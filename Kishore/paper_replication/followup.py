"""Complete-case reconstruction on the previously inspected catalog splits."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import main  # Configure thread counts before importing the scientific stack.
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from catalogs import object_ids, prepare
from common import CLASSES, FEATURES, ROOT, THRESHOLDS, read_json, sha256, write_csv, write_json
from evaluation import MODELS, evaluation, read_predictions, run, split_plan, validate_saved, verify_plan

BASE = ROOT / "results/catalog"
COMPLETE = ROOT / "results/catalog_complete_case"
DR7_QUERY = """SELECT DISTINCT p.objID AS objid, s.z AS z, p.petroMag_r AS petromag_r
INTO MyDB.barchi_dr7
FROM SpecObj AS s JOIN PhotoObjAll AS p ON s.bestObjID = p.objID
WHERE s.bestObjID > 0"""
POLICY = {
    "name": "complete_case_v1", "sentinels": [-9999.999999, -9999.99],
    "bounds": {"c": [0, None], "a": [-1, 2], "s": [-1, 2], "g2": [0, 2], "h": [0, 1]},
    "bound_tolerance": 1e-8,
    "author_error_required": 0,
    "source": "Barchi Mendeley v1 Error definitions and observed release codes; CyMorph equations",
}


def feature_conditions(frame, policy=POLICY):
    values = frame[list(FEATURES)].apply(pd.to_numeric, errors="coerce")
    sentinel = values.isin(policy["sentinels"])
    nonfinite = ~np.isfinite(values)
    impossible = sentinel.copy()
    for name, (low, high) in policy["bounds"].items():
        tol = policy["bound_tolerance"]
        outside = values[name].lt(low - tol)
        if high is not None:
            outside |= values[name].gt(high + tol)
        impossible[name] = outside & ~sentinel[name] & ~nonfinite[name]
    return sentinel, nonfinite, impossible


def prediction(bundle, values, error=0):
    frame = pd.DataFrame([values], columns=FEATURES)
    policy = bundle["measurement_policy"]
    flags = []
    for flag, mask in zip(("sentinel", "nonfinite", "outside_domain"), feature_conditions(frame, policy)):
        flags.extend(f"{flag}:{name}" for name in FEATURES if mask.at[0, name])
    if error != policy["author_error_required"]:
        flags.append("author_error")
    if flags:
        return {"class": None, "flags": flags}
    model = bundle["model"]
    frame = frame.apply(pd.to_numeric)
    result = {"class": bundle["classes"][int(model.predict(frame)[0])], "flags": []}
    if hasattr(model, "predict_proba"):
        result["spiral_model_probability"] = float(model.predict_proba(frame)[0, 1])
    else:
        result["spiral_decision_score"] = float(model.decision_function(frame)[0])
    return result


def read_metadata(path):
    fields = ["objid", "z", "petromag_r"]
    frame = pd.read_csv(path, dtype=str)
    frame.columns = frame.columns.str.strip().str.lower()
    frame = frame[fields].copy()
    frame.objid = object_ids(frame.objid.str.strip())
    for name in fields[1:]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.drop_duplicates()
    if frame.objid.duplicated().any():
        raise ValueError("SDSS metadata has conflicting measurements for the same exact object ID")
    return frame.set_index("objid")


def count(table, mask):
    part = table.loc[mask]
    result = {"rows": len(part), "objects": int(part.objid.nunique())}
    for idx, label in enumerate(CLASSES):
        rows = part.loc[part.label.eq(idx)]
        result[label] = {"rows": len(rows), "objects": int(rows.objid.nunique())}
    return result


def apply_policy(table, metadata=None):
    table = table.copy()
    kinds = dict(zip(("sentinel", "nonfinite", "outside_domain"), feature_conditions(table)))
    bad = np.logical_or.reduce([v.to_numpy() for v in kinds.values()])
    bad_frame = pd.DataFrame(bad, index=table.index, columns=FEATURES)
    any_bad = bad_frame.any(axis=1)
    good_meta = pd.Series(True, index=table.index)
    metadata_audit = None
    if metadata is not None:
        for name in ("z", "petromag_r"):
            table[name] = table.objid.map(metadata[name])
        available = table.objid.isin(metadata.index)
        measured = np.isfinite(table.z) & np.isfinite(table.petromag_r) & table.petromag_r.gt(-100)
        z_cut = table.z.gt(.03) & table.z.lt(.1)
        mag_cut = table.petromag_r.lt(17.78)
        good_meta = measured & z_cut & mag_cut
        metadata_audit = {"source": "SDSS CASJobs DR7, SpecObj and PhotoObjAll", "query": DR7_QUERY,
                          "units": {"z": "dimensionless", "petromag_r": "magnitudes"},
                          "unique_exported_ids": len(metadata), "by_cut": {}}
    joined = table.label_provenance.eq("GZ1 Table 2")
    labeled = joined & table.label.isin([0, 1])
    author_ok = labeled & table.error.eq(0)
    b_ok = author_ok & table.b.abs().ge(30) if "b" in table else author_ok
    audit = {"policy": POLICY, "by_cut": {}, "metadata": metadata_audit}
    for k in THRESHOLDS:
        field = f"selection_k{k}"
        prior = table[field].eq("included")
        retained = prior & ~any_bad & good_meta
        flow = {"Author catalog": count(table, pd.Series(True, index=table.index)),
                "Joined GZ1": count(table, joined), "Defined E/S label": count(table, labeled),
                "Author Error = 0": count(table, author_ok), "|b| ≥ 30°": count(table, b_ok),
                f"Prior eligible at K ≥ {k}": count(table, prior),
                "Complete measurements": count(table, prior & ~any_bad)}
        if metadata is not None:
            current = prior & ~any_bad
            flow["Matched DR7 metadata"] = count(table, current & available)
            flow["Finite DR7 redshift and magnitude"] = count(table, current & measured)
            flow["0.03 < z < 0.1"] = count(table, current & measured & z_cut)
            flow["r Petrosian magnitude < 17.78"] = count(table, retained)
            metadata_audit["by_cut"][str(k)] = {"unmatched": count(table, current & ~available),
                "missing_measurement": count(table, current & available & ~measured)}
        overlap = Counter("|".join(name for name, yes in zip(FEATURES, row) if yes)
                          for row in bad_frame.loc[prior & any_bad].to_numpy())
        by_feature = {}
        for name in FEATURES:
            by_feature[name] = {reason: count(table, prior & mask[name]) for reason, mask in kinds.items()}
            by_feature[name]["author_flags"] = {
                str(flag): int(n) for flag, n in table.loc[labeled & table.k.ge(k) & bad_frame[name],
                                                          "error"].value_counts().items()}
            by_feature[name]["codes"] = {str(code): count(table, prior & table[name].eq(code))
                                         for code in POLICY["sentinels"]}
        audit["by_cut"][str(k)] = {"flow": flow, "invalid_by_feature": by_feature,
                                  "overlap_patterns": dict(overlap),
                                  "removed_invalid": count(table, prior & any_bad),
                                  "retained": count(table, retained)}
        table.loc[prior & any_bad, field] = "invalid_cymorph_measurement"
        table.loc[prior & ~any_bad & ~good_meta, field] = "dr7_metadata_missing_or_cut"
    return table, audit


def restrict_plan(plan, table):
    def folds(entries, keep):
        result = []
        for entry in entries:
            next_entry = {key: [i for i in entry[key] if i in keep]
                          for key in ("train", "validation")}
            if "fold" in entry:
                next_entry["fold"] = entry["fold"]
                next_entry["inner"] = folds(entry["inner"], set(next_entry["train"]))
            result.append(next_entry)
        return result

    corrected = {}
    for k in THRESHOLDS:
        keep = set(table.loc[table[f"selection_k{k}"].eq("included"), "row_id"])
        prior = plan[str(k)]
        corrected[str(k)] = {name: [i for i in prior[name] if i in keep]
                             for name in ("development", "test")}
        corrected[str(k)]["outer"] = folds(prior["outer"], keep)
        corrected[str(k)]["final_tuning"] = folds(prior["final_tuning"], keep)
    verify_plan(table, corrected)
    return corrected


def prepare_run(out, metadata_path=None, smoke=False):
    if smoke:
        table, provenance = main.synthetic_table()
        table[list(FEATURES)] = 1 / (1 + np.exp(-table[list(FEATURES)]))
        original = split_plan(table, 42)
        table.loc[table.row_id.isin([7, 29, 95, 186]), "c"] = POLICY["sentinels"][0]
        baseline_hash = "synthetic"
    else:
        table, provenance = prepare(ROOT / "data/raw", ROOT / "cache/catalogs")
        original_path = BASE / "splits.json"
        baseline_hash = sha256(original_path)
        original = read_json(original_path)["thresholds"]
    metadata = read_metadata(metadata_path) if metadata_path else None
    corrected_provenance = {**provenance, "measurement_policy": POLICY,
                            "baseline_splits_sha256": baseline_hash,
                            "followup_source_sha256": sha256(Path(__file__)),
                            "metadata_sha256": sha256(metadata_path) if metadata_path else None,
                            "holdout": "Previously inspected, original membership retained"}
    key = hashlib.sha256(json.dumps(corrected_provenance, sort_keys=True).encode()).hexdigest()
    folder = ROOT / "cache/followup" / key
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / "selection.csv"
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / "audit.json").is_file():
            audit = read_json(folder / "audit.json")
            if audit["selection_sha256"] != sha256(cache):
                raise ValueError("Corrected selection cache checksum mismatch")
            table = pd.read_csv(cache, dtype={"objid": str}, float_precision="round_trip")
        else:
            table, audit = apply_policy(table, metadata)
            write_csv(cache, table)
            audit["selection_sha256"] = sha256(cache)
            write_json(folder / "audit.json", audit)
    corrected_provenance["corrected_selection"] = {"fingerprint": key, "sha256": audit["selection_sha256"]}
    plan = restrict_plan(original, table)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        audit_path = out / "selection_audit.json"
        if audit_path.is_file() and read_json(audit_path) != audit:
            raise ValueError("Output directory already contains a different selection audit")
        if not audit_path.is_file():
            write_json(audit_path, audit)
    return table, corrected_provenance, plan, audit


def comparison(out, plan):
    result = {}
    for cut, split in plan.items():
        result[cut] = {}
        for name in MODELS:
            result[cut][name] = {}
            for part, partition in (("oof", "development"), ("test", "test")):
                postfix = f"k{cut}/{name}/{part}_predictions.csv"
                prior = read_predictions(BASE / postfix)
                after = read_predictions(out / postfix)
                common = prior.loc[prior.row_id.isin(split[partition])].set_index("row_id").loc[after.row_id]
                for col in ("objid", "group_id", "label", *(("fold",) if part == "oof" else ())):
                    assert np.array_equal(common[col], after[col])
                result[cut][name][part] = {"original_rows": len(prior), "retained_rows": len(after),
                    "coverage": len(after)/len(prior), "before_full": evaluation(prior),
                    "before_on_retained": evaluation(common), "after_on_retained": evaluation(after),
                    "changed_predictions": int(np.count_nonzero(common.prediction.to_numpy() != after.prediction))}
    return result


def report_followup(table, provenance, out, plan, audit, smoke):
    from reporting import report
    import matplotlib.pyplot as plt
    report(table, provenance, out, smoke)
    lines = ["", "## Complete-case selection and follow-up evaluation", "",
             "The original holdout was previously inspected. Every split is the original assignment "
             "intersected with eligible rows, including the inner folds and final tuning folds.", "",
             "The Barchi release documents quality flags 0 (success), 1 (nearby bright sources), "
             "2 (Petrosian radius), 3 (G2), 4 (H), 5 (C), 6 (A), and 7 (S). It does not define numeric "
             "sentinels. The original CSV contains the exact codes −9999.999999 and −9999.99. "
             "They are outside every feature's mathematical domain and can coexist with Error = 0.", "",
             "Complete measurements require finite values and physically possible ranges. C is "
             "nonnegative, A and S use the union [−1, 2] covering correlation and one minus "
             "correlation, G2 is in [0, 2], and normalized H is in [0, 1]. Endpoints are valid "
             "with a numerical tolerance of 10⁻⁸. Negative A/S and zeros can be valid.", "",
             "The release supplies K = (2Rp/FWHM)² in consistent angular or pixel units. "
             "Latitude uses converted GZ1 coordinates. Labels use exclusive GZ1 clean flags, "
             "exclude ambiguous and majority-merger classifications, and ignore author model outputs. "
             "The catalog contains no raw image preprocessing data.",
             "", "[Exact per-code, feature, class, quality-flag and overlap counts](selection_audit.json)", ""]
    if audit["metadata"] is None:
        lines += ["Redshift and Petrosian magnitude were unavailable for this run. The separate "
                  "metadata-matched mode applies the strict paper cuts when the DR7 export is staged.", ""]
    else:
        lines += ["This separate DR7-matched run also requires measured 0.03 < z < 0.1 and "
                  "r-band Petrosian magnitude < 17.78. The selection audit records coverage and unmatched rows.", ""]
    for k, cut in audit["by_cut"].items():
        lines += [f"### Selection flow at K ≥ {k}", "",
                  "| Stage | Rows | Unique DR7 IDs | E rows / IDs | S rows / IDs |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for stage, counts in cut["flow"].items():
            e, s = (counts[label] for label in CLASSES)
            lines.append(f"| {stage} | {counts['rows']:,} | {counts['objects']:,} | "
                         f"{e['rows']:,} / {e['objects']:,} | {s['rows']:,} / {s['objects']:,} |")
        lines += ["", "| Invalid feature among prior eligible rows | Sentinel | Nonfinite | Outside domain |",
                  "| --- | ---: | ---: | ---: |"]
        for name, v in cut["invalid_by_feature"].items():
            lines.append(f"| {name} | {v['sentinel']['rows']:,} | {v['nonfinite']['rows']:,} | {v['outside_domain']['rows']:,} |")
        lines.append("")
    if not smoke:
        compare = comparison(out, plan)
        write_json(out / "common_object_comparison.json", compare)
        lines += ["## Before and after on common held-out objects", "",
                  "The old model is scored on the retained previously held-out rows alongside the "
                  "refitted complete-case model. Full-population and common-population metrics "
                  "are available for both OOF and held-out predictions.", "",
                  "| K | Model | Coverage | Old full accuracy | Old retained accuracy | New retained accuracy | Old / new retained macro-F1 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for k, models in compare.items():
            for name, parts in models.items():
                part = parts["test"]
                a, b, c = (part[key]["metrics"] for key in ("before_full", "before_on_retained", "after_on_retained"))
                lines.append(f"| {k} | {name} | {part['coverage']:.2%} | {a['accuracy']:.2%} | "
                             f"{b['accuracy']:.2%} | {c['accuracy']:.2%} | {b['macro_f1']:.4f} / {c['macro_f1']:.4f} |")
        lines += ["", "[Common-object metrics and prediction changes](common_object_comparison.json)", ""]
    lines += ["## MLP optimization", "",
              "The original 44/88/22 architecture, fold-local scaling, and 60-epoch Adam "
              "budget are retained. Curves are from the selected outer refits and final refit. "
              "Reaching the cap is reported without claiming convergence.", ""]
    for k in THRESHOLDS:
        runs = read_json(out / f"k{k}/mlp/results.json")
        fits = [r["optimization"] for r in (*runs["folds"], runs["final_fit"])]
        fig, ax = plt.subplots(layout="constrained")
        for i, fit in enumerate(fits):
            ax.plot(np.arange(1, len(fit["loss_curve"])+1), fit["loss_curve"],
                    label=f"Outer {i+1}" if i < 5 else "Final")
        ax.set(xlabel="Epoch", ylabel="Training loss", title=f"MLP selected fits, K ≥ {k}")
        ax.legend()
        fig.savefig(out / f"plots/k{k}_mlp_loss.png", dpi=140)
        plt.close(fig)
        lines += [f"K ≥ {k} selected refits reached the iteration cap in "
                  f"{sum(x['reached_iteration_cap'] for x in fits)} of 6 fits.",
                  "", f"![MLP training curves](plots/k{k}_mlp_loss.png)", ""]
    pilot = out / "svm_pilot.json"
    if pilot.is_file():
        result = read_json(pilot)
        lines += ["## Bounded exact RBF SVM timing pilot", "",
                  f"A single training-only inner partition of {result['training_rows']:,} rows "
                  f"was used with C = 2 and gamma = 1/5. The fit status was "
                  f"`{result['status']}` at a {result['deadline_seconds']}-second limit. "
                  "No validation or holdout data was scored. The reported benchmark uses the "
                  "unchanged Nyström approximation.", ""]
    path = out / "REPORT.md"
    with path.open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def pilot_fit():
    table, _ = prepare(ROOT / "data/raw", ROOT / "cache/catalogs")
    table, _ = apply_policy(table)
    split = read_json(BASE / "splits.json")["thresholds"]["5"]["outer"][0]["inner"][0]["train"]
    train = table.set_index("row_id").loc[split]
    train = train[train.selection_k5.eq("included")]
    print(len(train), flush=True)
    estimator = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2, gamma=.2, cache_size=256))
    start = perf_counter()
    estimator.fit(train[list(FEATURES)], train.label)
    print(perf_counter() - start, flush=True)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "check", "report", "smoke", "predict", "pilot", "pilot-fit"))
    parser.add_argument("--threshold", type=int, choices=THRESHOLDS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--values", help="JSON object with c, a, s, g2, h")
    parser.add_argument("--error", type=int, default=0)
    args = parser.parse_args()
    if args.command == "predict":
        print(json.dumps(prediction(joblib.load(args.model), json.loads(args.values), args.error), indent=2))
        return
    if args.command == "pilot-fit":
        pilot_fit()
        return
    smoke = args.command == "smoke"
    out = (args.out or (ROOT / "results/catalog_dr7_complete_case" if args.metadata else
                       ROOT / "results/followup_smoke" if smoke else COMPLETE)).resolve()
    if not out.is_relative_to(ROOT / "results") or out == ROOT / "results" or out.is_relative_to(BASE):
        raise ValueError("Use a new output directory under paper_replication/results")
    if args.command == "pilot":
        deadline = 180
        with subprocess.Popen([sys.executable, str(Path(__file__)), "pilot-fit"],
                              stdout=subprocess.PIPE, text=True) as worker:
            size = int(worker.stdout.readline())
            result = {"training_rows": size, "deadline_seconds": deadline}
            try:
                timing, _ = worker.communicate(timeout=deadline)
                if worker.returncode:
                    raise RuntimeError(f"Timing pilot returned {worker.returncode}")
                result.update(status="fit_completed", fit_seconds=float(timing.strip()))
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.communicate()
                result.update(status="time_limit")
        write_json(out / "svm_pilot.json", result)
        print(json.dumps(result))
        return
    table, provenance, plan, audit = prepare_run(out, args.metadata, smoke)
    if args.command in ("run", "smoke"):
        chosen = (args.threshold,) if args.threshold else THRESHOLDS
        run(table, provenance, out, workers=args.workers, smoke=smoke, thresholds=chosen, fixed_plan=plan)
    if args.command in ("check", "report", "smoke"):
        validate_saved(out, table)
        assert read_json(out / "splits.json")["thresholds"] == plan
        if not smoke:
            comparison(out, plan)
        print("Verified retained assignments, OOF coverage, metrics, and saved models.")
        if args.command != "check":
            report_followup(table, provenance, out, plan, audit, smoke)


if __name__ == "__main__":
    cli()
