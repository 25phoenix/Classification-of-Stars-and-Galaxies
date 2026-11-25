"""Check metadata eligibility and compare retained predictions with the complete-case run."""

import main  # Configure threads before the scientific stack.
import joblib
import numpy as np
import pandas as pd

from common import FEATURES, ROOT, THRESHOLDS, read_json, sha256, write_json
from evaluation import MODELS, evaluation, read_predictions
from validate_dr7 import OUT

REFERENCE = ROOT / "results/catalog_complete_case"
PAPER_COUNTS = {5: 104787, 10: 89829, 20: 58030}


def matched_metrics(previous, current, part):
    assert previous.row_id.is_unique and current.row_id.is_unique
    common = previous.set_index("row_id").loc[current.row_id]
    columns = ["objid", "group_id", "label"] + (["fold"] if part == "oof" else [])
    pd.testing.assert_frame_equal(common[columns], current.set_index("row_id")[columns])
    return {"reference_full": evaluation(previous), "reference_retained": evaluation(common),
            "metadata_retained": evaluation(current), "reference_rows": len(previous),
            "retained_rows": len(current), "coverage": len(current) / len(previous),
            "predictions_changed": int(np.count_nonzero(common.prediction.to_numpy() != current.prediction))}


def verify_metadata():
    run = read_json(OUT / "provenance.json")["input"]
    export = read_json(OUT / "metadata_export_audit.json")
    assert sha256(ROOT / export["path"]) == export["sha256"]
    assert sha256(ROOT / export["clean_export"]["path"]) == export["clean_export"]["sha256"] == run["metadata_sha256"]
    cache = ROOT / "cache/followup" / run["corrected_selection"]["fingerprint"] / "selection.csv"
    assert sha256(cache) == run["corrected_selection"]["sha256"]
    table = pd.read_csv(cache, dtype={"objid": str}, float_precision="round_trip").set_index("row_id")
    clean = pd.read_csv(ROOT / export["clean_export"]["path"], dtype={"objid": str},
                        float_precision="round_trip").set_index("objid")
    conflicts = pd.read_csv(OUT / "conflicting_dr7_measurements.csv", dtype={"objid": str})
    assert conflicts.objid.nunique() == export["conflicting_ids"]
    assert not clean.index.isin(conflicts.objid).any()
    assert clean.index.is_unique
    old = read_json(REFERENCE / "splits.json")["thresholds"]
    plan = read_json(OUT / "splits.json")["thresholds"]
    partitions, result = {}, {}
    for k in THRESHOLDS:
        cut = str(k)
        reference, filtered = old[cut], plan[cut]
        old_rows = table.loc[reference["development"] + reference["test"]]
        z = old_rows.objid.map(clean.z)
        r = old_rows.objid.map(clean.petromag_r)
        keep = set(old_rows.index[np.isfinite(z) & np.isfinite(r) & (z > .03) & (z < .1) & (r > -100) & (r < 17.78)])
        assert filtered["development"] == [i for i in reference["development"] if i in keep]
        assert filtered["test"] == [i for i in reference["test"] if i in keep]
        rows = table.loc[list(keep)]
        assert rows[f"selection_k{k}"].eq("included").all()
        assert rows.k.ge(k).all() and rows.b.abs().ge(30).all()
        partitions[cut] = {
            name: {str(label): int(count) for label, count in table.loc[filtered[name]].label.value_counts().items()}
            for name in ("development", "test")}
        ambiguous = old_rows.loc[old_rows.objid.isin(conflicts.objid), "label"]
        partitions[cut]["conflict_exclusions"] = {
            str(label): int(ambiguous.eq(label).sum()) for label in (0, 1)}
        result[cut] = {}
        for name in MODELS:
            prefix = f"k{k}/{name}"
            folder = OUT / prefix
            parts = {}
            for part in ("oof", "test"):
                current = read_predictions(folder / f"{part}_predictions.csv")
                previous = read_predictions(REFERENCE / prefix / f"{part}_predictions.csv")
                parts[part] = matched_metrics(previous, current, part)
            trained = joblib.load(folder / f"{name}_final_model.joblib")
            model = trained["model"]
            assert tuple(trained["features"]) == tuple(model.feature_names_in_) == FEATURES
            x = table.loc[current.row_id, list(FEATURES)]
            np.testing.assert_array_equal(model.predict(x), current.prediction)
            if hasattr(model, "predict_proba"):
                scores, saved = model.predict_proba(x)[:, 1], current.spiral_probability
            else:
                scores, saved = model.decision_function(x), current.spiral_decision_score
            np.testing.assert_allclose(scores, saved, atol=1e-12, rtol=1e-12)
            parts["reloaded_maximum_score_difference"] = float(np.max(np.abs(scores - saved)))
            result[cut][name] = parts
    return export, partitions, result


def report(export, partitions, result):
    lines = ["## DR7 export validation and comparison with the complete-case run", "",
             f"The staged export contained {export['rows']:,} rows and {export['unique_ids']:,} "
             f"exact decimal IDs. All {export['conflicting_ids']:,} IDs with conflicting measurements "
             "were excluded in full. The unmodified export and the derived unambiguous table "
             "have independently recorded hashes.", "",
             f"The raw join matched {export['raw_join']['matched_rows']:,} author rows. "
             f"After conflict exclusion it matched {export['unambiguous_join']['matched_rows']:,} "
             f"and left {export['unambiguous_join']['unmatched_rows']:,} unmatched.", "",
             f"The export is {export['size_bytes']:,} bytes. Its SHA-256 is `{export['sha256']}`. "
             "The columns are `objid`, `z` and `petromag_r`. "
             f"{export['petromag_le_minus_100']:,} export rows have magnitude ≤ −100, "
             "which the existing selection excludes as invalid.", "",
             "[Export audit](metadata_export_audit.json) · [Conflicting measurements](conflicting_dr7_measurements.csv) · "
             "[Independent fold audit](verification.json) · [Paired results](dr7_complete_case_comparison.json)", "",
             "| K ≥ | Conflicting metadata removed E / S | Development E / S | Holdout E / S | Selected rows | Paper Table 1 | Difference |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k in THRESHOLDS:
        split = partitions[str(k)]
        dev, test, conflict = (split[name] for name in ("development", "test", "conflict_exclusions"))
        n = sum(dev.values()) + sum(test.values())
        lines.append(f"| {k} | {conflict['0']:,} / {conflict['1']:,} | "
                     f"{dev['0']:,} / {dev['1']:,} | {test['0']:,} / {test['1']:,} "
                     f"| {n:,} | {PAPER_COUNTS[k]:,} | {n-PAPER_COUNTS[k]:+,} |")
    lines.extend(["", "The exact original roster and split indices remain unavailable. The reconstruction "
                  "uses the released GZ1 clean flags and CyMorph measurements, excludes invalid measurements "
                  "and ambiguous metadata, and uses a group-preserving nested assessment optimized for macro-F1. "
                  "The holdout was previously inspected.", "",
                  "Scores below are accuracy / balanced accuracy / macro-F1. Coverage is relative to the "
                  "earlier complete-case holdout. The common-row columns hold the evaluated population fixed.", "",
                  "| K ≥ | Model | Coverage | Complete-case full holdout | Complete-case on retained rows | DR7 refit on retained rows |",
                  "| ---: | --- | ---: | --- | --- | --- |"])
    for cut, models in result.items():
        for name, parts in models.items():
            item = parts["test"]
            cells = []
            for key in ("reference_full", "reference_retained", "metadata_retained"):
                m = item[key]["metrics"]
                cells.append(f"{m['accuracy']:.2%} / {m['balanced_accuracy']:.2%} / {m['macro_f1']:.4f}")
            lines.append(f"| {cut} | {name} | {item['coverage']:.2%} | " + " | ".join(cells) + " |")
    learned = [parts["test"] for models in result.values() for name, parts in models.items() if name != "dummy"]
    paired = [(item["reference_retained"]["metrics"], item["metadata_retained"]["metrics"]) for item in learned]
    if all(new["accuracy"] > old["accuracy"] and new["macro_f1"] > old["macro_f1"]
           and new["balanced_accuracy"] < old["balanced_accuracy"] for old, new in paired):
        lines.extend(["", "On identical retained objects, all nine learned refits improved accuracy and macro-F1 "
                      "while reducing balanced accuracy. The class-specific results and confusion matrices above "
                      "show how the refits changed elliptical and spiral recall."])
    path = OUT / "REPORT.md"
    body = path.read_text().split("\n## DR7 export validation and comparison", maxsplit=1)[0]
    body = body.replace("# Barchi et al. binary morphology experiment",
                        "# Barchi et al. binary morphology reconstruction with DR7 cuts", 1)
    body = body.replace("new group-aware splits", "frozen group-aware splits")
    body = body.replace(
        "Neither staged catalog includes redshift or r-band Petrosian magnitude, so we could not independently "
        "reapply those two cuts. Table 1 sample membership cannot be recovered from this join alone.",
        "The separate DR7 metadata export supplies spectroscopic redshift and r-band Petrosian magnitude. "
        "Both documented strict cuts are applied. The available catalogs do not identify the full Table 1 roster.")
    body = body.replace(
        "A preliminary exact-kernel attempt was interrupted for runtime before its first SVM outer fold "
        "completed. The paper does not specify its kernel.",
        "The exact-RBF sensitivity experiment was run separately on the earlier complete-case sample. "
        "This metadata branch retains Nyström. The paper's kernel was unspecified.")
    body = body.replace("| |b| ≥ 30° |", r"| \|b\| ≥ 30° |")
    temporary = path.with_suffix(".part")
    temporary.write_text(body.rstrip() + "\n\n" + "\n".join(lines) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    export, partitions, comparisons = verify_metadata()
    write_json(OUT / "dr7_complete_case_comparison.json", {"passed": True, "partitions": partitions,
                                                          "results": comparisons})
    report(export, partitions, comparisons)
    print("Verified DR7 eligibility, identical retained IDs, and reloaded model scores at all three thresholds.")
