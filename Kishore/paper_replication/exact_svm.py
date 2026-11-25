"""Exact RBF SVM sensitivity run on frozen complete-case assignments."""
import argparse
from hashlib import sha256 as digest
import json
from pathlib import Path
from time import perf_counter

import main
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CLASSES, FEATURES, PAPER_ACCURACY, ROOT, THRESHOLDS, read_json, sha256, write_csv, write_json
from evaluation import SCORER, evaluation, predict, read_predictions, training_environment, verify_plan

REFERENCE = ROOT / "results/catalog_complete_case"
OUTPUT = ROOT / "results/catalog_complete_case_exact_svm"
CONFIG = {"kernel": "rbf", "gamma": .2, "cache_size": 256, "C": [.5, 2.]}


def fit(train, folds, workers):
    position = {row: i for i, row in enumerate(train.row_id)}
    cv = [(np.array([position[i] for i in fold["train"]]),
           np.array([position[i] for i in fold["validation"]])) for fold in folds]
    model = make_pipeline(StandardScaler(), SVC(kernel=CONFIG["kernel"], gamma=CONFIG["gamma"],
                                                cache_size=CONFIG["cache_size"]))
    search = GridSearchCV(model, {"svc__C": CONFIG["C"]}, cv=cv, scoring=SCORER, n_jobs=workers,
                          pre_dispatch="n_jobs", error_score="raise", refit=True)
    start = perf_counter()
    search.fit(train[list(FEATURES)], train.label)
    return search, perf_counter() - start


def inputs():
    identity = read_json(REFERENCE / "provenance.json")
    selection = identity["input"]["corrected_selection"]
    path = ROOT / "cache/followup" / selection["fingerprint"] / "selection.csv"
    if sha256(path) != selection["sha256"]:
        raise ValueError("Corrected feature cache checksum mismatch")
    fields = ["row_id", "objid", "group_id", "label", *FEATURES, *(f"selection_k{k}" for k in THRESHOLDS)]
    table = pd.read_csv(path, dtype={"objid": str}, usecols=fields, float_precision="round_trip")
    split_path = REFERENCE / "splits.json"
    plan = read_json(split_path)["thresholds"]
    verify_plan(table, plan)
    stamp = {"splits_sha256": sha256(split_path), "feature_sha256": selection["sha256"],
             "code": sha256(Path(__file__)), "evaluation_code": sha256(ROOT / "evaluation.py"),
             "environment": training_environment(), "fit_configuration": CONFIG,
             "measurement_policy": identity["input"]["measurement_policy"],
             "features": list(FEATURES), "followup_holdout": True}
    return table.set_index("row_id", drop=False), plan, stamp


def execute(folder, name, fingerprint, train, valid, cv, workers, policy, fold=None):
    destination = folder / f"{name}.json"
    pred_path = folder / f"{name}_predictions.csv"
    search_path = folder / f"{name}_search.csv"
    model_path = folder / "svm_exact_final_model.joblib"
    products = [pred_path, search_path] + ([model_path] if fold is None else [])
    if destination.is_file():
        saved = read_json(destination)
        if saved["fingerprint"] != fingerprint or any(
                not p.is_file() or saved["sha256"][p.name] != sha256(p) for p in products):
            raise ValueError(f"Incompatible or corrupted checkpoint {destination}")
        return read_predictions(pred_path), saved
    search, seconds = fit(train, cv, workers)
    model = search.best_estimator_
    prediction = predict(model, valid.reset_index(drop=True), fold)
    write_csv(pred_path, prediction)
    write_csv(search_path, pd.DataFrame(search.cv_results_))
    if fold is None:
        temp = model_path.with_suffix(".part")
        joblib.dump({"model": model, "features": FEATURES, "classes": CLASSES,
                     "fingerprint": fingerprint, "measurement_policy": policy}, temp, compress=3)
        temp.replace(model_path)
    item = {**evaluation(prediction), "parameters": search.best_params_, "seconds": seconds,
            "fingerprint": fingerprint, "sha256": {p.name: sha256(p) for p in products}}
    write_json(destination, item)
    return prediction, item


def validate(directory, table, split):
    summary = read_json(directory / "results.json")
    for part, field in (("oof", "development"), ("test", "test")):
        rows = read_predictions(directory / f"{part}_predictions.csv")
        assert rows.row_id.is_unique and set(rows.row_id) == set(split[field])
        reference = table.loc[rows.row_id]
        for name in ("objid", "group_id", "label"):
            assert np.array_equal(rows[name], reference[name])
        assert evaluation(rows) == summary[part]
        if part == "oof":
            for spec, saved in zip(split["outer"], summary["folds"]):
                fold = rows.loc[rows.fold.eq(spec["fold"])]
                assert set(fold.row_id) == set(spec["validation"])
                assert evaluation(fold) == {k: saved[k] for k in ("metrics", "class_report", "confusion_matrix")}
        else:
            bundle = joblib.load(directory / "svm_exact_final_model.joblib")
            assert bundle["fingerprint"] == summary["fingerprint"]
            assert np.array_equal(bundle["model"].predict(reference[list(FEATURES)]), rows.prediction)


def run_cut(out, table, split, stamp, k, workers):
    folder = out / f"k{k}"
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = digest(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    records, oof = [], []
    for spec in split["outer"]:
        i = spec["fold"]
        rows, item = execute(folder, f"outer{i}", fingerprint, table.loc[spec["train"]],
                             table.loc[spec["validation"]], spec["inner"], workers,
                             stamp["measurement_policy"], i)
        records.append(item)
        oof.append(rows)
        print(f"K >= {k} exact RBF SVM outer {i}/5 saved", flush=True)
    oof = pd.concat(oof, ignore_index=True).sort_values("row_id")
    test, final = execute(folder, "final", fingerprint, table.loc[split["development"]],
                          table.loc[split["test"]], split["final_tuning"], workers, stamp["measurement_policy"])
    write_csv(folder / "oof_predictions.csv", oof)
    write_csv(folder / "test_predictions.csv", test)
    stats = {}
    for key in final["metrics"]:
        values = [r["metrics"][key] for r in records]
        stats[key] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
    write_json(folder / "results.json", {"fingerprint": fingerprint, "outer_summary": stats,
                                        "folds": records, "final_fit": final,
                                        "oof": evaluation(oof), "test": evaluation(test)})
    validate(folder, table, split)
    print(f"K >= {k} exact RBF SVM held-out artifacts verified", flush=True)


def report(out, table, plan):
    lines = ["# Exact RBF SVM sensitivity comparison", "",
             "This run uses identical complete-case rows and frozen assignments. "
             "The scaler is fitted separately inside each training fold. Both SVMs tune C over "
             "0.5 and 2 using macro-F1 and gamma 1/5. The exact-kernel SVC uses hinge loss "
             "and the original Nyström pipeline uses a squared-hinge LinearSVC. The comparison "
             "therefore includes both kernel representation and solver objective. The paper "
             "does not specify the kernel it used.", "",
             "| K | Exact outer accuracy | Exact outer macro-F1 | Exact test accuracy | Exact test balanced accuracy | Exact test macro-F1 | Nyström test accuracy | Nyström test macro-F1 | Paper SVM accuracy |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    details = []
    for k in THRESHOLDS:
        folder = out / f"k{k}"
        split = plan[str(k)]
        validate(folder, table, split)
        exact = read_json(folder / "results.json")
        approx = read_json(REFERENCE / f"k{k}/svm/results.json")
        a, f = [exact["outer_summary"][col] for col in ("accuracy", "macro_f1")]
        x, y = exact["test"]["metrics"], approx["test"]["metrics"]
        lines.append(f"| {k} | {a['mean']:.4f} ± {a['sample_sd']:.4f} | {f['mean']:.4f} ± {f['sample_sd']:.4f} "
                     f"| {x['accuracy']:.4f} | {x['balanced_accuracy']:.4f} | {x['macro_f1']:.4f} "
                     f"| {y['accuracy']:.4f} | {y['macro_f1']:.4f} | {PAPER_ACCURACY[k]['svm']:.4f} |")
        details += ["", f"## K ≥ {k}", "", "| Fold | Accuracy | Balanced accuracy | Macro-F1 | Seconds | C |",
                    "| --- | --- | --- | --- | --- | --- |"]
        for i, item in enumerate(exact["folds"], 1):
            score = item["metrics"]
            details.append(f"| {i} | {score['accuracy']:.4f} | {score['balanced_accuracy']:.4f} | "
                           f"{score['macro_f1']:.4f} | {item['seconds']:.1f} | {item['parameters']['svc__C']} |")
        details += ["", f"[Held-out confusion matrix and per-class report](k{k}/results.json)", ""]
    (out / "REPORT.md").write_text("\n".join(lines + details) + "\n")


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--threshold", type=int, choices=THRESHOLDS)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    table, plan, stamp = inputs()
    if args.command == "report":
        write_json(OUTPUT / "provenance.json", stamp)
        report(OUTPUT, table, plan)
    else:
        for k in ((args.threshold,) if args.threshold else THRESHOLDS):
            run_cut(OUTPUT, table, plan[str(k)], stamp, k, args.workers)


if __name__ == "__main__":
    cli()
