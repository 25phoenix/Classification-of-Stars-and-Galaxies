"""Verify the exact RBF sensitivity artifacts against frozen inputs."""

from collections import Counter
import json
from statistics import mean, stdev

import joblib
import numpy as np
import pandas as pd

from common import FEATURES, THRESHOLDS, read_json, sha256, write_json
from evaluation import read_predictions
from exact_svm import CONFIG, OUTPUT, REFERENCE, digest, inputs, validate


def metrics(rows):
    matrix = Counter(zip(rows.label, rows.prediction))
    recall, f1 = [], []
    for label in (0, 1):
        correct = matrix[label, label]
        actual = correct + matrix[label, 1 - label]
        predicted = correct + matrix[1 - label, label]
        recall.append(correct / actual)
        f1.append(2 * correct / (actual + predicted))
    return {"accuracy": (matrix[0, 0] + matrix[1, 1]) / len(rows),
            "balanced_accuracy": mean(recall), "macro_f1": mean(f1)}


def assert_metrics(observed, recorded):
    assert observed.keys() == recorded.keys()
    for name, value in observed.items():
        np.testing.assert_allclose(value, recorded[name], rtol=1e-12, atol=1e-14)


def audit_threshold(k, table, split, fingerprint, policy):
    folder = OUTPUT / f"k{k}"
    validate(folder, table, split)
    result = read_json(folder / "results.json")
    assert result["fingerprint"] == fingerprint
    oof = read_predictions(folder / "oof_predictions.csv")
    test = read_predictions(folder / "test_predictions.csv")
    assert set(oof.group_id).isdisjoint(test.group_id)
    fold_metrics = []
    partitions = [(f"outer{i}", 3, oof.loc[oof.fold.eq(i)], result["folds"][i - 1])
                  for i in range(1, 6)]
    partitions.append(("final", 5, test, result["final_fit"]))
    for name, cv_folds, rows, recorded in partitions:
        checkpoint = read_json(folder / f"{name}.json")
        assert checkpoint == recorded and checkpoint["fingerprint"] == fingerprint
        assert all(sha256(folder / name) == checksum
                   for name, checksum in checkpoint["sha256"].items())
        source = read_predictions(folder / f"{name}_predictions.csv")
        pd.testing.assert_frame_equal(source.sort_values("row_id").reset_index(drop=True),
                                      rows.sort_values("row_id").reset_index(drop=True))
        search = pd.read_csv(folder / f"{name}_search.csv")
        assert search["param_svc__C"].tolist() == CONFIG["C"]
        assert {c for c in search if c.startswith("split") and c.endswith("_test_score")} == {
            f"split{i}_test_score" for i in range(cv_folds)}
        chosen = search.loc[search["rank_test_score"].eq(1)].iloc[0]
        assert chosen["param_svc__C"] == recorded["parameters"]["svc__C"]
        measured = metrics(rows)
        assert_metrics(measured, recorded["metrics"])
        if name != "final":
            fold_metrics.append(measured)

    summaries = {}
    for metric in fold_metrics[0]:
        values = [fold[metric] for fold in fold_metrics]
        summaries[metric] = {"mean": mean(values), "sample_sd": stdev(values)}
        assert_metrics(summaries[metric], result["outer_summary"][metric])
    bundle = joblib.load(folder / "svm_exact_final_model.joblib")
    assert tuple(bundle["features"]) == tuple(FEATURES)
    assert bundle["measurement_policy"] == policy
    x = table.loc[test.row_id, list(FEATURES)]
    assert tuple(bundle["model"].feature_names_in_) == tuple(FEATURES)
    scores = bundle["model"].decision_function(x)
    saved_scores = test.spiral_decision_score.to_numpy()
    np.testing.assert_allclose(scores, saved_scores, rtol=1e-12, atol=1e-12)

    comparison = {}
    for part, rows in (("oof", oof), ("test", test)):
        other = read_predictions(REFERENCE / f"k{k}/svm/{part}_predictions.csv")
        columns = ["objid", "group_id", "label"] + (["fold"] if part == "oof" else [])
        rows, other = (frame.set_index("row_id").sort_index() for frame in (rows, other))
        pd.testing.assert_frame_equal(rows[columns], other[columns])
        exact, approx = metrics(rows), metrics(other)
        assert_metrics(exact, result[part]["metrics"])
        comparison[part] = {"rows": len(rows), "exact_rbf": exact, "nystrom": approx,
                            "prediction_disagreements": int(rows.prediction.ne(other.prediction).sum())}
    return {"outer_folds": fold_metrics, "outer_summary": summaries,
            "holdout_max_score_delta": float(np.max(np.abs(scores - saved_scores))),
            "paired_comparison": comparison}


def main():
    table, plan, stamp = inputs()
    fingerprint = digest(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    results = {str(k): audit_threshold(k, table, plan[str(k)], fingerprint, stamp["measurement_policy"])
               for k in THRESHOLDS}
    destination = OUTPUT / "verification.json"
    write_json(destination, {"passed": True, "fingerprint": fingerprint, "thresholds": results})
    write_comparison(results)
    print(f"Verified {len(results)} exact-RBF thresholds. Wrote {destination}")


def write_comparison(results):
    lines = ["# Exact RBF and Nyström on identical objects", "",
             "Both pipelines use the same corrected feature rows and frozen split assignments. "
             "The exact RBF model uses hinge loss, and the Nyström linear solver uses squared-hinge "
             "loss. The paper's kernel was unspecified. The holdout was previously inspected.", "",
             "| K | Partition | Rows | Exact accuracy | Exact balanced accuracy | Exact macro-F1 | "
             "Nyström accuracy | Nyström balanced accuracy | Nyström macro-F1 | Different predictions |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k, result in results.items():
        for part, comparison in result["paired_comparison"].items():
            exact, approx = (comparison[name] for name in ("exact_rbf", "nystrom"))
            scores = " | ".join(f"{estimator[name]:.4f}" for estimator in (exact, approx)
                                for name in ("accuracy", "balanced_accuracy", "macro_f1"))
            lines.append(f"| {k} | {part} | {comparison['rows']:,} | {scores} | "
                         f"{comparison['prediction_disagreements']:,} |")
    lines.extend(["", "## Exact RBF five-fold summary", "",
                  "SD is the sample standard deviation across the five outer folds.", "",
                  "| K | Accuracy mean ± SD | Balanced accuracy mean ± SD | Macro-F1 mean ± SD |",
                  "| --- | ---: | ---: | ---: |"])
    for k, result in results.items():
        stats = result["outer_summary"]
        summary = " | ".join(f"{stats[name]['mean']:.4f} ± {stats[name]['sample_sd']:.4f}"
                             for name in ("accuracy", "balanced_accuracy", "macro_f1"))
        lines.append(f"| {k} | {summary} |")
    lines.extend(["", "[Per-fold results and Table 3 comparison](REPORT.md) · "
                  "[Recomputed metrics and model checks](verification.json)", ""])
    destination = OUTPUT / "PAIRED_REPORT.md"
    temporary = destination.with_suffix(".part")
    temporary.write_text("\n".join(lines))
    temporary.replace(destination)


if __name__ == "__main__":
    main()
