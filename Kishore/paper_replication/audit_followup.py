"""Recompute headline statistics directly from saved real predictions."""
from collections import Counter
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("results/catalog_complete_case"))
OUT = parser.parse_args().out
BASE = Path("results/catalog")
MODELS = ("dummy", "dt", "svm", "mlp")
CANDIDATES = {"dummy": 1, "dt": 6, "svm": 2, "mlp": 2}
LABELS = (0, 1)


def read_json(path):
    return json.loads(path.read_text())


def prediction_rows(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert all(r["objid"].isdecimal() for r in rows)
    assert len({r["row_id"] for r in rows}) == len(rows)
    return rows


def metrics(rows):
    matrix = Counter((int(r["label"]), int(r["prediction"])) for r in rows)
    recall, f1 = [], []
    for cl in LABELS:
        true_pos = matrix[cl, cl]
        actual = sum(matrix[cl, p] for p in LABELS)
        predicted = sum(matrix[y, cl] for y in LABELS)
        recall.append(true_pos / actual)
        f1.append(2 * true_pos / (actual + predicted))
    return {"accuracy": sum(matrix[i, i] for i in LABELS) / len(rows),
            "balanced_accuracy": statistics.mean(recall), "macro_f1": statistics.mean(f1)}


def equal(a, b):
    assert all(math.isclose(a[k], b[k], rel_tol=1e-12, abs_tol=1e-14) for k in a), (a, b)


def check_search(path, folds, name):
    with path.open(newline="") as file:
        search = csv.DictReader(file)
        assert set(c for c in search.fieldnames if c.startswith("split") and c.endswith("_test_score")) == {
            f"split{i}_test_score" for i in range(folds)}
        assert len(list(search)) == CANDIDATES[name]


def ids(rows):
    return {int(r["row_id"]) for r in rows}


plan = read_json(OUT / "splits.json")["thresholds"]
original = read_json(BASE / "splits.json")["thresholds"]


def compare_folds(older, newer, keep):
    assert len(older) == len(newer)
    for old, new in zip(older, newer):
        for key in ("train", "validation"):
            assert new[key] == [row for row in old[key] if row in keep]
        if "inner" in old:
            assert new["fold"] == old["fold"]
            compare_folds(old["inner"], new["inner"], set(new["train"]))


for threshold, current in plan.items():
    former = original[threshold]
    keep = set(current["development"] + current["test"])
    for key in ("development", "test"):
        assert current[key] == [row for row in former[key] if row in keep]
    compare_folds(former["outer"], current["outer"], keep)
    compare_folds(former["final_tuning"], current["final_tuning"], keep)
del original

input_info = read_json(OUT / "provenance.json")["input"]
folder = Path("cache/followup") / input_info["corrected_selection"]["fingerprint"]
policy = input_info["measurement_policy"]
eligible = set(plan["5"]["development"] + plan["5"]["test"])
measured_rows = 0
with (folder / "selection.csv").open(newline="") as stream:
    for row in csv.DictReader(stream):
        if int(row["row_id"]) not in eligible:
            continue
        assert row["selection_k5"] == "included" and float(row["error"]) == 0
        assert row["label_provenance"] == "GZ1 Table 2"
        for name, (low, high) in policy["bounds"].items():
            value = float(row[name])
            assert math.isfinite(value) and value not in policy["sentinels"]
            assert value >= low - policy["bound_tolerance"]
            assert high is None or value <= high + policy["bound_tolerance"]
        measured_rows += 1
assert measured_rows == len(eligible)
result = {}
anchor = {"development": set(plan["5"]["development"]), "test": set(plan["5"]["test"])}
for threshold, split in plan.items():
    report = result[threshold] = {}
    for name in MODELS:
        folder = OUT / f"k{threshold}" / name
        saved = read_json(folder / "results.json")
        dev = prediction_rows(folder / "oof_predictions.csv")
        test = prediction_rows(folder / "test_predictions.csv")
        assert ids(dev) == set(split["development"]) and ids(dev) <= anchor["development"]
        assert ids(test) == set(split["test"]) and ids(test) <= anchor["test"]
        assert {r["group_id"] for r in dev}.isdisjoint(r["group_id"] for r in test)
        all_folds = []
        for i, fold in enumerate(split["outer"], 1):
            rows = [r for r in dev if int(r["fold"]) == i]
            assert ids(rows) == set(fold["validation"]) and len(fold["inner"]) == 3
            score = metrics(rows)
            equal(score, saved["folds"][i - 1]["metrics"])
            check_search(folder / f"{name}_outer{i}_search.csv", 3, name)
            all_folds.append(score)
        check_search(folder / f"{name}_final_search.csv", 5, name)
        summaries = {k: {"mean": statistics.mean(f[k] for f in all_folds),
                         "sample_sd": statistics.stdev(f[k] for f in all_folds)}
                     for k in all_folds[0]}
        for key in summaries:
            equal(summaries[key], saved["outer_summary"][key])
        equal(metrics(dev), saved["oof"]["metrics"])
        equal(metrics(test), saved["test"]["metrics"])
        report[name] = {"outer_folds": all_folds, "outer_summary": summaries, "oof": metrics(dev),
                        "test": metrics(test), "test_classes": dict(Counter(r["label"] for r in test)),
                        "final_fit_seconds": saved["final_fit"]["seconds"]}
print(json.dumps({"verified_classifier_threshold_runs": 12, "checked_measurement_rows": measured_rows,
                  "all_original_assignments_preserved": True, "results": result}, indent=2))
