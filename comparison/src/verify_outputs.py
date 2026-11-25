"""Independently verify the exported comparison using CSV counts and stdlib math."""

import csv
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1]
RESULTS = OUTPUT / "results"


def read_rows(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def check(condition, context):
    if not condition:
        raise ValueError(f"Verification failed: {context}")


def equal(actual, expected, context):
    check(math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-11), context)


def main():
    paired = read_rows("paired_predictions.csv")
    classes = read_rows("per_class_comparison.csv")
    stats = json.loads((RESULTS / "statistical_tests.json").read_text())
    overall = {row["metric"]: row for row in read_rows("comparison_metrics.csv")}
    audit = json.loads((OUTPUT / "audit/alignment_audit.json").read_text())
    names = [row["class_name"] for row in classes]
    n = len(paired)
    check(n == len({row["row_id"] for row in paired}) == audit["test_rows"] == 3545, "paired IDs")
    check(len(names) == 10 and names == [audit["class_mapping"][str(i)] for i in range(10)],
          "official class ordering")
    y = [int(row["true_label"]) for row in paired]
    support = Counter(y)
    computed = {}
    for key, model, field in (("rf", "Random Forest", "random_forest"),
                               ("mobile", "MobileNetV2", "mobilenetv2")):
        pred = [int(row[field]) for row in paired]
        check([row[field + "_correct"] for row in paired]
              == [str(a == b) for a, b in zip(y, pred)], f"{model} flags")
        pairs = Counter(zip(y, pred))
        frequencies = Counter(pred)
        raw = read_rows(f"confusion_{key}_counts.csv")
        norm = read_rows(f"confusion_{key}_row_normalized.csv")
        precisions, recalls, f1s = [], [], []
        for i, row in enumerate(classes):
            check(int(row["class_id"]) == i and int(row["support"]) == support[i], "class support")
            tp = pairs[i, i]
            precision = tp / frequencies[i] if frequencies[i] else 0
            recall = tp / support[i]
            f1 = 2 * tp / (support[i] + frequencies[i])
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            equal(row[f"{key}_precision"], precision, f"{model} precision {i}")
            equal(row[f"{key}_recall"], recall, f"{model} recall {i}")
            equal(row[f"{key}_f1"], f1, f"{model} F1 {i}")
            check(raw[i]["true_class"] == norm[i]["true_class"] == names[i], "matrix label")
            for j, name in enumerate(names):
                check(int(raw[i][name]) == pairs[i, j], f"{model} confusion {i}, {j}")
                equal(norm[i][name], pairs[i, j] / support[i], "normalized confusion")
        computed[model] = {
            "accuracy": sum(pairs[i, i] for i in range(10)) / n,
            "macro_precision": math.fsum(precisions) / 10,
            "macro_recall": math.fsum(recalls) / 10,
            "balanced_accuracy": math.fsum(recalls) / 10,
            "macro_f1": math.fsum(f1s) / 10,
            "weighted_f1": math.fsum(support[i] * f1s[i] for i in range(10)) / n,
        }
        for metric, value in computed[model].items():
            equal(overall[metric][model], value, f"{model} {metric}")

    rf, mobile = computed.values()
    for metric, row in overall.items():
        difference = mobile[metric] - rf[metric]
        equal(row["difference_mobile_minus_rf"], difference, "overall difference")
        equal(row["relative_difference_percent"], 100 * difference / rf[metric],
              "relative difference")
    equal(sum(float(row["accuracy_delta_contribution"]) for row in classes),
          mobile["accuracy"] - rf["accuracy"], "class accuracy contributions")
    for row in classes:
        for metric in ("precision", "recall", "f1"):
            difference = float(row[f"mobile_{metric}"]) - float(row[f"rf_{metric}"])
            equal(row[f"difference_{metric}"], difference, "per-class difference")

    pairs = Counter((int(row["random_forest"]), int(row["mobilenetv2"])) for row in paired)
    outcomes = Counter((int(row["random_forest"]) == label, int(row["mobilenetv2"]) == label)
                       for row, label in zip(paired, y))
    for key, pair in (("both_correct", (True, True)), ("rf_only_correct", (True, False)),
                      ("mobile_only_correct", (False, True)), ("both_wrong", (False, False))):
        check(stats["paired_outcomes"][key] == outcomes[pair], key)
    agreement = sum(pairs[i, i] for i in range(10))
    check(stats["prediction_agreement_count"] == agreement, "agreement count")
    equal(stats["prediction_agreement_fraction"], agreement / n, "agreement rate")
    saved_agreement = read_rows("prediction_agreement_matrix.csv")
    for i, row in enumerate(saved_agreement):
        check(row["random_forest_predicted_class"] == names[i], "agreement row label")
        for j, name in enumerate(names):
            check(int(row[name]) == pairs[i, j], "agreement matrix count")
    disagreement = read_rows("disagreement_analysis.csv")
    check(len(disagreement) == 90 and all(
        int(row["count"]) == pairs[int(row["rf_id"]), int(row["mobile_id"])]
        for row in disagreement), "off-diagonal disagreement counts")
    check(sum(int(row["count"]) for row in disagreement) == n - agreement, "disagreement total")

    b, c = outcomes[True, False], outcomes[False, True]
    p = min(1, 2 * sum(math.comb(b + c, i) for i in range(min(b, c) + 1)) / (1 << (b + c)))
    equal(stats["mcnemar"]["p_value"], p, "exact McNemar")
    groups = Counter(row["group_id"] for row in paired)
    group_sizes = {str(size): amount for size, amount in Counter(groups.values()).items()}
    check(group_sizes == audit["test_group_size_distribution"], "group sizes")
    signed = Counter()
    for row, label in zip(paired, y):
        signed[row["group_id"]] += (int(row["mobilenetv2"]) == label) - (
            int(row["random_forest"]) == label)
    # Independent integer count of sign allocations, using subset sums.
    weights = [abs(value) for value in signed.values() if value]
    coefficients = [1] + [0] * sum(weights)
    extent = 0
    for weight in weights:
        extent += weight
        for i in range(extent, weight - 1, -1):
            coefficients[i] += coefficients[i - weight]
    tail = sum(count for i, count in enumerate(coefficients)
               if abs(2 * i - extent) >= abs(c - b))
    equal(stats["group_randomization_sensitivity"]["p_value"], tail / (1 << len(weights)),
          "exact group randomization")
    boot = stats["preferred_group_bootstrap"]
    check(boot["groups"] == len(groups) and boot["replicates"] >= 10000, "bootstrap settings")
    for metric, ci in boot["intervals"].items():
        equal(ci["estimate"], mobile[metric] - rf[metric], f"{metric} bootstrap estimate")
        check(-1 <= ci["lower"] <= ci["upper"] <= 1, f"{metric} interval")

    figures = OUTPUT / "figures"
    stems = ("overall_metrics", "per_class_f1", "per_class_recall", "confusion_matrices",
             "prediction_disagreements", "paired_outcomes", "paired_difference_intervals")
    for stem in stems:
        png = (figures / f"{stem}.png").read_bytes()
        check(png[:8] == b"\x89PNG\r\n\x1a\n" and min(struct.unpack(">II", png[16:24])) > 1000,
              f"{stem} high-resolution PNG")
        check((figures / f"{stem}.pdf").read_bytes().startswith(b"%PDF-"), f"{stem} PDF export")

    provenance = json.loads((OUTPUT / "audit/provenance.json").read_text())
    path = OUTPUT / "src/comparison_analysis.py"
    check(hashlib.sha256(path.read_bytes()).hexdigest() == provenance["analysis_source_sha256"],
          "analysis source fingerprint")
    print("PASS: independently verified both models’ metrics, per-class scores and matrices, "
          "agreement counts, exact p-values, bootstrap settings, and seven figure pairs.")


if __name__ == "__main__":
    main()
