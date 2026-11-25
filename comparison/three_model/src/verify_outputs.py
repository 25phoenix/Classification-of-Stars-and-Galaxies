"""Verify three-model exports from CSV rows with independent standard-library calculations."""
import csv
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
RESULTS = BASE / "results"
KEYS = ("random_forest", "frozen_mobilenetv2", "finetuned_mobilenetv2")
MODELS = dict(zip(KEYS, ("Random Forest", "Frozen MobileNetV2", "Fine-Tuned MobileNetV2")))
PAIRS = dict(zip("ABC", ((KEYS[0], KEYS[1]), (KEYS[0], KEYS[2]), (KEYS[1], KEYS[2]))))
CLASSES = range(10)


def require(condition, context):
    if not condition:
        raise ValueError(f"Verification failed: {context}")


def close(a, b, context):
    require(math.isclose(float(a), b, rel_tol=0, abs_tol=1e-12), context)


def close_p(a, b, context):
    require(math.isclose(float(a), b, rel_tol=1e-10, abs_tol=0), context)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_scores(paired, names):
    summary = {r["metric"]: r for r in read_csv(RESULTS / "comparison_metrics.csv")}
    details = read_csv(RESULTS / "per_class_comparison.csv")
    y = [int(row["true_label"]) for row in paired]
    support = Counter(y)
    results, predictions, matrices = {}, {}, {}
    require(len(details) == 10 and [r["class_name"] for r in details] == names, "class table order")
    for key, model in MODELS.items():
        prediction = predictions[key] = [int(r[key]) for r in paired]
        require(set(prediction) <= set(CLASSES), f"{model} invalid labels")
        matrix = matrices[key] = Counter(zip(y, prediction))
        predicted = Counter(prediction)
        require([r[key + "_correct"] for r in paired] == [str(i == j) for i, j in zip(y, prediction)],
                f"{model} correctness flags")
        raw = read_csv(RESULTS / f"confusion_{key}_counts.csv")
        normalized = read_csv(RESULTS / f"confusion_{key}_row_normalized.csv")
        require(len(raw) == len(normalized) == 10 and
                list(raw[0]) == list(normalized[0]) == ["true_class", *names], f"{model} matrix headers")
        pr, rc, f1 = [], [], []
        for i, name in enumerate(names):
            tp = matrix[i, i]
            precision = tp / predicted[i] if predicted[i] else 0
            recall = tp / support[i]
            score = 2 * tp / (support[i] + predicted[i])
            pr.append(precision)
            rc.append(recall)
            f1.append(score)
            row = details[i]
            require(int(row["support"]) == support[i] and int(row["class_id"]) == i, "class support")
            for metric, value in (("precision", precision), ("recall", recall), ("f1", score)):
                close(row[f"{key}_{metric}"], value, f"{model} {name} {metric}")
            require(raw[i]["true_class"] == normalized[i]["true_class"] == name, "matrix row label")
            for j, target in enumerate(names):
                require(int(raw[i][target]) == matrix[i, j], f"{model} confusion count")
                close(normalized[i][target], matrix[i, j] / support[i], f"{model} normalized confusion")
        results[key] = {"accuracy": sum(matrix[i, i] for i in CLASSES) / len(y),
                        "macro_f1": math.fsum(f1) / 10,
                        "weighted_f1": math.fsum(f1[i] * support[i] for i in CLASSES) / len(y),
                        "macro_precision": math.fsum(pr) / 10,
                        "macro_recall": math.fsum(rc) / 10, "balanced_accuracy": math.fsum(rc) / 10}
        for metric, value in results[key].items():
            close(summary[metric][model], value, f"{model} {metric}")
    for pair, (first, second) in PAIRS.items():
        for metric, row in summary.items():
            delta = results[second][metric] - results[first][metric]
            close(row[f"{pair}_difference"], delta, f"{pair} score differences")
            close(row[f"{pair}_percentage_points"], 100 * delta, f"{pair} percentage points")
        for row in details:
            for metric in ("precision", "recall", "f1"):
                delta = float(row[f"{second}_{metric}"]) - float(row[f"{first}_{metric}"])
                close(row[f"{pair}_{metric}_difference"], delta, f"{pair} class differences")
                close(row[f"{pair}_{metric}_percentage_points"], 100 * delta, f"{pair} class percentage points")
    return y, predictions, results, matrices


def exact_swap_probability(signed, observed):
    weights = [abs(value) for value in signed.values() if value]
    coefficients = [1] + [0] * sum(weights)
    extent = 0
    for weight in weights:
        extent += weight
        for i in range(extent, weight - 1, -1):
            coefficients[i] += coefficients[i - weight]
    return sum(count for i, count in enumerate(coefficients)
               if abs(2 * i - extent) >= abs(observed)) / (1 << len(weights))


def verify_pairs(paired, names, y, predictions, metrics, stats):
    pvalues = {}
    agreement_rows = {r["pair"]: r for r in read_csv(RESULTS / "pairwise_agreement.csv")}
    disagreed_rows = read_csv(RESULTS / "pairwise_prediction_disagreements.csv")
    for pair, (first, second) in PAIRS.items():
        comparison = stats["pairs"][pair]
        require(comparison["difference_direction"] == f"{MODELS[second]} minus {MODELS[first]}",
                "sign convention")
        a, b = predictions[first], predictions[second]
        correct = Counter((x == t, z == t) for t, x, z in zip(y, a, b))
        outcome = dict(zip(("both_correct", "first_only_correct", "second_only_correct", "both_wrong"),
                           [correct[True, True], correct[True, False],
                            correct[False, True], correct[False, False]]))
        require(comparison["paired_outcomes"] == outcome and all(
            int(agreement_rows[pair][field]) == value for field, value in outcome.items()),
            f"{pair} paired correctness")
        agreement = sum(i == j for i, j in zip(a, b))
        require(comparison["prediction_agreement_count"] == int(agreement_rows[pair]["agreement_count"])
                == agreement and comparison["disagreeing_labels"]
                == int(agreement_rows[pair]["disagreement_count"]) == len(y) - agreement,
                f"{pair} label agreement")
        close(comparison["prediction_agreement_fraction"], agreement / len(y), f"{pair} agreement rate")
        observed_matrix = Counter((names[i], names[j]) for i, j in zip(a, b) if i != j)
        saved = [r for r in disagreed_rows if r["pair"] == pair]
        require(len(saved) == 90 and observed_matrix == Counter(
            {(r["first_predicted_class"], r["second_predicted_class"]): int(r["count"]) for r in saved}),
            f"{pair} disagreements")
        c1, c2 = outcome["first_only_correct"], outcome["second_only_correct"]
        n = c1 + c2
        pvalue = pvalues[pair] = min(1., 2 * sum(math.comb(n, i)
                                                for i in range(min(c1, c2) + 1)) / (1 << n))
        test = comparison["mcnemar"]
        require((test["first_only_correct"], test["second_only_correct"], test["total_discordant"])
                == (c1, c2, n), f"{pair} McNemar discordance")
        close_p(test["p_value"], pvalue, f"{pair} McNemar p-value")
        signed = Counter()
        for row, t, x, z in zip(paired, y, a, b):
            signed[row["group_id"]] += (z == t) - (x == t)
        random_test = comparison["group_randomization_sensitivity"]
        require(random_test["observed_net_correct_second_minus_first"] == c2 - c1,
                f"{pair} randomization direction")
        close_p(random_test["p_value"], exact_swap_probability(signed, c2 - c1),
                f"{pair} exact group randomization")
        boot = comparison["preferred_group_bootstrap"]
        require(boot["groups"] == len(signed) == 3535 and boot["seed"] == 20261124
                and boot["replicates"] == 20000 and boot["confidence_level"] == .95, "bootstrap settings")
        for metric, interval in boot["intervals"].items():
            delta = metrics[second][metric] - metrics[first][metric]
            close(interval["estimate"], delta, f"{pair} {metric} CI sign and center")
            require(-1 <= interval["lower"] <= delta <= interval["upper"] <= 1,
                    f"{pair} {metric} CI limits")
    running = 0.
    for i, (pair, pvalue) in enumerate(sorted(pvalues.items(), key=lambda item: item[1])):
        running = max(running, min(1., (3 - i) * pvalue))
        close_p(stats["pairs"][pair]["mcnemar"]["holm_adjusted_p_value"], running, "Holm adjustment")
    phase3 = read_json(BASE.parent / "results/statistical_tests.json")
    a = stats["pairs"]["A"]
    require(a["preferred_group_bootstrap"]["intervals"] ==
            phase3["preferred_group_bootstrap"]["intervals"]
            and a["mcnemar"]["p_value"] == phase3["mcnemar"]["p_value"]
            and a["group_randomization_sensitivity"]["p_value"] ==
            phase3["group_randomization_sensitivity"]["p_value"], "exact Phase 3 reproduction")


def main():
    paired = read_csv(RESULTS / "aligned_predictions.csv")
    audit = read_json(BASE / "audit/alignment_audit.json")
    provenance = read_json(BASE / "audit/provenance.json")
    stats = read_json(RESULTS / "pairwise_statistical_tests.json")
    k = ROOT / "Kishore/results/final"
    names = [read_json(k / "class_mapping.json")[str(i)] for i in CLASSES]
    require(names == [audit["class_mapping"][str(i)] for i in CLASSES], "official class mapping")
    assignments = read_csv(k / "split_assignments.csv")
    test = {r["row_id"]: (r["label"], r["group_id"]) for r in assignments if r["partition"] == "test"}
    dev = {r["group_id"] for r in assignments if r["partition"] == "development"}
    require(len(paired) == len(test) == 3545
            and test == {r["row_id"]: (r["true_label"], r["group_id"]) for r in paired}, "locked ID/label/group join")
    groups = Counter(r["group_id"] for r in paired)
    require(not dev & groups.keys() and Counter(groups.values()) == {1: 3525, 2: 10}, "group integrity")
    exports = [(k / "test_predictions.csv", "label", "forest"),
               (ROOT / "Sagar/results/frozen_baseline_full/test_predictions.csv", "true_label", "predicted_label"),
               (ROOT / "Sagar/results/finetuned_mobilenetv2/finetuned_test_predictions.csv",
                "true_label", "predicted_label")]
    for key, (path, truth, field) in zip(KEYS, exports):
        rows = read_csv(path)
        orig = {r["row_id"]: (r[truth], r[field]) for r in rows}
        require(len(rows) == len(orig) == 3545
                and orig == {r["row_id"]: (r["true_label"], r[key]) for r in paired}, f"{key} source export")
    y, predictions, metrics, matrices = verify_scores(paired, names)
    verify_pairs(paired, names, y, predictions, metrics, stats)
    frozen, fine = (matrices[key] for key in KEYS[1:])
    ids = {name: i for i, name in enumerate(names)}
    changes = read_csv(RESULTS / "confusion_changes_after_finetuning.csv")
    require(len(changes) == 90, "off-diagonal confusion change count")
    for row in changes:
        i, j = ids[row["true_class"]], ids[row["predicted_class"]]
        expect = frozen[i, j] - fine[i, j]
        require(int(row["errors_avoided"]) == expect, "confusion error reduction")
        close(row["error_rate_reduction_pp"], 100 * expect / y.count(i), "confusion rate reduction")

    for filename, expected in provenance["inputs_sha256"].items():
        # Full HDF5 was read and hashed by analysis. Verify its recorded checksum against source metadata.
        value = read_json(k / "dataset.json")["sha256"] if filename.endswith(".h5") else digest(ROOT / filename)
        require(value == expected, f"input fingerprint {filename}")
    for filename, expected in provenance["computed_sha256"].items():
        require(digest(BASE / filename) == expected, f"output fingerprint {filename}")
    require(digest(BASE / "src/three_model_analysis.py") == provenance["analysis_source_sha256"],
            "analysis source fingerprint")
    figures = BASE / "figures"
    pngs = sorted(figures.glob("*.png"))
    require(len(pngs) == 8 and len(list(figures.glob("*.pdf"))) == 8, "eight figure pairs")
    for path in pngs:
        with path.open("rb") as stream:
            header = stream.read(24)
        require(header[:8] == b"\x89PNG\r\n\x1a\n" and min(struct.unpack(">II", header[16:24])) >= 1000,
                f"high-resolution PNG {path.name}")
        with path.with_suffix(".pdf").open("rb") as stream:
            require(stream.read(5) == b"%PDF-", f"PDF export {path.name}")
    print("PASS · 3,545 locked rows, 3,535 groups, all metrics and matrices, three exact test pairs, "
          "Holm correction, interval directions, exact Phase 3 match, fingerprints, and eight figure pairs.")


if __name__ == "__main__":
    main()
