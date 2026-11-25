"""Compare the two frozen Galaxy10 test exports without fitting either model."""

import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from collections import Counter
from pathlib import Path
from textwrap import fill

OUTPUT = Path(__file__).resolve().parents[1]
ROOT = OUTPUT.parent
# The existing contract loader lives in Sagar. Keep all imports read-only.
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, precision_score, recall_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from Sagar.src.data_contract import load_and_verify_contract

RF = "Random Forest"
MOBILE = "MobileNetV2"
LABELS = np.arange(10)
SEED = 20261124
REPLICATES = 20_000
COLORS = ["#2166ac", "#ba4a00"]

KISHORE = ROOT / "Kishore/results/final"
SAGAR = ROOT / "Sagar/results/frozen_baseline_full"


def require(condition, description):
    if not condition:
        raise ValueError(description)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def indexed_csv(path):
    table = pd.read_csv(path, float_precision="round_trip")
    require(pd.api.types.is_integer_dtype(table.row_id), f"Non-integer IDs in {path}")
    return table.set_index("row_id", verify_integrity=True).sort_index()


def verify_probabilities(table, label, prefix):
    values = table[[f"{prefix}{i}" for i in LABELS]].to_numpy()
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
            f"Invalid probabilities in {prefix}")
    error = float(abs(values.sum(axis=1) - 1).max())
    require(error < 1e-6 and np.array_equal(values.argmax(axis=1), table[label]),
            f"Probability-sum or argmax mismatch in {prefix}")
    return error


def align():
    dataset_path = KISHORE / "dataset.json"
    dataset = read_json(dataset_path)
    rf_meta = read_json(KISHORE / "model_metadata.json")
    rf_metrics = read_json(KISHORE / "metrics.json")
    mobile_meta = read_json(SAGAR / "model_config.json")
    mobile_prov = read_json(SAGAR / "provenance.json")
    manifest = read_json(ROOT / "Sagar/handoff/file_manifest.json")
    model = KISHORE / "model.joblib"
    h5 = ROOT / "Kishore/data/raw/Galaxy10_DECals.h5"
    features_path = ROOT / "Kishore" / dataset["feature_table"]
    split_path = KISHORE / "splits.json"
    assignments_path = KISHORE / "split_assignments.csv"

    print("Validating the Galaxy10 HDF5 and shared split contract.", flush=True)
    contract = load_and_verify_contract(h5, dataset_path, split_path, assignments_path)
    for name in ("dataset_sha256", "dataset_manifest_sha256", "splits_manifest_sha256",
                 "assignments_csv_sha256"):
        require(mobile_prov[name] == getattr(contract, name), f"Shared {name} mismatch")
    require(dataset["mode"] == rf_meta["mode"] == "official"
            and dataset["sha256"] == contract.dataset_sha256
            and dataset["usable"] == mobile_prov["eligible_rows"] == 17733
            and len(contract.development) == mobile_prov["development_rows"] == 14188
            and len(contract.test) == mobile_prov["test_rows"] == 3545,
            "Dataset identity or population mismatch")
    classes = {str(i): name for i, name in contract.class_names.items()}
    require(set(classes) == {str(i) for i in LABELS}
            and dataset["class_mapping"] == rf_meta["class_mapping"]
            == read_json(KISHORE / "class_mapping.json") == mobile_meta["class_names"] == classes,
            "Class mapping mismatch")
    split_plan = read_json(split_path)
    require(rf_meta["training_partition"] == "development"
            and rf_meta["development_size"] == 14188
            and rf_meta["split_manifest_sha256"] == contract.splits_manifest_sha256
            and split_plan["feature_fingerprint"] == dataset["fingerprint"]
            and rf_meta["provenance"] == dataset,
            "Random Forest training or feature provenance mismatch")
    require(mobile_meta["experiment_type"] == "frozen_mobilenetv2_baseline"
            and mobile_meta["mode"] == "full" and mobile_meta["weights"] == "imagenet",
            "Unexpected MobileNetV2 configuration")

    feature_order = rf_meta["feature_order"]
    feature_sha = digest(features_path)
    require(len(feature_order) == len(set(feature_order)) == 7
            and feature_order == dataset["features"]
            and rf_meta["settings"] == dataset["settings"]
            and feature_sha == dataset["feature_table_sha256"]
            and read_json(features_path.with_name("dataset.json")) == dataset,
            "Feature table, order, or preprocessing metadata mismatch")
    code_hashes = {name: digest(ROOT / "Kishore" / name) for name in dataset["code"]}
    require(code_hashes == dataset["code"]
            and code_hashes["morphology.py"] == rf_meta["feature_code_sha256"],
            "Kishore feature-generation source fingerprints differ")
    feature_table = indexed_csv(features_path)
    assignments = indexed_csv(assignments_path)
    test = assignments.loc[assignments.partition == "test", ["label", "group_id"]]
    development = assignments.loc[assignments.partition == "development"]
    require(set(test.index) == set(contract.test)
            and set(development.index) == set(contract.development)
            and not set(test.group_id) & set(development.group_id),
            "Saved partition or held-out group mismatch")
    with h5py.File(h5, "r") as file:
        labels = file["ans"][:]
    require(np.array_equal(labels[test.index], test.label), "HDF5 test labels mismatch")
    require(set(feature_table.index[feature_table.usable]) == set(contract.eligible_row_ids)
            and test.equals(feature_table.loc[test.index, ["label", "group_id"]])
            and np.isfinite(feature_table.loc[test.index, feature_order]).all().all(),
            "Cached test features or duplicate-object associations mismatch")

    rf_path = KISHORE / "test_predictions.csv"
    mobile_path = SAGAR / "test_predictions.csv"
    rf_hash, mobile_hash = digest(rf_path), digest(mobile_path)
    handoff_hashes = {item["relative_path"]: item["sha256"] for item in manifest["files"]}
    require(rf_hash == rf_metrics["prediction_sha256"]["test_predictions.csv"]
            and mobile_hash == handoff_hashes["results/frozen_baseline_full/test_predictions.csv"],
            "Frozen prediction hash mismatch")
    rf, mobile = indexed_csv(rf_path), indexed_csv(mobile_path)
    require(len(test) == 3545 and rf.index.equals(test.index) and mobile.index.equals(test.index),
            "Paired test IDs differ")
    require(test.equals(rf[["label", "group_id"]]) and test.label.equals(mobile.true_label),
            "Paired test labels or duplicate groups differ")
    sums = {RF: verify_probabilities(rf, "forest", "p_"),
            MOBILE: verify_probabilities(mobile, "predicted_label", "prob_")}
    paired = test.rename(columns={"label": "true_label"}).join(
        rf[["forest"]].rename(columns={"forest": "random_forest"}), validate="one_to_one"
    ).join(mobile[["predicted_label"]].rename(columns={"predicted_label": "mobilenetv2"}),
           validate="one_to_one")
    paired["random_forest_correct"] = paired.random_forest.eq(paired.true_label)
    paired["mobilenetv2_correct"] = paired.mobilenetv2.eq(paired.true_label)
    sizes = paired.groupby("group_id").size()
    audit = {
        "status": "PASS", "join_key": "row_id", "join": "validated one-to-one",
        "test_rows": len(paired), "eligible_rows": len(contract.eligible_row_ids),
        "development_rows": len(contract.development), "class_mapping": classes,
        "unique_test_ids_per_export": 3545, "missing_ids": 0, "duplicate_ids": 0,
        "true_label_mismatches": 0, "test_groups": len(sizes),
        "test_group_size_distribution": {int(s): int(n) for s, n in sizes.value_counts().items()},
        "shared_development_test_groups": 0, "eligible_labels_checked_against_hdf5": True,
        "test_group_ids_match_features_and_rf_export": True,
        "probability_argmax_matches": True, "probability_max_sum_error": sums,
        "development_oof_comparison": "Outside the held-out analysis scope",
        "rf_prediction_origin": "Saved forest column, verified against published SHA-256 and metrics",
        "test_inference_executed": False,
    }
    known_hashes = {
        h5: contract.dataset_sha256, dataset_path: contract.dataset_manifest_sha256,
        split_path: contract.splits_manifest_sha256, assignments_path: contract.assignments_csv_sha256,
        features_path: feature_sha, rf_path: rf_hash, mobile_path: mobile_hash,
        **{ROOT / "Kishore" / name: value for name, value in code_hashes.items()},
    }
    paths = [model, features_path.with_name("dataset.json"), KISHORE / "model_metadata.json",
             KISHORE / "metrics.json", KISHORE / "class_mapping.json",
             SAGAR / "head_model.keras", SAGAR / "test_metrics.json", SAGAR / "model_config.json",
             SAGAR / "provenance.json", ROOT / "Sagar/handoff/file_manifest.json",
             ROOT / "Sagar/handoff/HANDOFF_VERIFICATION.md",
             *(ROOT / "Sagar/src" / name for name in (
                 "config.py", "data_contract.py", "embeddings.py", "train_baseline.py"))]
    known_hashes.update({path: digest(path) for path in paths})
    for path in (SAGAR / "head_model.keras", SAGAR / "test_metrics.json",
                 SAGAR / "model_config.json", SAGAR / "provenance.json"):
        require(known_hashes[path] == handoff_hashes[str(path.relative_to(ROOT / "Sagar"))],
                f"Audited handoff hash differs for {path.name}")
    provenance = {
        "dataset": "Galaxy10 DECaLS", "inputs_sha256": {
            str(path.relative_to(ROOT)): value for path, value in sorted(known_hashes.items())
        },
        "analysis_source_sha256": digest(Path(__file__)),
        "rf_feature_order": feature_order, "rf_feature_settings": rf_meta["settings"],
        "rf_selected_parameters": rf_meta["selected_parameters"],
        "mobilenetv2_configuration": mobile_meta,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "h5py")},
        "bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES,
    }
    return paired, list(classes[str(i)] for i in LABELS), audit, provenance, rf_metrics


def metrics_for(y, prediction):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=LABELS, zero_division=0)
    overall = {
        "accuracy": accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, prediction, labels=LABELS, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "macro_precision": precision_score(
            y, prediction, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(y, prediction, labels=LABELS, average="macro", zero_division=0),
    }
    return overall, dict(precision=precision, recall=recall, f1=f1, support=support)


def normalized(matrix):
    return matrix / matrix.sum(axis=1, keepdims=True)


def scores_from_matrices(matrices):
    # Last axes are true and predicted label. Leave the leading replicate/model axes intact.
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    support = matrices.sum(axis=-1)
    denominators = support + matrices.sum(axis=-2)
    f1 = np.divide(2 * diagonal, denominators, where=denominators > 0,
                   out=np.zeros_like(diagonal, dtype=float))
    return np.stack((diagonal.sum(axis=-1) / support.sum(axis=-1), f1.mean(axis=-1)), axis=-1)


def bootstrap(paired):
    group_codes, group_ids = pd.factorize(paired.group_id, sort=True)
    y = paired.true_label.to_numpy()
    predictions = paired[["random_forest", "mobilenetv2"]].to_numpy()
    contributions = np.zeros((len(group_ids), 2, 10, 10), dtype=np.int64)
    for index in range(2):
        np.add.at(contributions, (group_codes, index, y, predictions[:, index]), 1)
    # Identical group contributions can be pooled into multinomial categories.
    # The resulting distribution is exactly that of drawing individual groups uniformly.
    patterns, frequencies = np.unique(contributions.reshape(-1, 200), axis=0, return_counts=True)
    rng = np.random.default_rng(SEED)
    differences = []
    for start in range(0, REPLICATES, 500):
        size = min(500, REPLICATES - start)
        weights = rng.multinomial(len(group_ids), frequencies / len(group_ids), size=size)
        scores = scores_from_matrices((weights @ patterns).reshape(-1, 2, 10, 10))
        differences.append(scores[:, 1] - scores[:, 0])
    return np.concatenate(differences)


def group_randomization(paired):
    difference = paired.mobilenetv2_correct.astype(int) - paired.random_forest_correct.astype(int)
    signed_sums = difference.groupby(paired.group_id).sum()
    weights = Counter(abs(int(v)) for v in signed_sums if v)
    # Convolve exact Binomial(m, 1/2) PMFs for sets of equally weighted groups.
    distribution = np.ones(1)
    for weight, count in sorted(weights.items()):
        pmf = np.zeros(weight * count + 1)
        pmf[::weight] = binom.pmf(np.arange(count + 1), count, 0.5)
        distribution = np.convolve(distribution, pmf)
    observed = int(difference.sum())
    scores = 2 * np.arange(len(distribution)) - (len(distribution) - 1)
    return {
        "method": "Exact two-sided within-group model-label swap for accuracy",
        "groups_with_nonzero_signed_difference": sum(weights.values()),
        "absolute_group_difference_frequencies": dict(sorted(weights.items())),
        "observed_net_correct_mobile_minus_rf": observed,
        "p_value": min(1.0, float(distribution[abs(scores) >= abs(observed)].sum()
                                 / distribution.sum())),
        "assumption": "Joint model labels are exchangeable within independent groups under the null.",
    }


def paired_statistics(paired, overall):
    rf = paired.random_forest_correct
    mobile = paired.mobilenetv2_correct
    outcomes = {
        "both_correct": int((rf & mobile).sum()), "rf_only_correct": int((rf & ~mobile).sum()),
        "mobile_only_correct": int((mobile & ~rf).sum()), "both_wrong": int((~rf & ~mobile).sum()),
    }
    agreement = int(paired.random_forest.eq(paired.mobilenetv2).sum())
    b, c = outcomes["rf_only_correct"], outcomes["mobile_only_correct"]
    differences = bootstrap(paired)
    limits = np.quantile(differences, [0.025, 0.975], axis=0, method="linear")
    ci = {name: {"estimate": float(overall[MOBILE][name] - overall[RF][name]),
                 "lower": float(limits[0, i]), "upper": float(limits[1, i])}
          for i, name in enumerate(("accuracy", "macro_f1"))}
    return {
        "difference_direction": "MobileNetV2 minus Random Forest",
        "rows": len(paired), "paired_outcomes": outcomes, "prediction_agreement_count": agreement,
        "prediction_agreement_fraction": agreement / len(paired),
        "mcnemar": {"method": "Exact two-sided conditional binomial test",
                    "rf_only_correct": b, "mobile_only_correct": c, "total_discordant": b + c,
                    "p_value": float(binomtest(c, b + c).pvalue) if b + c else 1.0,
                    "assumption": "Independent row pairs and equal orientations of discordance under the null."},
        "preferred_group_bootstrap": {
            "seed": SEED, "rng": "NumPy default_rng (PCG64)", "replicates": REPLICATES,
            "groups": int(paired.group_id.nunique()), "confidence_level": 0.95,
            "method": "Percentile intervals. Uniformly sample groups with replacement, drawing the original number of groups. Retain all rows and both predictions in each sampled group. Replicate row counts may vary.",
            "estimand": "Row-weighted accuracy and fixed-ten-class Macro-F1 differences, conditional on the fitted models.",
            "intervals": ci,
        },
        "group_randomization_sensitivity": group_randomization(paired),
    }


def save_figure(figures, name, fig):
    fig.savefig(figures / f"{name}.png", dpi=330, facecolor="white")
    fig.savefig(figures / f"{name}.pdf", facecolor="white",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def metric_bars(figures, name, labels, values, xlabel, title):
    fig, ax = plt.subplots(figsize=(9.5, 0.48 * len(labels) + 1.8), layout="constrained")
    y = np.arange(len(labels))
    for index, (model, color) in enumerate(zip((RF, MOBILE), COLORS)):
        bars = ax.barh(y + (index - 0.5) * 0.34, values[index], 0.32, label=model, color=color)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values[index]], fontsize=8, padding=3)
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 1.06), xlabel=xlabel, title=title)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2, frameon=False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    save_figure(figures, name, fig)


def plot_all(figures, names, overall, per_class, matrices, agreement, stats):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.titleweight": "bold"})
    metrics = list(overall[RF])
    display = ["Accuracy", "Macro-F1", "Weighted-F1", "Balanced accuracy",
               "Macro-precision", "Macro-recall"]
    metric_bars(figures, "overall_metrics", display,
                [[overall[m][k] for k in metrics] for m in (RF, MOBILE)],
                "Score", "Held-out Galaxy10 performance")
    for metric, title in (("f1", "F1"), ("recall", "recall")):
        metric_bars(figures, f"per_class_{metric}", names,
                    [per_class[m][metric] for m in (RF, MOBILE)], title.capitalize(),
                    f"Held-out {title} by Galaxy10 class")

    wrapped = [fill(name, width=19) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), layout="constrained")
    for ax, model in zip(axes, (RF, MOBILE)):
        matrix = normalized(matrices[model])
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set(title=model, ylabel="True class", xlabel="Predicted class")
        ax.set_xticks(LABELS, wrapped, rotation=55, ha="right", rotation_mode="anchor", fontsize=8)
        ax.set_yticks(LABELS, wrapped, fontsize=8)
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > 0.55 else "#17202a")
    fig.colorbar(image, ax=axes, shrink=0.7, format=PercentFormatter(1),
                 label="Share of true class")
    fig.suptitle("Test confusion matrices · normalized by true class")
    save_figure(figures, "confusion_matrices", fig)

    fig, ax = plt.subplots(figsize=(9.5, 8), layout="constrained")
    off_diag = agreement.copy()
    np.fill_diagonal(off_diag, 0)
    image = ax.imshow(off_diag, cmap="Purples")
    ax.set(title="Disagreements in predicted labels", xlabel=f"{MOBILE} prediction",
           ylabel=f"{RF} prediction")
    ax.set_xticks(LABELS, wrapped, rotation=55, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticks(LABELS, wrapped, fontsize=8)
    for i, j in np.ndindex(agreement.shape):
        ax.text(j, i, "—" if i == j else str(off_diag[i, j]), ha="center", va="center",
                color="white" if off_diag[i, j] > off_diag.max() * 0.6 else "#17202a", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.7, label="Test galaxies")
    save_figure(figures, "prediction_disagreements", fig)

    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    labels = ["Both correct", f"{RF} only correct", f"{MOBILE} only correct", "Both wrong"]
    counts = list(stats["paired_outcomes"].values())
    bars = ax.barh(labels, counts, color=["#327757", *COLORS, "#666666"])
    ax.bar_label(bars, labels=[f"{n:,} ({n/stats['rows']:.1%})" for n in counts], padding=4)
    ax.set(xlim=(0, max(counts) * 1.30), xlabel="Test galaxies",
           title="Paired correctness outcomes")
    ax.invert_yaxis()
    save_figure(figures, "paired_outcomes", fig)

    intervals = stats["preferred_group_bootstrap"]["intervals"]
    fig, ax = plt.subplots(figsize=(8, 3.3), layout="constrained")
    for y, metric in enumerate(("accuracy", "macro_f1")):
        row = intervals[metric]
        estimate, low, high = (100 * row[key] for key in ("estimate", "lower", "upper"))
        ax.hlines(y, low, high, color=COLORS[1], linewidth=2)
        ax.plot(estimate, y, "o", color=COLORS[1])
        ax.annotate(f"{estimate:+.2f} [{low:+.2f}, {high:+.2f}]",
                    (estimate, y), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=9)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set(yticks=[0, 1], yticklabels=["Accuracy", "Macro-F1"], ylim=(1.55, -0.55),
           xlabel="MobileNetV2 − Random Forest (percentage points)",
           title="Paired differences with 95% group bootstrap intervals")
    ax.margins(x=0.18)
    save_figure(figures, "paired_difference_intervals", fig)


def export(results, names, paired, overall, per_class, matrices, agreement, stats):
    paired.to_csv(results / "paired_predictions.csv")
    overall_table = pd.DataFrame(overall).rename_axis("metric")
    overall_table["difference_mobile_minus_rf"] = overall_table[MOBILE] - overall_table[RF]
    overall_table["relative_difference_percent"] = (
        100 * overall_table.difference_mobile_minus_rf / overall_table[RF].replace(0, np.nan))
    overall_table.to_csv(results / "comparison_metrics.csv")
    details = pd.DataFrame({"class_id": LABELS, "class_name": names,
                            "support": per_class[RF]["support"]})
    confusions = []
    for key, model in (("rf", RF), ("mobile", MOBILE)):
        matrix = matrices[model]
        for mode, values in (("counts", matrix), ("row_normalized", normalized(matrix))):
            pd.DataFrame(values, index=names, columns=names).rename_axis(
                "true_class").to_csv(results / f"confusion_{key}_{mode}.csv")
        for metric in ("precision", "recall", "f1"):
            details[f"{key}_{metric}"] = per_class[model][metric]
        for i, j in np.ndindex(matrix.shape):
            if i != j:
                confusions.append((model, i, names[i], j, names[j], matrix[i, j],
                                   matrix[i, j] / matrix[i].sum()))
    for metric in ("precision", "recall", "f1"):
        details[f"difference_{metric}"] = details[f"mobile_{metric}"] - details[f"rf_{metric}"]
        details[f"relative_{metric}_difference_percent"] = (
            100 * details[f"difference_{metric}"] / details[f"rf_{metric}"].replace(0, np.nan))
    details["accuracy_delta_contribution"] = (
        matrices[MOBILE].diagonal() - matrices[RF].diagonal()) / len(paired)
    details.to_csv(results / "per_class_comparison.csv", index=False)
    pd.DataFrame(confusions, columns=["model", "true_id", "true_class", "predicted_id",
                                    "predicted_class", "count", "rate_within_true_class"]
                 ).sort_values(["model", "count"], ascending=[True, False], kind="stable"
                               ).to_csv(results / "off_diagonal_confusions.csv", index=False)
    disagreements = [(i, names[i], j, names[j], int(agreement[i, j]))
                     for i, j in np.ndindex(agreement.shape) if i != j]
    pd.DataFrame(disagreements, columns=["rf_id", "rf_class", "mobile_id", "mobile_class", "count"]
                 ).sort_values(["count", "rf_id", "mobile_id"], ascending=[False, True, True]
                               ).to_csv(results / "disagreement_analysis.csv", index=False)
    pd.DataFrame(agreement, index=names, columns=names).rename_axis(
        "random_forest_predicted_class").to_csv(results / "prediction_agreement_matrix.csv")
    write_json(results / "statistical_tests.json", stats)


def main():
    if sys.flags.optimize:
        raise SystemExit("Run without Python optimization so the shared contract checks are active.")
    try:
        paired, names, audit, provenance, rf_saved = align()
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
        write_json(OUTPUT / "audit/alignment_audit.json",
                   {"status": "FAIL", "error": str(error), "analysis_executed": False})
        raise
    print("Paired test alignment passed. Computing metrics and group resampling.", flush=True)
    y = paired.true_label
    overall, per_class, matrices = {}, {}, {}
    for model, column in ((RF, "random_forest"), (MOBILE, "mobilenetv2")):
        overall[model], per_class[model] = metrics_for(y, paired[column])
        matrices[model] = confusion_matrix(y, paired[column], labels=LABELS)
    for key, saved in rf_saved["test"]["forest"]["metrics"].items():
        require(np.isclose(saved, overall[RF][key], rtol=0, atol=1e-12), f"RF {key} mismatch")
    require(np.array_equal(matrices[RF], rf_saved["test"]["forest"]["confusion_counts"]),
            "Random Forest saved confusion counts differ")
    require(np.allclose(normalized(matrices[RF]), rf_saved["test"]["forest"]["confusion_normalized"],
                        rtol=0, atol=1e-12), "Random Forest saved normalized confusion matrix differs")
    mobile_saved = read_json(SAGAR / "test_metrics.json")
    for key in ("accuracy", "macro_f1"):
        require(np.isclose(mobile_saved[key], overall[MOBILE][key], rtol=0, atol=1e-12),
                f"MobileNetV2 {key} mismatch")
    require(mobile_saved["num_samples"] == len(paired), "MobileNetV2 saved sample count differs")
    audit["rf_saved_metrics_and_confusion_match"] = True
    audit["mobile_saved_accuracy_macro_f1_and_count_match"] = True

    agreement = confusion_matrix(paired.random_forest, paired.mobilenetv2, labels=LABELS)
    stats = paired_statistics(paired, overall)
    # Stage outputs so a plotting or export error leaves any prior run intact.
    with tempfile.TemporaryDirectory(prefix=".comparison-", dir=OUTPUT) as directory:
        stage = Path(directory)
        for name in ("results", "figures", "audit"):
            (stage / name).mkdir()
        export(stage / "results", names, paired, overall, per_class, matrices, agreement, stats)
        plot_all(stage / "figures", names, overall, per_class, matrices, agreement, stats)
        write_json(stage / "audit/alignment_audit.json", audit)
        write_json(stage / "audit/provenance.json", provenance)
        for path in sorted(stage.glob("*/*")):
            path.replace(OUTPUT / path.relative_to(stage))
    print(json.dumps({"metrics": overall, "statistics": stats}, indent=2))


if __name__ == "__main__":
    main()
