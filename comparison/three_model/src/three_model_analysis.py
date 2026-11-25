"""Audit and compare three saved Galaxy10 test exports. No model is loaded."""
import sys
import tempfile
import zipfile
from pathlib import Path
from textwrap import fill

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(OUT.parent / "src"))

import comparison_analysis as prior
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix

plt = prior.plt
check = prior.require
FINE = ROOT / "Sagar/results/finetuned_mobilenetv2"
ARCHIVE = ROOT / "Sagar/Sagar_FineTuned_Only_For_Kishore.zip"
KEYS = ("random_forest", "frozen_mobilenetv2", "finetuned_mobilenetv2")
MODELS = dict(zip(KEYS, ("Random Forest", "Frozen MobileNetV2", "Fine-Tuned MobileNetV2")))
PAIRS = dict(zip("ABC", ((KEYS[0], KEYS[1]), (KEYS[0], KEYS[2]), (KEYS[1], KEYS[2]))))
COLORS = ("#2166ac", "#ba4a00", "#087c70")
LABELS = prior.LABELS


def verify_archive():
    inventory = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        check(archive.testzip() is None, "Fine-tuned ZIP CRC failure")
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = FINE / Path(info.filename).name
            check(path.name not in inventory, f"Duplicate archive basename {path.name}")
            check(path.read_bytes() == archive.read(info), f"Archive/extracted mismatch {path.name}")
            inventory[path.name] = {"archive_member": info.filename, "size_bytes": info.file_size,
                                    "sha256": prior.digest(path)}
    check(len(inventory) == 11, "Unexpected fine-tuned archive member count")
    return inventory


def align():
    inventory = verify_archive()
    paired, names, audit, provenance, rf_saved = prior.align()
    fine_path = FINE / "finetuned_test_predictions.csv"
    fine = prior.indexed_csv(fine_path)
    check(fine.index.equals(paired.index) and len(fine) == 3545,
          f"Fine-tuned test IDs differ, {len(paired.index.difference(fine.index))} missing and "
          f"{len(fine.index.difference(paired.index))} unexpected")
    check(paired.true_label.equals(fine.true_label), "Fine-tuned truth labels differ for row IDs "
          f"{list(paired.index[paired.true_label.ne(fine.true_label)])[:10]}")
    check(pd.api.types.is_integer_dtype(fine.predicted_label) and fine.predicted_label.isin(LABELS).all(),
          "Fine-tuned predicted labels outside the ten classes")
    error = prior.verify_probabilities(fine, "predicted_label", "prob_")
    meta = prior.read_json(FINE / "finetune_config_and_timings.json")
    frozen = prior.read_json(prior.SAGAR / "model_config.json")
    expected = {"experiment_type": "optional_two_stage_end_to_end_finetuning", "batch_size": 16,
                "warmup_epochs": 5, "warmup_lr": .001, "unfreeze_top_layers": 40,
                "unfrozen_non_bn_layers": 27, "finetune_lr": .0001, "selected_finetune_epochs": 15,
                "use_class_weight": False, "mixed_precision_float16": True,
                "augmentation_active_training_only": True, "batchnorm_frozen": True,
                "preprocessing": frozen["preprocessing"]}
    check(all(meta[k] == v for k, v in expected.items()), "Fine-tuning metadata differs")
    check(np.isclose(meta["val_search_seconds"] + meta["final_refit_seconds"],
                     meta["total_finetuning_seconds"], rtol=0, atol=1e-9), "Timing total differs")
    assignments = prior.indexed_csv(prior.KISHORE / "split_assignments.csv")
    plan = prior.read_json(prior.KISHORE / "splits.json")
    fold = plan["final_tuning"][0]
    train, val = set(fold["train"]), set(fold["validation"])
    check(not train & val and train | val == set(plan["development"])
          and not (train | val) & set(plan["test"]), "Development epoch-selection membership differs")
    val_groups = set(assignments.loc[sorted(val)].group_id)
    check(not val_groups & set(assignments.loc[sorted(train)].group_id)
          and not val_groups & set(paired.group_id), "Development epoch-selection group overlap")
    check(audit["test_groups"] == 3535 and audit["test_group_size_distribution"] == {1: 3525, 2: 10},
          "Unexpected held-out group structure")
    paired = paired.rename(columns={"mobilenetv2": KEYS[1], "mobilenetv2_correct": KEYS[1] + "_correct"})
    paired = paired.join(fine[["predicted_label"]].rename(columns={"predicted_label": KEYS[2]}),
                         validate="one_to_one")
    paired[KEYS[2] + "_correct"] = paired[KEYS[2]].eq(paired.true_label)
    for key in KEYS:
        check(paired[key].isin(LABELS).all(), f"Missing or invalid prediction for {key}")
    audit.update({"models": list(MODELS.values()), "fine_tuned_members_match_zip": True,
                  "compared_prediction_exports": 3, "exact_locked_test_membership_and_labels": True,
                  "fine_tuned_selection_fold": {"index": 0, "train_rows": len(train),
                                                "validation_rows": len(val), "group_overlap": 0},
                  "fine_tuned_training_evidence": "Archived addendum and config; training weights, OOF, run code and independent split hashes absent.",
                  "class_mapping_matches_all_sources": True})
    audit["probability_max_sum_error"][MODELS[KEYS[2]]] = error
    inputs = provenance["inputs_sha256"]
    for name, member in inventory.items():
        inputs[str((FINE / name).relative_to(ROOT))] = member["sha256"]
    for path in (ARCHIVE, OUT.parent / "results/statistical_tests.json",
                 OUT.parent / "src/comparison_analysis.py", ROOT / "Sagar/src/finetune_optional.py"):
        inputs[str(path.relative_to(ROOT))] = prior.digest(path)
    provenance.update({"analysis_source_sha256": prior.digest(Path(__file__)),
                       "fine_tuned_configuration": meta, "fine_tuned_archive_members": inventory,
                       "fine_tuned_archive_original": "/mnt/c/Users/kisho/Downloads/" + ARCHIVE.name,
                       "phase3_reused": "Shared data and baseline audit, metric functions, pairwise bootstrap and exact group randomization",
                       "bootstrap_streams": "Each pair starts PCG64 at the Phase 3 seed. Pair A exactly reproduces Phase 3. Intervals are marginal."})
    return paired, names, audit, provenance, rf_saved


def verify_reported_scores(overall, details, matrices, names, rf_saved, audit):
    rf = rf_saved["test"]["forest"]
    for metric, value in rf["metrics"].items():
        check(np.isclose(overall[KEYS[0]][metric], value, rtol=0, atol=1e-12), f"RF {metric} mismatch")
    check(np.array_equal(matrices[KEYS[0]], rf["confusion_counts"])
          and np.allclose(prior.normalized(matrices[KEYS[0]]), rf["confusion_normalized"],
                          rtol=0, atol=1e-12), "RF archived confusion matrix mismatch")
    official_headers = [f"{i}: {name}" for i, name in enumerate(names)]
    for key, directory, prefix in ((KEYS[1], prior.SAGAR, ""), (KEYS[2], FINE, "finetuned_")):
        saved = prior.read_json(directory / f"{prefix}test_metrics.json")
        check(saved["num_samples"] == 3545, f"{key} archived sample mismatch")
        for metric in ("accuracy", "macro_f1"):
            check(np.isclose(overall[key][metric], saved[metric], rtol=0, atol=1e-12),
                  f"{key} archived {metric} mismatch")
        csv_classes = pd.read_csv(directory / f"{prefix}test_per_class_metrics.csv").to_dict("records")
        for rows in (saved["per_class"], csv_classes):
            check(len(rows) == 10, f"{key} archived class count")
            for i, row in enumerate(rows):
                check(row["class_id"] == i and row["class_name"] == names[i]
                      and all(np.isclose(row[m], details[key][m][i], rtol=0, atol=1e-12)
                              for m in ("precision", "recall", "f1", "support")),
                      f"{key} archived per-class metrics mismatch for class {i}")
        for mode, values in (("raw", matrices[key]), ("normalized", prior.normalized(matrices[key]))):
            matrix = pd.read_csv(directory / f"{prefix}test_confusion_matrix_{mode}.csv", index_col=0)
            check(list(matrix.index) == list(matrix.columns) == official_headers
                  and np.allclose(matrix.to_numpy(), values, rtol=0, atol=1e-12),
                  f"{key} archived {mode} confusion matrix mismatch")
    audit["source_metrics_and_confusions_match"] = True


def pairwise(paired, overall, details, names):
    results = {}
    disagreements = []
    for pair, (first, second) in PAIRS.items():
        # Phase 3 helper expects its original two prediction column names.
        frame = paired[["group_id", "true_label", first, second, first + "_correct",
                        second + "_correct"]].copy()
        frame.columns = ["group_id", "true_label", "random_forest", "mobilenetv2",
                         "random_forest_correct", "mobilenetv2_correct"]
        stats = prior.paired_statistics(frame, {prior.RF: overall[first], prior.MOBILE: overall[second]})
        stats["difference_direction"] = f"{MODELS[second]} minus {MODELS[first]}"
        stats["first_model"], stats["second_model"] = MODELS[first], MODELS[second]
        outcome = stats["paired_outcomes"]
        outcome["first_only_correct"] = outcome.pop("rf_only_correct")
        outcome["second_only_correct"] = outcome.pop("mobile_only_correct")
        test = stats["mcnemar"]
        test["first_only_correct"] = test.pop("rf_only_correct")
        test["second_only_correct"] = test.pop("mobile_only_correct")
        randomization = stats["group_randomization_sensitivity"]
        randomization["observed_net_correct_second_minus_first"] = randomization.pop(
            "observed_net_correct_mobile_minus_rf")
        stats["disagreeing_labels"] = len(paired) - stats["prediction_agreement_count"]
        diff = details[second]["f1"] - details[first]["f1"]
        stats["largest_class_f1_differences"] = [
            {"class_id": int(i), "class_name": names[i], "difference": float(diff[i])}
            for i in np.argsort(-abs(diff), kind="stable")[:3]]
        cross = confusion_matrix(paired[first], paired[second], labels=LABELS)
        for i, j in np.ndindex(cross.shape):
            if i != j:
                disagreements.append((pair, names[i], names[j], int(cross[i, j])))
        results[pair] = stats
    cumulative = 0.0
    for rank, pair in enumerate(sorted(results, key=lambda p: results[p]["mcnemar"]["p_value"])):
        test = results[pair]["mcnemar"]
        cumulative = max(cumulative, min(1.0, (len(results) - rank) * test["p_value"]))
        test["holm_adjusted_p_value"] = cumulative
    previous = prior.read_json(OUT.parent / "results/statistical_tests.json")
    a = results["A"]
    check(a["preferred_group_bootstrap"]["intervals"]
          == previous["preferred_group_bootstrap"]["intervals"]
          and a["mcnemar"]["p_value"] == previous["mcnemar"]["p_value"]
          and a["group_randomization_sensitivity"]["p_value"]
          == previous["group_randomization_sensitivity"]["p_value"], "Phase 3 statistical rerun differs")
    return results, disagreements


def tables(folder, paired, names, overall, details, matrices, pairs, disagreements):
    paired.to_csv(folder / "aligned_predictions.csv")
    metrics = pd.DataFrame({MODELS[k]: overall[k] for k in KEYS}).rename_axis("metric")
    classes = pd.DataFrame({"class_id": LABELS, "class_name": names, "support": details[KEYS[0]]["support"]})
    for key in KEYS:
        matrix = matrices[key]
        for mode, values in (("counts", matrix), ("row_normalized", prior.normalized(matrix))):
            pd.DataFrame(values, index=names, columns=names).rename_axis("true_class").to_csv(
                folder / f"confusion_{key}_{mode}.csv")
        for metric in ("precision", "recall", "f1"):
            classes[f"{key}_{metric}"] = details[key][metric]
    for pair, (first, second) in PAIRS.items():
        diff = metrics[MODELS[second]] - metrics[MODELS[first]]
        metrics[f"{pair}_difference"] = diff
        metrics[f"{pair}_percentage_points"] = 100 * diff
        for metric in ("precision", "recall", "f1"):
            diff = classes[f"{second}_{metric}"] - classes[f"{first}_{metric}"]
            classes[f"{pair}_{metric}_difference"] = diff
            classes[f"{pair}_{metric}_percentage_points"] = 100 * diff
    metrics.to_csv(folder / "comparison_metrics.csv")
    classes.to_csv(folder / "per_class_comparison.csv", index=False)
    pd.DataFrame([{"pair": p, "first_model": s["first_model"], "second_model": s["second_model"],
                   **s["paired_outcomes"], "agreement_count": s["prediction_agreement_count"],
                   "agreement_fraction": s["prediction_agreement_fraction"],
                   "disagreement_count": s["disagreeing_labels"],
                   "accuracy_difference": s["preferred_group_bootstrap"]["intervals"]["accuracy"]["estimate"]}
                  for p, s in pairs.items()]).to_csv(folder / "pairwise_agreement.csv", index=False)
    pd.DataFrame(disagreements, columns=["pair", "first_predicted_class", "second_predicted_class",
                                         "count"]).sort_values(["pair", "count"], ascending=[True, False],
                                                               kind="stable").to_csv(
        folder / "pairwise_prediction_disagreements.csv", index=False)
    frozen, fine = (matrices[k] for k in KEYS[1:])
    changes = pd.DataFrame([
        {"true_class": names[i], "predicted_class": names[j], "support": int(frozen[i].sum()),
         "frozen_errors": int(frozen[i, j]), "fine_tuned_errors": int(fine[i, j]),
         "errors_avoided": int(frozen[i, j] - fine[i, j]),
         "error_rate_reduction_pp": 100 * (frozen[i, j] - fine[i, j]) / frozen[i].sum()}
        for i, j in np.ndindex(frozen.shape) if i != j
    ]).sort_values("errors_avoided", ascending=False, kind="stable")
    changes.to_csv(folder / "confusion_changes_after_finetuning.csv", index=False)
    prior.write_json(folder / "pairwise_statistical_tests.json", {
        "holm_family": "Three two-sided exact row-level McNemar accuracy tests (A, B, C).",
        "phase3_pair_a": "Exact agreement for p-values and bootstrap interval endpoints.",
        "interval_adjustment": "Marginal 95% percentile intervals; no multiplicity adjustment.",
        "pairs": pairs})
    return changes


def bars(folder, name, rows, values, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, 1.8 + .53 * len(rows)), layout="constrained")
    y = np.arange(len(rows))
    for i, (key, color) in enumerate(zip(KEYS, COLORS)):
        bars = ax.barh(y + (i - 1) * .24, values[key], .23, color=color, label=MODELS[key])
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values[key]], padding=3, fontsize=8)
    ax.set(yticks=y, yticklabels=rows, xlim=(0, 1.07), xlabel=xlabel, title=title)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", alpha=.22)
    ax.set_axisbelow(True)
    fig.legend(loc="outside lower center", ncols=3, frameon=False)
    prior.save_figure(folder, name, fig)


def plots(folder, names, overall, details, matrices, pairs, changes):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.titleweight": "bold"})
    labels = ["Accuracy", "Macro-F1", "Weighted-F1", "Balanced accuracy"]
    fields = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"]
    bars(folder, "overall_metrics", labels, {k: [overall[k][f] for f in fields] for k in KEYS},
         "Held-out Galaxy10 performance", "Score")
    for field, label in (("f1", "F1"), ("recall", "Recall")):
        bars(folder, f"per_class_{field}", names, {k: details[k][field] for k in KEYS},
             f"{label} by Galaxy10 class", label)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), layout="constrained")
    wrapped = [fill(name, width=19) for name in names]
    for ax, key in zip(axes, KEYS):
        m = prior.normalized(matrices[key])
        image = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        ax.set(title=MODELS[key], xlabel="Predicted class", ylabel="True class")
        ax.set_xticks(LABELS, wrapped, rotation=55, ha="right", rotation_mode="anchor", fontsize=8)
        ax.set_yticks(LABELS, wrapped, fontsize=8)
        for i, j in np.ndindex(m.shape):
            ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if m[i, j] > .55 else "#17202a")
    fig.colorbar(image, ax=axes, shrink=.65, format=PercentFormatter(1), label="Share of true class")
    fig.suptitle("Held-out confusion matrices · normalized by true class")
    prior.save_figure(folder, "confusion_matrices", fig)

    comparison_labels = ["Frozen − Random Forest", "Fine-Tuned − Random Forest", "Fine-Tuned − Frozen"]
    bounds = [100 * v[end] for s in pairs.values()
              for v in s["preferred_group_bootstrap"]["intervals"].values()
              for end in ("lower", "upper")]
    for field, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
        for i, s in enumerate(pairs.values()):
            v = s["preferred_group_bootstrap"]["intervals"][field]
            est, low, high = (100 * v[k] for k in ("estimate", "lower", "upper"))
            ax.hlines(i, low, high, color=COLORS[2], linewidth=2)
            ax.plot(est, i, "o", color=COLORS[2])
            ax.annotate(f"{est:+.2f} [{low:+.2f}, {high:+.2f}]", (est, i), xytext=(0, 11),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.axvline(0, linestyle="--", color="#666666")
        ax.set(yticks=range(3), yticklabels=comparison_labels, ylim=(2.5, -.65),
               xlim=(min(bounds)-4, max(bounds)+5), xlabel="Second model − first model (percentage points)",
               title=f"{label} differences · 95% paired group bootstrap intervals")
        prior.save_figure(folder, f"paired_{field}_intervals", fig)

    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    outcome_labels = ("Both correct", "First only correct", "Second only correct", "Both wrong")
    totals = np.zeros(3, dtype=int)
    for key, label, color in zip(("both_correct", "first_only_correct", "second_only_correct", "both_wrong"),
                                 outcome_labels, ("#327757", COLORS[0], COLORS[1], "#5f6368")):
        values = np.array([s["paired_outcomes"][key] for s in pairs.values()])
        bars_ = ax.barh(range(3), values, left=totals, label=label, color=color, height=.58)
        ax.bar_label(bars_, labels=[str(v) for v in values], label_type="center", color="white")
        totals += values
    ax.set(yticks=range(3), yticklabels=["Random Forest / Frozen", "Random Forest / Fine-Tuned",
                                         "Frozen / Fine-Tuned"], xlim=(0, 3545),
           xlabel="Test galaxies", title="Paired correctness by model pair")
    ax.invert_yaxis()
    fig.legend(loc="outside lower center", ncols=4, frameon=False)
    prior.save_figure(folder, "paired_correctness", fig)

    improvements = changes.nlargest(8, "errors_avoided")
    labels = [fill(f"{row.true_class} → {row.predicted_class}", width=54)
              for row in improvements.itertuples()]
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    bars_ = ax.barh(labels, improvements.errors_avoided, color=COLORS[2])
    ax.bar_label(bars_, padding=4, fontsize=9)
    ax.set(xlim=(0, improvements.errors_avoided.max()*1.14), xlabel="Fewer held-out errors",
           title="Largest reductions in confusion counts after fine-tuning")
    ax.invert_yaxis()
    prior.save_figure(folder, "top_confusion_improvements", fig)


def main():
    if sys.flags.optimize:
        raise SystemExit("Run without Python optimization.")
    try:
        paired, names, audit, provenance, rf_saved = align()
        overall, details, matrices = {}, {}, {}
        for key in KEYS:
            overall[key], details[key] = prior.metrics_for(paired.true_label, paired[key])
            matrices[key] = confusion_matrix(paired.true_label, paired[key], labels=LABELS)
        verify_reported_scores(overall, details, matrices, names, rf_saved, audit)
    except (ValueError, OSError, KeyError, TypeError, AssertionError, zipfile.BadZipFile) as error:
        prior.write_json(OUT / "audit/alignment_audit.json",
                         {"status": "FAIL", "analysis_executed": False, "error": str(error)})
        raise
    print("Three-way alignment and source metrics passed. Running three paired comparisons.", flush=True)
    pairs, disagreements = pairwise(paired, overall, details, names)
    audit["phase3_pair_a_reproduced_exactly"] = True
    with tempfile.TemporaryDirectory(prefix=".three-model-", dir=OUT) as directory:
        stage = Path(directory)
        for name in ("audit", "results", "figures"):
            (stage / name).mkdir()
        changes = tables(stage / "results", paired, names, overall, details, matrices, pairs, disagreements)
        plots(stage / "figures", names, overall, details, matrices, pairs, changes)
        prior.write_json(stage / "audit/alignment_audit.json", audit)
        provenance["computed_sha256"] = {
            str(path.relative_to(stage)): prior.digest(path) for path in sorted(stage.glob("*/*"))}
        prior.write_json(stage / "audit/provenance.json", provenance)
        for path in sorted(stage.glob("*/*")):
            path.replace(OUT / path.relative_to(stage))
    print("Saved three-model outputs.", flush=True)


if __name__ == "__main__":
    main()
