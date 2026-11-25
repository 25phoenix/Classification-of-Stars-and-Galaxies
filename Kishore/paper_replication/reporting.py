"""Render measured catalog reports and diagnostic plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CLASSES, FEATURES, PAPER_ACCURACY, THRESHOLDS, read_json, write_json
from evaluation import MODELS

PAPER_ROWS = {5: 104787, 10: 89829, 20: 58030}
SENTINEL_VALUES = (-9999.999999, -9999.99)

def report(table, provenance, out, smoke=False):
    lines = ["# Barchi et al. binary morphology experiment", ""]
    if smoke:
        lines += ["> Synthetic software integration run. Scores are software fixtures.", ""]
    else:
        lines += ["This reconstruction uses released CyMorph features joined to independent GZ1 labels. "
                  "The image measurements are supplied by the authors and the evaluation uses nested, " +
                  ("group-aware cross-validation and the previously inspected holdout."
                   if provenance.get("measurement_policy") else
                   "group-aware cross-validation and a new held-out partition."), "",
                  f"Author catalog rows {provenance['catalog_rows']:,}. "
                  f"Galaxy Zoo source rows {provenance['gz1_rows']:,}.",
                  "", f"The exact string-ID join matched {provenance['exact_id_join']['matched']:,} rows "
                  f"and left {provenance['exact_id_join']['unmatched']:,} unmatched. "
                  f"The joint identity audit found {provenance['grouping']['non_singleton_groups']:,} "
                  "groups with multiple rows. All splits enforce group separation. "
                  "The catalog supplies released features and K. Image-level duplicate hashing requires FITS inputs.",
                  "", "[Full provenance](provenance.json) · [Pretraining checks](preflight.json) · "
                  "[Per-row selection](sample_manifest.csv)"]
    splits = read_json(out / "splits.json")["thresholds"]
    output = out / "plots"
    output.mkdir(exist_ok=True)
    selected = table.loc[table.selection_k5.eq("included")]
    sentinel = table[list(FEATURES)].isin(SENTINEL_VALUES)
    diagnostics = {"sentinel_values": SENTINEL_VALUES, "k_cuts": {}}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    fig.suptitle("Synthetic fixture features" if smoke else "Released CyMorph features, K ≥ 5")
    for name, ax in zip(FEATURES, axes.flat):
        values = selected.loc[~sentinel.loc[selected.index, name]]
        bins = np.histogram_bin_edges(values[name], bins=40)
        for i, label in enumerate(CLASSES):
            ax.hist(values.loc[values.label == i, name], bins=bins, density=True,
                    histtype="step", label=label)
        ax.set(xlabel=name.upper(), ylabel="Density")
    axes.flat[0].legend()
    axes.flat[-1].set_axis_off()
    fig.savefig(output / "features.png", dpi=140)
    plt.close(fig)
    lines += ["", "![K ≥ 5 feature distributions](plots/features.png)", ""]
    if not smoke and not provenance.get("measurement_policy"):
        lines += ["Feature histograms omit the observed finite sentinel codes (−9999.999999 and "
                  "−9999.99) from each plotted feature only. The fitted pipelines used the "
                  "released columns as staged, including those codes on rows with Error = 0.",
                  "", "[Catalog data-quality diagnostics](catalog_diagnostics.json)", ""]
    models = dict(zip(MODELS, ("Majority", "Decision Tree", "SVM", "MLP")))
    for k in THRESHOLDS:
        sample = table[table[f"selection_k{k}"].eq("included")]
        counts = sample.label.value_counts()
        split = splits[str(k)]
        dev, test = len(split["development"]), len(split["test"])
        lines += [f"## K ≥ {k}", "", f"Usable {len(sample):,}. Ellipticals {counts[0]:,} "
                  f"({counts[0]/len(sample):.1%}), spirals {counts[1]:,} ({counts[1]/len(sample):.1%}). "
                  f"Development {dev:,}, held-out test {test:,}.", ""]
        if not smoke:
            reasons = table.groupby([f"selection_k{k}", "label"]).size()
            cut_flags = sentinel.loc[sample.index]
            any_flag = cut_flags.any(axis=1)
            diagnostics["k_cuts"][str(k)] = {
                "rows_with_sentinel": int(any_flag.sum()),
                "rows_by_feature": {name: int(cut_flags[name].sum()) for name in FEATURES},
                "rows_by_class": {label: int((any_flag & sample.label.eq(i)).sum())
                                  for i, label in enumerate(CLASSES)},
                "held_out_rows_with_sentinel": int(any_flag[sample.row_id.isin(split["test"])].sum()),
            }
            if not provenance.get("measurement_policy"):
                lines += [f"The sentinel audit found {int(any_flag.sum()):,} of {len(sample):,} rows "
                          f"({any_flag.mean():.2%}) containing at least one of these codes.", ""]
            lines += [f"Paper Table 1 lists {PAPER_ROWS[k]:,} Galaxy Zoo 1 rows at this cut. "
                      f"The reconstructed catalog sample contains {len(sample) - PAPER_ROWS[k]:+,} rows "
                      "relative to that published count.", ""]
            lines += ["| Selection status | Elliptical | Spiral | Undefined or unmatched |",
                      "| --- | ---: | ---: | ---: |"]
            for reason in sorted(set(table[f"selection_k{k}"]) - {"included"}):
                lines.append(f"| {reason} | {reasons.get((reason, 0), 0):,} | "
                             f"{reasons.get((reason, 1), 0):,} | {reasons.get((reason, -1), 0):,} |")
            lines.append("")
        lines += [
                  "| Model | Outer accuracy mean ± SD | Outer balanced accuracy mean ± SD | "
                  "Outer macro-F1 mean ± SD | Test accuracy | Test balanced accuracy | Test macro-F1 | Paper Table 3 accuracy |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        summaries = {name: read_json(out / f"k{k}" / name / "results.json") for name in MODELS}
        for name, result in summaries.items():
            what = ["accuracy", "balanced_accuracy", "macro_f1"]
            outer = [f"{result['outer_summary'][q]['mean']:.4f} ± {result['outer_summary'][q]['sample_sd']:.4f}"
                     for q in what]
            test = [f"{result['test']['metrics'][q]:.4f}" for q in what]
            target = f"{PAPER_ACCURACY[k][name]:.4f}" if name in PAPER_ACCURACY[k] and not smoke else "—"
            lines.append("| " + " | ".join([models[name], *outer, *test, target]) + " |")
        lines += ["", "SD is the sample standard deviation across the five outer folds.", ""]
        for name, result in summaries.items():
            lines += [f"### {models[name]}", "",
                      "| Fold | Train E/S | Validation E/S | Accuracy | Balanced accuracy | Macro-F1 | Seconds | Selected parameters |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for item in result["folds"]:
                m = item["metrics"]
                counts = [f"{item[key]['0']}/{item[key]['1']}" for key in ("train_counts", "validation_counts")]
                lines.append(f"| {item['fold']} | {' | '.join(counts)} | {m['accuracy']:.4f} | "
                             f"{m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} | "
                             f"{item['seconds']:.2f} | `{item['parameters']}` |")
            final = result["final_fit"]
            lines += ["", f"Final tuning and fit took {final['seconds']:.2f} seconds. "
                      f"Selected parameters `{final['parameters']}`.", "",
                      "| Partition | Class | Precision | Recall | F1 | Support |",
                      "| --- | --- | --- | --- | --- | --- |"]
            fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
            if smoke:
                fig.suptitle("Synthetic software fixture")
            for panel, part in zip(axes, ("oof", "test")):
                data = result[part]
                for cl in CLASSES:
                    v = data["class_report"][cl]
                    lines.append(f"| {part} | {cl} | {v['precision']:.4f} | {v['recall']:.4f} | "
                                 f"{v['f1-score']:.4f} | {v['support']:.0f} |")
                cm = np.array(data["confusion_matrix"])
                panel.imshow(cm, cmap="Blues")
                panel.set(title=f"{part.upper()} counts", xlabel="Predicted",
                          ylabel="Fixture label" if smoke else "Galaxy Zoo 1")
                panel.set_xticks([0, 1], CLASSES)
                panel.set_yticks([0, 1], CLASSES)
                for a in range(2):
                    for b in range(2):
                        panel.text(b, a, str(cm[a, b]), ha="center", va="center",
                                   color="white" if cm[a, b] > .55*cm.max() else "black")
            name_figure = f"k{k}_{name}_confusion.png"
            fig.savefig(output / name_figure, dpi=140)
            plt.close(fig)
            lines += ["", f"![Confusion matrices](plots/{name_figure})", ""]
    if not smoke:
        lines += ["## Interpretation", "",
                  "Table 3 uses a roughly 90/10 original development/test split, with accuracy as its "
                  "main metric. Our test uses roughly 20%, new group-aware splits, GZ1 released clean flags, "
                  "and macro-F1 model selection. Original split IDs, exact cut membership, and complete "
                  "classifier settings were unavailable. The published entries provide context for the "
                  "corresponding supervised classes and thresholds.",
                  "", "The SVM uses a 128-landmark Nyström approximation of the RBF kernel and "
                  "a primal linear SVM on the mapped features. Landmarks and scaling are fitted within "
                  "each training partition, and each solver sees every eligible training row. "
                  "A preliminary exact-kernel attempt was interrupted for runtime before its first "
                  "SVM outer fold completed. The paper does not specify its kernel.",
                  "", "The MLP keeps the paper's 44/88/22 ReLU layers. We fixed a 60-epoch Adam "
                  "budget, batch size 2,048, and training-loss stopping after exploratory runtime "
                  "monitoring of the 200-epoch configuration. The paper does not specify these optimizer settings.",
                  "", "The Galaxy Zoo release provides sexagesimal coordinates. We converted them to "
                  "degrees and independently applied the Galactic latitude cut. Neither staged "
                  "catalog includes redshift or r-band Petrosian magnitude, so we could not independently "
                  "reapply those two cuts. Table 1 sample membership cannot be recovered from this "
                  "join alone. Class counts and accuracy differences therefore reflect a different "
                  "catalog reconstruction as well as a different evaluation protocol.",
                  "", "This is a classification-stage reproduction. It uses the released values for "
                  "C, A, S, G2, and H. Image-to-feature agreement remains unverified on paired real images.",
                  "", "## Separate Galaxy10 DECaLS experiment", "",
                  "The existing ten-class, rendered RGB, seven-feature Random Forest had 53.31% ± 0.62 "
                  "percentage points outer accuracy and 0.5032 ± 0.0038 macro-F1. Its held-out accuracy "
                  "was 55.26% with 0.5178 macro-F1. It addresses a different dataset and label task.",
                  "", "[Specification](../../SPECIFICATION.md)"]
        if not provenance.get("measurement_policy"):
            write_json(out / "catalog_diagnostics.json", diagnostics)
    (out / "REPORT.md").write_text("\n".join(lines)+"\n")
