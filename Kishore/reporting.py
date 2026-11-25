"""Build plots and a Markdown report from persisted run artifacts."""
import json

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from data_io import CLASSES, ROOT
from experiment import LABELS, validate_predictions
from morphology import FEATURES, extract


def diagnostic_plot(ax, image, details):
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Rendered input")
    if details:
        ax[1].imshow(details.get("signal", details["gray"]), cmap="gray")
        ax[1].set_title("Resampled, sky subtracted")
        if "target" in details:
            ax[2].imshow(details.get("population", details["target"]), cmap="gray", vmin=0, vmax=1)
            if "population" in details:
                ax[2].contour(details["target"], levels=[.5], colors=["orange"], linewidths=.7)
        if "center" in details:
            ax[2].plot(*details["center"], "c+", markersize=7)
            for r, color in zip((details["aperture"], *details["radii"]), ("cyan", "lime", "yellow")):
                ax[2].add_patch(Circle(details["center"], r, fill=False, color=color, linewidth=.8))
    ax[2].set_title("Mask, center, apertures")
    for panel in ax:
        panel.set_axis_off()


def render(table, image_path, out):
    validate_predictions(out)
    result = json.loads((out / "metrics.json").read_text())
    dataset = json.loads((out / "dataset.json").read_text())
    if not table.row_id.is_unique:
        raise ValueError("Feature table has duplicate row IDs")
    assignments = pd.read_csv(out / "split_assignments.csv")
    plot = out / "plots"
    plot.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(plot / f"{name}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    def matrix(values, name, caption, normalized=False, labels=LABELS):
        fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
        image = ax.imshow(values, cmap="Blues", vmin=-1 if name == "correlation" else 0,
                          vmax=1 if normalized or name == "correlation" else None)
        fig.colorbar(image, ax=ax, shrink=.75)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45 if name == "correlation" else 0,
                      ha="right" if name == "correlation" else "center")
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_title(caption)
        if name != "correlation":
            ax.set_xlabel("Predicted class ID")
            ax.set_ylabel("True class ID")
        fmt = ".2f" if normalized or name == "correlation" else "d"
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, format(value, fmt), ha="center", va="center",
                        color="white" if value > .60 * image.norm.vmax else "black", fontsize=8)
        save(fig, name)

    for part, title in (("pooled_oof", "Pooled out-of-fold"), ("test", "Final held-out test")):
        for model in ("forest", "baseline"):
            for form in ("counts", "normalized"):
                matrix(np.array(result[part][model][f"confusion_{form}"]),
                       f"{part}_{model}_{form}", f"{title} · {model} · {form}", form == "normalized")

    counts = pd.crosstab(assignments.label, assignments.partition).reindex(
        index=LABELS, columns=["development", "test", "excluded"], fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    counts.plot.bar(stacked=True, ax=ax, rot=0)
    ax.set(xlabel="Class ID", ylabel="Rows", title="Original dataset and final partitions")
    save(fig, "class_distribution")

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    usable = table.loc[table.usable, list(FEATURES)]
    for name, ax in zip(FEATURES, axes.flat):
        ax.hist(usable[name], bins=35)
        ax.set(xlabel=name, ylabel="Rows")
    axes.flat[-1].set_axis_off()
    save(fig, "feature_distributions")
    matrix(usable.corr().to_numpy(), "correlation", "Pearson feature correlation", labels=FEATURES)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    importance = pd.Series(result["impurity_importance"]).sort_values()
    ax.barh(importance.index, importance)
    ax.set(xlabel="Normalized mean decrease in impurity", title="Final development-fit forest")
    save(fig, "importance")

    test = pd.read_csv(out / "test_predictions.csv").sort_values("row_id")
    examples = pd.concat([test[test.label == test.forest].drop_duplicates("label").head(3),
                          test[test.label != test.forest].drop_duplicates("label").head(3)])
    examples.to_csv(out / "examples.csv", index=False)
    with h5py.File(image_path, "r") as file:
        for row in examples.itertuples():
            image = file["images"][row.row_id]
            _, _, details = extract(image, diagnostic=True)
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), layout="constrained")
            diagnostic_plot(axes, image, details)
            fig.suptitle(f"Row {row.row_id} · True {row.label} · Predicted {row.forest}")
            save(fig, f"example_{row.row_id}")

    names = list(result["outer"][0]["forest"])

    def cell(value):
        return f"{value:.4f}"

    def md(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |",
                          "| " + " | ".join("---" for _ in headers) + " |",
                          *["| " + " | ".join(map(str, row)) + " |" for row in rows]])

    lines = ["# Galaxy10 DECaLS results", ""]
    if result["mode"] != "official":
        lines += ["> SYNTHETIC SOFTWARE SMOKE TEST. These scores do not measure galaxy classification.", ""]
    lines += [
        f"The run processed {dataset['rows']:,} rows. It retained {dataset['usable']:,} and "
        f"excluded {dataset['excluded']:,}. The development and final test partitions contain "
        f"{result['final_fit']['train_size']:,} and {result['final_fit']['validation_size']:,} usable rows.",
        "", "## Dataset, grouping, and exclusions", "",
        md(["ID", "Official class", "All", "Development", "Test", "Excluded"],
           [[i, CLASSES[i], dataset["class_counts"][i], *counts.loc[i].tolist()] for i in LABELS]),
        "", "The dataset manifest records the source, SHA-256, byte count, and HDF5 schema.",
        "", "```json", json.dumps({"duplicate_audit": dataset["audit"],
                                     "quality_flag_counts": dataset["flag_counts"],
                                     "exclusion_reasons_by_class": dataset["exclusion_reasons_by_class"]}, indent=2), "```",
        "", "## Five outer folds", "",
        "The forest parameters were selected by three-fold inner macro-F1. "
        "Standard deviations are sample standard deviations across the five outer scores.",
    ]
    for model in ("forest", "baseline"):
        rows = [[fold["fold"], fold["train_size"], fold["validation_size"],
                 *[cell(fold[model][n]) for n in names]] for fold in result["outer"]]
        summary = result["outer_mean_and_sample_sd"][model]
        rows.append(["Mean ± SD", "", "", *[
            f"{cell(summary[n]['mean'])} ± {cell(summary[n]['sample_sd'])}" for n in names]])
        lines += ["", f"### {model.capitalize()}", "",
                  md(["Fold", "Train", "Validation", *names], rows)]
    lines += ["", "### Selected parameters, timing, and class counts", "",
              md(["Fold", "Depth", "Min leaf", "Class weight", "Seconds", "Train counts by ID",
                  "Validation counts by ID"],
                 [[f["fold"], f["params"]["forest__max_depth"], f["params"]["forest__min_samples_leaf"],
                   f["params"]["forest__class_weight"], f"{f['tuning_and_fit_seconds']:.1f}",
                   f["train_counts"], f["validation_counts"]] for f in result["outer"]]),
              "", f"Final development fit used {result['final_fit']['params']} and took "
              f"{result['final_fit']['tuning_and_fit_seconds']:.1f} seconds including tuning.", ""]
    for part, title in (("pooled_oof", "Pooled outer validation"), ("test", "Held-out test")):
        lines += [f"## {title}", "",
                  md(["Model", *names], [[m, *[cell(result[part][m]["metrics"][n]) for n in names]]
                                        for m in ("forest", "baseline")])]
        for model in ("forest", "baseline"):
            classification = result[part][model]["classification_report"]
            lines += ["", f"### {model.capitalize()} per-class results", "",
                      md(["ID", "Class", "Precision", "Recall", "F1", "Support"],
                         [[i, c, cell(classification[c]["precision"]), cell(classification[c]["recall"]),
                           cell(classification[c]["f1-score"]), int(classification[c]["support"])]
                          for i, c in enumerate(CLASSES)]), ""]
            for form in ("counts", "normalized"):
                lines += [f"![{title} {model} {form}](plots/{part}_{model}_{form}.png)", ""]
    test_report = result["test"]["forest"]["classification_report"]
    low_f1 = sorted(CLASSES, key=lambda c: test_report[c]["f1-score"])[:3]
    lines += ["The three lowest held-out forest F1 scores were " + ", ".join(
              f"{c} ({cell(test_report[c]['f1-score'])})" for c in low_f1) + ".",
              "", "## Measured features and example predictions", "",
              "Mean decrease in impurity credits features for training split improvements averaged "
              "across trees. Correlated features can share credit and this measure can favor "
              "features with more split opportunities.", ""]
    for name in ("class_distribution", "feature_distributions", "correlation", "importance"):
        lines += [f"![{name}](plots/{name}.png)", ""]
    for row in examples.itertuples():
        lines += [f"Row {row.row_id} · Catalog label {CLASSES[row.label]} · Forest {CLASSES[row.forest]}",
                  "", f"![Diagnostics for row {row.row_id}](plots/example_{row.row_id}.png)", ""]
    demo_path = out / "local_unlabeled_predictions.json"
    if demo_path.is_file() and result["mode"] == "official":
        demo = json.loads(demo_path.read_text())
        lines += ["## Unlabeled local inference", "", demo["note"], "",
                  f"The demonstration processed {len(demo['predictions'])} cutouts.", "",
                  "[Local predictions](local_unlabeled_predictions.json)", ""]
    lines += ["## Methods, limitations, and reproduction", "",
              (ROOT / "METHODS.md").read_text()]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
