from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import ALL_CLASS_IDS, NUM_CLASSES
from .data_contract import read_h5_images


def validate_predictions_frame(df: pd.DataFrame, expected_row_ids: np.ndarray) -> None:
    """Strict checks on prediction schema, row IDs, shapes, and probability validity."""
    prob_cols = [f"prob_{i}" for i in range(NUM_CLASSES)]
    required = ["row_id", "true_label", "predicted_label"] + prob_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction DataFrame missing columns: {missing}")

    actual_ids = df["row_id"].astype(np.int64).to_numpy()
    if set(actual_ids.tolist()) != set(int(r) for r in expected_row_ids) or len(actual_ids) != len(expected_row_ids):
        raise ValueError("Prediction row_ids do not exactly match expected partition row_ids.")

    probs = df[prob_cols].to_numpy(dtype=np.float64)
    if np.isnan(probs).any() or np.isinf(probs).any():
        raise ValueError("Prediction probabilities contain NaN or Inf.")
    if (probs < -1e-6).any() or (probs > 1.0 + 1e-6).any():
        raise ValueError("Prediction probabilities fall outside [0, 1].")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(f"Probabilities do not sum to 1 (min={row_sums.min()}, max={row_sums.max()}).")

    argmax_labels = np.argmax(probs, axis=1)
    if not np.array_equal(argmax_labels, df["predicted_label"].to_numpy(dtype=np.int64)):
        raise ValueError("predicted_label does not equal argmax(prob_0..prob_9).")


def build_predictions_dataframe(
    row_ids: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
) -> pd.DataFrame:
    row_ids = np.asarray(row_ids, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    y_pred = np.argmax(probs, axis=1).astype(np.int64)

    data: dict[str, Any] = {
        "row_id": row_ids,
        "true_label": y_true,
        "predicted_label": y_pred,
    }
    for i in range(NUM_CLASSES):
        data[f"prob_{i}"] = probs[:, i]
    df = pd.DataFrame(data)
    validate_predictions_frame(df, expected_row_ids=row_ids)
    return df


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: dict[int, str],
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute 10-class accuracy, macro-F1, per-class metrics, and raw/normalized confusion matrices."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=ALL_CLASS_IDS, average="macro", zero_division=0)
    )
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=ALL_CLASS_IDS, average=None, zero_division=0
    )

    per_class_df = pd.DataFrame(
        {
            "class_id": ALL_CLASS_IDS,
            "class_name": [class_names[i] for i in ALL_CLASS_IDS],
            "precision": prec.astype(float),
            "recall": rec.astype(float),
            "f1": f1.astype(float),
            "support": supp.astype(int),
        }
    )

    cm_raw = confusion_matrix(y_true, y_pred, labels=ALL_CLASS_IDS)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_raw.astype(np.float64),
        np.where(row_sums == 0, 1, row_sums),
    )

    summary = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "num_samples": int(len(y_true)),
        "per_class": per_class_df.to_dict(orient="records"),
    }
    return summary, per_class_df, cm_raw, cm_norm


def save_confusion_matrices(
    cm_raw: np.ndarray,
    cm_norm: np.ndarray,
    class_names: dict[int, str],
    out_dir: Path,
    prefix: str = "test",
) -> None:
    labels = [f"{i}: {class_names[i]}" for i in ALL_CLASS_IDS]
    pd.DataFrame(cm_raw, index=labels, columns=labels).to_csv(
        out_dir / f"{prefix}_confusion_matrix_raw.csv"
    )
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(
        out_dir / f"{prefix}_confusion_matrix_normalized.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    short_labels = [str(i) for i in ALL_CLASS_IDS]
    for ax, mat, title, fmt in (
        (axes[0], cm_raw, f"{prefix.title()} Raw Confusion Matrix", "d"),
        (axes[1], cm_norm, f"{prefix.title()} Row-Normalized Confusion Matrix", ".2f"),
    ):
        im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted Class ID")
        ax.set_ylabel("True Class ID")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(short_labels)
        ax.set_yticklabels(short_labels)
        thresh = mat.max() / 2.0 if mat.size else 0.0
        for r in range(NUM_CLASSES):
            for c in range(NUM_CLASSES):
                val = mat[r, c]
                ax.text(
                    c,
                    r,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if val > thresh else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion_matrices.png", dpi=150)
    plt.close(fig)


def save_learning_curves(fold_histories: list[dict[str, list[float]]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for idx, hist in enumerate(fold_histories):
        epochs = range(1, len(hist.get("loss", [])) + 1)
        axes[0].plot(epochs, hist.get("loss", []), alpha=0.6, label=f"fold{idx} train")
        if "val_loss" in hist:
            axes[0].plot(epochs, hist["val_loss"], linestyle="--", alpha=0.8, label=f"fold{idx} val")

        axes[1].plot(epochs, hist.get("accuracy", []), alpha=0.6)
        if "val_accuracy" in hist:
            axes[1].plot(epochs, hist["val_accuracy"], linestyle="--", alpha=0.8)

        if "val_macro_f1" in hist:
            axes[2].plot(epochs, hist["val_macro_f1"], marker="o", markersize=3, label=f"fold{idx}")

    axes[0].set_title("Loss (Final-Tuning Folds)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Accuracy (Final-Tuning Folds)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[2].set_title("Validation Macro-F1 (Model Selection Metric)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=150)
    plt.close(fig)


def save_prediction_gallery(
    h5_path: Path,
    preds_df: pd.DataFrame,
    class_names: dict[int, str],
    out_dir: Path,
    n_correct: int = 8,
    n_incorrect: int = 8,
) -> None:
    prob_cols = [f"prob_{i}" for i in range(NUM_CLASSES)]
    probs = preds_df[prob_cols].to_numpy(dtype=np.float32)
    max_probs = probs.max(axis=1)

    df = preds_df.copy()
    df["max_prob"] = max_probs
    correct_df = df[df["true_label"] == df["predicted_label"]].sort_values("max_prob", ascending=False).head(n_correct)
    wrong_df = df[df["true_label"] != df["predicted_label"]].sort_values("max_prob", ascending=False).head(n_incorrect)

    selected = pd.concat([correct_df, wrong_df], ignore_index=True)
    if selected.empty:
        return

    imgs = read_h5_images(h5_path, selected["row_id"].to_numpy(dtype=np.int64))
    n_total = len(selected)
    cols = 4
    rows = int(np.ceil(n_total / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax in axes_flat:
        ax.axis("off")

    for idx, (_, row) in enumerate(selected.iterrows()):
        ax = axes_flat[idx]
        ax.imshow(imgs[idx])
        is_ok = int(row["true_label"]) == int(row["predicted_label"])
        status = "CORRECT" if is_ok else "WRONG"
        color = "green" if is_ok else "red"
        t_id, p_id = int(row["true_label"]), int(row["predicted_label"])
        ax.set_title(
            f"{status} | row {int(row['row_id'])}\n"
            f"T:{t_id} ({class_names[t_id][:16]})\n"
            f"P:{p_id} ({class_names[p_id][:16]}) [{row['max_prob']:.2f}]",
            fontsize=8,
            color=color,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "prediction_gallery.png", dpi=150)
    plt.close(fig)


def write_results_report(
    out_path: Path,
    mode: str,
    oof_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    per_class_df: pd.DataFrame,
    best_hparams: dict[str, Any],
    refit_epochs: int,
    timings: dict[str, Any],
) -> None:
    sorted_classes = per_class_df.sort_values("f1", ascending=False)
    strongest = sorted_classes.head(3)
    weakest = sorted_classes.tail(3)
    min_supp_row = per_class_df.loc[per_class_df["support"].idxmin()]
    max_supp_row = per_class_df.loc[per_class_df["support"].idxmax()]

    mode_banner = (
        "> **EXPLORATORY SMOKE RUN:** This output was produced in `--mode smoke` with a "
        "reduced hyperparameter grid and epoch budget. It is NOT equivalent to the full comparison.\n"
        if mode != "full"
        else ""
    )

    lines = [
        "# MobileNetV2 Transfer Learning Results Report (Galaxy10 DECaLS)",
        "",
        mode_banner,
        "## Experimental Scope",
        "- **Task:** Ten-class galaxy morphology classification on `Galaxy10_DECals.h5` using the exact shared eligible population and nested splits saved by Kishore.",
        "- **Comparative framing:** Inspired by Barchi et al. (2020), not an exact reproduction (different dataset, backbone, and feature set).",
        "- **No star classification claims:** Galaxy10 DECaLS has no verified stellar subtype labels. Softmax outputs assume a galaxy image from these 10 classes and are not a star detector or calibrated confidence estimate.",
        "",
        "## Headline Measured Performance",
        f"- **Pooled Outer Out-of-Fold (Development):** Accuracy = `{oof_metrics['accuracy']:.4f}`, Macro-F1 = `{oof_metrics['macro_f1']:.4f}` (n = {oof_metrics['num_samples']})",
        f"- **Held-Out Test Partition:** Accuracy = `{test_metrics['accuracy']:.4f}`, Macro-F1 = `{test_metrics['macro_f1']:.4f}` (n = {test_metrics['num_samples']})",
        f"- **Selected Final Head Hyperparameters (development-only):** `{json.dumps(best_hparams)}`",
        f"- **Refit Epochs (median of final-tuning early-stopped epochs):** `{refit_epochs}`",
        "",
        "## Measured Strengths (Top 3 Test Classes by F1)",
    ]
    for _, r in strongest.iterrows():
        lines.append(
            f"- Class {int(r['class_id'])} ({r['class_name']}): F1 = `{r['f1']:.4f}`, "
            f"Precision = `{r['precision']:.4f}`, Recall = `{r['recall']:.4f}`, Support = `{int(r['support'])}`"
        )

    lines += ["", "## Measured Failure Modes (Lowest 3 Test Classes by F1)"]
    for _, r in weakest.iterrows():
        lines.append(
            f"- Class {int(r['class_id'])} ({r['class_name']}): F1 = `{r['f1']:.4f}`, "
            f"Precision = `{r['precision']:.4f}`, Recall = `{r['recall']:.4f}`, Support = `{int(r['support'])}`"
        )

    lines += [
        "",
        "## Class Imbalance & Limitations",
        f"- **Test support range:** smallest class is Class {int(min_supp_row['class_id'])} ({min_supp_row['class_name']}, n={int(min_supp_row['support'])}); "
        f"largest class is Class {int(max_supp_row['class_id'])} ({max_supp_row['class_name']}, n={int(max_supp_row['support'])}).",
        "- **Frozen features limitation:** Because one deterministic ImageNet-pretrained MobileNetV2 embedding is cached per image, no image augmentation is active during head training, and convolutional filters are not adapted to astronomical bandpass structure.",
        "- **Visual ambiguity:** Morphologically adjacent categories (e.g., tight vs. loose spirals, disturbed vs. merging, in-between smooth vs. round smooth) account for much of the confusion visible in `test_confusion_matrices.png`.",
        "",
        "## Measured Timings (seconds)",
        "```json",
        json.dumps(timings, indent=2),
        "```",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    