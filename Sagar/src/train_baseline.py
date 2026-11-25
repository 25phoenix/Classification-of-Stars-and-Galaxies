from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import (
    DEFAULT_SEED,
    FULL_SEARCH_GRID,
    PREPROCESSING_SPEC,
    SMOKE_SEARCH_GRID,
    HeadHyperparams,
    default_repo_paths,
)
from .data_contract import FoldSplit, SharedDataContract, load_and_verify_contract
from .embeddings import extract_or_load_embeddings, get_runtime_versions
from .evaluation import (
    build_predictions_dataframe,
    compute_classification_metrics,
    save_confusion_matrices,
    save_learning_curves,
    save_prediction_gallery,
    write_results_report,
)
from .model_head import (
    refit_head_fixed_epochs,
    select_refit_epochs,
    train_head_with_validation,
)


def _get_hardware_info() -> dict[str, Any]:
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": tf.config.threading.get_intra_op_parallelism_threads(),
        "gpus_detected": [g.name for g in gpus],
    }


def _tune_on_folds(
    folds: list[FoldSplit],
    row_to_idx: dict[int, int],
    embeddings: np.ndarray,
    row_to_label: dict[int, int],
    grid: list[HeadHyperparams],
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> tuple[HeadHyperparams, int, list[dict[str, Any]], list[dict[str, list[float]]], float]:
    """Select best hyperparameters strictly by mean validation macro-F1 across `folds`."""
    candidates_summary: list[dict[str, Any]] = []
    best_score = -1.0
    best_hp = grid[0]
    best_epochs_per_fold: list[int] = []
    best_histories: list[dict[str, list[float]]] = []
    total_train_sec = 0.0

    for hp_idx, hp in enumerate(grid):
        fold_f1s: list[float] = []
        fold_epochs: list[int] = []
        fold_histories: list[dict[str, list[float]]] = []

        for f_idx, fold in enumerate(folds):
            tr_idx = [row_to_idx[int(r)] for r in fold.train]
            va_idx = [row_to_idx[int(r)] for r in fold.validation]
            x_tr = embeddings[tr_idx]
            y_tr = np.array([row_to_label[int(r)] for r in fold.train], dtype=np.int64)
            x_va = embeddings[va_idx]
            y_va = np.array([row_to_label[int(r)] for r in fold.validation], dtype=np.int64)

            res = train_head_with_validation(
                x_train=x_tr,
                y_train=y_tr,
                x_val=x_va,
                y_val=y_va,
                hparams=hp,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                seed=seed + hp_idx * 100 + f_idx,
            )
            total_train_sec += res.train_seconds
            fold_f1s.append(res.best_val_macro_f1)
            fold_epochs.append(res.best_epoch)
            fold_histories.append(res.history)

        mean_f1 = float(np.mean(fold_f1s))
        candidates_summary.append(
            {
                "hparams": hp.to_dict(),
                "fold_val_macro_f1": fold_f1s,
                "mean_val_macro_f1": mean_f1,
                "fold_best_epochs": fold_epochs,
                "refit_epochs_if_selected": select_refit_epochs(fold_epochs),
            }
        )

        if mean_f1 > best_score:
            best_score = mean_f1
            best_hp = hp
            best_epochs_per_fold = fold_epochs
            best_histories = fold_histories

    return (
        best_hp,
        select_refit_epochs(best_epochs_per_fold),
        candidates_summary,
        best_histories,
        total_train_sec,
    )


def run_frozen_baseline(
    contract: SharedDataContract,
    output_dir: Path,
    cache_dir: Path,
    mode: str = "full",
    image_size: int = 224,
    embed_batch_size: int = 64,
    head_batch_size: int = 64,
    weights: str | None = "imagenet",
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Execute full nested CV + final tuning + held-out test evaluation on frozen MobileNetV2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)

    is_smoke = mode == "smoke"
    grid = SMOKE_SEARCH_GRID if is_smoke else FULL_SEARCH_GRID
    max_epochs = 8 if is_smoke else 40
    patience = 3 if is_smoke else 6

    # 1. Extract or load validated deterministic embeddings for ALL eligible rows
    ordered_ids, embeddings, embed_info = extract_or_load_embeddings(
        h5_path=contract.h5_path,
        dataset_sha256=contract.dataset_sha256,
        row_ids=contract.eligible_row_ids,
        cache_dir=cache_dir,
        image_size=image_size,
        batch_size=embed_batch_size,
        weights=weights,
        seed=seed,
    )
    row_to_idx = {int(r): idx for idx, r in enumerate(ordered_ids)}

    # 2. Nested Outer Cross-Validation (development partition only)
    outer_results: list[dict[str, Any]] = []
    oof_rows_list: list[np.ndarray] = []
    oof_true_list: list[np.ndarray] = []
    oof_probs_list: list[np.ndarray] = []
    head_train_seconds = 0.0
    inference_seconds = 0.0

    for o_idx, outer_fold in enumerate(contract.outer_folds):
        best_hp, refit_ep, inner_summary, _, inner_sec = _tune_on_folds(
            folds=outer_fold.inner,
            row_to_idx=row_to_idx,
            embeddings=embeddings,
            row_to_label=contract.row_to_label,
            grid=grid,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=head_batch_size,
            seed=seed + 1000 * (o_idx + 1),
        )
        head_train_seconds += inner_sec

        # Refit on outer training partition using inner-fold-selected epochs
        tr_idx = [row_to_idx[int(r)] for r in outer_fold.train]
        va_idx = [row_to_idx[int(r)] for r in outer_fold.validation]
        x_tr = embeddings[tr_idx]
        y_tr = np.array([contract.row_to_label[int(r)] for r in outer_fold.train], dtype=np.int64)
        x_va = embeddings[va_idx]
        y_va = np.array([contract.row_to_label[int(r)] for r in outer_fold.validation], dtype=np.int64)

        outer_model, _, refit_sec = refit_head_fixed_epochs(
            x_train=x_tr,
            y_train=y_tr,
            hparams=best_hp,
            epochs=refit_ep,
            batch_size=head_batch_size,
            seed=seed + 1000 * (o_idx + 1) + 99,
        )
        head_train_seconds += refit_sec

        t0 = time.perf_counter()
        val_probs = outer_model.predict(x_va, batch_size=256, verbose=0).astype(np.float32)
        inference_seconds += time.perf_counter() - t0

        val_preds = np.argmax(val_probs, axis=1)
        fold_metrics, _, _, _ = compute_classification_metrics(y_va, val_preds, contract.class_names)

        outer_results.append(
            {
                "fold": outer_fold.name,
                "selected_hparams": best_hp.to_dict(),
                "refit_epochs": refit_ep,
                "accuracy": fold_metrics["accuracy"],
                "macro_f1": fold_metrics["macro_f1"],
                "num_val_samples": int(len(y_va)),
                "inner_tuning": inner_summary,
            }
        )
        oof_rows_list.append(outer_fold.validation)
        oof_true_list.append(y_va)
        oof_probs_list.append(val_probs)

    oof_df = build_predictions_dataframe(
        row_ids=np.concatenate(oof_rows_list),
        y_true=np.concatenate(oof_true_list),
        probs=np.concatenate(oof_probs_list, axis=0),
    )
    oof_df = oof_df.sort_values("row_id").reset_index(drop=True)
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    oof_metrics, oof_per_class_df, oof_cm_raw, oof_cm_norm = compute_classification_metrics(
        oof_df["true_label"].to_numpy(),
        oof_df["predicted_label"].to_numpy(),
        contract.class_names,
    )
    oof_per_class_df.to_csv(output_dir / "oof_per_class_metrics.csv", index=False)
    save_confusion_matrices(oof_cm_raw, oof_cm_norm, contract.class_names, output_dir, prefix="oof")

    # 3. Final Model Selection on Development (saved final_tuning folds)
    final_hp, final_refit_epochs, final_tuning_summary, final_histories, ft_sec = _tune_on_folds(
        folds=contract.final_tuning_folds,
        row_to_idx=row_to_idx,
        embeddings=embeddings,
        row_to_label=contract.row_to_label,
        grid=grid,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=head_batch_size,
        seed=seed + 50000,
    )
    head_train_seconds += ft_sec
    save_learning_curves(final_histories, output_dir)

    # 4. Final Refit on ALL Development rows (zero test leakage)
    dev_idx = [row_to_idx[int(r)] for r in contract.development]
    x_dev = embeddings[dev_idx]
    y_dev = np.array([contract.row_to_label[int(r)] for r in contract.development], dtype=np.int64)

    final_model, final_refit_hist, final_refit_sec = refit_head_fixed_epochs(
        x_train=x_dev,
        y_train=y_dev,
        hparams=final_hp,
        epochs=final_refit_epochs,
        batch_size=head_batch_size,
        seed=seed + 99999,
    )
    head_train_seconds += final_refit_sec

    model_path = output_dir / "head_model.keras"
    final_model.save(model_path)

    # 5. One-Time Evaluation on Held-Out Test Partition
    test_idx = [row_to_idx[int(r)] for r in contract.test]
    x_test = embeddings[test_idx]
    y_test = np.array([contract.row_to_label[int(r)] for r in contract.test], dtype=np.int64)

    t0 = time.perf_counter()
    test_probs = final_model.predict(x_test, batch_size=256, verbose=0).astype(np.float32)
    test_infer_sec = time.perf_counter() - t0
    inference_seconds += test_infer_sec

    test_df = build_predictions_dataframe(
        row_ids=contract.test,
        y_true=y_test,
        probs=test_probs,
    )
    test_df = test_df.sort_values("row_id").reset_index(drop=True)
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)

    test_metrics, test_per_class_df, test_cm_raw, test_cm_norm = compute_classification_metrics(
        test_df["true_label"].to_numpy(),
        test_df["predicted_label"].to_numpy(),
        contract.class_names,
    )
    test_per_class_df.to_csv(output_dir / "test_per_class_metrics.csv", index=False)
    save_confusion_matrices(test_cm_raw, test_cm_norm, contract.class_names, output_dir, prefix="test")
    save_prediction_gallery(contract.h5_path, test_df, contract.class_names, output_dir)

    # 6. Save Provenance, Config, Timings, Metrics, and Honest Results Report
    timings = {
        "preprocessing_seconds": embed_info["preprocessing_seconds"],
        "embedding_extraction_seconds": embed_info["embedding_extraction_seconds"],
        "embedding_cache_hit": embed_info["cache_hit"],
        "head_training_seconds": float(head_train_seconds),
        "optional_finetuning_seconds": 0.0,
        "total_inference_seconds": float(inference_seconds),
        "test_inference_seconds": float(test_infer_sec),
    }
    model_config = {
        "experiment_type": "frozen_mobilenetv2_baseline",
        "mode": mode,
        "exploratory_reduced_protocol": is_smoke,
        "backbone": "tf.keras.applications.MobileNetV2",
        "weights": "none" if weights is None else weights,
        "weights_fingerprint_sha256": embed_info["metadata"]["weights_fingerprint_sha256"],
        "image_size": image_size,
        "preprocessing": PREPROCESSING_SPEC,
        "augmentation_active": False,
        "selected_hparams": final_hp.to_dict(),
        "refit_epochs": final_refit_epochs,
        "refit_epoch_rule": "ceil(median(final_tuning_fold_best_epochs))",
        "class_names": {str(k): v for k, v in contract.class_names.items()},
    }
    provenance = {
        "seed": seed,
        "dataset_path": str(contract.h5_path),
        "dataset_sha256": contract.dataset_sha256,
        "dataset_manifest_sha256": contract.dataset_manifest_sha256,
        "splits_manifest_sha256": contract.splits_manifest_sha256,
        "assignments_csv_sha256": contract.assignments_csv_sha256,
        "eligible_rows": int(len(contract.eligible_row_ids)),
        "development_rows": int(len(contract.development)),
        "test_rows": int(len(contract.test)),
        "embedding_cache_path": embed_info["cache_path"],
        "package_versions": get_runtime_versions(),
        "hardware": _get_hardware_info(),
    }

    (output_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    (output_dir / "outer_fold_results.json").write_text(json.dumps(outer_results, indent=2), encoding="utf-8")
    (output_dir / "final_tuning_results.json").write_text(
        json.dumps(
            {
                "selected_hparams": final_hp.to_dict(),
                "refit_epochs": final_refit_epochs,
                "candidates": final_tuning_summary,
                "final_refit_history": final_refit_hist,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "oof_metrics.json").write_text(json.dumps(oof_metrics, indent=2), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    write_results_report(
        out_path=output_dir / "results_report.md",
        mode=mode,
        oof_metrics=oof_metrics,
        test_metrics=test_metrics,
        per_class_df=test_per_class_df,
        best_hparams=final_hp.to_dict(),
        refit_epochs=final_refit_epochs,
        timings=timings,
    )

    return {
        "oof_metrics": oof_metrics,
        "test_metrics": test_metrics,
        "timings": timings,
        "model_path": str(model_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen MobileNetV2 baseline on Galaxy10 DECaLS.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--h5-path", type=Path, default=None)
    parser.add_argument("--dataset-json", type=Path, default=None)
    parser.add_argument("--splits-json", type=Path, default=None)
    parser.add_argument("--assignments-csv", type=Path, default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    defaults = default_repo_paths(args.project_root.resolve())
    h5_path = args.h5_path or defaults["h5_path"]
    dataset_json = args.dataset_json or defaults["dataset_json"]
    splits_json = args.splits_json or defaults["splits_json"]
    assignments_csv = args.assignments_csv or defaults["assignments_csv"]
    sagar_root = defaults["sagar_root"]

    out_dir = args.output_dir or (sagar_root / "results" / f"frozen_baseline_{args.mode}")
    cache_dir = args.cache_dir or (sagar_root / "cache" / "embeddings")

    contract = load_and_verify_contract(
        h5_path=h5_path,
        dataset_json_path=dataset_json,
        splits_json_path=splits_json,
        assignments_csv_path=assignments_csv,
    )
    summary = run_frozen_baseline(
        contract=contract,
        output_dir=out_dir,
        cache_dir=cache_dir,
        mode=args.mode,
        image_size=args.image_size,
        embed_batch_size=args.embed_batch_size,
        head_batch_size=args.head_batch_size,
        weights="imagenet",
        seed=args.seed,
    )
    print(f"Done ({args.mode}). Outputs saved to: {out_dir}")
    print(f"  OOF  -> acc={summary['oof_metrics']['accuracy']:.4f}, macro_f1={summary['oof_metrics']['macro_f1']:.4f}")
    print(f"  Test -> acc={summary['test_metrics']['accuracy']:.4f}, macro_f1={summary['test_metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()