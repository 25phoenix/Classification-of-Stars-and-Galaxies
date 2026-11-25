from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .config import DEFAULT_SEED, NUM_CLASSES, PREPROCESSING_SPEC, default_repo_paths
from .data_contract import iter_h5_images_by_row_ids, load_and_verify_contract
from .embeddings import get_runtime_versions
from .evaluation import (
    build_predictions_dataframe,
    compute_classification_metrics,
    save_confusion_matrices,
    save_prediction_gallery,
)
from .model_head import compute_fold_class_weights


def ensure_uncompressed_image_memmap(
    h5_path: Path,
    dataset_sha256: str,
    eligible_row_ids: np.ndarray,
    cache_dir: Path,
) -> tuple[np.memmap, np.ndarray, float]:
    """
    Stage all eligible raw uint8 (256x256x3) images once in sorted order into an
    uncompressed NVMe memmap (~3.25 GiB). Eliminates HDF5 gzip chunk thrashing
    during shuffled fine-tuning while keeping RAM usage minimal.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    ordered_ids = np.sort(np.asarray(eligible_row_ids, dtype=np.int64))
    n_rows = int(len(ordered_ids))
    shape = (n_rows, 256, 256, 3)

    dat_path = cache_dir / "galaxy10_eligible_256_uint8.dat"
    meta_path = cache_dir / "galaxy10_eligible_256_uint8.meta.json"
    expected_meta = {
        "dataset_sha256": dataset_sha256,
        "num_rows": n_rows,
        "shape": list(shape),
        "dtype": "uint8",
    }

    if dat_path.exists() and meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_bytes = int(np.prod(shape))
        if saved == expected_meta and dat_path.stat().st_size == expected_bytes:
            print(f"[stage] Reusing validated NVMe image memmap: {dat_path}", flush=True)
            mm = np.memmap(dat_path, dtype=np.uint8, mode="r", shape=shape)
            return mm, ordered_ids, 0.0

    tmp_path = cache_dir / "galaxy10_eligible_256_uint8.dat.tmp"
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[stage] Staging {n_rows} raw images once into {dat_path} (~3.25 GiB)...", flush=True)
    t0 = time.perf_counter()
    mm_w = np.memmap(tmp_path, dtype=np.uint8, mode="w+", shape=shape)
    offset = 0
    next_log = 2000
    for _, batch_imgs in iter_h5_images_by_row_ids(h5_path, ordered_ids, batch_size=256):
        n = len(batch_imgs)
        mm_w[offset : offset + n] = batch_imgs
        offset += n
        if offset >= next_log or offset == n_rows:
            elapsed = time.perf_counter() - t0
            print(f"[stage] {offset}/{n_rows} ({100.0 * offset / n_rows:.1f}%) | {elapsed:.1f}s", flush=True)
            next_log = offset + 2000
    mm_w.flush()
    del mm_w

    assert offset == n_rows, f"Expected {n_rows} staged rows, got {offset}"
    tmp_path.replace(dat_path)
    meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    stage_seconds = time.perf_counter() - t0
    print(f"[stage] Complete in {stage_seconds:.1f}s", flush=True)

    mm = np.memmap(dat_path, dtype=np.uint8, mode="r", shape=shape)
    return mm, ordered_ids, stage_seconds


def _make_memmap_dataset(
    mm: np.memmap,
    ordered_eligible_ids: np.ndarray,
    row_ids: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    row_ids = np.asarray(row_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    mm_indices = np.searchsorted(ordered_eligible_ids, row_ids)
    assert np.array_equal(ordered_eligible_ids[mm_indices], row_ids), "Row ID lookup mismatch in memmap!"

    epoch_counter = [0]

    def gen():
        ep = epoch_counter[0]
        epoch_counter[0] += 1
        order = np.arange(len(mm_indices))
        if shuffle:
            rng = np.random.default_rng(seed + ep)
            rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch_imgs = np.array(mm[mm_indices[idx]], copy=True)
            batch_y = labels[idx]
            yield batch_imgs, batch_y

    def _preprocess(imgs: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = tf.cast(imgs, tf.float32)
        if image_size != 256:
            x = tf.image.resize(x, [image_size, image_size], method="bilinear")
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        return x, y

    return (
        tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.uint8),
                tf.TensorSpec(shape=(None,), dtype=tf.int64),
            ),
        )
        .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(2)
    )


def build_two_stage_model(
    image_size: int = 224,
    dropout: float = 0.3,
    l2_reg: float = 1e-4,
    seed: int = DEFAULT_SEED,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Build MobileNetV2 with training-only astronomical augmentation and frozen
    backbone initially. Calling backbone(x, training=False) guarantees all
    BatchNormalization layers stay in inference mode across both stages.
    """
    tf.keras.utils.set_random_seed(seed)

    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", dtype="float32"),
            tf.keras.layers.RandomRotation(0.25, fill_mode="reflect", dtype="float32"),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", dtype="float32"),
        ],
        name="train_only_augmentation",
    )

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), dtype="float32", name="preprocessed_rgb")
    x = aug(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        dtype="float32",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_finetuned_galaxy10")
    return model, backbone


def configure_finetune_unfreeze(backbone: tf.keras.Model, unfreeze_top_layers: int) -> int:
    """Unfreeze top non-BatchNorm layers while keeping ALL BatchNorm layers frozen."""
    backbone.trainable = True
    for layer in backbone.layers[:-unfreeze_top_layers]:
        layer.trainable = False
    unfrozen_count = 0
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        elif layer.trainable:
            unfrozen_count += 1
    return unfrozen_count


def save_finetune_curves(
    hist_warmup: dict[str, list[float]],
    hist_ft: dict[str, list[float]],
    out_path: Path,
) -> None:
    loss = hist_warmup.get("loss", []) + hist_ft.get("loss", [])
    val_loss = hist_warmup.get("val_loss", []) + hist_ft.get("val_loss", [])
    acc = hist_warmup.get("accuracy", []) + hist_ft.get("accuracy", [])
    val_acc = hist_warmup.get("val_accuracy", []) + hist_ft.get("val_accuracy", [])
    split_ep = len(hist_warmup.get("loss", []))
    epochs = np.arange(1, len(loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].axvline(split_ep + 0.5, color="gray", linestyle="--", label="Unfreeze Top Layers")
    axes[0].set_title("Two-Stage Fine-Tuning Loss (Dev Fold 0)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, acc, label="Train Acc")
    axes[1].plot(epochs, val_acc, label="Val Acc")
    axes[1].axvline(split_ep + 0.5, color="gray", linestyle="--", label="Unfreeze Top Layers")
    axes[1].set_title("Two-Stage Fine-Tuning Accuracy (Dev Fold 0)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional separate two-stage MobileNetV2 fine-tuning (GPU).")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--unfreeze-top-layers", type=int, default=40)
    parser.add_argument("--warmup-lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use-class-weight", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    used_mixed_precision = False
    if gpus and not args.no_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        used_mixed_precision = True
        print(f"[gpu] Enabled mixed_float16 on {len(gpus)} GPU(s)", flush=True)
    else:
        print(f"[gpu] Running in float32 (GPUs detected: {len(gpus)})", flush=True)

    defaults = default_repo_paths(args.project_root.resolve())
    out_dir = defaults["sagar_root"] / "results" / "finetuned_optional"
    cache_dir = defaults["sagar_root"] / "cache" / "staged_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    contract = load_and_verify_contract(
        h5_path=defaults["h5_path"],
        dataset_json_path=defaults["dataset_json"],
        splits_json_path=defaults["splits_json"],
        assignments_csv_path=defaults["assignments_csv"],
    )

    mm, ordered_eligible_ids, stage_seconds = ensure_uncompressed_image_memmap(
        h5_path=contract.h5_path,
        dataset_sha256=contract.dataset_sha256,
        eligible_row_ids=contract.eligible_row_ids,
        cache_dir=cache_dir,
    )

    # 1. Development-only validation on final_tuning fold 0 to pick fine-tuning epochs
    ft_fold = contract.final_tuning_folds[0]
    tr_y = np.array([contract.row_to_label[int(r)] for r in ft_fold.train], dtype=np.int64)
    va_y = np.array([contract.row_to_label[int(r)] for r in ft_fold.validation], dtype=np.int64)

    tr_ds = _make_memmap_dataset(mm, ordered_eligible_ids, ft_fold.train, tr_y, args.image_size, args.batch_size, True, args.seed)
    va_ds = _make_memmap_dataset(mm, ordered_eligible_ids, ft_fold.validation, va_y, args.image_size, args.batch_size, False, args.seed)
    cw_tr = compute_fold_class_weights(tr_y) if args.use_class_weight else None

    model, backbone = build_two_stage_model(args.image_size, seed=args.seed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.warmup_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    t0 = time.perf_counter()
    print(f"\n=== Validation Stage 1/2: Head Warmup ({args.warmup_epochs} epochs, frozen backbone) ===", flush=True)
    hist_w = model.fit(tr_ds, validation_data=va_ds, epochs=args.warmup_epochs, class_weight=cw_tr, verbose=1)

    unfrozen_layers = configure_finetune_unfreeze(backbone, args.unfreeze_top_layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(
        f"\n=== Validation Stage 2/2: Fine-Tuning ({args.epochs} max epochs, "
        f"{unfrozen_layers} non-BN layers unfrozen) ===",
        flush=True,
    )
    hist_ft = model.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=args.epochs,
        class_weight=cw_tr,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
            ),
        ],
        verbose=1,
    )
    val_search_seconds = time.perf_counter() - t0
    best_ft_epochs = int(np.argmin(hist_ft.history["val_loss"])) + 1
    best_val_acc = float(hist_ft.history["val_accuracy"][best_ft_epochs - 1])
    best_val_loss = float(hist_ft.history["val_loss"][best_ft_epochs - 1])
    print(
        f"\n[select] Best dev-fold epoch={best_ft_epochs} "
        f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})",
        flush=True,
    )
    save_finetune_curves(hist_w.history, hist_ft.history, out_dir / "learning_curves.png")

    # Release GPU VRAM before building the final refit model
    del model, backbone, tr_ds, va_ds
    tf.keras.backend.clear_session()
    gc.collect()

    # 2. Refit on full development partition (14,188 rows) using selected epoch schedule
    dev_y = np.array([contract.row_to_label[int(r)] for r in contract.development], dtype=np.int64)
    dev_ds = _make_memmap_dataset(
        mm, ordered_eligible_ids, contract.development, dev_y, args.image_size, args.batch_size, True, args.seed + 100
    )
    cw_dev = compute_fold_class_weights(dev_y) if args.use_class_weight else None

    final_model, final_backbone = build_two_stage_model(args.image_size, seed=args.seed + 1)
    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.warmup_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    t1 = time.perf_counter()
    print(f"\n=== Final Refit Stage 1/2: Full Dev Head Warmup ({args.warmup_epochs} epochs) ===", flush=True)
    final_model.fit(dev_ds, epochs=args.warmup_epochs, class_weight=cw_dev, verbose=1)

    configure_finetune_unfreeze(final_backbone, args.unfreeze_top_layers)
    steps_per_epoch = math.ceil(len(contract.development) / args.batch_size)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=max(1, best_ft_epochs * steps_per_epoch),
        alpha=0.1,
    )
    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\n=== Final Refit Stage 2/2: Full Dev Fine-Tuning ({best_ft_epochs} epochs, cosine decay) ===", flush=True)
    final_model.fit(dev_ds, epochs=best_ft_epochs, class_weight=cw_dev, verbose=1)
    refit_seconds = time.perf_counter() - t1

    final_model.save(out_dir / "finetuned_model.keras")

    # 3. Held-out test evaluation (3,545 locked rows)
    test_y = np.array([contract.row_to_label[int(r)] for r in contract.test], dtype=np.int64)
    test_ds = _make_memmap_dataset(
        mm, ordered_eligible_ids, contract.test, test_y, args.image_size, args.batch_size, False, args.seed
    )
    t2 = time.perf_counter()
    test_probs = final_model.predict(test_ds, verbose=1).astype(np.float32)
    infer_seconds = time.perf_counter() - t2

    test_df = build_predictions_dataframe(contract.test, test_y, test_probs).sort_values("row_id")
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    test_metrics, per_class_df, cm_raw, cm_norm = compute_classification_metrics(
        test_df["true_label"].to_numpy(), test_df["predicted_label"].to_numpy(), contract.class_names
    )
    per_class_df.to_csv(out_dir / "test_per_class_metrics.csv", index=False)
    save_confusion_matrices(cm_raw, cm_norm, contract.class_names, out_dir, prefix="test")
    save_prediction_gallery(contract.h5_path, test_df, contract.class_names, out_dir)

    meta = {
        "experiment_type": "optional_two_stage_end_to_end_finetuning",
        "preprocessing": PREPROCESSING_SPEC,
        "mixed_precision_float16": used_mixed_precision,
        "augmentation_active_training_only": True,
        "batchnorm_frozen": True,
        "batch_size": args.batch_size,
        "warmup_epochs": args.warmup_epochs,
        "warmup_lr": args.warmup_lr,
        "unfreeze_top_layers": args.unfreeze_top_layers,
        "unfrozen_non_bn_layers": unfrozen_layers,
        "finetune_lr": args.lr,
        "use_class_weight": args.use_class_weight,
        "selected_finetune_epochs": best_ft_epochs,
        "dev_fold0_best_val_loss": best_val_loss,
        "dev_fold0_best_val_accuracy": best_val_acc,
        "image_staging_seconds": float(stage_seconds),
        "val_search_seconds": float(val_search_seconds),
        "final_refit_seconds": float(refit_seconds),
        "total_finetuning_seconds": float(val_search_seconds + refit_seconds),
        "test_inference_seconds": float(infer_seconds),
        "package_versions": get_runtime_versions(),
    }
    (out_dir / "finetune_config_and_timings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    top3 = per_class_df.sort_values("f1", ascending=False).head(3)
    bot3 = per_class_df.sort_values("f1", ascending=True).head(3)
    lines = [
        "# Optional Two-Stage Fine-Tuned MobileNetV2 Report (Galaxy10 DECaLS)\n",
        f"- **Held-Out Test Accuracy:** `{test_metrics['accuracy']:.4f}` (vs. Frozen Baseline `0.5633`)",
        f"- **Held-Out Test Macro-F1:** `{test_metrics['macro_f1']:.4f}` (vs. Frozen Baseline `0.5224`)",
        f"- **Protocol:** {args.warmup_epochs} warmup epochs (`lr={args.warmup_lr}`) + "
        f"{best_ft_epochs} fine-tuning epochs (`lr={args.lr}`, top {args.unfreeze_top_layers} layers, BN frozen, training-only flip/rot/shift augmentation).",
        f"- **Total Fine-Tuning Time:** `{val_search_seconds + refit_seconds:.1f}s` | **Test Inference:** `{infer_seconds:.2f}s`\n",
        "## Top 3 Test Classes by F1",
    ]
    for _, r in top3.iterrows():
        lines.append(f"- Class {int(r['class_id'])} ({r['class_name']}): F1=`{r['f1']:.4f}`, P=`{r['precision']:.4f}`, R=`{r['recall']:.4f}`, n=`{int(r['support'])}`")
    lines.append("\n## Bottom 3 Test Classes by F1")
    for _, r in bot3.iterrows():
        lines.append(f"- Class {int(r['class_id'])} ({r['class_name']}): F1=`{r['f1']:.4f}`, P=`{r['precision']:.4f}`, R=`{r['recall']:.4f}`, n=`{int(r['support'])}`")
    (out_dir / "results_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n============================================================", flush=True)
    print(f"Optional fine-tuning complete -> {out_dir}", flush=True)
    print(f"  Test Acc      : {test_metrics['accuracy']:.4f}  (Frozen baseline: 0.5633)", flush=True)
    print(f"  Test Macro-F1 : {test_metrics['macro_f1']:.4f}  (Frozen baseline: 0.5224)", flush=True)
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()