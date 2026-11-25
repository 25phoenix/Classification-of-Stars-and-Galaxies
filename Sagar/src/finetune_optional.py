from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import DEFAULT_SEED, NUM_CLASSES, PREPROCESSING_SPEC, default_repo_paths
from .data_contract import iter_h5_images_by_row_ids, load_and_verify_contract
from .embeddings import get_runtime_versions, preprocess_rgb_batch
from .evaluation import (
    build_predictions_dataframe,
    compute_classification_metrics,
    save_confusion_matrices,
    save_prediction_gallery,
)
from .model_head import compute_fold_class_weights


def _make_h5_dataset(
    h5_path: Path,
    row_ids: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    row_ids = np.asarray(row_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    def gen():
        order = np.arange(len(row_ids))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        ordered_ids = row_ids[order]
        ordered_labels = labels[order]
        offset = 0
        for _, batch_imgs in iter_h5_images_by_row_ids(h5_path, ordered_ids, batch_size=batch_size):
            n = len(batch_imgs)
            x = preprocess_rgb_batch(batch_imgs, image_size=image_size)
            y = ordered_labels[offset : offset + n]
            offset += n
            yield x, y

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, image_size, image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
        ),
    ).prefetch(tf.data.AUTOTUNE)


def build_finetune_model(
    image_size: int = 224,
    unfreeze_top_layers: int = 30,
    learning_rate: float = 1e-5,
    seed: int = DEFAULT_SEED,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)

    # Training-only astronomical augmentation (inactive at inference)
    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.25, fill_mode="reflect"),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
        ],
        name="train_only_augmentation",
    )

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = True
    for layer in backbone.layers[:-unfreeze_top_layers]:
        layer.trainable = False
    # Keep ALL BatchNormalization layers frozen for stable fine-tuning
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="preprocessed_rgb")
    x = aug(inputs)
    x = backbone(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_finetuned_galaxy10")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional separate fine-tuning experiment (GPU recommended).")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--unfreeze-top-layers", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    defaults = default_repo_paths(args.project_root.resolve())
    out_dir = defaults["sagar_root"] / "results" / "finetuned_optional"
    out_dir.mkdir(parents=True, exist_ok=True)

    contract = load_and_verify_contract(
        h5_path=defaults["h5_path"],
        dataset_json_path=defaults["dataset_json"],
        splits_json_path=defaults["splits_json"],
        assignments_csv_path=defaults["assignments_csv"],
    )

    # Use first final_tuning fold to choose best epoch via development-only validation
    ft_fold = contract.final_tuning_folds[0]
    tr_y = np.array([contract.row_to_label[int(r)] for r in ft_fold.train], dtype=np.int64)
    va_y = np.array([contract.row_to_label[int(r)] for r in ft_fold.validation], dtype=np.int64)

    tr_ds = _make_h5_dataset(contract.h5_path, ft_fold.train, tr_y, args.image_size, args.batch_size, True, args.seed)
    va_ds = _make_h5_dataset(contract.h5_path, ft_fold.validation, va_y, args.image_size, args.batch_size, False, args.seed)

    model = build_finetune_model(args.image_size, args.unfreeze_top_layers, args.lr, args.seed)
    cw = compute_fold_class_weights(tr_y)

    t0 = time.perf_counter()
    hist = model.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=args.epochs,
        class_weight=cw,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=1,
    )
    ft_seconds = time.perf_counter() - t0
    best_epochs = int(np.argmin(hist.history["val_loss"])) + 1

    # Refit on full development partition for best_epochs
    dev_y = np.array([contract.row_to_label[int(r)] for r in contract.development], dtype=np.int64)
    dev_ds = _make_h5_dataset(contract.h5_path, contract.development, dev_y, args.image_size, args.batch_size, True, args.seed + 1)
    final_model = build_finetune_model(args.image_size, args.unfreeze_top_layers, args.lr, args.seed + 1)
    t1 = time.perf_counter()
    final_model.fit(dev_ds, epochs=best_epochs, class_weight=compute_fold_class_weights(dev_y), verbose=1)
    ft_seconds += time.perf_counter() - t1

    final_model.save(out_dir / "finetuned_model.keras")

    # Held-out test evaluation
    test_y = np.array([contract.row_to_label[int(r)] for r in contract.test], dtype=np.int64)
    test_ds = _make_h5_dataset(contract.h5_path, contract.test, test_y, args.image_size, args.batch_size, False, args.seed)
    t2 = time.perf_counter()
    test_probs = final_model.predict(test_ds, verbose=0).astype(np.float32)
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
        "experiment_type": "optional_end_to_end_finetuning",
        "preprocessing": PREPROCESSING_SPEC,
        "augmentation_active_training_only": True,
        "batchnorm_frozen": True,
        "unfreeze_top_layers": args.unfreeze_top_layers,
        "learning_rate": args.lr,
        "selected_refit_epochs": best_epochs,
        "finetuning_seconds": float(ft_seconds),
        "test_inference_seconds": float(infer_seconds),
        "package_versions": get_runtime_versions(),
    }
    (out_dir / "finetune_config_and_timings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"Optional fine-tuning complete -> {out_dir}")


if __name__ == "__main__":
    main()
    