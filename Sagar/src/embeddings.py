from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tensorflow as tf

from .config import PREPROCESSING_SPEC
from .data_contract import iter_h5_images_by_row_ids


def get_runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "keras": getattr(tf.keras, "__version__", "bundled"),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }


def preprocess_rgb_batch(images: np.ndarray, image_size: int = 224) -> tf.Tensor:
    """
    Canonical MobileNetV2 preprocessing applied ONCE:
    uint8 RGB [0, 255] -> float32 -> bilinear resize -> mobilenet_v2.preprocess_input [-1, 1].
    """
    arr = np.asarray(images)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected 4D RGB batch (N, H, W, 3); got shape {arr.shape}")
    if arr.dtype != np.uint8:
        min_v, max_v = float(arr.min()), float(arr.max())
        if min_v < 0.0 or max_v > 255.0:
            raise ValueError(
                f"Input pixels out of [0, 255] range ({min_v}..{max_v}); "
                "possible accidental double preprocessing."
            )
        if max_v <= 1.0 and min_v >= -1.0:
            raise ValueError(
                "Input appears already normalized to [-1, 1] or [0, 1]; "
                "pass raw RGB pixels in [0, 255] to avoid double preprocessing."
            )

    x = tf.convert_to_tensor(arr, dtype=tf.float32)
    if x.shape[1] != image_size or x.shape[2] != image_size:
        x = tf.image.resize(x, [image_size, image_size], method="bilinear")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return x


def build_frozen_mobilenetv2_backbone(
    image_size: int = 224,
    weights: str | None = "imagenet",
    seed: int = 42,
) -> tuple[tf.keras.Model, str]:
    """
    Build frozen MobileNetV2 feature extractor with GlobalAveragePooling2D output.
    Returns (backbone_model, weights_fingerprint_sha256).
    """
    tf.keras.utils.set_random_seed(seed)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=weights,
        pooling="avg",
    )
    backbone.trainable = False

    h = hashlib.sha256()
    for w in backbone.get_weights():
        h.update(np.asarray(w, dtype=np.float32).tobytes())
    weights_fingerprint = h.hexdigest()
    return backbone, weights_fingerprint


def _hash_row_ids(row_ids: np.ndarray) -> str:
    arr = np.asarray(row_ids, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def extract_or_load_embeddings(
    h5_path: Path,
    dataset_sha256: str,
    row_ids: np.ndarray,
    cache_dir: Path,
    image_size: int = 224,
    batch_size: int = 64,
    weights: str | None = "imagenet",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Compute or load deterministic frozen MobileNetV2 embeddings for `row_ids`.
    Strictly validates cache compatibility and rejects mismatched caches.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    ordered_row_ids = np.asarray(row_ids, dtype=np.int64)
    row_ids_hash = _hash_row_ids(ordered_row_ids)
    weights_name = "none" if weights is None else str(weights)

    cache_npz = cache_dir / f"mobilenetv2_{weights_name}_{image_size}.npz"
    cache_meta = cache_dir / f"mobilenetv2_{weights_name}_{image_size}.meta.json"

    backbone, weights_fingerprint = build_frozen_mobilenetv2_backbone(
        image_size=image_size,
        weights=weights,
        seed=seed,
    )

    expected_meta: dict[str, Any] = {
        "dataset_sha256": dataset_sha256,
        "row_ids_sha256": row_ids_hash,
        "num_rows": int(len(ordered_row_ids)),
        "weights": weights_name,
        "weights_fingerprint_sha256": weights_fingerprint,
        "image_size": int(image_size),
        "preprocessing": PREPROCESSING_SPEC,
        "backbone_mode": "frozen_inference_training_false_global_avg_pool",
        "augmentation": "none_cached_single_embedding_per_image",
        "package_versions": get_runtime_versions(),
    }

    if cache_npz.exists() or cache_meta.exists():
        if not (cache_npz.exists() and cache_meta.exists()):
            raise ValueError(
                f"Incomplete embedding cache in {cache_dir}; delete partial cache files and re-run."
            )
        with cache_meta.open("r", encoding="utf-8") as f:
            saved_meta = json.load(f)

        mismatches = []
        for key, exp_val in expected_meta.items():
            if saved_meta.get(key) != exp_val:
                mismatches.append(f"  - {key}: cached={saved_meta.get(key)!r} vs expected={exp_val!r}")
        if mismatches:
            raise ValueError(
                "Incompatible embedding cache rejected! Delete or move the old cache before running.\n"
                + "\n".join(mismatches)
            )

        data = np.load(cache_npz)
        cached_row_ids = data["row_ids"].astype(np.int64)
        cached_embeddings = data["embeddings"].astype(np.float32)
        if not np.array_equal(cached_row_ids, ordered_row_ids):
            raise ValueError("Cached row_ids array does not match expected row_ids order.")
        if cached_embeddings.shape[0] != len(ordered_row_ids):
            raise ValueError("Cached embeddings row count mismatch.")

        print(f"[embeddings] Validated cache hit: {cache_npz} ({len(cached_row_ids)} rows)", flush=True)
        return cached_row_ids, cached_embeddings, {
            "cache_hit": True,
            "cache_path": str(cache_npz),
            "preprocessing_seconds": 0.0,
            "embedding_extraction_seconds": 0.0,
            "metadata": saved_meta,
        }

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, image_size, image_size, 3], dtype=tf.float32)]
    )
    def _extract_batch(x: tf.Tensor) -> tf.Tensor:
        return backbone(x, training=False)

    preprocessing_seconds = 0.0
    extraction_seconds = 0.0
    all_embeddings: list[np.ndarray] = []
    seen_ids: list[np.ndarray] = []

    total_rows = len(ordered_row_ids)
    done_rows = 0
    next_log = 500
    loop_start = time.perf_counter()
    print(f"[embeddings] Extracting {total_rows} images (batch_size={batch_size})...", flush=True)

    for batch_ids, batch_imgs in iter_h5_images_by_row_ids(h5_path, ordered_row_ids, batch_size=batch_size):
        t0 = time.perf_counter()
        preprocessed = preprocess_rgb_batch(batch_imgs, image_size=image_size)
        t1 = time.perf_counter()
        feats = _extract_batch(preprocessed).numpy().astype(np.float32)
        t2 = time.perf_counter()

        preprocessing_seconds += t1 - t0
        extraction_seconds += t2 - t1
        seen_ids.append(batch_ids)
        all_embeddings.append(feats)

        done_rows += len(batch_ids)
        if done_rows >= next_log or done_rows == total_rows:
            elapsed = time.perf_counter() - loop_start
            rate = done_rows / max(elapsed, 1e-6)
            eta = (total_rows - done_rows) / max(rate, 1e-6)
            print(
                f"[embeddings] {done_rows}/{total_rows} ({100.0 * done_rows / total_rows:.1f}%) "
                f"| elapsed {elapsed:.1f}s | {rate:.1f} img/s | ETA {eta:.0f}s",
                flush=True,
            )
            next_log = done_rows + 500

    extracted_ids = np.concatenate(seen_ids, axis=0)
    embeddings = np.concatenate(all_embeddings, axis=0)
    assert np.array_equal(extracted_ids, ordered_row_ids), "Row ID order drift during batch extraction!"

    print(f"[embeddings] Saving cache to {cache_npz}...", flush=True)
    np.savez_compressed(cache_npz, row_ids=extracted_ids, embeddings=embeddings)
    with cache_meta.open("w", encoding="utf-8") as f:
        json.dump(expected_meta, f, indent=2)

    return extracted_ids, embeddings, {
        "cache_hit": False,
        "cache_path": str(cache_npz),
        "preprocessing_seconds": float(preprocessing_seconds),
        "embedding_extraction_seconds": float(extraction_seconds),
        "metadata": expected_meta,
    }