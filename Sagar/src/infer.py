from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import NUM_CLASSES
from .embeddings import build_frozen_mobilenetv2_backbone, preprocess_rgb_batch


def predict_single_image(
    image_path: Path,
    model_dir: Path,
) -> dict[str, Any]:
    """
    Run inference on a single RGB image using the saved frozen-MobileNetV2 baseline.
    Uses the exact same preprocessing and backbone configuration recorded during training.
    """
    config_path = model_dir / "model_config.json"
    head_path = model_dir / "head_model.keras"
    if not config_path.exists() or not head_path.exists():
        raise FileNotFoundError(f"Expected model_config.json and head_model.keras in {model_dir}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    image_size = int(cfg["image_size"])
    weights = None if cfg["weights"] == "none" else cfg["weights"]
    class_names = {int(k): v for k, v in cfg["class_names"].items()}

    img = Image.open(image_path).convert("RGB")
    img_arr = np.asarray(img, dtype=np.uint8)[None, ...]

    x = preprocess_rgb_batch(img_arr, image_size=image_size)
    backbone, fp = build_frozen_mobilenetv2_backbone(image_size=image_size, weights=weights)
    if weights is not None and fp != cfg["weights_fingerprint_sha256"]:
        raise ValueError("Backbone weights fingerprint does not match trained model_config.json!")

    emb = backbone(x, training=False).numpy().astype(np.float32)
    head = tf.keras.models.load_model(head_path)
    probs = head.predict(emb, verbose=0)[0].astype(float)
    pred_id = int(np.argmax(probs))

    ranked = [
        {"class_id": i, "class_name": class_names[i], "probability": float(probs[i])}
        for i in np.argsort(probs)[::-1]
    ]
    return {
        "image_path": str(image_path),
        "predicted_class_id": pred_id,
        "predicted_class_name": class_names[pred_id],
        "top_probability": float(probs[pred_id]),
        "all_probabilities": {str(i): float(probs[i]) for i in range(NUM_CLASSES)},
        "ranked_predictions": ranked,
        "disclaimer": (
            "This classifier assumes the input image belongs to one of the 10 Galaxy10 DECaLS "
            "galaxy morphology classes. Its softmax probabilities are NOT a validated star "
            "detector and NOT a calibrated confidence estimate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Galaxy10 morphology class for a new RGB image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    result = predict_single_image(args.image, args.model_dir)
    print(json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()