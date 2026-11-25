"""Shared extraction and quality-aware model inference."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import morphology
from data_io import CLASSES, digest
from morphology import FEATURES, Settings, extract

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def read_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode == "P":
            if "transparency" in image.info:
                raise ValueError("Transparent palette images are unsupported")
            image = image.convert("RGB")
        return np.asarray(image)


def load_model(path):
    bundle = joblib.load(path)
    if tuple(bundle["feature_order"]) != FEATURES:
        raise ValueError("Incompatible feature order")
    if bundle["feature_code_sha256"] != digest(Path(morphology.__file__)):
        raise ValueError("Feature implementation differs from model's training version")
    if bundle["class_mapping"] != dict(enumerate(CLASSES)):
        raise ValueError("Incompatible class mapping")
    return bundle


def predict_image(image, bundle, diagnostic=False):
    features, flags, details = extract(image, Settings(**bundle["settings"]), diagnostic)
    result = {"class_id": None, "class_name": None, "probabilities": None, "flags": flags,
              "features": {k: float(v) if np.isfinite(v) else None for k, v in features.items()}}
    if np.isfinite(list(features.values())).all():
        frame = pd.DataFrame([features], columns=FEATURES)
        pipeline = bundle["pipeline"]
        label = int(pipeline.predict(frame)[0])
        result.update(class_id=label, class_name=bundle["class_mapping"][label],
                      probabilities={str(k): float(p) for k, p in
                                     zip(pipeline.classes_, pipeline.predict_proba(frame)[0])})
    return result, details


def predict_path(path, bundle, limit=None):
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES
                   ) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No supported images in {path}")
    for file in files[:limit]:
        try:
            image = read_image(file)
        except (OSError, ValueError) as error:
            yield {"image": str(file), "class_id": None, "class_name": None,
                   "probabilities": None, "flags": ["unreadable"], "error": str(error)}
            continue
        result, _ = predict_image(image, bundle)
        yield {"image": str(file), **result}
