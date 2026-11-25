"""Audit the archived model and optionally reproduce predictions with TensorFlow."""
import argparse
import hashlib
import importlib.util
import io
import json
import platform
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
VERSION = BASE / "finetuned_v1"
RESULTS = BASE / "verification"
ORIGINAL = ROOT / "Sagar/results/finetuned_mobilenetv2"
MODEL_NAME = "finetuned_model.keras"
ARCHIVE = ROOT / "Sagar/Sagar_FineTuned_Model_And_Code_Archive.zip"
ARCHIVED_MODEL = VERSION / "model" / MODEL_NAME
DATA = ROOT / "Kishore/results/final"
H5 = ROOT / "Kishore/data/raw/Galaxy10_DECals.h5"
PROBS = [f"prob_{i}" for i in range(10)]

sys.dont_write_bytecode = True
sys.path.insert(0, str(VERSION / "code"))
from Sagar.src.data_contract import load_and_verify_contract


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
        return checksum.hexdigest()


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def structure():
    model_hash = digest(ARCHIVED_MODEL)
    check(model_hash == digest(ORIGINAL / MODEL_NAME), "Installed model differs from versioned model")
    manifest = {"archive": str(ARCHIVE.relative_to(ROOT)), "archive_sha256": digest(ARCHIVE),
                "installed_model": str((ORIGINAL / MODEL_NAME).relative_to(ROOT)), "model_sha256": model_hash,
                "members": {}, "source_matches_baseline": {}}
    with zipfile.ZipFile(ARCHIVE) as archive:
        check(archive.testzip() is None, "Outer archive CRC failure")
        for member in archive.infolist():
            if member.is_dir():
                continue
            path = VERSION.joinpath(*Path(member.filename).parts[1:])
            value = hashlib.sha256(archive.read(member)).hexdigest()
            check(value == digest(path), f"Archive member differs: {member.filename}")
            manifest["members"][member.filename] = {"path": str(path.relative_to(ROOT)),
                                                     "sha256": value, "size_bytes": member.file_size}
            if path.suffix == ".py":
                compile(path.read_bytes(), str(path), "exec")
                manifest["source_matches_baseline"][path.name] = (
                    value == digest(ROOT / "Sagar/src" / path.name))
    sources = manifest["source_matches_baseline"]
    check(len(sources) == 9, "Unexpected source count")
    config_path = VERSION / "environment/finetune_config_and_timings.json"
    check(digest(config_path) == digest(ORIGINAL / config_path.name), "Run configuration differs from addendum")
    run = json_file(config_path)
    with zipfile.ZipFile(ARCHIVED_MODEL) as model:
        check(model.testzip() is None, "Keras archive CRC failure")
        check(set(model.namelist()) == {"metadata.json", "config.json", "model.weights.h5"},
              "Unexpected Keras member set")
        metadata = json.loads(model.read("metadata.json"))
        layers = json.loads(model.read("config.json"))["config"]["layers"]
        inp, aug, backbone, dropout, out = layers
        check([l["class_name"] for l in layers] == ["InputLayer", "Sequential", "Functional", "Dropout", "Dense"],
              "Unexpected top-level architecture")
        check(inp["config"]["batch_shape"] == [None, 224, 224, 3]
              and dropout["config"]["rate"] == 0.3
              and out["config"]["units"] == 10 and out["config"]["activation"] == "softmax"
              and out["build_config"]["input_shape"] == [None, 1280], "Unexpected input or head")
        bb = backbone["config"]["layers"]
        last = run["unfreeze_top_layers"]
        bn = [l for l in bb if l["class_name"] == "BatchNormalization"]
        trainable = [l for l in bb if l["config"].get("trainable")]
        check(backbone["config"]["name"].startswith("mobilenetv2") and backbone["config"]["trainable"]
              and bb[-1]["class_name"] == "GlobalAveragePooling2D"
              and not any(l["config"].get("trainable") for l in bb[:-last] + bn)
              and len(trainable) == run["unfrozen_non_bn_layers"]
              and backbone["inbound_nodes"][0]["kwargs"]["training"] is False, "Unexpected backbone state")
        a = aug["config"]["layers"][1:]
        check([l["class_name"] for l in a] == ["RandomFlip", "RandomRotation", "RandomTranslation"]
              and a[0]["config"]["mode"] == "horizontal_and_vertical"
              and a[1]["config"]["factor"] == [-.25, .25]
              and a[2]["config"]["height_factor"] == a[2]["config"]["width_factor"] == .05
              and all(l["config"]["fill_mode"] == "reflect" for l in a[1:]), "Unexpected augmentation")
        with h5py.File(io.BytesIO(model.read("model.weights.h5")), "r") as weights:
            tensors = {}
            def validate(name, obj):
                if isinstance(obj, h5py.Dataset):
                    check(np.isfinite(obj[()]).all(), f"Nonfinite serialized values: {name}")
                    tensors[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype), "elements": obj.size}
            weights.visititems(validate)
        count = sum(v["elements"] for p, v in tensors.items() if p.startswith("layers/"))
        check(count == 2270794 and tensors["layers/dense/vars/0"]["shape"] == [1280, 10]
              and tensors["layers/dense/vars/1"]["shape"] == [10], "Unexpected serialized weights")
    manifest["keras_members_sha256"] = {}
    with zipfile.ZipFile(ARCHIVED_MODEL) as model:
        manifest["keras_members_sha256"] = {name: hashlib.sha256(model.read(name)).hexdigest()
                                            for name in model.namelist()}
    return manifest, {"keras_metadata": metadata, "model_input_shape": [None, 224, 224, 3],
                      "model_output_shape": [None, 10], "model_weight_elements": count,
                      "model_weight_tensors": sum(p.startswith("layers/") for p in tensors),
                      "optimizer_tensors": sum(p.startswith("optimizer/") for p in tensors),
                      "backbone_layers": len(bb), "frozen_batchnorm_layers": len(bn),
                      "trainable_backbone_layers": len(trainable),
                      "trainable_backbone_convolutions": sum(
                          l["class_name"] in ("Conv2D", "DepthwiseConv2D") for l in trainable),
                      "backbone_serialized_training_argument": False,
                      "preprocessing": run["preprocessing"], "run_configuration": run}


def scores(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=list(range(10)), average="macro", zero_division=0))}


def validate_probabilities(prob, n):
    check(prob.shape == (n, 10) and np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
          and abs(prob.sum(axis=1) - 1).max() < 1e-6, "Invalid probability matrix")


def infer(tf, h5, ids, batch_size):
    model = tf.keras.models.load_model(ARCHIVED_MODEL, compile=False)
    check(list(model.input_shape) == [None, 224, 224, 3]
          and list(model.output_shape) == [None, 10], "Loaded model shape differs")
    with h5py.File(h5, "r") as file:
        images = file["images"]
        def predict(raw):
            check(raw.dtype == np.uint8 and raw.shape[1:] == (256, 256, 3), "Unexpected raw image layout")
            x = tf.image.resize(tf.cast(raw, tf.float32), [224, 224], method="bilinear")
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
            p = np.asarray(model(x, training=False), dtype=np.float32)
            validate_probabilities(p, len(raw))
            return p
        smoke = predict(images[int(ids[0]):int(ids[0])+1])
        validate_probabilities(smoke, 1)
        batches = [predict(np.stack([images[int(row)] for row in ids[start:start + batch_size]]))
                   for start in range(0, len(ids), batch_size)]
    return np.concatenate(batches)


def compare(tf, h5, original, batch_size):
    ids = original.index.to_numpy()
    probs = infer(tf, h5, ids, batch_size)
    generated = pd.DataFrame(probs, index=original.index, columns=PROBS)
    generated["true_label"] = original.true_label
    generated["predicted_label"] = probs.argmax(axis=1)
    generated = generated[["true_label", "predicted_label", *PROBS]]
    aligned = original.join(generated, lsuffix="_original", rsuffix="_regenerated", validate="one_to_one")
    mismatch = aligned.predicted_label_original.ne(aligned.predicted_label_regenerated)
    errors = abs(aligned[[f"{c}_original" for c in PROBS]].to_numpy()
                 - aligned[[f"{c}_regenerated" for c in PROBS]].to_numpy())
    generated.to_csv(RESULTS / "regenerated_test_predictions.csv")
    aligned.loc[mismatch, ["true_label_original", "predicted_label_original",
                           "predicted_label_regenerated"]].to_csv(RESULTS / "class_mismatches.csv")
    return {"aligned_rows": len(aligned), "matching_classes": int((~mismatch).sum()),
            "matching_percent": float(100 * (~mismatch).mean()),
            "mismatched_classes": int(mismatch.sum()), "mismatched_row_ids": ids[mismatch].tolist(),
            "absolute_probability_difference": {"maximum": float(errors.max()), "mean": float(errors.mean()),
                                                "p99": float(np.quantile(errors, .99))},
            "regenerated_metrics": scores(generated.true_label, generated.predicted_label),
            "class_decisions_identical": not bool(mismatch.any())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", type=Path, default=H5)
    parser.add_argument("--dataset-json", type=Path, default=DATA / "dataset.json")
    parser.add_argument("--splits-json", type=Path, default=DATA / "splits.json")
    parser.add_argument("--assignments-csv", type=Path, default=DATA / "split_assignments.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    check(args.batch_size > 0, "Batch size must be positive")
    RESULTS.mkdir(exist_ok=True)
    manifest, audit = structure()
    print("Archive and model structures verified. Checking the locked data contract.", flush=True)
    contract = load_and_verify_contract(args.h5_path, args.dataset_json, args.splits_json,
                                        args.assignments_csv)
    published_contract = json_file(ROOT / "Sagar/results/frozen_baseline_full/provenance.json")
    hashes = {field: getattr(contract, field) for field in ("dataset_sha256", "dataset_manifest_sha256",
                                                          "splits_manifest_sha256", "assignments_csv_sha256")}
    check(all(value == published_contract[field] for field, value in hashes.items()), "Contract hash mismatch")
    p = ORIGINAL / "finetuned_test_predictions.csv"
    original = pd.read_csv(p, float_precision="round_trip").set_index("row_id", verify_integrity=True).sort_index()
    check(len(original) == 3545 and set(original.index) == set(contract.test)
          and original.true_label.tolist() == [contract.row_to_label[r] for r in original.index],
          "Published row IDs or labels differ from locked test")
    validate_probabilities(original[PROBS].to_numpy(), len(original))
    check(np.array_equal(original.predicted_label, original[PROBS].to_numpy().argmax(axis=1)),
          "Published prediction does not match probability argmax")
    reference = json_file(ORIGINAL / "finetuned_test_metrics.json")
    check([(p["class_id"], p["class_name"]) for p in reference["per_class"]]
          == sorted(contract.class_names.items()), "Published class mapping differs")
    original_scores = scores(original.true_label, original.predicted_label)
    check(all(abs(reference[key] - value) < 1e-12 for key, value in original_scores.items()),
          "Published metric differs")
    manifest.update({"data_contract_hashes": hashes, "dataset_path": str(args.h5_path),
                     "contract_paths": {key: str(getattr(args, key)) for key in
                                        ("dataset_json", "splits_json", "assignments_csv")},
                     "reference_predictions_sha256": digest(p),
                     "reference_metrics_sha256": digest(ORIGINAL / "finetuned_test_metrics.json"),
                     "verification_script_sha256": digest(Path(__file__))})
    audit.update({"static_model_and_archive": "PASS", "locked_test_contract": "PASS",
                  "class_mapping": contract.class_names, "published_rows": len(original),
                  "published_metrics_recomputed": original_scores, "python": platform.python_version(),
                  "python_executable": sys.executable, "batch_size": args.batch_size})
    write_json(RESULTS / "provenance.json", manifest)
    if not all(importlib.util.find_spec(package) for package in ("tensorflow", "keras")):
        audit.update({"model_load": "NOT TESTED", "inference": "BLOCKED",
                      "blocker": "TensorFlow and/or Keras are absent in this interpreter."})
    else:
        try:
            import tensorflow as tf
            import keras
            audit["tensorflow_version"], audit["keras_version"] = tf.__version__, keras.__version__
            audit["inference_devices"] = [device.name for device in tf.config.list_logical_devices()]
            audit["inference_comparison"] = compare(tf, args.h5_path, original, args.batch_size)
            audit.update({"model_load": "PASS", "inference": "PASS"})
        except Exception as error:
            audit.update({"model_load_and_inference": "FAILED", "error": repr(error)})
            write_json(RESULTS / "verification.json", audit)
            raise
    write_json(RESULTS / "verification.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
