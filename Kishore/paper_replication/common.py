"""Project constants and atomic artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEATURES = ("c", "a", "s", "g2", "h")
CLASSES = ("Elliptical", "Spiral")
THRESHOLDS = (5, 10, 20)
PAPER_ACCURACY = {5: {"dt": .948, "svm": .946, "mlp": .946},
                  10: {"dt": .957, "svm": .958, "mlp": .956},
                  20: {"dt": .985, "svm": .986, "mlp": .986}}


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    part.replace(path)


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    frame.to_csv(part, index=False)
    part.replace(path)
