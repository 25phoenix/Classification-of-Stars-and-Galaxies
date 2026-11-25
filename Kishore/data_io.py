"""Acquisition, integrity checks, grouping, and feature cache."""
import hashlib
import fcntl
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

import morphology
from acquisition import DATA, ROOT, SHA256, URL, digest, verify
from morphology import FEATURES, Settings, extract

SOURCE = "https://astronn.readthedocs.io/en/latest/galaxy10.html"
CLASSES = (
    "Disturbed Galaxies", "Merging Galaxies", "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies", "Cigar Shaped Smooth Galaxies",
    "Barred Spiral Galaxies", "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies", "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge",
)
COUNTS = [1081, 1853, 2645, 2027, 334, 2043, 1829, 2628, 1423, 1873]
SETTINGS = Settings()
CACHE = ROOT / "cache"


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".tmp")
    part.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    part.replace(path)


def inspect_file(file, official=True):
    schema = {}
    file.visititems(lambda key, value: schema.update({
        key: {"shape": list(value.shape), "dtype": str(value.dtype),
              "chunks": value.chunks, "compression": value.compression}
    }) if isinstance(value, h5py.Dataset) else None)
    images = file["images"]
    raw_labels = file["ans"][:]
    if raw_labels.shape != (len(images),) or not np.isin(raw_labels, np.arange(10)).all():
        raise ValueError("Invalid label schema or values")
    labels = raw_labels.astype(int)
    counts = np.bincount(labels, minlength=10).tolist()
    if official and (images.shape != (17736, 256, 256, 3) or counts != COUNTS
                     or images.dtype != np.uint8):
        raise ValueError(f"Unexpected original image schema or class counts: {schema}, {counts}")
    for field in ("ra", "dec"):
        if file[field].shape != labels.shape:
            raise ValueError(f"Expected one {field} per image")
    if official and any(field not in schema for field in ("redshift", "pxscale")):
        raise ValueError("Missing documented redshift or pxscale dataset")
    return schema, labels, counts


def groups_for(hashes, ra, dec):
    parent = list(range(len(hashes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def merge(i, j):
        a, b = sorted((root(i), root(j)))
        parent[b] = a

    first = {}
    for i, image_hash in enumerate(hashes):
        merge(i, first.setdefault(image_hash, i))
    valid = np.isfinite(ra) & np.isfinite(dec) & (ra >= 0) & (ra <= 360) & (abs(dec) <= 90)
    indices = np.flatnonzero(valid)
    a, d = np.deg2rad(ra[valid]), np.deg2rad(dec[valid])
    xyz = np.column_stack((np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)))
    pairs = cKDTree(xyz).query_pairs(2 * np.sin(np.deg2rad(1 / 3600) / 2))
    for i, j in pairs:
        merge(int(indices[i]), int(indices[j]))
    groups = np.array([root(i) for i in range(len(parent))])
    sizes = Counter(groups)
    duplicates = Counter(hashes)
    return groups, {
        "exact_duplicate_groups": sum(n > 1 for n in duplicates.values()),
        "exact_duplicate_rows": sum(n for n in duplicates.values() if n > 1),
        "coordinate_pairs": len(pairs), "invalid_coordinates": int((~valid).sum()),
        "joint_groups": len(sizes), "joint_non_singleton_groups": sum(n > 1 for n in sizes.values()),
        "joint_non_singleton_rows": sum(n for n in sizes.values() if n > 1),
        "largest_group": max(sizes.values(), default=0),
        "rule": "Connected components of identical decoded RGB SHA-256 or RA/Dec within 1 arcsec.",
    }


def process_image(image):
    values, flags, _ = extract(image, SETTINGS)
    return hashlib.sha256(image.tobytes()).hexdigest(), values, "|".join(flags)


def measure_batches(images, folder, workers):
    """Checkpoint ordered rows within the cache identified by the dataset and code."""
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    with folder.joinpath("extract.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with Parallel(n_jobs=workers, prefer="processes") as pool:
            for start in range(0, len(images), 128):
                stop = min(start + 128, len(images))
                checkpoint = folder / f"{start:05d}.json"
                if checkpoint.is_file():
                    batch = json.loads(checkpoint.read_text())
                else:
                    measured = pool(delayed(process_image)(image) for image in images[start:stop])
                    batch = [[sha, {k: v if np.isfinite(v) else None for k, v in values.items()}, flags]
                             for sha, values, flags in measured]
                    write_json(checkpoint, batch)
                if len(batch) != stop - start:
                    raise ValueError(f"Unexpected batch length in {checkpoint}")
                records.extend(batch)
                if start % 2048 == 0 or stop == len(images):
                    print(f"Measured {len(records)}/{len(images)}", flush=True)
    return records


def extract_table(path=DATA, cache=CACHE, workers=4, official=True):
    if official:
        verify()
    checksum = SHA256 if official else digest(path)
    config = {"sha256": checksum, "settings": asdict(SETTINGS), "features": FEATURES,
              "mode": "official" if official else "synthetic_smoke",
              "code": {p.name: digest(p) for p in (Path(__file__), Path(morphology.__file__))},
              "numpy": np.__version__, "scipy": __import__("scipy").__version__}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    folder = cache / key
    table_path = folder / "features.csv"
    provenance_path = folder / "dataset.json"
    if table_path.exists() and provenance_path.exists():
        table = pd.read_csv(table_path, keep_default_na=False, float_precision="round_trip",
                            na_values={f: [""] for f in FEATURES})
        provenance = json.loads(provenance_path.read_text())
        if provenance["fingerprint"] != key or provenance["feature_table_sha256"] != digest(table_path):
            raise ValueError("Feature cache integrity failure")
        return table, provenance

    with h5py.File(path, "r") as file:
        schema, labels, counts = inspect_file(file, official)
        ra, dec = file["ra"][:], file["dec"][:]
        records = measure_batches(file["images"], folder / "batches", workers)
    hashes, features, flags = zip(*records)
    groups, audit = groups_for(hashes, ra, dec)
    table = pd.DataFrame(features, columns=FEATURES, dtype=float)
    table.insert(0, "row_id", np.arange(len(labels)))
    table["label"] = labels
    table["group_id"] = groups
    table["image_sha256"] = hashes
    table["flags"] = flags
    table["usable"] = np.isfinite(table[list(FEATURES)]).all(axis=1)
    audit["mixed_label_groups"] = int((table.groupby("group_id").label.nunique() > 1).sum())
    folder.mkdir(parents=True, exist_ok=True)
    partial = table_path.with_suffix(".tmp")
    table.to_csv(partial, index=False)
    partial.replace(table_path)
    provenance = {**config, "fingerprint": key, "feature_table_sha256": digest(table_path),
                  "source": URL if official else "synthetic software smoke fixture",
                  "class_mapping_source": SOURCE, "class_mapping": dict(enumerate(CLASSES)),
                  "size_bytes": path.stat().st_size, "schema": schema, "class_counts": counts,
                  "audit": audit, "rows": len(table),
                  "usable": int(table.usable.sum()), "excluded": int((~table.usable).sum()),
                  "excluded_by_class": np.bincount(labels[~table.usable], minlength=10).tolist(),
                  "exclusion_reasons_by_class": {
                      str(i): dict(Counter(table.loc[(table.label == i) & ~table.usable, "flags"]))
                      for i in range(len(CLASSES))},
                  "flag_counts": dict(Counter(flag for entry in flags for flag in entry.split("|") if flag)),
                  "feature_table": str(table_path.relative_to(ROOT))}
    write_json(provenance_path, provenance)
    return table, json.loads(provenance_path.read_text())
