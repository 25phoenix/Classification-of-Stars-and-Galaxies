from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import pandas as pd

from .config import GALAXY10_DECALS_CLASS_NAMES, NUM_CLASSES


@dataclass(frozen=True)
class FoldSplit:
    name: str
    train: np.ndarray
    validation: np.ndarray


@dataclass(frozen=True)
class OuterFoldSplit:
    name: str
    train: np.ndarray
    validation: np.ndarray
    inner: list[FoldSplit]


@dataclass(frozen=True)
class SharedDataContract:
    h5_path: Path
    dataset_sha256: str
    dataset_manifest_sha256: str
    splits_manifest_sha256: str
    assignments_csv_sha256: str
    class_names: dict[int, str]
    eligible_row_ids: np.ndarray
    row_to_label: dict[int, int]
    row_to_group: dict[int, str]
    development: np.ndarray
    test: np.ndarray
    outer_folds: list[OuterFoldSplit]
    final_tuning_folds: list[FoldSplit]


def compute_file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _extract_row_ids(obj: Any, context: str) -> np.ndarray:
    if isinstance(obj, list):
        arr = np.asarray(obj, dtype=np.int64)
    elif isinstance(obj, dict):
        for key in ("row_ids", "row_id", "indices", "ids", "rows"):
            if key in obj:
                arr = np.asarray(obj[key], dtype=np.int64)
                break
        else:
            raise ValueError(
                f"Cannot extract row IDs from {context}; found keys={list(obj.keys())}"
            )
    else:
        raise ValueError(f"Unsupported type for {context}: {type(obj).__name__}")

    if arr.ndim != 1:
        raise ValueError(f"{context} must be a 1D list of integer row IDs, got shape {arr.shape}")
    if len(np.unique(arr)) != len(arr):
        raise ValueError(f"{context} contains duplicate row IDs.")
    return arr


def _extract_fold_list(obj: Any, context: str) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(obj, list):
        out = []
        for idx, item in enumerate(obj):
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict for {context}[{idx}], got {type(item).__name__}")
            name = str(item.get("fold", item.get("name", f"{context}_{idx}")))
            out.append((name, item))
        return out
    if isinstance(obj, dict):
        out = []
        for key, item in obj.items():
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict for {context}[{key}], got {type(item).__name__}")
            out.append((str(key), item))
        return out
    raise ValueError(f"Expected list or dict for {context}, got {type(obj).__name__}")


def _extract_manifest_sha256(manifest: dict[str, Any]) -> str | None:
    candidate_keys = ("sha256", "dataset_sha256", "file_sha256", "checksum", "h5_sha256")
    for k in candidate_keys:
        val = manifest.get(k)
        if isinstance(val, str) and len(val.strip()) == 64:
            return val.strip().lower()
    for sub in manifest.values():
        if isinstance(sub, dict):
            found = _extract_manifest_sha256(sub)
            if found:
                return found
    return None


def _extract_manifest_classes(manifest: dict[str, Any]) -> dict[int, str] | None:
    for key in ("class_mapping", "class_names", "classes", "label_names", "id_to_name"):
        val = manifest.get(key)
        if isinstance(val, list) and len(val) == NUM_CLASSES:
            return {i: str(name) for i, name in enumerate(val)}
        if isinstance(val, dict) and len(val) == NUM_CLASSES:
            return {int(k): str(v) for k, v in val.items()}
    for sub in manifest.values():
        if isinstance(sub, dict):
            found = _extract_manifest_classes(sub)
            if found:
                return found
    return None


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...], required: bool, role: str) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(
            f"Missing required column for {role} in split_assignments.csv. "
            f"Tried {candidates}; actual columns={list(df.columns)}"
        )
    return None


def _assert_disjoint_and_group_safe(
    rows_a: np.ndarray,
    rows_b: np.ndarray,
    row_to_group: dict[int, str],
    name_a: str,
    name_b: str,
) -> None:
    set_a = set(int(r) for r in rows_a)
    set_b = set(int(r) for r in rows_b)
    row_overlap = set_a.intersection(set_b)
    if row_overlap:
        raise ValueError(
            f"ROW LEAKAGE between {name_a} and {name_b}: "
            f"{len(row_overlap)} overlapping rows (sample: {sorted(row_overlap)[:5]})"
        )
    groups_a = {row_to_group[r] for r in set_a}
    groups_b = {row_to_group[r] for r in set_b}
    group_overlap = groups_a.intersection(groups_b)
    if group_overlap:
        raise ValueError(
            f"GROUP LEAKAGE between {name_a} and {name_b}: "
            f"{len(group_overlap)} duplicate/object groups cross the boundary "
            f"(sample: {sorted(group_overlap)[:5]})"
        )


def load_and_verify_contract(
    h5_path: Path,
    dataset_json_path: Path,
    splits_json_path: Path,
    assignments_csv_path: Path,
    verify_checksum: bool = True,
) -> SharedDataContract:
    """Load Kishore's saved manifests and strictly validate the shared contract."""
    missing = [
        str(p)
        for p in (h5_path, dataset_json_path, splits_json_path, assignments_csv_path)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required shared files are missing. Do NOT substitute a random split.\n"
            f"Missing paths:\n  - " + "\n  - ".join(missing)
        )

    with dataset_json_path.open("r", encoding="utf-8") as f:
        dataset_manifest = json.load(f)
    with splits_json_path.open("r", encoding="utf-8") as f:
        splits_manifest = json.load(f)
    assignments_df = pd.read_csv(assignments_csv_path)

    # 1. Verify dataset checksum
    actual_h5_sha256 = compute_file_sha256(h5_path)
    expected_sha256 = _extract_manifest_sha256(dataset_manifest)
    if verify_checksum and expected_sha256 is not None:
        if actual_h5_sha256.lower() != expected_sha256.lower():
            raise ValueError(
                f"Dataset SHA256 mismatch!\n"
                f"  Expected (dataset.json): {expected_sha256}\n"
                f"  Actual   ({h5_path.name}): {actual_h5_sha256}"
            )

    # 2. Verify class mapping matches Galaxy10 DECaLS
    manifest_classes = _extract_manifest_classes(dataset_manifest)
    if manifest_classes is not None:
        if set(manifest_classes.keys()) != set(GALAXY10_DECALS_CLASS_NAMES.keys()):
            raise ValueError(f"Class IDs in dataset.json do not match 0..9: {manifest_classes}")
        for cid, official_name in GALAXY10_DECALS_CLASS_NAMES.items():
            if _normalize_name(manifest_classes[cid]) != _normalize_name(official_name):
                raise ValueError(
                    f"Class {cid} name mismatch: dataset.json has '{manifest_classes[cid]}', "
                    f"expected Galaxy10 DECaLS '{official_name}'."
                )
        class_names = dict(GALAXY10_DECALS_CLASS_NAMES)
    else:
        class_names = dict(GALAXY10_DECALS_CLASS_NAMES)

    # 3. Parse split_assignments.csv
    row_col = _resolve_column(
        assignments_df,
        ("row_id", "h5_row_id", "original_row_id", "h5_index", "original_index", "index"),
        required=True,
        role="row_id",
    )
    label_col = _resolve_column(
        assignments_df,
        ("label", "ans", "class_id", "y", "target"),
        required=True,
        role="label",
    )
    group_col = _resolve_column(
        assignments_df,
        ("group_id", "group", "object_group", "duplicate_group", "cluster_id"),
        required=True,
        role="duplicate/object group",
    )
    eligible_col = _resolve_column(
        assignments_df,
        ("eligible", "is_eligible", "valid", "is_valid", "included"),
        required=False,
        role="eligibility flag",
    )
    partition_col = _resolve_column(
        assignments_df,
        ("partition", "split"),
        required=False,
        role="partition",
    )

    if eligible_col is not None:
        eligible_df = assignments_df.loc[assignments_df[eligible_col].astype(bool)].copy()
    elif partition_col is not None:
        part_norm = assignments_df[partition_col].fillna("").astype(str).str.strip().str.lower()
        eligible_df = assignments_df.loc[part_norm.isin({"development", "test"})].copy()
    else:
        eligible_df = assignments_df.copy()

    eligible_row_ids = eligible_df[row_col].astype(np.int64).to_numpy()
    if len(np.unique(eligible_row_ids)) != len(eligible_row_ids):
        raise ValueError("split_assignments.csv contains duplicate eligible row IDs.")

    if isinstance(dataset_manifest.get("usable"), int):
        if len(eligible_row_ids) != int(dataset_manifest["usable"]):
            raise ValueError(
                f"Eligible row count ({len(eligible_row_ids)}) does not match "
                f"dataset.json usable count ({dataset_manifest['usable']})."
            )

    csv_labels = eligible_df[label_col].astype(np.int64).to_numpy()
    csv_groups = eligible_df[group_col].astype(str).to_numpy()

    row_to_label = {int(r): int(y) for r, y in zip(eligible_row_ids, csv_labels)}
    row_to_group = {int(r): str(g) for r, g in zip(eligible_row_ids, csv_groups)}
    eligible_set = set(int(r) for r in eligible_row_ids)

    # 4. Verify HDF5 existence, shape, and exact label agreement
    with h5py.File(h5_path, "r") as f:
        if "images" not in f or "ans" not in f:
            raise ValueError(f"HDF5 file must contain 'images' and 'ans' datasets; found {list(f.keys())}")
        n_total = f["images"].shape[0]
        if eligible_row_ids.min() < 0 or eligible_row_ids.max() >= n_total:
            raise ValueError(
                f"Row IDs out of bounds for HDF5 length {n_total}: "
                f"min={eligible_row_ids.min()}, max={eligible_row_ids.max()}"
            )
        h5_all_labels = np.asarray(f["ans"][:], dtype=np.int64)

    for r in eligible_row_ids:
        r_int = int(r)
        if int(h5_all_labels[r_int]) != row_to_label[r_int]:
            raise ValueError(
                f"Label disagreement at row {r_int}: HDF5 has {h5_all_labels[r_int]}, "
                f"split_assignments.csv has {row_to_label[r_int]}"
            )

    # 5. Parse splits.json (development, test, outer, final_tuning)
    for required_key in ("development", "test", "outer", "final_tuning"):
        if required_key not in splits_manifest:
            raise ValueError(
                f"splits.json missing required key '{required_key}'; "
                f"actual keys={list(splits_manifest.keys())}"
            )

    development = _extract_row_ids(splits_manifest["development"], "splits.development")
    test = _extract_row_ids(splits_manifest["test"], "splits.test")

    def _assert_subset_of_eligible(rows: np.ndarray, context: str) -> None:
        unknown = set(int(r) for r in rows) - eligible_set
        if unknown:
            raise ValueError(
                f"{context} references {len(unknown)} rows not in eligible population "
                f"(sample: {sorted(unknown)[:5]})"
            )

    _assert_subset_of_eligible(development, "splits.development")
    _assert_subset_of_eligible(test, "splits.test")
    _assert_disjoint_and_group_safe(development, test, row_to_group, "development", "test")

    dev_set = set(int(r) for r in development)
    test_set = set(int(r) for r in test)
    if dev_set.union(test_set) != eligible_set:
        raise ValueError(
            "Union of development and test does not equal the eligible population in split_assignments.csv."
        )

    # Cross-check partition labels in split_assignments.csv against splits.json
    if partition_col is not None:
        part_norm = eligible_df[partition_col].fillna("").astype(str).str.strip().str.lower()
        csv_dev = set(eligible_df.loc[part_norm == "development", row_col].astype(np.int64).tolist())
        csv_test = set(eligible_df.loc[part_norm == "test", row_col].astype(np.int64).tolist())
        if csv_dev != dev_set or csv_test != test_set:
            raise ValueError(
                "Partition assignments in split_assignments.csv disagree with splits.json!"
            )

    # Parse outer folds (each with train, validation, inner)
    outer_folds: list[OuterFoldSplit] = []
    outer_val_union: set[int] = set()
    for fname, fdict in _extract_fold_list(splits_manifest["outer"], "splits.outer"):
        if "train" not in fdict or "validation" not in fdict or "inner" not in fdict:
            raise ValueError(f"Outer fold '{fname}' must have 'train', 'validation', and 'inner'.")
        o_train = _extract_row_ids(fdict["train"], f"outer.{fname}.train")
        o_val = _extract_row_ids(fdict["validation"], f"outer.{fname}.validation")
        _assert_subset_of_eligible(o_train, f"outer.{fname}.train")
        _assert_subset_of_eligible(o_val, f"outer.{fname}.validation")
        _assert_disjoint_and_group_safe(o_train, o_val, row_to_group, f"outer.{fname}.train", f"outer.{fname}.val")
        _assert_disjoint_and_group_safe(o_train, test, row_to_group, f"outer.{fname}.train", "test")
        _assert_disjoint_and_group_safe(o_val, test, row_to_group, f"outer.{fname}.val", "test")

        o_train_set = set(int(r) for r in o_train)
        o_val_set = set(int(r) for r in o_val)
        if o_train_set.union(o_val_set) != dev_set:
            raise ValueError(f"Outer fold '{fname}' train+validation does not cover development set exactly.")

        if outer_val_union.intersection(o_val_set):
            raise ValueError(f"Outer validation fold '{fname}' overlaps with a previous outer validation fold.")
        outer_val_union.update(o_val_set)

        inner_list: list[FoldSplit] = []
        for iname, idict in _extract_fold_list(fdict["inner"], f"outer.{fname}.inner"):
            i_train = _extract_row_ids(idict["train"], f"outer.{fname}.inner.{iname}.train")
            i_val = _extract_row_ids(idict["validation"], f"outer.{fname}.inner.{iname}.validation")
            _assert_disjoint_and_group_safe(
                i_train, i_val, row_to_group, f"inner.{iname}.train", f"inner.{iname}.val"
            )
            if not set(int(r) for r in i_train).union(int(r) for r in i_val).issubset(o_train_set):
                raise ValueError(
                    f"Inner fold '{iname}' in outer fold '{fname}' leaks outside outer train partition!"
                )
            inner_list.append(FoldSplit(name=iname, train=i_train, validation=i_val))

        outer_folds.append(OuterFoldSplit(name=fname, train=o_train, validation=o_val, inner=inner_list))

    if outer_val_union != dev_set:
        raise ValueError("Union of outer validation folds does not equal development set.")

    # Parse final_tuning folds (must stay strictly inside development)
    final_tuning_folds: list[FoldSplit] = []
    for fname, fdict in _extract_fold_list(splits_manifest["final_tuning"], "splits.final_tuning"):
        ft_train = _extract_row_ids(fdict["train"], f"final_tuning.{fname}.train")
        ft_val = _extract_row_ids(fdict["validation"], f"final_tuning.{fname}.validation")
        _assert_disjoint_and_group_safe(
            ft_train, ft_val, row_to_group, f"final_tuning.{fname}.train", f"final_tuning.{fname}.val"
        )
        _assert_disjoint_and_group_safe(ft_train, test, row_to_group, f"final_tuning.{fname}.train", "test")
        _assert_disjoint_and_group_safe(ft_val, test, row_to_group, f"final_tuning.{fname}.val", "test")
        if not set(int(r) for r in ft_train).union(int(r) for r in ft_val).issubset(dev_set):
            raise ValueError(f"final_tuning fold '{fname}' contains rows outside development.")
        final_tuning_folds.append(FoldSplit(name=fname, train=ft_train, validation=ft_val))

    return SharedDataContract(
        h5_path=h5_path,
        dataset_sha256=actual_h5_sha256,
        dataset_manifest_sha256=compute_file_sha256(dataset_json_path),
        splits_manifest_sha256=compute_file_sha256(splits_json_path),
        assignments_csv_sha256=compute_file_sha256(assignments_csv_path),
        class_names=class_names,
        eligible_row_ids=np.sort(eligible_row_ids),
        row_to_label=row_to_label,
        row_to_group=row_to_group,
        development=development,
        test=test,
        outer_folds=outer_folds,
        final_tuning_folds=final_tuning_folds,
    )


def iter_h5_images_by_row_ids(
    h5_path: Path,
    row_ids: np.ndarray,
    batch_size: int = 64,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield (batch_row_ids, batch_images_uint8) in bounded memory while preserving
    exact row_ids order. When row_ids are sorted, reads aligned HDF5 chunks once
    (~109 MB for Galaxy10's 555-row gzip chunks) instead of re-decompressing the
    same chunk on every small GPU batch.
    """
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if len(row_ids) == 0:
        return

    with h5py.File(h5_path, "r") as f:
        images_ds = f["images"]
        n_total = images_ds.shape[0]
        is_sorted = bool(np.all(row_ids[:-1] <= row_ids[1:]))

        if is_sorted:
            chunk_rows = images_ds.chunks[0] if images_ds.chunks else 512
            chunk_rows = max(1, min(int(chunk_rows), 1024))
            for block_lo in range(0, n_total, chunk_rows):
                block_hi = min(n_total, block_lo + chunk_rows)
                i0 = int(np.searchsorted(row_ids, block_lo, side="left"))
                i1 = int(np.searchsorted(row_ids, block_hi, side="left"))
                if i0 == i1:
                    continue
                block_ids = row_ids[i0:i1]
                block_imgs = np.asarray(images_ds[block_lo:block_hi])
                selected_imgs = block_imgs[block_ids - block_lo]
                for start in range(0, len(block_ids), batch_size):
                    yield (
                        block_ids[start : start + batch_size],
                        selected_imgs[start : start + batch_size],
                    )
                del block_imgs, selected_imgs
            return

        for start in range(0, len(row_ids), batch_size):
            batch_ids = row_ids[start : start + batch_size]
            unique_sorted, inv_idx = np.unique(batch_ids, return_inverse=True)
            lo = int(unique_sorted[0])
            hi = int(unique_sorted[-1]) + 1
            if (hi - lo) <= 2 * len(unique_sorted):
                slice_imgs = images_ds[lo:hi]
                batch_unique_imgs = slice_imgs[unique_sorted - lo]
            else:
                batch_unique_imgs = images_ds[unique_sorted]
            batch_imgs = batch_unique_imgs[inv_idx]
            yield batch_ids, np.asarray(batch_imgs)


def read_h5_images(h5_path: Path, row_ids: np.ndarray) -> np.ndarray:
    """Read a small list of row IDs (e.g., for prediction gallery) in exact order."""
    chunks = [imgs for _, imgs in iter_h5_images_by_row_ids(h5_path, row_ids, batch_size=64)]
    if not chunks:
        return np.empty((0, 256, 256, 3), dtype=np.uint8)
    return np.concatenate(chunks, axis=0)