"""Audit the staged DR7 export and exclude every ambiguous exact object ID."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from catalogs import prepare
from common import ROOT, sha256, write_csv, write_json


OUT = ROOT / "results/catalog_dr7_complete_case"
FIELDS = ["objid", "z", "petromag_r"]


def inspect_export(raw):
    if list(raw.columns) != FIELDS:
        raise ValueError(f"Expected {FIELDS}, found {list(raw.columns)}")
    if not raw.objid.str.fullmatch(r"[0-9]+", na=False).all():
        raise ValueError("Export contains missing or non-decimal object IDs")
    numeric = raw[FIELDS[1:]].apply(pd.to_numeric, errors="coerce")
    # Keep unparseable values distinct for duplicate validation. An object
    # with both a missing and a valid measurement is also ambiguous.
    pairs = numeric.astype(object).where(numeric.notna(), raw[FIELDS[1:]])
    distinct = pairs.assign(objid=raw.objid).drop_duplicates()
    conflicts = set(distinct.loc[distinct.objid.duplicated(keep=False), "objid"])
    clean = raw.loc[~raw.objid.isin(conflicts)].drop_duplicates(subset=["objid"])
    return clean, conflicts, {
        "rows": len(raw), "columns": list(raw), "unique_ids": raw.objid.nunique(),
        "duplicate_ids": raw.loc[raw.objid.duplicated(keep=False), "objid"].nunique(),
        "conflicting_ids": len(conflicts), "extra_rows_beyond_unique_measurements": len(raw) - len(distinct),
        "missing": {name: int(raw[name].eq("").sum()) for name in FIELDS},
        "nonfinite_or_unparseable": (~np.isfinite(numeric)).sum().to_dict(),
        "petromag_le_minus_100": int(numeric.petromag_r.le(-100).sum()),
    }


def coverage(author_ids, export_ids):
    matched = author_ids.isin(export_ids)
    return {"matched_rows": int(matched.sum()), "unmatched_rows": int((~matched).sum()),
            "matched_ids": author_ids[matched].nunique(), "unmatched_ids": author_ids[~matched].nunique()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    args = parser.parse_args()
    raw = pd.read_csv(args.export, dtype=str, keep_default_na=False)
    clean, conflicts, summary = inspect_export(raw)
    table, _ = prepare(ROOT / "data/raw", ROOT / "cache/catalogs")
    summary.update({"source": "User-staged SDSS CasJobs DR7 CSV, dr7_metadata.sql",
                    "path": str(args.export), "sha256": sha256(args.export),
                    "size_bytes": args.export.stat().st_size,
                    "author_rows": len(table), "author_unique_ids": table.objid.nunique(),
                    "raw_join": coverage(table.objid, raw.objid),
                    "unambiguous_join": coverage(table.objid, clean.objid),
                    "policy": "Exclude all rows of any ID with conflicting measurements. "
                              "Preserve unambiguous missing measurements for workflow exclusion."})
    conflicting = raw.loc[raw.objid.isin(conflicts)].assign(
        in_author_catalog=lambda rows: rows.objid.isin(table.objid))
    write_csv(OUT / "conflicting_dr7_measurements.csv", conflicting)
    cleaned = OUT / "unambiguous_dr7_metadata.csv"
    write_csv(cleaned, clean)
    summary["clean_export"] = {"path": str(cleaned.relative_to(ROOT)), "rows": len(clean),
                               "sha256": sha256(cleaned)}
    write_json(OUT / "metadata_export_audit.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
