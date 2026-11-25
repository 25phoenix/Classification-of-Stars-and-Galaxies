"""Join released CyMorph measurements to independent GZ1 classifications."""
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import FEATURES, THRESHOLDS, sha256, write_csv, write_json

AUTHOR_FILE = "Barchi19_Morph-catalog_670k-galaxies.zip"
GZ_FILE = "GalaxyZoo1_DR_table2.csv.gz"
AUTHOR_URL = "https://data.mendeley.com/datasets/tg7c985c2n/1"
GZ_URL = "https://galaxy-zoo-1.s3.amazonaws.com/GalaxyZoo1_DR_table2.csv.gz"

ALIASES = {
    "objid": {"sdssobjid", "dr7objid", "objid", "sdssid", "sdssdr7objid", "sdssdr7id"},
    "k": {"k"}, "error": {"error", "cymorpherror", "errorflag"},
    "c": {"c", "concentration"}, "a": {"a", "asymmetry"},
    "s": {"s", "smoothness"}, "g2": {"g2", "ga", "gpa"},
    "h": {"h", "entropy"}, "ra": {"ra"}, "dec": {"dec", "declination"},
    "elliptical": {"elliptical"}, "spiral": {"spiral"},
    "uncertain": {"uncertain"}, "p_mg": {"pmg"},
    "z": {"z", "redshift"}, "petromag_r": {"petromagr", "rpetromag"},
    "b": {"b", "galacticlatitude", "galacticb"}, "image_sha256": {"imagesha256"},
}


def read_table(path):
    if path.suffix != ".zip":
        return pd.read_csv(path, dtype=str, skipinitialspace=True), [path.name]
    with ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")
                   and not Path(n).name.startswith("._")]
        if len(members) != 1:
            raise ValueError(f"Expected one catalog CSV in {path.name}. ZIP entries: {archive.namelist()}")
        with archive.open(members[0]) as stream:
            return pd.read_csv(stream, dtype=str, skipinitialspace=True), members


def columns(frame, required, optional=()):
    result = {}
    for field in (*required, *optional):
        matches = [name for name in frame if re.sub(r"[^a-z0-9]", "", name.lower()) in ALIASES[field]]
        if len(matches) > 1 or (not matches and field in required):
            raise ValueError(f"Expected one column for {field}. Found {matches}. Headers: {list(frame)}")
        if matches:
            result[field] = frame[matches[0]].str.strip()
    return pd.DataFrame(result, index=frame.index)


def object_ids(values):
    # Parsing via float would round 18-digit SDSS object IDs.
    if not values.str.fullmatch(r"[1-9][0-9]{14,18}", na=False).all():
        examples = values[~values.str.fullmatch(r"[1-9][0-9]{14,18}", na=False)].head().tolist()
        raise ValueError(f"Expected exact decimal SDSS object identifiers. Examples: {examples}")
    return values


def angle_degrees(values, is_ra):
    """Read numeric degrees, or GZ1 RA as hours and DEC as signed DMS."""
    numeric = pd.to_numeric(values, errors="coerce")
    sexagesimal = values.astype("string").str.contains(":", na=False)
    if sexagesimal.any():
        parts = values[sexagesimal].str.extract(
            r"^([+-]?)([0-9]{1,3}):([0-5][0-9]):([0-5][0-9](?:\.[0-9]+)?)$")
        sign = parts[0].map({"+": 1, "": 1, "-": -1})
        scale = 15 if is_ra else 1
        converted = scale * sign * (parts[1].astype(float) + parts[2].astype(float)/60
                                   + parts[3].astype(float)/3600)
        numeric.loc[sexagesimal] = converted
    valid = numeric.between(0, 360) if is_ra else numeric.between(-90, 90)
    return numeric.where(valid)


def k_value(radius, fwhm, radius_unit="arcsec", fwhm_unit="arcsec", pixel_scale=.396):
    units = {"arcsec": 1, "pixel": pixel_scale}
    radius = np.asarray(radius, dtype=float) * units[radius_unit]
    fwhm = np.asarray(fwhm, dtype=float) * units[fwhm_unit]
    good = np.isfinite(radius) & np.isfinite(fwhm) & (radius > 0) & (fwhm > 0)
    result = np.full(np.broadcast_shapes(radius.shape, fwhm.shape), np.nan)
    np.divide(2 * radius, fwhm, where=good, out=result)
    return result ** 2


def grouping(frame):
    parent = np.arange(len(frame))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def merge(a, b):
        a, b = sorted((root(a), root(b)))
        parent[b] = a

    repeated = {}
    for field in ("objid", "image_sha256"):
        if field not in frame:
            continue
        first = {}
        pairs = 0
        for i, value in enumerate(frame[field]):
            if pd.notna(value) and value:
                j = first.setdefault(value, i)
                if i != j:
                    merge(i, j)
                    pairs += 1
        repeated[field] = pairs
    coordinate_pairs = 0
    if {"ra", "dec"}.issubset(frame.columns):
        ra, dec = frame.ra.to_numpy(), frame.dec.to_numpy()
        good = np.isfinite(ra) & np.isfinite(dec) & (0 <= ra) & (ra <= 360) & (abs(dec) <= 90)
        rows = np.flatnonzero(good)
        ra, dec = np.deg2rad(ra[good]), np.deg2rad(dec[good])
        xyz = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))
        matches = cKDTree(xyz).query_pairs(2*np.sin(np.deg2rad(1/3600)/2), output_type="ndarray")
        for i, j in matches:
            merge(rows[i], rows[j])
        coordinate_pairs = len(matches)
        repeated["unavailable_coordinates"] = int((~good).sum())
    repeated["one_arcsec_pairs"] = coordinate_pairs
    groups = np.array([root(i) for i in range(len(frame))])
    repeated["joint_groups"] = int(len(np.unique(groups)))
    repeated["non_singleton_groups"] = int(np.sum(np.unique(groups, return_counts=True)[1] > 1))
    return groups, repeated


def select(author, zoo):
    source = columns(author, ("objid", "k", "error", *FEATURES),
                     ("ra", "dec", "z", "petromag_r", "b", "image_sha256"))
    votes = columns(zoo, ("objid", "elliptical", "spiral", "uncertain"), ("ra", "dec", "p_mg"))
    source["objid"], votes["objid"] = object_ids(source.objid), object_ids(votes.objid)
    votes = votes.drop_duplicates()
    if votes.objid.duplicated().any():
        raise ValueError("Galaxy Zoo label catalog has conflicting repeated object IDs")
    votes = votes.rename(columns={"ra": "gz_ra", "dec": "gz_dec"})
    table = source.merge(votes, on="objid", how="left", validate="many_to_one", indicator=True)
    if not table.objid.equals(source.objid.reset_index(drop=True)):
        raise ValueError("Exact object IDs or author row order changed during the catalog join")
    table.insert(0, "row_id", np.arange(len(table)))
    table["label_provenance"] = np.where(table._merge.eq("both"), "GZ1 Table 2", "")
    coordinate_fields = {"ra", "dec", "gz_ra", "gz_dec"}
    for name in coordinate_fields & set(table.columns):
        table[name] = angle_degrees(table[name], is_ra=name.endswith("ra"))
    for name in set(table.columns) - {"objid", "_merge", "image_sha256", "label_provenance"} - coordinate_fields:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    for field in ("ra", "dec"):
        zoo_name = f"gz_{field}"
        if zoo_name in table:
            table[field] = table[field].fillna(table[zoo_name]) if field in table else table[zoo_name]
    if "b" not in table and {"ra", "dec"}.issubset(table.columns):
        # IAU Galactic north pole in ICRS (degrees).
        ra, dec = np.deg2rad(table.ra), np.deg2rad(table.dec)
        pole_ra, pole_dec = np.deg2rad([192.85948, 27.12825])
        table["b"] = np.rad2deg(np.arcsin(np.clip(np.sin(dec)*np.sin(pole_dec) +
            np.cos(dec)*np.cos(pole_dec)*np.cos(ra-pole_ra), -1, 1)))
    is_e = table.elliptical.eq(1) & table.spiral.eq(0) & table.uncertain.eq(0)
    is_s = table.spiral.eq(1) & table.elliptical.eq(0) & table.uncertain.eq(0)
    if "p_mg" in table:
        # An explicit merger-majority vote cannot supply an E/S training label.
        is_e &= ~table.p_mg.ge(.5)
        is_s &= ~table.p_mg.ge(.5)
    table["label"] = np.select([is_e, is_s], [0, 1], default=-1)
    reasons = [
        (table._merge.ne("both"), "no_gz1_match"),
        (table.label.lt(0), "no_defined_gz1_label"),
        (~table.error.eq(0), "cymorph_error"),
        (~np.isfinite(table[list(FEATURES)]).all(axis=1), "nonfinite_features"),
        (~np.isfinite(table.k), "invalid_k"),
    ]
    independent = {}
    for name, predicate in (("z", table.get("z", pd.Series(np.nan, index=table.index)).between(.03, .1, inclusive="neither")),
                            ("petromag_r", table.get("petromag_r", pd.Series(np.nan, index=table.index)).lt(17.78)),
                            ("b", table.get("b", pd.Series(np.nan, index=table.index)).abs().ge(30))):
        if name in table:
            measured = np.isfinite(table[name])
            independent[name] = int(measured.sum())
            reasons.append((measured & ~predicate, f"outside_{name}_selection"))
    table["common_reason"] = np.select([r for r, _ in reasons], [v for _, v in reasons], default="")
    for k in THRESHOLDS:
        table[f"selection_k{k}"] = np.where(table.common_reason.ne(""), table.common_reason,
                                                np.where(table.k.ge(k), "included", "below_k"))
    table["group_id"], audit = grouping(table)
    audit["mixed_label_groups"] = int((table.loc[table.label.ge(0)].groupby("group_id").label.nunique() > 1).sum())
    return table.drop(columns="_merge"), {
        "grouping": audit, "independently_checked_catalog_rows": independent,
        "exact_id_join": {"author_rows": len(source), "unique_author_ids": int(source.objid.nunique()),
                          "gz1_unique_ids": len(votes), "matched": int(table._merge.eq("both").sum()),
                          "unmatched": int(table._merge.ne("both").sum()),
                          "key": "DR7 decimal object-ID strings",
                          "labels": "Independent GZ1 Table 2 clean flags; reject majority-merger votes"},
        "reason_counts": {str(k): table.groupby([f"selection_k{k}", "label"]).size()
                          .rename("rows").reset_index().to_dict("records") for k in THRESHOLDS},
    }


def prepare(raw, cache):
    files = [raw / AUTHOR_FILE, raw / GZ_FILE]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing authoritative inputs: " + ", ".join(missing))
    info = {"inputs": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p), "url": url}
                       for p, url in zip(files, (AUTHOR_URL, GZ_URL))],
            "author_release": "10.17632/tg7c985c2n.1, CC BY 4.0",
            "selection_code_sha256": sha256(Path(__file__))}
    from hashlib import sha256 as digest
    import json
    stamp = digest(json.dumps(info, sort_keys=True).encode()).hexdigest()
    feature_dir = cache / stamp
    table_path = feature_dir / "catalog_selection.csv"
    metadata = feature_dir / "provenance.json"
    if table_path.is_file() and metadata.is_file():
        from common import read_json
        provenance = read_json(metadata)
        if provenance["feature_table_sha256"] != sha256(table_path):
            raise ValueError("Cached catalog selection checksum mismatch")
        return pd.read_csv(table_path, dtype={"objid": str}, float_precision="round_trip"), provenance
    (author, members), (zoo, _) = [read_table(p) for p in files]
    table, audit = select(author, zoo)
    provenance = {**info, **audit, "fingerprint": stamp, "catalog_rows": len(author),
                  "gz1_rows": len(zoo), "author_members": members,
                  "author_headers": author.columns.tolist(), "gz_headers": zoo.columns.tolist()}
    write_csv(table_path, table)
    provenance["feature_table_sha256"] = sha256(table_path)
    write_json(metadata, provenance)
    return table, provenance
