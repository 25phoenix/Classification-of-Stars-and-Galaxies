"""SDSS DR7 / Galaxy Zoo 1 catalog reconstruction and image feature diagnostics."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for name, value in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "MPLCONFIGDIR": str(ROOT / "cache/matplotlib"),
                    "JOBLIB_TEMP_FOLDER": str(ROOT / "cache/joblib")}.items():
    os.environ[name] = value


def prepared_measurement(path):
    import hashlib
    import math
    from dataclasses import asdict

    from common import read_json, sha256, write_json
    from image_features import Settings, extract_prepared

    stamp = {"image_sha256": sha256(path), "configuration": asdict(Settings()),
             "feature_code_sha256": sha256(ROOT / "image_features.py")}
    fingerprint = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    cached = ROOT / "cache/prepared" / f"{fingerprint}.json"
    if cached.is_file():
        return read_json(cached)
    values, flags = extract_prepared(path)
    response = {**stamp, "fingerprint": fingerprint,
                "features": {k: v if math.isfinite(v) else None for k, v in values.items()},
                "flags": flags,
                "method": "prepared scientific intermediates; image-to-catalog equivalence unvalidated"}
    write_json(cached, response)
    return response


def compare_images(table, pairs, out):
    import pandas as pd

    from common import FEATURES, write_csv

    fixtures = pd.read_csv(pairs, dtype={"objid": str, "path": str})
    if list(fixtures.columns) != ["objid", "path"] or not len(fixtures):
        raise ValueError("Expected a nonempty objid,path CSV manifest")
    source = table.set_index("objid")
    records = []
    for pair in fixtures.itertuples():
        raw = source.loc[pair.objid]
        if isinstance(raw, pd.DataFrame):
            raise ValueError(f"Multiple author rows for {pair.objid}; a unique object is required")
        row = {"objid": pair.objid, "path": pair.path, "catalog_error": raw.error}
        try:
            measured = prepared_measurement(pairs.parent / pair.path)
        except (OSError, ValueError, KeyError) as error:
            row["flags"] = str(error)
        else:
            row["flags"] = "|".join(measured["flags"])
            row["fingerprint"] = measured["fingerprint"]
            for key in FEATURES:
                value = measured["features"][key]
                row[f"catalog_{key}"] = raw[key]
                row[f"image_{key}"] = value
                row[f"delta_{key}"] = None if value is None else value - raw[key]
            for key in ("a", "s"):
                value = measured["features"][key]
                row[f"delta_if_catalog_stores_correlation_{key}"] = (
                    None if value is None else (1 - value) - raw[key])
        records.append(row)
    write_csv(out / "image_catalog_comparison.csv", pd.DataFrame(records))


def synthetic_table():
    import hashlib

    import numpy as np
    import pandas as pd

    from catalogs import select
    from common import FEATURES

    rng = np.random.default_rng(1234)
    labels = np.tile(np.arange(2), 126)
    size = len(labels)
    objid = pd.Series([str(587700000000000001+i) for i in range(size)])
    features = rng.normal(labels[:, None], 1, (size, 5))
    author = pd.DataFrame(features, columns=FEATURES).assign(
        objid=objid, k=np.repeat([6, 14, 24], size//3), error=0).astype(str)
    votes = pd.DataFrame(dict(objid=objid, elliptical=1-labels, spiral=labels, uncertain=0)).astype(str)
    author.loc[2, "objid"] = author.loc[0, "objid"]
    table, _ = select(author, votes)
    checksum = hashlib.sha256(table.to_csv(index=False).encode()).hexdigest()
    return table, {"mode": "synthetic fixture", "generator_seed": 1234, "rows": size,
                   "feature_table_sha256": checksum}


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report", "check", "smoke", "extract",
                                            "predict", "compare-images"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--threshold", type=int, choices=(5, 10, 20))
    parser.add_argument("--check-smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.check_smoke and args.command != "check":
        parser.error("--check-smoke requires the check command")
    if args.threshold is not None and args.command != "run":
        parser.error("--threshold is only supported for the real catalog run")
    smoke = args.command == "smoke" or args.check_smoke
    out = args.out or ROOT / "results" / ("smoke" if smoke else "catalog")
    out = out.resolve()
    if out == ROOT or not out.is_relative_to(ROOT):
        parser.error("--out must be a child directory of paper_replication")

    from common import FEATURES
    if args.command in ("extract", "predict"):
        if not args.image:
            parser.error("--image is required")
        response = prepared_measurement(args.image)
        if args.command == "predict":
            import joblib
            import pandas as pd
            if not args.model:
                parser.error("--model is required")
            bundle = joblib.load(args.model)
            if tuple(bundle["features"]) != FEATURES:
                raise ValueError("Incompatible model feature order")
            response["class"] = None
            if not response["flags"] and bundle.get("measurement_policy"):
                from followup import prediction
                response.update(prediction(bundle, response["features"]))
            elif not response["flags"]:
                model = bundle["model"]
                x = pd.DataFrame([response["features"]], columns=FEATURES)
                response["class"] = bundle["classes"][int(model.predict(x)[0])]
                if hasattr(model, "predict_proba"):
                    response["spiral_model_probability"] = float(model.predict_proba(x)[0, 1])
                else:
                    response["spiral_decision_score"] = float(model.decision_function(x)[0])
        print(json.dumps(response, indent=2, allow_nan=False))
        return
    from catalogs import prepare
    from evaluation import run, validate_saved
    table, provenance = synthetic_table() if smoke else prepare(ROOT / "data/raw", ROOT / "cache/catalogs")
    if not smoke and provenance["catalog_rows"] != 670_560:
        raise ValueError(f"Version 1 is documented with 670,560 rows, found {provenance['catalog_rows']:,}")
    if args.command == "prepare":
        print(json.dumps(provenance, indent=2))
    elif args.command == "compare-images":
        if not args.pairs:
            parser.error("--pairs is required")
        compare_images(table, args.pairs, out)
    elif args.command == "check":
        validate_saved(out, table)
        print("Verified group separation, OOF coverage, recomputed metrics and model predictions.")
    elif args.threshold is not None:
        out.mkdir(parents=True, exist_ok=True)
        run(table, provenance, out, args.seed, args.workers, thresholds=(args.threshold,))
    else:
        from reporting import report
        if args.command == "report":
            validate_saved(out, table)
        else:
            out.mkdir(parents=True, exist_ok=True)
            run(table, provenance, out, args.seed, args.workers, smoke)
        report(table, provenance, out, smoke)
        print(f"Report written to {out / 'REPORT.md'}")


if __name__ == "__main__":
    try:
        cli()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from None
