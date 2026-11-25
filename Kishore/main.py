"""Galaxy10 workflow command line."""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for key, value in {
    "MPLCONFIGDIR": str(ROOT / "cache/matplotlib"),
    "JOBLIB_TEMP_FOLDER": str(ROOT / "cache/joblib"),
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
}.items():
    os.environ[key] = value


def child(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("Output must be inside Kishore")
    return path


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("download", "verify"):
        commands.add_parser(name)
    for name in ("extract", "inspect", "run", "all", "smoke", "report", "check"):
        command = commands.add_parser(name)
        command.add_argument("--out", default=str(ROOT / "results" / (
            "smoke" if name == "smoke" else "final")))
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    predict = commands.add_parser("predict")
    predict.add_argument("image", type=Path)
    predict.add_argument("--model", type=Path, default=ROOT / "results/final/model.joblib")
    predict.add_argument("--output", type=Path)
    predict.add_argument("--limit", type=int)
    return cli


def smoke_data(path):
    import h5py
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1729)
    y, x = np.indices((64, 64))
    images = []
    labels = np.repeat(np.arange(10), 30)
    for label in labels:
        cx, cy = rng.uniform(30, 34, 2)
        width = 3 + label * .65
        light = .7 * np.exp(-.5 * (((x - cx) / width) ** 2 + ((y - cy) / 5) ** 2))
        image = np.uint8(np.clip((.07 + light + rng.normal(0, .012, light.shape)) * 255, 0, 255))
        images.append(np.repeat(image[..., None], 3, axis=-1))
    images[1] = images[0].copy()
    ra = np.linspace(1, 359, len(labels))
    ra[2] = ra[1] + 0.00001
    with h5py.File(path, "w") as file:
        file["images"] = images
        file["ans"] = labels
        file["ra"] = ra
        file["dec"] = np.zeros(len(labels))


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        from acquisition import DATA, download, verify
        command = args.command
        if command in ("download", "verify"):
            (download if command == "download" else verify)()
            print(f"Verified {DATA}")
            return
        if command == "all":
            download()
        from data_io import CACHE, extract_table, write_json
        if command == "predict":
            from predictor import load_model, predict_path
            if args.limit is not None and args.limit < 1:
                cli.error("--limit must be positive")
            model = load_model(args.model)
            rows = list(predict_path(args.image, model, args.limit))
            output = {"model": str(args.model), "mode": model["mode"],
                      "note": "Probabilities are model outputs. Labeled truth is required to measure accuracy.",
                      "predictions": rows}
            if args.output:
                write_json(child(args.output), output)
            else:
                print(json.dumps(output, indent=2, allow_nan=False))
            return
        from experiment import fit_experiment, persist_splits, validate_predictions, verify_plan
        if args.workers < 1:
            cli.error("--workers must be positive")
        out = child(args.out)
        smoke = command == "smoke" or (
            command in ("report", "check") and (out / "synthetic.h5").is_file())
        path = out / "synthetic.h5" if smoke else DATA
        if command in ("run", "all", "smoke", "inspect") and (out / "metrics.json").is_file():
            raise FileExistsError(f"Evaluation already exists in {out}. Choose a new run directory.")
        if command == "smoke":
            if out == ROOT / "results/final":
                raise ValueError("Use a separate synthetic smoke output directory")
            smoke_data(path)
        table, provenance = extract_table(path, out / "cache" if smoke else CACHE,
                                          args.workers, official=not smoke)
        if command == "extract":
            print(json.dumps(provenance, indent=2))
            return
        if command == "inspect":
            persist_splits(table, provenance, out, args.seed)
            write_json(out / "dataset.json", provenance)
            write_json(out / "class_mapping.json", provenance["class_mapping"])
            print(json.dumps(provenance, indent=2))
            return
        if command == "check":
            plan = json.loads((out / "splits.json").read_text())
            if plan["feature_fingerprint"] != provenance["fingerprint"]:
                raise ValueError("Run split manifest uses a different feature cache fingerprint")
            verify_plan(table, plan)
            validate_predictions(out, table)
            from predictor import load_model
            from acquisition import digest
            bundle = load_model(out / "model.joblib")
            if bundle["split_manifest_sha256"] != digest(out / "splits.json"):
                raise ValueError("Saved model uses a different split manifest")
            print(f"Verified splits, prediction coverage, metrics, and model in {out}")
            return
        if command in ("run", "all", "smoke"):
            fit_experiment(table, provenance, out, args.seed, args.workers, smoke)
        if command == "all":
            from predictor import load_model, predict_path
            local = ROOT.parent / "Classification-of-Stars-and-Galaxies/Cutout Files/galaxy"
            if local.is_dir():
                rows = list(predict_path(local, load_model(out / "model.joblib"), 6))
                write_json(out / "local_unlabeled_predictions.json", {
                    "note": "Unlabeled inference demonstration. Accuracy is unknown. "
                            "These images may differ from the Galaxy10 DECaLS training distribution.",
                    "predictions": rows,
                })
        from reporting import render
        render(table, path, out)
        print(f"Report written to {out / 'REPORT.md'}")
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as error:
        cli.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
