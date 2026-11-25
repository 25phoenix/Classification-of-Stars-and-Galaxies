# Galaxy10 DECaLS morphology classifier

The workflow measures seven morphology features and classifies the ten DECaLS
morphologies with a Random Forest. A majority-class baseline uses the same
partitions. [Measurement definitions and evaluation protocol](METHODS.md).

## WSL setup and full run

Run these commands in Ubuntu under WSL using Linux Python 3.10 or later.

```bash
cd /mnt/c/Acads/4th_year/1st_sem/DOP/Kishore
python3 -m venv .venv
PIP_CACHE_DIR="$PWD/cache/pip" .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py smoke
.venv/bin/python main.py all
.venv/bin/python main.py check
```

If Ubuntu lacks `ensurepip` and Linux `uv` is already available, initialize
the environment without pip and install directly into it.

```bash
python3 -m venv --without-pip .venv
UV_CACHE_DIR="$PWD/cache/uv" uv pip install --python .venv/bin/python -r requirements.txt
```

The synthetic smoke run uses eight trees per forest and writes to
`results/smoke`. Its output verifies the workflow on generated intensity
profiles. The real run uses every official row and the full search. It
requires a verified file of roughly 2.54 GB and can take substantial CPU time.
Use `--workers 2` on memory-constrained systems.

## Run stages individually

```bash
.venv/bin/python main.py download
.venv/bin/python main.py verify
.venv/bin/python main.py extract
.venv/bin/python main.py inspect
.venv/bin/python main.py run
.venv/bin/python main.py report
.venv/bin/python main.py check
```

`inspect` creates the dataset manifest, class mapping, and split indices.
Split construction requires measuring image validity and auditing all row
hashes, so a cold inspect invocation also builds the feature cache. Use a
consistent `--seed` (default 42) for inspect and run. The runner recalculates
the same checked split assignments before fitting. Commands with `--out`
accept directories inside Kishore. Use a fresh directory for an independent
run. If report generation fails after the metrics are saved, use `report`
with the same output directory to regenerate the plots and report. Run
`check` with the same output directory after an independent run.

## Inference

```bash
.venv/bin/python main.py predict /path/to/new-image.jpg
.venv/bin/python main.py predict /path/to/images --output results/predictions.json
.venv/bin/python main.py predict "../Classification-of-Stars-and-Galaxies/Cutout Files/galaxy" \
  --limit 6 --output results/final/local_unlabeled_predictions.json
```

JSON includes the predicted class, model probabilities, measured features and
quality flags. Invalid images get a null prediction. Use numeric class IDs with
the saved class mapping to interpret probability fields. Probability outputs
have no calibration guarantee. Only load joblib bundles from trusted sources.
Local cutouts have no verified subtype labels, so accuracy cannot be measured
and their image distribution may differ from the supervised dataset.

## Files

| Path | Contents |
| --- | --- |
| `acquisition.py` and `data_io.py` | Verified download, schema, duplicate audit, and feature table |
| `morphology.py` and `predictor.py` | Shared extraction and model inference |
| `experiment.py` and `reporting.py` | Nested evaluation, persisted predictions, plots, and report |
| `main.py` | Workflow CLI |
| `tests/` | Acquisition, features, splits, serialization, and metrics checks |
| `data/raw/Galaxy10_DECals.h5` | Original release after checksum verification |
| `cache/<fingerprint>/features.csv` | All rows, features, group IDs, and quality flags |
| `results/final/dataset.json` | Checksum, schema, audit, and exclusion counts |
| `results/final/splits.json` | Holdout, outer, inner, and final CV row IDs |
| `results/final/split_assignments.csv` | Row-wise assignment, including exclusions |
| `results/final/metrics.json` and `folds.json` | Metrics and timings |
| `results/final/oof_predictions.csv` and `test_predictions.csv` | Row-wise predictions and probabilities |
| `results/final/search_*.csv` | Candidate results by search |
| `results/final/model.joblib` and `model_metadata.json` | Development-fit pipeline and provenance |
| `results/final/environment.txt` and `runtime.json` | Exact installed package versions and runtime information |
| `results/final/REPORT.md` and `plots/` | Measured results and figures |
| `results/final/local_unlabeled_predictions.json` | Full runner's inference demo when local cutouts are present |

## Troubleshooting

- For DNS errors, verify that WSL can resolve both `pypi.org` and `zenodo.org`.
  With no network, provide Linux wheels in a Kishore-local wheelhouse and
  the exact release file in `data/raw/`. Run `verify` to confirm the checksum.
- The downloader resumes `.h5.part` when ranges are supported and promotes
  it only after SHA-256 verification. If a checksum mismatch persists, move
  the suspect file aside within Kishore and retry `download`.
- An existing verified dataset is reused. A completed evaluation is kept
  intact when you specify its directory again. Use `--out results/new-run`
  for a deliberate new evaluation.
- After an interruption during extraction or outer CV, rerun the same command
  with the same output directory and seed. Completed feature batches and
  matching outer folds are reused. If evaluation completed before report
  generation failed, use the `report` command for that run.
- The feature fingerprint changes with the data checksum, extraction settings,
  extraction code, and numerical library versions. Model loading checks the
  feature implementation and class mapping. Use the original feature code
  when loading an existing model.
