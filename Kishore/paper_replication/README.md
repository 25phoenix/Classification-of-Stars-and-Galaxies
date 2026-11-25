# Barchi et al. catalog reconstruction

This project evaluates the traditional binary Galaxy Zoo 1 task on the
authors' released CyMorph measurements. It uses a new nested split protocol.
The separate image feature diagnostic requires scientific preprocessing
intermediates. [Paper specification and methodological limits](SPECIFICATION.md).

## Run in Ubuntu WSL

The existing project Linux environment already contains the packages in
`requirements.txt`. All new outputs live below this directory.

```bash
cd /mnt/c/Acads/4th_year/1st_sem/DOP/Kishore/paper_replication
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python main.py smoke --workers 1
```

For an isolated fresh environment with network access, use
`uv venv --python /usr/bin/python3 .venv` followed by
`UV_CACHE_DIR="$PWD/cache/uv" uv pip install --python .venv/bin/python -r requirements.txt`
and substitute `.venv/bin/python` below.

## Stage the two public catalogs

In an ordinary Ubuntu WSL terminal with internet access, the acquisition
helper requests both released archives, handles the optional Mendeley outer
ZIP, checks CRCs, and saves the original catalog containers.

```bash
python3 acquire.py
```

For manual acquisition, save `Barchi19_Morph-catalog_670k-galaxies.zip` from the [version 1 release](https://data.mendeley.com/datasets/tg7c985c2n/1)
under `data/raw/`. The publisher lists a 45 MB ZIP. Keep the original ZIP.
Download the independent GZ1 class flags in ordinary WSL.

```bash
mkdir -p data/raw
curl -fL --retry 3 --connect-timeout 15 -C - \
  https://galaxy-zoo-1.s3.amazonaws.com/GalaxyZoo1_DR_table2.csv.gz \
  -o data/raw/GalaxyZoo1_DR_table2.csv.gz.part &&
mv data/raw/GalaxyZoo1_DR_table2.csv.gz.part data/raw/GalaxyZoo1_DR_table2.csv.gz
```

Skip acquisition if both complete files are already in place. The loader
records SHA-256, checks the schemas, and preserves exact integer object
identities during the join. Author predictions never supply labels.

```bash
../.venv/bin/python main.py prepare
../.venv/bin/python main.py run --workers 4
../.venv/bin/python main.py check
```

The default results path is `results/catalog/`. `run` persists a common
object-level holdout and outer assignments across all K cuts. It commits
each fold and final model with hashes and resumes only under the same data,
code, and configuration. Each threshold uses all eligible matched rows. The
SVM uses a fold-local 128-landmark Nyström RBF map followed by a primal
linear SVM on all training rows. Use `../.venv/bin/python main.py report`
to regenerate plots and the report after a completed fit.
To distribute the work, run `main.py run --threshold 5`, `--threshold 10`,
and `--threshold 20` separately. Different thresholds may run in parallel,
using the same locked split initialization. Run `main.py report` and
`main.py check` after all three commands have finished.
The catalog cache and sample manifest retain the selected and excluded
rows. `REPORT.md`, `plots/`, and each `k*/MODEL/` contain figures,
searches, model bundles, predictions, metrics, and per-class results.

## Scientific feature extraction and inference

```bash
../.venv/bin/python main.py extract --image data/prepared/object.npz
../.venv/bin/python main.py predict --image data/prepared/object.npz \
  --model results/catalog/k20/mlp/mlp_final_model.joblib
```

Only load trusted joblib models. Images must be scientific SDSS-DR7 r-band
FITS-derived intermediates packaged as a NumPy NPZ with scalar string entries
`survey=SDSS-DR7` and `band=r`. All 2-D arrays must be aligned on the same original
pixel grid.

| NPZ field | Meaning |
| --- | --- |
| `clean_g`, `mask_g` | Isophote-cleaned input and valid G2 mask |
| `raw_h`, `mask_h` | Original scientific intensities and the H segmentation |
| `enhanced_a`, `mask_a` | A gradient-enhanced input and A segmentation |
| `sky_subtracted_s`, `filter_input_s`, `mask_s` | S rank input, Butterworth input, and S segmentation |
| `cumulative_flux` | Clean cumulative aperture flux sampled at integer radii, through beyond 2 Rp |
| `rp_pixels`, `fwhm_pixels`, `max_rad_pixels` | Positive Petrosian radius, source FWHM, and ellipse radius in pixels |
| `cx_pixels`, `cy_pixels` | Zero-based G2 center on the supplied pixel grid |

Masks use 1 for valid source pixels. Undefined measurements produce flags
and abstention. This diagnostic follows the paper's rank distance convention.
The public source exposes raw correlation outputs as well, and prepared image
feature parity with the published catalog remains unverified. Prediction
results from this adapter are provisional until a dataset of paired author
measurements and prepared scientific stamps is available. The SVM emits a
decision score. Other estimators expose uncalibrated model probabilities.

To audit a scientific subset, prepare a CSV with columns `objid,path`,
where each path is relative to the manifest and refers to a prepared NPZ.
Use a subset defined without consulting held-out labels or performance.

```bash
../.venv/bin/python main.py compare-images --pairs data/prepared/pairs.csv
```

The comparison CSV records each input and its feature deltas against the
same DR7 catalog row. It includes A and S deltas for both the paper's
distance and the wrapper's correlation conventions. Extraction caches are
keyed by input bytes, feature code, and explicit settings.

## Complete-case follow-up on frozen assignments

The original as-released run remains in `results/catalog/`. The follow-up
filters the previously assigned rows and keeps every surviving holdout,
outer, inner and final-tuning assignment. It does not change the four
estimators or their search grids. The holdout was previously inspected.

```bash
cd /mnt/c/Acads/4th_year/1st_sem/DOP/Kishore/paper_replication
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python followup.py smoke --workers 1
../.venv/bin/python followup.py run --threshold 5 --workers 3
../.venv/bin/python followup.py run --threshold 10 --workers 3
../.venv/bin/python followup.py run --threshold 20 --workers 3
../.venv/bin/python followup.py pilot
../.venv/bin/python followup.py report
../.venv/bin/python followup.py check
```

The three threshold commands may run concurrently. Their initialization is
locked and each threshold has independent checkpoints. Results live in
`results/catalog_complete_case/`. The corrected selection cache is keyed by
input and baseline split hashes, policy, optional metadata, and correction
code. Checkpoints also bind the code, estimators, environment, and seed.
The reference artifacts are read without modification.

Complete-case prediction requires all five valid catalog measurements.
A sentinel, nonfinite value, out-of-domain value, or nonzero author Error
results in abstention. Set `--error` if the original quality flag is available.

```bash
../.venv/bin/python followup.py predict \
  --model results/catalog_complete_case/k20/dt/dt_final_model.joblib \
  --values '{"c":0.4,"a":0.7,"s":0.8,"g2":1.4,"h":0.65}' --error 0
```

For the separate metadata variant, submit the exact query in
`dr7_metadata.sql` to SDSS CasJobs under DR7 context. Export
`MyDB.barchi_dr7` as plain CSV to `data/raw/dr7_metadata.csv`. Preserve the object
IDs as decimal strings by exporting directly rather than opening and saving
the file in a spreadsheet. The loader caches input by hash, rejects conflicting
same-ID measurements, and records coverage, missing metadata, and both strict
cuts. The staged export contains conflicting measurements. The validator
records those rows and excludes every entry for a conflicting exact ID.
It leaves the staged source unchanged. Use its derived unambiguous table
for the workflow. Results go to `results/catalog_dr7_complete_case/`.
[SDSS CasJobs guide](https://preprod.skyserver.sdss.org/casjobs/Guide.aspx) ·
[SDSS DR7 primary spectra and photometry](https://classic.sdss.org/dr7/help/faq.php).

```bash
cd /mnt/c/Acads/4th_Year/1st_sem/DOP/Kishore/paper_replication
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PWD/cache"
out=results/catalog_dr7_complete_case
../.venv/bin/python validate_dr7.py data/raw/dr7_metadata.csv
metadata="$out/unambiguous_dr7_metadata.csv"
for k in 5 10 20; do
  ../.venv/bin/python followup.py run --threshold "$k" --workers 2 --metadata "$metadata" || exit 1
done
../.venv/bin/python followup.py report --metadata "$metadata"
../.venv/bin/python followup.py check --metadata "$metadata"
(
  set -e
  audit_tmp="$(mktemp "$out/verification.XXXXXX.part")"
  trap 'rm -f "$audit_tmp"' EXIT
  ../.venv/bin/python audit_followup.py --out "$out" > "$audit_tmp"
  mv "$audit_tmp" "$out/verification.json"
)
../.venv/bin/python audit_dr7.py
```

[DR7 report](results/catalog_dr7_complete_case/REPORT.md) ·
[Export validation](results/catalog_dr7_complete_case/metadata_export_audit.json) ·
[Independent fold audit](results/catalog_dr7_complete_case/verification.json) ·
[Paired comparison](results/catalog_dr7_complete_case/dr7_complete_case_comparison.json).

The exact RBF SVM sensitivity run uses the same corrected inputs and folds.
Resume from the existing checkpoints in WSL with the following commands.
The run command verifies checkpoint fingerprints and output hashes before
reuse. Keep one writer per threshold. Run the thresholds sequentially if RAM
is limited. The worker counts below fit the tested 7.6 GiB WSL environment
with BLAS threads limited to one.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PWD/cache"
../.venv/bin/python exact_svm.py run --threshold 10 --workers 3
../.venv/bin/python exact_svm.py run --threshold 5 --workers 6
../.venv/bin/python exact_svm.py report
../.venv/bin/python audit_exact_svm.py
../.venv/bin/python followup.py check
(
  set -e
  audit_tmp="$(mktemp results/catalog_complete_case/verification.XXXXXX.part)"
  trap 'rm -f "$audit_tmp"' EXIT
  ../.venv/bin/python audit_followup.py > "$audit_tmp"
  mv "$audit_tmp" results/catalog_complete_case/verification.json
)
sha256sum -c cache/followup_original.sha256 --quiet
```

For a fresh sensitivity run, also execute
`../.venv/bin/python exact_svm.py run --threshold 20 --workers 2`
before generating the report. Exact SVM outputs, paired comparison and
verification go to `results/catalog_complete_case_exact_svm/`. The exact
kernel uses hinge loss and the Nyström pipeline uses squared-hinge loss.
Their measured difference includes the solver objective. The paper did not
specify its SVM kernel. Both use the previously inspected holdout.

[Exact-SVM report and all outer folds](results/catalog_complete_case_exact_svm/REPORT.md) ·
[Paired comparison](results/catalog_complete_case_exact_svm/PAIRED_REPORT.md) ·
[Exact-SVM verification](results/catalog_complete_case_exact_svm/verification.json) ·
[Primary complete-case report](results/catalog_complete_case/REPORT.md).
