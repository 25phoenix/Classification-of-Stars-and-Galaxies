# Paired Galaxy10 comparison

This directory compares the frozen seven-feature Random Forest and frozen ImageNet MobileNetV2 on the same 3,545 Galaxy10 DECaLS test rows. The [comparison report](COMPARISON_REPORT.md) gives the findings and statistical assumptions.

From the project root, run with the existing Kishore virtual environment.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dop-phase3-mpl \
  Kishore/.venv/bin/python comparison/src/comparison_analysis.py
python3 comparison/src/verify_outputs.py
```

The analysis rereads and hashes the local HDF5 dataset, so runtime depends in part on disk throughput. It uses the saved prediction exports and writes computed tables, figures, and audit records under `comparison/`. TensorFlow, model fitting, and regenerated predictions are unnecessary for this run. Use the package versions in `audit/provenance.json` to reproduce the seeded intervals. The report is an interpretation of the outputs from these recorded inputs.

- `results/paired_predictions.csv` is the one-to-one test join by `row_id`, including the duplicate-object group and each prediction.
- `results/statistical_tests.json` contains the paired outcomes, exact tests, and 20,000-replicate group-bootstrap confidence intervals.
- `results/` also contains overall and per-class metrics, confusion matrices, and disagreement counts.
- `figures/` contains seven figures as 330-DPI PNGs and vector PDFs.
- `audit/` contains input fingerprints, environment versions, and alignment checks.

All differences are MobileNetV2 minus Random Forest. CSV score differences are fractions, and columns ending in `_percent` report relative changes in percent. The report expresses absolute score differences in percentage points.
