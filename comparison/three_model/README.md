# Three-model Galaxy10 comparison

This directory contains the held-out comparison of Random Forest, Frozen MobileNetV2, and Fine-Tuned MobileNetV2. The two-model analysis remains in its original location. The fine-tuned ZIP and its 11 files are preserved in `Sagar/` and `Sagar/results/finetuned_mobilenetv2/`, respectively.

From the DOP project root, run:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dop-phase4-mpl \
  Kishore/.venv/bin/python comparison/three_model/src/three_model_analysis.py
python3 comparison/three_model/src/verify_outputs.py
```

The analysis verifies the shared HDF5 hash, locked split, exports, and fine-tuned ZIP. It updates the derived output files only. The scripts use the saved predictions and require no TensorFlow or model weights. The validation script independently checks the CSV calculations and provenance, including the previously verified HDF5 checksum as recorded in the dataset manifest.

- [Scientific report](THREE_MODEL_REPORT.md) describes the protocols, effect sizes, tests, and limitations.
- [Alignment audit](audit/alignment_audit.json) records the row, label, group, and source-artifact checks.
- [Provenance](audit/provenance.json) records input and output hashes, archive member paths, software versions, and seed.
- [Pairwise statistics](results/pairwise_statistical_tests.json) gives estimates, marginal intervals, exact tests, and Holm-adjusted McNemar p-values.
- `figures/` contains eight comparisons as 330-DPI PNG and PDF pairs.

A is Frozen − Random Forest, B is Fine-Tuned − Random Forest, and C is Fine-Tuned − Frozen. Percentage-point columns are 100 times the score differences.
