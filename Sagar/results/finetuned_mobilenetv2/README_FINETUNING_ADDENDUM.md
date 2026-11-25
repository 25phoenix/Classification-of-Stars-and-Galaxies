# Continuation Addendum: Two-Stage Fine-Tuned MobileNetV2 (Galaxy10 DECaLS)

This archive contains **only** the new end-to-end fine-tuning experiment completed after the Frozen MobileNetV2 Baseline. All earlier frozen-baseline results remain unchanged (`0.5633` Test Acc / `0.5224` Macro-F1).

## Protocol Summary
- **Same leak-free contract:** Exact same `17,733` eligible Galaxy10 DECaLS rows (`14,188` dev / `3,545` locked test; `60` duplicate groups isolated) and identical `224x224` `mobilenet_v2.preprocess_input`.
- **Training-only augmentation:** Random H/V flip, rotation (±25%, reflect), translation (±5%, reflect). Inactive at test time.
- **Two-stage schedule (RTX 3050 4GB, `mixed_float16`, `batch_size=16`, `dropout=0.3`, `l2=1e-4`, `use_class_weight=False`):**
  1. **Warmup:** `5` epochs, frozen backbone, `lr=0.001`.
  2. **Fine-Tuning:** Top `40` layers targeted (`27` non-BN conv layers unfrozen; **all BatchNorm layers frozen** with `training=False`), `lr=0.0001`.
- **Epoch selection & refit:** Validated strictly on Dev Fold 0 (`final_tuning_folds[0]`, best at Epoch `15`, `val_acc=0.7594`, `val_loss=0.7079`), then refit from scratch on all `14,188` dev rows (cosine LR decay) and evaluated once on the `3,545` test rows.
- **Validation note:** Unlike the 5x3 nested-CV frozen baseline, this GPU run used Dev Fold 0 for epoch selection, so 3-way comparison (RF vs. Frozen vs. Fine-Tuned) aligns strictly on the `3,545` held-out test `row_id`s.

## Measured Results (Held-Out Test, n = 3,545)
- **Test Accuracy:** `0.7622` (up from `0.5633` frozen, `+0.1989`)
- **Test Macro-F1:** `0.7342` (up from `0.5224` frozen, `+0.2118`)
- **Timings:** `34.8s` NVMe image staging | `1777.3s` (~29.6 min) GPU fine-tuning | `8.85s` GPU test inference (`3,545` raw images)

## Per-Class Held-Out Test Metrics (Fine-Tuned)
| ID | Class | Support | Precision | Recall | F1 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | Disturbed Galaxies | 217 | 0.4600 | 0.3180 | 0.3760 |
| 1 | Merging Galaxies | 370 | 0.8222 | 0.8000 | 0.8110 |
| 2 | Round Smooth Galaxies | 529 | 0.7915 | 0.9471 | 0.8623 |
| 3 | In-between Round Smooth Galaxies | 406 | 0.7704 | 0.8596 | 0.8126 |
| 4 | Cigar Shaped Smooth Galaxies | 65 | 0.6418 | 0.6615 | 0.6515 |
| 5 | Barred Spiral Galaxies | 409 | 0.7685 | 0.7628 | 0.7656 |
| 6 | Unbarred Tight Spiral Galaxies | 365 | 0.7381 | 0.5945 | 0.6586 |
| 7 | Unbarred Loose Spiral Galaxies | 525 | 0.6318 | 0.5981 | 0.6145 |
| 8 | Edge-on Galaxies without Bulge | 284 | 0.8982 | 0.9014 | 0.8998 |
| 9 | Edge-on Galaxies with Bulge | 375 | 0.8625 | 0.9200 | 0.8903 |

## Caveats
- Still 10 galaxy classes only (no verified star labels; uncalibrated closed-set softmax).
- Verified in-distribution on DECaLS only (cross-survey transfer not tested).
- Validation loss was still edging down at Epoch 15 (`val_loss=0.7079`), so longer training remains future work.

## Files Included
- `data/` — Prefixed CSV/JSON/MD metrics, timings, and `finetuned_test_predictions.csv` (`row_id`-aligned with earlier test predictions).
- `plots/` — Ready-to-embed slide figures (`finetuned_learning_curves.png`, `finetuned_test_confusion_matrices.png`, `finetuned_prediction_gallery.png`).
