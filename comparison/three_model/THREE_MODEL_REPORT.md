# Three-model comparison on Galaxy10 DECaLS

## Executive summary

Fine-Tuned MobileNetV2 correctly classified **2,702 of 3,545 held-out galaxies (76.22%)**, compared with **1,997 (56.33%)** for Frozen MobileNetV2 and **1,959 (55.26%)** for the seven-feature Random Forest. Its Macro-F1 was **0.7342**, versus **0.5224** and **0.5178**, respectively. The gains against both baselines were supported by group-aware confidence intervals and paired accuracy tests. The two frozen baselines performed similarly overall.

## Experimental methodology and audit

The official Galaxy10 DECaLS HDF5 contains 17,736 images, of which 17,733 are eligible under the shared contract. The split contains 14,188 development and 3,545 locked test rows across ten classes. The analysis rehashed the HDF5 and all three contract files and checked them against frozen-baseline provenance. It joined all three original prediction exports one-to-one by `row_id`. True labels and class order agreed with the saved contract, the HDF5 labels, and the supplied class metadata. All rows had valid class probabilities, unit sums within tolerance, and predicted labels matching the probability argmax.

The 3,545 test rows contain 3,535 duplicate-object groups, including ten groups of two. The shared rule links identical decoded RGB images or sky positions within one arcsecond, then takes connected components. No development group intersects the held-out groups. The first final-tuning split also has disjoint train and validation groups. Inspect [alignment checks](audit/alignment_audit.json) and [SHA-256 provenance](audit/provenance.json).

| Model | Representation and classifier | Development selection and refit |
|---|---|---|
| Random Forest | Concentration, asymmetry, smoothness, entropy, Gini, M20 and ellipticity from rendered cutouts. 160-tree classifier with balanced subsample class weights. | Five outer folds with three inner folds, then final development tuning and refit. |
| Frozen MobileNetV2 | ImageNet backbone at 224 × 224, MobileNetV2 normalization to [−1, 1], global average pooling to 1,280 features, dropout 0.2 and ten-class softmax head. | Five outer folds with three inner folds, then final development tuning and 18-epoch head refit. Backbone stayed frozen. |
| Fine-Tuned MobileNetV2 | Same backbone family, input size, normalization and pooling as the frozen model, with dropout 0.3 and ten-class softmax. Uses two-stage training and training-only flips, rotations and translations. | One pre-existing development validation fold for epoch selection, followed by a fresh full-development refit. |

The fine-tuned archive reports mixed float16, batch size 16, no class weighting, five frozen warmup epochs with Adam at 0.001, then partial unfreezing at 0.0001. Of the top 40 layers targeted, 27 non-BatchNormalization layers were unfrozen. BatchNormalization was kept frozen. The addendum specifies horizontal and vertical flips, rotations up to ±90°, and ±5% spatial translations with reflection. Epoch selection used development fold index 0 (11,349 training and 2,839 validation rows), with a reported best validation loss of 0.7079 and accuracy of 75.94% at epoch 15. The reported final refit used all 14,188 development rows, five warmup epochs, 15 fine-tuning epochs, and cosine learning-rate decay. The supplied JSON records approximately 1,777.3 seconds for validation and final refit combined, and 8.85 seconds for test inference.

The ZIP passed CRC and exact-byte verification for all 11 extracted members. Its SHA-256 is `7cadaad9d62925c73339cff75933b30cd5c20cdac63bfc59fa68c52853c92812`. It was copied from Windows Downloads to `Sagar/Sagar_FineTuned_Only_For_Kishore.zip`, and its members were installed under `Sagar/results/finetuned_mobilenetv2/` with original filenames. The addendum and configuration provide the training record for this run. The local optional fine-tuning script implements a different schedule and weighting policy. The test membership is verified directly. The fine-tuned archive has no independent dataset or split hashes, specific executed training script, or weights to independently verify the recorded training procedure.

## Overall held-out performance

A = Frozen − Random Forest, B = Fine-Tuned − Random Forest, and C = Fine-Tuned − Frozen. Differences are in percentage points (pp) and use unrounded scores.

| Metric | Random Forest | Frozen MobileNetV2 | Fine-Tuned MobileNetV2 | A Δ | B Δ | C Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 55.26% | 56.33% | 76.22% | +1.07 | +20.96 | +19.89 |
| Macro-F1 | 51.78% | 52.24% | 73.42% | +0.45 | +21.64 | +21.19 |
| Weighted-F1 | 54.57% | 55.11% | 75.48% | +0.54 | +20.91 | +20.37 |
| Balanced accuracy | 52.62% | 52.43% | 73.63% | -0.19 | +21.01 | +21.20 |
| Macro-precision | 51.84% | 54.49% | 73.85% | +2.65 | +22.01 | +19.36 |
| Macro-recall | 52.62% | 52.43% | 73.63% | -0.19 | +21.01 | +21.20 |

Balanced accuracy equals macro-recall because all ten classes are present. The fine-tuned accuracy and Macro-F1 recompute to 0.7622002821 and 0.7342220992, matching the archived values. The large gains in Macro-F1 and balanced accuracy indicate that the overall gain extends beyond performance on the more prevalent classes.

## Class-level performance

Values within each cell follow the order Random Forest / Frozen / Fine-Tuned.

| Galaxy10 class | Support | F1 | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| Disturbed Galaxies | 217 | 25.00% / 15.66% / 37.60% | 21.20% / 10.14% / 31.80% | 30.46% / 34.38% / 46.00% |
| Merging Galaxies | 370 | 57.39% / 49.73% / 81.10% | 71.89% / 48.92% / 80.00% | 47.76% / 50.56% / 82.22% |
| Round Smooth Galaxies | 529 | 70.15% / 67.63% / 86.23% | 72.40% / 70.70% / 94.71% | 68.03% / 64.82% / 79.15% |
| In-between Round Smooth Galaxies | 406 | 60.82% / 55.64% / 81.26% | 60.59% / 60.10% / 85.96% | 61.04% / 51.80% / 77.04% |
| Cigar Shaped Smooth Galaxies | 65 | 30.22% / 34.62% / 65.15% | 32.31% / 27.69% / 66.15% | 28.38% / 46.15% / 64.18% |
| Barred Spiral Galaxies | 409 | 45.65% / 45.52% / 76.56% | 41.08% / 37.90% / 76.28% | 51.38% / 56.99% / 76.85% |
| Unbarred Tight Spiral Galaxies | 365 | 44.17% / 52.11% / 65.86% | 44.11% / 57.53% / 59.45% | 44.23% / 47.62% / 73.81% |
| Unbarred Loose Spiral Galaxies | 525 | 41.03% / 46.84% / 61.45% | 37.90% / 50.86% / 59.81% | 44.72% / 43.41% / 63.18% |
| Edge-on Galaxies without Bulge | 284 | 77.83% / 79.86% / 89.98% | 80.99% / 83.10% / 90.14% | 74.92% / 76.87% / 89.82% |
| Edge-on Galaxies with Bulge | 375 | 65.57% / 74.74% / 89.03% | 63.73% / 77.33% / 92.00% | 67.51% / 72.32% / 86.25% |

F1 increased after fine-tuning for all ten classes relative to both baselines. Compared with Frozen MobileNetV2, the largest gains were for **Merging Galaxies**, **Barred Spiral Galaxies**, and **Cigar Shaped Smooth Galaxies**. Both edge-on classes reached F1 of about 0.89 or higher, and Round Smooth Galaxies reached 94.71% recall. The cigar-shaped class has only 65 test examples, so its large point gain is less precisely characterized. Disturbed Galaxies remain the weakest class despite improvement. Fine-tuned F1 is 37.60%, with 31.80% recall and 46.00% precision. Its recall and precision exceed both baselines. Class-level comparisons are descriptive.

The [per-class table](results/per_class_comparison.csv) contains exact precision, recall, F1, and support for every model and all three pairwise deltas.

## Paired outcomes and statistical inference

| Pair | Both correct | First only | Second only | Both wrong | Label agreement | Label disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A. Frozen − Random Forest | 1,334 | 625 | 663 | 923 | 1,689 (47.64%) | 1,856 |
| B. Fine-Tuned − Random Forest | 1,701 | 258 | 1,001 | 585 | 1,942 (54.78%) | 1,603 |
| C. Fine-Tuned − Frozen | 1,806 | 191 | 896 | 652 | 2,170 (61.21%) | 1,375 |

Agreement means the predicted labels are identical, including matched wrong labels. The largest absolute class F1 differences per pair were:

- **A.** Disturbed Galaxies (-9.34 pp), Edge-on Galaxies with Bulge (+9.17 pp), Unbarred Tight Spiral Galaxies (+7.94 pp).
- **B.** Cigar Shaped Smooth Galaxies (+34.94 pp), Barred Spiral Galaxies (+30.91 pp), Merging Galaxies (+23.71 pp).
- **C.** Merging Galaxies (+31.37 pp), Barred Spiral Galaxies (+31.04 pp), Cigar Shaped Smooth Galaxies (+30.54 pp).

The group bootstrap samples 3,535 groups with replacement per replicate and includes every row of each drawn group, with multiplicity. It uses 20,000 replicates per pair, PCG64 seed `20261124`, row-weighted accuracy, and equal weighting of the ten class F1 scores. Each pair starts at the same seed. The reported 95% percentile intervals are marginal. They condition on the fitted models and assume independent sampling units at the group level.

| Pair | Accuracy Δ [95% CI], pp | Macro-F1 Δ [95% CI], pp | McNemar p | Holm p | Exact group-swap p |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | +1.07 [-0.90, +3.02] | +0.45 [-1.87, +2.75] | 0.302555 | 0.302555 | 0.301803 |
| B | +20.96 [+19.17, +22.80] | +21.64 [+19.54, +23.79] | 1.5068e-103 | 3.0136e-103 | 6.3865e-104 |
| C | +19.89 [+18.17, +21.62] | +21.19 [+19.05, +23.36] | 1.3290e-109 | 3.9871e-109 | 7.6983e-110 |

Exact row-level McNemar conditions on discordant correctness pairs and assumes independent row pairs with equally likely directions of discordance under the null. Duplicate-object dependence can affect the row-level p-values. The Holm adjustment applies only to the three two-sided McNemar accuracy comparisons. The separate sensitivity test swaps model labels jointly within each object group under within-group exchangeability and independence across groups. Exact convolution of group sign allocations gives the two-sided p-value for the absolute net difference in correct counts. Sensitivity p-values are unadjusted.

Comparison A exactly reproduced the earlier run’s McNemar and group-swap p-values and both bootstrap interval endpoints. Comparisons B and C retain strong evidence of higher fine-tuned accuracy after adjustment, with group-bootstrap intervals entirely above zero for both accuracy and Macro-F1. The frozen-versus-forest intervals include zero. These are conditional in-distribution comparisons and provide no causal identification or universal ranking.

## Confusion patterns

The largest reduction in error counts was Round Smooth Galaxies predicted as In-between Round Smooth Galaxies (71 → 3 from frozen to fine-tuned). Barred spirals mistaken for loose or tight unbarred spirals fell from 102 → 41 and 80 → 20. Loose spirals mistaken for tight spirals fell from 83 → 37. Edge-on galaxies with bulge mistaken for edge-on galaxies without bulge fell from 34 → 6. The change was uneven. Tight spirals mistaken for loose spirals remained nearly unchanged (74 → 73), and 48 loose spirals were predicted as disturbed after fine-tuning. Counts describe directional errors against the true labels. The [confusion-change table](results/confusion_changes_after_finetuning.csv) includes increases as well as reductions.

## Scientific interpretation

The measured gains are consistent with the value of adapting convolutional features to the galaxy morphology task. The frozen network uses natural-image filters and compresses spatial information into one globally pooled vector. Handcrafted features capture physically interpretable shape and light-distribution cues, so comparable held-out performance from these two representations is plausible. Fine-tuning may help the backbone retain distinctions in roundness, bars, and bulges. Training-only geometric augmentation may help the model learn visual cues that remain informative under changes in orientation and position. Augmentation, partial unfreezing, head settings, and the training schedule changed together. The study cannot isolate their individual effects without ablations.

## Limitations

- Fine-tuning selected its duration using one development validation fold. Both baselines used nested cross-validation. No fine-tuned development OOF predictions were supplied, so all three-way comparisons use the locked test set.
- The fine-tuned full-model weights and exact executed training script were absent. Scores and statistical comparisons are reproducible from the saved outputs. Weight-level inference and reproduction of the reported training run remain unverified.
- Galaxy10 DECaLS contains ten galaxy morphology classes. This evaluation covers neither verified stellar labels nor cross-survey generalization. The separate Barchi/DR7 experiment was excluded.
- The paired procedures condition on fitted models and rely on independence across object groups. They quantify uncertainty on this holdout under stated assumptions and exclude variation from retraining. The test set remains locked against further model selection.

## Conclusion

The seven-feature Random Forest and frozen ImageNet MobileNetV2 achieve similar accuracy and Macro-F1 on the shared holdout. The reported fine-tuned pipeline substantially outperforms both across overall and class-balanced metrics. This supports the usefulness of the complete adaptation pipeline for Galaxy10 DECaLS, while leaving the contribution of each training change open.

## Reproducibility

From the project root, run:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dop-phase4-mpl \
  Kishore/.venv/bin/python comparison/three_model/src/three_model_analysis.py
python3 comparison/three_model/src/verify_outputs.py
```

The independent verifier passed checks for sample counts, class mapping, source predictions, metrics, confusion matrices, exact tests, Holm adjustment, confidence interval direction, provenance, and figure exports. It recomputes metrics and p-values independently and checks the stored bootstrap conventions and Phase 3 interval match. Only the analysis recomputes the complete group-bootstrap draws. The analysis and verification loaded no model and performed no training.

### Output inventory

- [README.md](README.md)
- `THREE_MODEL_REPORT.md`
- [audit/alignment_audit.json](audit/alignment_audit.json)
- [audit/provenance.json](audit/provenance.json)
- [audit/verification.txt](audit/verification.txt)
- [figures/confusion_matrices.pdf](figures/confusion_matrices.pdf)
- [figures/confusion_matrices.png](figures/confusion_matrices.png)
- [figures/overall_metrics.pdf](figures/overall_metrics.pdf)
- [figures/overall_metrics.png](figures/overall_metrics.png)
- [figures/paired_accuracy_intervals.pdf](figures/paired_accuracy_intervals.pdf)
- [figures/paired_accuracy_intervals.png](figures/paired_accuracy_intervals.png)
- [figures/paired_correctness.pdf](figures/paired_correctness.pdf)
- [figures/paired_correctness.png](figures/paired_correctness.png)
- [figures/paired_macro_f1_intervals.pdf](figures/paired_macro_f1_intervals.pdf)
- [figures/paired_macro_f1_intervals.png](figures/paired_macro_f1_intervals.png)
- [figures/per_class_f1.pdf](figures/per_class_f1.pdf)
- [figures/per_class_f1.png](figures/per_class_f1.png)
- [figures/per_class_recall.pdf](figures/per_class_recall.pdf)
- [figures/per_class_recall.png](figures/per_class_recall.png)
- [figures/top_confusion_improvements.pdf](figures/top_confusion_improvements.pdf)
- [figures/top_confusion_improvements.png](figures/top_confusion_improvements.png)
- [results/aligned_predictions.csv](results/aligned_predictions.csv)
- [results/comparison_metrics.csv](results/comparison_metrics.csv)
- [results/confusion_changes_after_finetuning.csv](results/confusion_changes_after_finetuning.csv)
- [results/confusion_finetuned_mobilenetv2_counts.csv](results/confusion_finetuned_mobilenetv2_counts.csv)
- [results/confusion_finetuned_mobilenetv2_row_normalized.csv](results/confusion_finetuned_mobilenetv2_row_normalized.csv)
- [results/confusion_frozen_mobilenetv2_counts.csv](results/confusion_frozen_mobilenetv2_counts.csv)
- [results/confusion_frozen_mobilenetv2_row_normalized.csv](results/confusion_frozen_mobilenetv2_row_normalized.csv)
- [results/confusion_random_forest_counts.csv](results/confusion_random_forest_counts.csv)
- [results/confusion_random_forest_row_normalized.csv](results/confusion_random_forest_row_normalized.csv)
- [results/pairwise_agreement.csv](results/pairwise_agreement.csv)
- [results/pairwise_prediction_disagreements.csv](results/pairwise_prediction_disagreements.csv)
- [results/pairwise_statistical_tests.json](results/pairwise_statistical_tests.json)
- [results/per_class_comparison.csv](results/per_class_comparison.csv)
- [src/three_model_analysis.py](src/three_model_analysis.py)
- [src/verify_outputs.py](src/verify_outputs.py)
