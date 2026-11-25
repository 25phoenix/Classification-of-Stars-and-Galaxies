# Paired Galaxy10 DECaLS model comparison

## Executive summary

On 3,545 aligned held-out galaxies, the frozen MobileNetV2 head correctly classified **1,997** galaxies and the seven-feature Random Forest correctly classified **1,959**. MobileNetV2 led by **1.07 percentage points in accuracy** and **0.45 points in Macro-F1**. Both preferred group-bootstrap intervals include zero. The exact group-aware sensitivity test and row-level McNemar test also give insufficient evidence of an accuracy advantage at the 0.05 level. The per-class results show substantial descriptive trade-offs that the pooled scores obscure.

## Methodology and dataset alignment

Both frozen baselines used the official Galaxy10 DECaLS HDF5 file with 17,736 raw rows, 17,733 eligible rows, 14,188 development rows, and 3,545 held-out rows. The comparison verified the HDF5 SHA-256 and SHA-256 values of `dataset.json`, `splits.json`, and `split_assignments.csv` against Sagar’s run provenance. The shared loader checked the HDF5 labels and split definitions. The two prediction exports joined one-to-one by `row_id` with matching true labels, official class mappings, and exact test membership. The duplicate-object group IDs agree with the cached morphology table and the Random Forest export. Kishore’s grouping rule links identical decoded RGB images or sky positions within one arcsecond, then takes the connected components. Development and test share no group IDs.

- **Random Forest.** The cached inputs match the recorded feature-table hash and use concentration, asymmetry, smoothness, entropy, Gini, M20, and ellipticity in that order. The extraction converts rendered RGB to luminance, applies antialiasing and resizes to a 128-pixel maximum side, then estimates background and segments the central source before measuring morphology. Extraction settings and code hashes agree with the final model metadata. The development-refit forest has 160 trees, maximum depth 14, and balanced subsample class weighting. Its `model.joblib` fingerprint is in the provenance record. The saved prediction hash, probability argmax, all six test metrics, and both confusion matrices agree with the Random Forest artifacts.
- **MobileNetV2.** The pipeline resizes RGB images to 224 × 224, applies MobileNetV2 normalization, and uses frozen ImageNet weights with global average pooling. The trained head applies dropout before a ten-class linear softmax layer on 1,280-dimensional embeddings. The prediction hash matches the integrated handoff manifest. Test accuracy (0.5633286319), Macro-F1 (0.5223538500), and both confusion matrices agree with the handoff.
- **Evaluation.** Both frozen baselines used the shared development-only five-outer, three-inner nested protocol and a final development tuning stage. The analysis uses the saved held-out exports. Neither model was loaded for inference or fitted during comparison. Development OOF metrics were outside this held-out analysis.

## Overall held-out results

All deltas are MobileNetV2 minus Random Forest. Absolute changes are in percentage points (pp). Relative changes divide by the Random Forest score.

| Metric | Random Forest | MobileNetV2 | Δ (pp) | Relative Δ |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 55.26% | 56.33% | +1.07 | +1.94% |
| Macro-F1 | 51.78% | 52.24% | +0.45 | +0.87% |
| Weighted-F1 | 54.57% | 55.11% | +0.54 | +0.98% |
| Balanced accuracy | 52.62% | 52.43% | -0.19 | -0.37% |
| Macro-precision | 51.84% | 54.49% | +2.65 | +5.11% |
| Macro-recall | 52.62% | 52.43% | -0.19 | -0.37% |

Accuracy weighs each galaxy equally. Macro-F1, macro-precision, and macro-recall give each of the ten classes equal weight. Weighted-F1 weights class F1 by support. Balanced accuracy equals macro-recall here because all ten classes occur in the test data.

## Per-class performance

| Galaxy10 class | Support | RF F1 | MobileNetV2 F1 | Δ F1 (pp) | RF recall | MobileNetV2 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Disturbed Galaxies | 217 | 25.00% | 15.66% | -9.34 | 21.20% | 10.14% |
| Merging Galaxies | 370 | 57.39% | 49.73% | -7.66 | 71.89% | 48.92% |
| Round Smooth Galaxies | 529 | 70.15% | 67.63% | -2.52 | 72.40% | 70.70% |
| In-between Round Smooth Galaxies | 406 | 60.82% | 55.64% | -5.17 | 60.59% | 60.10% |
| Cigar Shaped Smooth Galaxies | 65 | 30.22% | 34.62% | +4.40 | 32.31% | 27.69% |
| Barred Spiral Galaxies | 409 | 45.65% | 45.52% | -0.13 | 41.08% | 37.90% |
| Unbarred Tight Spiral Galaxies | 365 | 44.17% | 52.11% | +7.94 | 44.11% | 57.53% |
| Unbarred Loose Spiral Galaxies | 525 | 41.03% | 46.84% | +5.81 | 37.90% | 50.86% |
| Edge-on Galaxies without Bulge | 284 | 77.83% | 79.86% | +2.03 | 80.99% | 83.10% |
| Edge-on Galaxies with Bulge | 375 | 65.57% | 74.74% | +9.17 | 63.73% | 77.33% |

The largest Random Forest F1 advantages occur in **Disturbed Galaxies** (+9.34 pp for the forest), **Merging Galaxies** (+7.66 pp), and **In-between Round Smooth Galaxies** (+5.17 pp). MobileNetV2 has its largest F1 advantages on **Edge-on Galaxies with Bulge** (+9.17 pp), **Unbarred Tight Spiral Galaxies** (+7.94 pp), and **Unbarred Loose Spiral Galaxies** (+5.81 pp). For **Cigar Shaped Smooth Galaxies**, MobileNetV2 improves F1 despite recovering 18 of 65 examples versus 21 for the forest. Its higher precision accounts for the difference. These class-level differences are descriptive.

Class support ranges from 65 to 529. MobileNetV2 gains 68 correct predictions on loose spirals, 51 on edge-on galaxies with bulge, and 49 on tight spirals. It loses 85 on mergers and 24 on disturbed galaxies. The resulting accuracy gain is larger than the Macro-F1 gain, and its equal-class recall drops by 0.19 pp. Class prevalence therefore affects the aggregate result, while gains and losses are spread across several support levels. The gains cannot be attributed to class imbalance alone. The models also differ in their training weights, representation, and classifier.

## Paired statistics

| Outcome | Galaxies |
|---|---:|
| Both correct | 1,334 |
| Random Forest only correct | 625 |
| MobileNetV2 only correct | 663 |
| Both wrong | 923 |

The models assign the same label to **1,689 of 3,545 galaxies (47.64%)**, including agreements on incorrect labels. The 38 additional correct MobileNetV2 predictions are the net result of 1,288 discordant correctness pairs.

**Exact two-sided McNemar test.** The conventional test conditions on the 1,288 discordant rows and uses a binomial probability of one half under equal marginal accuracy. It gives **p = 0.302555**. Its row-level interpretation assumes independent paired observations. Repeated observations can correlate their outcomes, so the conventional p-value is reported alongside a group-aware analysis.

**Preferred paired group bootstrap.** The held-out sample contains **3,535 groups**, with 3,525 singleton groups and ten groups of size two. Each of 20,000 replicates draws 3,535 groups uniformly with replacement, retaining all rows and both models’ predictions from every draw. The number of rows may vary across replicates. The procedure recomputes row-weighted accuracy and fixed-ten-class Macro-F1 and takes the 2.5th and 97.5th percentiles of their paired differences. It uses NumPy PCG64 with seed `20261124`. Confidence intervals treat groups as independent sampling units and condition on the two fitted models.

| Difference | Estimate (pp) | Group bootstrap 95% CI (pp) |
|---|---:|---:|
| Accuracy | +1.07 | [-0.90, +3.02] |
| Macro-F1 | +0.45 | [-1.87, +2.75] |

**Group randomization sensitivity test.** The exact two-sided test independently swaps the two model labels for all rows in each group, using the absolute net difference in correct counts as the statistic. Under joint within-group model-label exchangeability and independence across groups, convolution over group sign allocations gives **p = 0.301803**. This stronger null gives a sensitivity check for the accuracy contrast. Neither test rejects at 0.05, and the intervals remain compatible with modest advantages in either direction. These estimates and tests measure differences on a shared evaluation sample and support no causal attribution.

## Confusion and disagreements

The most frequent ordered disagreements in predicted labels were Random Forest **Round Smooth** versus MobileNetV2 **In-between Round Smooth** (108), Random Forest **Merging** versus MobileNetV2 **Unbarred Loose Spiral** (97), and Random Forest **Barred Spiral** versus MobileNetV2 **Unbarred Tight Spiral** (95). The reverse smooth-class direction occurred 85 times, making those two classes the largest unordered predicted-label disagreement pair (193). These counts compare model outputs, irrespective of the true class.

Both models struggle with spiral categories. The forest labels 73 true barred spirals as loose spirals and 63 as tight spirals. MobileNetV2 makes these errors 102 and 80 times, respectively. They also confuse the smooth roundness categories. The forest labels 56 true in-between smooth galaxies as round smooth, and MobileNetV2 does so 73 times. Both identify fewer than a quarter of the true disturbed galaxies. See the labeled raw and row-normalized matrices for the full pattern.

## Scientific interpretation and limitations

Concentration, asymmetry, and related summaries retain explicit information about light profiles and irregular structure. This could help explain the Random Forest’s advantage on mergers and disturbed galaxies. The MobileNetV2 embedding retains image cues unavailable to seven scalar measurements and may help distinguish tight or loose spiral structure and edge-on profiles. The evaluation compares the complete pipelines, so it cannot isolate any particular cue, model component, or training choice.

Frozen ImageNet filters were learned on natural images. They may underrepresent low-surface-brightness structure and subtle distinctions in bulges, bars, or spiral pitch, especially after resizing and global average pooling. Only the head adapts to Galaxy10. The fixed holdout and shared group definitions support a paired comparison within this dataset. The intervals do not include retraining variability, label uncertainty, or transfer to another survey. Differences across individual classes were explored descriptively without class-specific hypothesis tests.

The experiment takes its motivation from Barchi et al.’s comparison of nonparametric morphology with deep learning. Their principal data came from SDSS DR7, with CyMorph measurements and two- and three-class tasks. This project uses Galaxy10 DECaLS, ten classes, its own seven-feature implementation, and a frozen ImageNet MobileNetV2 representation. The separate Barchi/DR7 work was outside the analysis. [Barchi et al., Astronomy and Computing 30, 100334](https://doi.org/10.1016/j.ascom.2019.100334)

## Conclusion

MobileNetV2 has the higher observed accuracy and Macro-F1 on the shared holdout. The group-aware intervals and paired tests leave the overall advantage unresolved. The clearest observed differences are class-specific and favor different classifiers.

## Reproducibility and file inventory

From the project root, use the existing Kishore environment to rebuild the computed outputs, then run the independent standard-library verifier.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dop-phase3-mpl \
  Kishore/.venv/bin/python comparison/src/comparison_analysis.py
python3 comparison/src/verify_outputs.py
```

The analysis updates derived outputs under `comparison/`. The verifier recomputes all six metrics per model, per-class scores, confusion and agreement counts, and both exact p-values from the paired export. It also checks the figure formats and analysis source fingerprint.

- `COMPARISON_REPORT.md`
- [README.md](README.md)
- [audit/alignment_audit.json](audit/alignment_audit.json)
- [audit/provenance.json](audit/provenance.json)
- [figures/confusion_matrices.pdf](figures/confusion_matrices.pdf)
- [figures/confusion_matrices.png](figures/confusion_matrices.png)
- [figures/overall_metrics.pdf](figures/overall_metrics.pdf)
- [figures/overall_metrics.png](figures/overall_metrics.png)
- [figures/paired_difference_intervals.pdf](figures/paired_difference_intervals.pdf)
- [figures/paired_difference_intervals.png](figures/paired_difference_intervals.png)
- [figures/paired_outcomes.pdf](figures/paired_outcomes.pdf)
- [figures/paired_outcomes.png](figures/paired_outcomes.png)
- [figures/per_class_f1.pdf](figures/per_class_f1.pdf)
- [figures/per_class_f1.png](figures/per_class_f1.png)
- [figures/per_class_recall.pdf](figures/per_class_recall.pdf)
- [figures/per_class_recall.png](figures/per_class_recall.png)
- [figures/prediction_disagreements.pdf](figures/prediction_disagreements.pdf)
- [figures/prediction_disagreements.png](figures/prediction_disagreements.png)
- [results/comparison_metrics.csv](results/comparison_metrics.csv)
- [results/confusion_mobile_counts.csv](results/confusion_mobile_counts.csv)
- [results/confusion_mobile_row_normalized.csv](results/confusion_mobile_row_normalized.csv)
- [results/confusion_rf_counts.csv](results/confusion_rf_counts.csv)
- [results/confusion_rf_row_normalized.csv](results/confusion_rf_row_normalized.csv)
- [results/disagreement_analysis.csv](results/disagreement_analysis.csv)
- [results/off_diagonal_confusions.csv](results/off_diagonal_confusions.csv)
- [results/paired_predictions.csv](results/paired_predictions.csv)
- [results/per_class_comparison.csv](results/per_class_comparison.csv)
- [results/prediction_agreement_matrix.csv](results/prediction_agreement_matrix.csv)
- [results/statistical_tests.json](results/statistical_tests.json)
- [src/comparison_analysis.py](src/comparison_analysis.py)
- [src/verify_outputs.py](src/verify_outputs.py)
