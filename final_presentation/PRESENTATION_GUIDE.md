# Presentation guide

**Deck:** Classification of Stars and Galaxies  
**Presenters:** Kishore and Sagar  
**Length:** 28 main slides plus 6 appendix slides. Plan 17–20 minutes for the main presentation and keep the appendix for questions.

## Slide list

| # | Title | Purpose | Suggested lead |
|---:|---|---|---|
| 1 | Classification of Stars and Galaxies | Title and honest project framing. | Shared |
| 2 | Why galaxy morphology matters | Survey motivation and visual difficulty. | Shared |
| 3 | Research questions | Three Galaxy10 questions plus the literature track. | Shared |
| 4 | Two experimental tracks | Separate experimental tracks and contributor roles. | Shared |
| 5 | Galaxy10 DECaLS dataset | Dataset counts, exclusions, and image format. | Shared |
| 6 | Shared split and model selection | Common object groups, nested baseline selection, and fine-tuned selection distinction. | Shared |
| 7 | Seven morphological features | Exact seven morphological measurements used by Kishore. | Kishore |
| 8 | Random Forest model selection | Random Forest selection and locked-test result. | Kishore |
| 9 | Frozen MobileNetV2 representation | Sagar’s frozen backbone and ten-class classification head. | Sagar |
| 10 | Nested evaluation of frozen embeddings | Cached embeddings and nested head evaluation. | Sagar |
| 11 | Two-stage MobileNetV2 adaptation | Warmup, partial unfreezing, and full-development refit. | Sagar |
| 12 | Training-time image augmentation | Training-only flips, rotations, and translations. | Sagar |
| 13 | Fine-tuning learning curves | Reported learning curves and stage transition. | Sagar |
| 14 | Held-out test performance | Three-way accuracy and Macro-F1 on the identical test rows. | Shared |
| 15 | Class-balanced metrics | Weighted F1, balanced accuracy, and macro averages. | Shared |
| 16 | F1 across ten morphologies | Class-wise F1 from the comparison export. | Shared |
| 17 | Error structure on the locked test | Three normalized confusion matrices. | Shared |
| 18 | Disturbed galaxies remain difficult | Remaining errors in disturbed galaxies. | Shared |
| 19 | Paired accuracy differences | Paired accuracy differences, intervals, and adjusted p-values. | Kishore |
| 20 | What the model comparison supports | Interpretation and non-isolation of training changes. | Kishore |
| 21 | Barchi / SDSS DR7 reconstruction | CyMorph, GZ1, DR7, and reconstruction method. | Kishore |
| 22 | DR7 sample selection changes the task | K-cut sample counts against the published table. | Kishore |
| 23 | DR7 held-out results | Binary held-out results for tree, SVM, and MLP. | Kishore |
| 24 | How the tracks relate | Synthesis without equating the two tasks. | Kishore |
| 25 | Scientific limitations | Evaluation, generalization, and reconstruction limitations. | Kishore |
| 26 | Conclusions | Main conclusions and score progression. | Kishore |
| 27 | Future experiments | Clearly labeled proposed follow-up experiments. | Shared |
| 28 | Thank you / references | Thank you and abbreviated references. | Shared |
| 29 | Random Forest implementation details | RF extraction and final hyperparameters. | Kishore |
| 30 | MobileNetV2 configuration | MobileNetV2 setup and training schedule. | Shared |
| 31 | Per-class F1 and support | Full ten-class support and F1 values. | Kishore |
| 32 | Paired inference details | Exact paired intervals and adjusted tests. | Kishore |
| 33 | DR7 reconstruction details | DR7 sample sizes and selected accuracies. | Kishore |
| 34 | Fine-tuned prediction gallery | Actual fine-tuned prediction examples. | Sagar |

## Narrative and timing

- Slides 1–6: problem and shared data, about 3 minutes. Both presenters should know the split and ten labels.
- Slides 7–8: Kishore’s feature pipeline, about 1.5 minutes.
- Slides 9–13: Sagar’s CNN experiments, about 3.5 minutes.
- Slides 14–20: comparison and inference, about 4.5 minutes. Kishore can lead the paired statistics while Sagar explains class behavior.
- Slides 21–23: Kishore’s DR7 reconstruction, about 2.5 minutes.
- Slides 24–28: synthesis, limitations, conclusion, and questions, about 2.5 minutes.

## Source map

| Topic | Authoritative project source |
|---|---|
| Galaxy10 dataset, RF, seven features | `Kishore/results/final/REPORT.md`, `model_metadata.json`, `class_mapping.json` |
| Frozen CNN | `Sagar/results/frozen_baseline_full/results_report.md`, `model_config.json` |
| Fine-tuned CNN | `Sagar/results/finetuned_mobilenetv2/finetuned_results_report.md`, `finetune_config_and_timings.json` |
| Three-way comparison, figures, paired tests | `comparison/three_model/THREE_MODEL_REPORT.md`, `results/`, `figures/` |
| DR7 reconstruction | `Kishore/paper_replication/results/catalog_dr7_complete_case/REPORT.md`, `selection_audit.json`, `metadata_export_audit.json` |
| Reference paper | Project `Paper.pdf` and `Kishore/paper_replication/SPECIFICATION.md` |

## Likely professor questions

**Do these models distinguish stars from galaxies?**
No. The implemented Galaxy10 benchmark distinguishes ten galaxy morphologies. A separate labeled stellar benchmark would be required.

**Why use Macro-F1?**
Class supports vary. Macro-F1 gives each class equal weight, which exposes weaknesses obscured by overall accuracy.

**Were all three models tuned identically?**
No. RF and the frozen CNN used five outer and three inner folds. The fine-tuned run used one development validation fold for epoch selection, then refit on all development rows. All three share the same locked test rows.

**Is the 20-point gain caused by fine-tuning alone?**
The complete adaptation pipeline improved performance. Augmentation, head choices, schedule, and partial unfreezing changed together, so an ablation is needed to isolate causes.

**How was significance assessed?**
Predictions were paired by the same 3,545 galaxies. The analysis used exact McNemar tests with Holm correction and 20,000 group-bootstrap replicates across 3,535 duplicate-object groups.

**Why is the DR7 accuracy much higher?**
DR7 is a separate binary elliptical/spiral task on a filtered feature catalog. The Galaxy10 task has ten image classes. The numbers do not form a like-for-like ranking.

**Did the reconstruction exactly reproduce Barchi et al.?**
No. The authors' exact roster and split indices were unavailable. Released CyMorph measurements were matched to GZ1 labels and DR7 metadata with documented cuts. The selected sample counts differ from the paper.

## Scientific cautions

- Sagar's supplied scores are accepted as reported. The deck does not claim independent weight-level rerun of fine-tuned inference.
- The DR7 holdout had been inspected before the final refit; treat it as a reconstruction analysis rather than a new confirmatory test.
- Paired intervals condition on fitted models and do not capture retraining variability.
- No cross-survey generalization was measured.

## Presenting the figures

Slides 13, 17, and 34 embed authentic project figures. Slide 16 recreates the per-class chart as editable bars from the comparison CSV. Slide text, comparison bars, statistical intervals, and tables remain editable. The PDF reproduces the rendered slides.

