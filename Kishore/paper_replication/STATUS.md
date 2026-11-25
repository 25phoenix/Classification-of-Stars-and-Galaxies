# Experiment status

The real classification-stage reconstruction is complete for the as-released
CyMorph protocol. All 12 classifier-and-threshold runs have complete outer
folds, final held-out predictions, and fitted model bundles. Twelve unit tests,
the workflow check, and the independent prediction audit passed. The Galaxy10
metrics, model and report matched their recorded checksums.

Both official catalogs are staged. The author release contains 670,560 rows,
the Galaxy Zoo 1 spectroscopic release contains 667,944 rows, and the exact
DR7 string-ID join matched 653,355 author rows. The prepared samples contain
221,810 eligible rows at K ≥ 5, 173,106 at K ≥ 10, and 92,376 at K ≥ 20.

The model inputs include finite sentinel codes identified in the released
CyMorph columns during final plot review. The data-quality diagnostics quantify
this limitation, and the scored model artifacts use the frozen as-released
protocol. The two original catalogs lack the fields needed to independently reapply
the redshift and Petrosian magnitude cuts. Agreement on paired real SDSS-DR7
images remains unverified. The sentinel-aware refit is documented below.

[Final report](results/catalog/REPORT.md) · [Independent audit](results/catalog/verification.json) · [Paper specification](SPECIFICATION.md)

## Complete-case follow-up

The available-data follow-up is complete in
`results/catalog_complete_case/`. It retains 197,985, 157,949 and 84,927
rows across K ≥ 5, 10 and 20. All original assignments were preserved for
surviving rows. The independent audit validated all 12 model-and-cut
runs and every retained feature vector. A real model abstained on a
sentinel input. All 339 protected input and output files matched their
pre-run hashes.

The original and complete-case scores are compared on common held-out
rows in the follow-up report. The highest primary score is the K ≥ 20
MLP with 95.45% accuracy and 0.9211 macro-F1. The holdout was previously
inspected.

This complete-case experiment predates the DR7 export. The metadata-filtered
variant lives separately in `results/catalog_dr7_complete_case/`.

## Exact RBF sensitivity recovery

The training-only pilot completed in 97.49 seconds. Recovery verified the
fingerprints, product hashes, and coverage of 11 saved outer checkpoints.
K ≥ 20 already had a complete model and evaluation. K ≥ 10 reused all
five outer folds and finished final tuning, refit, and holdout scoring
during recovery. K ≥ 5 reused its first outer fold and finished the
remaining four folds, final tuning, refit, and holdout scoring. All 15
outer folds and all three final model bundles and test predictions exist.
The regenerated exact-SVM report and independent audit are complete.
The audit recomputed every fold and holdout metric and reproduced all
saved exact-SVM holdout labels and decision scores after reloading.
All 17 unit tests, the primary workflow check, and the primary
independent prediction audit passed again.

The sensitivity run uses the same corrected rows and previously inspected
holdout as the primary Nyström model. The paper's kernel was unspecified.
The exact RBF SVC and Nyström pipeline also differ in their loss and solver
objective.

[Exact-SVM report](results/catalog_complete_case_exact_svm/REPORT.md) ·
[Paired exact/Nyström report](results/catalog_complete_case_exact_svm/PAIRED_REPORT.md) ·
[Exact-SVM verification](results/catalog_complete_case_exact_svm/verification.json)

[Primary follow-up report](results/catalog_complete_case/REPORT.md) ·
[Selection audit](results/catalog_complete_case/selection_audit.json) ·
[Independent verification](results/catalog_complete_case/verification.json)

## DR7 metadata-filtered experiment

All 12 classifier/threshold runs, 60 outer folds, final development refits,
and follow-up holdout predictions are complete. The CasJobs export contained
1,049,595 rows. The validator excluded all entries for 140 conflicting
exact IDs. Seven of those IDs occurred in the author catalog, accounting
for 14 author rows. The staged input was preserved.

The strict DR7 cuts retained 95,209, 82,736, and 53,220 objects for K ≥ 5,
10, and 20. These counts are 9,578, 7,093, and 4,810 below paper Table 1.
The K ≥ 20 MLP scored 97.22% accuracy, 92.32% balanced accuracy, and
0.9362 macro-F1. The Nyström SVM had the highest K ≥ 20 balanced accuracy
at 93.27%. The paired report compares earlier complete-case predictions
on both their full holdout and the same DR7-retained objects.

All 20 tests, the metadata workflow check, and both metadata audits
passed. Reloaded model scores matched saved outputs. The 339 original
protected files and 365 earlier follow-up files matched their recorded
hashes. The holdout was previously inspected. The original paper roster
and paired raw image feature agreement remain unresolved.

[Final DR7 report](results/catalog_dr7_complete_case/REPORT.md) ·
[Export audit](results/catalog_dr7_complete_case/metadata_export_audit.json) ·
[Independent fold audit](results/catalog_dr7_complete_case/verification.json) ·
[Paired comparison and score checks](results/catalog_dr7_complete_case/dr7_complete_case_comparison.json).
