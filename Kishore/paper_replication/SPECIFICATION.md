# Barchi et al. paper-to-implementation specification

Reference is the supplied 21-page `Paper.pdf`, SHA-256
`8c667e53ae090b3accdf4b19a1b4fed84c487efeafbcb8154515eb917c0f8f4d`.
Page numbers below are printed journal pages. The local copy was read with
PDFBox text extraction and Figure 2 was checked visually. [Version of record](https://doi.org/10.1016/j.ascom.2019.100334).

| Paper location | Requirement | Implementation and unresolved details |
| --- | --- | --- |
| §2, p. 3 | Defined Galaxy Zoo 1 E and S classes for binary supervision | Join authors' catalog to Lintott et al. GZ1 Table 2 by exact-string DR7 photometric OBJID. Use exclusive E/S published clean flags, `UNCERTAIN=0`, and reject merger-majority votes when provided. Galaxy Zoo states the flags incorporate the 0.8 clean threshold and threshold-aware debiasing. The paper does not identify the precise input table or vote reprocessing, so use of the released flags is a reconstruction decision. |
| §2, p. 3 | SDSS DR7 r band, 0.03 < z < 0.1, r Petrosian mag < 17.78, Galactic latitude abs(b) ≥ 30° | Rely on the parent selection declared for the author release. Cross-check any available z and r magnitudes and calculate ICRS Galactic latitude from catalog RA/Dec when necessary. Missing fields are counted in the provenance through the number independently checked. |
| §2, Eq. 1 and Table 1, p. 3 | K = (Rp / (FWHM / 2))², dimensionless | Use released K for the primary catalog analysis. Implement explicit pixel/arcsec conversions for checks when the matching original radius and seeing are available. Eq. 1 has no extra axis-ratio term. The paper's 0.396 arcsec per pixel and typical 1.5 arcsec seeing describe the survey, and the typical seeing cannot stand in for object-level FWHM. |
| Table 1, p. 3 | GZ1 counts for K ≥ 5, 10, 20 are 104,787, 89,829, and 58,030 | Treat as paper counts, with observed merged, defined-label, successful-measurement counts reported separately. Original per-object split IDs and criterion-by-criterion membership were not found. |
| §3, pp. 3–4 | Cut r-band FITS stamps, remove secondary sources by median along intersecting isophote, segment for A/S/G2/H, use the clean radial profile for C | Prefer original author features. Source performs multiple SExtractor runs, one per metric threshold, an isophotal interpolation, a 32-pixel background pass, and gradient enhancement for A. Native DR7 fpC corrected FITS and field/target metadata are needed for an independent end-to-end remeasurement. The extractor accepts explicitly prepared intermediate arrays, and the current evidence does not establish equivalence from the raw frames. |
| §3, p. 3, and §3.2, Fig. 2, pp. 4–5 | C = log10(R75/R35) | Author measurements in the catalog. Prepared extractor uses the supplied cumulative curve on integer radii and the legacy 2Rp aperture interpolation in `indexes.pyx`. The legacy fractional factor uses floor(Rp), as written in the source. |
| §3, p. 3 | A = 1 − Spearman(original, π-rotated) | Prepared extractor uses the enhanced author input, a cubic rotation around frame center with nearest boundary extension and the common mask. Spearman uses mean ranks for ties. It keeps the asymmetric mask conventions of the reviewed code. The public wrapper emits correlation under `sA3`, whereas the paper prints distance. The release conversion has not been independently confirmed. |
| §3, p. 3, §3.2, Fig. 2c | S = 1 − Spearman(original, Butterworth-filtered) | Fourier radial response 1 / (1 + (D / (0.2 Rmax))^4), with a central radius 0.2 FWHM omitted and at least six eligible pixels, consistent with public code. The public wrapper emits `sS3` as correlation, so the same release convention caveat applies. |
| §3, p. 4, and Fig. 2d | Entropy H | 130 equal-width histogram bins across the selected nonzero intensities; normalized Shannon entropy divided by log(130). Log base cancels. |
| §3, pp. 3–4, Fig. 2e/f, Rosa et al. (2018) | Gradient-pattern second moment G2 | First differences, truncated integer distance rings, opposite-phase/matched-modulus removal, residual fraction times (2 − alignment). Follow public code's filter excluding vectors with either component zero and its scan-order pairing. Use radial .01 pixel, relative modulus .02 and phase .03 rad from `ParallelConfig.ini`. The fine-tuning figure and configuration alone do not prove the precise catalog production revision. |
| §3.2, pp. 4–5 | GHS-based optimization of segmentation and feature settings using 1,000 E and 1,000 S | No rerun of label-dependent feature optimization. Use released author features and the separately documented fixed parallel config for the diagnostic route. The author's historic optimization sample membership is unknown, so independence of the published settings from new held-out objects cannot be established. |
| §4, pp. 4–6 | sklearn CART, SVM and ReLU MLP with layers 44/88/22 | Reuse the model families and exact published MLP architecture. Use Gini CART and tune depth 6/12/unlimited and min leaf 1/10. The SVM fits C .5 or 2 using a 128-landmark Nyström RBF map with gamma 1/5 and a primal linear SVM. MLP uses Adam, batch 2,048, up to 60 epochs, tolerance 0.0003 with patience 8, and alpha .0001 or .001. These choices fill unspecified settings. Fold-local standardization for SVM/MLP and fold-local kernel landmarks for SVM. No imputation for failed rows. |
| §4, p. 4, §2, p. 3, and §6.1, p. 6 | Approximately 90/10 development/test and CV inside development, summarized as 80/10/10; accuracy emphasized | Our predeclared evaluation uses approximately 80/20 across object groups, five outer and three inner folds, plus five development folds for final selection. Select via macro-F1 and also report accuracy. Original seed, exact fold count and split lists were not found. Common holdout and outer fold assignments derive from a master K ≥ 5 sample stratified by label and disjoint K bands. |
| §6.2, pp. 7–8 and appendix pp. 14–16 | Binary task retains original imbalance | Keep the original label proportions in each eligible sample. Add a matched majority dummy. Figure 5 SMOTE and other balancing results belong to the three-class investigation. |
| §6.1 and Table 3, pp. 6–7 | Binary traditional OA targets | DT/SVM/MLP are 94.8/94.6/94.6% for K ≥ 5, 95.7/95.8/95.6% for K ≥ 10, and 98.5/98.6/98.6% for K ≥ 20. The CNN column is a separate deep-learning result. |
| §6.3–6.5, pp. 8–11 | Classification diagnostics | Use held-out and OOF confusion matrices, per-class metrics and saved decision outputs. Figure 7's 6,763-object discussion is not a published split manifest. Figure 9 uses model-assigned labels for GZ1 Undefined galaxies and cannot provide their training labels. |
| Appendix, pp. 14–16 | 670,560-row catalog, five features, K, error, output classifications | Error 0 and finite measurements are required. Include the published flags in the manifest. Ignore all published ML and CNN classes/probabilities as inputs and labels. Record all exclusions by label status and K threshold. Coordinate matches within one arcsecond supplement exact DR7 IDs. Exact image hashes are possible only where images or released checksums are present. |

## Sources and settings audit

- Author data, Mendeley version 1, CC BY 4.0, (670{,}560) documented galaxies. Release lists a 45 MB ZIP. Hash and member names are recorded when acquired. [Data release](https://data.mendeley.com/datasets/tg7c985c2n/1).
- Galaxy Zoo 1 original Table 2, classified SDSS DR7 spectroscopic sample and independent labels. Version precedes GZ2. Hash recorded at acquisition. [Galaxy Zoo documentation](https://data.galaxyzoo.org/) · [Lintott et al. 2011](https://doi.org/10.1111/j.1365-2966.2010.17432.x).
- Author CyMorph source at commit `1ed0ee4c3de7ad1282aa24d5ada0666229e59d7f`, January 21, 2019. The repository indicates GPL-3.0. File identities reviewed include `CyMorph.pyx` `722c7240c526f89ca84451b523936ee375d5f803`, `ParallelConfig.ini` `a3f390b8c35ad8c09a55894f2d231ff7c69aec1d`, and `gridAlg.pyx` `1fa33cccd066ed502fee50f461fcf58b15468003`. Its README requires Python 2.7, Cython, SExtractor, PyFITS and MPI. The project implements scientific measurements in Linux Python 3 using explicit intermediates and keeps the original author feature catalog as the training source. [Author code](https://github.com/paulobarchi/CyMorph/tree/1ed0ee4c3de7ad1282aa24d5ada0666229e59d7f) · [Gradient method](https://doi.org/10.1093/mnrasl/sly054).
- The parallel config uses r band, `stamp_size=5`, and detection thresholds 1.8 for A, 1.4 for S, 1.0 for G2, and 0.8 for H. The ordinary `config.ini` uses different values. The release's exact compiler and segmentation runtime remain unverified. [Parallel configuration](https://github.com/paulobarchi/CyMorph/blob/1ed0ee4c3de7ad1282aa24d5ada0666229e59d7f/ParallelConfig.ini).
- Scientific data product is an SDSS DR7 r-band fpC corrected FITS frame. Catalog reconstruction uses two compact release files, with no bulk image acquisition. Full-frame storage depends on the distinct run/rerun/camcol/field tuples and should be estimated after the joined sample is available. [SDSS DR7 data access](https://classic.sdss.org/dr7/access/) · [fpC format](https://classic.sdss.org/dr7/dm/flatFiles/fpC.php).
- The author driver `PCyMorph.py` reads `run, rerun, camcol, field, ra, dec, dr7objid, petroRad_r, petroMag_r, rowc_r, colc_r`, converts `petroRad_r` from arcseconds using 0.396 arcsec/pixel, and requests the r-band corrected frames. The first-author thesis describes the Petrosian profile, source removal and excludes nonzero error flags from traditional-model training. These details do not identify the original split roster. [Parallel driver](https://github.com/paulobarchi/CyMorph/blob/1ed0ee4c3de7ad1282aa24d5ada0666229e59d7f/PCyMorph.py) · [Barchi thesis, §§3.3.1–3.3.3](https://mtc-m21c.sid.inpe.br/col/sid.inpe.br/mtc-m21c/2020/03.09.19.33/doc/publicacao.pdf).

The earlier Galaxy10 DECaLS baseline is a separate ten-class rendered RGB experiment. None of its images, predictions, seven features, or models are used here.

## Actual catalog format and grouping

GZ1 Table 2 expresses RA as sexagesimal hours and Dec as signed sexagesimal
degrees. The loader converts those before the Galactic latitude and sky
separation calculations. The released author table contains DR7 ID, K,
features, quality flags, T-Type, and model outputs. It does not supply
redshift, Petrosian magnitude, or coordinates. The reconstructed samples
use author-provided K and independently enforce the latitude cut from GZ1.

For at most one extra row per thousand and label-homogeneous groups,
stratification operates on one representative per object group, using its
highest eligible K band for the master stratification. Every member inherits
the group assignment. Conventional StratifiedGroupKFold handles other
repeated-group cases, and ordinary StratifiedKFold handles unique groups.
All three routes verify separation and coverage.

The initial exact-kernel SVC run spent over seven minutes on four workers
without completing its first outer fold at K ≥ 5. We stopped it based on
runtime while the SVM had no measured fold or held-out result. The same
approximation was fixed for all three K cuts. The classifier fits every
eligible training row. Only the kernel map uses a landmark approximation.
The interrupted artifacts remain under `results/exact_svc_attempt/`.

A second exploratory run reached the 200-epoch MLP cap in inner fits
while the first largest-sample outer fold was still running. The final
workflow used the 60-epoch budget and larger batch shown above for all
cuts. This change was driven by wall time before a completed MLP outer
score and without using held-out results. The archival attempts have
separate fingerprints and are excluded from the final report.

## Complete-case follow-up

The Mendeley V1 archive README documents quality flags 0 (success), 1 (bright
nearby objects), 2 (Petrosian radius), 3 (G2), 4 (H), 5 (C), 6 (A) and 7 (S).
It does not enumerate numeric sentinels. Review of the author `CyMorph.pyx`
at the pinned commit confirms that it sets measurement-specific errors,
without documenting the release's two literal codes. The released CSV
contains −9999.999999 and −9999.99 in these columns, including rows marked
successful. The observed values are impossible for the published measurements
and are treated as missing wherever present. This is a complete-case
reconstruction rule.

| Feature | Valid inclusive domain | Basis |
| --- | --- | --- |
| C | [0, ∞) | log10(R75/R35) with positive ordered radii |
| A, S | [−1, 2] | Union of the Spearman coefficient and its one-minus form |
| G2 | [0, 2] | Surviving gradient fraction times one plus confluence |
| H | [0, 1] | Shannon entropy divided by log of the number of bins |

The classifier rejects nonfinite values and allows absolute numerical
boundary tolerance of 10⁻⁸. It requires author Error = 0 and abstains at
inference for invalid feature vectors. Zero is allowed when mathematically
valid. No imputation is used. The audit distinguishes literal codes,
nonfinite values, other domain failures, per-feature quality flags, and
co-occurrence patterns by K and class.
[Mendeley V1 documentation](https://data.mendeley.com/datasets/tg7c985c2n/1) ·
[CyMorph source](https://github.com/paulobarchi/CyMorph/tree/1ed0ee4c3de7ad1282aa24d5ada0666229e59d7f).

The baseline assignment manifest is hashed and each train, validation, test,
inner, and final-tuning list is intersected with eligibility. Object groups
stay fixed and all retained folds are checked for both classes, exhaustive
coverage, and disjoint groups. Old and new predictions are compared on common
rows with unchanged object identity, class, partition, and outer fold.
The held-out results are follow-up evaluations on the previously inspected
holdout. No selection or model setting was optimized using those scores.

A separate DR7 export variant joins `SpecObj.bestObjID` to
`PhotoObjAll.objID` as exact decimal strings. It records spectroscopic z
(dimensionless), r Petrosian magnitude (magnitudes), unmatched rows, and
exclusions for (0.03 < z < 0.1) and (r < 17.78). The choice of the DR7
science-primary view is an explicit duplicate-spectra policy. The
available-data follow-up has no such metadata. The SQL and provenance
remain separate from the five-feature model matrix. The SVM timing pilot is
training-only and keeps the Nyström estimator in the main run. MLP uses
the same fold-local scaling and optimization settings as before.
