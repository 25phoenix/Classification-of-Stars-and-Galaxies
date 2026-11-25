# Sagar handoff verification

Workspace date 2026-11-24. Verification covered the supplied frozen MobileNetV2 baseline and its saved records. Training, optional fine-tuning, and comparative scoring were outside this operation.

## A. Initial state

The initial `find Sagar -print` inventory contained six root files and no subdirectories or hidden entries.

```text
Sagar
Sagar/PROJECT_GUIDE.md
Sagar/Sagar_Handoff_For_Kishore.zip
Sagar/Star_Galaxy_Classification_Colab.ipynb
Sagar/head_model.keras
Sagar/head_model.keras.zip
Sagar/src.zip
```

`PROJECT_GUIDE.md` covers a broader Colab and cutout workflow and includes illustrative performance ranges. The extracted `results_report.md`, configuration, provenance and predictions document the measured frozen-feature experiment.

### Complete original archive inventory

**src.zip**

```text
src/
src/__init__.py
src/config.py
src/data_contract.py
src/embeddings.py
src/model_head.py
src/evaluation.py
src/train_baseline.py
src/finetune_optional.py
src/infer.py
src/__pycache__/
src/__pycache__/__init__.cpython-312.pyc
src/__pycache__/config.cpython-312.pyc
src/__pycache__/train_baseline.cpython-312.pyc
src/__pycache__/evaluation.cpython-312.pyc
src/__pycache__/model_head.cpython-312.pyc
src/__pycache__/data_contract.cpython-312.pyc
src/__pycache__/embeddings.cpython-312.pyc
src/__pycache__/__init__.cpython-313.pyc
src/__pycache__/infer.cpython-313.pyc
src/__pycache__/infer.cpython-312.pyc
```

**Sagar_Handoff_For_Kishore.zip**

```text
Sagar_Handoff_For_Kishore/
Sagar_Handoff_For_Kishore/final_tuning_results.json
Sagar_Handoff_For_Kishore/gpu_embedding_extraction_timings.json
Sagar_Handoff_For_Kishore/learning_curves.png
Sagar_Handoff_For_Kishore/model_config.json
Sagar_Handoff_For_Kishore/oof_confusion_matrices.png
Sagar_Handoff_For_Kishore/oof_confusion_matrix_normalized.csv
Sagar_Handoff_For_Kishore/oof_confusion_matrix_raw.csv
Sagar_Handoff_For_Kishore/oof_metrics.json
Sagar_Handoff_For_Kishore/oof_per_class_metrics.csv
Sagar_Handoff_For_Kishore/oof_predictions.csv
Sagar_Handoff_For_Kishore/outer_fold_results.json
Sagar_Handoff_For_Kishore/prediction_gallery.png
Sagar_Handoff_For_Kishore/provenance.json
Sagar_Handoff_For_Kishore/results_report.md
Sagar_Handoff_For_Kishore/test_confusion_matrices.png
Sagar_Handoff_For_Kishore/test_confusion_matrix_normalized.csv
Sagar_Handoff_For_Kishore/test_confusion_matrix_raw.csv
Sagar_Handoff_For_Kishore/test_metrics.json
Sagar_Handoff_For_Kishore/test_per_class_metrics.csv
Sagar_Handoff_For_Kishore/test_predictions.csv
Sagar_Handoff_For_Kishore/timings.json
```

**head_model.keras.zip**

```text
metadata.json
config.json
model.weights.h5
```

The model ZIP already contains the native Keras internals. It is byte-identical to the existing root model. The source archive also contains ten interpreter-generated cache files under `src/__pycache__/`. Those entries remain available in the original ZIP. No destination conflict was found. The archives provide `config.py`, experiment configuration and provenance, and a results report. They contain no dependency lockfile or requirements file.

## B. Operations performed

- Extracted nine Python files and all 21 result files through temporary staging. Checked member integrity and staged bytes before installation. Removed the single archive directory prefix when mapping files to their destinations.
- Copied the root model directly to `results/frozen_baseline_full/head_model.keras`. The two root containers and the installed copy have SHA-256 `1a6e5b762824993907414162694b5f285c86e56f2b66e553192cc1e1c51437c8`.
- Preserved the guide, notebook, original model, and all three archives. Existing identical files are reused and differing destinations trigger a conflict. No source or configuration edits were needed.
- Generated this report and `handoff/file_manifest.json`. The manifest records exact size, SHA-256, origin, and action for each other file, including this report. It excludes its own digest because including it would change the file bytes.

## C. Final directory structure

Output of `find Sagar -print | LC_ALL=C sort` from the project root.

```text
Sagar
Sagar/PROJECT_GUIDE.md
Sagar/Sagar_Handoff_For_Kishore.zip
Sagar/Star_Galaxy_Classification_Colab.ipynb
Sagar/handoff
Sagar/handoff/HANDOFF_VERIFICATION.md
Sagar/handoff/file_manifest.json
Sagar/head_model.keras
Sagar/head_model.keras.zip
Sagar/results
Sagar/results/frozen_baseline_full
Sagar/results/frozen_baseline_full/final_tuning_results.json
Sagar/results/frozen_baseline_full/gpu_embedding_extraction_timings.json
Sagar/results/frozen_baseline_full/head_model.keras
Sagar/results/frozen_baseline_full/learning_curves.png
Sagar/results/frozen_baseline_full/model_config.json
Sagar/results/frozen_baseline_full/oof_confusion_matrices.png
Sagar/results/frozen_baseline_full/oof_confusion_matrix_normalized.csv
Sagar/results/frozen_baseline_full/oof_confusion_matrix_raw.csv
Sagar/results/frozen_baseline_full/oof_metrics.json
Sagar/results/frozen_baseline_full/oof_per_class_metrics.csv
Sagar/results/frozen_baseline_full/oof_predictions.csv
Sagar/results/frozen_baseline_full/outer_fold_results.json
Sagar/results/frozen_baseline_full/prediction_gallery.png
Sagar/results/frozen_baseline_full/provenance.json
Sagar/results/frozen_baseline_full/results_report.md
Sagar/results/frozen_baseline_full/test_confusion_matrices.png
Sagar/results/frozen_baseline_full/test_confusion_matrix_normalized.csv
Sagar/results/frozen_baseline_full/test_confusion_matrix_raw.csv
Sagar/results/frozen_baseline_full/test_metrics.json
Sagar/results/frozen_baseline_full/test_per_class_metrics.csv
Sagar/results/frozen_baseline_full/test_predictions.csv
Sagar/results/frozen_baseline_full/timings.json
Sagar/src
Sagar/src.zip
Sagar/src/__init__.py
Sagar/src/config.py
Sagar/src/data_contract.py
Sagar/src/embeddings.py
Sagar/src/evaluation.py
Sagar/src/finetune_optional.py
Sagar/src/infer.py
Sagar/src/model_head.py
Sagar/src/train_baseline.py
```

## D. Verification results

| Check | Status | Evidence |
|---|---|---|
| Archive integrity | PASS | ZIP CRC checks passed for all three archives and the root model. |
| Source extraction | PASS | All nine Python files match their archive entries byte for byte. |
| Model file integrity | PASS | Native Keras container with readable JSON metadata/config and readable HDF5 weights. Dense kernel and bias have finite values and shapes `(1280, 10)` and `(10,)`. |
| Model architecture consistency | PASS | Saved configuration has 1,280 inputs, dropout 0.2, and a single ten-unit softmax Dense layer with L2 0.0001. This matches the recorded final hyperparameters. |
| Python syntax | PASS | All nine modules compiled in memory. |
| Relative module references | PASS | Static checks resolved all imported package modules and their referenced symbols. |
| Module import checks | FAIL | Package initializer, configuration, data contract, and evaluation imported using Kishore’s existing virtual environment. The five TensorFlow-dependent modules stopped at the missing TensorFlow import. |
| Keras model load | NOT TESTED | TensorFlow and Keras were unavailable in the checked Python environments. |
| Development predictions | PASS | 14,188 unique IDs, ten probability columns, zero argmax mismatches, maximum probability-sum error 2.17e-7. |
| Test predictions | PASS | 3,545 unique IDs, ten probability columns, zero argmax mismatches, maximum probability-sum error 1.84e-7. |
| Metric consistency | PASS | Both headline metrics match their JSON values within 1e-12. All cells in the four raw and row-normalized confusion matrices match the predictions. |
| Split and contract compatibility | PASS | The supplied contract loader accepted the local dataset and manifests. All four input hashes match Sagar’s provenance. Saved prediction IDs and labels match the shared assignments and Kishore’s corresponding exports. |

Both CSVs contain `row_id`, `true_label`, `predicted_label`, and `prob_0` through `prob_9`. Probabilities are finite and within [0, 1]. No row ID occurs in both partitions.

| Evaluation | Rows | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Development OOF | 14,188 | 0.5368621370 | 0.5177920605 |
| Held-out test | 3,545 | 0.5633286319 | 0.5223538500 |

### Implementation and provenance

`train_baseline.py` loads the shared data contract and uses its outer, inner, and final-tuning fold definitions. `embeddings.py` uses ImageNet MobileNetV2 with the classification top removed, average pooling, and `trainable=False`. It converts RGB to float32, resizes to 224 × 224 when needed, and applies MobileNetV2 preprocessing once to map inputs to [-1, 1]. `model_config.json` identifies a full frozen baseline with no augmentation, a linear softmax head, and an 18-epoch development refit that agrees with `final_tuning_results.json`. `finetune_optional.py` has a separate CLI and unfreezes top backbone layers.

`timings.json` records a cache hit, 1,587.23 s of head training and 0.126 s of held-out inference. `gpu_embedding_extraction_timings.json` records a cache miss, 7.06 s of preprocessing, 38.93 s of extraction, 447.41 s of head training and 1.128 s of held-out inference. Both record zero fine-tuning. The supplied report matches `timings.json`. The additional timing file has no separate hardware record, and main provenance lists no detected GPUs. The additional filename alone cannot establish hardware or attribute those timings to the exported predictions.

Inference resolves `model_config.json` and `head_model.keras` in the same explicit `--model-dir`. From the project root its module entry point is `python -m Sagar.src.infer --model-dir Sagar/results/frozen_baseline_full --image /path/to/image`. Source and saved-model structure were checked. Numerical reproduction of the saved predictions from weights was outside the executed checks.

### Shared Galaxy10 contract

| Input (relative to project root) | Local SHA-256, also recorded by Sagar |
|---|---|
| `Kishore/data/raw/Galaxy10_DECals.h5` | `19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571` |
| `Kishore/results/final/dataset.json` | `e9d22ccd6a948e25330ade871255a0b1d3770d246cdc7dc791454b284fe5e69e` |
| `Kishore/results/final/splits.json` | `3019bad56d5014490366510d2956c82d1da74effb2ac0dd025088cb3c08ca884` |
| `Kishore/results/final/split_assignments.csv` | `47e842d5a31837db8fe43f852b3d5f83e3ffc9ee79398b7a528bb88bf93a3de8` |

The official Galaxy10 dataset manifest, its class mapping, and the Random Forest model metadata identify `Kishore/results/final/` as the matching experiment. The 10-class mapping agrees with Sagar’s configuration. The contract has 17,733 eligible images, five outer folds with three inner folds each, and 5 final-tuning folds. Sagar’s outer fold summary sizes match the assignments. The Sagar OOF CSV supplies row IDs and labels, with fold membership available through the shared assignments. Kishore’s OOF fold column matches those assignments. The Barchi/DR7 work under `Kishore/paper_replication/` was outside this integration.

The baseline CLI accepts `--project-root`, `--h5-path`, `--dataset-json`, `--splits-json`, and `--assignments-csv`. The three manifest defaults already resolve under the project root. The HDF5 default points to the project root, where the file is absent. A future baseline invocation from that root can pass `--h5-path Kishore/data/raw/Galaxy10_DECals.h5`. The separate optional fine-tuning CLI uses the project-root defaults and has no individual dataset override.

## E. Missing dependencies or blockers

- The system Python and Kishore virtual environment lack TensorFlow and Keras. The latter has h5py, NumPy, pandas, scikit-learn, Matplotlib, and Pillow.
- ImageNet MobileNetV2 weights were absent from the checked local Keras cache paths. The original embedding cache and metadata are absent from `Sagar/cache/embeddings/` and the path recorded in provenance.
- The HDF5 dataset exists under `Kishore/data/raw/`, and the three shared manifests, model configuration, and provenance are present. No expected handoff result file is missing.
- The absent runtime, weights, and cache limit model execution and numerical reproduction. They do not prevent verification of saved predictions.

## F. Readiness for paired evaluation

The saved predictions are ready for paired Random Forest and MobileNetV2 evaluation by `row_id`. Both partitions have exact row and true-label matches across implementations. Kishore stores the true label in `label` and the Random Forest prediction in `forest`. Sagar stores them in `true_label` and `predicted_label`. No remaining blocker was found for a comparison using the saved predictions.
