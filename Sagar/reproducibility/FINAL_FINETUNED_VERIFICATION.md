# Final fine-tuned MobileNetV2 verification

**Integration and structural audit passed. Independent Keras load and held-out inference were blocked by the missing runtime.** The published predictions were audited separately. Their verified metrics describe the supplied CSV.

## Archive and source provenance

- Original archive `Sagar/Sagar_FineTuned_Model_And_Code_Archive.zip` passed its ZIP CRC check. SHA-256 `576f6eec43e094e1a5e87d19f52e337a210a9822b24801d37941a0ed93d83072`.
- Extracted all 13 file members, preserving paths below the archive's top-level directory, into [finetuned_v1](finetuned_v1). All extracted bytes matched their archive members.
- The versioned native Keras file and the copy installed at `Sagar/results/finetuned_mobilenetv2/finetuned_model.keras` share SHA-256 `142efcacd36ba4b95d0956abf3a43d326bc6195ffe5bbbb08c7e1bbfb6efd0a1`. The native container's configuration, metadata, and weights passed CRC checks.
- All nine archived Python files compiled in memory. Eight are byte-identical to the earlier source. The other is `finetune_optional.py`, which differs from the original optional fine-tuning module. Both versions are preserved. The archived run JSON is byte-identical to the Phase 4 addendum configuration.
- [Provenance](verification/provenance.json) contains archive member and model component hashes, dataset and contract hashes, and reference prediction and metric file hashes. [Source comparison](verification/source_comparison.csv) gives exact per-module hashes.

## Saved model structure

| Check | Result |
|---|---|
| Native Keras structure | Keras 3.15.1 metadata, readable graph JSON, readable HDF5 weight and optimizer data |
| Input | Float32 preprocessed RGB, shape `(None, 224, 224, 3)` |
| Backbone | Serialized MobileNetV2, 155 layers including input and global average pooling |
| Trainability | Top 40 layers targeted. Of those, 27 non-BatchNormalization layers have trainable flags, including 13 Conv2D or DepthwiseConv2D layers. All 52 BatchNormalization layers are frozen. |
| Backbone call | Serialized with `training=False` |
| Augmentation | Horizontal and vertical flips, rotation of ±0.25 turns, translation of ±5% on each spatial axis, reflection padding for rotation and translation |
| Head | 1,280 pooled features, dropout 0.3, ten-unit float32 softmax, L2 coefficient 0.0001 |
| Weights | 262 model-weight tensors with 2,270,794 scalar values. Dense kernel and bias shapes are `(1280, 10)` and `(10,)`. All model and optimizer tensors inspected contain finite values. |
| Class order | Source and published metric metadata agree on the official Galaxy10 index order from 0 to 9. Class names are external to the Keras graph. |
| Normalization | The archived data pipeline casts uint8 RGB to float32, resizes 256 × 256 to 224 × 224 with bilinear interpolation, and applies MobileNetV2 normalization to [−1, 1] before the model. |

## Training snapshot and configuration

The archived script first fits the classification head for five epochs with the ImageNet backbone frozen and Adam at 0.001. It then unfreezes the configured layers and uses Adam at 0.0001, retaining BatchNormalization in inference mode. It selects the fine-tuning duration by minimum validation loss on the first saved final-tuning fold. The validation stage uses learning-rate reduction and early stopping. The reported minimum occurs at fine-tuning epoch 15, with validation loss 0.7079075 and accuracy 0.7594223. It builds a fresh model for the full 14,188-row development refit, with five warmup epochs followed by the selected duration under cosine decay. Class weights are optional and were off for the reported run. Code, metadata, and graph agree on this high-level procedure.

The earlier optional script used a single training stage with a smaller default unfreeze window, class weights, and no separate warmup or cosine refit. The original frozen baseline uses a separate head on cached embeddings and keeps the backbone frozen throughout. The fine-tuned run used one development validation fold for epoch selection.

Two serialized details limit claims about the exact run history:

- The addendum reports mixed float16 training. The saved backbone's layer policies and the saved head and dropout policies are float32. The saved optimizer is serialized as Adam. The archive provides no export log explaining the difference between the recorded training policy and the saved policies.
- The saved Adam schedule records an initial rate of 0.0001, alpha 0.1, and 13,305 decay steps, consistent with 15 × ceil(14,188 ÷ 16). The serialized optimizer iteration counter is 13,320. The archive does not establish the cause of the 15-step difference.

See the [serialized training review](verification/serialized_training_review.json). The verification performs no retraining and cannot certify training history from the saved state.

## Held-out prediction checks

The local HDF5 is `Kishore/data/raw/Galaxy10_DECals.h5`. The archived contract loader verified its SHA-256, HDF5 labels, and saved development/test split using the three manifests in `Kishore/results/final/`. All four hashes match the established frozen-run provenance. Exactly 3,545 unique published prediction rows joined the original locked test IDs and matched their true labels. All ten probability columns were finite, summed to one within tolerance, and produced the saved class labels by argmax.

| Check | Result |
|---|---|
| Published prediction rows aligned to locked test | 3,545 |
| Accuracy recomputed from published CSV | 0.7622002821, or 76.22% |
| Macro-F1 recomputed from published CSV | 0.7342220992 |
| `tf.keras.models.load_model(..., compile=False)` | NOT TESTED |
| Inference-mode forward pass and full test prediction | BLOCKED |
| Regenerated rows, matching classes, mismatch IDs, probability error statistics | Unavailable |
| Regenerated accuracy, Macro-F1, and material equivalence | Unavailable |

The archive README reports an eight-image reload check. That supplier check is separate from independent runtime and full-test verification. No regenerated CSV exists for this run.

## Runtime requirements and rerun

The checked WSL Python 3.10 and 3.12 interpreters and Kishore virtual environment lack TensorFlow and Keras. The checked Windows Python 3.12 package locations also lack them. No compatible installed runtime was located, and no packages were installed. The supplied [environment freeze](finetuned_v1/environment/requirements_freeze.txt) records Python 3.12.16, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.5, h5py 3.14.0, pandas 2.3.3, and scikit-learn 1.9.1, along with the original CUDA training stack. A compatible isolated environment with TensorFlow, Keras, and the array, HDF5, and metric packages is needed to exercise the verifier. A CPU-capable build can be used for inference if it supports the saved model operations. Reproducing the original training hardware is a separate requirement.

From the project root, run the [verifier](verify_finetuned.py) with a compatible interpreter:

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/compatible/python \
  Sagar/reproducibility/verify_finetuned.py \
  --h5-path Kishore/data/raw/Galaxy10_DECals.h5 \
  --dataset-json Kishore/results/final/dataset.json \
  --splits-json Kishore/results/final/splits.json \
  --assignments-csv Kishore/results/final/split_assignments.csv
```

It loads the versioned model with `compile=False`, performs a one-image smoke test, then uses inference mode on the full held-out split. It writes the separately regenerated CSV, mismatch rows, and comparison JSON under `Sagar/reproducibility/verification/`. The probability summaries aggregate absolute errors over all aligned class probabilities. The verifier leaves the original data, predictions, model artifacts, and comparison files untouched.

## Files created

- `finetuned_v1/README.md`, `finetuned_v1/model/finetuned_model.keras`, and the two supplied environment files.
- `finetuned_v1/code/Sagar/src/` with all nine original Python files.
- `Sagar/results/finetuned_mobilenetv2/finetuned_model.keras`.
- This report, `verify_finetuned.py`, and `verification/{provenance.json,verification.json,source_comparison.csv,serialized_training_review.json}`.
