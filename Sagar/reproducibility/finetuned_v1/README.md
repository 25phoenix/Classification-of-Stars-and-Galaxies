# Fine-Tuned MobileNetV2 — Model, Code & Reproducibility Archive

Archive of Sagar's final Two-Stage Fine-Tuned MobileNetV2 model (`0.7622` Test Accuracy, `0.7342` Test Macro-F1 across all `3,545` held-out Galaxy10 DECaLS test galaxies), exact executed source code, and environment specifications.

## 1. Archive Contents
- `model/finetuned_model.keras` — Complete saved Keras 3 model (2,270,794 parameters), containing training-only augmentation layers (inactive at inference), the MobileNetV2 backbone (top 27 non-BatchNorm layers fine-tuned), and the 10-class softmax head.
- `code/Sagar/src/` — Exact Python scripts executed (`finetune_optional.py` plus its imported contract/preprocessing/evaluation modules).
- `environment/requirements_freeze.txt` — Full `pip freeze` from the training virtual environment.
- `environment/finetune_config_and_timings.json` — Recorded hyperparameters, package versions, selected epochs, and timings.

## 2. Environment & Hardware
- **OS / Python:** Linux (`7.2.15-200.fc44.x86_64`), Python `3.12.16`
- **Core Stack:** TensorFlow `2.21.0`, Keras `3.15.1`, NumPy `2.5.5`
- **Training Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), `mixed_float16` policy (`float32` augmentation & softmax head), `TF_GPU_ALLOCATOR=cuda_malloc_async`.

Recreate the virtual environment:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements_freeze.txt
```

## 3. Loading the Saved Model & Reproducing the 3,545 Test Predictions
Load with `compile=False` for inference (augmentation layers are automatically inactive during `predict()`):

```python
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model("model/finetuned_model.keras", compile=False)

splits = pd.read_csv("split_assignments.csv")  # shared split contract
row_ids = splits[splits["partition"] == "test"].sort_values("row_id")["row_id"].to_numpy(dtype=np.int64)

probs_list = []
batch_size = 16
with h5py.File("Galaxy10_DECals.h5", "r") as f:
    images_ds = f["images"]
    for start in range(0, len(row_ids), batch_size):
        batch_ids = row_ids[start : start + batch_size]
        imgs = np.stack([images_ds[int(r)] for r in batch_ids], axis=0)
        x = tf.cast(imgs, tf.float32)
        x = tf.image.resize(x, [224, 224], method="bilinear")
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        probs_list.append(model.predict(x, verbose=0).astype(np.float32))

test_probs = np.concatenate(probs_list, axis=0)
pred_labels = np.argmax(test_probs, axis=1)
```

*(Reload verification on 8 held-out test galaxies: 8/8 exact predicted-class match, max probability diff `1.27e-03`.)*

## 4. Reproducing Two-Stage Fine-Tuning From Scratch
Place `code/Sagar/` inside `project_root/` alongside `Galaxy10_DECals.h5` and the shared split files, then run:

```bash
TF_GPU_ALLOCATOR=cuda_malloc_async TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONPATH=. \
  python -m Sagar.src.finetune_optional \
  --project-root . \
  --batch-size 16 \
  --warmup-epochs 5 \
  --epochs 15 \
  --unfreeze-top-layers 40 \
  --warmup-lr 0.001 \
  --lr 0.0001 \
  --seed 42
```

### Random Seeds & Deterministic Protocol
- **Base seed (`--seed` / `DEFAULT_SEED`):** `42`
- **Dev Fold 0 validation:** Model init seed = `42`; per-epoch shuffle seed = `42 + epoch_index` (best epoch = `15`, `val_acc = 0.7594`, `val_loss = 0.7079`).
- **Full development refit (14,188 rows):** Model init seed = `43`; per-epoch shuffle seed = `142 + epoch_index` (`5` warmup epochs at `lr=0.001` + `15` fine-tuning epochs with cosine decay to `1.0e-05`).
- **Held-out test evaluation (3,545 rows):** Fixed order (`shuffle=False`), training-only augmentation inactive.
