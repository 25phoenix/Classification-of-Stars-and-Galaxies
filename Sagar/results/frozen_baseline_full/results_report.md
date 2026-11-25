# MobileNetV2 Transfer Learning Results Report (Galaxy10 DECaLS)


## Experimental Scope
- **Task:** Ten-class galaxy morphology classification on `Galaxy10_DECals.h5` using the exact shared eligible population and nested splits saved by Kishore.
- **Comparative framing:** Inspired by Barchi et al. (2020), not an exact reproduction (different dataset, backbone, and feature set).
- **No star classification claims:** Galaxy10 DECaLS has no verified stellar subtype labels. Softmax outputs assume a galaxy image from these 10 classes and are not a star detector or calibrated confidence estimate.

## Headline Measured Performance
- **Pooled Outer Out-of-Fold (Development):** Accuracy = `0.5369`, Macro-F1 = `0.5178` (n = 14188)
- **Held-Out Test Partition:** Accuracy = `0.5633`, Macro-F1 = `0.5224` (n = 3545)
- **Selected Final Head Hyperparameters (development-only):** `{"hidden_units": 0, "dropout": 0.2, "learning_rate": 0.001, "l2_reg": 0.0001, "use_class_weight": false}`
- **Refit Epochs (median of final-tuning early-stopped epochs):** `18`

## Measured Strengths (Top 3 Test Classes by F1)
- Class 8 (Edge-on Galaxies without Bulge): F1 = `0.7986`, Precision = `0.7687`, Recall = `0.8310`, Support = `284`
- Class 9 (Edge-on Galaxies with Bulge): F1 = `0.7474`, Precision = `0.7232`, Recall = `0.7733`, Support = `375`
- Class 2 (Round Smooth Galaxies): F1 = `0.6763`, Precision = `0.6482`, Recall = `0.7070`, Support = `529`

## Measured Failure Modes (Lowest 3 Test Classes by F1)
- Class 5 (Barred Spiral Galaxies): F1 = `0.4552`, Precision = `0.5699`, Recall = `0.3790`, Support = `409`
- Class 4 (Cigar Shaped Smooth Galaxies): F1 = `0.3462`, Precision = `0.4615`, Recall = `0.2769`, Support = `65`
- Class 0 (Disturbed Galaxies): F1 = `0.1566`, Precision = `0.3438`, Recall = `0.1014`, Support = `217`

## Class Imbalance & Limitations
- **Test support range:** smallest class is Class 4 (Cigar Shaped Smooth Galaxies, n=65); largest class is Class 2 (Round Smooth Galaxies, n=529).
- **Frozen features limitation:** Because one deterministic ImageNet-pretrained MobileNetV2 embedding is cached per image, no image augmentation is active during head training, and convolutional filters are not adapted to astronomical bandpass structure.
- **Visual ambiguity:** Morphologically adjacent categories (e.g., tight vs. loose spirals, disturbed vs. merging, in-between smooth vs. round smooth) account for much of the confusion visible in `test_confusion_matrices.png`.

## Measured Timings (seconds)
```json
{
  "preprocessing_seconds": 0.0,
  "embedding_extraction_seconds": 0.0,
  "embedding_cache_hit": true,
  "head_training_seconds": 1587.2275123939908,
  "optional_finetuning_seconds": 0.0,
  "total_inference_seconds": 0.6868565960030537,
  "test_inference_seconds": 0.12604760599788278
}
```
