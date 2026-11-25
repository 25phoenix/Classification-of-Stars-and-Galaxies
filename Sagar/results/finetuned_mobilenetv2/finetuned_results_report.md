# Optional Two-Stage Fine-Tuned MobileNetV2 Report (Galaxy10 DECaLS)

- **Held-Out Test Accuracy:** `0.7622` (vs. Frozen Baseline `0.5633`)
- **Held-Out Test Macro-F1:** `0.7342` (vs. Frozen Baseline `0.5224`)
- **Protocol:** 5 warmup epochs (`lr=0.001`) + 15 fine-tuning epochs (`lr=0.0001`, top 40 layers, BN frozen, training-only flip/rot/shift augmentation).
- **Total Fine-Tuning Time:** `1777.3s` | **Test Inference:** `8.85s`

## Top 3 Test Classes by F1
- Class 8 (Edge-on Galaxies without Bulge): F1=`0.8998`, P=`0.8982`, R=`0.9014`, n=`284`
- Class 9 (Edge-on Galaxies with Bulge): F1=`0.8903`, P=`0.8625`, R=`0.9200`, n=`375`
- Class 2 (Round Smooth Galaxies): F1=`0.8623`, P=`0.7915`, R=`0.9471`, n=`529`

## Bottom 3 Test Classes by F1
- Class 0 (Disturbed Galaxies): F1=`0.3760`, P=`0.4600`, R=`0.3180`, n=`217`
- Class 7 (Unbarred Loose Spiral Galaxies): F1=`0.6145`, P=`0.6318`, R=`0.5981`, n=`525`
- Class 4 (Cigar Shaped Smooth Galaxies): F1=`0.6515`, P=`0.6418`, R=`0.6615`, n=`65`
