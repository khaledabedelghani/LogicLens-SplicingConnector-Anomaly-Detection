# LogicLens Training Plan

## Task

Unsupervised visual anomaly detection and localization for industrial splicing connectors.

## Dataset

Dataset: MVTec LOCO AD / splicing_connectors

Total images: 732

| Split | Label | Count |
|---|---|---:|
| Train | good | 360 |
| Validation | good | 60 |
| Test | good | 119 |
| Test | logical_anomalies | 108 |
| Test | structural_anomalies | 85 |

Image resolution: 1700 x 850 RGB.

Masks exist for all anomaly test images.

## Dataset Decisions

- Use original dataset only.
- Do not create augmented dataset.
- Do not move anomalies into training.
- Do not use masks for training.
- Masks are used only for localization evaluation.
- Do not crop.
- ROI audit showed crop gain is small.
- Use full image.

## Preprocessing

Primary input size:
- 1024 x 512

Fallback if GPU memory fails:
- 768 x 384

Never use:
- 224 x 224
- square resize
- crop

## Models

### PatchCore

Role: strong baseline.

Settings:
- Train only on train/good.
- No augmentation.
- Full image resize to 1024 x 512.
- Prefer wide_resnet50_2 backbone if supported.

### EfficientAD

Role: fast / real-time-style model.

Settings:
- Train only on train/good.
- Full image resize to 1024 x 512.
- Light augmentation only if safely supported.

- No external augmentation.
- This decision reduces overfitting risk and avoids hiding tiny defects or altering color/location-based anomalies.

Forbidden augmentation:
- crop
- flip
- random erasing
- cutout
- strong color jitter
- mixup/cutmix

## Metrics

Image-level:
- AUROC
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Localization:
- Pixel AUROC
- AUPRO / PRO if available
- IoU as secondary metric

## Evaluation Breakdown

Report:
- Overall
- Logical anomalies only
- Structural anomalies only

## Important Notes

Structural anomalies are tiny, so use high resolution.

Some logical masks are global/full-image, so IoU is not always the best localization metric.

Main localization metrics should be Pixel AUROC and AUPRO.
IoU is secondary.