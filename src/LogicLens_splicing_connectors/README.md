# LogicLens: Unsupervised Anomaly Detection for MVTec LOCO Splicing Connectors

This repository contains a GitHub-ready training and evaluation package for unsupervised anomaly detection and localization on the **MVTec LOCO AD `splicing_connectors`** category.

The project compares standard Anomalib models with object-focused preprocessing and LogicLens-style rule/embedding evaluation.

## Project goal

Detect and localize two anomaly families:

- **Structural anomalies**: physical/local visual defects, usually small and pixel-localized.
- **Logical anomalies**: wrong cable/connector arrangement or global configuration problems.

Training remains unsupervised: only `train/good` images are used for model training. Test anomalies and masks are used only for evaluation.

## Repository structure

```text
.
├── src/
│   ├── data_prep/        # Dataset conversion, resizing, object focus, augmentation
│   ├── train/            # Anomalib training scripts
│   ├── eval/             # LogicLens + embedding evaluation scripts
│   ├── diagnostics/      # API/checkpoint test scripts
│   └── ... includes ROI canonical dataset + Patch/DeiT fusion
├── docs/                 # Audit notes + archived old instructions
├── scripts/              # Command runners
├── training_workdir/     # Generated prepared datasets; ignored by Git
├── training_outputs/     # Generated model outputs; ignored by Git
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset expected layout

Put the original MVTec LOCO `splicing_connectors` files at the repository root:

```text
train/good/
validation/good/
test/good/
test/logical_anomalies/
test/structural_anomalies/
ground_truth/logical_anomalies/
ground_truth/structural_anomalies/
outputs/dataset_manifest.csv
```

The scripts prepare Anomalib-compatible folders under `training_workdir/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / cloud
# .venv\Scripts\activate       # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

## Recommended run order

### 1. Prepare full-image Anomalib dataset

```bash
python src/data_prep/prepare_anomalib_dataset.py
```

Creates:

```text
training_workdir/anomalib_splicing/
```

### 2. Train full-image baselines

```bash
python src/train/train_patchcore.py
python src/train/train_efficientad.py
python src/train/train_stfpm.py
python src/train/train_reverse_distillation_full_50.py
```

### 3. Prepare object-focus dataset

```bash
python src/data_prep/prepare_object_focus_dataset.py
```

Creates:

```text
training_workdir/anomalib_splicing_object_focus/
```

### 4. Train object-focus models

```bash
python src/train/train_efficientad_object_focus.py
python src/train/train_reverse_distillation_object_focus.py
```

### 5. Prepare augmented object-focus dataset

```bash
python src/data_prep/prepare_object_focus_aug_1080_fast.py
```

Creates:

```text
training_workdir/anomalib_splicing_object_focus_aug_1080/
```

### 6. Train augmented Reverse Distillation

```bash
python src/train/train_reverse_distillation_object_focus_aug_70.py
```

### 7. Prepare ROI canonical dataset, then run PatchMemory + DeiT fusion

These two scripts are important for the stronger LogicLens metrics because they create a canonical 448×448 ROI dataset and then fuse local patch anomaly scores with global DeiT embeddings.

```bash
python src/data_prep/prepare_roi_canonical_dataset.py
python src/eval/run_roi_patch_deit_fusion.py
```

Compatibility copies are also included under the original requested paths:

```bash
python training_package/prepare_roi_canonical_dataset.py
python training_package/run_roi_patch_deit_fusion.py
```

Or run both together:

```bash
bash scripts/run_roi_fusion.sh
```

Creates:

```text
training_workdir/logiclens_roi448/
training_outputs/logiclens_roi_patch_deit_fusion/
```

## Evaluation scripts

Logic/rule-based evaluation:

```bash
python src/eval/logiclens_logic_eval.py
python src/eval/logiclens_cable_inspection.py
```

Embedding-based evaluation:

```bash
python src/eval/logiclens_embedding_knn_eval.py
python src/eval/logiclens_vit_embedding_eval.py
python src/eval/run_roi_patch_deit_fusion.py
```

Placeholder output collector:

```bash
python src/eval/evaluate_results.py
```

## Model families included

| Family | Script | Input dataset |
|---|---|---|
| PatchCore | `src/train/train_patchcore.py` | full image |
| EfficientAD | `src/train/train_efficientad.py` | full image |
| STFPM | `src/train/train_stfpm.py` | full image |
| Reverse Distillation | `src/train/train_reverse_distillation_full_50.py` | full image |
| EfficientAD object-focus | `src/train/train_efficientad_object_focus.py` | object-focused |
| Reverse Distillation object-focus | `src/train/train_reverse_distillation_object_focus.py` | object-focused |
| Reverse Distillation augmented | `src/train/train_reverse_distillation_object_focus_aug_70.py` | object-focused + light augmentation |
| LogicLens rules | `src/eval/logiclens_logic_eval.py`, `src/eval/logiclens_cable_inspection.py` | full image |
| Embedding KNN / ViT | `src/eval/logiclens_embedding_knn_eval.py`, `src/eval/logiclens_vit_embedding_eval.py` | full/object-focused |
| ROI PatchMemory + DeiT fusion | `src/data_prep/prepare_roi_canonical_dataset.py`, `src/eval/run_roi_patch_deit_fusion.py` | ROI canonical 448×448 |

## GitHub upload steps

```bash
git init
git add README.md requirements.txt .gitignore src docs scripts training_package
git commit -m "Initial LogicLens anomaly detection package"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Do **not** push the original dataset, checkpoints, generated outputs, or virtual environment.
