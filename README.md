# LogicLens: Explainable Anomaly Detection for Splicing Connectors

LogicLens is a Computer Vision final project for **unsupervised anomaly detection and localization** on industrial splicing connector images. The project detects whether a connector image is **normal** or **anomalous**, produces anomaly heatmaps to highlight suspicious regions, and compares several anomaly detection approaches under the same evaluation setup.

This repository focuses on the **MVTec LOCO AD `splicing_connectors`** category and studies both:

- **Structural anomalies**: local physical defects such as damaged, broken, or contaminated components.
- **Logical anomalies**: configuration errors such as missing, misplaced, flipped, or incorrectly arranged components.

The goal is not only to report metrics, but also to understand the strengths and weaknesses of each method, especially when comparing image-level detection and pixel-level localization.

---

## 1. Project Description

Industrial visual inspection requires models that can detect defects even when only normal samples are available for training. This project follows an unsupervised anomaly detection setup:

- Training uses only normal images from `train/good`.
- Validation uses normal images from `validation/good`.
- Testing uses normal images and anomalous images from `test/good`, `test/logical_anomalies`, and `test/structural_anomalies`.
- Ground-truth masks are used only during evaluation, not during training.

The system produces:

1. Image-level anomaly scores.
2. Binary normal/anomalous predictions after thresholding.
3. Pixel-level anomaly maps.
4. Heatmap visualizations.
5. Quantitative comparison between methods.
6. Failure-case analysis for logical and structural anomalies.

---

## 2. Main Contributions

This project includes:

- A complete anomaly detection pipeline for MVTec LOCO splicing connectors.
- Multiple model families:
  - EfficientAD
  - PatchCore
  - PaDiM
  - STFPM
  - Reverse Distillation
  - LogicLens ROI + Patch/DeiT fusion evaluation
- Dataset preparation scripts for full-image, ROI-based, object-focused, and augmented variants.
- Image-level and pixel-level evaluation.
- Threshold tuning and subset-level analysis.
- Anomaly map saving and mAP-style localization evaluation.
- Organized GitHub-ready code structure with setup and reproduction instructions.

---

## 3. Dataset

### Dataset Name

**MVTec LOCO AD - Splicing Connectors**

Official dataset page:

```text
https://www.mvtec.com/company/research/datasets/mvtec-loco
```

### Dataset Type

The dataset is designed for unsupervised anomaly detection and localization.

The `splicing_connectors` category contains normal samples, logical anomalies, structural anomalies, and pixel-level ground-truth masks for anomalous test images.

### Expected Original Dataset Layout

Place the original `splicing_connectors` folder in your local project directory using this structure:

```text
splicing_connectors/
├── train/
│   └── good/
├── validation/
│   └── good/
├── test/
│   ├── good/
│   ├── logical_anomalies/
│   └── structural_anomalies/
├── ground_truth/
│   ├── logical_anomalies/
│   └── structural_anomalies/
├── defects_config.json
├── license.txt
└── readme.txt
```

### Dataset Availability

The dataset is **not included** in this GitHub repository because of size and licensing restrictions.  
To reproduce the experiments, download the dataset from the official MVTec website and place it in the expected local path.

---

## 4. Repository Structure

The repository is organized as follows:

```text
LogicLens-SplicingConnector-Anomaly-Detection/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── LogicLens_splicing_connectors/
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── docs/
    │   │   ├── PROJECT_AUDIT.md
    │   │   └── PY_COMPILE_RESULT.txt
    │   ├── scripts/
    │   │   ├── run_full_baselines.sh
    │   │   ├── run_object_focus.sh
    │   │   └── run_roi_fusion.sh
    │   ├── src/
    │   │   ├── data_prep/
    │   │   │   ├── prepare_anomalib_dataset.py
    │   │   │   ├── prepare_object_focus_dataset.py
    │   │   │   ├── prepare_object_focus_augmented_dataset.py
    │   │   │   ├── prepare_object_focus_aug_1080_fast.py
    │   │   │   └── prepare_roi_canonical_dataset.py
    │   │   ├── train/
    │   │   │   ├── train_efficientad.py
    │   │   │   ├── train_efficientad_object_focus.py
    │   │   │   ├── train_patchcore.py
    │   │   │   ├── train_reverse_distillation_full_50.py
    │   │   │   ├── train_reverse_distillation_object_focus.py
    │   │   │   ├── train_reverse_distillation_object_focus_aug_70.py
    │   │   │   └── train_stfpm.py
    │   │   ├── eval/
    │   │   │   ├── evaluate_results.py
    │   │   │   ├── logiclens_logic_eval.py
    │   │   │   ├── logiclens_cable_inspection.py
    │   │   │   ├── logiclens_embedding_knn_eval.py
    │   │   │   ├── logiclens_vit_embedding_eval.py
    │   │   │   └── run_roi_patch_deit_fusion.py
    │   │   └── diagnostics/
    │   │       └── test_efficientad_object_focus_api.py
    │   └── training_package/
    │       ├── prepare_roi_canonical_dataset.py
    │       └── run_roi_patch_deit_fusion.py
    ├── src_all_training_efficient_and_padim/
    │   ├── run_efficientad.py
    │   ├── run_padim.py
    │   ├── test_checkpoint.py
    │   ├── save_anomaly_maps.py
    │   ├── compute_map_from_maps.py
    │   ├── sweep_thresholds.py
    │   ├── threshold_scores.py
    │   ├── find_best_threshold.py
    │   ├── f1overall.py
    │   ├── make_roi_dataset.py
    │   ├── make_roi_bg_dataset.py
    │   ├── make_train_aug_dataset.py
    │   ├── make_roi_padim_dataset.py
    │   └── make_train_aug_padim_dataset.py
    └── src_pathcore/
        ├── dataset.py
        ├── train_patchcore.py
        ├── patchcore_final.py
        ├── patchcore_roi_final.py
        ├── patchcore_heatmaps.py
        ├── patchcore_split_eval.py
        └── visual_check.py
```

---

## 5. What Is Excluded from GitHub

Large files and generated outputs are excluded using `.gitignore`.

The following are intentionally not pushed:

```text
data/
datasets/
data_roi/
data_roi_bg/
data_roi_bg_aug/
data_roi_padim/
data_roi_padim_aug/
dev_validation/
outputs/
training_outputs/
training_workdir/
pred_maps/
results_pred_maps/
pre_trained/
*.ckpt
*.pt
*.pth
*.onnx
*.zip
.venv/
.env
```

This keeps the repository clean, lightweight, and reproducible.

---

## 6. Installation and Setup

### 6.1 Clone the Repository

```bash
git clone https://github.com/khaledabedelghani/LogicLens-SplicingConnector-Anomaly-Detection.git
cd LogicLens-SplicingConnector-Anomaly-Detection
```

### 6.2 Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS / Cloud

```bash
python -m venv .venv
source .venv/bin/activate
```

### 6.3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If CUDA is required, install the correct PyTorch version for your GPU from the official PyTorch installation page.

---

## 7. Requirements

Main dependencies:

```text
numpy
pandas
pillow
opencv-python
matplotlib
scikit-learn
tqdm
albumentations
einops
kornia
lightning
anomalib==2.1.0
timm
```

The exact dependency list is provided in:

```text
requirements.txt
```

---

## 8. Dataset Preparation

Before training, download the MVTec LOCO AD dataset and place the `splicing_connectors` category in the project folder.

Example local path:

```text
D:\project computer vision\data\splicing_connectors
```

or, inside the cloned repository:

```text
data/splicing_connectors/
```

### 8.1 Prepare Full-Image Anomalib Dataset

From inside:

```text
src/LogicLens_splicing_connectors/
```

run:

```bash
python src/data_prep/prepare_anomalib_dataset.py
```

This creates an Anomalib-compatible dataset under:

```text
training_workdir/anomalib_splicing/
```

### 8.2 Prepare Object-Focus Dataset

```bash
python src/data_prep/prepare_object_focus_dataset.py
```

This creates:

```text
training_workdir/anomalib_splicing_object_focus/
```

### 8.3 Prepare Augmented Object-Focus Dataset

```bash
python src/data_prep/prepare_object_focus_aug_1080_fast.py
```

This creates:

```text
training_workdir/anomalib_splicing_object_focus_aug_1080/
```

### 8.4 Prepare ROI Canonical Dataset

```bash
python src/data_prep/prepare_roi_canonical_dataset.py
```

This creates:

```text
training_workdir/logiclens_roi448/
```

---

## 9. Training Instructions

### 9.1 Train Full-Image Baselines

From inside:

```text
src/LogicLens_splicing_connectors/
```

run:

```bash
python src/train/train_patchcore.py
python src/train/train_efficientad.py
python src/train/train_stfpm.py
python src/train/train_reverse_distillation_full_50.py
```

These scripts train standard anomaly detection baselines on the prepared full-image dataset.

### 9.2 Train Object-Focus Models

```bash
python src/train/train_efficientad_object_focus.py
python src/train/train_reverse_distillation_object_focus.py
```

### 9.3 Train Augmented Object-Focus Reverse Distillation

```bash
python src/train/train_reverse_distillation_object_focus_aug_70.py
```

### 9.4 Run ROI Patch/DeiT Fusion

```bash
python src/data_prep/prepare_roi_canonical_dataset.py
python src/eval/run_roi_patch_deit_fusion.py
```

or use the provided runner:

```bash
bash scripts/run_roi_fusion.sh
```

Outputs are saved under:

```text
training_outputs/
```

---

## 10. EfficientAD and PaDiM Legacy Experiment Scripts

Additional experiment scripts are available under:

```text
src/src_all_training_efficient_and_padim/
```

This folder contains scripts used for:

- EfficientAD training
- PaDiM evaluation
- ROI dataset creation
- Black-background ROI dataset creation
- Train-only augmentation
- Threshold sweeping
- Checkpoint testing
- Anomaly map saving
- mAP-style evaluation from anomaly maps

Example commands:

```bash
cd src/src_all_training_efficient_and_padim
python run_efficientad.py
python run_padim.py
python test_checkpoint.py
python save_anomaly_maps.py
python compute_map_from_maps.py
```

Before running these scripts, update local dataset and checkpoint paths inside the files if needed.

---

## 11. PatchCore Scripts

PatchCore scripts are available under:

```text
src/src_pathcore/
```

Example commands:

```bash
cd src/src_pathcore
python train_patchcore.py
python patchcore_final.py
python patchcore_roi_final.py
python patchcore_heatmaps.py
python patchcore_split_eval.py
```

PatchCore is used as a strong baseline and for comparison with EfficientAD, PaDiM, and LogicLens-style evaluation.

---

## 12. How to Run Inference / Evaluation

### Option A: Evaluate Trained Anomalib Models

After training, outputs are stored in:

```text
training_outputs/
```

Use the evaluation scripts:

```bash
python src/eval/evaluate_results.py
python src/eval/logiclens_logic_eval.py
python src/eval/logiclens_cable_inspection.py
```

### Option B: Run EfficientAD Checkpoint Evaluation

From:

```text
src/src_all_training_efficient_and_padim/
```

run:

```bash
python test_checkpoint.py
```

Before running, update:

```python
CKPT_PATH = "path/to/model.ckpt"
root = "path/to/prepared/dataset"
```

The script reports metrics such as:

```text
image_AUROC
image_F1
pixel_AUROC
pixel_F1
```

### Option C: Generate Anomaly Heatmaps

```bash
python save_anomaly_maps.py
```

This saves predicted anomaly maps under a folder such as:

```text
pred_maps/
```

### Option D: Compute mAP-Style Localization Metrics

After anomaly maps are generated:

```bash
python compute_map_from_maps.py
```

This computes localization-style metrics such as:

```text
mAP@50
mAP@50:95
per-image IoU
```

---

## 13. Evaluation Metrics

The project reports both image-level and pixel-level metrics.

### Image-Level Metrics

- Image AUROC
- Image F1-score
- Accuracy
- Precision
- Recall
- Threshold-based prediction quality

### Pixel-Level / Localization Metrics

- Pixel AUROC
- Pixel F1-score
- IoU
- mAP@50
- mAP@50:95
- Qualitative heatmaps

### Subset-Level Analysis

Results are analyzed separately for:

- Overall test set
- Good images
- Logical anomalies
- Structural anomalies

This is important because logical anomalies are often more difficult than structural anomalies.

---

## 14. Results

Final numeric results should be inserted after running the evaluation scripts.

Example result table:

| Method | Setup | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 |
|---|---|---:|---:|---:|---:|
| EfficientAD | Baseline | TBD | TBD | TBD | TBD |
| EfficientAD | ROI / black background | TBD | TBD | TBD | TBD |
| PatchCore | Best setup | TBD | TBD | TBD | TBD |
| PaDiM | ROI setup | TBD | TBD | TBD | TBD |
| LogicLens | ROI Patch/DeiT fusion | TBD | TBD | TBD | TBD |

Localization-style evaluation:

| Method | mAP@50 | mAP@50:95 |
|---|---:|---:|
| EfficientAD | TBD | TBD |
| PatchCore | TBD | TBD |
| PaDiM | TBD | TBD |
| LogicLens Fusion | TBD | TBD |

---

## 15. Failure Case Analysis

The project includes qualitative analysis of difficult cases.

Failure analysis focuses on questions such as:

- Does the model fail more on logical anomalies or structural anomalies?
- Does the heatmap focus on the correct cable or connector?
- Are small defects missed?
- Does background texture affect the prediction?
- Is the threshold too strict or too permissive?
- Do ROI preprocessing and object-focus transformations improve localization?

This analysis helps explain the model behavior beyond numerical metrics.

---

## 16. Weights

Model weights are not included in this GitHub repository because checkpoint files are large.

### Option 1: Download Pretrained Weights

If weights are provided externally, download them from:

```text
ADD_GOOGLE_DRIVE_OR_ONEDRIVE_LINK_HERE
```

Place downloaded weights under:

```text
training_outputs/
```

or under the path expected by the evaluation script.

### Option 2: Reproduce Training

To reproduce the weights:

1. Download the MVTec LOCO AD dataset.
2. Place the `splicing_connectors` folder in the expected dataset path.
3. Install requirements.
4. Prepare the dataset:

```bash
python src/data_prep/prepare_anomalib_dataset.py
```

5. Train the desired model:

```bash
python src/train/train_efficientad.py
python src/train/train_patchcore.py
```

6. Evaluate the trained model:

```bash
python src/eval/evaluate_results.py
```

For ROI fusion:

```bash
python src/data_prep/prepare_roi_canonical_dataset.py
python src/eval/run_roi_patch_deit_fusion.py
```

---

## 17. Reproducibility Notes

To reproduce the experiments correctly:

- Do not train on anomalous test images.
- Do not use ground-truth masks during training.
- Do not apply augmentation to test images.
- Keep the same train / validation / test split.
- Use the same preprocessing for training and evaluation.
- Keep large datasets, checkpoints, and outputs outside GitHub.
- Report metrics separately for logical and structural anomalies.
- Save final results as CSV files for reporting.

---

## 18. Academic Integrity and Attribution

This project uses the MVTec LOCO AD dataset for academic purposes. The dataset is not redistributed in this repository.

Please cite:

```text
MVTec LOCO AD Dataset
https://www.mvtec.com/company/research/datasets/mvtec-loco
```

Main tools and libraries used:

- PyTorch
- Anomalib
- Lightning
- OpenCV
- scikit-learn
- Albumentations
- timm

---

## 19. Authors

```text
Khaled Abed el ghani
Ali abou chakra
```

Course:

```text
Computer Vision Final Project
Saint Joseph University
Spring 2026
```

---

## 20. License

This repository is intended for academic and educational use.

The dataset is not included and must be downloaded from the official source. Users must respect the dataset license and cite the dataset properly.
