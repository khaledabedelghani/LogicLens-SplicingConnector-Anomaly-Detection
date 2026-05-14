# LogicLens: Splicing Connector Anomaly Detection

Explainable visual anomaly detection for MVTec LOCO splicing connectors using EfficientAD, PatchCore, and PaDiM.

## Project Goal

Detect normal vs anomalous splicing connector images and visualize suspicious regions using anomaly heatmaps.

## Methods

- EfficientAD
- PatchCore
- PaDiM
- Threshold tuning
- Failure-case analysis
- mAP / localization evaluation from anomaly maps

## Dataset

MVTec LOCO AD - Splicing Connectors category.

Dataset and trained weights are not included in this repository because of size and licensing.
Please download the dataset from the official MVTec website.

## Repository Structure

src/
notes/
results/
README.md
requirements.txt

## How to Run

Install dependencies:

pip install -r requirements.txt

Then run the training or evaluation scripts from the src folder.

## Important Note

Large datasets, generated outputs, prediction maps, and model checkpoints are excluded from GitHub using .gitignore.