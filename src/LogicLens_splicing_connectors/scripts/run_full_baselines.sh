#!/usr/bin/env bash
set -e
python src/data_prep/prepare_anomalib_dataset.py
python src/train/train_patchcore.py
python src/train/train_efficientad.py
python src/train/train_stfpm.py
python src/train/train_reverse_distillation_full_50.py
