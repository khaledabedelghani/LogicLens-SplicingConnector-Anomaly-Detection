#!/usr/bin/env bash
set -e
python src/data_prep/prepare_object_focus_dataset.py
python src/train/train_efficientad_object_focus.py
python src/train/train_reverse_distillation_object_focus.py
python src/data_prep/prepare_object_focus_aug_1080_fast.py
python src/train/train_reverse_distillation_object_focus_aug_70.py
