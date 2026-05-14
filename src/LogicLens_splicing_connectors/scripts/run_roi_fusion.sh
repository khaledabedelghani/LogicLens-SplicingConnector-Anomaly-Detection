#!/usr/bin/env bash
set -euo pipefail

python src/data_prep/prepare_roi_canonical_dataset.py
python src/eval/run_roi_patch_deit_fusion.py
