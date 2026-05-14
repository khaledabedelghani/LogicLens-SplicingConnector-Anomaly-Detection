# Project Audit Notes

## What was cleaned

- Removed `.ipynb_checkpoints/` duplicate files from the GitHub-ready package.
- Removed the unknown empty file `sedd2oQOn`.
- Split scripts into `src/data_prep`, `src/train`, `src/eval`, and `src/diagnostics`.
- Added `.gitignore` to avoid pushing datasets, checkpoints, virtual environments, and outputs.
- Added `timm` to `requirements.txt` because the embedding evaluation scripts import it.

## Important fixes applied

1. `train_reverse_distillation_full_50.py`
   - Fixed printed labels and dataset name so it is clearly the full-image Reverse Distillation run.

2. `train_reverse_distillation_object_focus.py`
   - Fixed `DATA_ROOT` to use `training_workdir/anomalib_splicing_object_focus`.
   - Fixed `OUT_DIR` to use `training_outputs/reverse_distillation_object_focus_50`.
   - Fixed printed labels and Anomalib dataset name.

3. `train_reverse_distillation_object_focus_aug_70.py`
   - Changed output from `/tmp/...` to `training_outputs/...` so results stay inside the repo folder and are easier to collect.

## Warning about old instructions

The old `README_TRAINING_PLAN.md` and `DO_NOT_DO.txt` from the zip said: no object focus and no augmented dataset. That is outdated compared with the current code package, which includes object-focus preprocessing and augmented 1080-image training. For GitHub, use the new root `README.md` as the main source of truth.

## Added ROI canonical + Patch/DeiT fusion scripts

Added the two high-metric LogicLens scripts requested for the GitHub package:

- `src/data_prep/prepare_roi_canonical_dataset.py`
- `src/eval/run_roi_patch_deit_fusion.py`

Compatibility copies are also preserved under:

- `training_package/prepare_roi_canonical_dataset.py`
- `training_package/run_roi_patch_deit_fusion.py`

These scripts generate the canonical ROI dataset at `training_workdir/logiclens_roi448/` and save fusion scores/summaries under `training_outputs/logiclens_roi_patch_deit_fusion/`.
