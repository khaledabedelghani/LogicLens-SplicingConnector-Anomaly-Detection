from pathlib import Path
import csv
import torch

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import EfficientAd

torch.set_float32_matmul_precision("high")

PROJECT_ROOT = Path(r"D:\project computer vision")
SAMPLES_ROOT = PROJECT_ROOT / "dev_validation"
CSV_PATH = PROJECT_ROOT / "notes" / "threshold_scores_roi_bg.csv"

CKPT_PATH = PROJECT_ROOT / "outputs" / "efficientad_roi_bg" / "EfficientAd" / "MVTecLOCO" / "splicing_connectors" / "v1" / "weights" / "lightning" / "model.ckpt"
if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")


def to_scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0].item())
    return float(x)


def main():
    model = EfficientAd(
        imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
    )

    engine = Engine(
        default_root_dir=str(PROJECT_ROOT / "outputs" / "threshold_predict_roi"),
        accelerator="gpu",
        devices=1,
    )

    rows = []

    for group_name in ["good", "structural", "logical"]:
        folder = SAMPLES_ROOT / group_name

        dataset = PredictDataset(
            path=folder,
            image_size=(256, 256),
        )

        predictions = engine.predict(
            model=model,
            dataset=dataset,
            ckpt_path=str(CKPT_PATH),
        )

        for pred in predictions:
            raw_path = pred.image_path
            if isinstance(raw_path, (list, tuple)):
                image_name = Path(raw_path[0]).name
            else:
                image_name = Path(str(raw_path)).name

            score = to_scalar(pred.pred_score)

            gt = 0 if group_name == "good" else 1

            rows.append({
                "image_name": image_name,
                "group": group_name,
                "gt_label": gt,
                "score": score,
            })

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "group", "gt_label", "score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved scores to: {CSV_PATH}")


if __name__ == "__main__":
    main()