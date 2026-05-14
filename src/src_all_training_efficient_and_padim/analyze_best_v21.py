from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import EfficientAd

torch.set_float32_matmul_precision("high")

PROJECT_ROOT = Path(r"D:\project computer vision")
SAMPLES_ROOT = PROJECT_ROOT / "analysis_samples"
OUTPUT_ROOT = PROJECT_ROOT / "analysis_outputs"
CSV_PATH = PROJECT_ROOT / "notes" / "hard_cases.csv"

# CKPT_PATH = PROJECT_ROOT / "outputs" / "efficientad" / "best_v21_model.ckpt"
CKPT_PATH = PROJECT_ROOT / "outputs" / "efficientad_roi" / "best_efficientad_roi_v1.ckpt"
if not CKPT_PATH.exists():
    CKPT_PATH = PROJECT_ROOT / "outputs" / "efficientad" / "EfficientAd" / "MVTecLOCO" / "splicing_connectors" / "v21" / "weights" / "lightning" / "model.ckpt"


def to_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.reshape(-1)[0])
    try:
        return float(x)
    except Exception:
        return x


def to_numpy_image(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.array(x)

    # squeeze batch/channel if needed
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]

    # if CHW -> HWC for RGB-like arrays
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))

    # if single-channel image, squeeze last dim
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    return x


def expected_gt_from_group(group_name: str) -> str:
    if group_name == "good":
        return "normal"
    if group_name == "structural":
        return "structural"
    if group_name == "logical":
        return "logical"
    return "unknown"


def detection_ok(group_name: str, pred_label_name: str) -> str:
    if group_name == "good":
        return "yes" if pred_label_name == "normal" else "no"
    return "yes" if pred_label_name == "anomalous" else "no"


def save_heatmap(anomaly_map, out_path: Path):
    arr = to_numpy_image(anomaly_map)
    plt.figure(figsize=(5, 4))
    plt.imshow(arr, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = EfficientAd(
        imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
    )
    engine = Engine(
        default_root_dir=str(PROJECT_ROOT / "outputs" / "analysis_predict"),
        accelerator="gpu",
        devices=1,
    )

    rows = []

    for group_name in ["good", "structural", "logical"]:
        folder = SAMPLES_ROOT / group_name
        out_group = OUTPUT_ROOT / group_name
        out_group.mkdir(parents=True, exist_ok=True)

        dataset = PredictDataset(
            path=folder,
            image_size=(256, 256),
        )

        predictions = engine.predict(
            model=model,
            dataset=dataset,
            ckpt_path=str(CKPT_PATH),
        )

        if predictions is None:
            continue

        for pred in predictions:
            image_path = Path(str(pred.image_path))
            pred_label = to_scalar(pred.pred_label)
            pred_score = to_scalar(pred.pred_score)

            pred_label_name = "anomalous" if int(pred_label) == 1 else "normal"
            gt_name = expected_gt_from_group(group_name)
            det_ok = detection_ok(group_name, pred_label_name)

            heatmap_path = out_group / f"{image_path.stem}_heatmap.png"
            save_heatmap(pred.anomaly_map, heatmap_path)

            rows.append({
                "image_name": image_path.name,
                "group": group_name,
                "ground_truth": gt_name,
                "prediction": pred_label_name,
                "score": pred_score,
                "detection_correct": det_ok,
                "heatmap_correct": "",
                "note": "",
            })

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "group",
                "ground_truth",
                "prediction",
                "score",
                "detection_correct",
                "heatmap_correct",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. CSV saved to: {CSV_PATH}")
    print(f"Heatmaps saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()