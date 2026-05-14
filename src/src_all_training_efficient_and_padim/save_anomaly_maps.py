from pathlib import Path
import csv
import numpy as np
from PIL import Image
import torch

from anomalib.data import MVTecLOCO
from anomalib.models import EfficientAd
from anomalib.engine import Engine

PROJECT_ROOT = Path(r"D:\project computer vision")

CKPT_PATH = PROJECT_ROOT / "outputs" / "efficientad_roi_bg_aug" / "best_efficientad_roi_bg_aug_v0.ckpt"
DATA_ROOT = PROJECT_ROOT / "data_roi_bg_aug"
IMAGENET_DIR = PROJECT_ROOT / "datasets" / "imagenette2-320" / "train"
OUT_DIR = PROJECT_ROOT / "pred_maps"
SCORES_CSV = OUT_DIR / "pred_scores.csv"
TEST_ROOT = DATA_ROOT / "splicing_connectors" / "test"

OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.set_float32_matmul_precision("high")


def get_field(obj, *names):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def to_numpy_map(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    else:
        x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    return x.astype(np.float32)


def save_png_preview(arr: np.ndarray, out_path: Path):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    mn = float(arr.min())
    mx = float(arr.max())

    if mx > mn:
        vis = (arr - mn) / (mx - mn)
    else:
        vis = np.zeros_like(arr, dtype=np.float32)

    img = Image.fromarray((vis * 255).astype(np.uint8), mode="L")
    img.save(out_path)


def to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).tolist()

    arr = np.array(x)
    if arr.ndim == 0:
        return [arr.item()]
    return arr.reshape(-1).tolist()


def make_map_key_from_image_path(img_path: Path) -> str:
    rel = img_path.relative_to(TEST_ROOT).with_suffix("")
    return "__".join(rel.parts)


def main():
    datamodule = MVTecLOCO(
        root=str(DATA_ROOT),
        category="splicing_connectors",
        train_batch_size=1,
        eval_batch_size=4,
        num_workers=0,
    )

    model = EfficientAd(
        imagenet_dir=str(IMAGENET_DIR)
    )

    engine = Engine(
        default_root_dir=str(PROJECT_ROOT / "results_pred_maps"),
        accelerator="gpu",
        devices=1,
    )

    preds = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CKPT_PATH),
    )

    saved = 0
    score_rows = []

    for batch in preds:
        image_paths = get_field(batch, "image_path", "image_paths")
        anomaly_maps = get_field(batch, "anomaly_map", "anomaly_maps")
        pred_scores = get_field(batch, "pred_score", "pred_scores")

        if image_paths is None or anomaly_maps is None:
            continue

        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        if isinstance(anomaly_maps, torch.Tensor):
            if anomaly_maps.ndim == 2:
                anomaly_maps = anomaly_maps.unsqueeze(0)
            elif anomaly_maps.ndim == 4 and anomaly_maps.shape[1] == 1:
                anomaly_maps = anomaly_maps[:, 0]
        else:
            anomaly_maps = np.asarray(anomaly_maps)
            if anomaly_maps.ndim == 2:
                anomaly_maps = anomaly_maps[None, ...]

        pred_scores = to_list(pred_scores)
        if len(pred_scores) == 0:
            pred_scores = [0.0] * len(image_paths)
        if len(pred_scores) == 1 and len(image_paths) > 1:
            pred_scores = pred_scores * len(image_paths)

        for img_path, amap, score in zip(image_paths, anomaly_maps, pred_scores):
            img_path = Path(img_path)
            arr = to_numpy_map(amap)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            key = make_map_key_from_image_path(img_path)

            png_path = OUT_DIR / f"{key}.png"
            npy_path = OUT_DIR / f"{key}.npy"

            save_png_preview(arr, png_path)
            np.save(npy_path, arr)

            score_rows.append({
                "key": key,
                "image_path": str(img_path),
                "pred_score": float(score),
                "png_path": str(png_path),
                "npy_path": str(npy_path),
            })

            saved += 1

    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["key", "image_path", "pred_score", "png_path", "npy_path"]
        )
        writer.writeheader()
        writer.writerows(score_rows)

    print(f"Saved {saved} anomaly maps to: {OUT_DIR}")
    print(f"Saved score CSV to: {SCORES_CSV}")
    print("Both PNG previews and raw NPY maps were saved.")


if __name__ == "__main__":
    main()