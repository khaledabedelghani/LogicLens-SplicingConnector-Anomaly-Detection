from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(".")
SRC = ROOT / "training_workdir" / "anomalib_splicing"
OUT = ROOT / "training_workdir" / "logiclens_roi448"

TARGET_SIZE = 448
MARGIN = 0.14
SEED = 42


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def foreground_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Object-centric mask for colored cable/connector foreground.
    Keeps cable + colored connector, suppresses metallic/background noise.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    color = (s > 28) & (v > 35)

    red = ((h <= 12) | (h >= 165)) & (s > 25) & (v > 35)
    orange = (h >= 4) & (h <= 32) & (s > 25) & (v > 35)
    yellow = (h >= 18) & (h <= 52) & (s > 25) & (v > 35)
    blue = (h >= 80) & (h <= 140) & (s > 20) & (v > 30)

    mask = color | red | orange | yellow | blue
    mask = mask.astype(np.uint8) * 255

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)
    mask = cv2.dilate(mask, k2, iterations=1)

    # Keep largest connected component group
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return np.ones(mask.shape, dtype=np.uint8) * 255

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    clean = (labels == largest).astype(np.uint8) * 255

    # If largest component too small, fallback to all colored mask
    if clean.mean() < 3:
        clean = mask

    return clean


def bbox_from_mask(mask: np.ndarray, img_shape, margin=MARGIN):
    H, W = img_shape[:2]
    ys, xs = np.where(mask > 0)

    if len(xs) < 50:
        return 0, 0, W, H

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    dx = int(bw * margin)
    dy = int(bh * margin)

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(W, x2 + dx)
    y2 = min(H, y2 + dy)

    # Make square crop around object for ViT/PatchCore consistency
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    side = max(x2 - x1, y2 - y1)
    side = int(side * 1.05)

    sx1 = max(0, cx - side // 2)
    sy1 = max(0, cy - side // 2)
    sx2 = min(W, sx1 + side)
    sy2 = min(H, sy1 + side)

    # adjust if clipped
    sx1 = max(0, sx2 - side)
    sy1 = max(0, sy2 - side)

    return sx1, sy1, sx2, sy2


def crop_resize_image(src: Path, dst: Path, mask_src: Path | None = None, mask_dst: Path | None = None):
    img = read_rgb(src)
    m = foreground_mask(img)
    x1, y1, x2, y2 = bbox_from_mask(m, img.shape)

    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop).save(dst)

    if mask_src is not None and mask_dst is not None:
        if mask_src.exists():
            gt = np.array(Image.open(mask_src).convert("L"))
            gt_crop = gt[y1:y2, x1:x2]
            gt_crop = cv2.resize(gt_crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
        else:
            gt_crop = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(gt_crop).save(mask_dst)


def family_from_name(p: Path):
    n = p.name.lower()
    if n.startswith("logical"):
        return "logical"
    if n.startswith("structural"):
        return "structural"
    return "anomaly"


def copy_split():
    if OUT.exists():
        print("Removing old:", OUT)
        shutil.rmtree(OUT)

    rows = []

    # train good
    train_good = sorted((SRC / "train" / "good").glob("*.png"))
    for p in tqdm(train_good, desc="ROI train/good"):
        dst = OUT / "train" / "good" / p.name
        crop_resize_image(p, dst)
        rows.append({"path": str(dst), "label": 0, "family": "good", "source_split": "train"})

    # validation good
    val_good = sorted((SRC / "val" / "good").glob("*.png"))
    for p in tqdm(val_good, desc="ROI val/good"):
        dst = OUT / "val" / "good" / p.name
        crop_resize_image(p, dst)
        rows.append({"path": str(dst), "label": 0, "family": "good", "source_split": "val_good"})

    # test good
    test_good = sorted((SRC / "test" / "good").glob("*.png"))
    for p in tqdm(test_good, desc="ROI test/good"):
        dst = OUT / "test" / "good" / p.name
        crop_resize_image(p, dst)
        rows.append({"path": str(dst), "label": 0, "family": "good", "source_split": "test_good"})

    # test anomaly + masks
    test_anom = sorted((SRC / "test" / "anomaly").glob("*.png"))
    for p in tqdm(test_anom, desc="ROI test/anomaly"):
        fam = family_from_name(p)
        dst = OUT / "test" / "anomaly" / p.name
        mask_src = SRC / "ground_truth" / "anomaly" / p.name
        mask_dst = OUT / "ground_truth" / "anomaly" / p.name

        crop_resize_image(p, dst, mask_src=mask_src, mask_dst=mask_dst)
        rows.append({"path": str(dst), "label": 1, "family": fam, "source_split": "test_anomaly"})

    df = pd.DataFrame(rows)

    # Build calibration split from original test only, no train leakage.
    eval_df = df[df["source_split"].isin(["test_good", "test_anomaly"])].copy()
    strat = eval_df["label"].astype(str) + "_" + eval_df["family"].astype(str)

    # Some families may be small, fallback if stratification fails
    try:
        val_eval, test_eval = train_test_split(
            eval_df,
            test_size=0.60,
            random_state=SEED,
            stratify=strat,
        )
    except Exception:
        val_eval, test_eval = train_test_split(
            eval_df,
            test_size=0.60,
            random_state=SEED,
            stratify=eval_df["label"],
        )

    df["eval_split"] = "none"
    df.loc[df["source_split"] == "train", "eval_split"] = "train_good"
    df.loc[df["source_split"] == "val_good", "eval_split"] = "val_good_only"

    df.loc[val_eval.index, "eval_split"] = "val_calib"
    df.loc[test_eval.index, "eval_split"] = "test_final"

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "roi_manifest.csv", index=False)

    print("\nDONE ROI DATASET")
    print("Output:", OUT.resolve())
    print(df.groupby(["eval_split", "label", "family"]).size())


def main():
    print("Preparing LogicLens ROI canonical dataset")
    print("Source:", SRC.resolve())
    print("Output:", OUT.resolve())
    copy_split()


if __name__ == "__main__":
    main()