from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATASET_ROOT = Path(".")
MANIFEST_PATH = DATASET_ROOT / "outputs" / "dataset_manifest.csv"

WORKDIR = DATASET_ROOT / "training_workdir"
OUT = WORKDIR / "anomalib_splicing_object_focus"

TARGET_SIZE = (1024, 512)  # width, height


def object_focus_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """
    Keep full image. No crop.
    Suppress low-saturation metallic background while preserving cable/connectors.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Main foreground: colored cables/connectors have higher saturation.
    color_mask = (s > 35) & (v > 40)

    # Keep strong red/orange/yellow/blue regions even if saturation varies.
    red_orange = ((h < 25) | (h > 165)) & (s > 25) & (v > 40)
    yellow = (h >= 20) & (h <= 40) & (s > 25) & (v > 40)
    blue = (h >= 85) & (h <= 135) & (s > 20) & (v > 30)

    mask = color_mask | red_orange | yellow | blue

    mask = mask.astype(np.uint8) * 255

    # Connect nearby regions: cable + connectors.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Soft mask edges.
    mask_soft = cv2.GaussianBlur(mask, (41, 41), 0).astype(np.float32) / 255.0
    mask_soft = np.clip(mask_soft, 0.0, 1.0)

    # Background suppression, not removal.
    # Keep some background visibility so model still sees context.
    background = cv2.GaussianBlur(img_rgb, (45, 45), 0)
    background = (background * 0.25).astype(np.uint8)

    focused = (
        img_rgb.astype(np.float32) * mask_soft[..., None]
        + background.astype(np.float32) * (1.0 - mask_soft[..., None])
    )

    return np.clip(focused, 0, 255).astype(np.uint8)


def save_resized_image(src: Path, dst: Path, is_mask: bool = False, focus: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if is_mask:
        img = Image.open(src).convert("L")
        img = img.resize(TARGET_SIZE, Image.Resampling.NEAREST)
        img.save(dst)
        return

    img = Image.open(src).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img)

    if focus:
        arr = object_focus_rgb(arr)

    Image.fromarray(arr).save(dst)


def main():
    print("Preparing OBJECT-FOCUSED Anomalib dataset...")
    print("No crop. Full image preserved.")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    if OUT.exists():
        print(f"Removing old object-focused dataset: {OUT}")
        shutil.rmtree(OUT)

    df = pd.read_csv(MANIFEST_PATH)

    train_good = df[(df["split"] == "train") & (df["label"] == "good")]
    for _, row in tqdm(train_good.iterrows(), total=len(train_good), desc="train/good"):
        src = DATASET_ROOT / row["image_path"]
        dst = OUT / "train" / "good" / src.name
        save_resized_image(src, dst, is_mask=False, focus=True)

    val_good = df[(df["split"] == "validation") & (df["label"] == "good")]
    for _, row in tqdm(val_good.iterrows(), total=len(val_good), desc="val/good"):
        src = DATASET_ROOT / row["image_path"]
        dst = OUT / "val" / "good" / src.name
        save_resized_image(src, dst, is_mask=False, focus=True)

    test_good = df[(df["split"] == "test") & (df["label"] == "good")]
    for _, row in tqdm(test_good.iterrows(), total=len(test_good), desc="test/good"):
        src = DATASET_ROOT / row["image_path"]
        dst = OUT / "test" / "good" / src.name
        save_resized_image(src, dst, is_mask=False, focus=True)

    anomaly_rows = df[(df["split"] == "test") & (df["is_anomaly"] == 1)]
    family_rows = []

    for _, row in tqdm(anomaly_rows.iterrows(), total=len(anomaly_rows), desc="test/anomaly"):
        src = DATASET_ROOT / row["image_path"]
        family = row["anomaly_family"]
        image_id = Path(row["image_path"]).stem
        new_name = f"{family}_{image_id}.png"

        dst_img = OUT / "test" / "anomaly" / new_name
        save_resized_image(src, dst_img, is_mask=False, focus=True)

        dst_family = OUT / "test_by_family" / family / new_name
        save_resized_image(src, dst_family, is_mask=False, focus=True)

        mask_paths = str(row["mask_path"]).split(";")
        if len(mask_paths) == 0 or mask_paths[0] == "":
            raise RuntimeError(f"Missing mask for {row['image_path']}")

        mask_src = DATASET_ROOT / mask_paths[0]

        dst_mask = OUT / "ground_truth" / "anomaly" / new_name
        save_resized_image(mask_src, dst_mask, is_mask=True, focus=False)

        dst_mask_family = OUT / "ground_truth_by_family" / family / new_name
        save_resized_image(mask_src, dst_mask_family, is_mask=True, focus=False)

        family_rows.append({
            "prepared_image": str(dst_img.relative_to(OUT)),
            "prepared_mask": str(dst_mask.relative_to(OUT)),
            "original_image": row["image_path"],
            "original_mask": row["mask_path"],
            "family": family,
            "label": row["label"],
            "prepared_width": TARGET_SIZE[0],
            "prepared_height": TARGET_SIZE[1],
        })

    pd.DataFrame(family_rows).to_csv(OUT / "family_mapping.csv", index=False)

    print("\nPrepared object-focused dataset:")
    print(OUT.resolve())
    print("\nDONE")


if __name__ == "__main__":
    main()