from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tqdm import tqdm
import random

DATASET_ROOT = Path(".")
MANIFEST_PATH = DATASET_ROOT / "outputs" / "dataset_manifest.csv"

OUT = DATASET_ROOT / "training_workdir" / "anomalib_splicing_object_focus_aug"

TARGET_SIZE = (1024, 512)  # width, height
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def object_focus_rgb(img_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    color_mask = (s > 35) & (v > 40)

    red_orange = ((h < 25) | (h > 165)) & (s > 25) & (v > 40)
    yellow = (h >= 20) & (h <= 45) & (s > 25) & (v > 40)
    blue = (h >= 85) & (h <= 135) & (s > 20) & (v > 30)

    mask = color_mask | red_orange | yellow | blue
    mask = mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    mask_soft = cv2.GaussianBlur(mask, (41, 41), 0).astype(np.float32) / 255.0
    mask_soft = np.clip(mask_soft, 0.0, 1.0)

    background = cv2.GaussianBlur(img_rgb, (45, 45), 0)
    background = (background * 0.25).astype(np.uint8)

    focused = (
        img_rgb.astype(np.float32) * mask_soft[..., None]
        + background.astype(np.float32) * (1.0 - mask_soft[..., None])
    )

    return np.clip(focused, 0, 255).astype(np.uint8)


def light_augment_pil(img: Image.Image, idx: int) -> Image.Image:
    """
    Very light normal-only augmentation.
    No crop, no flip, no blur, no random erasing.
    """
    rng = random.Random(1000 + idx)

    # tiny rotation, keep full canvas
    angle = rng.uniform(-2.0, 2.0)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

    # light brightness/contrast
    brightness = rng.uniform(0.92, 1.08)
    contrast = rng.uniform(0.92, 1.08)

    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # tiny gaussian noise
    arr = np.array(img).astype(np.float32)
    noise_std = rng.uniform(0.0, 3.0)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
        arr = arr + noise

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_image(src: Path, dst: Path, is_mask: bool = False, focus: bool = True, augment_idx: int | None = None):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if is_mask:
        img = Image.open(src).convert("L")
        img = img.resize(TARGET_SIZE, Image.Resampling.NEAREST)
        img.save(dst)
        return

    img = Image.open(src).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

    if augment_idx is not None:
        img = light_augment_pil(img, augment_idx)

    arr = np.array(img)
    if focus:
        arr = object_focus_rgb(arr)

    Image.fromarray(arr).save(dst)


def main():
    print("Preparing OBJECT-FOCUS AUGMENTED dataset")
    print("Train/good: original + light augmentations")
    print("Test/good + test/anomaly: NO augmentation")
    print("Masks: resized only")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    if OUT.exists():
        print(f"Removing old dataset: {OUT}")
        shutil.rmtree(OUT)

    df = pd.read_csv(MANIFEST_PATH)

    train_good = df[(df["split"] == "train") & (df["label"] == "good")]

    for i, (_, row) in enumerate(tqdm(train_good.iterrows(), total=len(train_good), desc="train/good original+aug")):
        src = DATASET_ROOT / row["image_path"]
        stem = src.stem

        # original object-focused
        save_image(src, OUT / "train" / "good" / f"{stem}.png", is_mask=False, focus=True)

        # two light augmentations only for training normal images
        save_image(src, OUT / "train" / "good" / f"{stem}_aug1.png", is_mask=False, focus=True, augment_idx=i * 2 + 1)
        save_image(src, OUT / "train" / "good" / f"{stem}_aug2.png", is_mask=False, focus=True, augment_idx=i * 2 + 2)

    val_good = df[(df["split"] == "validation") & (df["label"] == "good")]
    for _, row in tqdm(val_good.iterrows(), total=len(val_good), desc="val/good"):
        src = DATASET_ROOT / row["image_path"]
        save_image(src, OUT / "val" / "good" / src.name, is_mask=False, focus=True)

    test_good = df[(df["split"] == "test") & (df["label"] == "good")]
    for _, row in tqdm(test_good.iterrows(), total=len(test_good), desc="test/good"):
        src = DATASET_ROOT / row["image_path"]
        save_image(src, OUT / "test" / "good" / src.name, is_mask=False, focus=True)

    anomaly_rows = df[(df["split"] == "test") & (df["is_anomaly"] == 1)]
    family_rows = []

    for _, row in tqdm(anomaly_rows.iterrows(), total=len(anomaly_rows), desc="test/anomaly"):
        src = DATASET_ROOT / row["image_path"]
        family = row["anomaly_family"]
        image_id = Path(row["image_path"]).stem
        new_name = f"{family}_{image_id}.png"

        save_image(src, OUT / "test" / "anomaly" / new_name, is_mask=False, focus=True)
        save_image(src, OUT / "test_by_family" / family / new_name, is_mask=False, focus=True)

        mask_paths = str(row["mask_path"]).split(";")
        if len(mask_paths) == 0 or mask_paths[0] == "":
            raise RuntimeError(f"Missing mask for {row['image_path']}")

        mask_src = DATASET_ROOT / mask_paths[0]
        save_image(mask_src, OUT / "ground_truth" / "anomaly" / new_name, is_mask=True, focus=False)
        save_image(mask_src, OUT / "ground_truth_by_family" / family / new_name, is_mask=True, focus=False)

        family_rows.append({
            "prepared_image": str((OUT / "test" / "anomaly" / new_name).relative_to(OUT)),
            "prepared_mask": str((OUT / "ground_truth" / "anomaly" / new_name).relative_to(OUT)),
            "original_image": row["image_path"],
            "original_mask": row["mask_path"],
            "family": family,
            "label": row["label"],
            "prepared_width": TARGET_SIZE[0],
            "prepared_height": TARGET_SIZE[1],
        })

    pd.DataFrame(family_rows).to_csv(OUT / "family_mapping.csv", index=False)

    print("\nDONE")
    print("Dataset:", OUT.resolve())
    print("Train good count:", len(list((OUT / "train" / "good").glob("*.png"))))


if __name__ == "__main__":
    main()