from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image

DATASET_ROOT = Path(".")
MANIFEST_PATH = DATASET_ROOT / "outputs" / "dataset_manifest.csv"

WORKDIR = DATASET_ROOT / "training_workdir"
OUT = WORKDIR / "anomalib_splicing"

TARGET_SIZE = (1024, 512)  # width, height


def save_resized_image(src: Path, dst: Path, is_mask: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)

    if is_mask:
        img = img.convert("L")
        img = img.resize(TARGET_SIZE, Image.Resampling.NEAREST)
    else:
        img = img.convert("RGB")
        img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

    img.save(dst)


def main():
    print("Preparing Anomalib-compatible resized dataset...")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    if OUT.exists():
        print(f"Removing old working dataset: {OUT}")
        shutil.rmtree(OUT)

    df = pd.read_csv(MANIFEST_PATH)

    # Train good
    train_good = df[(df["split"] == "train") & (df["label"] == "good")]
    for _, row in tqdm(train_good.iterrows(), total=len(train_good), desc="train/good"):
        src = DATASET_ROOT / row["image_path"]
        dst = OUT / "train" / "good" / src.name
        save_resized_image(src, dst, is_mask=False)

    # Validation good kept for reference, not directly used by Folder CLI
    val_good = df[(df["split"] == "validation") & (df["label"] == "good")]
    for _, row in tqdm(val_good.iterrows(), total=len(val_good), desc="val/good"):
        src = DATASET_ROOT / row["image_path"]
        dst = OUT / "val" / "good" / src.name
        save_resized_image(src, dst, is_mask=False)

    # Test good
    test_good = df[(df["split"] == "test") & (df["label"] == "good")]
    for _, row in tqdm(test_good.iterrows(), total=len(test_good), desc="test/good"):
        src = DATASET_ROOT / row["image_path"]
        dst = OUT / "test" / "good" / src.name
        save_resized_image(src, dst, is_mask=False)

    # Test anomalies combined for Anomalib, plus family-specific dirs for our analysis
    anomaly_rows = df[(df["split"] == "test") & (df["is_anomaly"] == 1)]

    family_rows = []

    for _, row in tqdm(anomaly_rows.iterrows(), total=len(anomaly_rows), desc="test/anomaly"):
        src = DATASET_ROOT / row["image_path"]
        family = row["anomaly_family"]
        image_id = Path(row["image_path"]).stem
        new_name = f"{family}_{image_id}.png"

        dst_img = OUT / "test" / "anomaly" / new_name
        save_resized_image(src, dst_img, is_mask=False)

        dst_family = OUT / "test_by_family" / family / new_name
        save_resized_image(src, dst_family, is_mask=False)

        mask_paths = str(row["mask_path"]).split(";")
        if len(mask_paths) == 0 or mask_paths[0] == "":
            raise RuntimeError(f"Missing mask for {row['image_path']}")

        mask_src = DATASET_ROOT / mask_paths[0]

        dst_mask = OUT / "ground_truth" / "anomaly" / new_name
        save_resized_image(mask_src, dst_mask, is_mask=True)

        dst_mask_family = OUT / "ground_truth_by_family" / family / new_name
        save_resized_image(mask_src, dst_mask_family, is_mask=True)

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

    family_df = pd.DataFrame(family_rows)
    family_df.to_csv(OUT / "family_mapping.csv", index=False)

    print("\nPrepared resized dataset:")
    print(OUT.resolve())
    print("\nStructure:")
    print("train/good")
    print("val/good")
    print("test/good")
    print("test/anomaly")
    print("ground_truth/anomaly")
    print("test_by_family/logical")
    print("test_by_family/structural")
    print("ground_truth_by_family/logical")
    print("ground_truth_by_family/structural")
    print("\nDONE")


if __name__ == "__main__":
    main()
