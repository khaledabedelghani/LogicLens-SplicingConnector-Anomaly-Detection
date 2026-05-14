from pathlib import Path
from PIL import Image
import shutil

PROJECT_ROOT = Path(r"D:\project computer vision")

# source = original data
SRC_ROOT = PROJECT_ROOT / "data" / "splicing_connectors"

# new cropped dataset for PaDiM
DST_PARENT = PROJECT_ROOT / "data_roi_padim"
DST_ROOT = DST_PARENT / "splicing_connectors"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Based on your sample images:
# - keep horizontal crop very small
# - crop more from top/bottom
LEFT_CROP = 0.005
RIGHT_CROP = 0.005
TOP_CROP = 0.1
BOTTOM_CROP = 0.1


def crop_box(width: int, height: int):
    left = int(width * LEFT_CROP)
    right = width - int(width * RIGHT_CROP)
    top = int(height * TOP_CROP)
    bottom = height - int(height * BOTTOM_CROP)

    if right <= left:
        left, right = 0, width
    if bottom <= top:
        top, bottom = 0, height

    return (left, top, right, bottom)


def process_image(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = img.copy()
        cropped = img.crop(crop_box(img.width, img.height))
        cropped.save(dst_path)


def process_tree():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source dataset not found: {SRC_ROOT}")

    if DST_ROOT.exists():
        print(f"Warning: destination already exists: {DST_ROOT}")
        print("Files with same names will be overwritten.")

    for src_path in SRC_ROOT.rglob("*"):
        rel_path = src_path.relative_to(SRC_ROOT)
        dst_path = DST_ROOT / rel_path

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        if src_path.suffix.lower() in IMAGE_EXTS:
            process_image(src_path, dst_path)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)

    print("Done.")
    print(f"New dataset saved to: {DST_ROOT}")
    print("All images and masks were cropped with the same rule.")
    print(f"Crop used: left={LEFT_CROP}, right={RIGHT_CROP}, top={TOP_CROP}, bottom={BOTTOM_CROP}")


if __name__ == "__main__":
    process_tree()