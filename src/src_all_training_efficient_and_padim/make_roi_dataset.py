from pathlib import Path
from PIL import Image
import shutil

PROJECT_ROOT = Path(r"D:\project computer vision")

# المصدر الحالي تبع الداتا
SRC_ROOT = PROJECT_ROOT / "data" / "splicing_connectors"

# الداتا الجديدة بعد الـROI crop
DST_PARENT = PROJECT_ROOT / "data_roi"
DST_ROOT = DST_PARENT / "splicing_connectors"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# جرب crop خفيف أول شي حتى ما نقص anomaly مهمة
LEFT_CROP = 0.10
RIGHT_CROP = 0.10
TOP_CROP = 0.08
BOTTOM_CROP = 0.08


def crop_box(width: int, height: int):
    left = int(width * LEFT_CROP)
    right = width - int(width * RIGHT_CROP)
    top = int(height * TOP_CROP)
    bottom = height - int(height * BOTTOM_CROP)

    # حماية بسيطة
    if right <= left:
        left = 0
        right = width
    if bottom <= top:
        top = 0
        bottom = height

    return (left, top, right, bottom)


def crop_and_save_image(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = img.copy()
        box = crop_box(img.width, img.height)
        cropped = img.crop(box)
        cropped.save(dst_path)


def process_tree():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source dataset not found: {SRC_ROOT}")

    if DST_ROOT.exists():
        print(f"Warning: destination already exists: {DST_ROOT}")
        print("New files will overwrite existing ones with the same names.")

    for src_path in SRC_ROOT.rglob("*"):
        rel_path = src_path.relative_to(SRC_ROOT)
        dst_path = DST_ROOT / rel_path

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        if src_path.suffix.lower() in IMAGE_EXTS:
            crop_and_save_image(src_path, dst_path)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)

    print("Done.")
    print(f"Cropped dataset saved to: {DST_ROOT}")
    print("Important: images and masks were cropped with the same rule.")


if __name__ == "__main__":
    process_tree()