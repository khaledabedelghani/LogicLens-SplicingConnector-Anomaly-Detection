from pathlib import Path
from PIL import Image
import shutil

PROJECT_ROOT = Path(r"D:\project computer vision")

SRC_ROOT = PROJECT_ROOT / "data" / "splicing_connectors"
DST_PARENT = PROJECT_ROOT / "data_roi_bg"
DST_ROOT = DST_PARENT / "splicing_connectors"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ROI v2: أقل قص من اليمين/الشمال، أكثر من فوق/تحت
LEFT_CROP = 0.04
RIGHT_CROP = 0.04
TOP_CROP = 0.18
BOTTOM_CROP = 0.18

# Black background داخل الصورة المقصوصة
# نخلي المنطقة المهمة بالنص ونصبغ الحواف أسود
KEEP_LEFT = 0.05
KEEP_RIGHT = 0.95
KEEP_TOP = 0.08
KEEP_BOTTOM = 0.92


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


def apply_black_background(img: Image.Image) -> Image.Image:
    width, height = img.size

    keep_left = int(width * KEEP_LEFT)
    keep_right = int(width * KEEP_RIGHT)
    keep_top = int(height * KEEP_TOP)
    keep_bottom = int(height * KEEP_BOTTOM)

    if keep_right <= keep_left or keep_bottom <= keep_top:
        return img

    black_canvas = Image.new(img.mode, img.size, 0)
    keep_region = img.crop((keep_left, keep_top, keep_right, keep_bottom))
    black_canvas.paste(keep_region, (keep_left, keep_top))
    return black_canvas


def process_image(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = img.copy()
        cropped = img.crop(crop_box(img.width, img.height))

        # ground_truth masks: crop only, no black background
        if "ground_truth" not in src_path.parts:
            cropped = apply_black_background(cropped)

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
    print("Images were cropped with ROI v2.")
    print("Non-ground-truth images got black background suppression.")


if __name__ == "__main__":
    process_tree()