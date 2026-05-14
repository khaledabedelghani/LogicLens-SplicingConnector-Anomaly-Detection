from pathlib import Path
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter

random.seed(42)

PROJECT_ROOT = Path(r"D:\project computer vision")

SRC_ROOT = PROJECT_ROOT / "data_roi_bg" / "splicing_connectors"
DST_ROOT = PROJECT_ROOT / "data_roi_bg_aug" / "splicing_connectors"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def copy_dataset():
    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    shutil.copytree(SRC_ROOT, DST_ROOT)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def augment_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")

    # 1) very small rotation
    angle = random.uniform(-3.5, 3.5)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    # 2) very slight brightness / contrast
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.97, 1.03))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.97, 1.03))

    # 3) tiny blur sometimes
    if random.random() < 0.30:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    return img


def main():
    copy_dataset()

    train_good_dir = DST_ROOT / "train" / "good"
    if not train_good_dir.exists():
        raise FileNotFoundError(f"train/good not found: {train_good_dir}")

    files = sorted([
        p for p in train_good_dir.iterdir()
        if p.is_file() and is_image(p) and "_aug1" not in p.stem
    ])

    print(f"Original train/good images: {len(files)}")

    for p in files:
        with Image.open(p) as img:
            aug = augment_image(img.copy())
            out_path = train_good_dir / f"{p.stem}_aug1{p.suffix}"
            aug.save(out_path)

    new_count = len([
        p for p in train_good_dir.iterdir()
        if p.is_file() and is_image(p)
    ])

    print("Done.")
    print(f"Augmented dataset saved to: {DST_ROOT}")
    print(f"New train/good count: {new_count}")


if __name__ == "__main__":
    main()