from pathlib import Path
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter

random.seed(42)

PROJECT_ROOT = Path(r"D:\project computer vision")

SRC_ROOT = PROJECT_ROOT / "data_roi_padim" / "splicing_connectors"
DST_ROOT = PROJECT_ROOT / "data_roi_padim_aug" / "splicing_connectors"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def copy_dataset():
    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    shutil.copytree(SRC_ROOT, DST_ROOT)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def augment_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")

    # small rotation
    angle = random.uniform(-3.0, 3.0)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    # slight brightness / contrast
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.97, 1.03))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.97, 1.03))

    # tiny blur sometimes
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.35))

    return img


def main():
    copy_dataset()

    train_good_dir = DST_ROOT / "train" / "good"
    if not train_good_dir.exists():
        raise FileNotFoundError(f"train/good not found: {train_good_dir}")

    # only original images, never re-augment augmented ones
    files = sorted([
        p for p in train_good_dir.iterdir()
        if p.is_file()
        and is_image(p)
        and "_aug1" not in p.stem
        and "_aug2" not in p.stem
        and "_aug3" not in p.stem
    ])

    print(f"Original train/good images: {len(files)}")

    for p in files:
        with Image.open(p) as img:
            base = img.copy()

            aug1 = augment_image(base.copy())
            aug1.save(train_good_dir / f"{p.stem}_aug1{p.suffix}")

            aug2 = augment_image(base.copy())
            aug2.save(train_good_dir / f"{p.stem}_aug2{p.suffix}")

            aug3 = augment_image(base.copy())
            aug3.save(train_good_dir / f"{p.stem}_aug3{p.suffix}")

    new_count = len([
        p for p in train_good_dir.iterdir()
        if p.is_file() and is_image(p)
    ])

    print("Done.")
    print(f"Augmented dataset saved to: {DST_ROOT}")
    print(f"New train/good count: {new_count}")
    print("Expected total = original x 4")
    print("Only train/good was augmented. Test, validation, and ground_truth stayed unchanged.")


if __name__ == "__main__":
    main()