from pathlib import Path
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

SRC = Path("training_workdir/anomalib_splicing_object_focus")
OUT = Path("training_workdir/anomalib_splicing_object_focus_aug_1080")

random.seed(42)
np.random.seed(42)


def light_augment(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)

    # very small rotation only
    angle = rng.uniform(-2.0, 2.0)
    img = img.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )

    # light brightness/contrast
    brightness = rng.uniform(0.94, 1.06)
    contrast = rng.uniform(0.94, 1.06)

    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # tiny noise
    arr = np.array(img).astype(np.float32)
    noise_std = rng.uniform(0.0, 2.0)
    if noise_std > 0:
        arr += np.random.normal(0, noise_std, arr.shape).astype(np.float32)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    print("Creating FAST object-focus augmented dataset 1080...")
    print("Source:", SRC.resolve())
    print("Output:", OUT.resolve())

    if not SRC.exists():
        raise FileNotFoundError(
            "Object-focus dataset not found. Run prepare_object_focus_dataset.py first."
        )

    if OUT.exists():
        print("Removing old:", OUT)
        shutil.rmtree(OUT)

    print("Copying existing object-focus dataset...")
    shutil.copytree(SRC, OUT)

    train_good = sorted((OUT / "train" / "good").glob("*.png"))
    print("Original train/good:", len(train_good))

    if len(train_good) != 360:
        print("WARNING: expected 360 original train/good images.")

    for i, img_path in enumerate(tqdm(train_good, desc="adding aug1+aug2")):
        img = Image.open(img_path).convert("RGB")

        aug1 = light_augment(img, seed=1000 + i * 2 + 1)
        aug1.save(img_path.with_name(img_path.stem + "_aug1.png"))

        aug2 = light_augment(img, seed=1000 + i * 2 + 2)
        aug2.save(img_path.with_name(img_path.stem + "_aug2.png"))

    final_count = len(list((OUT / "train" / "good").glob("*.png")))
    print("Final train/good:", final_count)

    if final_count != 1080:
        raise RuntimeError(f"Expected 1080 train/good images, got {final_count}")

    print("DONE")


if __name__ == "__main__":
    main()