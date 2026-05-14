from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image

from dataset import build_dataset


def load_image(path):
    return Image.open(path).convert("RGB")


def load_mask(path):
    return Image.open(path).convert("L")


def show_sample(sample):
    image = load_image(sample["image_path"])

    if sample["mask_path"] is not None:
        mask = load_mask(sample["mask_path"])

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(
            f'Image\nlabel={sample["label"]}, type={sample["anomaly_type"]}'
        )
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

    else:
        plt.figure(figsize=(5, 4))
        plt.imshow(image)
        plt.title(
            f'Image\nlabel={sample["label"]}, type={sample["anomaly_type"]}'
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    samples = build_dataset(project_root)

    train_good = [s for s in samples if s["split"] == "train"]
    logical_test = [s for s in samples if s["split"] == "test" and s["anomaly_type"] == "logical"]
    structural_test = [s for s in samples if s["split"] == "test" and s["anomaly_type"] == "structural"]

    print("Showing 2 train good samples...")
    for sample in random.sample(train_good, 2):
        show_sample(sample)

    print("Showing 2 logical anomaly samples...")
    for sample in random.sample(logical_test, 2):
        show_sample(sample)

    print("Showing 2 structural anomaly samples...")
    for sample in random.sample(structural_test, 2):
        show_sample(sample)