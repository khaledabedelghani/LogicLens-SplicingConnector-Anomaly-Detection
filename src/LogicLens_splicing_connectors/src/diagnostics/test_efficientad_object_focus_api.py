from pathlib import Path

import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd

ROOT = Path(".")
DATA_ROOT = ROOT / "training_workdir" / "anomalib_splicing_object_focus"

CKPT = (
    ROOT
    / "training_outputs"
    / "efficientad_object_focus_20"
    / "EfficientAd"
    / "splicing_connectors_efficientad_object_focus"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

OUT_DIR = ROOT / "training_outputs" / "efficientad_object_focus_20_api_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\nTEST EFFICIENTAD OBJECT-FOCUS FROM CHECKPOINT USING PYTHON API")
    print("=" * 80)
    print("Dataset:", DATA_ROOT.resolve())
    print("Checkpoint:", CKPT.resolve())
    print("Output:", OUT_DIR.resolve())
    print("Visualizer: OFF")
    print("=" * 80)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_ROOT}")

    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    datamodule = Folder(
        name="splicing_connectors_efficientad_object_focus",
        root=DATA_ROOT,
        normal_dir="train/good",
        abnormal_dir="test/anomaly",
        normal_test_dir="test/good",
        mask_dir="ground_truth/anomaly",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=4,
    )

    model = EfficientAd(
        visualizer=False,
    )

    print("\nLoading checkpoint manually with weights_only=False...")
    checkpoint = torch.load(str(CKPT.resolve()), map_location="cpu", weights_only=False)

    state_dict = checkpoint["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Checkpoint loaded.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    engine = Engine(
        accelerator="gpu",
        devices=1,
        default_root_dir=OUT_DIR,
    )

    results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print("\nTEST RESULTS:")
    print(results)


if __name__ == "__main__":
    main()