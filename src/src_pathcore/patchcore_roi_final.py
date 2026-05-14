from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


def build_datamodule(roi_root: Path, mode: str):
    if mode == "overall":
        abnormal_dir = [
            "test/logical_anomalies",
            "test/structural_anomalies",
        ]
        mask_dir = [
            "ground_truth_flat/logical_anomalies",
            "ground_truth_flat/structural_anomalies",
        ]
    elif mode == "logical":
        abnormal_dir = "test/logical_anomalies"
        mask_dir = "ground_truth_flat/logical_anomalies"
    elif mode == "structural":
        abnormal_dir = "test/structural_anomalies"
        mask_dir = "ground_truth_flat/structural_anomalies"
    else:
        raise ValueError("mode must be overall, logical, or structural")

    return Folder(
        name=f"splicing_connectors_roi_{mode}",
        root=roi_root,
        normal_dir="train/good",
        abnormal_dir=abnormal_dir,
        normal_test_dir="test/good",
        mask_dir=mask_dir,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        normal_split_ratio=0.0,
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    roi_root = project_root / "splicing_Connectors_EfficientAD" / "data_roi" / "splicing_connectors"

    model = Patchcore(
        pre_processor=Patchcore.configure_pre_processor(image_size=(224, 224)),
        num_neighbors=3,
    )

    engine = Engine(
        default_root_dir=roi_root / "results_patchcore_roi_final"
    )

    print("Training PatchCore on ROI...")
    engine.fit(model=model, datamodule=build_datamodule(roi_root, "overall"))

    print("\nTesting overall...")
    overall = engine.test(model=model, datamodule=build_datamodule(roi_root, "overall"))

    print("\nTesting logical only...")
    logical = engine.test(model=model, datamodule=build_datamodule(roi_root, "logical"))

    print("\nTesting structural only...")
    structural = engine.test(model=model, datamodule=build_datamodule(roi_root, "structural"))

    print("\n=== ROI FINAL SUMMARY ===")
    print(f"overall   : {overall}")
    print(f"logical   : {logical}")
    print(f"structural: {structural}")