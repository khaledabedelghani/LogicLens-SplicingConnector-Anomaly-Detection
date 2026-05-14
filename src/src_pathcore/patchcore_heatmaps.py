from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.visualization import ImageVisualizer


def build_datamodule(project_root: Path):
    return Folder(
        name="splicing_connectors_overall",
        root=project_root,
        normal_dir="train/good",
        abnormal_dir=[
            "test/logical_anomalies",
            "test/structural_anomalies",
        ],
        normal_test_dir="test/good",
        mask_dir=[
            "ground_truth_flat/logical_anomalies",
            "ground_truth_flat/structural_anomalies",
        ],
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        normal_split_ratio=0.0,
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    visualizer = ImageVisualizer()

    model = Patchcore(
        pre_processor=Patchcore.configure_pre_processor(image_size=(224, 224)),
        num_neighbors=3,
        visualizer=visualizer,
    )

    engine = Engine(
        default_root_dir=project_root / "results" / "patchcore_heatmaps"
    )

    datamodule = build_datamodule(project_root)

    print("Training PatchCore final model for heatmaps...")
    engine.fit(model=model, datamodule=datamodule)

    print("\nGenerating heatmaps on test set...")
    engine.test(model=model, datamodule=datamodule)

    print("\nDone. Check the results folder:")
    print(project_root / "results" / "patchcore_heatmaps")