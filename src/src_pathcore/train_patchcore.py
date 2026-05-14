from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    datamodule = Folder(
        name="splicing_connectors",
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

    model = Patchcore()
    engine = Engine()

    print("Starting PatchCore training...")
    engine.fit(model=model, datamodule=datamodule)

    print("\nStarting PatchCore testing...")
    test_results = engine.test(model=model, datamodule=datamodule)

    print("\n=== PATCHCORE TEST RESULTS ===")
    print(test_results)