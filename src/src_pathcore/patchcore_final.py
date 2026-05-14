from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


def build_datamodule(project_root: Path, mode: str):
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
        name=f"splicing_connectors_{mode}",
        root=project_root,
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
    project_root = Path(__file__).resolve().parent.parent

    model = Patchcore(
        pre_processor=Patchcore.configure_pre_processor(image_size=(224, 224)),
        num_neighbors=3,
    )

    engine = Engine(
        default_root_dir=project_root / "results" / "patchcore_final"
    )

    print("Training final PatchCore...")
    engine.fit(model=model, datamodule=build_datamodule(project_root, "overall"))

    print("\nTesting overall...")
    overall = engine.test(model=model, datamodule=build_datamodule(project_root, "overall"))

    print("\nTesting logical only...")
    logical = engine.test(model=model, datamodule=build_datamodule(project_root, "logical"))

    print("\nTesting structural only...")
    structural = engine.test(model=model, datamodule=build_datamodule(project_root, "structural"))

    summary = []
    summary.append("PATCHCORE FINAL RESULTS")
    summary.append("image_size = 224")
    summary.append("num_neighbors = 3")
    summary.append("")
    summary.append(f"overall   : {overall}")
    summary.append(f"logical   : {logical}")
    summary.append(f"structural: {structural}")

    summary_text = "\n".join(summary)

    print("\n=== FINAL SUMMARY ===")
    print(summary_text)

    out_file = project_root / "results" / "patchcore_final" / "final_metrics.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(summary_text, encoding="utf-8")

    print(f"\nSaved to: {out_file}")