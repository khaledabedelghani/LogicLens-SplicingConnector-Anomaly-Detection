from pathlib import Path
import shutil


def flatten_masks(src_root: Path, dst_root: Path):
    dst_root.mkdir(parents=True, exist_ok=True)
    count = 0

    for sample_dir in sorted(src_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        src_mask = sample_dir / "000.png"
        if not src_mask.exists():
            print(f"Missing mask: {src_mask}")
            continue

        dst_mask = dst_root / f"{sample_dir.name}.png"
        shutil.copy2(src_mask, dst_mask)
        count += 1

    return count


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    roi_root = project_root / "splicing_Connectors_EfficientAD" / "data_roi" / "splicing_connectors"

    logical_count = flatten_masks(
        roi_root / "ground_truth" / "logical_anomalies",
        roi_root / "ground_truth_flat" / "logical_anomalies",
    )

    structural_count = flatten_masks(
        roi_root / "ground_truth" / "structural_anomalies",
        roi_root / "ground_truth_flat" / "structural_anomalies",
    )

    print("Done.")
    print(f"Logical masks: {logical_count}")
    print(f"Structural masks: {structural_count}")