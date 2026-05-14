from pathlib import Path


def add_good_samples(folder, split_name, samples):
    for img_path in sorted(folder.glob("*.png")):
        samples.append({
            "split": split_name,
            "image_path": str(img_path),
            "label": 0,
            "anomaly_type": "good",
            "mask_path": None,
        })


def add_anomaly_samples(test_folder, gt_folder, split_name, anomaly_type, samples):
    for img_path in sorted(test_folder.glob("*.png")):
        sample_id = img_path.stem   # مثلا 004 من 004.png
        mask_path = gt_folder / sample_id / "000.png"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found for {img_path.name} -> expected: {mask_path}"
            )

        samples.append({
            "split": split_name,
            "image_path": str(img_path),
            "label": 1,
            "anomaly_type": anomaly_type,
            "mask_path": str(mask_path),
        })


def build_dataset(root_dir):
    root = Path(root_dir)
    samples = []

    # train/good
    add_good_samples(
        root / "train" / "good",
        "train",
        samples
    )

    # validation/good
    add_good_samples(
        root / "validation" / "good",
        "validation",
        samples
    )

    # test/good
    add_good_samples(
        root / "test" / "good",
        "test",
        samples
    )

    # test/logical_anomalies
    add_anomaly_samples(
        root / "test" / "logical_anomalies",
        root / "ground_truth" / "logical_anomalies",
        "test",
        "logical",
        samples
    )

    # test/structural_anomalies
    add_anomaly_samples(
        root / "test" / "structural_anomalies",
        root / "ground_truth" / "structural_anomalies",
        "test",
        "structural",
        samples
    )

    return samples


def print_summary(samples):
    train_count = sum(1 for s in samples if s["split"] == "train")
    val_count = sum(1 for s in samples if s["split"] == "validation")
    test_count = sum(1 for s in samples if s["split"] == "test")

    test_good = sum(1 for s in samples if s["split"] == "test" and s["anomaly_type"] == "good")
    test_logical = sum(1 for s in samples if s["split"] == "test" and s["anomaly_type"] == "logical")
    test_structural = sum(1 for s in samples if s["split"] == "test" and s["anomaly_type"] == "structural")

    print("=== DATASET SUMMARY ===")
    print(f"Train samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Test samples: {test_count}")
    print(f"  - Test good: {test_good}")
    print(f"  - Test logical anomalies: {test_logical}")
    print(f"  - Test structural anomalies: {test_structural}")
    print()

    print("=== FIRST 5 TEST ANOMALY MAPPINGS ===")
    shown = 0
    for s in samples:
        if s["split"] == "test" and s["label"] == 1:
            print(f'{s["image_path"]}  -->  {s["mask_path"]}')
            shown += 1
            if shown == 5:
                break


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    samples = build_dataset(project_root)
    print_summary(samples)