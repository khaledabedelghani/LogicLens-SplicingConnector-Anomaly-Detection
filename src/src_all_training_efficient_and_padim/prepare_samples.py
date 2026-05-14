from pathlib import Path
import shutil

project_root = Path(r"D:\project computer vision")
data_root = project_root / "data" / "splicing_connectors" / "test"
out_root = project_root / "analysis_samples"

mapping = {
    "good": data_root / "good",
    "structural": data_root / "structural_anomalies",
    "logical": data_root / "logical_anomalies",
}

for group_name, src_dir in mapping.items():
    dst_dir = out_root / group_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]])

    for img_path in images[:5]:
        shutil.copy2(img_path, dst_dir / img_path.name)

print("Done. Samples copied to analysis_samples.")