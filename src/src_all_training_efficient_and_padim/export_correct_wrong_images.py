from pathlib import Path
import shutil
import pandas as pd

THRESHOLD = 0.005025

CSV_PATH = Path(r"D:\project computer vision\per_image_predictions_aug.csv")
OUT_DIR = Path(r"D:\project computer vision\review_threshold_0_005025")

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required_cols = {"image_path", "group", "gt_label", "pred_score"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# recompute predictions using the tuned threshold
df["pred_label_thr"] = (df["pred_score"] >= THRESHOLD).astype(int)

def classify_row(row):
    group = str(row["group"]).lower()
    pred = int(row["pred_label_thr"])

    if group == "good":
        return "correct" if pred == 0 else "wrong"
    elif group in {"logical", "structural"}:
        return "correct" if pred == 1 else "wrong"
    else:
        return "unknown"

df["result"] = df.apply(classify_row, axis=1)

# clean old output
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

# create folders
for result in ["correct", "wrong", "unknown"]:
    for group in ["good", "logical", "structural", "other"]:
        (OUT_DIR / result / group).mkdir(parents=True, exist_ok=True)

copied_rows = []

for _, row in df.iterrows():
    src = Path(str(row["image_path"]))
    if not src.exists():
        copied_rows.append({
            "image_path": str(src),
            "group": row["group"],
            "gt_label": row["gt_label"],
            "pred_score": row["pred_score"],
            "pred_label_thr": row["pred_label_thr"],
            "result": row["result"],
            "copied": False,
            "reason": "source_not_found",
        })
        continue

    group = str(row["group"]).lower()
    if group not in {"good", "logical", "structural"}:
        group = "other"

    result = str(row["result"]).lower()
    if result not in {"correct", "wrong", "unknown"}:
        result = "unknown"

    dst_dir = OUT_DIR / result / group
    dst_path = dst_dir / src.name

    # avoid collisions
    if dst_path.exists():
        dst_path = dst_dir / f"{src.stem}_{abs(hash(str(src))) % 100000}{src.suffix}"

    shutil.copy2(src, dst_path)

    copied_rows.append({
        "image_path": str(src),
        "copied_to": str(dst_path),
        "group": row["group"],
        "gt_label": int(row["gt_label"]),
        "pred_score": float(row["pred_score"]),
        "pred_label_thr": int(row["pred_label_thr"]),
        "result": row["result"],
        "copied": True,
        "reason": "",
    })

out_df = pd.DataFrame(copied_rows)
out_df.to_csv(OUT_DIR / "review_index.csv", index=False)

summary = (
    out_df[out_df["copied"] == True]
    .groupby(["result", "group"])
    .size()
    .reset_index(name="count")
)

summary.to_csv(OUT_DIR / "summary.csv", index=False)

print("\nSaved review folders to:")
print(OUT_DIR)

print("\nSummary:")
print(summary if not summary.empty else "No copied images.")
print("\nIndex file:")
print(OUT_DIR / "review_index.csv")