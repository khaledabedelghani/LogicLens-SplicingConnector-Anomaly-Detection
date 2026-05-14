import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

CSV_PATH = r"D:\project computer vision\per_image_predictions_aug.csv"

df = pd.read_csv(CSV_PATH)

required_cols = {"image_path", "group", "gt_label", "pred_score"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

def binary_stats(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return p, r, f1

def eval_threshold(dataframe, threshold):
    tmp = dataframe.copy()
    tmp["pred_label_thr"] = (tmp["pred_score"] >= threshold).astype(int)

    rows = []

    # 1) overall anomaly
    y_true = tmp["gt_label"].astype(int)
    y_pred = tmp["pred_label_thr"].astype(int)
    p, r, f1 = binary_stats(y_true, y_pred)
    rows.append({
        "subset": "overall_anomaly",
        "threshold": threshold,
        "precision": p,
        "recall": r,
        "f1": f1,
        "n": len(tmp),
    })

    # 2) good class F1 (positive = good)
    good_true = (tmp["group"] == "good").astype(int)
    good_pred = (tmp["pred_label_thr"] == 0).astype(int)
    p, r, f1 = binary_stats(good_true, good_pred)
    rows.append({
        "subset": "good_class",
        "threshold": threshold,
        "precision": p,
        "recall": r,
        "f1": f1,
        "n": len(tmp),
    })

    # 3) logical subset (good + logical), positive = logical anomaly
    logical_df = tmp[tmp["group"].isin(["good", "logical"])].copy()
    logical_true = (logical_df["group"] == "logical").astype(int)
    logical_pred = logical_df["pred_label_thr"].astype(int)
    p, r, f1 = binary_stats(logical_true, logical_pred)
    rows.append({
        "subset": "logical_subset",
        "threshold": threshold,
        "precision": p,
        "recall": r,
        "f1": f1,
        "n": len(logical_df),
    })

    # 4) structural subset (good + structural), positive = structural anomaly
    struct_df = tmp[tmp["group"].isin(["good", "structural"])].copy()
    struct_true = (struct_df["group"] == "structural").astype(int)
    struct_pred = struct_df["pred_label_thr"].astype(int)
    p, r, f1 = binary_stats(struct_true, struct_pred)
    rows.append({
        "subset": "structural_subset",
        "threshold": threshold,
        "precision": p,
        "recall": r,
        "f1": f1,
        "n": len(struct_df),
    })

    return rows

# Threshold range based on observed scores
score_min = df["pred_score"].min()
score_max = df["pred_score"].max()

thresholds = np.linspace(score_min, score_max, 200)

all_rows = []
for thr in thresholds:
    all_rows.extend(eval_threshold(df, thr))

results = pd.DataFrame(all_rows)

# Save full sweep
results.to_csv(r"D:\project computer vision\threshold_sweep_results_aug.csv", index=False)

# Best threshold per subset
best_rows = (
    results.sort_values(["subset", "f1"], ascending=[True, False])
           .groupby("subset", as_index=False)
           .first()
)

best_rows.to_csv(r"D:\project computer vision\best_thresholds_by_subset_aug.csv", index=False)

print("\n=== Best threshold per subset ===")
print(best_rows[["subset", "threshold", "precision", "recall", "f1", "n"]])

# Optional: one global score that favors logical more
pivot = results.pivot_table(
    index="threshold",
    columns="subset",
    values="f1"
).reset_index()

pivot["custom_score"] = (
    0.50 * pivot["logical_subset"].fillna(0) +
    0.25 * pivot["overall_anomaly"].fillna(0) +
    0.15 * pivot["good_class"].fillna(0) +
    0.10 * pivot["structural_subset"].fillna(0)
)

best_global = pivot.sort_values("custom_score", ascending=False).iloc[0]
print("\n=== Best custom threshold (logical-focused) ===")
print(best_global)

pivot.to_csv(r"D:\project computer vision\threshold_sweep_summary_aug.csv", index=False)