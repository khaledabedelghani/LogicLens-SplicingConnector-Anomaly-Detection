import csv
import numpy as np
CSV_PATH = r"D:\project computer vision\notes\threshold_scores_roi_bg.csv"
rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["gt_label"] = int(row["gt_label"])
        row["score"] = float(row["score"])
        rows.append(row)

scores = [r["score"] for r in rows]
gts = [r["gt_label"] for r in rows]

def f1_score(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

best_thr = None
best_f1 = -1

for thr in np.linspace(0.0, 1.0, 201):
    preds = [1 if s >= thr else 0 for s in scores]
    f1 = f1_score(gts, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print(f"Best threshold = {best_thr:.4f}")
print(f"Best F1 = {best_f1:.4f}")