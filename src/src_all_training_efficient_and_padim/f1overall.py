# import torch
# import numpy as np
# import pandas as pd
# from sklearn.metrics import f1_score, precision_recall_fscore_support
# from anomalib.data import MVTecLOCO
# from anomalib.models import EfficientAd
# from anomalib.engine import Engine

# torch.set_float32_matmul_precision("high")

# ROOT = r"D:\project computer vision\data_roi_bg_aug"
# CKPT_PATH = r"D:\project computer vision\outputs\efficientad_roi_bg_aug\best_efficientad_roi_bg_aug_v0.ckpt"

# datamodule = MVTecLOCO(
#     root=ROOT,
#     category="splicing_connectors",
#     train_batch_size=1,
#     eval_batch_size=4,
#     num_workers=0,
# )

# model = EfficientAd(
#     imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
# )

# engine = Engine(
#     accelerator="gpu",
#     devices=1,
# )

# # If your anomalib version complains about return_predictions, remove that line.
# pred_batches = engine.predict(
#     model=model,
#     datamodule=datamodule,
#     ckpt_path=CKPT_PATH,
#     return_predictions=True,
# )

# def to_list(x):
#     if x is None:
#         return []
#     if isinstance(x, list):
#         return x
#     if hasattr(x, "detach"):
#         x = x.detach().cpu().numpy()
#     elif hasattr(x, "cpu"):
#         x = x.cpu().numpy()
#     elif hasattr(x, "numpy"):
#         x = x.numpy()

#     arr = np.array(x)
#     if arr.ndim == 0:
#         return [arr.item()]
#     return arr.reshape(-1).tolist()

# def get_group_from_path(path_str: str) -> str:
#     p = str(path_str).replace("\\", "/").lower()
#     if "/good/" in p:
#         return "good"
#     if "/logical_anomalies/" in p:
#         return "logical"
#     if "/structural_anomalies/" in p:
#         return "structural"
#     return "unknown"

# rows = []

# for batch in pred_batches:
#     paths = list(batch.image_path)
#     gt_labels = to_list(batch.gt_label)
#     pred_labels = to_list(batch.pred_label)
#     pred_scores = to_list(batch.pred_score)

#     for path, gt, pred, score in zip(paths, gt_labels, pred_labels, pred_scores):
#         group = get_group_from_path(path)
#         rows.append({
#             "image_path": str(path),
#             "group": group,
#             "gt_label": int(gt),       # 0=normal, 1=anomaly
#             "pred_label": int(pred),   # 0=normal, 1=anomaly
#             "pred_score": float(score),
#         })

# df = pd.DataFrame(rows)
# df.to_csv("per_image_predictions_aug.csv", index=False)

# def binary_stats(y_true, y_pred, name):
#     p, r, f1, _ = precision_recall_fscore_support(
#         y_true, y_pred, average="binary", zero_division=0
#     )
#     return {
#         "subset": name,
#         "precision": p,
#         "recall": r,
#         "f1": f1,
#         "n": len(y_true),
#     }

# results = []

# # 1) Overall anomaly F1
# results.append(
#     binary_stats(
#         df["gt_label"].astype(int),
#         df["pred_label"].astype(int),
#         "overall_anomaly"
#     )
# )

# # 2) Good class F1
# # Here "positive" means GOOD/NORMAL
# good_true = (df["group"] == "good").astype(int)
# good_pred = (df["pred_label"] == 0).astype(int)
# results.append(binary_stats(good_true, good_pred, "good_class"))

# # 3) Logical subset F1 = evaluate only on good + logical
# logical_df = df[df["group"].isin(["good", "logical"])].copy()
# logical_true = (logical_df["group"] == "logical").astype(int)
# logical_pred = logical_df["pred_label"].astype(int)
# results.append(binary_stats(logical_true, logical_pred, "logical_subset"))

# # 4) Structural subset F1 = evaluate only on good + structural
# struct_df = df[df["group"].isin(["good", "structural"])].copy()
# struct_true = (struct_df["group"] == "structural").astype(int)
# struct_pred = struct_df["pred_label"].astype(int)
# results.append(binary_stats(struct_true, struct_pred, "structural_subset"))

# results_df = pd.DataFrame(results)
# print(results_df)
# results_df.to_csv("f1_by_subset_aug.csv", index=False)

# print("\nCounts by group:")
# print(df["group"].value_counts())





















import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from anomalib.data import MVTecLOCO
from anomalib.models import EfficientAd
from anomalib.engine import Engine

torch.set_float32_matmul_precision("high")

ROOT = r"D:\project computer vision\data_roi_bg_aug"
CKPT_PATH = r"D:\project computer vision\outputs\efficientad_roi_bg_aug\best_efficientad_roi_bg_aug_v0.ckpt"
THRESHOLD = 0.005025

datamodule = MVTecLOCO(
    root=ROOT,
    category="splicing_connectors",
    train_batch_size=1,
    eval_batch_size=4,
    num_workers=0,
)

model = EfficientAd(
    imagenet_dir=r"D:\project computer vision\datasets\imagenette2-320\train"
)

engine = Engine(
    accelerator="gpu",
    devices=1,
)

pred_batches = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=CKPT_PATH,
    return_predictions=True,
)

def to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()
    elif hasattr(x, "numpy"):
        x = x.numpy()

    arr = np.array(x)
    if arr.ndim == 0:
        return [arr.item()]
    return arr.reshape(-1).tolist()

def get_group_from_path(path_str: str) -> str:
    p = str(path_str).replace("\\", "/").lower()
    if "/good/" in p:
        return "good"
    if "/logical_anomalies/" in p:
        return "logical"
    if "/structural_anomalies/" in p:
        return "structural"
    return "unknown"

rows = []

for batch in pred_batches:
    paths = list(batch.image_path)
    gt_labels = to_list(batch.gt_label)
    pred_scores = to_list(batch.pred_score)

    for path, gt, score in zip(paths, gt_labels, pred_scores):
        group = get_group_from_path(path)
        pred = int(float(score) >= THRESHOLD)

        rows.append({
            "image_path": str(path),
            "group": group,
            "gt_label": int(gt),       # 0=normal, 1=anomaly
            "pred_label": int(pred),   # thresholded manually
            "pred_score": float(score),
        })

df = pd.DataFrame(rows)
df.to_csv("per_image_predictions_aug_tuned.csv", index=False)

def binary_stats(y_true, y_pred, name):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "subset": name,
        "precision": p,
        "recall": r,
        "f1": f1,
        "n": len(y_true),
    }

results = []

# 1) Overall anomaly F1
results.append(
    binary_stats(
        df["gt_label"].astype(int),
        df["pred_label"].astype(int),
        "overall_anomaly"
    )
)

# 2) Good class F1 (positive = good / normal)
good_true = (df["group"] == "good").astype(int)
good_pred = (df["pred_label"] == 0).astype(int)
results.append(binary_stats(good_true, good_pred, "good_class"))

# 3) Logical subset F1 = evaluate only on good + logical
logical_df = df[df["group"].isin(["good", "logical"])].copy()
logical_true = (logical_df["group"] == "logical").astype(int)
logical_pred = logical_df["pred_label"].astype(int)
results.append(binary_stats(logical_true, logical_pred, "logical_subset"))

# 4) Structural subset F1 = evaluate only on good + structural
struct_df = df[df["group"].isin(["good", "structural"])].copy()
struct_true = (struct_df["group"] == "structural").astype(int)
struct_pred = struct_df["pred_label"].astype(int)
results.append(binary_stats(struct_true, struct_pred, "structural_subset"))

results_df = pd.DataFrame(results)
print(f"\nThreshold used: {THRESHOLD}")
print(results_df)

results_df.to_csv("f1_by_subset_aug_tuned.csv", index=False)

print("\nCounts by group:")
print(df["group"].value_counts())