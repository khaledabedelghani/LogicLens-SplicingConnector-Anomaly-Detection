from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

DATA_ROOT = Path("training_workdir/anomalib_splicing")
OUT_DIR = Path("training_outputs/logiclens_logic_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-6


def read_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def masks_from_rgb(img_rgb: np.ndarray):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # colored object mask: cable + connectors
    color = (s > 35) & (v > 35)

    # orange connector regions
    orange = (h >= 5) & (h <= 25) & (s > 45) & (v > 40)

    # cable-like colors: blue, yellow, red
    blue = (h >= 85) & (h <= 135) & (s > 35) & (v > 35)
    yellow = (h >= 22) & (h <= 45) & (s > 35) & (v > 35)
    red = ((h < 8) | (h > 170)) & (s > 35) & (v > 35)

    cable = blue | yellow | red

    # clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_u = cv2.morphologyEx(color.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    cable_u = cv2.morphologyEx(cable.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    orange_u = cv2.morphologyEx(orange.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)

    return color_u > 0, cable_u > 0, orange_u > 0, hsv


def connected_stats(mask: np.ndarray):
    mask_u = mask.astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u, connectivity=8)

    areas = []
    boxes = []
    centers = []

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        areas.append(area)
        boxes.append((x, y, w, h))
        centers.append(centroids[i])

    if len(areas) == 0:
        return {
            "num_components": 0,
            "largest_area": 0,
            "total_area": 0,
            "largest_ratio": 0,
            "bbox_w": 0,
            "bbox_h": 0,
            "bbox_area": 0,
        }

    areas = np.array(areas)
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2s = [b[0] + b[2] for b in boxes]
    y2s = [b[1] + b[3] for b in boxes]

    x1, y1, x2, y2 = min(xs), min(ys), max(x2s), max(y2s)

    return {
        "num_components": len(areas),
        "largest_area": float(areas.max()),
        "total_area": float(areas.sum()),
        "largest_ratio": float(areas.max() / (areas.sum() + EPS)),
        "bbox_w": float(x2 - x1),
        "bbox_h": float(y2 - y1),
        "bbox_area": float((x2 - x1) * (y2 - y1)),
    }


def gap_score_from_cable(cable_mask: np.ndarray):
    """
    Measures cable continuity.
    If cable columns have big internal gaps, score increases.
    """
    ys, xs = np.where(cable_mask)
    if len(xs) < 50:
        return 999.0

    x_min, x_max = int(xs.min()), int(xs.max())
    col_has = cable_mask[:, x_min:x_max + 1].any(axis=0).astype(np.uint8)

    # count long gaps inside cable span
    gaps = []
    in_gap = False
    start = 0

    for i, val in enumerate(col_has):
        if val == 0 and not in_gap:
            in_gap = True
            start = i
        elif val == 1 and in_gap:
            in_gap = False
            gaps.append(i - start)

    if in_gap:
        gaps.append(len(col_has) - start)

    long_gaps = [g for g in gaps if g >= 8]
    max_gap = max(long_gaps) if long_gaps else 0

    return float(max_gap)


def hue_hist(hsv: np.ndarray, mask: np.ndarray, bins: int = 18):
    h = hsv[:, :, 0][mask]
    if len(h) == 0:
        return np.zeros(bins, dtype=np.float32)
    hist, _ = np.histogram(h, bins=bins, range=(0, 180), density=True)
    return hist.astype(np.float32)


def extract_features(path: Path):
    img = read_rgb(path)
    h_img, w_img = img.shape[:2]

    color_mask, cable_mask, orange_mask, hsv = masks_from_rgb(img)

    color_stats = connected_stats(color_mask)
    cable_stats = connected_stats(cable_mask)
    orange_stats = connected_stats(orange_mask)

    cable_ys, cable_xs = np.where(cable_mask)

    if len(cable_xs) > 0:
        cable_y_mean = float(np.mean(cable_ys) / h_img)
        cable_y_std = float(np.std(cable_ys) / h_img)
        cable_x_span = float((cable_xs.max() - cable_xs.min()) / w_img)
        cable_area_pct = float(cable_mask.mean())
    else:
        cable_y_mean = 0.0
        cable_y_std = 0.0
        cable_x_span = 0.0
        cable_area_pct = 0.0

    color_area_pct = float(color_mask.mean())
    orange_area_pct = float(orange_mask.mean())
    gap = gap_score_from_cable(cable_mask)

    feats = {
        "color_area_pct": color_area_pct,
        "orange_area_pct": orange_area_pct,
        "cable_area_pct": cable_area_pct,

        "color_num_components": color_stats["num_components"],
        "color_largest_ratio": color_stats["largest_ratio"],
        "color_bbox_w_norm": color_stats["bbox_w"] / w_img,
        "color_bbox_h_norm": color_stats["bbox_h"] / h_img,
        "color_bbox_area_norm": color_stats["bbox_area"] / (w_img * h_img),

        "cable_num_components": cable_stats["num_components"],
        "cable_largest_ratio": cable_stats["largest_ratio"],
        "cable_bbox_w_norm": cable_stats["bbox_w"] / w_img,
        "cable_bbox_h_norm": cable_stats["bbox_h"] / h_img,
        "cable_bbox_area_norm": cable_stats["bbox_area"] / (w_img * h_img),

        "orange_num_components": orange_stats["num_components"],
        "orange_largest_ratio": orange_stats["largest_ratio"],
        "orange_bbox_w_norm": orange_stats["bbox_w"] / w_img,
        "orange_bbox_h_norm": orange_stats["bbox_h"] / h_img,

        "cable_y_mean": cable_y_mean,
        "cable_y_std": cable_y_std,
        "cable_x_span": cable_x_span,
        "cable_gap_max": gap,
    }

    hist = hue_hist(hsv, color_mask, bins=18)
    for i, val in enumerate(hist):
        feats[f"hue_hist_{i:02d}"] = float(val)

    return feats


def robust_stats(df: pd.DataFrame):
    med = df.median()
    mad = (df - med).abs().median()
    mad = mad.replace(0, EPS)
    return med, mad


def anomaly_score(row, med, mad):
    z = ((row - med).abs() / (1.4826 * mad + EPS)).values
    z = np.nan_to_num(z, nan=0.0, posinf=999.0, neginf=999.0)

    # Use top abnormal feature deviations instead of average all features
    z_sorted = np.sort(z)[::-1]
    top5 = z_sorted[:5]
    return float(np.mean(top5))


def best_f1_threshold(y_true, scores):
    thresholds = np.unique(scores)
    best = {"thr": None, "f1": -1, "precision": 0, "recall": 0}

    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)

        if f1 > best["f1"]:
            best = {"thr": float(thr), "f1": float(f1), "precision": float(p), "recall": float(r)}

    return best


def collect_images():
    train_good = sorted((DATA_ROOT / "train" / "good").glob("*.png"))
    test_good = sorted((DATA_ROOT / "test" / "good").glob("*.png"))
    test_anom = sorted((DATA_ROOT / "test" / "anomaly").glob("*.png"))

    return train_good, test_good, test_anom


def family_from_name(path: Path):
    name = path.name.lower()
    if name.startswith("logical"):
        return "logical"
    if name.startswith("structural"):
        return "structural"
    return "good"


def main():
    print("LOGICLENS HYBRID LOGIC EVALUATION")
    print("=" * 80)
    print("Dataset:", DATA_ROOT.resolve())
    print("Output:", OUT_DIR.resolve())
    print("Training stats source: train/good only")
    print("=" * 80)

    train_good, test_good, test_anom = collect_images()

    print("Train good:", len(train_good))
    print("Test good:", len(test_good))
    print("Test anomaly:", len(test_anom))

    if len(train_good) == 0:
        raise RuntimeError("No train/good images found.")

    train_rows = []
    for p in tqdm(train_good, desc="extract train/good features"):
        f = extract_features(p)
        f["path"] = str(p)
        train_rows.append(f)

    train_df = pd.DataFrame(train_rows)
    feature_cols = [c for c in train_df.columns if c != "path"]

    med, mad = robust_stats(train_df[feature_cols])

    test_rows = []

    for p in tqdm(test_good, desc="extract test/good features"):
        f = extract_features(p)
        row = {**f}
        row["path"] = str(p)
        row["label"] = 0
        row["family"] = "good"
        test_rows.append(row)

    for p in tqdm(test_anom, desc="extract test/anomaly features"):
        f = extract_features(p)
        row = {**f}
        row["path"] = str(p)
        row["label"] = 1
        row["family"] = family_from_name(p)
        test_rows.append(row)

    test_df = pd.DataFrame(test_rows)

    scores = []
    for _, row in test_df[feature_cols].iterrows():
        scores.append(anomaly_score(row, med, mad))

    test_df["logic_score"] = scores

    y_true = test_df["label"].values.astype(int)
    y_score = test_df["logic_score"].values.astype(float)

    auc = roc_auc_score(y_true, y_score)
    best = best_f1_threshold(y_true, y_score)

    y_pred = (y_score >= best["thr"]).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    print("\n========== LOGIC SCORE RESULTS ==========")
    print(f"image_AUROC: {auc:.4f}")
    print(f"best_threshold: {best['thr']:.4f}")
    print(f"best_F1: {best['f1']:.4f}")
    print(f"precision: {best['precision']:.4f}")
    print(f"recall: {best['recall']:.4f}")
    print("confusion_matrix:")
    print(cm)

    print("\n========== FAMILY AUC ==========")
    for fam in ["logical", "structural"]:
        sub = test_df[(test_df["family"].isin(["good", fam]))]
        if sub["label"].nunique() == 2:
            fam_auc = roc_auc_score(sub["label"].values, sub["logic_score"].values)
            print(f"{fam}_AUROC: {fam_auc:.4f}")

    train_df.to_csv(OUT_DIR / "logic_train_features.csv", index=False)
    test_df.to_csv(OUT_DIR / "logic_test_scores.csv", index=False)

    # Save top suspicious examples
    top = test_df.sort_values("logic_score", ascending=False).head(30)
    top.to_csv(OUT_DIR / "top_suspicious_examples.csv", index=False)

    print("\nSaved:")
    print(OUT_DIR / "logic_train_features.csv")
    print(OUT_DIR / "logic_test_scores.csv")
    print(OUT_DIR / "top_suspicious_examples.csv")


if __name__ == "__main__":
    main()