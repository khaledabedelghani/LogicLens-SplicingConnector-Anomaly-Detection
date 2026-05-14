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
OUT_DIR = Path("training_outputs/logiclens_cable_inspection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-6


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def get_masks(img_rgb: np.ndarray):
    """
    Extract cable + connector masks using HSV.
    This is not supervised; it uses color/structure assumptions only.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Cable colors: red / yellow / blue-ish colored wires
    red = ((h <= 10) | (h >= 168)) & (s > 35) & (v > 35)
    yellow = (h >= 18) & (h <= 48) & (s > 35) & (v > 35)
    blue = (h >= 85) & (h <= 135) & (s > 25) & (v > 30)

    cable = red | yellow | blue

    # Orange connector-like parts
    orange = (h >= 4) & (h <= 28) & (s > 40) & (v > 35)

    # General colored foreground
    foreground = (s > 32) & (v > 32)

    # Morphological cleanup
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cable_u = cv2.morphologyEx(cable.astype(np.uint8) * 255, cv2.MORPH_OPEN, small_kernel)
    orange_u = cv2.morphologyEx(orange.astype(np.uint8) * 255, cv2.MORPH_OPEN, small_kernel)
    fg_u = cv2.morphologyEx(foreground.astype(np.uint8) * 255, cv2.MORPH_OPEN, small_kernel)

    # Connect near cable pieces but do not over-close huge missing defects
    cable_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
    cable_u = cv2.morphologyEx(cable_u, cv2.MORPH_CLOSE, cable_close_kernel, iterations=1)

    return {
        "hsv": hsv,
        "cable": cable_u > 0,
        "orange": orange_u > 0,
        "foreground": fg_u > 0,
    }


def component_stats(mask: np.ndarray, min_area: int = 30):
    mask_u = mask.astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u, connectivity=8)

    comps = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        comps.append({
            "area": float(area),
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
            "cx": float(centroids[i][0]),
            "cy": float(centroids[i][1]),
        })

    return comps


def cable_profile_features(cable_mask: np.ndarray):
    """
    Analyze cable continuity along x-axis.
    Good cable should have long horizontal presence and limited internal gaps.
    """
    H, W = cable_mask.shape
    ys, xs = np.where(cable_mask)

    if len(xs) < 50:
        return {
            "cable_present": 0.0,
            "cable_area_pct": 0.0,
            "x_span": 0.0,
            "y_mean": 0.0,
            "y_std": 0.0,
            "max_internal_gap_norm": 1.0,
            "num_internal_gaps": 999.0,
            "presence_ratio_inside_span": 0.0,
            "centerline_residual": 999.0,
            "centerline_slope_abs": 999.0,
        }

    x_min, x_max = int(xs.min()), int(xs.max())
    y_mean = float(np.mean(ys) / H)
    y_std = float(np.std(ys) / H)
    x_span = float((x_max - x_min + 1) / W)

    sub = cable_mask[:, x_min:x_max + 1]
    col_has = sub.any(axis=0).astype(np.uint8)
    span_len = len(col_has)

    # Internal gaps
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
        gaps.append(span_len - start)

    # Ignore tiny gaps from segmentation noise
    meaningful_gaps = [g for g in gaps if g >= 6]

    max_gap = max(meaningful_gaps) if meaningful_gaps else 0
    num_gaps = len(meaningful_gaps)
    presence_ratio = float(col_has.mean())

    # Centerline from median y per column
    x_vals = []
    y_vals = []
    for local_x in range(span_len):
        col = sub[:, local_x]
        yy = np.where(col)[0]
        if len(yy) > 0:
            x_vals.append(x_min + local_x)
            y_vals.append(np.median(yy))

    if len(x_vals) >= 20:
        x_arr = np.array(x_vals, dtype=np.float32)
        y_arr = np.array(y_vals, dtype=np.float32)

        x_norm = (x_arr - x_arr.min()) / (x_arr.max() - x_arr.min() + EPS)
        y_norm = y_arr / H

        # Fit line, measure residual
        coeff = np.polyfit(x_norm, y_norm, deg=1)
        y_pred = coeff[0] * x_norm + coeff[1]
        residual = float(np.mean(np.abs(y_norm - y_pred)))
        slope_abs = float(abs(coeff[0]))
    else:
        residual = 999.0
        slope_abs = 999.0

    return {
        "cable_present": 1.0,
        "cable_area_pct": float(cable_mask.mean()),
        "x_span": x_span,
        "y_mean": y_mean,
        "y_std": y_std,
        "max_internal_gap_norm": float(max_gap / max(span_len, 1)),
        "num_internal_gaps": float(num_gaps),
        "presence_ratio_inside_span": presence_ratio,
        "centerline_residual": residual,
        "centerline_slope_abs": slope_abs,
    }


def connector_overlap_features(cable_mask: np.ndarray, orange_mask: np.ndarray):
    """
    Check whether cable overlaps/touches expected left/right connector zones.
    """
    H, W = cable_mask.shape

    orange_comps = component_stats(orange_mask, min_area=50)

    if len(orange_comps) == 0:
        return {
            "orange_components": 0.0,
            "left_connector_area_pct": 0.0,
            "right_connector_area_pct": 0.0,
            "left_cable_overlap": 0.0,
            "right_cable_overlap": 0.0,
            "connector_balance": 999.0,
        }

    # Split orange components by x position
    left_area = 0.0
    right_area = 0.0
    left_mask = np.zeros_like(cable_mask, dtype=np.uint8)
    right_mask = np.zeros_like(cable_mask, dtype=np.uint8)

    for comp in orange_comps:
        x, y, w, h = int(comp["x"]), int(comp["y"]), int(comp["w"]), int(comp["h"])
        area = comp["area"]
        cx_norm = comp["cx"] / W

        if cx_norm < 0.5:
            left_area += area
            left_mask[y:y+h, x:x+w] = 1
        else:
            right_area += area
            right_mask[y:y+h, x:x+w] = 1

    # Dilate connector regions to test cable proximity
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    left_zone = cv2.dilate(left_mask, kernel, iterations=1) > 0
    right_zone = cv2.dilate(right_mask, kernel, iterations=1) > 0

    cable_area = cable_mask.sum() + EPS
    left_overlap = float((cable_mask & left_zone).sum() / cable_area)
    right_overlap = float((cable_mask & right_zone).sum() / cable_area)

    total_connector = left_area + right_area + EPS
    balance = float(abs(left_area - right_area) / total_connector)

    return {
        "orange_components": float(len(orange_comps)),
        "left_connector_area_pct": float(left_area / (H * W)),
        "right_connector_area_pct": float(right_area / (H * W)),
        "left_cable_overlap": left_overlap,
        "right_cable_overlap": right_overlap,
        "connector_balance": balance,
    }


def color_features(hsv: np.ndarray, cable_mask: np.ndarray):
    """
    Measure cable hue distribution.
    """
    h = hsv[:, :, 0][cable_mask]
    s = hsv[:, :, 1][cable_mask]
    v = hsv[:, :, 2][cable_mask]

    if len(h) == 0:
        feats = {
            "cable_h_mean": 999.0,
            "cable_h_std": 999.0,
            "cable_s_mean": 0.0,
            "cable_v_mean": 0.0,
        }
        for i in range(18):
            feats[f"cable_hue_hist_{i:02d}"] = 0.0
        return feats

    hist, _ = np.histogram(h, bins=18, range=(0, 180), density=True)

    feats = {
        "cable_h_mean": float(np.mean(h)),
        "cable_h_std": float(np.std(h)),
        "cable_s_mean": float(np.mean(s)),
        "cable_v_mean": float(np.mean(v)),
    }

    for i, val in enumerate(hist):
        feats[f"cable_hue_hist_{i:02d}"] = float(val)

    return feats


def extract_features(path: Path):
    img = read_rgb(path)
    masks = get_masks(img)

    cable = masks["cable"]
    orange = masks["orange"]
    fg = masks["foreground"]
    hsv = masks["hsv"]

    H, W = cable.shape

    cable_feats = cable_profile_features(cable)
    connector_feats = connector_overlap_features(cable, orange)
    color_feats = color_features(hsv, cable)

    fg_comps = component_stats(fg, min_area=50)
    cable_comps = component_stats(cable, min_area=30)

    feats = {}
    feats.update(cable_feats)
    feats.update(connector_feats)
    feats.update(color_feats)

    feats["foreground_area_pct"] = float(fg.mean())
    feats["foreground_components"] = float(len(fg_comps))
    feats["cable_components"] = float(len(cable_comps))

    if len(cable_comps) > 0:
        areas = np.array([c["area"] for c in cable_comps])
        feats["cable_largest_component_ratio"] = float(areas.max() / (areas.sum() + EPS))
    else:
        feats["cable_largest_component_ratio"] = 0.0

    return feats


def robust_model(train_df, feature_cols):
    med = train_df[feature_cols].median()
    mad = (train_df[feature_cols] - med).abs().median()
    mad = mad.replace(0, EPS)
    return med, mad


def score_row(row, med, mad):
    z = ((row - med).abs() / (1.4826 * mad + EPS)).values
    z = np.nan_to_num(z, nan=0.0, posinf=999.0, neginf=999.0)

    # Targeted scoring: use strongest abnormal signals but avoid averaging too many useless features
    z_sorted = np.sort(z)[::-1]
    return float(np.mean(z_sorted[:3]))


def train_threshold(train_scores):
    # Strict unsupervised threshold from train/good only
    return float(np.quantile(train_scores, 0.99))


def best_f1_threshold(y_true, scores):
    best = {"thr": None, "f1": -1, "precision": 0, "recall": 0}
    for thr in np.unique(scores):
        pred = (scores >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {"thr": float(thr), "f1": float(f1), "precision": float(p), "recall": float(r)}
    return best


def family_from_path(path: Path):
    n = path.name.lower()
    if n.startswith("logical"):
        return "logical"
    if n.startswith("structural"):
        return "structural"
    return "good"


def main():
    print("LOGICLENS CABLE-SPECIFIC INSPECTION")
    print("=" * 80)
    print("Dataset:", DATA_ROOT.resolve())
    print("Output:", OUT_DIR.resolve())
    print("Mode: unsupervised statistics from train/good only")
    print("=" * 80)

    train_good = sorted((DATA_ROOT / "train" / "good").glob("*.png"))
    test_good = sorted((DATA_ROOT / "test" / "good").glob("*.png"))
    test_anom = sorted((DATA_ROOT / "test" / "anomaly").glob("*.png"))

    print("Train good:", len(train_good))
    print("Test good:", len(test_good))
    print("Test anomaly:", len(test_anom))

    train_rows = []
    for p in tqdm(train_good, desc="extract train/good cable features"):
        f = extract_features(p)
        f["path"] = str(p)
        train_rows.append(f)

    train_df = pd.DataFrame(train_rows)
    feature_cols = [c for c in train_df.columns if c != "path"]

    med, mad = robust_model(train_df, feature_cols)

    train_scores = []
    for _, row in train_df[feature_cols].iterrows():
        train_scores.append(score_row(row, med, mad))
    train_df["cable_logic_score"] = train_scores

    unsup_thr = train_threshold(np.array(train_scores))

    test_rows = []

    for p in tqdm(test_good, desc="extract test/good cable features"):
        f = extract_features(p)
        f["path"] = str(p)
        f["label"] = 0
        f["family"] = "good"
        test_rows.append(f)

    for p in tqdm(test_anom, desc="extract test/anomaly cable features"):
        f = extract_features(p)
        f["path"] = str(p)
        f["label"] = 1
        f["family"] = family_from_path(p)
        test_rows.append(f)

    test_df = pd.DataFrame(test_rows)

    scores = []
    for _, row in test_df[feature_cols].iterrows():
        scores.append(score_row(row, med, mad))

    test_df["cable_logic_score"] = scores

    y_true = test_df["label"].values.astype(int)
    y_score = test_df["cable_logic_score"].values.astype(float)

    auc = roc_auc_score(y_true, y_score)

    # Honest train-only threshold
    y_pred_unsup = (y_score >= unsup_thr).astype(int)
    f1_unsup = f1_score(y_true, y_pred_unsup, zero_division=0)
    p_unsup = precision_score(y_true, y_pred_unsup, zero_division=0)
    r_unsup = recall_score(y_true, y_pred_unsup, zero_division=0)
    cm_unsup = confusion_matrix(y_true, y_pred_unsup)

    # Diagnostic best threshold only to see upper bound
    best = best_f1_threshold(y_true, y_score)
    y_pred_best = (y_score >= best["thr"]).astype(int)
    cm_best = confusion_matrix(y_true, y_pred_best)

    print("\n========== CABLE LOGIC RESULTS ==========")
    print(f"image_AUROC: {auc:.4f}")

    print("\n--- Unsupervised train-only threshold ---")
    print(f"threshold_from_train_99pct: {unsup_thr:.4f}")
    print(f"F1: {f1_unsup:.4f}")
    print(f"precision: {p_unsup:.4f}")
    print(f"recall: {r_unsup:.4f}")
    print("confusion_matrix:")
    print(cm_unsup)

    print("\n--- Diagnostic best-F1 threshold on test ---")
    print(f"best_threshold: {best['thr']:.4f}")
    print(f"best_F1: {best['f1']:.4f}")
    print(f"precision: {best['precision']:.4f}")
    print(f"recall: {best['recall']:.4f}")
    print("confusion_matrix:")
    print(cm_best)

    print("\n========== FAMILY AUC ==========")
    for fam in ["logical", "structural"]:
        sub = test_df[test_df["family"].isin(["good", fam])]
        if sub["label"].nunique() == 2:
            fam_auc = roc_auc_score(sub["label"].values, sub["cable_logic_score"].values)
            print(f"{fam}_AUROC: {fam_auc:.4f}")

    train_df.to_csv(OUT_DIR / "cable_train_features.csv", index=False)
    test_df.to_csv(OUT_DIR / "cable_test_scores.csv", index=False)

    top = test_df.sort_values("cable_logic_score", ascending=False).head(40)
    top.to_csv(OUT_DIR / "top_cable_suspicious_examples.csv", index=False)

    print("\nSaved:")
    print(OUT_DIR / "cable_train_features.csv")
    print(OUT_DIR / "cable_test_scores.csv")
    print(OUT_DIR / "top_cable_suspicious_examples.csv")


if __name__ == "__main__":
    main()