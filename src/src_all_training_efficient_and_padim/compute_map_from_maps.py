from pathlib import Path
import csv
import numpy as np
from PIL import Image
from scipy import ndimage

# =========================
# SETTINGS
# =========================
PROJECT_ROOT = Path(r"D:\project computer vision")
TEST_ROOT = PROJECT_ROOT / "data_roi_bg_aug" / "splicing_connectors" / "test"
GT_ROOT = PROJECT_ROOT / "data_roi_bg_aug" / "splicing_connectors" / "ground_truth"
MAP_ROOT = PROJECT_ROOT / "pred_maps"
SCORES_CSV = MAP_ROOT / "pred_scores.csv"
OUT_CSV = PROJECT_ROOT / "map_eval_results_raw.csv"

RAW_MAP_BIN_THR = 0.005
MIN_BLOB_SIZE = 20

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def make_test_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return "__".join(rel.parts)


def make_gt_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "000":
        parts = parts[:-1]
    return "__".join(parts)


def load_raw_map(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    return arr.astype(np.float32)


def load_binary_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    return arr > 0


def resize_float_map(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((w, h), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return inter / union


def voc_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = precisions[recalls >= r]
        ap += (p.max() if p.size > 0 else 0.0)
    return ap / 101.0


def compute_ap(records, iou_thr: float) -> float:
    npos = sum(r["has_gt"] for r in records)
    if npos == 0:
        return 0.0

    preds = [r for r in records if r["has_pred"]]
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)

    tp = []
    fp = []

    for r in preds:
        if r["has_gt"] and r["iou"] >= iou_thr:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    if len(tp) == 0:
        return 0.0

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recalls = tp / npos
    precisions = tp / np.maximum(tp + fp, 1e-9)

    return voc_ap_101(recalls, precisions)
def make_test_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return "__".join(rel.parts)

def make_gt_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)

    if parts and parts[-1] == "000":
        parts = parts[:-1]

    return "__".join(parts)

def clean_mask(pred_mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(pred_mask)
    if num > 0:
        sizes = ndimage.sum(pred_mask, labeled, range(1, num + 1))
        cleaned = np.zeros_like(pred_mask, dtype=bool)
        for i, size in enumerate(sizes, start=1):
            if size >= MIN_BLOB_SIZE:
                cleaned |= (labeled == i)
        pred_mask = cleaned

    pred_mask = ndimage.binary_closing(pred_mask, structure=np.ones((3, 3)))
    return pred_mask


def main():
    if not TEST_ROOT.exists():
        raise FileNotFoundError(f"TEST_ROOT not found: {TEST_ROOT}")
    if not GT_ROOT.exists():
        raise FileNotFoundError(f"GT_ROOT not found: {GT_ROOT}")
    if not MAP_ROOT.exists():
        raise FileNotFoundError(f"MAP_ROOT not found: {MAP_ROOT}")

    test_files = []
    for ext in IMAGE_EXTS:
        test_files.extend(TEST_ROOT.rglob(f"*{ext}"))
    test_files = sorted(test_files)

    if not test_files:
        raise FileNotFoundError(f"No test images found inside: {TEST_ROOT}")

    npy_files = sorted(MAP_ROOT.rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No raw NPY maps found inside: {MAP_ROOT}")

    gt_files = []
    for ext in IMAGE_EXTS:
        gt_files.extend(GT_ROOT.rglob(f"*{ext}"))
    gt_files = sorted(gt_files)

    map_index_by_key = {p.stem: p for p in npy_files}
    gt_index_by_key = {make_gt_key(p, GT_ROOT): p for p in gt_files}

    score_by_key = {}
    if SCORES_CSV.exists():
        with open(SCORES_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                score_by_key[row["key"]] = float(row["pred_score"])

    records = []
    missing_maps = []
    missing_gts_for_anomalies = []

    for test_path in test_files:
        key = make_test_key(test_path, TEST_ROOT)
        map_path = map_index_by_key.get(key, None)
        gt_path = gt_index_by_key.get(key, None)

        is_good = "good" in test_path.parts
        is_anomaly = not is_good

        if map_path is None:
            missing_maps.append(str(test_path))
            continue

        anomaly_map = load_raw_map(map_path)

        if gt_path is not None:
            gt_mask = load_binary_mask(gt_path)
            if anomaly_map.shape != gt_mask.shape:
                anomaly_map = resize_float_map(anomaly_map, gt_mask.shape)
        else:
            gt_mask = np.zeros_like(anomaly_map, dtype=bool)
            if is_anomaly:
                missing_gts_for_anomalies.append(str(test_path))

        pred_mask = anomaly_map >= RAW_MAP_BIN_THR
        pred_mask = clean_mask(pred_mask)

        has_gt = bool(gt_mask.any())
        has_pred = bool(pred_mask.any())
        score = score_by_key.get(key, float(anomaly_map.max()) if has_pred else 0.0)
        iou = iou_score(pred_mask, gt_mask) if (has_gt or has_pred) else 1.0

        records.append({
            "image_id": key,
            "test_path": str(test_path),
            "gt_path": str(gt_path) if gt_path is not None else "",
            "map_path": str(map_path),
            "score": score,
            "has_gt": has_gt,
            "has_pred": has_pred,
            "iou": iou,
        })

    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    ap_values = {thr: compute_ap(records, thr) for thr in iou_thresholds}

    ap50 = ap_values[0.50]
    ap50_95 = float(np.mean(list(ap_values.values())))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id", "score", "has_gt", "has_pred", "iou",
            "test_path", "gt_path", "map_path"
        ])
        for r in records:
            writer.writerow([
                r["image_id"],
                f"{r['score']:.6f}",
                int(r["has_gt"]),
                int(r["has_pred"]),
                f"{r['iou']:.6f}",
                r["test_path"],
                r["gt_path"],
                r["map_path"],
            ])

    print("\n==============================")
    print(f"Images evaluated           : {len(records)}")
    print(f"Missing maps               : {len(missing_maps)}")
    print(f"Missing GT for anomalies   : {len(missing_gts_for_anomalies)}")
    print(f"Raw map threshold          : {RAW_MAP_BIN_THR:.6f}")
    print(f"Min blob size              : {MIN_BLOB_SIZE}")
    print("------------------------------")
    for thr in iou_thresholds:
        print(f"AP@{thr:.2f} = {ap_values[thr]:.6f}")
    print("------------------------------")
    print(f"mAP@50      = {ap50:.6f}")
    print(f"mAP@50:95   = {ap50_95:.6f}")
    print(f"CSV saved   = {OUT_CSV}")
    print("==============================\n")

    if missing_maps:
        print("First missing map examples:")
        for x in missing_maps[:10]:
            print(" -", x)

    if missing_gts_for_anomalies:
        print("\nFirst anomaly images with missing GT:")
        for x in missing_gts_for_anomalies[:10]:
            print(" -", x)


if __name__ == "__main__":
    main()