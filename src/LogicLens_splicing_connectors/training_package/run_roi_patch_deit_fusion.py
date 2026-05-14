from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
)

ROOT = Path(".")
ROI_ROOT = ROOT / "training_workdir" / "logiclens_roi448"
MANIFEST = ROI_ROOT / "roi_manifest.csv"
OUT_DIR = ROOT / "training_outputs" / "logiclens_roi_patch_deit_fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 8
IMG_SIZE = 448

# Patch memory settings
PATCHES_PER_TRAIN_IMAGE = 180
MAX_MEMORY_PATCHES = 80000
PATCH_K = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.tf = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = Path(self.paths[idx])
        img = Image.open(p).convert("RGB")
        return self.tf(img), str(p)


class ResNetPatchExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "wide_resnet50_2",
            pretrained=True,
            features_only=True,
            out_indices=(2, 3),  # mid-level + high-level
        )
        self.backbone.eval()

    def forward(self, x):
        feats = self.backbone(x)
        f2, f3 = feats[0], feats[1]
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3_up], dim=1)
        f = F.normalize(f, p=2, dim=1)
        return f


class DeiTExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "deit_base_patch16_384",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.model.eval()

    def forward(self, x):
        y = self.model(x)
        y = F.normalize(y, p=2, dim=1)
        return y


def loader(paths, batch_size=BATCH_SIZE):
    return DataLoader(
        ImageDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def extract_patch_features(model, paths):
    all_img_scores_feats = []
    all_paths = []

    model.eval().to(DEVICE)

    with torch.no_grad():
        for x, ps in tqdm(loader(paths), desc="extract patch features"):
            x = x.to(DEVICE)
            f = model(x)  # B,C,H,W
            B, C, H, W = f.shape
            f = f.permute(0, 2, 3, 1).reshape(B, H * W, C)
            all_img_scores_feats.append(f.cpu().numpy())
            all_paths.extend(list(ps))

    return all_img_scores_feats, all_paths


def build_patch_memory(train_patch_batches):
    rng = np.random.RandomState(SEED)
    memory = []

    for img_patches in tqdm(train_patch_batches, desc="sample memory patches"):
        # img_patches shape B,N,C
        for patches in img_patches:
            n = patches.shape[0]
            take = min(PATCHES_PER_TRAIN_IMAGE, n)
            idx = rng.choice(n, size=take, replace=False)
            memory.append(patches[idx])

    memory = np.concatenate(memory, axis=0)

    if len(memory) > MAX_MEMORY_PATCHES:
        idx = rng.choice(len(memory), size=MAX_MEMORY_PATCHES, replace=False)
        memory = memory[idx]

    print("Patch memory:", memory.shape)
    return memory.astype(np.float32)


def patchcore_scores(patch_batches, nn_model):
    scores = []
    for batch in tqdm(patch_batches, desc="score PatchCore-like"):
        for patches in batch:
            dists, _ = nn_model.kneighbors(patches.astype(np.float32))
            d = dists.mean(axis=1)

            # top-k percent patch distance = image anomaly score
            top_n = max(1, int(0.01 * len(d)))
            score = float(np.mean(np.sort(d)[-top_n:]))
            scores.append(score)
    return np.array(scores, dtype=np.float32)


def extract_global_embeddings(model, paths):
    embs = []
    names = []

    model.eval().to(DEVICE)

    with torch.no_grad():
        for x, ps in tqdm(loader(paths), desc="extract DeiT embeddings"):
            x = x.to(DEVICE)
            y = model(x).cpu().numpy()
            embs.append(y)
            names.extend(list(ps))

    return np.concatenate(embs, axis=0).astype(np.float32), names


def knn_global_scores(train_emb, eval_emb, metric="cosine"):
    scaler = StandardScaler()
    tr = scaler.fit_transform(train_emb)
    ev = scaler.transform(eval_emb)

    nn = NearestNeighbors(n_neighbors=5, metric=metric)
    nn.fit(tr)
    d, _ = nn.kneighbors(ev)
    return d.mean(axis=1).astype(np.float32)


def robust_train_norm(train_scores, eval_scores):
    med = float(np.median(train_scores))
    q75 = float(np.quantile(train_scores, 0.75))
    q25 = float(np.quantile(train_scores, 0.25))
    iqr = max(q75 - q25, 1e-6)
    return (eval_scores - med) / iqr, {"median": med, "iqr": iqr}


def best_f1_threshold(y, s):
    thresholds = np.linspace(float(np.min(s)), float(np.max(s)), 500)
    best = {"thr": thresholds[0], "f1": -1, "precision": 0, "recall": 0}
    for t in thresholds:
        pred = (s >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        if f1 > best["f1"]:
            best = {"thr": float(t), "f1": float(f1), "precision": float(p), "recall": float(r)}
    return best


def eval_scores(name, df, score_col, split):
    sub = df[df["eval_split"] == split].copy()
    y = sub["label"].values.astype(int)
    s = sub[score_col].values.astype(float)

    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    best = best_f1_threshold(y, s)
    pred = (s >= best["thr"]).astype(int)
    cm = confusion_matrix(y, pred)

    out = {
        "name": name,
        "split": split,
        "score_col": score_col,
        "image_AUROC": float(auc),
        "image_AUPR": float(ap),
        "best_F1": float(best["f1"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "threshold": float(best["thr"]),
        "confusion_matrix": cm.tolist(),
    }

    fam_results = {}
    for fam in ["logical", "structural"]:
        fam_sub = sub[sub["family"].isin(["good", fam])]
        if fam_sub["label"].nunique() == 2:
            fam_auc = roc_auc_score(
                fam_sub["label"].values.astype(int),
                fam_sub[score_col].values.astype(float),
            )
            fam_results[f"{fam}_AUROC"] = float(fam_auc)

    out.update(fam_results)
    return out


def print_result(r):
    print("\n==========", r["name"], "|", r["split"], "|", r["score_col"], "==========")
    print("image_AUROC:", round(r["image_AUROC"], 4))
    print("image_AUPR:", round(r["image_AUPR"], 4))
    print("best_F1:", round(r["best_F1"], 4))
    print("precision:", round(r["precision"], 4))
    print("recall:", round(r["recall"], 4))
    print("threshold:", round(r["threshold"], 4))
    if "logical_AUROC" in r:
        print("logical_AUROC:", round(r["logical_AUROC"], 4))
    if "structural_AUROC" in r:
        print("structural_AUROC:", round(r["structural_AUROC"], 4))
    print("confusion_matrix:")
    print(np.array(r["confusion_matrix"]))


def main():
    print("LogicLens ROI PatchMemory + DeiT Fusion")
    print("=" * 80)
    print("Device:", DEVICE)
    print("ROI root:", ROI_ROOT.resolve())
    print("Output:", OUT_DIR.resolve())

    if not MANIFEST.exists():
        raise FileNotFoundError("Run prepare_roi_canonical_dataset.py first.")

    df = pd.read_csv(MANIFEST)

    train_df = df[df["eval_split"] == "train_good"].copy()
    val_df = df[df["eval_split"] == "val_calib"].copy()
    test_df = df[df["eval_split"] == "test_final"].copy()

    train_paths = train_df["path"].tolist()
    all_eval_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_paths = all_eval_df["path"].tolist()

    print("Train good:", len(train_df))
    print("Val calib:", len(val_df), val_df["label"].value_counts().to_dict())
    print("Test final:", len(test_df), test_df["label"].value_counts().to_dict())

    # 1) PatchCore-like local branch
    patch_model = ResNetPatchExtractor()
    train_patch_batches, _ = extract_patch_features(patch_model, train_paths)
    memory = build_patch_memory(train_patch_batches)

    nn_patch = NearestNeighbors(n_neighbors=PATCH_K, metric="euclidean")
    nn_patch.fit(memory)

    all_patch_batches, _ = extract_patch_features(patch_model, all_paths)
    patch_score_all = patchcore_scores(all_patch_batches, nn_patch)

    # Train scores are first len(train_df)
    patch_train_scores = patch_score_all[:len(train_df)]
    patch_norm_all, patch_stats = robust_train_norm(patch_train_scores, patch_score_all)

    all_eval_df["patch_score"] = patch_score_all
    all_eval_df["patch_z"] = patch_norm_all

    # Free GPU memory
    del patch_model
    torch.cuda.empty_cache()

    # 2) DeiT global logical branch
    deit = DeiTExtractor()
    train_emb, _ = extract_global_embeddings(deit, train_paths)
    all_emb, _ = extract_global_embeddings(deit, all_paths)

    deit_cos_all = knn_global_scores(train_emb, all_emb, metric="cosine")
    deit_train_scores = deit_cos_all[:len(train_df)]
    deit_norm_all, deit_stats = robust_train_norm(deit_train_scores, deit_cos_all)

    all_eval_df["deit_cos_score"] = deit_cos_all
    all_eval_df["deit_z"] = deit_norm_all

    # 3) Unsupervised fusion: no anomaly labels used
    all_eval_df["fusion_mean_z"] = 0.55 * all_eval_df["patch_z"] + 0.45 * all_eval_df["deit_z"]
    all_eval_df["fusion_max_z"] = np.maximum(all_eval_df["patch_z"], all_eval_df["deit_z"])

    # 4) Optional validation-calibrated fusion: uses val_calib labels, no test leakage
    cols = ["patch_z", "deit_z"]
    val_part = all_eval_df[all_eval_df["eval_split"] == "val_calib"].copy()
    test_part = all_eval_df[all_eval_df["eval_split"] == "test_final"].copy()

    clf = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=SEED)
    clf.fit(val_part[cols].values, val_part["label"].values.astype(int))

    all_eval_df["fusion_val_logreg"] = clf.predict_proba(all_eval_df[cols].values)[:, 1]

    # Evaluate
    results = []
    for score_col in ["patch_z", "deit_z", "fusion_mean_z", "fusion_max_z", "fusion_val_logreg"]:
        for split in ["val_calib", "test_final"]:
            r = eval_scores("logiclens_roi_patch_deit", all_eval_df, score_col, split)
            results.append(r)
            print_result(r)

    all_eval_df.to_csv(OUT_DIR / "roi_patch_deit_scores.csv", index=False)
    pd.DataFrame(results).to_csv(OUT_DIR / "roi_patch_deit_summary.csv", index=False)

    config = {
        "PATCHES_PER_TRAIN_IMAGE": PATCHES_PER_TRAIN_IMAGE,
        "MAX_MEMORY_PATCHES": MAX_MEMORY_PATCHES,
        "PATCH_K": PATCH_K,
        "IMG_SIZE": IMG_SIZE,
        "patch_stats": patch_stats,
        "deit_stats": deit_stats,
        "logreg_coef": clf.coef_.tolist(),
        "logreg_intercept": clf.intercept_.tolist(),
    }

    with open(OUT_DIR / "fusion_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nSaved:")
    print(OUT_DIR / "roi_patch_deit_scores.csv")
    print(OUT_DIR / "roi_patch_deit_summary.csv")
    print(OUT_DIR / "fusion_config.json")


if __name__ == "__main__":
    main()