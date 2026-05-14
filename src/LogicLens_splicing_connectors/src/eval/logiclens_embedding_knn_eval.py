from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix


DATA_ROOT = Path("training_workdir/anomalib_splicing_object_focus")
OUT_DIR = Path("training_outputs/logiclens_embedding_knn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ImagePathDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.tf = T.Compose([
            T.Resize((384, 768)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.tf(img), str(p)


def extract_embeddings(model, paths):
    ds = ImagePathDataset(paths)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_embs = []
    all_paths = []

    model.eval()
    with torch.no_grad():
        for x, p in tqdm(dl, desc="extract embeddings"):
            x = x.to(DEVICE)
            emb = model(x)
            emb = emb.detach().cpu().numpy()
            all_embs.append(emb)
            all_paths.extend(list(p))

    return np.concatenate(all_embs, axis=0), all_paths


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


def family_from_path(path):
    name = Path(path).name.lower()
    if name.startswith("logical"):
        return "logical"
    if name.startswith("structural"):
        return "structural"
    return "good"


def evaluate_score(name, y_true, scores, families):
    auc = roc_auc_score(y_true, scores)
    best = best_f1_threshold(y_true, scores)
    pred = (scores >= best["thr"]).astype(int)
    cm = confusion_matrix(y_true, pred)

    print(f"\n========== {name} ==========")
    print(f"image_AUROC: {auc:.4f}")
    print(f"best_threshold: {best['thr']:.4f}")
    print(f"best_F1: {best['f1']:.4f}")
    print(f"precision: {best['precision']:.4f}")
    print(f"recall: {best['recall']:.4f}")
    print("confusion_matrix:")
    print(cm)

    fam_arr = np.array(families)
    for fam in ["logical", "structural"]:
        idx = (fam_arr == "good") | (fam_arr == fam)
        if len(np.unique(y_true[idx])) == 2:
            fam_auc = roc_auc_score(y_true[idx], scores[idx])
            print(f"{fam}_AUROC: {fam_auc:.4f}")

    return {
        "score_name": name,
        "image_AUROC": auc,
        "best_F1": best["f1"],
        "precision": best["precision"],
        "recall": best["recall"],
        "threshold": best["thr"],
    }


def main():
    print("LOGICLENS EMBEDDING KNN UNSUPERVISED EVAL")
    print("=" * 80)
    print("Dataset:", DATA_ROOT.resolve())
    print("Device:", DEVICE)
    print("Mode: pretrained features + train/good kNN only")
    print("=" * 80)

    train_good = sorted((DATA_ROOT / "train" / "good").glob("*.png"))
    test_good = sorted((DATA_ROOT / "test" / "good").glob("*.png"))
    test_anom = sorted((DATA_ROOT / "test" / "anomaly").glob("*.png"))

    print("Train good:", len(train_good))
    print("Test good:", len(test_good))
    print("Test anomaly:", len(test_anom))

    model = timm.create_model(
        "wide_resnet50_2",
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    ).to(DEVICE)

    train_emb, train_paths = extract_embeddings(model, train_good)
    test_paths = test_good + test_anom
    test_emb, test_paths = extract_embeddings(model, test_paths)

    y_true = np.array([0] * len(test_good) + [1] * len(test_anom))
    families = [family_from_path(p) for p in test_paths]

    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_emb)
    test_z = scaler.transform(test_emb)

    results = []

    # Euclidean kNN distance
    nn_euc = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn_euc.fit(train_z)
    dist_euc, _ = nn_euc.kneighbors(test_z)
    score_euc = dist_euc.mean(axis=1)
    results.append(evaluate_score("embedding_knn_euclidean", y_true, score_euc, families))

    # Cosine kNN distance
    nn_cos = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn_cos.fit(train_z)
    dist_cos, _ = nn_cos.kneighbors(test_z)
    score_cos = dist_cos.mean(axis=1)
    results.append(evaluate_score("embedding_knn_cosine", y_true, score_cos, families))

    # Mean distance score
    center = train_z.mean(axis=0)
    score_center = np.linalg.norm(test_z - center[None, :], axis=1)
    results.append(evaluate_score("embedding_center_distance", y_true, score_center, families))

    # Combined normalized score
    def norm01(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    score_combined = (
        norm01(score_euc)
        + norm01(score_cos)
        + norm01(score_center)
    ) / 3.0

    results.append(evaluate_score("embedding_combined", y_true, score_combined, families))

    out_df = pd.DataFrame({
        "path": test_paths,
        "label": y_true,
        "family": families,
        "score_euclidean": score_euc,
        "score_cosine": score_cos,
        "score_center": score_center,
        "score_combined": score_combined,
    })

    out_df.to_csv(OUT_DIR / "embedding_test_scores.csv", index=False)
    pd.DataFrame(results).to_csv(OUT_DIR / "embedding_summary.csv", index=False)

    print("\nSaved:")
    print(OUT_DIR / "embedding_test_scores.csv")
    print(OUT_DIR / "embedding_summary.csv")


if __name__ == "__main__":
    main()