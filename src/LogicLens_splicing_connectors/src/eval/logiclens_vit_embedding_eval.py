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


DATA_ROOTS = {
    "full": Path("training_workdir/anomalib_splicing"),
    "object_focus": Path("training_workdir/anomalib_splicing_object_focus"),
}

OUT_DIR = Path("training_outputs/logiclens_vit_embedding")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.tf = T.Compose([
            T.Resize((384, 384)),
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
    ds = ImageDataset(paths)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    embs = []
    names = []

    model.eval()
    with torch.no_grad():
        for x, p in tqdm(dl, desc="extract embeddings"):
            x = x.to(DEVICE)
            y = model(x)
            embs.append(y.detach().cpu().numpy())
            names.extend(list(p))

    return np.concatenate(embs, axis=0), names


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
    n = Path(path).name.lower()
    if n.startswith("logical"):
        return "logical"
    if n.startswith("structural"):
        return "structural"
    return "good"


def evaluate(name, y_true, scores, families):
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
        "name": name,
        "image_AUROC": auc,
        "best_F1": best["f1"],
        "precision": best["precision"],
        "recall": best["recall"],
        "threshold": best["thr"],
    }


def run_one_dataset(dataset_name, root, model_name):
    print("\n" + "=" * 100)
    print(f"DATASET: {dataset_name}")
    print(f"MODEL: {model_name}")
    print("ROOT:", root.resolve())
    print("=" * 100)

    train_good = sorted((root / "train" / "good").glob("*.png"))
    test_good = sorted((root / "test" / "good").glob("*.png"))
    test_anom = sorted((root / "test" / "anomaly").glob("*.png"))

    print("Train good:", len(train_good))
    print("Test good:", len(test_good))
    print("Test anomaly:", len(test_anom))

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    ).to(DEVICE)

    train_emb, _ = extract_embeddings(model, train_good)
    test_paths = test_good + test_anom
    test_emb, test_paths = extract_embeddings(model, test_paths)

    y_true = np.array([0] * len(test_good) + [1] * len(test_anom))
    families = [family_from_path(p) for p in test_paths]

    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_emb)
    test_z = scaler.transform(test_emb)

    results = []
    all_scores = {}

    for metric in ["cosine", "euclidean"]:
        nn = NearestNeighbors(n_neighbors=5, metric=metric)
        nn.fit(train_z)
        dist, _ = nn.kneighbors(test_z)
        score = dist.mean(axis=1)
        score_name = f"{dataset_name}_{model_name}_{metric}"
        results.append(evaluate(score_name, y_true, score, families))
        all_scores[metric] = score

    center = train_z.mean(axis=0)
    center_score = np.linalg.norm(test_z - center[None, :], axis=1)
    results.append(evaluate(f"{dataset_name}_{model_name}_center", y_true, center_score, families))
    all_scores["center"] = center_score

    def norm01(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    combined = (
        norm01(all_scores["cosine"])
        + norm01(all_scores["euclidean"])
        + norm01(all_scores["center"])
    ) / 3.0

    results.append(evaluate(f"{dataset_name}_{model_name}_combined", y_true, combined, families))

    out = pd.DataFrame({
        "path": test_paths,
        "label": y_true,
        "family": families,
        "cosine": all_scores["cosine"],
        "euclidean": all_scores["euclidean"],
        "center": all_scores["center"],
        "combined": combined,
    })

    safe_model = model_name.replace("/", "_")
    out.to_csv(OUT_DIR / f"{dataset_name}_{safe_model}_scores.csv", index=False)

    return results


def main():
    print("VIT / DINO-LIKE EMBEDDING KNN EVAL")
    print("Device:", DEVICE)

    # Start with a strong ViT available in timm.
    model_names = [
        "vit_base_patch16_384",
        "deit_base_patch16_384",
    ]

    all_results = []

    for model_name in model_names:
        for ds_name, root in DATA_ROOTS.items():
            if root.exists():
                try:
                    all_results.extend(run_one_dataset(ds_name, root, model_name))
                except Exception as e:
                    print(f"\nFAILED: {model_name} on {ds_name}")
                    print(e)

    pd.DataFrame(all_results).to_csv(OUT_DIR / "vit_embedding_summary.csv", index=False)

    print("\nSaved summary:")
    print(OUT_DIR / "vit_embedding_summary.csv")


if __name__ == "__main__":
    main()