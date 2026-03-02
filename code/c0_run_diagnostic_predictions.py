"""
run_c0_baseline.py

Runs C0 (DenseNet121 trained on CheXpert) on the CheXpert dataset and saves
per-image predictions for a single target disease.

Output CSVs (one per split) contain:
  - path    : image path
  - prob    : C0 sigmoid probability for the target disease
  - pred    : binary prediction using Youden-optimal threshold
  - true    : ground truth label (NaN and -1 rows are dropped)
  - correct : 1 if pred == true  ← C2 training target
Usage:
    python run_c0_baseline.py
"""

import os
import torch
import torchxrayvision as xrv
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_DIR       = "/zhome/d0/a/221493/thesis"
DATA_PATH      = os.path.join(BASE_DIR, "data/CheXpert-v1.0-small")
OUTPUT_DIR     = os.path.join(BASE_DIR, "results/C0_baseline")
TARGET_DISEASE = "Cardiomegaly"
BATCH_SIZE     = 32
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────

class CheXpertDataset(xrv.datasets.CheX_Dataset):
    """CheXpert dataset with robust image loading and path return."""

    def __getitem__(self, idx):
        sample   = self.csv.iloc[idx]
        img_path = os.path.join(BASE_DIR, "data", sample["Path"])

        try:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            img = xrv.datasets.normalize(img, 255)   # [0,255] → [-1024, 1024]
            img = img.mean(axis=2, keepdims=True)     # RGB → greyscale
            img = np.transpose(img, (2, 0, 1))        # HWC → CHW
        except Exception:
            return None

        if self.transform:
            img = self.transform(img)

        return {
            "img":  torch.tensor(img, dtype=torch.float32),
            "lab":  self.labels[idx],
            "path": sample["Path"]
        }


def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


# ── Inference ──────────────────────────────────────────────────────────────────

def run_c0(model, dataset, disease_idx, split_name):
    """Run C0 on a dataset split and return a raw DataFrame (no thresholding yet)."""

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_skip_none
    )

    rows = []
    for batch in tqdm(loader, desc=f"Running C0 on {split_name}"):
        if batch is None:
            continue

        images = batch["img"].to(DEVICE)
        labels = batch["lab"].numpy()
        paths  = batch["path"]

        with torch.no_grad():
            probs = torch.sigmoid(model(images))

        disease_probs = probs[:, disease_idx].cpu().numpy()
        disease_true  = labels[:, disease_idx]

        for i in range(len(paths)):
            rows.append({
                "path": paths[i],
                "prob": disease_probs[i],
                "true": disease_true[i],
            })

    return pd.DataFrame(rows)


# ── Thresholding ───────────────────────────────────────────────────────────────

def apply_threshold(df, thresh):
    """Add pred, correct, and margin columns given a threshold."""
    df = df.copy()
    df["pred"]    = (df["prob"] > thresh).astype(int)
    df["correct"] = (df["pred"] == df["true"]).astype(int)
    df["margin"]  = abs(df["prob"] - thresh)
    return df


def find_optimal_threshold(df):
    """Find Youden-optimal threshold on a clean (no NaN/-1) DataFrame."""
    fpr, tpr, thresholds = roc_curve(df["true"], df["prob"])
    auc   = roc_auc_score(df["true"], df["prob"])
    thresh = thresholds[np.argmax(tpr - fpr)]
    return thresh, auc


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device      : {DEVICE}")
    print(f"Target      : {TARGET_DISEASE}")
    print(f"Output dir  : {OUTPUT_DIR}\n")

    # Load model
    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model = model.to(DEVICE)
    model.eval()

    disease_idx = model.pathologies.index(TARGET_DISEASE)
    print(f"'{TARGET_DISEASE}' → output index {disease_idx}\n")

    # Preprocessing
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    # Datasets
    train_dataset = CheXpertDataset(
        imgpath=DATA_PATH,
        csvpath=os.path.join(DATA_PATH, "train.csv"),
        views=["PA", "AP"],
        transform=transform
    )
    valid_dataset = CheXpertDataset(
        imgpath=DATA_PATH,
        csvpath=os.path.join(DATA_PATH, "valid.csv"),
        views=["PA", "AP"],
        transform=transform
    )
    print(f"Train samples : {len(train_dataset)}")
    print(f"Valid samples : {len(valid_dataset)}\n")

    # Run inference
    train_df = run_c0(model, train_dataset, disease_idx, "train")
    valid_df = run_c0(model, valid_dataset, disease_idx, "valid")

    # Drop NaN and uncertain (-1) labels — these cannot be used as C2 supervision
    train_clean = train_df[train_df["true"].isin([0.0, 1.0])].copy()
    valid_clean = valid_df[valid_df["true"].isin([0.0, 1.0])].copy()
    print(f"Train after dropping NaN/-1 : {len(train_clean)}")
    print(f"Valid after dropping NaN/-1 : {len(valid_clean)}\n")

    # Find optimal threshold on train only, then apply to both
    optimal_thresh, auc = find_optimal_threshold(train_clean)
    print(f"Train AUC               : {auc:.3f}")
    print(f"Optimal threshold       : {optimal_thresh:.4f}\n")

    train_clean = apply_threshold(train_clean, optimal_thresh)
    valid_clean = apply_threshold(valid_clean, optimal_thresh)

    # Report
    for name, df in [("Train", train_clean), ("Valid", valid_clean)]:
        acc = df["correct"].mean()
        counts = df["correct"].value_counts().to_dict()
        print(f"{name} → correct: {counts} | Accuracy: {acc:.3f}")

    # Save
    disease_tag = TARGET_DISEASE.lower().replace(" ", "_")
    train_out = os.path.join(OUTPUT_DIR, f"train_c0_{disease_tag}.csv")
    valid_out = os.path.join(OUTPUT_DIR, f"valid_c0_{disease_tag}.csv")

    train_clean.to_csv(train_out, index=False)
    valid_clean.to_csv(valid_out, index=False)
    print(f"\nSaved → {train_out}")
    print(f"Saved → {valid_out}")


if __name__ == "__main__":
    main()