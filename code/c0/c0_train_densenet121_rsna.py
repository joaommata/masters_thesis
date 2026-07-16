# c0_train_densenet121_rsna.py
"""
C0 DenseNet121 classifier for the RSNA Pneumonia binary task
('Lung Opacity' vs 'Normal'; the 'No Lung Opacity / Not Normal' class is
excluded upstream in c0_rsna_split.py).

Same structure/hyperparameters as c0_train_densenet121.py -- the only real
difference is the dataset reads DICOM instead of JPEG.
Run c0_rsna_split.py once first to create the split CSVs.
"""
import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ── Config Constants ─────────────────────────────────────────────────────────
DATA_ROOT = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
DATA_DIR = DATA_ROOT + "/"
TARGET_COL = "Pneumonia"

N_EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ───────────────────────────────────────────────────────────────────
class RSNAPneumoniaDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df[df[TARGET_COL].isin([0.0, 1.0])].reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # RSNA ships 12-bit-ish MONOCHROME2 DICOM; rescale to 8-bit so the
        # ImageNet normalization below sees the same range as the CheXpert JPEGs.
        arr = pydicom.dcmread(os.path.join(self.data_dir, row["Path"])).pixel_array
        arr = arr.astype(np.float32)
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")

        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row[TARGET_COL], dtype=torch.float32)
        return img, label

# ── Transforms ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.densenet121(weights='IMAGENET1K_V1')
    # Replace classifier head with single binary output
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model

# ── One epoch ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs).squeeze(1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).squeeze(1)
            total_loss += criterion(logits, labels).item()
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), auc

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DenseNet121 on RSNA Pneumonia (Lung Opacity vs Normal)."
    )
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    SPLIT_DIR = f"{DATA_ROOT}/rsna_pneumonia/"
    OUTPUT_DIR = "/work3/s251710/thesis_results/C0_custom/rsna_pneumonia/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting training pipeline for: RSNA_PNEUMONIA (Lung Opacity vs Normal)")
    print(f"Outputs will be saved to: {OUTPUT_DIR}")

    train_losses, val_losses, val_aucs = [], [], []

    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "c0_train_split.csv"))
    val_df = pd.read_csv(os.path.join(SPLIT_DIR, "c0_val_split.csv"))
    print(f"train: {len(train_df):,} | val: {len(val_df):,} "
          f"| prevalence {train_df[TARGET_COL].mean():.1%}")

    train_loader = DataLoader(RSNAPneumoniaDataset(train_df, DATA_DIR, transform),
                              batch_size=args.batch_size, shuffle=True, num_workers=4,
                              persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(RSNAPneumoniaDataset(val_df, DATA_DIR, transform),
                            batch_size=args.batch_size, shuffle=False, num_workers=4,
                            persistent_workers=True, pin_memory=True)

    model = build_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_auc = evaluate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "c0_best.pt"))
            print(f"  -> Saved new best model (AUC={best_auc:.4f})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.title("Loss (rsna_pneumonia)")
    plt.savefig(os.path.join(OUTPUT_DIR, "loss.png")); plt.close()

    plt.figure()
    plt.plot(val_aucs, label="Val AUC")
    plt.legend(); plt.title("AUC (rsna_pneumonia)")
    plt.savefig(os.path.join(OUTPUT_DIR, "auc.png")); plt.close()
