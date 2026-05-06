# c0_train_device_detector_frozen.py
# Trains a DenseNet121 binary classifier for Support Device presence/absence.

import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR        = "/zhome/d0/a/221493/thesis/data/"
OUTPUT_DIR      = "/zhome/d0/a/221493/thesis/results/device_classifier/densenet/"
N_EPOCHS_FROZEN = 3
N_EPOCHS_FULL   = 7
LR_HEAD         = 1e-3
LR_FULL         = 1e-4
BATCH_SIZE      = 32
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
class DeviceDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(os.path.join(self.data_dir, row["Path"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(float(row["Support Devices"]), dtype=torch.float32)
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
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier.in_features, 1)
    )
    return model

def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    # classifier is now Sequential(Dropout, Linear)
    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

# ── One epoch ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs).squeeze(1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_probs, all_labels = 0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs).squeeze(1)
            total_loss += criterion(logits, labels).item()
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), auc

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    train_df = pd.read_csv(os.path.join(DATA_DIR, "device_train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "device_val.csv"))

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    print("Train balance:\n", train_df['Support Devices'].value_counts(normalize=True))

    # ── pos_weight ────────────────────────────────────────────────────────────
    n_neg      = (train_df['Support Devices'] == 0).sum()
    n_pos      = (train_df['Support Devices'] == 1).sum()
    pos_weight = torch.tensor([n_pos / n_neg], dtype=torch.float32).to(DEVICE)
    print(f"pos_weight: {pos_weight.item():.4f}")

    train_loader = DataLoader(
        DeviceDataset(train_df, DATA_DIR, transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        DeviceDataset(val_df, DATA_DIR, transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model     = build_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses, val_losses, val_aucs = [], [], []
    best_auc = 0

    # ── Phase 1: frozen backbone ──────────────────────────────────────────────
    print(f"\n── Phase 1: Frozen backbone ({N_EPOCHS_FROZEN} epochs, LR={LR_HEAD}) ──")
    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD
    )

    for epoch in range(N_EPOCHS_FROZEN):
        train_loss        = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        print(f"  Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "device_best.pt"))
            print(f"  -> Saved best model (AUC={best_auc:.4f})")

    # ── Phase 2: full fine-tuning ─────────────────────────────────────────────
    print(f"\n── Phase 2: Full fine-tuning ({N_EPOCHS_FULL} epochs, LR={LR_FULL}) ──")
    unfreeze_all(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2,
    )

    for epoch in range(N_EPOCHS_FULL):
        train_loss        = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        scheduler.step(val_auc)
        print(f"  Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "device_best.pt"))
            print(f"  -> Saved best model (AUC={best_auc:.4f})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses,   label="Val Loss")
    ax1.axvline(x=N_EPOCHS_FROZEN - 0.5, color='gray', linestyle='--', label="Unfreeze")
    ax1.legend(); ax1.set_title("Loss")
    ax2.plot(val_aucs, label="Val AUC")
    ax2.axvline(x=N_EPOCHS_FROZEN - 0.5, color='gray', linestyle='--', label="Unfreeze")
    ax2.legend(); ax2.set_title("AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))
    plt.close()

    print(f"\nDone. Best Val AUC: {best_auc:.4f}")
    print(f"Model saved to: {OUTPUT_DIR}")