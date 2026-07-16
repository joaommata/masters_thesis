# c0_train_vit.py
import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# ── Config Constants ─────────────────────────────────────────────────────────
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
disease_col = {
    "pneumothorax": "Pneumothorax",
    "effusion": "Pleural Effusion",
    "cardiomegaly": "Cardiomegaly",
}

DATA_DIR   = DATA_ROOT + "/"                
N_EPOCHS   = 10
BATCH_SIZE = 16  # ViT is heavier, reduce batch size
LR         = 1e-4

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ───────────────────────────────────────────────────────────────────
class CheXpertDataset(Dataset):
    def __init__(self, df, data_dir, disease, disease_map, transform=None):
        # Dynamically target the requested column and clear out missing values
        self.target_col = disease_map[disease]
        self.df = df[df[self.target_col].isin([0.0, 1.0])].reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.data_dir, row["Path"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row[self.target_col], dtype=torch.float32)
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
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    # ViT classifier head is at model.heads.head
    model.heads.head = nn.Linear(model.heads.head.in_features, 1)
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
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Train ViT on CheXpert data for a specific disease.")
    parser.add_argument(
        "--disease", 
        type=str, 
        required=True, 
        choices=list(disease_col.keys()),
        help="The disease target to train the model on."
    )
    args = parser.parse_args()
    
    DISEASE = args.disease
    
    # Resolve dynamic paths using the parsed disease selection
    SPLIT_DIR   = f"{DATA_ROOT}/{DISEASE}/"
    OUTPUT_DIR = f"/work3/s251710/thesis_results/C0_vit/{DISEASE}/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting ViT training pipeline for: {DISEASE.upper()}")
    print(f"Outputs will be saved to: {OUTPUT_DIR}")

    train_losses, val_losses, val_aucs = [], [], []

    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "c0_train_split.csv"))
    val_df   = pd.read_csv(os.path.join(SPLIT_DIR, "c0_val_split.csv"))

    train_loader = DataLoader(CheXpertDataset(train_df, DATA_DIR, DISEASE, disease_col, transform),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader   = DataLoader(CheXpertDataset(val_df, DATA_DIR, DISEASE, disease_col, transform),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    model = build_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc = 0

    for epoch in range(N_EPOCHS):
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
    plt.legend(); plt.title(f"ViT Loss ({DISEASE})")
    plt.savefig(os.path.join(OUTPUT_DIR, "loss.png")); plt.close()

    plt.figure()
    plt.plot(val_aucs, label="Val AUC")
    plt.legend(); plt.title(f"ViT AUC ({DISEASE})")
    plt.savefig(os.path.join(OUTPUT_DIR, "auc.png")); plt.close()