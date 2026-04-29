# c0_train_device_detector.py
# Trains a ResNet18 binary classifier for Support Device presence/absence.
# Mirrors c0_train_densenet121.py in structure.

import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR   = "/zhome/d0/a/221493/thesis/data/"
OUTPUT_DIR = "/zhome/d0/a/221493/thesis/results/device_classifier/densenet/"
N_EPOCHS   = 5
BATCH_SIZE = 32
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ─────────────────────────────────────────────────────────────────
class DeviceDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        # Assumes df already filtered to Support Devices in {0, 1}
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

# ── Transforms ──────────────────────────────────────────────────────────────
# Same as your DenseNet training — ImageNet stats, 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Model ────────────────────────────────────────────────────────────────────
def build_model():
    model = models.resnet18(weights='IMAGENET1K_V1')
    # Replace the final FC layer with a single binary output
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# FGONNA TRY DENSENET :
def build_model():
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model

# ── One epoch ────────────────────────────────────────────────────────────────
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

# ── Evaluation ───────────────────────────────────────────────────────────────
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

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load your pre-filtered CSVs (Support Devices already in {0,1})
    train_df = pd.read_csv(os.path.join(DATA_DIR, "device_train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "device_val.csv"))
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    #print class balance as %
    print("Class balance in train set:")
    print(train_df['Support Devices'].value_counts(normalize=True))
    print("Class balance in val set:")
    print(val_df['Support Devices'].value_counts(normalize=True))

    # Store losses
    train_losses = []
    val_losses = []
    val_aucs = []

    # Create weighted sampler to handle class imbalance in training set
    # Count classes
    class_counts = train_df['Support Devices'].value_counts().sort_index()

    # Weight = inverse frequency
    class_weights = 1. / class_counts
    print(class_weights)

    # Assign weight to each sample
    sample_weights = train_df['Support Devices'].map(class_weights).values

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(DeviceDataset(train_df, DATA_DIR, transform),
                              batch_size=BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=4)
    val_loader   = DataLoader(DeviceDataset(val_df,   DATA_DIR, transform),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model     = build_model().to(DEVICE)
    print(f"Model initialized on {DEVICE}")
    
    # After loading train_df, before building the criterion:
    pos_weight = torch.tensor([(train_df['Support Devices'] == 0).sum() / 
                                (train_df['Support Devices'] == 1).sum()]).to(DEVICE)
    
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("Loss function defined.")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print("Optimizer set up.")

    best_auc = 0

    print("Starting training...")
    for epoch in range(N_EPOCHS):
        train_loss          = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc   = evaluate(model, val_loader, criterion)

        # Save metrics for plotting later
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Save best model by validation AUC
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "device_best.pt"))
            print(f"  -> Saved new best model (AUC={best_auc:.4f})")


    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss")
    loss_path = os.path.join(OUTPUT_DIR, "loss.png")
    plt.savefig(loss_path)
    plt.close()

    plt.figure()
    plt.plot(val_aucs, label="Val AUC")
    plt.legend()
    plt.title("AUC")
    auc_path = os.path.join(OUTPUT_DIR, "auc.png")
    plt.savefig(auc_path)
    plt.close()
    # Plot loss and AUC curves
    # (add your own tracking lists here if needed, same pattern as DenseNet script)
    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")
    print(f"Model saved to: {OUTPUT_DIR}")