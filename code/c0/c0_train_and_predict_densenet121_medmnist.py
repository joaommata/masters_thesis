# c0_train_medmnist.py
#
# Train DenseNet121 on ChestMNIST (single disease) and save predictions
# with Grad-CAM activations and embeddings — matching the output format
# of c0_custom_predictions_densenet121.py so downstream C1/C2 pipelines
# can be reused without changes.
#
# Usage:
#   python c0_train_medmnist.py --disease effusion
#   python c0_train_medmnist.py --disease cardiomegaly --epochs 15

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import medmnist
from medmnist import ChestMNIST

# ── Disease index mapping ─────────────────────────────────────────────────────
# Matches ChestMNIST label order (INFO['chestmnist']['label'])
DISEASE_IDX = {
    "atelectasis":   0,
    "cardiomegaly":  1,
    "effusion":      2,
    "infiltration":  3,
    "mass":          4,
    "nodule":        5,
    "pneumonia":     6,
    "pneumothorax":  7,
    "consolidation": 8,
    "edema":         9,
    "emphysema":     10,
    "fibrosis":      11,
    "pleural":       12,
    "hernia":        13,
}

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--disease", type=str, default="effusion",
                    choices=list(DISEASE_IDX.keys()))
parser.add_argument("--epochs",  type=int, default=10)
parser.add_argument("--batch",   type=int, default=32)
parser.add_argument("--lr",      type=float, default=1e-4)
args = parser.parse_args()

DISEASE    = args.disease
LABEL_IDX  = DISEASE_IDX[DISEASE]
N_EPOCHS   = args.epochs
BATCH_SIZE = args.batch
LR         = args.lr

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
DATA_DIR   = os.path.join(DATA_ROOT, "medmnist")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "", "C0_medmnist", DISEASE)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Disease: {DISEASE} (label index {LABEL_IDX}) | Device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
# MedMNIST images are 1-channel grayscale; replicate to 3 channels for
# ImageNet-pretrained models. Use the 224×224 size for torchxrayvision
# compatibility in the downstream C1 step.

transform = transforms.Compose([
    transforms.ToTensor(),                                   # [0,255] uint8 → [0,1] float
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),         # 1-ch → 3-ch
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class ChestMNISTSingle(Dataset):
    """Wraps ChestMNIST and exposes one binary label for a single disease."""

    def __init__(self, split, label_idx, transform=None):
        self.base = ChestMNIST(split=split, size=224, download=True,
                               root=DATA_DIR, transform=None)
        self.label_idx = label_idx
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, labels = self.base[idx]        # img: PIL Image, labels: (14,) ndarray
        label = float(labels[self.label_idx])
        if self.transform:
            img = self.transform(img)
        return img, label, f"{self.split}/{idx:06d}"


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.densenet121(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


# ── Training helpers ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs).squeeze(1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).squeeze(1)
            total_loss += criterion(logits, labels).item()
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), auc


# ── Grad-CAM + embedding inference ───────────────────────────────────────────
_cache = {}


def _hook_fn(module, inp, out):
    acts = out
    _cache["acts"] = torch.relu(acts).detach()
    _cache["emb"]  = _cache["acts"].mean(dim=(2, 3)).detach()

    def _save_grad(grad):
        _cache["grads"] = grad.detach()

    acts.register_hook(_save_grad)


def run_inference(model, loader, cam_dir, device):
    os.makedirs(cam_dir, exist_ok=True)
    rows = []

    for imgs, labels, paths in tqdm(loader, desc=f"Inference {cam_dir.split('/')[-1]}"):
        imgs = imgs.to(device)
        _cache.clear()
        model.zero_grad(set_to_none=True)

        logits = model(imgs).squeeze(1)
        probs  = torch.sigmoid(logits)
        logits.backward(torch.ones_like(logits))

        acts  = _cache["acts"]
        grads = _cache["grads"]
        emb   = _cache["emb"].cpu().numpy()

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cams    = torch.relu((weights * acts).sum(dim=1)).cpu().numpy()
        probs   = probs.detach().cpu().numpy()

        for i, path in enumerate(paths):
            cam = cams[i]
            cam -= cam.min()
            cam /= cam.max() + 1e-8

            cam_path = os.path.join(cam_dir, path.replace("/", "_") + "_cam.npz")
            np.savez_compressed(cam_path, cam=cam)

            row = {"path": path, "prob": float(probs[i]),
                   "true": float(labels[i]), "cam_path": cam_path}
            for j, v in enumerate(emb[i]):
                row[f"emb_{j}"] = float(v)
            rows.append(row)

    return pd.DataFrame(rows)


def find_threshold(df):
    fpr, tpr, thr = roc_curve(df["true"], df["prob"])
    return float(thr[np.argmax(tpr - fpr)])


def apply_threshold(df, t):
    df = df.copy()
    df["pred"]    = (df["prob"] > t).astype(int)
    df["correct"] = (df["pred"] == df["true"]).astype(int)
    df["margin"]  = (df["prob"] - t).abs()
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    train_ds = ChestMNISTSingle("train", LABEL_IDX, transform)
    val_ds   = ChestMNISTSingle("val",   LABEL_IDX, transform)
    test_ds  = ChestMNISTSingle("test",  LABEL_IDX, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    pos = sum(train_ds.base.labels[:, LABEL_IDX])
    neg = len(train_ds) - pos
    print(f"Train size: {len(train_ds):,}  |  pos: {int(pos):,}  neg: {int(neg):,}")

    model     = build_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses, val_aucs = [], [], []
    best_auc = 0.0

    for epoch in range(N_EPOCHS):
        train_loss          = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_auc   = evaluate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        print(f"Epoch {epoch+1:02d}/{N_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "c0_best.pt"))
            print(f"  -> Saved best model (AUC={best_auc:.4f})")

    # ── Training curves ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(train_losses, label="Train"); ax1.plot(val_losses, label="Val")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(val_aucs, label="Val AUC"); ax2.set_title("AUC"); ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=120)
    plt.close(fig)

    # ── Inference: GradCAM + embeddings ──────────────────────────────────────
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "c0_best.pt"),
                                     map_location=DEVICE))
    model.eval()
    model.features.register_forward_hook(_hook_fn)

    # Use non-shuffled loaders for inference
    train_inf_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=4)
    test_inf_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=4)

    train_df = run_inference(model, train_inf_loader,
                             os.path.join(OUTPUT_DIR, "gradcam", "train"), DEVICE)
    test_df  = run_inference(model, test_inf_loader,
                             os.path.join(OUTPUT_DIR, "gradcam", "test"),  DEVICE)

    thresh = find_threshold(train_df)
    print(f"\nOptimal threshold (Youden's J on train): {thresh:.4f}")
    print(f"Train AUC: {roc_auc_score(train_df['true'], train_df['prob']):.4f}")
    print(f"Test  AUC: {roc_auc_score(test_df['true'],  test_df['prob']):.4f}")

    train_df = apply_threshold(train_df, thresh)
    test_df  = apply_threshold(test_df,  thresh)

    for name, df in [("Train", train_df), ("Test", test_df)]:
        acc = df["correct"].mean()
        print(f"{name} accuracy: {acc:.3f} ({df['correct'].sum():,}/{len(df):,})")

    train_df.to_csv(os.path.join(OUTPUT_DIR, f"train_c0_{DISEASE}.csv"), index=False)
    test_df.to_csv( os.path.join(OUTPUT_DIR, f"test_c0_{DISEASE}.csv"),  index=False)

    # Save threshold so downstream scripts can read it without recomputing
    with open(os.path.join(OUTPUT_DIR, "threshold.txt"), "w") as f:
        f.write(str(thresh))

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
