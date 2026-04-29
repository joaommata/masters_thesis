# c0_device_predictions.py
"""
Runs the custom-trained ResNet18 device classifier on train/val splits
and saves predictions.

Output CSVs contain:
path, prob, pred, true, correct, margin
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
#need the classification report and confusion matrix for the val set
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import WeightedRandomSampler

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = "/zhome/d0/a/221493/thesis"
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "results/device_classifier/densenet")
MODEL_PATH = os.path.join(OUTPUT_DIR, "device_best.pt")

BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
class DeviceDataset(Dataset):
    """
    Dataset for Support Device classification.
    Returns image + label + path.
    """

    def __init__(self, df, data_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row   = self.df.iloc[idx]

        path  = row["Path"]
        label = float(row["Support Devices"])

        img = Image.open(
            os.path.join(self.data_dir, path)
        ).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, path


# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_path, device):

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        1
    )

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model = model.to(device)
    model.eval()

    return model

# FOR THE DENSENET VERSION:
def load_model(model_path, device):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, loader, device):

    rows = []

    with torch.no_grad():

        for imgs, labels, paths in tqdm(loader):

            imgs = imgs.to(device)

            logits = model(imgs).squeeze(1)

            probs = torch.sigmoid(logits).cpu().numpy()

            labels = np.array(labels)

            for i in range(len(paths)):

                rows.append({
                    "path": paths[i],
                    "prob": float(probs[i]),
                    "true": float(labels[i]),
                })

    return pd.DataFrame(rows)


# ── Thresholding ──────────────────────────────────────────────────────────────
def find_optimal_threshold(df):

    fpr, tpr, thresholds = roc_curve(
        df["true"],
        df["prob"]
    )

    auc = roc_auc_score(
        df["true"],
        df["prob"]
    )

    thresh = thresholds[np.argmax(tpr - fpr)]

    return thresh, auc


def apply_threshold(df, thresh):

    df = df.copy()

    df["pred"] = (
        df["prob"] > thresh
    ).astype(int)

    df["correct"] = (
        df["pred"] == df["true"]
    ).astype(int)

    df["margin"] = np.abs(
        df["prob"] - thresh
    )

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():

    # Load splits
    train_df = pd.read_csv(
        os.path.join(DATA_DIR, "device_train.csv")
    )

    val_df = pd.read_csv(
        os.path.join(DATA_DIR, "device_val.csv")
    )

    print(f"Train: {len(train_df):,}")
    print(f"Val:   {len(val_df):,}")
    

    # DataLoaders
    train_loader = DataLoader(
        DeviceDataset(train_df, DATA_DIR, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    val_loader = DataLoader(
        DeviceDataset(val_df, DATA_DIR, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # Load model
    model = load_model(MODEL_PATH, DEVICE)

    print(f"Model loaded from:")
    print(MODEL_PATH)

    # Run inference
    print("\nRunning inference on TRAIN...")
    train_raw = run_inference(
        model,
        train_loader,
        DEVICE
    )

    print("\nRunning inference on VAL...")
    val_raw = run_inference(
        model,
        val_loader,
        DEVICE
    )

    # Find threshold on train only
    thresh, auc = find_optimal_threshold(train_raw)

    print(f"\nTrain AUC: {auc:.4f}")
    print(f"Optimal threshold: {thresh:.4f}")

    # Apply threshold
    train_out = apply_threshold(
        train_raw,
        thresh
    )

    val_out = apply_threshold(
        val_raw,
        thresh
    )

    print("\nValidation Classification Report:")
    print(
        classification_report(
            val_out["true"],
            val_out["pred"],
            digits=4
        )
    )

    print("\nConfusion Matrix:")
    print(
    confusion_matrix(
        val_out["true"],
        val_out["pred"]
    )
)
    
    # Metrics
    for name, df in [
        ("Train", train_out),
        ("Val", val_out)
    ]:

        acc = df["correct"].mean()
        auc = roc_auc_score(
            df["true"],
            df["prob"]
        )

        print(
            f"{name} accuracy: {acc:.4f} "
            f"AUC: {auc:.4f} "
            f"({df['correct'].sum():,}/{len(df):,})"
        )

    # Save
    train_path = os.path.join(
        OUTPUT_DIR,
        "train_device_predictions.csv"
    )

    val_path = os.path.join(
        OUTPUT_DIR,
        "val_device_predictions.csv"
    )

    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)

    print("\nSaved predictions:")
    print(train_path)
    print(val_path)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()