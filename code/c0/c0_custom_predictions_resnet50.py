# c0_custom_predictions_resnet50.py

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

# ── Config ────────────────────────────────────────────────────────────────────

DISEASE = 'effusion'
disease_col = {
    "pneumothorax": "Pneumothorax", 
    "effusion": "Pleural Effusion",
    "cardiomegaly": "Cardiomegaly",
}


DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
DATA_DIR   = DATA_ROOT
SPLIT_DIR  = os.path.join(DATA_ROOT, f"{DISEASE}")
OUTPUT_DIR = os.path.join(RESULTS_DIR, f"C0_resnet50/{DISEASE}")
MODEL_PATH = os.path.join(RESULTS_DIR, f"C0_resnet50/{DISEASE}/c0_best.pt")
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
class CheXpertEffusionDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df       = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = row["Path"]
        label = float(row[disease_col[DISEASE]])
        img   = Image.open(os.path.join(self.data_dir, path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_path, device):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 1)  # match training head
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, loader, embeddings_cache, device):
    rows = []
    for imgs, labels, paths in tqdm(loader):
        imgs = imgs.to(device)
        with torch.no_grad():
            logits = model(imgs).squeeze(1)  # hook fires here
            probs  = torch.sigmoid(logits).cpu().numpy()

        emb_batch = embeddings_cache['last']  # (B, 2048)

        for i in range(len(paths)):
            row = {
                "path": paths[i],
                "prob": float(probs[i]),
                "true": float(labels[i]),
            }
            for j, val in enumerate(emb_batch[i]):
                row[f"emb_{j}"] = float(val)
            rows.append(row)

    return pd.DataFrame(rows)

# ── Thresholding ──────────────────────────────────────────────────────────────
def find_optimal_threshold(df):
    fpr, tpr, thresholds = roc_curve(df["true"], df["prob"])
    auc    = roc_auc_score(df["true"], df["prob"])
    thresh = thresholds[np.argmax(tpr - fpr)]
    return thresh, auc

def apply_threshold(df, thresh):
    df = df.copy()
    df["pred"]    = (df["prob"] > thresh).astype(int)
    df["correct"] = (df["pred"] == df["true"]).astype(int)
    df["margin"]  = np.abs(df["prob"] - thresh)
    return df

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "c0_train_split.csv"))
    val_df   = pd.read_csv(os.path.join(SPLIT_DIR, "c0_val_split.csv"))
    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    train_loader = DataLoader(CheXpertEffusionDataset(train_df, DATA_DIR, transform),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    val_loader   = DataLoader(CheXpertEffusionDataset(val_df, DATA_DIR, transform),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = load_model(MODEL_PATH, DEVICE)
    print(f"Model loaded from {MODEL_PATH}")

    # Hook on layer4 — ResNet's final conv block, output (B, 2048, 7, 7)
    embeddings_cache = {}
    def hook_fn(module, input, output):
        embeddings_cache['last'] = output.mean(dim=[2, 3]).cpu().numpy()
    model.layer4.register_forward_hook(hook_fn)

    train_raw = run_inference(model, train_loader, embeddings_cache, DEVICE)
    val_raw   = run_inference(model, val_loader,   embeddings_cache, DEVICE)

    thresh, auc = find_optimal_threshold(train_raw)
    print(f"Train AUC: {auc:.4f}  |  Optimal threshold: {thresh:.4f}")

    train_out = apply_threshold(train_raw, thresh)
    val_out   = apply_threshold(val_raw,   thresh)

    for name, df in [("Train", train_out), ("Val", val_out)]:
        print(f"{name} accuracy: {df['correct'].mean():.3f}  "
              f"(correct={df['correct'].sum():,} / {len(df):,})")

    train_out.to_csv(os.path.join(OUTPUT_DIR, f"train_c0_{DISEASE}.csv"), index=False)
    val_out.to_csv(  os.path.join(OUTPUT_DIR, f"val_c0_{DISEASE}.csv"),   index=False)
    
    # save the threshold in a txt file for later use in C1 matching and C2 pipeline
    with open(os.path.join(OUTPUT_DIR, "threshold.txt"), "w") as f:
        f.write(f"{thresh:.4f}")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()