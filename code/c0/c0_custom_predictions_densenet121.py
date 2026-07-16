"""
c0_custom_predictions.py
Runs DenseNet121 inference + Grad-CAM + embeddings.
Now also includes internal classifier predictions.
"""

import os
import argparse
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

disease_col = {
    "pneumothorax": "Pneumothorax",
    "effusion": "Pleural Effusion",
    "cardiomegaly": "Cardiomegaly",
}

parser = argparse.ArgumentParser()
parser.add_argument("--disease", type=str, default="effusion",
                    choices=list(disease_col.keys()))
args = parser.parse_args()

DISEASE = args.disease

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
DATA_DIR   = DATA_ROOT
SPLIT_DIR  = os.path.join(DATA_ROOT, f"{DISEASE}")
OUTPUT_DIR = os.path.join(RESULTS_DIR, f"C0_custom/{DISEASE}")
MODEL_PATH = os.path.join(OUTPUT_DIR, "c0_best.pt")

# Internal classifiers (Shallow-Deep Networks, Kaya et al., ICML 2019).
# One IC per feature stage of DenseNet121; trained with the backbone frozen
# ("SDN conversion"), then cached at IC_PATH so reruns skip training.
IC_LAYERS = [
    "denseblock1", "transition1",
    "denseblock2", "transition2",
    "denseblock3", "transition3",
    "denseblock4",
]
IC_EPOCHS = 3
IC_LR = 1e-3
IC_PATH = os.path.join(OUTPUT_DIR, "c0_internal_classifiers.pt")

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def banner(msg):
    print("\n" + "─" * 70)
    print(msg)
    print("─" * 70)


banner("CONFIG")
print(f"Disease:        {DISEASE} (column: {disease_col[DISEASE]})")
print(f"Data dir:       {DATA_DIR}")
print(f"Split dir:      {SPLIT_DIR}")
print(f"Output dir:     {OUTPUT_DIR}")
print(f"Model path:     {MODEL_PATH}")
print(f"IC layers:      {IC_LAYERS}")
print(f"IC training:    {IC_EPOCHS} epochs @ lr={IC_LR}")
print(f"IC weights:     {IC_PATH}")
print(f"Batch size:     {BATCH_SIZE}")
print(f"Device:         {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────

class CheXpertDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["Path"]
        label = float(row[disease_col[DISEASE]])

        img = Image.open(os.path.join(self.data_dir, path)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, path


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(path):
    print(f"Building DenseNet121 with 1-logit head...")
    model = models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, 1)

    print(f"Loading trained C0 weights from: {path}")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"C0 loaded: {n_params:,} parameters | eval mode | "
          f"backbone will NOT be trained again")

    return model

# ── Internal classifiers (SDN) ────────────────────────────────────────────────

ic_cache = {}


class InternalClassifier(nn.Module):
    """Feature reduction (relu + global avg pool) + single linear layer."""

    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, feats):
        x = torch.relu(feats).mean(dim=(2, 3))
        return self.fc(x).squeeze(1)


def register_ic_hooks(model):
    """Capture (detached) intermediate feature maps at each IC layer."""

    def make_hook(name):
        def hook(module, inp, out):
            ic_cache[name] = out.detach()
        return hook

    for name, module in model.features.named_children():
        if name in IC_LAYERS:
            module.register_forward_hook(make_hook(name))
            print(f"  hooked feature tap: features.{name}")


def build_internal_classifiers(model):
    # dummy pass to infer channel counts at each attachment point
    print("Running dummy forward pass to infer feature shapes...")
    with torch.no_grad():
        model.features(torch.zeros(1, 3, 224, 224, device=DEVICE))

    ics = nn.ModuleDict({
        name: InternalClassifier(ic_cache[name].shape[1])
        for name in IC_LAYERS
    }).to(DEVICE)

    print(f"{'IC layer':<14} {'feature map':<18} {'head params':>12}")
    for name in IC_LAYERS:
        c, h, w = ic_cache[name].shape[1:]
        n = sum(p.numel() for p in ics[name].parameters())
        print(f"{name:<14} {f'{c} x {h} x {w}':<18} {n:>12,}")

    total = sum(p.numel() for p in ics.parameters())
    print(f"Total IC parameters (the ONLY thing trained): {total:,}")

    return ics


def train_internal_classifiers(model, ics, df):
    """SDN conversion: backbone frozen, only the IC heads are trained."""

    banner(f"TRAINING INTERNAL CLASSIFIERS ({IC_EPOCHS} epochs, backbone frozen)")
    print(f"Training images:  {len(df):,}")
    print(f"Optimizer:        Adam(lr={IC_LR}) over IC heads only")
    print(f"Loss:             BCEWithLogits vs. ground-truth "
          f"'{disease_col[DISEASE]}' label (same target as final layer)")
    print("Backbone runs under torch.no_grad() — its weights cannot change.")

    loader = DataLoader(
        CheXpertDataset(df, DATA_DIR, transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    opt = torch.optim.Adam(ics.parameters(), lr=IC_LR)
    loss_fn = nn.BCEWithLogitsLoss()

    ics.train()

    for epoch in range(IC_EPOCHS):
        totals = {name: 0.0 for name in IC_LAYERS}

        for imgs, labels, _ in tqdm(loader, desc=f"IC epoch {epoch+1}/{IC_EPOCHS}"):
            imgs = imgs.to(DEVICE)
            labels = labels.float().to(DEVICE)

            with torch.no_grad():
                model.features(imgs)  # populates ic_cache via hooks

            losses = {
                name: loss_fn(ics[name](ic_cache[name]), labels)
                for name in IC_LAYERS
            }
            loss = sum(losses.values())

            opt.zero_grad()
            loss.backward()
            opt.step()

            for name, l in losses.items():
                totals[name] += l.item() * imgs.size(0)

        n = len(loader.dataset)
        print(f"Epoch {epoch+1}/{IC_EPOCHS} mean BCE per head "
              f"(deeper heads should be lower):")
        for name in IC_LAYERS:
            print(f"    {name:<14} {totals[name] / n:.4f}")

    ics.eval()
    print("IC training done — heads frozen to eval mode.")

# ── Grad-CAM Cache ────────────────────────────────────────────────────────────

cache = {}

# ── Hook (modern Grad-CAM style) ──────────────────────────────────────────────

def hook_fn(module, inp, out):

    # IMPORTANT: do NOT detach, do NOT clone
    acts = out  # raw tensor used by model

    cache["acts"] = torch.relu(acts).detach()

    cache["emb"] = cache["acts"].mean(dim=(2, 3)).detach()

    def save_grad(grad):
        cache["grads"] = grad.detach()

    # register hook on RAW tensor that participates in backprop
    acts.register_hook(save_grad)

# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, ics, loader, cam_dir, split=""):

    banner(f"INFERENCE ({split})")
    print(f"Images:        {len(loader.dataset):,} in {len(loader)} batches")
    print(f"Grad-CAMs to:  {cam_dir}")
    print(f"Per image: final prob + {len(IC_LAYERS)} internal probs "
          f"({', '.join('prob_' + n for n in IC_LAYERS)}) + 1024-d embedding")

    os.makedirs(cam_dir, exist_ok=True)

    rows = []

    for imgs, labels, paths in tqdm(loader, desc=f"inference {split}"):

        imgs = imgs.to(DEVICE)

        cache.clear()
        ic_cache.clear()

        model.zero_grad(set_to_none=True)

        logits = model(imgs).squeeze(1)
        probs = torch.sigmoid(logits)

        with torch.no_grad():
            ic_probs = {
                name: torch.sigmoid(ics[name](ic_cache[name])).cpu().numpy()
                for name in IC_LAYERS
            }

        logits.backward(torch.ones_like(logits))

        acts = cache["acts"]
        grads = cache["grads"]
        emb   = cache["emb"]

        # Grad-CAM
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cams = (weights * acts).sum(dim=1)
        cams = torch.relu(cams)

        cams = cams.cpu().numpy()
        emb = emb.cpu().numpy()
        probs = probs.detach().cpu().numpy()

        for i, path in enumerate(paths):

            cam = cams[i]

            # normalize per image
            cam -= cam.min()
            cam /= (cam.max() + 1e-8)

            cam_path = os.path.join(
                cam_dir,
                path.replace("/", "_").replace(".png", "_cam.npz")
            )

            np.savez_compressed(cam_path, cam=cam)

            row = {
                "path": path,
                "prob": float(probs[i]),
                "true": float(labels[i]),
                "cam_path": cam_path,
            }

            for name in IC_LAYERS:
                row[f"prob_{name}"] = float(ic_probs[name][i])

            for j, v in enumerate(emb[i]):
                row[f"emb_{j}"] = float(v)

            rows.append(row)

    return pd.DataFrame(rows)

# ── Metrics ───────────────────────────────────────────────────────────────────

def find_threshold(df):
    fpr, tpr, thr = roc_curve(df["true"], df["prob"])
    auc = roc_auc_score(df["true"], df["prob"])
    best = thr[np.argmax(tpr - fpr)]
    return best, auc


def apply_threshold(df, t):
    df = df.copy()
    df["pred"] = (df["prob"] > t).astype(int)
    df["correct"] = (df["pred"] == df["true"]).astype(int)
    df["margin"] = abs(df["prob"] - t)
    return df

# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    banner("LOADING DATA SPLITS")
    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "c0_train_split.csv"))
    val_df   = pd.read_csv(os.path.join(SPLIT_DIR, "c0_val_split.csv"))

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    for name, df in [("Train", train_df), ("Val", val_df)]:
        pos = df[disease_col[DISEASE]].sum()
        print(f"{name} positives: {pos:,.0f}/{len(df):,} "
              f"({pos / len(df):.1%})")

    train_loader = DataLoader(
        CheXpertDataset(train_df, DATA_DIR, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    val_loader = DataLoader(
        CheXpertDataset(val_df, DATA_DIR, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    banner("LOADING TRAINED C0 MODEL")
    model = load_model(MODEL_PATH)

    banner("ATTACHING INTERNAL CLASSIFIERS (SDN, Kaya et al. 2019)")
    register_ic_hooks(model)
    ics = build_internal_classifiers(model)

    if os.path.exists(IC_PATH):
        ics.load_state_dict(torch.load(IC_PATH, map_location=DEVICE))
        print("Found cached IC weights — SKIPPING IC training.")
        print("Loaded internal classifiers from:", IC_PATH)
        print("(delete this file to retrain the IC heads)")
    else:
        print("No cached IC weights found — training IC heads once now.")
        train_internal_classifiers(model, ics, train_df)
        torch.save(ics.state_dict(), IC_PATH)
        print("Saved internal classifiers to:", IC_PATH)

    # IMPORTANT: hook last conv output (DenseNet121 feature extractor)
    model.features.register_forward_hook(hook_fn)

    train_cam_dir = os.path.join(OUTPUT_DIR, "gradcam/train")
    val_cam_dir   = os.path.join(OUTPUT_DIR, "gradcam/val")

    train_df_out = run_inference(model, ics, train_loader, train_cam_dir, split="train")
    val_df_out   = run_inference(model, ics, val_loader, val_cam_dir, split="val")

    banner("METRICS")
    thresh, auc = find_threshold(train_df_out)

    print(f"Train AUC (final layer): {auc:.4f}")
    print(f"Optimal threshold: {thresh:.4f}")

    train_df_out = apply_threshold(train_df_out, thresh)
    val_df_out   = apply_threshold(val_df_out, thresh)

    for name, df in [("Train", train_df_out), ("Val", val_df_out)]:
        print(
            f"\n{name} accuracy (final layer): {df['correct'].mean():.3f} "
            f"({df['correct'].sum():,}/{len(df):,})"
        )
        print(f"{name} AUC by network depth (shallow → deep):")
        for ic in IC_LAYERS:
            ic_auc = roc_auc_score(df["true"], df[f"prob_{ic}"])
            print(f"    {ic:<14} {ic_auc:.4f}")
        print(f"    {'final':<14} {roc_auc_score(df['true'], df['prob']):.4f}")

    banner("SAVING OUTPUTS")
    train_csv = os.path.join(OUTPUT_DIR, f"train_c0_{DISEASE}.csv")
    val_csv   = os.path.join(OUTPUT_DIR, f"val_c0_{DISEASE}.csv")

    train_df_out.to_csv(train_csv, index=False)
    print(f"Wrote {train_csv} ({len(train_df_out):,} rows)")

    val_df_out.to_csv(val_csv, index=False)
    print(f"Wrote {val_csv} ({len(val_df_out):,} rows)")

    print("Internal prediction columns:",
          ", ".join(f"prob_{n}" for n in IC_LAYERS))
    print("Saved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()