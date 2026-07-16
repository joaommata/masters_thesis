# c0_predict_medmnist.py
#
# Inference-only script: loads the saved C0 checkpoint and produces
# train_c0_{disease}.csv and test_c0_{disease}.csv with Grad-CAM + embeddings.
# Run this after training is done (or killed early) — checkpoint from best
# val epoch is already saved by c0_train__densenet121_medmnist.py.
#
# Usage:
#   python c0_predict_medmnist.py --disease effusion

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
from tqdm import tqdm
from medmnist import ChestMNIST

DISEASE_IDX = {
    "atelectasis":   0, "cardiomegaly":  1, "effusion":      2,
    "infiltration":  3, "mass":          4, "nodule":        5,
    "pneumonia":     6, "pneumothorax":  7, "consolidation": 8,
    "edema":         9, "emphysema":     10, "fibrosis":     11,
    "pleural":       12, "hernia":       13,
}

parser = argparse.ArgumentParser()
parser.add_argument("--disease", type=str, default="effusion",
                    choices=list(DISEASE_IDX.keys()))
parser.add_argument("--batch", type=int, default=32)
args = parser.parse_args()

DISEASE    = args.disease
LABEL_IDX  = DISEASE_IDX[DISEASE]
BATCH_SIZE = args.batch
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
DATA_DIR   = os.path.join(DATA_ROOT, "medmnist")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "", "C0_medmnist", DISEASE)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Internal classifiers (Shallow-Deep Networks, Kaya et al., ICML 2019).
# One IC per DenseNet121 feature stage; the backbone is frozen ("SDN conversion")
# and only the IC heads are trained, then cached at IC_PATH so reruns skip it.
IC_LAYERS = [
    "denseblock1", "transition1",
    "denseblock2", "transition2",
    "denseblock3", "transition3",
    "denseblock4",
]
IC_EPOCHS = 3
IC_LR = 1e-3
IC_PATH = os.path.join(OUTPUT_DIR, "c0_internal_classifiers.pt")

print(f"Disease: {DISEASE} | Device: {DEVICE}")
print(f"Loading checkpoint from: {OUTPUT_DIR}/c0_best.pt")

# ── Dataset ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ChestMNISTSingle(Dataset):
    def __init__(self, split, label_idx, transform=None):
        self.base      = ChestMNIST(split=split, size=224, download=False,
                                    root=DATA_DIR, transform=None)
        self.label_idx = label_idx
        self.transform = transform
        self.split     = split

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, labels = self.base[idx]
        label = float(labels[self.label_idx])
        if self.transform:
            img = self.transform(img)
        return img, label, f"{self.split}/{idx:06d}"


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(path):
    model = models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()


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


def build_internal_classifiers(model):
    # dummy pass to infer channel counts at each attachment point
    with torch.no_grad():
        model.features(torch.zeros(1, 3, 224, 224, device=DEVICE))
    return nn.ModuleDict({
        name: InternalClassifier(ic_cache[name].shape[1])
        for name in IC_LAYERS
    }).to(DEVICE)


def train_internal_classifiers(model, ics, loader):
    """SDN conversion: backbone frozen, only the IC heads are trained."""
    opt = torch.optim.Adam(ics.parameters(), lr=IC_LR)
    loss_fn = nn.BCEWithLogitsLoss()
    ics.train()

    for epoch in range(IC_EPOCHS):
        total = 0.0
        n = 0
        for imgs, labels, _ in tqdm(loader, desc=f"IC epoch {epoch+1}/{IC_EPOCHS}"):
            imgs = imgs.to(DEVICE)
            labels = labels.float().to(DEVICE)

            with torch.no_grad():
                model.features(imgs)  # populates ic_cache via hooks

            loss = sum(loss_fn(ics[name](ic_cache[name]), labels)
                       for name in IC_LAYERS)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * imgs.size(0)
            n += imgs.size(0)

        print(f"IC epoch {epoch+1}: loss {total / n:.4f}")

    ics.eval()


# ── Grad-CAM + embeddings ─────────────────────────────────────────────────────
_cache = {}


def _hook_fn(module, inp, out):
    acts = out
    _cache["acts"] = torch.relu(acts).detach()
    _cache["emb"]  = _cache["acts"].mean(dim=(2, 3)).detach()

    def _save_grad(grad):
        _cache["grads"] = grad.detach()

    acts.register_hook(_save_grad)


def run_inference(model, ics, loader, cam_dir):
    os.makedirs(cam_dir, exist_ok=True)
    rows = []

    for imgs, labels, paths in tqdm(loader, desc=os.path.basename(cam_dir)):
        imgs = imgs.to(DEVICE)
        _cache.clear()
        ic_cache.clear()
        model.zero_grad(set_to_none=True)

        logits = model(imgs).squeeze(1)
        probs  = torch.sigmoid(logits)

        with torch.no_grad():
            ic_probs = {
                name: torch.sigmoid(ics[name](ic_cache[name])).cpu().numpy()
                for name in IC_LAYERS
            }

        logits.backward(torch.ones_like(logits))

        acts   = _cache["acts"]
        grads  = _cache["grads"]
        emb    = _cache["emb"].cpu().numpy()
        cams   = torch.relu((grads.mean(dim=(2, 3), keepdim=True) * acts).sum(dim=1)).cpu().numpy()
        probs  = probs.detach().cpu().numpy()

        for i, path in enumerate(paths):
            cam = cams[i]
            cam -= cam.min()
            cam /= cam.max() + 1e-8
            cam_path = os.path.join(cam_dir, path.replace("/", "_") + "_cam.npz")
            np.savez_compressed(cam_path, cam=cam)

            row = {"path": path, "prob": float(probs[i]),
                   "true": float(labels[i]), "cam_path": cam_path}
            for name in IC_LAYERS:
                row[f"prob_{name}"] = float(ic_probs[name][i])
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
    model = load_model(os.path.join(OUTPUT_DIR, "c0_best.pt"))

    train_ds = ChestMNISTSingle("train", LABEL_IDX, transform)
    test_ds  = ChestMNISTSingle("test",  LABEL_IDX, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ── Attach internal classifiers (SDN) — backbone stays frozen ──────────────
    register_ic_hooks(model)
    ics = build_internal_classifiers(model)
    if os.path.exists(IC_PATH):
        ics.load_state_dict(torch.load(IC_PATH, map_location=DEVICE))
        print(f"Loaded internal classifiers from: {IC_PATH}")
    else:
        print("No cached IC weights — training IC heads once (backbone frozen).")
        ic_train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                     shuffle=True, num_workers=4)
        train_internal_classifiers(model, ics, ic_train_loader)
        torch.save(ics.state_dict(), IC_PATH)
        print(f"Saved internal classifiers to: {IC_PATH}")

    # register Grad-CAM hook AFTER IC hooks so both fire during inference
    model.features.register_forward_hook(_hook_fn)

    train_df = run_inference(model, ics, train_loader, os.path.join(OUTPUT_DIR, "gradcam", "train"))
    test_df  = run_inference(model, ics, test_loader,  os.path.join(OUTPUT_DIR, "gradcam", "test"))

    thresh = find_threshold(train_df)
    print(f"\nThreshold (Youden's J on train): {thresh:.4f}")
    print(f"Train AUC: {roc_auc_score(train_df['true'], train_df['prob']):.4f}")
    print(f"Test  AUC: {roc_auc_score(test_df['true'],  test_df['prob']):.4f}")

    for split, df in [("Train", train_df), ("Test", test_df)]:
        print(f"{split} AUC by depth (shallow -> deep):")
        for ic in IC_LAYERS:
            print(f"    {ic:<14} {roc_auc_score(df['true'], df[f'prob_{ic}']):.4f}")

    train_df = apply_threshold(train_df, thresh)
    test_df  = apply_threshold(test_df,  thresh)

    for name, df in [("Train", train_df), ("Test", test_df)]:
        print(f"{name} accuracy: {df['correct'].mean():.3f} "
              f"({df['correct'].sum():,}/{len(df):,})")

    train_df.to_csv(os.path.join(OUTPUT_DIR, f"train_c0_{DISEASE}.csv"), index=False)
    test_df.to_csv( os.path.join(OUTPUT_DIR, f"test_c0_{DISEASE}.csv"),  index=False)

    with open(os.path.join(OUTPUT_DIR, "threshold.txt"), "w") as f:
        f.write(str(thresh))

    print(f"\nSaved to: {OUTPUT_DIR}")
