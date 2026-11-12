"""
c0_multilabel_predictions.py
Inference + Grad-CAM + embeddings for the 14-label C0.

Port of c0_custom_predictions_densenet121.py to the multi-label model trained by
c0_train_densenet121_multilabel.py, minus the SDN internal classifiers (dropped
from the project). Output schema:

    path, prob, true, cam_path, emb_0..emb_1023,
    pred, correct, margin, label_raw, certain

which is the single-label schema without the seven prob_<IC layer> columns. That
is safe downstream: c2_feature_spec.C0_DERIVED_COLS is only ever subtracted from
the frame to form meta_cols, so the missing names are a no-op there.

What changes for a multi-label backbone
---------------------------------------
* `prob` is sigmoid(logit[target]) for ONE chosen observation (--disease), pulled
  out of the 14 by name via labels.json, which the training script wrote next to
  the weights. The label order is the checkpoint's contract; never hardcode it.
* Grad-CAM backpropagates the TARGET logit only. Backpropagating all 14 (or their
  sum) would produce a saliency map for "any finding", not for the disease.

Two columns the single-label script could not have: `label_raw` (the original
1/0/-1/blank CheXpert value) and `certain` (1 iff label_raw was an explicit 0 or
1). This cohort, unlike the old per-disease splits, keeps blank-labelled studies
as negatives -- see the note in c0_multilabel_split.py. Filtering to certain == 1
reproduces the old, easier, explicitly-mentioned-only cohort for comparison.

The 14 probabilities go to a SEPARATE all_probs_<split>.csv keyed by path. They are
deliberately NOT in the main CSV: C2 builds its design matrix by subtracting a
meta-column set from the frame, so any new numeric column there would silently
enter the feature space.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

disease_col = {
    "pneumothorax": "Pneumothorax",
    "effusion": "Pleural Effusion",
    "cardiomegaly": "Cardiomegaly",
    "atelectasis": "Atelectasis",
    "consolidation": "Consolidation",
    "edema": "Edema",
    "pneumonia": "Pneumonia",
}

parser = argparse.ArgumentParser()
parser.add_argument("--disease", type=str, default="effusion",
                    choices=list(disease_col.keys()))
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--no-cam", action="store_true",
                    help="Skip Grad-CAM computation and .npz writing (much faster).")
parser.add_argument("--threshold-mode", choices=["youden", "fixed"], default="youden",
                    help="youden: maximise TPR-FPR on train. fixed: 0.5.")
parser.add_argument("--limit", type=int, default=None,
                    help="Debug: only the first N rows of each split.")
args = parser.parse_args()

DISEASE = args.disease
TARGET_COL = disease_col[DISEASE]

DATA_ROOT = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
DATA_DIR = DATA_ROOT
SPLIT_DIR = os.path.join(DATA_ROOT, "chexpert_multilabel")
MODEL_DIR = os.path.join(RESULTS_DIR, "C0_custom/chexpert_multilabel")
OUTPUT_DIR = os.path.join(MODEL_DIR, DISEASE)
MODEL_PATH = os.path.join(MODEL_DIR, "c0_best.pt")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

BATCH_SIZE = args.batch_size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def banner(msg):
    print("\n" + "─" * 70)
    print(msg)
    print("─" * 70)


with open(LABELS_PATH) as f:
    spec = json.load(f)
LABELS = spec["labels"]
U_ONES = spec["u_ones"]
TARGET_IDX = LABELS.index(TARGET_COL)

banner("CONFIG")
print(f"Disease:        {DISEASE} (column: {TARGET_COL})")
print(f"Target logit:   index {TARGET_IDX} of {len(LABELS)} (from labels.json)")
print(f"Uncertainty:    {TARGET_COL} -> "
      f"{'U-Ones (-1 -> 1)' if TARGET_COL in U_ONES else 'U-Zeros (-1 -> 0)'}, blanks -> 0")
print(f"Split dir:      {SPLIT_DIR}")
print(f"Output dir:     {OUTPUT_DIR}")
print(f"Model path:     {MODEL_PATH}")
print(f"Grad-CAM:       {'DISABLED' if args.no_cam else 'enabled'}")
print(f"Threshold mode: {args.threshold_mode}")
print(f"Batch size:     {BATCH_SIZE}")
print(f"Device:         {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class CheXpertDataset(Dataset):
    """Yields (image, target, raw_label, path) for the ONE target observation.

    target applies the same uncertainty policy the backbone was trained under, so
    `true` in the output CSV means what the loss meant. raw_label is passed through
    untouched (NaN encoded as 2.0, since a DataLoader cannot batch None).
    """

    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

        raw = self.df[TARGET_COL].to_numpy(dtype=np.float32)
        self.raw = np.where(np.isnan(raw), 2.0, raw)          # 2.0 == blank
        target = np.where(np.isnan(raw), 0.0, raw)            # blank -> negative
        self.target = np.where(raw == -1.0,
                               1.0 if TARGET_COL in U_ONES else 0.0,
                               target).astype(np.float32)
        self.paths = self.df["Path"].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_dir, self.paths[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, float(self.target[idx]), float(self.raw[idx]), self.paths[idx]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(path):
    print(f"Building DenseNet121 with {len(LABELS)}-logit head...")
    model = models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))

    print(f"Loading trained C0 weights from: {path}")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"C0 loaded: {n_params:,} parameters | eval mode | "
          f"backbone will NOT be trained again")

    return model


# ── Grad-CAM cache / hook ─────────────────────────────────────────────────────

cache = {}


def hook_fn(module, inp, out):
    # IMPORTANT: do NOT detach, do NOT clone -- the raw tensor has to stay in the
    # graph so register_hook below actually catches the gradient.
    acts = out

    cache["acts"] = torch.relu(acts).detach()
    cache["emb"] = cache["acts"].mean(dim=(2, 3)).detach()

    def save_grad(grad):
        cache["grads"] = grad.detach()

    acts.register_hook(save_grad)


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, loader, cam_dir, split=""):

    banner(f"INFERENCE ({split})")
    print(f"Images:        {len(loader.dataset):,} in {len(loader)} batches")
    print(f"Grad-CAMs to:  {'(skipped)' if args.no_cam else cam_dir}")
    print(f"Per image: prob for {TARGET_COL} + 1024-d embedding "
          f"+ all {len(LABELS)} label probs (separate CSV)")

    if not args.no_cam:
        os.makedirs(cam_dir, exist_ok=True)

    rows, all_prob_rows = [], []

    for imgs, targets, raws, paths in tqdm(loader, desc=f"inference {split}"):

        imgs = imgs.to(DEVICE)

        cache.clear()
        model.zero_grad(set_to_none=True)

        if args.no_cam:
            with torch.no_grad():
                logits = model(imgs)                   # (B, 14)
            probs_all = torch.sigmoid(logits)
            emb = cache["emb"]
            cams = None
        else:
            logits = model(imgs)
            probs_all = torch.sigmoid(logits)
            # Backprop the target logit only: each sample gets d(its own target
            # logit)/d(features), never a gradient mixed across the 14 heads.
            target_logits = logits[:, TARGET_IDX]      # (B,) -- the ONE we explain
            target_logits.backward(torch.ones_like(target_logits))

            acts = cache["acts"]
            grads = cache["grads"]
            emb = cache["emb"]

            weights = grads.mean(dim=(2, 3), keepdim=True)
            cams = torch.relu((weights * acts).sum(dim=1)).cpu().numpy()

        probs = probs_all[:, TARGET_IDX]

        emb = emb.cpu().numpy()
        probs = probs.detach().cpu().numpy()
        probs_all = probs_all.detach().cpu().numpy()
        targets = targets.numpy()
        raws = raws.numpy()

        for i, path in enumerate(paths):

            cam_path = ""
            if not args.no_cam:
                cam = cams[i]
                cam -= cam.min()
                cam /= (cam.max() + 1e-8)
                # Explicit .npz: np.savez_compressed appends it anyway, which is why
                # the single-label CSVs carry a cam_path that does not exist on disk.
                # This name also matches _load_cam() in the C2 CNN baselines.
                cam_path = os.path.join(cam_dir, path.replace("/", "_") + ".npz")
                np.savez_compressed(cam_path, cam=cam)

            row = {
                "path": path,
                "prob": float(probs[i]),
                "true": float(targets[i]),
                "cam_path": cam_path,
            }

            for j, v in enumerate(emb[i]):
                row[f"emb_{j}"] = float(v)

            raw = float(raws[i])
            row["label_raw"] = "" if raw == 2.0 else raw    # 2.0 was the blank sentinel
            row["certain"] = int(raw in (0.0, 1.0))

            rows.append(row)

            all_prob_rows.append(
                {"path": path,
                 **{f"p_{lbl}": float(probs_all[i, j]) for j, lbl in enumerate(LABELS)}}
            )

    return pd.DataFrame(rows), pd.DataFrame(all_prob_rows)


# ── Metrics ───────────────────────────────────────────────────────────────────

def find_threshold(df):
    fpr, tpr, thr = roc_curve(df["true"], df["prob"])
    auc = roc_auc_score(df["true"], df["prob"])
    best = thr[np.argmax(tpr - fpr)]
    return float(best), auc


def apply_threshold(df, t):
    df = df.copy()
    df["pred"] = (df["prob"] > t).astype(int)
    df["correct"] = (df["pred"] == df["true"]).astype(int)
    df["margin"] = abs(df["prob"] - t)
    # keep the legacy column order: ... emb_1023, pred, correct, margin, then the new two
    tail = ["label_raw", "certain"]
    cols = [c for c in df.columns if c not in tail] + tail
    return df[cols]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    banner("LOADING DATA SPLITS")
    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "c0_train_split.csv"))
    val_df = pd.read_csv(os.path.join(SPLIT_DIR, "c0_val_split.csv"))
    if args.limit:
        train_df, val_df = train_df.head(args.limit), val_df.head(args.limit)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    for name, df in [("Train", train_df), ("Val", val_df)]:
        col = df[TARGET_COL]
        print(f"{name}: explicit 1 = {(col == 1).sum():,} | explicit 0 = {(col == 0).sum():,} "
              f"| uncertain = {(col == -1).sum():,} | blank = {col.isna().sum():,}")

    loaders = {
        "train": DataLoader(CheXpertDataset(train_df, DATA_DIR, transform),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=args.num_workers),
        "val": DataLoader(CheXpertDataset(val_df, DATA_DIR, transform),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=args.num_workers),
    }

    banner("LOADING TRAINED C0 MODEL")
    model = load_model(MODEL_PATH)

    # Hook the last conv output (DenseNet121 feature extractor) for CAM + embeddings.
    model.features.register_forward_hook(hook_fn)

    out = {}
    for split, loader in loaders.items():
        cam_dir = os.path.join(OUTPUT_DIR, "gradcam", split)
        out[split] = run_inference(model, loader, cam_dir, split=split)

    banner("METRICS")
    train_out = out["train"][0]
    if args.threshold_mode == "fixed":
        thresh, auc = 0.5, roc_auc_score(train_out["true"], train_out["prob"])
    else:
        thresh, auc = find_threshold(train_out)

    with open(os.path.join(OUTPUT_DIR, "threshold.txt"), "w") as f:
        f.write(str(thresh))
    with open(os.path.join(OUTPUT_DIR, "threshold_fixed.txt"), "w") as f:
        f.write("0.5")

    print(f"Train AUC: {auc:.4f}")
    print(f"Operating threshold ({args.threshold_mode}): {thresh:.4f}")

    banner("SAVING OUTPUTS")
    for split in ("train", "val"):
        df, all_probs = out[split]
        df = apply_threshold(df, thresh)

        print(f"\n{split} accuracy: {df['correct'].mean():.3f} "
              f"({df['correct'].sum():,}/{len(df):,})")
        print(f"{split} AUC: {roc_auc_score(df['true'], df['prob']):.4f}")

        # The old per-disease splits kept only explicitly-mentioned studies, which is
        # a much easier (and ~75% positive) cohort. Report it so the two are comparable.
        cert = df[df["certain"] == 1]
        print(f"{split} AUC (certain only): {roc_auc_score(cert['true'], cert['prob']):.4f} "
              f"on {len(cert):,} rows, {cert['true'].mean():.1%} positive")

        csv_path = os.path.join(OUTPUT_DIR, f"{split}_c0_{DISEASE}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path} ({len(df):,} rows)")

        ap_path = os.path.join(OUTPUT_DIR, f"all_probs_{split}.csv")
        all_probs.to_csv(ap_path, index=False)
        print(f"Wrote {ap_path} ({len(all_probs):,} rows x {len(LABELS)} labels)")

    print("\nSaved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
