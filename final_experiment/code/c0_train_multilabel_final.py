# c0_train_multilabel_final.py
"""
Final multi-label C0: DenseNet121 over all 14 CheXpert observations.

Standard CheXpert setup (Irvin et al., AAAI 2019): ImageNet-pretrained
DenseNet121, a 14-logit head, per-label BCE, Adam @ 1e-4, model selection on the
mean AUC of the 5 competition pathologies.

Data
----
Reads the frontal-only, patient-disjoint split built by
final_experiment/C0_custom_split.ipynb:

    C0_train.csv                85,664 imgs / 29,040 patients   <- fit
    C0_checkpoint_selection.csv  9,528 imgs /  3,227 patients   <- early stop + best ckpt
    C2_dataset.csv              95,835 imgs / 32,267 patients   <- NEVER seen here
    Original_Test.csv              202 imgs /    200 patients   <- official valid, scored once

All four are disjoint at patient level (verified: 0 patient and 0 path overlap
across every pair). C2_dataset is the downstream meta-classifier cohort and is
deliberately untouched by this script -- that disjointness is the whole point of
the rebuild, so do not add it to any loader here.

Labels
------
Blanks were already filled with 0.0 upstream (blank = observation not mentioned
= negative, the standard CheXpert reading), so the CSVs contain only 1 / 0 / -1.

Uncertainty policy: U-Ignore (fixed)
------------------------------------
Per-label mask: an uncertain (-1) entry contributes zero gradient for THAT label
while the row still trains the other 13.
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
REPO = "/zhome/d0/a/221493/thesis"
SPLIT_DIR = os.path.join(REPO, "final_experiment")
OUTPUT_ROOT = "/work3/s251710/thesis_results/C0_final"

# The 14 observations, in train.csv column order. This order is the contract
# between the checkpoint's 14 logits and everything that reads them -- it is
# written to labels.json next to the weights.
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]
# The 5 pathologies the CheXpert competition scores on.
COMPETITION = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
               "Pleural Effusion"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Dataset ───────────────────────────────────────────────────────────────────
class CheXpertMultiLabelDataset(Dataset):
    """Returns (image, target[14], certain[14]).
    target  raw 1/0 labels; value at uncertain entries is 0 (masked out anyway).
    certain 0 where the RAW label was -1. Masks both the loss and the AUC.
    """

    def __init__(self, df, data_dir, transform):
        self.paths = df["Path"].to_numpy()
        self.data_dir = data_dir
        self.transform = transform

        # The raw labels are used to determine which entries are uncertain,
        # and the target labels are used for training.
        raw = df[LABELS].to_numpy(dtype=np.float32)          # 1 / 0 / -1
        uncertain = raw == -1.0
        self.certain = (~uncertain).astype(np.float32)
        # Value at uncertain entries is arbitrary (masked out of the loss);
        # 0 keeps it well-defined.
        self.targets = np.where(uncertain, 0.0, raw).astype(np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_dir, self.paths[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (img,
                torch.from_numpy(self.targets[idx].copy()),
                torch.from_numpy(self.certain[idx]))


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(n_labels=len(LABELS)):
    model = models.densenet121(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(model.classifier.in_features, n_labels)
    return model


# ── Loss ──────────────────────────────────────────────────────────────────────
def masked_bce(logits, targets, certain):
    """Mean BCE over certain entries only; uncertain entries contribute no gradient."""
    per_elem = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none")
    return (per_elem * certain).sum() / certain.sum().clamp(min=1.0)


# ── One epoch ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss, n_batches = 0.0, 0
    for imgs, targets, certain in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        certain = certain.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=scaler is not None):
            loss = masked_bce(model(imgs), targets, certain)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """Masked loss; AUC on confidently-labelled rows only."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    probs, targets, certains = [], [], []
    for imgs, tgt, cer in loader:
        imgs = imgs.to(device, non_blocking=True)
        tgt_d, cer_d = tgt.to(device, non_blocking=True), cer.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            total_loss += masked_bce(logits, tgt_d, cer_d).item()
        n_batches += 1
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(tgt.numpy())
        certains.append(cer.numpy())

    probs = np.concatenate(probs)
    targets = np.concatenate(targets)
    certains = np.concatenate(certains).astype(bool)

    aucs = {}
    for j, lbl in enumerate(LABELS):
        keep = certains[:, j]
        y, p = targets[keep, j], probs[keep, j]
        aucs[lbl] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")

    comp = [aucs[c] for c in COMPETITION]
    mean_comp_auc = float(np.nanmean(comp))
    return total_loss / max(n_batches, 1), aucs, mean_comp_auc, probs


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train the final 14-label DenseNet121 C0 (U-Ignore) on the "
                    "frontal-only patient-disjoint CheXpert split.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3,
                   help="Stop after this many epochs with no selection-AUC gain.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--split-dir", default=SPLIT_DIR)
    p.add_argument("--out-dir", default=None,
                   help=f"Default: {OUTPUT_ROOT}/multilabel_ignore")
    p.add_argument("--limit", type=int, default=None, help="Debug: first N rows per split.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_amp = (not args.no_amp) and DEVICE.type == "cuda"
    out_dir = args.out_dir or os.path.join(OUTPUT_ROOT, "multilabel_ignore")
    os.makedirs(out_dir, exist_ok=True)

    print("Final multi-label C0 — 14 CheXpert observations, frontal only")
    print(f"  splits:      {args.split_dir}")
    print(f"  output:      {out_dir}")
    print(f"  device:      {DEVICE} | amp: {use_amp}")
    print("  uncertainty: ignore  (per-label mask; -1 contributes no gradient)")
    print("  augment:     off")

    # Read the CSVs
    train_df = pd.read_csv(os.path.join(args.split_dir, "C0_train.csv"))
    sel_df = pd.read_csv(os.path.join(args.split_dir, "C0_checkpoint_selection.csv"))
    test_df = pd.read_csv(os.path.join(args.split_dir, "Original_Test.csv"))
    if args.limit:
        train_df, sel_df = train_df.head(args.limit), sel_df.head(args.limit)

    # The rebuild exists to guarantee this. Fail loudly rather than silently leak.
    leak = set(train_df["patient_id"]) & set(sel_df["patient_id"])
    assert not leak, f"{len(leak)} patients leak between train and checkpoint selection"

    print(f"  train:       {len(train_df):,} imgs / {train_df['patient_id'].nunique():,} patients")
    print(f"  selection:   {len(sel_df):,} imgs / {sel_df['patient_id'].nunique():,} patients")
    print(f"  test:        {len(test_df):,} imgs / {test_df['patient_id'].nunique():,} patients "
          "(official valid; scored once, at the end)")

    tf = build_transform()
    train_ds = CheXpertMultiLabelDataset(train_df, DATA_ROOT, tf)
    sel_ds = CheXpertMultiLabelDataset(sel_df, DATA_ROOT, tf)
    test_ds = CheXpertMultiLabelDataset(test_df, DATA_ROOT, tf)

    print("\n  per-label uncertain rate in train (share dropped from the loss):")
    for j, lbl in enumerate(LABELS):
        print(f"    {lbl:<28} {1 - train_ds.certain[:, j].mean():6.1%}")

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=args.num_workers > 0)

    # Loaders
    train_loader = loader(train_ds, True)
    sel_loader = loader(sel_ds, False)
    test_loader = loader(test_ds, False)

    # Start model, optimizer, scaler
    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Label order is the checkpoint's contract -- write it next to the weights.
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump({"labels": LABELS, "competition": COMPETITION,
                   "uncertainty": "ignore",
                   "input_size": 224, "views": "frontal",
                   "split_dir": args.split_dir, "seed": args.seed}, f, indent=2)

    ckpt_path = os.path.join(out_dir, "c0_best.pt")
    history, best_auc, best_epoch = [], -1.0, -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train one epoch, evaluate on the checkpoint selection set, and record metrics
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, scaler)
        sel_loss, aucs, mean_comp_auc, _ = evaluate(model, sel_loader, DEVICE, use_amp)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "sel_loss": sel_loss, "mean_comp_auc": mean_comp_auc, **aucs})
        print(f"\nEpoch {epoch:02d} ({time.time()-t0:.0f}s) | train {train_loss:.4f} | "
              f"selection {sel_loss:.4f} | mean AUC (5 competition) {mean_comp_auc:.4f}")
        for lbl in LABELS:
            print(f"    {lbl:<28} {aucs[lbl]:.4f}{' *' if lbl in COMPETITION else ''}")

        # Check if the current epoch is the best so far in the competition tasks
        if mean_comp_auc > best_auc:
            best_auc, best_epoch = mean_comp_auc, epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> new best (mean competition AUC={best_auc:.4f}), saved")

        # If improvement doesn't happen in `args.patience` epochs, stop early (3 by default)
        elif epoch - best_epoch >= args.patience:
            print(f"  -> no gain for {args.patience} epochs (best was epoch "
                  f"{best_epoch}); stopping early")
            break

        pd.DataFrame(history).to_csv(os.path.join(out_dir, "train_history.csv"), index=False)

    # Save the training history to a CSV file for later analysis
    hist = pd.DataFrame(history)
    hist.to_csv(os.path.join(out_dir, "train_history.csv"), index=False)

    # ── Final, single evaluation on the held-out official valid set ───────────
    # This is the 200 sample official validation set, which is scored once at the
    # end. It is not used for model selection or training.
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    test_loss, test_aucs, test_mean, test_probs = evaluate(
        model, test_loader, DEVICE, use_amp)

    print(f"\nBest checkpoint: epoch {best_epoch}, selection mean AUC {best_auc:.4f}")
    print(f"Original_Test.csv ({len(test_df)} imgs) mean competition AUC: {test_mean:.4f}")
    for lbl in LABELS:
        print(f"    {lbl:<28} {test_aucs[lbl]:.4f}{' *' if lbl in COMPETITION else ''}")

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "selection_mean_comp_auc": best_auc,
                   "test_mean_comp_auc": test_mean, "test_loss": test_loss,
                   "test_auc_per_label": test_aucs, "n_test": len(test_df)}, f, indent=2)
    pd.DataFrame(test_probs, columns=LABELS).assign(Path=test_df["Path"].values).to_csv(
        os.path.join(out_dir, "test_probs.csv"), index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    # Training and selection losses over epochs, marking the best epoch.
    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="train")
    plt.plot(hist["epoch"], hist["sel_loss"], label="checkpoint selection")
    plt.axvline(best_epoch, color="k", ls=":", lw=1, label=f"best (ep {best_epoch})")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Masked BCE")
    plt.title("Loss — multi-label C0 (U-Ignore)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss.png"), dpi=150); plt.close()

    # AUC for each competition label over epochs, with mean and best-epoch marker.
    plt.figure()
    for lbl in COMPETITION:
        plt.plot(hist["epoch"], hist[lbl], alpha=0.6, label=lbl)
    plt.plot(hist["epoch"], hist["mean_comp_auc"], color="k", lw=2, label="mean (5)")
    plt.axvline(best_epoch, color="k", ls=":", lw=1)
    plt.legend(fontsize=8); plt.xlabel("Epoch"); plt.ylabel("Selection AUC")
    plt.title("Selection AUC — multi-label C0 (U-Ignore)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "auc.png"), dpi=150); plt.close()

    print(f"\nWrote {out_dir}")