"""
c0_final_predictions.py
=======================
Inference + Grad-CAM + embeddings for the final_experiment C0.

What it outputs, per split, under <model dir>/<disease>/:
    <split>_c0_<disease>.csv   path, prob, true, cam_path, emb_0..emb_1023,
                               pred, correct, margin, label_raw, certain
    all_probs_<split>.csv      path + p_<label> for all 14
    gradcam/<split>/*.npz      cam_pos, cam_neg  two (7,7) float32 maps in [0,1]
                               scale_pos, scale_neg  their pre-normalisation maxima
                               cam, orient_t, sign   net map, for old readers

1. U-Ignore. The backbone was trained with a per-label mask, so it never saw a
   target for an uncertain (-1) row. `true` is therefore NaN for those rows, and
   so are `pred`/`correct`/`margin`. They are kept in the CSV (with certain == 0)
   so the cohort stays traceable, but C2 must train on certain == 1 only -- a NaN
   correctness label is not a class.

2. The operating threshold is Youden's J on --threshold-split
   (default: C0_checkpoint_selection).

3. Grad-CAM is stored as TWO maps, not one: `cam_pos` = relu(signed) is the
   evidence arguing for the disease, `cam_neg` = relu(-signed) the evidence
   arguing for health. Backprop is on the target logit either way; only the
   handling of its sign changed.

   The original code kept relu(signed) alone, so every image C0 called negative
   -- where the signed map is negative nearly everywhere -- came out exactly
   all-zero: 77.5% of TN and 50.0% of FN. Blank-vs-not then encoded the
   predicted class, which a saliency encoder can learn as a shortcut.

   Orienting the single map toward the predicted class fixed those and broke the
   mirror case. `prob > threshold` is the reported DECISION, while the gradient
   follows the LOGIT, and the two disagree for every row between the operating
   threshold and 0.5 -- 99.9% of consolidation's positive predictions, 91.9% of
   atelectasis's. On those the flip never fired and the ReLU clipped the map
   away exactly as before.

   Two channels remove the choice: no threshold enters the map, nothing is
   annihilated in either direction, and a region that argues both ways appears
   in both channels instead of cancelling to a net near zero. `cam` is kept as
   the net map (oriented by the decision) so older readers still work.
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

# Chooses the prefix for each csv and whether grad-cam is needed.
# In this case, we don't save saliency maps for checkpoint selection.
SPLIT_SPEC = {
    "C2_dataset":              ("C2_dataset.csv", True),
    "Original_Test":           ("Original_Test.csv", True),
    "C0_checkpoint_selection": ("C0_checkpoint_selection.csv", False),
}

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--disease", type=str, required=True, choices=list(disease_col.keys()))
parser.add_argument("--splits", nargs="+", default=list(SPLIT_SPEC.keys()),
                    choices=list(SPLIT_SPEC.keys()))
parser.add_argument("--threshold-split", default="C0_checkpoint_selection",
                    choices=list(SPLIT_SPEC.keys()),
                    help="split the operating threshold is tuned on (Youden's J)")
parser.add_argument("--policy", default="ignore",
                    help="C0 uncertainty policy; picks the multilabel_<policy> model "
                         "dir. Each policy trains its own backbone into its own "
                         "results dir, so runs never overwrite each other.")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--no-cam", action="store_true", help="skip Grad-CAM entirely (much faster)")
parser.add_argument("--cam-threshold", type=float, default=None,
                    help="threshold used to ORIENT the Grad-CAM toward the predicted "
                         "class. Default: read threshold.txt from a prior run of this "
                         "disease; failing that, the threshold split is scored first "
                         "and its Youden J is used.")
parser.add_argument("--cam-only", action="store_true",
                    help="recompute and overwrite the Grad-CAM npz files only. Skips "
                         "the CSVs, embeddings, threshold.txt and the symlink -- none "
                         "of which the CAM orientation affects. Requires an existing "
                         "threshold.txt or --cam-threshold.")
parser.add_argument("--limit", type=int, default=None, help="debug: first N rows of each split")
parser.add_argument("--link-into", default=os.path.join(REPO, "final_experiment", "results"),
                    help='directory to symlink the output dir into ("" to skip)')
args = parser.parse_args()

if args.cam_only and args.no_cam:
    parser.error("--cam-only and --no-cam are mutually exclusive")

DISEASE = args.disease
TARGET_COL = disease_col[DISEASE]

DATA_ROOT = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
DATA_DIR = DATA_ROOT
SPLIT_DIR = os.path.join(REPO, "final_experiment")
MODEL_DIR = os.path.join(RESULTS_DIR, "C0_final", f"multilabel_{args.policy}")
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
TARGET_IDX = LABELS.index(TARGET_COL)

banner("CONFIG")
print(f"Disease:        {DISEASE} (column: {TARGET_COL})")
print(f"Target logit:   index {TARGET_IDX} of {len(LABELS)} (from labels.json)")
print(f"Uncertainty:    U-Ignore (-1 rows -> true = NaN; excluded from AUC, threshold, `correct`)")
print(f"Split dir:      {SPLIT_DIR}")
print(f"Splits:         {', '.join(args.splits)}")
print(f"Threshold from: {args.threshold_split} (Youden's J)")
print(f"Model path:     {MODEL_PATH}")
print(f"Output dir:     {OUTPUT_DIR}")
print(f"Grad-CAM:       {'DISABLED' if args.no_cam else 'enabled'}")
print(f"Device:         {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class CheXpertDataset(Dataset):
    """Yields (image, target, raw_label, path) for the ONE target observation.

    raw_label passes the original CheXpert cell through untouched (NaN -> 2.0,
    since a DataLoader cannot batch None). The final_experiment CSVs have blanks
    already filled with 0.0 upstream, so in practice raw is only 1/0/-1.

    Uncertain (-1) rows get true = NaN (U-Ignore, matching training).
    """

    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        
        
        raw = self.df[TARGET_COL].to_numpy(dtype=np.float32)
        self.raw = np.where(np.isnan(raw), 2.0, raw)          # 2.0 == blank sentinel
        # uncertain (-1) -> NaN; blank -> negative (already 0.0 upstream)
        self.target = np.where(raw == -1.0, np.nan,
                               np.where(np.isnan(raw), 0.0, raw)).astype(np.float32)
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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(path):
    print(f"Building DenseNet121 with {len(LABELS)}-logit head...")
    
    # Loads the pretrained DenseNet121 and replaces the classifier with a new linear layer
    model = models.densenet121()
    
    # This linear layer will output the number of logits equal to the number of labels
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))

    print(f"Loading trained C0 weights from: {path}")
    
    # Loads the state dictionary from the specified path and maps it to the appropriate device
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print(f"C0 loaded: {sum(p.numel() for p in model.parameters()):,} parameters | eval mode")
    return model


# ── Grad-CAM cache / hook ─────────────────────────────────────────────────────

cache = {}

# Hook is a forward hook on model.features (the densenet feature extractor, before the classifier head)
# For each batch that passes, the hook saves:
# + the activations (ReLU's feature maps (B,21024,7,7) 
# + the 1024--dim embedding, which is the spatial average of the feature maps (B,1024)) 
# + the gradients of the target logit w.r.t. the activations (B,21024,7,7) into the cache.

def hook_fn(module, inp, out):
    # IMPORTANT: do NOT detach/clone `out` -- the raw tensor has to stay in the
    # graph so register_hook below actually catches the gradient.
    acts = out
    cache["acts"] = torch.relu(acts).detach()
    cache["emb"] = cache["acts"].mean(dim=(2, 3)).detach()

    def save_grad(grad):
        cache["grads"] = grad.detach()

    # Under torch.no_grad() (the CAM-free splits) there is no graph to hook, and
    # register_hook would raise. The old script hits exactly this with --no-cam.
    if acts.requires_grad:
        acts.register_hook(save_grad)


# ── Inference ─────────────────────────────────────────────────────────────────

# Running inference depends on whether we want to write_cam or not.
# For checkpoint selection, we don't need CAMs, so we can skip the backward pass and save time.
def run_inference(model, loader, cam_dir, split, write_cam, cam_threshold=None):
    banner(f"INFERENCE ({split})")
    print(f"Images:        {len(loader.dataset):,} in {len(loader)} batches")
    print(f"Grad-CAMs to:  {cam_dir if write_cam else '(skipped for this split)'}")

    if write_cam:
        os.makedirs(cam_dir, exist_ok=True)

    rows, all_prob_rows = [], []

    for imgs, targets, raws, paths in tqdm(loader, desc=f"inference {split}"):
        imgs = imgs.to(DEVICE)
        cache.clear()
        model.zero_grad(set_to_none=True)

        if not write_cam:
            with torch.no_grad():
                logits = model(imgs)
            probs_all = torch.sigmoid(logits)
            emb = cache["emb"]
            cams_pos = cams_neg = None
        else:
            logits = model(imgs)
            probs_all = torch.sigmoid(logits)
            # Backprop the target logit only: each sample gets d(its own target logit)/d(features), never a gradient mixed across the 14 heads.
            # That's how we get the gradients specific to that disease
            target_logits = logits[:, TARGET_IDX]
            target_logits.backward(torch.ones_like(target_logits))

            acts, grads, emb = cache["acts"], cache["grads"], cache["emb"]
            weights = grads.mean(dim=(2, 3), keepdim=True)

            # Two channels from the SIGNED map, no orientation and no clipping:
            #   cam_pos = relu( signed)   regions arguing FOR the disease
            #   cam_neg = relu(-signed)   regions arguing FOR healthy
            #
            # The original code kept only relu(signed) of the positive logit, so
            # every image C0 called negative -- where the signed map is negative
            # almost everywhere -- came out exactly all-zero (77.5% of TN, 50.0%
            # of FN). Orienting by the predicted class fixed those but broke the
            # mirror case: `prob > threshold` is the reported DECISION while the
            # gradient follows the LOGIT, and between t and 0.5 they disagree --
            # 99.9% of consolidation's positive predictions, where the flip never
            # happened and the ReLU clipped the map away again.
            #
            # Splitting the two directions removes the choice entirely. Nothing
            # depends on a threshold, nothing is annihilated, and a region that
            # argues both ways (an opacity with an air bronchogram running
            # through it) shows up in both channels instead of silently
            # cancelling to a net near zero.
            cam_signed = (weights * acts).sum(dim=1)                    # (B,7,7)
            cams_pos = torch.relu(cam_signed).cpu().numpy()
            cams_neg = torch.relu(-cam_signed).cpu().numpy()



        probs = probs_all[:, TARGET_IDX]
        emb = emb.cpu().numpy()
        probs = probs.detach().cpu().numpy()
        probs_all = probs_all.detach().cpu().numpy()
        targets = targets.numpy()
        raws = raws.numpy()

        for i, path in enumerate(paths):
            cam_path = ""
            if write_cam:
            
                # Two (7,7) maps, each scaled to [0,1] by ITS OWN max. Scaling
                # them jointly would let the dominant direction flatten the other
                # into invisibility, and on most images one direction does
                # dominate. `scale_pos`/`scale_neg` keep the discarded magnitudes
                # so the raw balance between the two is recoverable.
                cam_pos, cam_neg = cams_pos[i], cams_neg[i]
                s_pos, s_neg = float(cam_pos.max()), float(cam_neg.max())
                cam_pos = cam_pos / (s_pos + 1e-8)
                cam_neg = cam_neg / (s_neg + 1e-8)

                # `cam` stays as the net map oriented toward the reported
                # decision, so readers written against the old single-key format
                # keep working. New consumers should take cam_pos/cam_neg.
                sign = 1.0 if float(probs_all[i, TARGET_IDX]) > cam_threshold else -1.0
                net = cam_pos - cam_neg if sign > 0 else cam_neg - cam_pos
                net = (net - net.min()) / (net.max() - net.min() + 1e-8)

                # Explicit .npz: np.savez_compressed appends it anyway, which is why
                # the single-label CSVs carry a cam_path that does not exist on disk.
                cam_path = os.path.join(cam_dir, path.replace("/", "_") + ".npz")
                np.savez_compressed(cam_path,
                                    cam_pos=cam_pos.astype(np.float32),
                                    cam_neg=cam_neg.astype(np.float32),
                                    scale_pos=np.float32(s_pos),
                                    scale_neg=np.float32(s_neg),
                                    cam=net.astype(np.float32),
                                    orient_t=np.float32(cam_threshold),
                                    sign=np.float32(sign))

            row = {
                "path": path,
                "prob": float(probs[i]),
                "true": float(targets[i]),
                "cam_path": cam_path,
            }
            for j, v in enumerate(emb[i]):
                row[f"emb_{j}"] = float(v)

            raw = float(raws[i])
            row["label_raw"] = "" if raw == 2.0 else raw     # 2.0 was the blank sentinel
            row["certain"] = int(raw in (0.0, 1.0))
            rows.append(row)

            all_prob_rows.append(
                {"path": path,
                 **{f"p_{lbl}": float(probs_all[i, j]) for j, lbl in enumerate(LABELS)}}
            )

    return pd.DataFrame(rows), pd.DataFrame(all_prob_rows)


# ── Metrics ───────────────────────────────────────────────────────────────────

def certain_rows(df):
    """Rows with an explicit 0/1 label -- the only ones with usable ground truth."""
    return df[(df["certain"] == 1) & df["true"].notna()]


def safe_auc(y, p):
    """AUC, or NaN when a split has only one class (e.g. a rare label on the
    202-row official test). Never let that kill a run that has already spent
    hours writing CAMs."""
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def find_threshold(df):
    # Optimizes for Youden's J statistic (maximizes TPR - FPR) on the certain rows of the given dataframe.
    c = certain_rows(df)
    fpr, tpr, thr = roc_curve(c["true"], c["prob"])
    return float(thr[np.argmax(tpr - fpr)]), safe_auc(c["true"], c["prob"])


def apply_threshold(df, t):
    df = df.copy()
    pred = (df["prob"] > t).astype(float)
    # An uncertain row has no ground truth, so it gets no pred/correct/margin
    # either. Leaving them NaN makes a downstream dropna() do the right thing;
    # filling them with 0 would silently create a class C2 could learn.
    usable = (df["certain"] == 1) & df["true"].notna()
    df["pred"] = pred.where(usable)
    df["correct"] = (df["pred"] == df["true"]).astype(float).where(usable)
    df["margin"] = (df["prob"] - t).abs().where(usable)
    # legacy column order: ... emb_1023, pred, correct, margin, then the new two
    tail = ["label_raw", "certain"]
    return df[[c for c in df.columns if c not in tail] + tail]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    banner("LOADING DATA SPLITS")
    dfs = {}
    for name in args.splits:
        fname, _ = SPLIT_SPEC[name]
        
        # Read the CSV for the split, optionally limiting the number of rows for debugging
        df = pd.read_csv(os.path.join(SPLIT_DIR, fname))
        if args.limit:
            df = df.head(args.limit)
        dfs[name] = df
        col = df[TARGET_COL]
        print(f"{name:26s} {len(df):>7,} rows | "
              f"pos {int((col == 1).sum()):>6,} | neg {int((col == 0).sum()):>6,} | "
              f"uncertain {int((col == -1).sum()):>5,} | blank {int(col.isna().sum()):>5,}")

    if args.threshold_split not in dfs and not args.cam_only:
        raise SystemExit(f"--threshold-split {args.threshold_split} is not in --splits")

    banner("LOADING DATA")
    banner("LOADING TRAINED C0 MODEL")
    model = load_model(MODEL_PATH)
    model.features.register_forward_hook(hook_fn)

    # ── CAM orientation threshold ─────────────────────────────────────────────
    # The map is oriented toward the PREDICTED class, so the threshold has to be
    # known before any CAM split runs -- but find_threshold() below needs that
    # split already scored. Resolve it up front, in order of preference:
    #   1. --cam-threshold
    #   2. threshold.txt from a prior run. This is the usual path, and it keeps
    #      the operating point bit-identical to the run the C2 folds and the CF
    #      levels were built against.
    #   3. score the threshold split first. It writes no CAMs (SPLIT_SPEC), so
    #      reordering costs nothing.
    cam_thresh = args.cam_threshold
    thresh_file = os.path.join(OUTPUT_DIR, "threshold.txt")
    if cam_thresh is None and os.path.exists(thresh_file):
        with open(thresh_file) as f:
            cam_thresh = float(f.read().split()[0])
        print(f"CAM orientation threshold: {cam_thresh:.6f}  (from {thresh_file})")
    elif cam_thresh is not None:
        print(f"CAM orientation threshold: {cam_thresh:.6f}  (--cam-threshold)")

    order = list(dfs.keys())
    want_cam = any(SPLIT_SPEC[n][1] for n in order) and not args.no_cam
    if args.cam_only:
        if cam_thresh is None:
            raise SystemExit(
                "--cam-only needs an orientation threshold, but no threshold.txt "
                f"exists at {thresh_file} and --cam-threshold was not given. Run a "
                "full pass first, or pass the threshold explicitly.")
        order = [n for n in order if SPLIT_SPEC[n][1]]
        print(f"--cam-only: writing CAMs for {', '.join(order)}; "
              f"skipping CSVs, embeddings, threshold and symlink.")
    elif cam_thresh is None and want_cam:
        order = ([args.threshold_split]
                 + [n for n in order if n != args.threshold_split])
        print(f"No threshold on disk -- scoring {args.threshold_split} first to "
              f"derive the CAM orientation threshold.")

    out = {}
    for name in order:
        df = dfs[name]
        _, cam_default = SPLIT_SPEC[name]
        write_cam = cam_default and not args.no_cam

        # Fallback (3): the threshold split has now been scored, so derive it.
        if write_cam and cam_thresh is None:
            cam_thresh, _ = find_threshold(out[args.threshold_split][0])
            print(f"CAM orientation threshold: {cam_thresh:.6f}  "
                  f"(Youden's J on {args.threshold_split}, this run)")

        # Load the dataset and create a DataLoader for the current split
        loader = DataLoader(CheXpertDataset(df, DATA_DIR, transform),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=args.num_workers)

        # Run inference on the current split and store the results in the output dictionary
        out[name] = run_inference(model, loader,
                                  os.path.join(OUTPUT_DIR, "gradcam", name),
                                  split=name, write_cam=write_cam,
                                  cam_threshold=cam_thresh)

    if args.cam_only:
        banner("DONE (--cam-only)")
        for name in order:
            n = len(dfs[name])
            print(f"{name:26s} {n:>7,} CAMs -> "
                  f"{os.path.join(OUTPUT_DIR, 'gradcam', name)}")
        print(f"\nOriented at threshold {cam_thresh:.6f}. "
              f"CSVs, embeddings and threshold.txt left untouched.")
        return

    banner("THRESHOLD")
    # Determine the operating threshold using the specified split for tuning
    src = out[args.threshold_split][0]
    thresh, auc = find_threshold(src)
    print(f"Tuned on {args.threshold_split}: {len(certain_rows(src)):,} certain rows")
    print(f"AUC: {auc:.4f} | operating threshold (Youden's J): {thresh:.4f}")

    # The CAMs on disk were oriented at cam_thresh. If this run's Youden J has
    # moved, every map for a row between the two thresholds now points the wrong
    # way relative to the `pred` column written below.
    if cam_thresh is not None and abs(cam_thresh - thresh) > 1e-6:
        n_between = int(((src["prob"] > min(cam_thresh, thresh)) &
                         (src["prob"] <= max(cam_thresh, thresh))).sum())
        print(f"  !! CAMs were oriented at t={cam_thresh:.6f} but this run's "
              f"Youden J is t={thresh:.6f}.\n"
              f"     {n_between:,} rows of {args.threshold_split} fall between the "
              f"two; their maps and `pred` disagree.\n"
              f"     Rerun with --cam-only --cam-threshold {thresh:.6f} to realign.")

    with open(os.path.join(OUTPUT_DIR, "threshold.txt"), "w") as f:
        f.write(str(thresh))
    with open(os.path.join(OUTPUT_DIR, "threshold_fixed.txt"), "w") as f:
        f.write("0.5")

    banner("SAVING OUTPUTS")
    for name in args.splits:
        df, all_probs = out[name]
        df = apply_threshold(df, thresh)
        c = certain_rows(df)

        print(f"\n{name}: {len(df):,} rows, {len(c):,} certain "
              f"({len(df) - len(c):,} uncertain -> NaN correctness)")
        print(f"  AUC (certain):      {safe_auc(c['true'], c['prob']):.4f} "
              f"on {c['true'].mean():.1%} positive")
        print(f"  accuracy (certain): {c['correct'].mean():.3f} "
              f"({int(c['correct'].sum()):,}/{len(c):,})")

        p = os.path.join(OUTPUT_DIR, f"{name}_c0_{DISEASE}.csv")
        df.to_csv(p, index=False)
        print(f"  wrote {p}")
        p = os.path.join(OUTPUT_DIR, f"all_probs_{name}.csv")
        all_probs.to_csv(p, index=False)
        print(f"  wrote {p} ({len(LABELS)} label probs)")

    if args.link_into:
        link = os.path.join(args.link_into, f"C0_ignore_{DISEASE}")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(OUTPUT_DIR, link)
        print(f"\nLinked -> {link}")

    print("\nSaved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()