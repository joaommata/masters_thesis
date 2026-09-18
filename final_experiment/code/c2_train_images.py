# c2_train_images.py
"""
Stage 5c: image-based (CNN) C2 models on the cached folds.

Same folds, same counterfactual pairing and same target as c2_train_attributes.py,
so the two ladders are directly comparable. The ladder:

    xi                query image only
    xi_sal            query image + its C0 Grad-CAM
    cf                counterfactual image only
    cf_sal            CF image + its Grad-CAM
    dual              query + CF, one encoder each
    dual_sal          query + CF + both Grad-CAMs, four encoders
    dual_sal_scalars  dual_sal plus the C0 scalars [p_xi, H(p_xi), p_cf, H(p_cf),
                      p_xi - p_cf] embedded at the classifier head

    python c2_train_images.py --disease effusion --cf-source knn --cf-count 1 \
        --models dual_sal,dual_sal_scalars

Folds come from c2_folds.py. This script does NOT recompute the CF pairing and
does not depend on the attribute trainer having run -- either can go first.

More than one counterfactual per query
--------------------------------------
K is combined by ENSEMBLING SCORES -- the same protocol c2_train_attributes.py
uses, which is what makes the two ladders comparable at every K. The models have
no notion of K: each is called once per (query, CF) pair, and _ensemble_probs
averages the K sigmoids into one probability per query.

    train   --train-k nearest CFs, 1 by default. An epoch costs the same at
            K=10 as at K=1.
    val     all K, so checkpoint selection is scored under the protocol the
            test set will be scored under.
    test    all K, averaged.

Two consequences worth knowing:

  * --eval-batch-size defaults to --batch-size // K, holding the number of images
    in flight constant. Peak GPU memory is (batch x CFs x encoders) images, so a
    fixed batch at K=10 OOMs the quad-encoder configs. Inference batches are
    independent, so this is throughput and not a result.
  * xi and xi_sal never look at a counterfactual, so at K > 1 their K=1 run is
    copied rather than repeated. See K_INVARIANT. They are the CNN counterpart of
    the attribute ladder's B1-B5.

--train-k other than 1 breaks the symmetry with the attribute ladder, which
cannot train on an ensembled probability the way this loss can. Use it as a
diagnostic, not for the headline numbers.

Relationship to code/CNN_baselines/c2_cnn_all.py
------------------------------------------------
The encoder, dataset and train/eval definitions are ported from that script and
are deliberately a copy rather than an import: c2_cnn_all.py parses its arguments
and resolves all of its paths at module scope, so it cannot be imported without
executing that. Its paths are also hardcoded to the superseded experiment
(C2_custom_corrected folds, C0_custom Grad-CAMs), which is exactly what this file
exists to replace. The duplicated part is stable model code, not the retrieval
logic -- the KNN matching is still imported from one place, via c2_cf_sources.

Grad-CAMs come from the multi-label C0 under C0_final/, whose split directories
are named C2_dataset/ and Original_Test/ -- not the val/ and train/ of the old
single-label model.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import (                                          # noqa: E402
    RESULTS_DIR, RANDOM_SEED, N_FOLDS,
    load_folds, add_cf_args, cf_source_from_args, fold_dir, c2_data_path,
)

DATA_ROOT = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")

# Grad-CAM split directories written by c0_final_predictions.py. The old
# single-label C0 used val/ and train/; this model uses the split names.
CAM_SPLITS = ("C2_dataset", "Original_Test")

MODEL_KEYS = ["xi", "xi_sal", "cf", "cf_sal", "dual", "dual_sal", "dual_sal_scalars"]

# These two never open a counterfactual image: their dataset reads `path` and
# `correct` only. Folds, seed and data are identical at every K, so their result
# is identical at every K too -- retraining them once per K would burn GPU hours
# reproducing a number to the last decimal. main() copies the K=1 run instead.
K_INVARIANT = ("xi", "xi_sal")

# Columns the CNN needs. Parquet is columnar, so naming them turns a ~3,400-column
# read into a handful of megabytes.
# cf_paths / cf_prob / cf_probs are not in c2_data.csv -- load_folds attaches
# them from the cached pairing.
FOLD_COLS = ["path", "patient_id", "correct", "cf_paths", "cf_prob"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
_cam_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

CAM_BASE = None          # set by main() once --disease/--policy are known


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_cam(img_path):
    """Load a pre-computed C0 Grad-CAM as a (3,224,224) normalized tensor.

    c0_final_predictions.py stores the two directions separately -- cam_pos is
    the evidence arguing for the disease, cam_neg the evidence arguing for
    health -- because collapsing them to one map annihilates whichever direction
    loses. The three channels here are [pos, neg, pos-neg]: both directions plus
    their contrast, which keeps the (3,224,224) shape the pretrained DenseNet
    stem expects while giving the encoder the full signal.

    Older files carry only `cam`; those are replicated across all three channels
    as before, so a half-regenerated directory still loads.
    """
    fname = img_path.replace("/", "_") + ".npz"
    for split in CAM_SPLITS:
        npz_path = os.path.join(CAM_BASE, split, fname)
        if os.path.exists(npz_path):
            break
    else:
        raise FileNotFoundError(
            f"no Grad-CAM for {img_path!r} under {CAM_BASE} "
            f"(looked in {list(CAM_SPLITS)}). Run c0_final_predictions.py for "
            f"this disease, or pass --no-cam-models to skip saliency configs.")
    z = np.load(npz_path)
    if "cam_pos" in z.files:
        pos = torch.from_numpy(z["cam_pos"])                        # (7,7) float32
        neg = torch.from_numpy(z["cam_neg"])
        cam_t = torch.stack([pos, neg, (pos - neg + 1.0) / 2.0])[None]   # (1,3,7,7)
    else:
        cam = torch.from_numpy(z["cam"])
        cam_t = cam[None, None].repeat(1, 3, 1, 1)
    cam_t = F.interpolate(cam_t, (224, 224), mode="bilinear", align_corners=False)
    return _cam_normalize(cam_t.squeeze(0))


def _split_cf_paths(value):
    return [p.strip() for p in str(value).split("|")]


def _split_cf_probs(value):
    return [float(p) for p in str(value).split("|")]


def _entropy(p):
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _make_scalars(p_xi, cf_probs):
    """(K, 5): [p_xi, H(p_xi), p_cf_k, H(p_cf_k), p_xi - p_cf_k]."""
    return torch.tensor(
        [[p_xi, _entropy(p_xi), p_cf, _entropy(p_cf), p_xi - p_cf]
         for p_cf in cf_probs], dtype=torch.float32)


def _truncate_cfs(df, k):
    """Keep the k nearest counterfactuals per row. cf_paths is nearest-first.

    Training defaults to k=1 while evaluation keeps all K. That asymmetry is
    deliberate and it is the reason a K sweep is affordable at all: cost per
    epoch is linear in the CFs per sample, so training on all 10 would cost ten
    forward passes per row per epoch, while the ensembling that K buys happens at
    prediction time either way. --train-k 0 trains on all K if you want to test
    that the asymmetry is harmless.
    """
    if not k:
        return df
    df = df.copy()
    df["cf_paths"] = df["cf_paths"].map(
        lambda v: "|".join(_split_cf_paths(v)[:k]))
    if "cf_probs" in df.columns:
        df["cf_probs"] = df["cf_probs"].map(
            lambda v: "|".join(str(v).split("|")[:k]))
    return df


def _attach_scalar_cols(dfs, disease, prob_col):
    """Add query_prob, and per-CF cf_probs if the pairing did not carry them.

    c2_folds now stores the K individual CF probabilities next to their mean, so
    normally this only renames the query column. The lookup path stays for the
    pairings written before that column existed: every KNN CF is a real image
    whose own C0 probability is in c2_data.csv, so the values are recoverable.
    A generated (diffusion) CF has no row there -- it will have to supply its own
    probability table.
    """
    lookup = None
    for df in dfs:
        df["query_prob"] = df[prob_col]
        if "cf_probs" in df.columns and df["cf_probs"].notna().all():
            continue
        if lookup is None:
            c2 = pd.read_csv(c2_data_path(disease), usecols=["path", "prob"])
            lookup = dict(zip(c2["path"], c2["prob"]))
        missing = [p for cell in df["cf_paths"] for p in _split_cf_paths(cell)
                   if p not in lookup]
        if missing:
            raise KeyError(
                f"{len(missing)} CF paths have no C0 probability in "
                f"{c2_data_path(disease)} (first: {missing[0]!r})")
        df["cf_probs"] = df["cf_paths"].map(
            lambda v: "|".join(str(lookup[p]) for p in _split_cf_paths(v)))


def _assert_k_alignment(df, k, where):
    """Every row must carry exactly k CFs, and its scalars must match them.

    _stack_cf_inputs infers K from the tensor shapes and reshapes the scalars to
    the same layout. A row with a different count would silently pair one CF's
    image with another CF's probability rather than fail, so it is checked here
    where the message can still say which row.
    """
    counts = df["cf_paths"].map(lambda v: len(_split_cf_paths(v)))
    bad = counts[counts != k]
    if len(bad):
        raise ValueError(
            f"{where}: {len(bad)} rows carry {sorted(set(bad))} counterfactuals, "
            f"not {k} (first: {df['path'].iloc[bad.index[0]]!r})")
    if "cf_probs" in df.columns:
        pc = df["cf_probs"].map(lambda v: len(str(v).split("|")))
        if (pc != counts).any():
            raise ValueError(
                f"{where}: cf_probs and cf_paths disagree on the number of "
                f"counterfactuals -- their order is what pairs a CF image with "
                f"its own C0 probability")


# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════

class _Base(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _img(self, rel):
        img = Image.open(os.path.join(self.data_dir, str(rel))).convert("RGB")
        return self.transform(img) if self.transform else img


class QueryDataset(_Base):
    """Query image xi only."""
    def __getitem__(self, i):
        row = self.df.iloc[i]
        return self._img(row["path"]), float(row["correct"])


class CFDataset(_Base):
    """All K counterfactual images, stacked (K,3,224,224)."""
    def __getitem__(self, i):
        row = self.df.iloc[i]
        imgs = [self._img(p) for p in _split_cf_paths(row["cf_paths"])]
        return torch.stack(imgs), float(row["correct"])


class QueryCFDataset(_Base):
    """Query image + K CF images."""
    def __getitem__(self, i):
        row = self.df.iloc[i]
        q = self._img(row["path"])
        cfs = [self._img(p) for p in _split_cf_paths(row["cf_paths"])]
        return q, torch.stack(cfs), float(row["correct"])


class QueryDatasetWithSaliency(_Base):
    """Query image + its C0 Grad-CAM."""
    def __getitem__(self, i):
        row = self.df.iloc[i]
        return self._img(row["path"]), _load_cam(str(row["path"])), float(row["correct"])


class CFDatasetWithSaliency(_Base):
    """K CF images + their Grad-CAMs."""
    def __getitem__(self, i):
        row = self.df.iloc[i]
        paths = _split_cf_paths(row["cf_paths"])
        imgs = [self._img(p) for p in paths]
        cams = [_load_cam(p) for p in paths]
        return torch.stack(imgs), torch.stack(cams), float(row["correct"])


class QueryCFDatasetWithSaliency(_Base):
    """Query + K CFs + both sides' Grad-CAMs (four streams)."""
    def __getitem__(self, i):
        row = self.df.iloc[i]
        xi = str(row["path"])
        q, q_cam = self._img(xi), _load_cam(xi)
        paths = _split_cf_paths(row["cf_paths"])
        cfs = [self._img(p) for p in paths]
        cf_cams = [_load_cam(p) for p in paths]
        return q, torch.stack(cfs), q_cam, torch.stack(cf_cams), float(row["correct"])


class QueryCFDatasetWithSaliencyScalars(_Base):
    """As above, plus the (K,5) C0 scalar tensor before the label."""
    def __getitem__(self, i):
        row = self.df.iloc[i]
        xi = str(row["path"])
        q, q_cam = self._img(xi), _load_cam(xi)
        paths = _split_cf_paths(row["cf_paths"])
        cfs = [self._img(p) for p in paths]
        cf_cams = [_load_cam(p) for p in paths]
        scalars = _make_scalars(float(row["query_prob"]),
                                _split_cf_probs(row["cf_probs"]))
        return (q, torch.stack(cfs), q_cam, torch.stack(cf_cams),
                scalars, float(row["correct"]))


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════
# Each model produces one logit per "instance" -- a single image, or one
# (query, CF) pair. When a sample carries K counterfactuals the model is simply
# called K times and the logits are combined by the ensembling helpers; the
# models have no notion of K.

class _ImageEncoder(nn.Module):
    """DenseNet121 backbone -> 1024-dim pooled feature vector."""
    def __init__(self):
        super().__init__()
        backbone = models.densenet121(weights="IMAGENET1K_V1")
        self.features = backbone.features
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return torch.flatten(self.pool(self.relu(self.features(x))), 1)


class SingleEncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = _ImageEncoder()
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        return self.classifier(self.enc(x))


class DualEncoderCNN(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.query_enc = _ImageEncoder()
        self.cf_enc = _ImageEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, q, cf):
        return self.classifier(torch.cat([self.query_enc(q), self.cf_enc(cf)], dim=1))


class QuadEncoderCNN(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.query_enc = _ImageEncoder()
        self.cf_enc = _ImageEncoder()
        self.qcam_enc = _ImageEncoder()
        self.cfcam_enc = _ImageEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 4, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, q, cf, qcam, cfcam):
        return self.classifier(torch.cat([
            self.query_enc(q), self.cf_enc(cf),
            self.qcam_enc(qcam), self.cfcam_enc(cfcam)], dim=1))


class QuadEncoderScalarCNN(nn.Module):
    """QuadEncoderCNN whose head also sees the C0 scalars. The head keeps the
    same shape as the base config (its input only widens by scalar_hidden) so
    the comparison isolates the scalar contribution."""
    def __init__(self, hidden_dim=256, dropout=0.3, scalar_dim=5, scalar_hidden=32):
        super().__init__()
        self.query_enc = _ImageEncoder()
        self.cf_enc = _ImageEncoder()
        self.qcam_enc = _ImageEncoder()
        self.cfcam_enc = _ImageEncoder()
        self.scalar_embed = nn.Sequential(
            nn.Linear(scalar_dim, scalar_hidden), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 4 + scalar_hidden, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, q, cf, qcam, cfcam, scalars):
        return self.classifier(torch.cat([
            self.query_enc(q), self.cf_enc(cf),
            self.qcam_enc(qcam), self.cfcam_enc(cfcam),
            self.scalar_embed(scalars)], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# CF ENSEMBLING
# ══════════════════════════════════════════════════════════════════════════════

def _stack_cf_inputs(*streams):
    """Broadcast a mix of (B,3,H,W) and (B,K,3,H,W) streams to a common K, then
    fold K into the batch dim. Returns (flattened_streams, B, K)."""
    k = max((s.shape[1] for s in streams if s.dim() == 5), default=1)
    flat, b = [], streams[0].shape[0]
    for s in streams:
        if s.dim() == 4:
            s = s.unsqueeze(1).expand(-1, k, *s.shape[1:])
        b = s.shape[0]
        flat.append(s.reshape(b * k, *s.shape[2:]))
    return flat, b, k


def _ensemble_probs(logits, b, k):
    """K logits per sample -> one probability, averaging sigmoid over the K CFs.

    Always fp32: BCELoss on fp16 probabilities can under/overflow near 0 and 1,
    and autocast rejects it outright, so callers invoke this OUTSIDE the autocast
    region even though the encoder forward pass runs inside it.
    """
    return torch.sigmoid(logits.float()).view(b, k).mean(dim=1)


def make_weighted_bce(pos_weight):
    """BCE on probabilities with the positive class upweighted -- the same
    rebalancing BCEWithLogitsLoss(pos_weight=...) does, reimplemented because
    training operates on ensembled probabilities rather than a single logit."""
    base = nn.BCELoss(reduction="none")

    def loss_fn(probs, labels):
        labels = labels.float()
        return (base(probs, labels) * (1.0 + labels * (pos_weight - 1.0))).mean()

    return loss_fn


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL PER MODE
# ══════════════════════════════════════════════════════════════════════════════
# Scalars are NOT routed through _stack_cf_inputs (its dim checks are
# image-specific); the (B,K,5) tensor flattens with the same sample-major,
# CF-minor ordering, so a plain reshape keeps each CF's scalars with its images.

def _run_batch(model, batch, device, mode):
    """Unpack one batch by mode -> (ensembled_probs, labels)."""
    *inputs, labels = batch
    if mode == "quad_scalars":
        *imgs, scal = inputs
        streams, b, k = _stack_cf_inputs(*[x.to(device) for x in imgs])
        logits = model(*streams, scal.to(device).reshape(-1, scal.shape[-1])).squeeze(1)
    else:
        streams, b, k = _stack_cf_inputs(*[x.to(device) for x in inputs])
        logits = model(*streams).squeeze(1)
    return logits, b, k, labels


def _train_epoch(model, loader, optimizer, criterion, device, scaler, mode):
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        with autocast("cuda"):
            logits, b, k, labels = _run_batch(model, batch, device, mode)
        loss = criterion(_ensemble_probs(logits, b, k), labels.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / max(len(loader), 1)


def _evaluate(model, loader, device, mode):
    """Returns (auc, ensembled probs, labels, per-slot probs of shape (n, K)).

    The per-slot matrix is what makes a smaller K readable off a larger run: the
    network never sees K, so column j is the same number whether the run was K=5
    or K=10, and mean(columns 0..4) IS the K=5 prediction.
    """
    model.eval()
    probs, labels_all, per_slot = [], [], []
    with torch.no_grad(), autocast("cuda"):
        for batch in loader:
            logits, b, k, labels = _run_batch(model, batch, device, mode)
            slot = torch.sigmoid(logits.float()).view(b, k)
            per_slot.append(slot.cpu().numpy())
            probs.extend(slot.mean(dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())
    return (roc_auc_score(labels_all, probs), probs, labels_all,
            np.concatenate(per_slot, axis=0))


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "xi":    dict(name="CNN Xi", dataset=QueryDataset,
                  make_model=SingleEncoderCNN, mode="single", needs_cam=False),
    "xi_sal": dict(name="CNN Xi + Saliency", dataset=QueryDatasetWithSaliency,
                   make_model=DualEncoderCNN, mode="dual", needs_cam=True),
    "cf":    dict(name="CNN CF", dataset=CFDataset,
                  make_model=SingleEncoderCNN, mode="single", needs_cam=False),
    "cf_sal": dict(name="CNN CF + Saliency", dataset=CFDatasetWithSaliency,
                   make_model=DualEncoderCNN, mode="dual", needs_cam=True),
    "dual":  dict(name="Dual Encoder", dataset=QueryCFDataset,
                  make_model=DualEncoderCNN, mode="dual", needs_cam=False),
    "dual_sal": dict(name="Dual Encoder + Saliency",
                     dataset=QueryCFDatasetWithSaliency,
                     make_model=QuadEncoderCNN, mode="quad", needs_cam=True),
    "dual_sal_scalars": dict(name="Dual Encoder + Saliency + Scalars",
                             dataset=QueryCFDatasetWithSaliencyScalars,
                             make_model=QuadEncoderScalarCNN,
                             mode="quad_scalars", needs_cam=True),
}


# ══════════════════════════════════════════════════════════════════════════════
# CV
# ══════════════════════════════════════════════════════════════════════════════

def reuse_k_invariant(key, out_dir, args, src_root):
    """Copy a K-invariant model's K=1 results into this K's output tree.

    Returns True if the copy happened. The marker file is the point: a later
    reader must be able to tell that cf_10/images/xi was not trained at K=10, and
    why that is not a shortcut but an identity.
    """
    src = os.path.join(src_root, "images", key)
    dst = os.path.join(out_dir, key)
    if not os.path.exists(os.path.join(src, "fold_aucs.csv")):
        return False
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name == "reused_from.json" or name.endswith(".pt"):
            continue
        shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    with open(os.path.join(dst, "reused_from.json"), "w") as fh:
        json.dump({"copied_from": src, "reason":
                   f"{key} never reads a counterfactual, so its result is "
                   f"identical at every K on these folds",
                   "cf_count": args.cf_count}, fh, indent=2)
    aucs = pd.read_csv(os.path.join(dst, "fold_aucs.csv"))["auc"]
    print(f"\n{'='*72}\n  {MODEL_REGISTRY[key]['name']}  ({key})\n{'='*72}")
    print(f"  K-invariant: copied the K=1 run ({aucs.mean():.4f} ± {aucs.std():.4f})"
          f"\n  from {src}\n  (--no-reuse-k-invariant to retrain instead)")
    return True


def run_cv(key, folds, out_dir, args, device):
    cfg = MODEL_REGISTRY[key]
    model_dir = os.path.join(out_dir, key)
    os.makedirs(model_dir, exist_ok=True)

    # Resume at model granularity. fold_aucs.csv is written only after all five
    # folds of THIS model finish, so its presence means the model is done. The
    # full seven-model ladder is ~40h of GPU -- more than the 24h queue maximum
    # -- so a run is expected to span submissions, and each one must pick up
    # where the last was killed rather than start over.
    done_path = os.path.join(model_dir, "fold_aucs.csv")
    if os.path.exists(done_path) and not args.overwrite:
        aucs = pd.read_csv(done_path)["auc"].tolist()
        print(f"\n{'='*72}\n  {cfg['name']}  ({key})\n{'='*72}")
        print(f"  already complete ({np.mean(aucs):.4f} ± {np.std(aucs):.4f})"
              f" -- skipping (--overwrite to retrain)")
        return aucs

    print(f"\n{'='*72}\n  {cfg['name']}  ({key})\n{'='*72}")

    fold_aucs, rows = [], []
    for fold_idx, train_df, test_df in folds:
        print(f"\n{'-'*60}\n  [{key}] FOLD {fold_idx + 1}/{args.n_folds}\n{'-'*60}")

        # Validation split for checkpoint selection, grouped by patient so the
        # test fold is touched exactly once, after training.
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac,
                                random_state=RANDOM_SEED)
        tr_idx, va_idx = next(gss.split(train_df, groups=train_df["patient_id"]))
        train_sub = train_df.iloc[tr_idx].reset_index(drop=True)
        val_df = train_df.iloc[va_idx].reset_index(drop=True)

        # Train on the k nearest CFs (default 1); val/test keep all K so
        # checkpoint selection sees the same ensemble protocol as the final
        # evaluation.
        train_sub = _truncate_cfs(train_sub, args.train_k)
        train_k = args.train_k or args.cf_count
        _assert_k_alignment(train_sub, train_k, f"fold {fold_idx} train")
        _assert_k_alignment(val_df, args.cf_count, f"fold {fold_idx} val")
        _assert_k_alignment(test_df, args.cf_count, f"fold {fold_idx} test")
        print(f"  train {len(train_sub):,} (K={train_k}) | val {len(val_df):,} "
              f"| test {len(test_df):,} (K={args.cf_count})")

        def loader(df, shuffle, batch_size):
            return DataLoader(cfg["dataset"](df, DATA_ROOT, transform),
                              batch_size=batch_size, shuffle=shuffle,
                              num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0,
                              pin_memory=True)

        test_loader = loader(test_df, False, args.eval_batch_size)
        model = cfg["make_model"]().to(device)
        ckpt = os.path.join(model_dir, f"fold_{fold_idx}_best.pt")

        if args.eval_from:
            # Training never sees a counterfactual beyond the nearest one, so a
            # checkpoint trained at K=1 IS the K=10 model. Re-running ten epochs
            # to arrive at the same weights would cost ~30 GPU-hours per disease
            # and change nothing. Loading it also holds model selection fixed
            # across K, which is what makes the K comparison clean: the only
            # thing that moves is how many counterfactuals the fixed model is
            # scored against.
            src_ckpt = os.path.join(args.eval_from, "images", key,
                                    f"fold_{fold_idx}_best.pt")
            if not os.path.exists(src_ckpt):
                raise FileNotFoundError(
                    f"--eval-from is set but {src_ckpt} does not exist. "
                    f"Train {key} at that K first, or drop --eval-from to train "
                    f"from scratch here.")
            model.load_state_dict(torch.load(src_ckpt, map_location=device))
            if os.path.abspath(src_ckpt) != os.path.abspath(ckpt):
                shutil.copy2(src_ckpt, ckpt)
            print(f"  loaded {src_ckpt}")
        else:
            train_loader = loader(train_sub, True, args.batch_size)
            val_loader = loader(val_df, False, args.eval_batch_size)
            pos_rate = train_sub["correct"].mean()
            criterion = make_weighted_bce((1 - pos_rate) / pos_rate)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scaler = GradScaler("cuda")

            best = -1.0
            for epoch in range(args.epochs):
                t0 = time.time()
                loss = _train_epoch(model, train_loader, optimizer, criterion,
                                    device, scaler, cfg["mode"])
                val_auc, _, _, _ = _evaluate(model, val_loader, device, cfg["mode"])
                flag = ""
                if val_auc > best:
                    best, flag = val_auc, "  *"
                    torch.save(model.state_dict(), ckpt)
                print(f"  epoch {epoch+1:02d} | loss {loss:.4f} | val AUC "
                      f"{val_auc:.4f} | {time.time()-t0:.0f}s{flag}")
            model.load_state_dict(torch.load(ckpt, map_location=device))

        auc, probs, labels, slot_probs = _evaluate(model, test_loader, device,
                                                   cfg["mode"])
        fold_aucs.append(auc)
        print(f"  fold {fold_idx} TEST AUC = {auc:.4f}  "
              f"(K={slot_probs.shape[1]}, per-slot "
              f"{', '.join(f'{roc_auc_score(labels, slot_probs[:, j]):.4f}' for j in range(min(slot_probs.shape[1], 4)))}"
              f"{', ...' if slot_probs.shape[1] > 4 else ''})")

        fpr, tpr, _ = roc_curve(labels, probs)
        np.savez(os.path.join(model_dir, f"fold_{fold_idx}_roc.npz"),
                 fpr=fpr, tpr=tpr, probs=np.asarray(probs),
                 labels=np.asarray(labels))
        # Per-slot probabilities: column j is the score against the j-th nearest
        # counterfactual, independent of K, so c2_derive_k.py can average any
        # prefix of them into a smaller-K result.
        np.savez_compressed(
            os.path.join(model_dir, f"fold_{fold_idx}_slot_probs.npz"),
            paths=test_df["path"].values.astype(str),
            patient_id=test_df["patient_id"].values.astype(str),
            labels=np.asarray(labels, dtype=np.int8),
            cf_paths=test_df["cf_paths"].values.astype(str),
            # float64: the derived-K arithmetic must reproduce a computed K
            # exactly, and float32 rounding of near-tied probabilities does not.
            probs=slot_probs.astype(np.float64))
        pd.DataFrame({"path": test_df["path"].values,
                      "patient_id": test_df["patient_id"].values,
                      "correct": labels, f"{key}_prob": probs}).to_csv(
            os.path.join(model_dir, f"fold_{fold_idx}_predictions.csv"), index=False)
        rows.append({"fold": fold_idx, "auc": auc, "n_test": len(test_df)})

    pd.DataFrame(rows).to_csv(os.path.join(model_dir, "fold_aucs.csv"), index=False)
    print(f"\n  {cfg['name']}: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    return fold_aucs


def main():
    global CAM_BASE

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--policy", default="ignore",
                    help="C0 uncertainty policy; picks the Grad-CAM directory")
    ap.add_argument("--models", default=",".join(MODEL_KEYS))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="training samples per step. Each sample costs "
                         "train_k x (number of encoders) images of GPU memory.")
    ap.add_argument("--eval-batch-size", type=int, default=None,
                    help="default: --batch-size scaled down by K, so a K sweep "
                         "keeps a constant number of images in flight")
    ap.add_argument("--train-k", type=int, default=1,
                    help="counterfactuals per sample during training (0 = all K). "
                         "Evaluation always uses all K.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if cv_summary.csv already exists")
    ap.add_argument("--eval-from", default=None, metavar="RESULTS_ROOT",
                    help="skip training and score the checkpoints under "
                         "RESULTS_ROOT/images/<model>/fold_<i>_best.pt against "
                         "this K. Training is K-independent (--train-k 1), so a "
                         "K=1 checkpoint is already the K=10 model.")
    ap.add_argument("--no-reuse-k-invariant", dest="reuse_k_invariant",
                    action="store_false",
                    help=f"retrain {', '.join(K_INVARIANT)} at K>1 instead of "
                         f"copying the K=1 run they are identical to")
    add_cf_args(ap)
    args = ap.parse_args()

    if args.train_k != 1:
        print(f"\n  !! --train-k {args.train_k} (not 1): the attribute ladder "
              f"always trains on the nearest CF, so this run is no longer a "
              f"like-for-like comparison with it.\n")
    if args.train_k and args.train_k > args.cf_count:
        raise SystemExit(f"--train-k {args.train_k} exceeds --cf-count "
                         f"{args.cf_count}: there are only {args.cf_count} "
                         f"counterfactuals per row to train on")

    # Peak GPU memory is (batch x CFs per sample x encoders) images, so a fixed
    # batch size at K=10 asks for ten times the activations that the same number
    # asked for at K=1 -- an immediate OOM on the quad-encoder configs. Scale the
    # EVAL batch by K instead, which changes throughput and nothing else: batches
    # are independent at inference, so predictions are identical either way.
    if args.eval_batch_size is None:
        args.eval_batch_size = max(1, args.batch_size // max(args.cf_count, 1))

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CAM_BASE = os.path.join(RESULTS_DIR, "C0_final", f"multilabel_{args.policy}",
                            args.disease, "gradcam")

    src = cf_source_from_args(args)
    keys = [k.strip() for k in args.models.split(",")]
    unknown = [k for k in keys if k not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown model keys {unknown}; expected {MODEL_KEYS}")

    if any(MODEL_REGISTRY[k]["needs_cam"] for k in keys) and not os.path.isdir(CAM_BASE):
        raise SystemExit(
            f"Grad-CAM directory not found: {CAM_BASE}\n"
            f"Run c0_final_predictions.py --disease {args.disease} first, or pick "
            f"only non-saliency models (--models xi,cf,dual).")

    out_dir = os.path.join(RESULTS_DIR, "C2_final_results", args.disease,
                           src.name, f"cf_{args.cf_count}", "images")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  IMAGE MODELS: {args.disease} | {src.describe()} | K={args.cf_count}")
    print(f"{'='*72}")
    print(f"  device:   {device}")
    print(f"  images:   {DATA_ROOT}")
    print(f"  gradcam:  {CAM_BASE}")
    print(f"  folds:    {fold_dir(args.disease, src.name, args.cf_count)}")
    print(f"  results:  {out_dir}")
    print(f"  models:   {', '.join(keys)}")
    print(f"  batch:    train {args.batch_size} (K={args.train_k or args.cf_count})"
          f" | eval {args.eval_batch_size} (K={args.cf_count})")

    # Read the folds once: every model trains on the same in-memory split.
    prob_col = f"{args.disease}_prob"
    cols = FOLD_COLS + [prob_col]
    # materialize=False: the CNN needs only paths, labels and the pairing, so the
    # ~3,400-column attribute space is never rebuilt.
    folds = list(load_folds(args.disease, src, args.cf_count,
                            n_folds=args.n_folds, materialize=False, columns=cols))

    if any(MODEL_REGISTRY[k]["mode"] == "quad_scalars" for k in keys):
        _attach_scalar_cols([d for _, tr, te in folds for d in (tr, te)],
                            args.disease, prob_col)

    # With --eval-from there is nothing to save by copying: xi and xi_sal are
    # cheap to score and DO need to be scored here, because the derived-K step
    # reads per-slot probabilities and a copied K=1 directory has none.
    if args.eval_from and args.reuse_k_invariant:
        args.reuse_k_invariant = False
        print("  (--eval-from: scoring the K-invariant models too, so they "
              "produce per-slot probabilities for c2_derive_k.py)")

    k1_root = os.path.join(RESULTS_DIR, "C2_final_results", args.disease,
                           src.name, "cf_1")
    if args.eval_from:
        # A K=1 ladder killed at the wall clock leaves some models untrained.
        # Score the ones that exist and name the ones that do not, rather than
        # aborting the whole pass on the first missing checkpoint.
        have = [k for k in keys
                if all(os.path.exists(os.path.join(args.eval_from, "images", k,
                                                   f"fold_{i}_best.pt"))
                       for i in range(args.n_folds))]
        skipped = [k for k in keys if k not in have]
        if skipped:
            print(f"  no complete checkpoint set under {args.eval_from} for: "
                  f"{', '.join(skipped)} -- skipping (train them at K=1 first, "
                  f"then rerun this)")
        if not have:
            raise SystemExit(
                f"none of {keys} has all {args.n_folds} checkpoints under "
                f"{args.eval_from}/images")
        keys = have

    for key in keys:
        if (args.cf_count > 1 and key in K_INVARIANT and args.reuse_k_invariant
                and not args.overwrite
                and not os.path.exists(os.path.join(out_dir, key, "fold_aucs.csv"))
                and reuse_k_invariant(key, out_dir, args, k1_root)):
            continue
        run_cv(key, folds, out_dir, args, device)

    # Summarize every model present on disk, not just this submission's keys:
    # the ladder is split across submissions (and may be split across jobs via
    # MODELS=), so rebuilding from only `keys` would drop earlier models.
    rows = []
    for k in MODEL_KEYS:
        p = os.path.join(out_dir, k, "fold_aucs.csv")
        if not os.path.exists(p):
            continue
        v = pd.read_csv(p)["auc"].tolist()
        rows.append({"model": k, "name": MODEL_REGISTRY[k]["name"],
                     "auc_mean": float(np.mean(v)), "auc_std": float(np.std(v)),
                     "auc_folds": ";".join(f"{a:.4f}" for a in v)})
    df = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    df.to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)

    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump({"disease": args.disease, "policy": args.policy,
                   "cf_source": src.name, "cf_source_description": src.describe(),
                   "cf_count": args.cf_count, "train_k": args.train_k,
                   "eval_from": args.eval_from,
                   "n_folds": args.n_folds,
                   "models": df["model"].tolist(), "epochs": args.epochs,
                   "batch_size": args.batch_size,
                   "eval_batch_size": args.eval_batch_size, "lr": args.lr,
                   "val_frac": args.val_frac, "seed": RANDOM_SEED}, fh, indent=2)

    print(f"\n{'='*72}\n  SUMMARY\n{'='*72}")
    for _, r in df.iterrows():
        print(f"  {r['name']:<34} {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")
    print(f"\n  wrote {os.path.join(out_dir, 'cv_summary.csv')}")


if __name__ == "__main__":
    main()
