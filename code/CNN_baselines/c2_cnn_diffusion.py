"""
c2_cnn_diffusion.py
====================
Late-fusion CNN baselines for C2 quality control using real diffusion-generated
counterfactuals (see cf_generate_diffusion_counterfactuals.py), instead of the
matched/simulated CF pool used by c2_cnn_all.py.

Counterpart to c2_cnn_diffusion_early_fusion.py: each input is encoded with its
own DenseNet and the pooled embeddings are concatenated before the classifier
(late fusion), rather than channel-concatenating the raw inputs into one
DenseNet (early fusion). Every input is loaded as grayscale, replicated to 3
channels, and normalized with ImageNet RGB stats — the only difference between
the two scripts is the fusion point.

Trains models in sequence using the same CV fold splits:
  1. CNN Xi                — query image xi only
  2. CNN CF                — diffusion counterfactual image only
  3. CNN Xi + CF            — query + CF via two encoders
  4. CNN Xi + Saliency      — query + C0 Grad-CAM of xi via two encoders
  5. CNN Xi + Saliency + CF — query + C0 Grad-CAM of xi + CF via three encoders
  6. CNN Xi + Saliency + CF + Scalars — config 5 plus a per-CF vector of C0
     prediction scalars [p_xi, H(p_xi), p_cf_k, H(p_cf_k), p_xi − p_cf_k]
     embedded and concatenated at the classifier head

Grad-CAMs exist only for xi (there is no CF-side CAM for diffusion CFs), so
every saliency config uses cam_xi.

Counterfactuals: every query has 3 diffusion counterfactuals available, from
the manifest cf_manifest_10_0.25_n3.csv (t_start=10, guidance_weight=0.25). The
manifest is long-format (3 rows per query, keyed by cf_idx); it is pivoted here
into the pipe-separated `cf_paths` convention the rest of the CNN family uses.
Pass --cf_count 3 (default) to use all of them, or --cf_count 1 to keep only
the single most confident CF per query — both read the same manifest, so
results are directly comparable. CF-facing datasets load all K images and the
corresponding encoder is run once per counterfactual to get K predictions,
which are combined into one final prediction by averaging their
probabilities — an ensemble over the K counterfactuals, mirroring
c2_cnn_diffusion_early_fusion.py and the rest of the CNN family.

Use --models to train only a subset (default: all five), e.g.:

    python c2_cnn_diffusion.py --models xi_cf xi_saliency_cf

Available keys: xi, cf, xi_cf, xi_saliency, xi_saliency_cf, xi_saliency_cf_scalars

Results for each model are written to their own output directory.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit   
from torchvision import models, transforms
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit  # was: StratifiedKFold, train_test_split
# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_CHOICES = ['xi', 'cf', 'xi_cf', 'xi_saliency', 'xi_saliency_cf',
                 'xi_saliency_cf_scalars']

# Both counts are read from the same manifest (t_start=10, w=0.25, 3 CFs/query
# generated); --cf_count 1 just keeps the single best CF per query instead of
# all 3, so K=1 and K=3 runs are directly comparable (same generation settings).
T_START           = 10
GUIDANCE_WEIGHT   = 0.25
MANIFEST_CF_COUNT = 3

parser = argparse.ArgumentParser()
parser.add_argument('--disease', type=str, default='effusion')
parser.add_argument('--cf_count', type=int, default=3, choices=[1, 3],
                    help="Counterfactuals per query to use. 3 uses every CF in the "
                         "manifest; 1 keeps only the single best CF per query "
                         "(flipped if any, else best-attempt).")
parser.add_argument(
    '--models', type=str, nargs='+', default=MODEL_CHOICES, choices=MODEL_CHOICES,
    help=(
        "Which input configurations to train. Choose any subset of: "
        f"{', '.join(MODEL_CHOICES)}. Defaults to all of them."
    ),
)
parser.add_argument('--val_frac', type=float, default=0.15,
                    help="Fraction of each fold's train split held out for validation "
                         "(checkpoint selection).")
parser.add_argument('--seed', type=int, default=42,
                    help="Seed for the train/val split (fixed for reproducibility, "
                         "and shared across separate --models job submissions so "
                         "every config sees the same split per fold).")
parser.add_argument('--batch_size', type=int, default=32,
                    help="Training/eval batch size. Lower if you hit CUDA OOM.")
args = parser.parse_args()

DISEASE      = args.disease
CF_COUNT     = args.cf_count
SELECTED     = args.models
VAL_FRAC     = args.val_frac
SEED         = args.seed
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
DATA_DIR     = DATA_ROOT
C2_DATA_CSV  = os.path.join(RESULTS_DIR, f"C2_custom/{DISEASE}/c2_data.csv")
MANIFEST_PATH = os.path.join(RESULTS_DIR, f"diffusion_cf/cf_manifest_{T_START}_{GUIDANCE_WEIGHT}_n{MANIFEST_CF_COUNT}.csv")
RES_BASE     = os.path.join(RESULTS_DIR, "C2_cnn_diffusion")
CAM_BASE     = os.path.join(RESULTS_DIR, f"C0_custom/{DISEASE}/gradcam")
N_FOLDS      = 5
RANDOM_SEED  = 42
N_EPOCHS     = 10
BATCH_SIZE   = args.batch_size
LR           = 1e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f"Disease: {DISEASE} | CF count: {CF_COUNT}")
print(f"Manifest: {os.path.basename(MANIFEST_PATH)}")
print(f"Models to train: {SELECTED}")
print("Using device:", DEVICE)

# ── Transforms ─────────────────────────────────────────────────────────────────
# Resize((224,224)) is an anisotropic squash, and deliberately so: it is the
# transform the CF generator applied, and it is what keeps xi pixel-aligned
# with its counterfactual. Do not replace it with a crop.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_cam_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# ── CAM helper ─────────────────────────────────────────────────────────────────
def _load_cam(img_path):
    """Load a pre-computed C0 Grad-CAM as a (3,224,224) normalized tensor."""
    fname = img_path.replace('/', '_') + '.npz'
    # CheXpert paths always contain /train/ regardless of model split; try both dirs.
    for split in ('val', 'train'):
        npz_path = os.path.join(CAM_BASE, split, fname)
        if os.path.exists(npz_path):
            break
    cam = np.load(npz_path)['cam']                                    # (7,7) float32
    cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)           # (1,1,7,7)
    cam_t = F.interpolate(cam_t, (224, 224), mode='bilinear', align_corners=False)
    cam_t = cam_t.repeat(1, 3, 1, 1).squeeze(0)                       # (3,224,224)
    return _cam_normalize(cam_t)


def _split_cf_paths(value):
    """cf_paths holds K pipe-separated paths (K = CF_COUNT)."""
    return [p.strip() for p in str(value).split('|')]


# ── C0 prediction scalars ──────────────────────────────────────────────────────
def _split_cf_probs(value):
    """cf_probs holds K pipe-separated C0 probabilities, aligned with cf_paths."""
    return [float(p) for p in str(value).split('|')]


def _entropy(p):
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _make_scalars(p_xi, cf_probs):
    """C0 prediction scalars per CF -> (K, 5) float32:
    [p_xi, H(p_xi), p_cf_k, H(p_cf_k), p_xi - p_cf_k]."""
    return torch.tensor(
        [[p_xi, _entropy(p_xi), p_cf, _entropy(p_cf), p_xi - p_cf]
         for p_cf in cf_probs], dtype=torch.float32)


# ── Fold data builder ──────────────────────────────────────────────────────────
def build_fold_dfs():
    """Attach the diffusion CF paths per query, then split into N_FOLDS
    stratified folds (stratified on `correct`).

    The manifest is long-format (columns path, cf_idx, cf_path, cf_prob,
    flipped — one row per CF, K=MANIFEST_CF_COUNT rows per query). It is
    pivoted into a single pipe-separated `cf_paths` cell per query, matching
    the convention the rest of the CNN family uses.

    CF selection mirrors c2_cv_pipeline_diffusion_cf.py: keep the CFs that
    actually flipped the classifier; for any query where none flipped, fall
    back to the best attempt (the CF whose probability is furthest from 0.5).
    Within each query, CFs are ranked by that same "furthest from 0.5"
    confidence and truncated to the requested CF_COUNT — so --cf_count 1 keeps
    the single most confident CF per query rather than an arbitrary one, and
    --cf_count 3 keeps all of them (order doesn't matter for the latter, since
    ensembling averages predictions over K). Queries left with fewer CFs than
    requested (only possible if fewer than CF_COUNT flipped) are padded by
    repeating their best CF, so every row carries exactly CF_COUNT.
    """
    df = pd.read_csv(C2_DATA_CSV, usecols=['path', 'prob', 'correct', 'patient_id'])
    df = df.rename(columns={'prob': 'query_prob'})
    manifest = pd.read_csv(MANIFEST_PATH)

    n_total, n_flip = len(manifest), int(manifest['flipped'].sum())
    print(f"  Manifest: {n_total:,} CFs, {n_flip:,} flipped ({n_flip/max(n_total,1):.1%})")

    manifest = manifest.copy()
    manifest['dist_from_mid'] = (manifest['cf_prob'] - 0.5).abs()
    ranked = manifest.sort_values(['path', 'dist_from_mid'], ascending=[True, False])

    flipped = ranked[ranked['flipped'] == 1]
    cf_lists = flipped.groupby('path')['cf_path'].apply(list)

    # Best-attempt fallback for queries where no CF flipped: rank all CFs
    # (flipped or not) by confidence and use the best ones instead.
    missing = set(manifest['path']) - set(cf_lists.index)
    if missing:
        print(f"  Fallback: {len(missing)} quer{'y' if len(missing)==1 else 'ies'} "
              f"had no flipped CF — using best-attempt CF")
        un = ranked[ranked['path'].isin(missing)]
        cf_lists = pd.concat([cf_lists, un.groupby('path')['cf_path'].apply(list)])

    # Truncate to the CF_COUNT most confident CFs, padding by cycling if a
    # query has fewer than CF_COUNT available.
    def _select(paths):
        return [paths[i % len(paths)] for i in range(CF_COUNT)]

    cf_paths = cf_lists.apply(_select).apply('|'.join).rename('cf_paths')
    best_cf = cf_paths.reset_index()

    df = df.merge(best_cf, on='path', how='inner')
    n_cfs = df['cf_paths'].map(lambda v: len(_split_cf_paths(v)))
    assert (n_cfs == CF_COUNT).all(), f"expected {CF_COUNT} CFs/query, got {n_cfs.unique()}"

    # Per-CF C0 probabilities for the scalar configs. Manifest cf_path values
    # are globally unique, so mapping each selected path through a lookup
    # reproduces the exact truncation/cycling applied to cf_paths above,
    # keeping paths and probs 1:1 by construction.
    prob_lookup = dict(zip(manifest['cf_path'], manifest['cf_prob']))
    df['cf_probs'] = df['cf_paths'].map(
        lambda v: '|'.join(str(prob_lookup[p]) for p in _split_cf_paths(v)))

    print(f"Samples with diffusion CF: {len(df):,}  "
          f"(Correct: {(df['correct']==1).sum():,} | Incorrect: {(df['correct']==0).sum():,})")

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_dfs = [None] * N_FOLDS
    for fold_idx, (_, test_idx) in enumerate(skf.split(df, df['correct'], groups=df['patient_id'])):
        fold_dfs[fold_idx] = df.iloc[test_idx].reset_index(drop=True)

    for i in range(N_FOLDS):
        for j in range(i + 1, N_FOLDS):
            overlap = set(fold_dfs[i]['patient_id']) & set(fold_dfs[j]['patient_id'])
            assert len(overlap) == 0, f"Patient overlap between fold {i} and {j}: {len(overlap)}"
    print(f"  Verified: zero patient overlap across {N_FOLDS} folds")

    return fold_dfs

# ── Datasets ───────────────────────────────────────────────────────────────────
# Every dataset yields (stack(s), label). CF-facing datasets yield K images per
# stream, stacked as (K, 3, 224, 224) — one per counterfactual — so the model
# can be run once per CF and the K resulting predictions ensembled.
class QueryDataset(Dataset):
    """Query image xi only. No CF, so K=1 (broadcast at ensembling time)."""
    def __init__(self, df, data_dir, transform=None):
        self.df, self.data_dir, self.transform = df.reset_index(drop=True), data_dir, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.data_dir, str(row['path']))).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, float(row['correct'])


class CFDataset(Dataset):
    """All K counterfactual images, stacked as (K, 3, 224, 224)."""
    def __init__(self, df, data_dir, transform=None):
        self.df, self.data_dir, self.transform = df.reset_index(drop=True), data_dir, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgs = []
        for cf_path in _split_cf_paths(row['cf_paths']):
            img = Image.open(os.path.join(self.data_dir, cf_path)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        return torch.stack(imgs), float(row['correct'])


class QueryCFDataset(Dataset):
    """Query image + all K CF images, the latter stacked as (K, 3, 224, 224)."""
    def __init__(self, df, data_dir, transform=None):
        self.df, self.data_dir, self.transform = df.reset_index(drop=True), data_dir, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q_img = Image.open(os.path.join(self.data_dir, str(row['path']))).convert('RGB')
        if self.transform:
            q_img = self.transform(q_img)
        cf_imgs = []
        for cf_path in _split_cf_paths(row['cf_paths']):
            cf_img = Image.open(os.path.join(self.data_dir, cf_path)).convert('RGB')
            if self.transform:
                cf_img = self.transform(cf_img)
            cf_imgs.append(cf_img)
        return q_img, torch.stack(cf_imgs), float(row['correct'])


class QueryDatasetWithSaliency(Dataset):
    """Query image xi + C0 Grad-CAM of xi as a second encoder stream. No CF,
    so K=1 (broadcast at ensembling time)."""
    def __init__(self, df, data_dir, transform=None):
        self.df, self.data_dir, self.transform = df.reset_index(drop=True), data_dir, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.data_dir, str(row['path']))).convert('RGB')
        if self.transform:
            img = self.transform(img)
        cam = _load_cam(str(row['path']))
        return img, cam, float(row['correct'])


class QueryCFSaliencyDataset(Dataset):
    """Query image + C0 Grad-CAM of xi + all K CF images, the latter stacked
    as (K, 3, 224, 224)."""
    def __init__(self, df, data_dir, transform=None):
        self.df, self.data_dir, self.transform = df.reset_index(drop=True), data_dir, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xi_path = str(row['path'])
        q_img = Image.open(os.path.join(self.data_dir, xi_path)).convert('RGB')
        if self.transform:
            q_img = self.transform(q_img)
        cam = _load_cam(xi_path)
        cf_imgs = []
        for cf_path in _split_cf_paths(row['cf_paths']):
            cf_img = Image.open(os.path.join(self.data_dir, cf_path)).convert('RGB')
            if self.transform:
                cf_img = self.transform(cf_img)
            cf_imgs.append(cf_img)
        return q_img, cam, torch.stack(cf_imgs), float(row['correct'])


class QueryCFSaliencyScalarsDataset(Dataset):
    """QueryCFSaliencyDataset plus the C0 prediction scalars as a (K, 5)
    tensor inserted before the label. Stream order (query, cam, cf) is
    preserved so encoder assignment matches TripleEncoderCNN."""
    def __init__(self, df, data_dir, transform=None):
        self.df, self.data_dir, self.transform = df.reset_index(drop=True), data_dir, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xi_path = str(row['path'])
        q_img = Image.open(os.path.join(self.data_dir, xi_path)).convert('RGB')
        if self.transform:
            q_img = self.transform(q_img)
        cam = _load_cam(xi_path)
        cf_imgs = []
        for cf_path in _split_cf_paths(row['cf_paths']):
            cf_img = Image.open(os.path.join(self.data_dir, cf_path)).convert('RGB')
            if self.transform:
                cf_img = self.transform(cf_img)
            cf_imgs.append(cf_img)
        scalars = _make_scalars(float(row['query_prob']),
                                _split_cf_probs(row['cf_probs']))
        return q_img, cam, torch.stack(cf_imgs), scalars, float(row['correct'])


# ── Models ─────────────────────────────────────────────────────────────────────
# Every model below produces one logit for one "instance" — a single image, or
# a (query, CF) pair, etc. When a sample carries K counterfactuals, the model
# is simply called K times (once per CF) and the K resulting logits are
# combined into one prediction by the ensembling helpers further down — the
# models themselves have no notion of K.
class _ImageEncoder(nn.Module):
    """DenseNet121 backbone → 1024-dim pooled feature vector."""
    def __init__(self):
        super().__init__()
        backbone      = models.densenet121(weights='IMAGENET1K_V1')
        self.features = backbone.features
        self.relu     = nn.ReLU(inplace=True)
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return torch.flatten(self.pool(self.relu(self.features(x))), 1)


class SingleEncoderCNN(nn.Module):
    """One image → one encoder → classifier."""
    def __init__(self):
        super().__init__()
        self.enc        = _ImageEncoder()
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        return self.classifier(self.enc(x))


class DualEncoderCNN(nn.Module):
    """Two images (e.g. query + CF), each with its own encoder, concatenated
    into a classifier."""
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.enc_a      = _ImageEncoder()
        self.enc_b      = _ImageEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, a, b):
        emb = torch.cat([self.enc_a(a), self.enc_b(b)], dim=1)
        return self.classifier(emb)


class TripleEncoderCNN(nn.Module):
    """Three images (e.g. query, CAM, CF), each with its own encoder,
    concatenated into a classifier."""
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.enc_a      = _ImageEncoder()
        self.enc_b      = _ImageEncoder()
        self.enc_c      = _ImageEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, a, b, c):
        emb = torch.cat([self.enc_a(a), self.enc_b(b), self.enc_c(c)], dim=1)
        return self.classifier(emb)


class TripleEncoderScalarCNN(nn.Module):
    """TripleEncoderCNN whose head also sees the C0 prediction scalars: the
    (B, 5) vector is embedded to 32 dims and concatenated with the three
    1024-dim encoder embeddings. The head keeps the same shape as the base
    config (only its input widens by 32) so the comparison isolates the
    scalar contribution."""
    def __init__(self, hidden_dim=256, dropout=0.3, scalar_dim=5, scalar_hidden=32):
        super().__init__()
        self.enc_a        = _ImageEncoder()
        self.enc_b        = _ImageEncoder()
        self.enc_c        = _ImageEncoder()
        self.scalar_embed = nn.Sequential(nn.Linear(scalar_dim, scalar_hidden),
                                          nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 3 + scalar_hidden, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, a, b, c, s):
        emb = torch.cat([self.enc_a(a), self.enc_b(b), self.enc_c(c),
                         self.scalar_embed(s)], dim=1)
        return self.classifier(emb)


def build_single_encoder():
    return SingleEncoderCNN()


# ── CF ensembling ──────────────────────────────────────────────────────────────
# A sample's inputs are either a single image per stream (B, 3, H, W) or K
# counterfactual images per stream (B, K, 3, H, W). To ensemble over K, every
# stream is broadcast to K (if it isn't already), K is folded into the batch
# dimension, the model runs once per (sample, CF) pair to get K logits per
# sample, and the K *probabilities* (not logits) are averaged into one final
# prediction — a genuine ensemble over the K counterfactuals.
def _stack_cf_inputs(*streams):
    """Broadcast a mix of (B,3,H,W) and (B,K,3,H,W) streams to a common K,
    then flatten K into the batch dim. Returns (flattened_streams, B, K)."""
    k = max((s.shape[1] for s in streams if s.dim() == 5), default=1)
    flat = []
    for s in streams:
        if s.dim() == 4:                        # (B,3,H,W) -> repeat K times
            s = s.unsqueeze(1).expand(-1, k, *s.shape[1:])
        b = s.shape[0]
        flat.append(s.reshape(b * k, *s.shape[2:]))
    return flat, b, k


def _ensemble_probs(logits, b, k):
    """K logits per sample -> one ensembled probability per sample, by
    averaging sigmoid(logit) over the K counterfactuals. Always computed in
    fp32: BCELoss on fp16 probabilities can under/overflow near 0 or 1, and
    autocast rejects it outright, so callers must invoke this outside the
    autocast region even though the encoder forward pass runs inside it."""
    return torch.sigmoid(logits.float()).view(b, k).mean(dim=1)


def make_weighted_bce(pos_weight):
    """BCELoss on probabilities with the positive class upweighted by
    `pos_weight` (the same rebalancing BCEWithLogitsLoss(pos_weight=...) does,
    reimplemented here because training operates on ensembled probabilities
    rather than a single logit)."""
    base = nn.BCELoss(reduction='none')

    def loss_fn(probs, labels):
        # Datasets emit labels as Python floats, which the default collate
        # turns into float64 — BCELoss requires input and target dtypes to
        # match, so bring labels down to the fp32 the probs are in.
        labels  = labels.float()
        weights = 1.0 + labels * (pos_weight - 1.0)
        return (base(probs, labels) * weights).mean()

    return loss_fn


# ── Train / eval — single input ────────────────────────────────────────────────
def _train_single(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total = 0.0
    for imgs, labels in loader:
        (imgs,), b, k = _stack_cf_inputs(imgs.to(device))
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs).squeeze(1)
        probs = _ensemble_probs(logits, b, k)
        loss  = criterion(probs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


def _eval_single(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad(), autocast():
        for imgs, lbls in loader:
            (imgs,), b, k = _stack_cf_inputs(imgs.to(device))
            logits = model(imgs).squeeze(1).float()
            probs.extend(_ensemble_probs(logits, b, k).cpu().numpy())
            labels.extend(lbls.numpy())
    return roc_auc_score(labels, probs), probs, labels


# ── Train / eval — dual input ──────────────────────────────────────────────────
def _train_dual(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total = 0.0
    for a_imgs, b_imgs, labels in loader:
        (a_imgs, b_imgs), b, k = _stack_cf_inputs(a_imgs.to(device), b_imgs.to(device))
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(a_imgs, b_imgs).squeeze(1)
        probs = _ensemble_probs(logits, b, k)
        loss  = criterion(probs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


def _eval_dual(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad(), autocast():
        for a_imgs, b_imgs, lbls in loader:
            (a_imgs, b_imgs), b, k = _stack_cf_inputs(a_imgs.to(device), b_imgs.to(device))
            logits = model(a_imgs, b_imgs).squeeze(1).float()
            probs.extend(_ensemble_probs(logits, b, k).cpu().numpy())
            labels.extend(lbls.numpy())
    return roc_auc_score(labels, probs), probs, labels


# ── Train / eval — triple input ────────────────────────────────────────────────
def _train_triple(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total = 0.0
    for a_imgs, b_imgs, c_imgs, labels in loader:
        streams = (a_imgs.to(device), b_imgs.to(device), c_imgs.to(device))
        (a_imgs, b_imgs, c_imgs), b, k = _stack_cf_inputs(*streams)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(a_imgs, b_imgs, c_imgs).squeeze(1)
        probs = _ensemble_probs(logits, b, k)
        loss  = criterion(probs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


def _eval_triple(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad(), autocast():
        for a_imgs, b_imgs, c_imgs, lbls in loader:
            streams = (a_imgs.to(device), b_imgs.to(device), c_imgs.to(device))
            (a_imgs, b_imgs, c_imgs), b, k = _stack_cf_inputs(*streams)
            logits = model(a_imgs, b_imgs, c_imgs).squeeze(1).float()
            probs.extend(_ensemble_probs(logits, b, k).cpu().numpy())
            labels.extend(lbls.numpy())
    return roc_auc_score(labels, probs), probs, labels


# ── Train / eval — triple input + scalars ──────────────────────────────────────
# The (B, K, 5) scalar tensor flattens with the same sample-major, CF-minor
# ordering _stack_cf_inputs applies to the image streams, so a plain reshape
# keeps each CF's scalars aligned with its images. Scalars are NOT routed
# through _stack_cf_inputs — its dim checks are image-specific.
def _train_triple_scalars(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total = 0.0
    for a_imgs, b_imgs, c_imgs, scal, labels in loader:
        streams = (a_imgs.to(device), b_imgs.to(device), c_imgs.to(device))
        (a_imgs, b_imgs, c_imgs), b, k = _stack_cf_inputs(*streams)
        scal   = scal.to(device).reshape(b * k, -1)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(a_imgs, b_imgs, c_imgs, scal).squeeze(1)
        probs = _ensemble_probs(logits, b, k)
        loss  = criterion(probs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


def _eval_triple_scalars(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad(), autocast():
        for a_imgs, b_imgs, c_imgs, scal, lbls in loader:
            streams = (a_imgs.to(device), b_imgs.to(device), c_imgs.to(device))
            (a_imgs, b_imgs, c_imgs), b, k = _stack_cf_inputs(*streams)
            scal   = scal.to(device).reshape(b * k, -1)
            logits = model(a_imgs, b_imgs, c_imgs, scal).squeeze(1).float()
            probs.extend(_ensemble_probs(logits, b, k).cpu().numpy())
            labels.extend(lbls.numpy())
    return roc_auc_score(labels, probs), probs, labels


# ── Embedding extraction ───────────────────────────────────────────────────────
# These save one feature vector per sample for downstream analysis (separate
# from how the classifier ensembles predictions above). When a stream carries
# K counterfactuals, its per-CF embeddings are mean-pooled into one vector.
def _encode_pooled(encoder, x):
    """Encode `x` with `encoder`; if x is (B,K,3,H,W), mean-pool over K."""
    if x.dim() == 5:
        b, k = x.shape[:2]
        emb = encoder(x.reshape(b * k, *x.shape[2:]))
        return emb.view(b, k, -1).mean(dim=1)
    return encoder(x)


def _embed_single(model, loader, device):
    """Extract 1024-dim DenseNet features before the linear classifier."""
    model.eval()
    embs, labels = [], []
    with torch.no_grad(), autocast():
        for imgs, lbls in loader:
            emb = _encode_pooled(model.enc, imgs.to(device)).float()
            embs.append(emb.cpu().numpy())
            labels.extend(lbls.numpy())
    return np.vstack(embs), np.array(labels)


def _embed_dual(model, loader, device):
    """Extract concatenated encoder embeddings (2048-dim) before the MLP head."""
    model.eval()
    embs, labels = [], []
    with torch.no_grad(), autocast():
        for a_imgs, b_imgs, lbls in loader:
            emb = torch.cat([
                _encode_pooled(model.enc_a, a_imgs.to(device)),
                _encode_pooled(model.enc_b, b_imgs.to(device)),
            ], dim=1).float()
            embs.append(emb.cpu().numpy())
            labels.extend(lbls.numpy())
    return np.vstack(embs), np.array(labels)


def _embed_triple(model, loader, device):
    """Extract concatenated encoder embeddings (3072-dim) before the MLP head."""
    model.eval()
    embs, labels = [], []
    with torch.no_grad(), autocast():
        for a_imgs, b_imgs, c_imgs, lbls in loader:
            emb = torch.cat([
                _encode_pooled(model.enc_a, a_imgs.to(device)),
                _encode_pooled(model.enc_b, b_imgs.to(device)),
                _encode_pooled(model.enc_c, c_imgs.to(device)),
            ], dim=1).float()
            embs.append(emb.cpu().numpy())
            labels.extend(lbls.numpy())
    return np.vstack(embs), np.array(labels)


# ── CV runner ──────────────────────────────────────────────────────────────────
def run_cv(name, fold_dfs, dataset_cls, make_model, output_dir, mode='single'):
    """
    mode: 'single' | 'dual' | 'triple' | 'triple_scalars'
      single         — dataset yields (img, label)
      dual           — dataset yields (img_a, img_b, label)
      triple         — dataset yields (img_a, img_b, img_c, label)
      triple_scalars — dataset yields (img_a, img_b, img_c, scalars, label)
    """
    os.makedirs(output_dir, exist_ok=True)
    train_fn = {'single': _train_single, 'dual': _train_dual, 'triple': _train_triple,
                'triple_scalars': _train_triple_scalars}[mode]
    eval_fn  = {'single': _eval_single,  'dual': _eval_dual,  'triple': _eval_triple,
                'triple_scalars': _eval_triple_scalars}[mode]

    fold_aucs, cv_results = [], []

    for test_fold in range(N_FOLDS):
        print(f"\n{'─'*60}")
        print(f"[{name}]  FOLD {test_fold + 1}/{N_FOLDS}")
        print(f"{'─'*60}")

        test_df  = fold_dfs[test_fold]
        train_df = pd.concat([fold_dfs[i] for i in range(N_FOLDS) if i != test_fold],
                             ignore_index=True)

        # Carve a validation split out of train for checkpoint selection, so the
        # test fold is only ever touched once (after training) — mirrors the
        # early-fusion pipeline so both are compared under the same protocol.
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
        train_sub_idx, val_idx = next(gss.split(train_df, groups=train_df['patient_id']))
        train_sub_df = train_df.iloc[train_sub_idx].reset_index(drop=True)
        val_df       = train_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"  Train: {len(train_sub_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

        train_loader = DataLoader(dataset_cls(train_sub_df, DATA_DIR, transform),
                                  batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, persistent_workers=True, pin_memory=True)
        val_loader   = DataLoader(dataset_cls(val_df,   DATA_DIR, transform),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
        test_loader  = DataLoader(dataset_cls(test_df,  DATA_DIR, transform),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

        model      = make_model().to(DEVICE)
        pos_weight = (1 - train_sub_df['correct'].mean()) / train_sub_df['correct'].mean()
        criterion  = make_weighted_bce(pos_weight)
        optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
        scaler     = GradScaler()

        best_auc = 0.0
        for epoch in range(N_EPOCHS):
            train_loss    = train_fn(model, train_loader, optimizer, criterion, DEVICE, scaler)
            val_auc, _, _ = eval_fn(model, val_loader, DEVICE)
            print(f"  Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(),
                           os.path.join(output_dir, f'fold_{test_fold}_best.pt'))

        # Evaluate the best checkpoint, not the last epoch's weights.
        best_ckpt = os.path.join(output_dir, f'fold_{test_fold}_best.pt')
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

        fold_auc, fold_probs, fold_labels = eval_fn(model, test_loader, DEVICE)
        fpr, tpr, _ = roc_curve(fold_labels, fold_probs)

        np.save(os.path.join(output_dir, f'fold_{test_fold}_fpr.npy'),    fpr)
        np.save(os.path.join(output_dir, f'fold_{test_fold}_tpr.npy'),    tpr)
        np.save(os.path.join(output_dir, f'fold_{test_fold}_probs.npy'),  np.asarray(fold_probs))
        np.save(os.path.join(output_dir, f'fold_{test_fold}_labels.npy'), np.asarray(fold_labels))

        pred_df = test_df.copy()
        pred_df['prob_pred'] = fold_probs
        pred_df.to_csv(os.path.join(output_dir, f'fold_{test_fold}_predictions.csv'), index=False)

        fold_aucs.append(fold_auc)
        cv_results.append({
            'fold':     test_fold,
            'auc':      float(fold_auc),
            'best_auc': float(best_auc),
            'fpr':      fpr.tolist(),
            'tpr':      tpr.tolist(),
            'y_prob':   list(map(float, fold_probs)),
            'y_true':   list(map(int,   fold_labels)),
        })
        print(f"  Best Val AUC: {best_auc:.4f}  |  Test AUC: {fold_auc:.4f}")

    pd.DataFrame([{'fold': r['fold'], 'auc': r['auc'], 'best_auc': r['best_auc']}
                  for r in cv_results]).to_csv(os.path.join(output_dir, 'cv_summary.csv'), index=False)
    with open(os.path.join(output_dir, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    print(f"\n[{name}] Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"[{name}] Saved to {output_dir}")
    return fold_aucs


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    fold_dfs = build_fold_dfs()
    
    
    # results/C2_cnn_diffusion/{disease}/late/{config}/cf_{K}/ — "late"
    # distinguishes this script's outputs from
    # c2_cnn_diffusion_early_fusion.py's under the same RES_BASE; config
    # names match 1:1 across both scripts.
    out = lambda name: os.path.join(RES_BASE, DISEASE, 'late', name, f'cf_{CF_COUNT}')

    # Registry of all five configurations. Each entry is the kwarg set passed
    # straight to run_cv(). Only the keys listed in --models are executed.
    MODEL_REGISTRY = {
        'xi': dict(
            name        = 'CNN Xi',
            dataset_cls = QueryDataset,
            make_model  = build_single_encoder,
            output_dir  = out('xi'),
            mode        = 'single',
        ),
        'cf': dict(
            name        = 'CNN CF',
            dataset_cls = CFDataset,
            make_model  = build_single_encoder,
            output_dir  = out('cf'),
            mode        = 'single',
        ),
        'xi_cf': dict(
            name        = 'CNN Xi + CF',
            dataset_cls = QueryCFDataset,
            make_model  = DualEncoderCNN,
            output_dir  = out('xi_cf'),
            mode        = 'dual',
        ),
        'xi_saliency': dict(
            name        = 'CNN Xi + Saliency',
            dataset_cls = QueryDatasetWithSaliency,
            make_model  = DualEncoderCNN,
            output_dir  = out('xi_saliency'),
            mode        = 'dual',
        ),
        'xi_saliency_cf': dict(
            name        = 'CNN Xi + Saliency + CF',
            dataset_cls = QueryCFSaliencyDataset,
            make_model  = TripleEncoderCNN,
            output_dir  = out('xi_saliency_cf'),
            mode        = 'triple',
        ),
        'xi_saliency_cf_scalars': dict(
            name        = 'CNN Xi + Saliency + CF + Scalars',
            dataset_cls = QueryCFSaliencyScalarsDataset,
            make_model  = TripleEncoderScalarCNN,
            output_dir  = out('xi_saliency_cf_scalars'),
            mode        = 'triple_scalars',
        ),
    }

    all_aucs = {}
    for key in SELECTED:
        cfg = MODEL_REGISTRY[key]
        all_aucs[cfg['name']] = run_cv(fold_dfs=fold_dfs, **cfg)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, aucs in all_aucs.items():
        print(f"  {name:<28}  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
