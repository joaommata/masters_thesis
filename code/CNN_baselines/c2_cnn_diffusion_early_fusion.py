"""
c2_cnn_diffusion_early_fusion.py
================================
Early-fusion CNN baselines for C2 quality control using real diffusion-generated
counterfactuals (see cf_generate_diffusion_counterfactuals.py).

Counterpart to c2_cnn_diffusion.py: instead of encoding each input with its own
DenseNet and concatenating pooled embeddings (late fusion), all inputs are
channel-concatenated into a single multi-channel image fed to one DenseNet, so
the very first conv layer can compare the same region across all inputs.

To keep results directly comparable with the late-fusion baselines, every input
is loaded exactly as in c2_cnn_diffusion.py: read as grayscale, replicated to 3
channels, and normalized with ImageNet RGB stats. The only difference between
the two scripts is the fusion point — here the inputs are channel-concatenated,
so each contributes 3 channels:

  1. Xi                — [xi]                     (3 channels)
  2. CF                 — [cf]                     (3 channels)
  3. Xi + CF            — [xi, cf]                 (6 channels)
  4. Xi + Saliency      — [xi, cam_xi]             (6 channels)
  5. Xi + Saliency + CF — [xi, cam_xi, cf]         (9 channels)
  6. Xi + Saliency + CF + Scalars — the 9-channel stack plus a per-CF vector
     of C0 prediction scalars [p_xi, H(p_xi), p_cf_k, H(p_cf_k), p_xi − p_cf_k]
     embedded and concatenated at the classifier head

Grad-CAMs exist only for xi (there is no CF-side CAM for diffusion CFs), so
every saliency config uses cam_xi.

The pretrained DenseNet conv0 (3-channel) is adapted to 3N channels by tiling
its RGB filters once per input, scaled by 1/N to preserve activation magnitude,
so each input enters through the same ImageNet weights as in the late-fusion
encoders; all other layers keep their ImageNet weights.

Counterfactuals: every query has 3 diffusion counterfactuals available, from
the manifest cf_manifest_10_0.25_n3.csv (t_start=10, guidance_weight=0.25). The
manifest is long-format (3 rows per query, keyed by cf_idx); it is pivoted here
into the pipe-separated `cf_paths` convention the rest of the CNN family uses.
Pass --cf_count 3 (default) to use all of them, or --cf_count 1 to keep only
the single most confident CF per query — both read the same manifest, so
results are directly comparable. CF-facing datasets build K channel-stacks of
shape (K, 3N, 224, 224), repeating the xi-side channels per CF. The model is
run once per counterfactual to get K predictions, which are combined into one
final prediction by averaging their probabilities — an ensemble over the K
counterfactuals, mirroring c2_cnn_diffusion.py and the rest of the CNN family.

Use --models to train only a subset (default: all five), e.g.:

    python c2_cnn_diffusion_early_fusion.py --models xi_cf xi_saliency_cf

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
from torchvision import models, transforms

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
        "Which early-fusion configurations to train. Choose any subset of: "
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
                    help="Training/eval batch size. Early fusion pushes batch*K stacks "
                         "through one DenseNet, so lower this if you hit CUDA OOM.")
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
# Identical to the late-fusion pipeline in c2_cnn_diffusion.py: every input —
# image or CAM — is replicated to 3 channels and normalized with ImageNet RGB
# stats, so the only difference between the two scripts is where the streams
# are fused. Resize((224,224)) is an anisotropic squash, and deliberately so:
# it is the transform the CF generator applied, and it is what keeps xi
# pixel-aligned with its counterfactual. Do not replace it with a crop.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_cam_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# ── Input helpers ──────────────────────────────────────────────────────────────
def _load_img(img_path, data_dir):
    """Load an image as a (3,224,224) normalized tensor (grayscale replicated).

    Manifest `cf_path` values are absolute, so os.path.join discards data_dir for
    counterfactuals; query `path` values are relative to it.
    """
    img = Image.open(os.path.join(data_dir, img_path)).convert('RGB')
    return transform(img)


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
    df = pd.read_csv(C2_DATA_CSV, usecols=['path', 'prob', 'correct'])
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

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_dfs = [None] * N_FOLDS
    for fold_idx, (_, test_idx) in enumerate(skf.split(df, df['correct'])):
        fold_dfs[fold_idx] = df.iloc[test_idx].reset_index(drop=True)
    return fold_dfs


# ── Datasets ───────────────────────────────────────────────────────────────────
# Every dataset yields (stack, label). Each input contributes 3 channels (its
# grayscale value replicated), exactly as in the late-fusion pipeline, so an
# N-input configuration produces 3N channels. Every dataset yields K
# channel-stacks as (K, 3N, 224, 224) — one per counterfactual, with the
# xi-side channels repeated — so the model can be run once per CF and the K
# resulting predictions ensembled (see the "CF ensembling" section below).
class XiDataset(Dataset):
    """[xi] → (3, 224, 224). No CF, so K=1 (broadcast at ensembling time)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        stack = _load_img(str(row['path']), self.data_dir)
        return stack, float(row['correct'])


class CFDataset(Dataset):
    """[cf_k] per CF → (K, 3, 224, 224)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        stacks = [_load_img(cf_path, self.data_dir)
                  for cf_path in _split_cf_paths(row['cf_paths'])]
        return torch.stack(stacks), float(row['correct'])


class XiCFDataset(Dataset):
    """[xi, cf_k] per CF → (K, 6, 224, 224)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xi = _load_img(str(row['path']), self.data_dir)
        stacks = [
            torch.cat([xi, _load_img(cf_path, self.data_dir)])
            for cf_path in _split_cf_paths(row['cf_paths'])
        ]
        return torch.stack(stacks), float(row['correct'])


class XiSalDataset(Dataset):
    """[xi, cam_xi] → (6, 224, 224). No CF, so K=1 (broadcast at ensembling time)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xi_path = str(row['path'])
        stack = torch.cat([_load_img(xi_path, self.data_dir), _load_cam(xi_path)])
        return stack, float(row['correct'])


class XiSalCFDataset(Dataset):
    """[xi, cam_xi, cf_k] per CF → (K, 9, 224, 224)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xi_path = str(row['path'])
        xi     = _load_img(xi_path, self.data_dir)
        xi_cam = _load_cam(xi_path)
        stacks = [
            torch.cat([xi, xi_cam, _load_img(cf_path, self.data_dir)])
            for cf_path in _split_cf_paths(row['cf_paths'])
        ]
        return torch.stack(stacks), float(row['correct'])


class XiSalCFScalarDataset(Dataset):
    """[xi, cam_xi, cf_k] per CF → (K, 9, 224, 224), plus the C0 prediction
    scalars → (K, 5)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xi_path = str(row['path'])
        xi     = _load_img(xi_path, self.data_dir)
        xi_cam = _load_cam(xi_path)
        stacks = [
            torch.cat([xi, xi_cam, _load_img(cf_path, self.data_dir)])
            for cf_path in _split_cf_paths(row['cf_paths'])
        ]
        scalars = _make_scalars(float(row['query_prob']),
                                _split_cf_probs(row['cf_probs']))
        return torch.stack(stacks), scalars, float(row['correct'])


# ── Model ──────────────────────────────────────────────────────────────────────
def _inflate_conv0(conv0, in_channels):
    """Adapt the pretrained 3-channel conv0 to `in_channels` for a stack of
    N = in_channels // 3 inputs, each a 3-channel (grayscale-replicated) image.
    The pretrained RGB filters are tiled once per input and scaled by 1/N so
    activation magnitudes match — every input enters through the same ImageNet
    weights it would in the late-fusion encoders."""
    assert in_channels % 3 == 0, "in_channels must be a multiple of 3 (3 per input)"
    n_inputs = in_channels // 3
    new_conv = nn.Conv2d(in_channels, conv0.out_channels,
                         kernel_size=conv0.kernel_size, stride=conv0.stride,
                         padding=conv0.padding, bias=False)
    with torch.no_grad():
        w = conv0.weight.repeat(1, n_inputs, 1, 1) / n_inputs         # (64,3N,7,7)
        new_conv.weight.copy_(w)
    return new_conv


class EarlyFusionCNN(nn.Module):
    """Single DenseNet over one channel-concatenated stack (B, C, H, W) ->
    one logit. Has no notion of K counterfactuals — when a sample carries K
    of them, the training/eval loops below call this model once per CF and
    ensemble the K resulting predictions."""
    def __init__(self, in_channels):
        super().__init__()
        backbone = models.densenet121(weights='IMAGENET1K_V1')
        backbone.features.conv0 = _inflate_conv0(backbone.features.conv0, in_channels)
        self.features   = backbone.features
        self.relu       = nn.ReLU(inplace=True)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, 1)

    def encode(self, x):
        return torch.flatten(self.pool(self.relu(self.features(x))), 1)

    def forward(self, x):
        return self.classifier(self.encode(x))


class EarlyFusionScalarCNN(EarlyFusionCNN):
    """EarlyFusionCNN whose head also sees the C0 prediction scalars: the
    (B, 5) vector is embedded to 32 dims and concatenated with the 1024-dim
    pooled image features. The head stays a single linear layer so the
    comparison with the base config isolates the scalar contribution."""
    def __init__(self, in_channels, scalar_dim=5, scalar_hidden=32):
        super().__init__(in_channels)
        self.scalar_embed = nn.Sequential(nn.Linear(scalar_dim, scalar_hidden),
                                          nn.ReLU(inplace=True))
        self.classifier = nn.Linear(1024 + scalar_hidden, 1)

    def forward(self, x, s):
        return self.classifier(
            torch.cat([self.encode(x), self.scalar_embed(s)], dim=1))


# ── CF ensembling ──────────────────────────────────────────────────────────────
# A sample's input is either a single stack (B, C, H, W), for configs with no
# CF, or K stacks (B, K, C, H, W) — one per counterfactual. We fold K into the
# batch dimension, run the model once per (sample, CF) pair to get K logits per
# sample, then average the K *probabilities* (not the logits) into one final
# prediction — an ensemble over the K counterfactuals.
def _flatten_k(x):
    """(B, C, H, W) -> (B, C, H, W) with K=1, or (B, K, C, H, W) -> (B*K, C, H, W).
    Always returns (flattened, B, K)."""
    if x.dim() == 4:
        return x, x.shape[0], 1
    b, k = x.shape[:2]
    return x.reshape(b * k, *x.shape[2:]), b, k


def _ensemble_probs(logits, b, k):
    """K logits per sample -> one ensembled probability per sample, by
    averaging sigmoid(logit) over the K counterfactuals. Cast to fp32 before
    the sigmoid: BCELoss on fp16 probabilities can under/overflow near 0/1,
    and autocast rejects it outright, so callers must invoke this outside the
    autocast region."""
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


# ── Train / eval / embed ───────────────────────────────────────────────────────
def _train(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total = 0.0
    for imgs, labels in loader:
        imgs_flat, b, k = _flatten_k(imgs.to(device))
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs_flat).squeeze(1)
        probs = _ensemble_probs(logits, b, k)
        loss  = criterion(probs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


def _eval(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad(), autocast():
        for imgs, lbls in loader:
            imgs_flat, b, k = _flatten_k(imgs.to(device))
            logits = model(imgs_flat).squeeze(1).float()
            probs.extend(_ensemble_probs(logits, b, k).cpu().numpy())
            labels.extend(lbls.numpy())
    return roc_auc_score(labels, probs), probs, labels


# Scalar-aware counterparts for datasets that yield (stack, scalars, label).
# The (B, K, 5) scalar tensor flattens with the same sample-major, CF-minor
# ordering _flatten_k applies to the image stacks, so a plain reshape keeps
# each CF's scalars aligned with its channel stack.
def _train_scalar(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total = 0.0
    for imgs, scalars, labels in loader:
        imgs_flat, b, k = _flatten_k(imgs.to(device))
        scalars_flat = scalars.to(device).reshape(b * k, -1)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs_flat, scalars_flat).squeeze(1)
        probs = _ensemble_probs(logits, b, k)
        loss  = criterion(probs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


def _eval_scalar(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad(), autocast():
        for imgs, scalars, lbls in loader:
            imgs_flat, b, k = _flatten_k(imgs.to(device))
            scalars_flat = scalars.to(device).reshape(b * k, -1)
            logits = model(imgs_flat, scalars_flat).squeeze(1).float()
            probs.extend(_ensemble_probs(logits, b, k).cpu().numpy())
            labels.extend(lbls.numpy())
    return roc_auc_score(labels, probs), probs, labels


def _embed(model, loader, device):
    """Extract 1024-dim DenseNet features before the linear classifier,
    mean-pooled over K (a separate concern from how predictions are
    ensembled above — this just saves one representative feature vector
    per sample for downstream analysis)."""
    model.eval()
    embs, labels = [], []
    with torch.no_grad(), autocast():
        for imgs, lbls in loader:
            imgs_flat, b, k = _flatten_k(imgs.to(device))
            emb = model.encode(imgs_flat).float().view(b, k, -1).mean(dim=1)
            embs.append(emb.cpu().numpy())
            labels.extend(lbls.numpy())
    return np.vstack(embs), np.array(labels)


# ── CV runner ──────────────────────────────────────────────────────────────────
def run_cv(name, fold_dfs, dataset_cls, make_model, output_dir, with_scalars=False):
    os.makedirs(output_dir, exist_ok=True)
    train_fn = _train_scalar if with_scalars else _train
    eval_fn  = _eval_scalar  if with_scalars else _eval

    fold_aucs, cv_results = [], []

    for test_fold in range(N_FOLDS):
        print(f"\n{'─'*60}")
        print(f"[{name}]  FOLD {test_fold + 1}/{N_FOLDS}")
        print(f"{'─'*60}")

        test_df  = fold_dfs[test_fold]
        train_df = pd.concat([fold_dfs[i] for i in range(N_FOLDS) if i != test_fold],
                             ignore_index=True)

        # Carve a validation split out of train for checkpoint selection, so the
        # test fold is only ever touched once (after training).
        train_sub_df, val_df = train_test_split(
            train_df, test_size=VAL_FRAC, stratify=train_df['correct'],
            random_state=SEED,
        )
        train_sub_df = train_sub_df.reset_index(drop=True)
        val_df       = val_df.reset_index(drop=True)
        print(f"  Train: {len(train_sub_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

        train_loader = DataLoader(dataset_cls(train_sub_df, DATA_DIR),
                                  batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, persistent_workers=True, pin_memory=True)
        val_loader   = DataLoader(dataset_cls(val_df,   DATA_DIR),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
        test_loader  = DataLoader(dataset_cls(test_df,  DATA_DIR),
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

    # results/C2_cnn_diffusion/{disease}/early/{config}/cf_{K}/ — "early"
    # distinguishes this script's outputs from c2_cnn_diffusion.py's under the
    # same RES_BASE; config names match 1:1 across both scripts.
    out = lambda name: os.path.join(RES_BASE, DISEASE, 'early', name, f'cf_{CF_COUNT}')

    # Registry of all five early-fusion configurations. Each entry is the kwarg
    # set passed straight to run_cv(). Only the keys in --models are executed.
    MODEL_REGISTRY = {
        'xi': dict(
            name        = 'Early Fusion Xi',
            dataset_cls = XiDataset,
            make_model  = lambda: EarlyFusionCNN(in_channels=3),
            output_dir  = out('xi'),
        ),
        'cf': dict(
            name        = 'Early Fusion CF',
            dataset_cls = CFDataset,
            make_model  = lambda: EarlyFusionCNN(in_channels=3),
            output_dir  = out('cf'),
        ),
        'xi_cf': dict(
            name        = 'Early Fusion Xi + CF',
            dataset_cls = XiCFDataset,
            make_model  = lambda: EarlyFusionCNN(in_channels=6),
            output_dir  = out('xi_cf'),
        ),
        'xi_saliency': dict(
            name        = 'Early Fusion Xi + Saliency',
            dataset_cls = XiSalDataset,
            make_model  = lambda: EarlyFusionCNN(in_channels=6),
            output_dir  = out('xi_saliency'),
        ),
        'xi_saliency_cf': dict(
            name        = 'Early Fusion Xi + Saliency + CF',
            dataset_cls = XiSalCFDataset,
            make_model  = lambda: EarlyFusionCNN(in_channels=9),
            output_dir  = out('xi_saliency_cf'),
        ),
        'xi_saliency_cf_scalars': dict(
            name        = 'Early Fusion Xi + Saliency + CF + Scalars',
            dataset_cls = XiSalCFScalarDataset,
            make_model  = lambda: EarlyFusionScalarCNN(in_channels=9),
            output_dir  = out('xi_saliency_cf_scalars'),
            with_scalars = True,
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
        print(f"  {name:<32}  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
