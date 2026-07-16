"""
c2_cnn_early_fusion.py
======================
Early-fusion CNN baselines for C2 quality control under the "correct CF" rule.

Counterpart to c2_cnn_all.py: instead of encoding each input with its own
DenseNet and concatenating pooled embeddings (late fusion), all inputs are
channel-concatenated into a single multi-channel image fed to one DenseNet.
Because xi, its CF, and their Grad-CAMs are pixel-aligned, the very first
conv layer can compare the same region across all inputs.

To keep results directly comparable with the late-fusion baselines, every input
is loaded exactly as in c2_cnn_all.py: read as grayscale, replicated to 3
channels, and normalized with ImageNet RGB stats. The only difference between
the two scripts is the fusion point — here the inputs are channel-concatenated,
so each contributes 3 channels:

  1. Early Xi              — [xi]                          (3 channels)
  2. Early CF              — [cf]                           (3 channels)
  3. Early Xi + Saliency   — [xi, cam_xi]                  (6 channels)
  4. Early CF + Saliency   — [cf, cam_cf]                  (6 channels)
  5. Early Dual            — [xi, cf]                      (6 channels)
  6. Early Dual + Saliency — [xi, cf, cam_xi, cam_cf]      (12 channels)
  7. Early Dual + Saliency + Scalars — the dual_sal stack plus a per-CF vector
     of C0 prediction scalars [p_xi, H(p_xi), p_cf_k, H(p_cf_k), p_xi − p_cf_k]
     embedded and concatenated at the classifier head

The pretrained DenseNet conv0 (3-channel) is adapted to 3N channels by tiling
its RGB filters once per input, scaled by 1/N to preserve activation magnitude,
so each input enters through the same ImageNet weights as in the late-fusion
encoders; all other layers keep their ImageNet weights.

Counterfactuals: when a fold's `cf_paths` column carries K pipe-separated CF
paths, every CF-facing dataset builds K channel-stacks (xi and cam_xi repeated
per CF) of shape (K, 3N, 224, 224). The model is run once per counterfactual
to get K predictions, which are combined into one final prediction by
averaging their probabilities — an ensemble over the K counterfactuals,
mirroring c2_cnn_all.py and the tabular pipeline. Pass --cf_count to select
which fold set (cf_{K}) to train on.

Use --models to train only a subset (default: all six), e.g.:

    python c2_cnn_early_fusion.py --models dual dual_sal

Available keys: xi, xi_sal, cf, cf_sal, dual, dual_sal, dual_sal_scalars

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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_CHOICES = ['xi', 'xi_sal', 'cf', 'cf_sal', 'dual', 'dual_sal', 'dual_sal_scalars']

parser = argparse.ArgumentParser()
parser.add_argument('--disease', type=str, default='effusion')
parser.add_argument('--cf_count', type=int, default=1)
parser.add_argument(
    '--models', type=str, nargs='+', default=MODEL_CHOICES, choices=MODEL_CHOICES,
    help=(
        "Which early-fusion configurations to train. Choose any subset of: "
        f"{', '.join(MODEL_CHOICES)}. Defaults to all of them."
    ),
)
parser.add_argument('--val_frac', type=float, default=0.15,
                    help="Fraction of each fold's train split held out for validation "
                         "(early stopping / checkpoint selection).")
parser.add_argument('--seed', type=int, default=42,
                    help="Seed for the train/val split (fixed for reproducibility).")
parser.add_argument('--batch_size', type=int, default=32,
                    help="Training/eval batch size. Lower for memory-heavy configs "
                         "(high cf_count) to avoid CUDA OOM.")
args = parser.parse_args()

DISEASE      = args.disease
CF_COUNT     = args.cf_count
SELECTED     = args.models
VAL_FRAC     = args.val_frac
SEED         = args.seed
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
DATA_DIR   = DATA_ROOT
CV_DIR     = os.path.join(RESULTS_DIR, f"C2_custom_corrected/{DISEASE}/cv_results_correct_cf/cf_{CF_COUNT}")
C2_DATA_CSV = os.path.join(RESULTS_DIR, f"C2_custom/{DISEASE}/c2_data.csv")
RES_BASE   = os.path.join(RESULTS_DIR, "C2_cnn")
CAM_BASE   = os.path.join(RESULTS_DIR, f"C0_custom/{DISEASE}/gradcam")
N_FOLDS    = 5
N_EPOCHS   = 10
BATCH_SIZE = args.batch_size
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f"Disease: {DISEASE} | CF count: {CF_COUNT}")
print(f"Models to train: {SELECTED}")
print("Using device:", DEVICE)

# ── Transforms ─────────────────────────────────────────────────────────────────
# Identical to the late-fusion pipeline in c2_cnn_all.py: every input — image or
# CAM — is replicated to 3 channels and normalized with ImageNet RGB stats, so
# the only difference between the two scripts is where the streams are fused.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_cam_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# ── Input helpers ──────────────────────────────────────────────────────────────
def _load_img(img_path, data_dir):
    """Load an image as a (3,224,224) normalized tensor (grayscale replicated)."""
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
    """cf_paths holds K pipe-separated paths (K = CF_COUNT for this fold set)."""
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


def _nearest_cf_only(df):
    """Train-time view of a fold df: keep only the nearest (first) CF, so each
    sample contributes ONE stack per epoch. cf_paths lists are nearest-first
    (verified: cf_16 first entry == cf_1's single entry on every row), and
    cf_probs is aligned with cf_paths. K-CF ensembling happens at eval only,
    matching the tabular pipelines' protocol (train base on nearest CF,
    combine k scores at test)."""
    df = df.copy()
    df['cf_paths'] = df['cf_paths'].map(lambda v: _split_cf_paths(v)[0])
    if 'cf_probs' in df.columns:
        df['cf_probs'] = df['cf_probs'].map(lambda v: str(v).split('|')[0])
    return df


def _attach_scalar_cols(fold_dfs):
    """Add `query_prob` and pipe-separated per-CF `cf_probs` columns in place.
    The fold CSVs only store the mean CF probability, but every KNN CF is a
    real training image whose C0 probability lives in c2_data.csv (column
    `prob`), so per-CF values are recovered by path lookup."""
    c2 = pd.read_csv(C2_DATA_CSV, usecols=['path', 'prob'])
    lookup = dict(zip(c2['path'], c2['prob']))
    for df in fold_dfs:
        df['query_prob'] = df[f'{DISEASE}_prob']
        df['cf_probs'] = df['cf_paths'].map(
            lambda v: '|'.join(str(lookup[p]) for p in _split_cf_paths(v)))


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


class XiSalDataset(Dataset):
    """[xi, cam_xi] → (6, 224, 224)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xi_path = str(row['path'])
        stack = torch.cat([_load_img(xi_path, self.data_dir), _load_cam(xi_path)])
        return stack, float(row['correct'])


class CFSalDataset(Dataset):
    """[cf_k, cam_cf_k] per CF → (K, 6, 224, 224)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        stacks = [
            torch.cat([_load_img(cf_path, self.data_dir), _load_cam(cf_path)])
            for cf_path in _split_cf_paths(row['cf_paths'])
        ]
        return torch.stack(stacks), float(row['correct'])


class DualDataset(Dataset):
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


class DualSalDataset(Dataset):
    """[xi, cf_k, cam_xi, cam_cf_k] per CF → (K, 12, 224, 224)."""
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
            torch.cat([xi, _load_img(cf_path, self.data_dir),
                       xi_cam, _load_cam(cf_path)])
            for cf_path in _split_cf_paths(row['cf_paths'])
        ]
        return torch.stack(stacks), float(row['correct'])


class DualSalScalarDataset(Dataset):
    """[xi, cf_k, cam_xi, cam_cf_k] per CF → (K, 12, 224, 224), plus the C0
    prediction scalars → (K, 5)."""
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
            torch.cat([xi, _load_img(cf_path, self.data_dir),
                       xi_cam, _load_cam(cf_path)])
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
# Every dataset yields K channel-stacks per sample as (B, K, C, H, W). We fold
# K into the batch dimension, run the model once per (sample, CF) pair to get
# K logits per sample, then average the K *probabilities* (not the logits)
# into one final prediction — an ensemble over the K counterfactuals.
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

        # Carve a validation split out of train for early stopping / checkpoint
        # selection, so the test fold is only ever touched once (after training).
        train_sub_df, val_df = train_test_split(
            train_df, test_size=VAL_FRAC, stratify=train_df['correct'],
            random_state=SEED,
        )
        train_sub_df = train_sub_df.reset_index(drop=True)
        val_df       = val_df.reset_index(drop=True)

        # Training uses only the nearest CF per sample; val/test keep all K so
        # checkpoint selection sees the same K-ensemble protocol as the final
        # test evaluation.
        if 'cf_paths' in train_sub_df.columns:
            train_sub_df = _nearest_cf_only(train_sub_df)
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

    fold_dfs = [
        pd.read_csv(os.path.join(CV_DIR, f'fold_{i}_predictions.csv'))
        for i in range(N_FOLDS)
    ]

    # Scalar configs need per-CF C0 probabilities, recovered by path lookup
    # from c2_data.csv (a wide file — only read it when actually needed).
    if any(key.endswith('_scalars') for key in SELECTED):
        _attach_scalar_cols(fold_dfs)

    # results/C2_cnn/{disease}/early/{config}/cf_{K}/ — "early" distinguishes
    # this script's outputs from c2_cnn_all.py's under the same RES_BASE;
    # config names match 1:1 across both scripts.
    out = lambda name: os.path.join(RES_BASE, DISEASE, 'early', name, f'cf_{CF_COUNT}')

    # Registry of all six early-fusion configurations. Each entry is the kwarg
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
        'xi_sal': dict(
            name        = 'Early Fusion Xi + Saliency',
            dataset_cls = XiSalDataset,
            make_model  = lambda: EarlyFusionCNN(in_channels=6),
            output_dir  = out('xi_sal'),
        ),
        'cf_sal': dict(
            name        = 'Early Fusion CF + Saliency',
            dataset_cls = CFSalDataset,
            make_model  = lambda: EarlyFusionCNN(in_channels=6),
            output_dir  = out('cf_sal'),
        ),
        'dual': dict(
            name        = 'Early Fusion Xi + CF',
            dataset_cls = DualDataset,
            make_model  = lambda: EarlyFusionCNN(in_channels=6),
            output_dir  = out('dual'),
        ),
        'dual_sal': dict(
            name        = 'Early Fusion Xi + CF + Saliency',
            dataset_cls = DualSalDataset,
            make_model  = lambda: EarlyFusionCNN(in_channels=12),
            output_dir  = out('dual_sal'),
        ),
        'dual_sal_scalars': dict(
            name        = 'Early Fusion Xi + CF + Saliency + Scalars',
            dataset_cls = DualSalScalarDataset,
            make_model  = lambda: EarlyFusionScalarCNN(in_channels=12),
            output_dir  = out('dual_sal_scalars'),
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
