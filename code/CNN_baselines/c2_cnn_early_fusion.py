"""
c2_cnn_early_fusion.py
======================
Early-fusion CNN baselines for C2 quality control under the "correct CF" rule.

Counterpart to c2_cnn_all.py: instead of encoding each input with its own
DenseNet and concatenating pooled embeddings (late fusion), all inputs are
channel-concatenated into a single multi-channel image fed to one DenseNet.
Because xi, its CF, and their Grad-CAMs are pixel-aligned, the very first
conv layer can compare the same region across all inputs.

All images are loaded as single-channel grayscale, so each input contributes
one channel:

  1. Xi                    — [xi]                          (1 channel)
  2. Early Xi + Saliency   — [xi, cam_xi]                  (2 channels)
  3. CF                    — [cf]                          (1 channel)
  4. Early CF + Saliency   — [cf, cam_cf]                  (2 channels)
  5. Early Dual            — [xi, cf]                      (2 channels)
  6. Early Dual + Saliency — [xi, cf, cam_xi, cam_cf]      (4 channels)

(The xi-only and cf-only configs have nothing to fuse; they are included as
single-channel grayscale baselines so all six models share this pipeline.)

The pretrained DenseNet conv0 (3-channel) is inflated to C channels by
averaging its RGB filters and tiling, scaled by 3/C to preserve activation
magnitude; all other layers keep their ImageNet weights.

Counterfactuals: when a fold's `cf_paths` column carries K pipe-separated CF
paths, every CF-facing dataset builds K channel-stacks (xi and cam_xi repeated
per CF) of shape (K, C, 224, 224). How those K stacks are combined is set by
--cf_agg, mirroring c2_cnn_all.py:

  mean       — score each stack separately (full forward pass per CF) and
               average the K predicted probabilities. Default. Note this
               averages probabilities, not logits: the classifier here is a
               single affine layer, so averaging logits would be algebraically
               identical to `embedding` below.
  mlp        — score each stack separately, then a small MLP over the K logits
               produces the final score. CFs are ordered by neighbour distance,
               so the MLP can weight nearer CFs more heavily.
  embedding  — mean-pool the K stack embeddings before the classifier, mirroring
               how the tabular pipeline averages diff vectors across the K
               nearest neighbours. One forward pass through the head.

The xi and xi_sal configs have no CF stack and are unaffected by --cf_agg.
Pass --cf_count to select which fold set (cf_{K}) to train on.

Use --models to train only a subset (default: all six), e.g.:

    python c2_cnn_early_fusion.py --models dual dual_sal

Available keys: xi, xi_sal, cf, cf_sal, dual, dual_sal

Results for each model are written to their own output directory: the
`embedding` runs land in cf_{K}/, the score-level runs in cf_{K}_{cf_agg}/.
"""

import argparse
import json
import os
from functools import partial

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
MODEL_CHOICES  = ['xi', 'xi_sal', 'cf', 'cf_sal', 'dual', 'dual_sal']
CF_AGG_CHOICES = ['mean', 'mlp', 'embedding']

parser = argparse.ArgumentParser()
parser.add_argument('--disease', type=str, default='effusion')
parser.add_argument('--cf_count', type=int, default=1)
parser.add_argument(
    '--cf_agg', type=str, default='mean', choices=CF_AGG_CHOICES,
    help="How to combine the K counterfactuals. 'mean'/'mlp' score each CF "
         "stack separately and aggregate the K logits; 'embedding' mean-pools "
         "the K stack embeddings before the classifier.",
)
parser.add_argument('--agg_hidden', type=int, default=16,
                    help="Hidden width of the score-aggregation MLP (--cf_agg mlp).")
parser.add_argument(
    '--models', type=str, nargs='+', default=MODEL_CHOICES, choices=MODEL_CHOICES,
    help=(
        "Which early-fusion configurations to train. Choose any subset of: "
        f"{', '.join(MODEL_CHOICES)}. Defaults to all of them."
    ),
)
parser.add_argument('--patience', type=int, default=3,
                    help="Early-stopping patience (epochs without val-AUC improvement).")
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
CF_AGG       = args.cf_agg
AGG_HIDDEN   = args.agg_hidden
SELECTED     = args.models
PATIENCE     = args.patience
VAL_FRAC     = args.val_frac
SEED         = args.seed
BASE_DIR   = "/zhome/d0/a/221493/thesis"
DATA_DIR   = os.path.join(BASE_DIR, "data")
CV_DIR     = os.path.join(BASE_DIR, f"results/C2_custom_corrected/{DISEASE}/cv_results_correct_cf/cf_{CF_COUNT}")
RES_BASE   = os.path.join(BASE_DIR, "results/C2_cnn")
CAM_BASE   = os.path.join(BASE_DIR, f"results/C0_custom/{DISEASE}/gradcam")
N_FOLDS    = 5
N_EPOCHS   = 10
BATCH_SIZE = args.batch_size
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Disease: {DISEASE} | CF count: {CF_COUNT} | CF aggregation: {CF_AGG}")
print(f"Models to train: {SELECTED}")
print("Using device:", DEVICE)

# ── Transforms ─────────────────────────────────────────────────────────────────
# Single-channel stats (ImageNet RGB stats averaged), applied per channel so
# every stream — image or CAM — enters the network on the same scale.
GRAY_MEAN, GRAY_STD = 0.449, 0.226

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([GRAY_MEAN], [GRAY_STD]),
])


# ── Input helpers ──────────────────────────────────────────────────────────────
def _load_gray(img_path, data_dir):
    """Load an image as a (1,224,224) normalized grayscale tensor."""
    img = Image.open(os.path.join(data_dir, img_path)).convert('L')
    return transform(img)


def _load_cam(img_path):
    """Load a pre-computed C0 Grad-CAM as a (1,224,224) normalized tensor."""
    fname = img_path.replace('/', '_') + '.npz'
    # CheXpert paths always contain /train/ regardless of model split; try both dirs.
    for split in ('val', 'train'):
        npz_path = os.path.join(CAM_BASE, split, fname)
        if os.path.exists(npz_path):
            break
    cam = np.load(npz_path)['cam']                                    # (7,7) float32
    cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)           # (1,1,7,7)
    cam_t = F.interpolate(cam_t, (224, 224), mode='bilinear', align_corners=False)
    return (cam_t.squeeze(0) - GRAY_MEAN) / GRAY_STD                  # (1,224,224)


def _split_cf_paths(value):
    """cf_paths holds K pipe-separated paths (K = CF_COUNT for this fold set)."""
    return [p.strip() for p in str(value).split('|')]


# ── Datasets ───────────────────────────────────────────────────────────────────
# Every dataset yields (stack, label). CF-facing datasets yield K channel-stacks
# as (K, C, 224, 224) — one per counterfactual, with the xi-side channels
# repeated — so the model can mean-pool embeddings over K.
class XiDataset(Dataset):
    """[xi] → (1, 224, 224)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return _load_gray(str(row['path']), self.data_dir), float(row['correct'])


class CFDataset(Dataset):
    """[cf_k] per CF → (K, 1, 224, 224)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        stacks = [
            _load_gray(cf_path, self.data_dir)
            for cf_path in _split_cf_paths(row['cf_paths'])
        ]
        return torch.stack(stacks), float(row['correct'])


class XiSalDataset(Dataset):
    """[xi, cam_xi] → (2, 224, 224)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xi_path = str(row['path'])
        stack = torch.cat([_load_gray(xi_path, self.data_dir), _load_cam(xi_path)])
        return stack, float(row['correct'])


class CFSalDataset(Dataset):
    """[cf_k, cam_cf_k] per CF → (K, 2, 224, 224)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        stacks = [
            torch.cat([_load_gray(cf_path, self.data_dir), _load_cam(cf_path)])
            for cf_path in _split_cf_paths(row['cf_paths'])
        ]
        return torch.stack(stacks), float(row['correct'])


class DualDataset(Dataset):
    """[xi, cf_k] per CF → (K, 2, 224, 224)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xi = _load_gray(str(row['path']), self.data_dir)
        stacks = [
            torch.cat([xi, _load_gray(cf_path, self.data_dir)])
            for cf_path in _split_cf_paths(row['cf_paths'])
        ]
        return torch.stack(stacks), float(row['correct'])


class DualSalDataset(Dataset):
    """[xi, cf_k, cam_xi, cam_cf_k] per CF → (K, 4, 224, 224)."""
    def __init__(self, df, data_dir):
        self.df, self.data_dir = df.reset_index(drop=True), data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xi_path = str(row['path'])
        xi     = _load_gray(xi_path, self.data_dir)
        xi_cam = _load_cam(xi_path)
        stacks = [
            torch.cat([xi, _load_gray(cf_path, self.data_dir),
                       xi_cam, _load_cam(cf_path)])
            for cf_path in _split_cf_paths(row['cf_paths'])
        ]
        return torch.stack(stacks), float(row['correct'])


# ── Model ──────────────────────────────────────────────────────────────────────
def _inflate_conv0(conv0, in_channels):
    """Adapt the pretrained 3-channel conv0 to `in_channels` by averaging its
    RGB filters and tiling, scaled by 3/C so activation magnitudes match."""
    new_conv = nn.Conv2d(in_channels, conv0.out_channels,
                         kernel_size=conv0.kernel_size, stride=conv0.stride,
                         padding=conv0.padding, bias=False)
    with torch.no_grad():
        w = conv0.weight.mean(dim=1, keepdim=True)                    # (64,1,7,7)
        new_conv.weight.copy_(w.repeat(1, in_channels, 1, 1) * (3.0 / in_channels))
    return new_conv


def _aggregate_scores(logits, cf_agg, score_mlp):
    """Collapse the K per-CF logits (B, K) into one logit per sample (B,).

    'mean' averages the per-CF *probabilities*, not the logits: the classifier
    is a single affine layer, so averaging logits is algebraically identical to
    mean-pooling the embeddings first and would silently reproduce
    cf_agg='embedding'. The mean probability is mapped back through logit() so
    callers can keep using BCEWithLogitsLoss and recover it with sigmoid().
    """
    if score_mlp is not None:
        return score_mlp(logits).squeeze(-1)
    if cf_agg == 'embedding':
        return logits.mean(dim=1)          # already pooled to K == 1
    # float32 keeps sigmoid/mean well-conditioned under autocast.
    p = torch.sigmoid(logits.float()).mean(dim=1).clamp(1e-6, 1 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


class EarlyFusionCNN(nn.Module):
    """Single DenseNet over channel-concatenated inputs. Accepts one stack
    (B, C, H, W) or K stacks (B, K, C, H, W) — one per counterfactual.

    `cf_agg` decides where the K collapses:

      mean/mlp  — the classifier scores each of the K stack embeddings, giving K
                  logits that are ensemble-averaged as probabilities, or fed to
                  a small MLP that learns to weight them (CFs arrive ordered by
                  neighbour distance).
      embedding — the K embeddings are mean-pooled first, so the classifier runs
                  once, matching how the tabular pipeline averages diff vectors.

    With K = 1 (or inputs that carry no CF at all) the three are equivalent up to
    the MLP's extra 1→hidden→1 mapping.
    """
    def __init__(self, in_channels, cf_agg='mean', k=1, agg_hidden=16):
        super().__init__()
        backbone = models.densenet121(weights='IMAGENET1K_V1')
        backbone.features.conv0 = _inflate_conv0(backbone.features.conv0, in_channels)
        self.features   = backbone.features
        self.relu       = nn.ReLU(inplace=True)
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, 1)
        self.cf_agg     = cf_agg
        self.score_mlp  = nn.Sequential(
            nn.Linear(k, agg_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(agg_hidden, 1),
        ) if cf_agg == 'mlp' else None

    def _encode_flat(self, x):
        return torch.flatten(self.pool(self.relu(self.features(x))), 1)

    def _encode_stacks(self, x):
        """Encode every stack, always returning (B, K, 1024). A single stack
        (B, C, H, W) yields K = 1; K stacks (B, K, C, H, W) yield K."""
        if x.dim() == 5:
            b, k = x.shape[:2]
            return self._encode_flat(x.reshape(b * k, *x.shape[2:])).view(b, k, -1)
        return self._encode_flat(x).unsqueeze(1)

    def forward(self, x):
        emb = self._encode_stacks(x)                      # (B, K, 1024)
        if self.cf_agg == 'embedding':
            emb = emb.mean(dim=1, keepdim=True)           # (B, 1, 1024)
        logits = self.classifier(emb).squeeze(-1)         # (B, K) — one per CF
        return _aggregate_scores(logits, self.cf_agg, self.score_mlp)

    def embed(self, x):
        """1024-dim DenseNet features before the classifier, mean-pooled over K."""
        return self._encode_stacks(x).mean(dim=1)


# ── Train / eval / embed ───────────────────────────────────────────────────────
def _train(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            loss = criterion(model(imgs), labels)
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
            logits = model(imgs.to(device)).float()
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(lbls.numpy())
    return roc_auc_score(labels, probs), probs, labels


def _embed(model, loader, device):
    model.eval()
    embs, labels = [], []
    with torch.no_grad(), autocast():
        for imgs, lbls in loader:
            embs.append(model.embed(imgs.to(device)).float().cpu().numpy())
            labels.extend(lbls.numpy())
    return np.vstack(embs), np.array(labels)


# ── CV runner ──────────────────────────────────────────────────────────────────
def run_cv(name, fold_dfs, dataset_cls, make_model, output_dir):
    os.makedirs(output_dir, exist_ok=True)

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
        print(f"  Train: {len(train_sub_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

        train_loader = DataLoader(dataset_cls(train_sub_df, DATA_DIR),
                                  batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
        val_loader   = DataLoader(dataset_cls(val_df,   DATA_DIR),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader  = DataLoader(dataset_cls(test_df,  DATA_DIR),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        # Full train (unshuffled) — used only for embedding extraction so saved
        # train embeddings still cover every training sample, not just the subset.
        train_full_loader = DataLoader(dataset_cls(train_df, DATA_DIR),
                                       batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model      = make_model().to(DEVICE)
        pos_weight = torch.tensor(
            [(1 - train_sub_df['correct'].mean()) / train_sub_df['correct'].mean()]
        ).to(DEVICE)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
        scaler     = GradScaler()

        best_auc, epochs_no_improve = 0.0, 0
        for epoch in range(N_EPOCHS):
            train_loss    = _train(model, train_loader, optimizer, criterion, DEVICE, scaler)
            val_auc, _, _ = _eval(model, val_loader, DEVICE)
            print(f"  Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
                torch.save(model.state_dict(),
                           os.path.join(output_dir, f'fold_{test_fold}_best.pt'))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"  Early stop at epoch {epoch+1} "
                          f"(no val-AUC improvement for {PATIENCE} epochs).")
                    break

        # Load best checkpoint and extract embeddings for train + test
        best_ckpt = os.path.join(output_dir, f'fold_{test_fold}_best.pt')
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        train_embs, train_emb_labels = _embed(model, train_full_loader, DEVICE)
        test_embs,  test_emb_labels  = _embed(model, test_loader,  DEVICE)
        np.save(os.path.join(output_dir, f'fold_{test_fold}_train_embeddings.npy'), train_embs)
        np.save(os.path.join(output_dir, f'fold_{test_fold}_train_emb_labels.npy'), train_emb_labels)
        np.save(os.path.join(output_dir, f'fold_{test_fold}_test_embeddings.npy'),  test_embs)
        np.save(os.path.join(output_dir, f'fold_{test_fold}_test_emb_labels.npy'),  test_emb_labels)

        fold_auc, fold_probs, fold_labels = _eval(model, test_loader, DEVICE)
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

    # Score-level aggregation writes to its own directory so the legacy
    # embedding-pooling results under cf_{K}/ stay intact.
    fold_tag = f'cf_{CF_COUNT}' if CF_AGG == 'embedding' else f'cf_{CF_COUNT}_{CF_AGG}'
    out = lambda name: os.path.join(RES_BASE, f'{name}/{DISEASE}/{fold_tag}')

    # Registry of all six configurations. `has_cf` marks the configs whose
    # dataset yields K stacks, and so are the only ones --cf_agg applies to.
    MODEL_REGISTRY = {
        'xi': dict(
            name        = 'Gray Xi',
            dataset_cls = XiDataset,
            in_channels = 1,
            has_cf      = False,
            output_dir  = out('early_baseline'),
        ),
        'cf': dict(
            name        = 'Gray CF',
            dataset_cls = CFDataset,
            in_channels = 1,
            has_cf      = True,
            output_dir  = out('early_cf_baseline'),
        ),
        'xi_sal': dict(
            name        = 'Early Fusion Xi + Saliency',
            dataset_cls = XiSalDataset,
            in_channels = 2,
            has_cf      = False,
            output_dir  = out('early_baseline_saliency'),
        ),
        'cf_sal': dict(
            name        = 'Early Fusion CF + Saliency',
            dataset_cls = CFSalDataset,
            in_channels = 2,
            has_cf      = True,
            output_dir  = out('early_cf_saliency'),
        ),
        'dual': dict(
            name        = 'Early Fusion Xi + CF',
            dataset_cls = DualDataset,
            in_channels = 2,
            has_cf      = True,
            output_dir  = out('early_dual'),
        ),
        'dual_sal': dict(
            name        = 'Early Fusion Xi + CF + Saliency',
            dataset_cls = DualSalDataset,
            in_channels = 4,
            has_cf      = True,
            output_dir  = out('early_dual_saliency'),
        ),
    }

    all_aucs = {}
    for key in SELECTED:
        cfg = dict(MODEL_REGISTRY[key])
        in_channels, has_cf = cfg.pop('in_channels'), cfg.pop('has_cf')
        # Without a CF stack there is nothing to aggregate: K is always 1, so a
        # score MLP would expect CF_COUNT inputs it never receives.
        agg = CF_AGG if has_cf else 'mean'
        cfg['make_model'] = partial(EarlyFusionCNN, in_channels=in_channels,
                                    cf_agg=agg, k=CF_COUNT, agg_hidden=AGG_HIDDEN)
        all_aucs[cfg['name']] = run_cv(fold_dfs=fold_dfs, **cfg)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, aucs in all_aucs.items():
        print(f"  {name:<32}  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
