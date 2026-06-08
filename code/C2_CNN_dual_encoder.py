"""
c2_cnn_dual_encoder.py
======================
Dual-encoder CNN baseline for C2 quality control.

Each sample uses two inputs:
  1. the original query image
  2. its nearest counterfactual image

The script trains a pair of separate DenseNet121 encoders, concatenates the two
feature vectors, and predicts whether the original sample was correctly
classified.

Cross-validation mirrors the saved fold CSVs produced by the C2 pipeline:
  - fold_i_predictions.csv is the test set for fold i
  - the remaining 4 folds are concatenated as training data

Outputs saved per fold:
  - fold_{i}_best.pt
  - fold_{i}_fpr.npy
  - fold_{i}_tpr.npy
  - fold_{i}_probs.npy
  - fold_{i}_labels.npy
  - fold_{i}_predictions.csv
  - cv_summary.csv
  - cv_detailed.json
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ── Config ────────────────────────────────────────────────────────────────────
DISEASE = 'effusion'
CF_COUNT = 1
BASE_DIR = "/zhome/d0/a/221493/thesis"
DATA_DIR = os.path.join(BASE_DIR, "data")
CV_DIR = os.path.join(BASE_DIR, f"results/C2_custom/{DISEASE}/cv_results/cf_{CF_COUNT}")
OUTPUT_DIR = os.path.join(BASE_DIR, f"results/C2_cnn_dual_baseline/{DISEASE}/cf_{CF_COUNT}")
N_FOLDS = 5
N_EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
class QueryCFDataset(Dataset):
    """Loads the query image and its CF image for each sample."""

    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _parse_cf_path(self, value):
        cf_path = str(value)
        if '|' in cf_path:
            cf_path = cf_path.split('|')[0]
        return cf_path.strip()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = float(row['correct'])

        query_path = str(row['path'])
        cf_path = self._parse_cf_path(row['cf_paths'])

        query_img = Image.open(os.path.join(self.data_dir, query_path)).convert('RGB')
        cf_img = Image.open(os.path.join(self.data_dir, cf_path)).convert('RGB')

        if self.transform:
            query_img = self.transform(query_img)
            cf_img = self.transform(cf_img)

        return query_img, cf_img, label


# ── Transforms ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ── Model ─────────────────────────────────────────────────────────────────────
class ImageEncoder(nn.Module):
    """DenseNet121 backbone that returns a 1024-D embedding."""

    def __init__(self):
        super().__init__()
        backbone = models.densenet121(weights='IMAGENET1K_V1')
        self.features = backbone.features
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class DualEncoderCNN(nn.Module):
    """Two separate encoders whose embeddings are concatenated for prediction."""

    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.query_encoder = ImageEncoder()
        self.cf_encoder = ImageEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_img, cf_img):
        query_feat = self.query_encoder(query_img)
        cf_feat = self.cf_encoder(cf_img)
        fused = torch.cat([query_feat, cf_feat], dim=1)
        return self.classifier(fused)


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for query_imgs, cf_imgs, labels in loader:
        query_imgs = query_imgs.to(device)
        cf_imgs = cf_imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(query_imgs, cf_imgs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    with torch.no_grad():
        for query_imgs, cf_imgs, labels in loader:
            query_imgs = query_imgs.to(device)
            cf_imgs = cf_imgs.to(device)
            labels = labels.to(device)

            logits = model(query_imgs, cf_imgs).squeeze(1)
            total_loss += criterion(logits, labels).item()
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), auc, all_probs, all_labels


def evaluate_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for query_imgs, cf_imgs, labels in loader:
            query_imgs = query_imgs.to(device)
            cf_imgs = cf_imgs.to(device)

            logits = model(query_imgs, cf_imgs).squeeze(1)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return auc, all_probs, all_labels


# ── Main CV loop ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fold_dfs = [
        pd.read_csv(os.path.join(CV_DIR, f'fold_{i}_predictions.csv'))
        for i in range(N_FOLDS)
    ]

    fold_aucs = []
    cv_results = []

    for test_fold in range(N_FOLDS):
        print(f"\n{'─'*50}")
        print(f"FOLD {test_fold + 1}/{N_FOLDS}")
        print(f"{'─'*50}")

        test_df = fold_dfs[test_fold]
        train_df = pd.concat(
            [fold_dfs[i] for i in range(N_FOLDS) if i != test_fold],
            ignore_index=True
        )

        print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

        train_loader = DataLoader(
            QueryCFDataset(train_df, DATA_DIR, transform),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
        )
        test_loader = DataLoader(
            QueryCFDataset(test_df, DATA_DIR, transform),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
        )

        model = DualEncoderCNN().to(DEVICE)
        pos_weight = torch.tensor([
            (1 - train_df['correct'].mean()) / train_df['correct'].mean()
        ]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_auc = 0.0
        for epoch in range(N_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_auc, _, _ = evaluate(model, test_loader, criterion, DEVICE)
            print(f"  Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'fold_{test_fold}_best.pt'))
                print(f"    -> Saved best model (AUC={best_auc:.4f})")

        fold_auc, fold_probs, fold_labels = evaluate_probs(model, test_loader, DEVICE)
        fpr, tpr, _ = roc_curve(fold_labels, fold_probs)

        np.save(os.path.join(OUTPUT_DIR, f'fold_{test_fold}_fpr.npy'), fpr)
        np.save(os.path.join(OUTPUT_DIR, f'fold_{test_fold}_tpr.npy'), tpr)
        np.save(os.path.join(OUTPUT_DIR, f'fold_{test_fold}_probs.npy'), np.asarray(fold_probs))
        np.save(os.path.join(OUTPUT_DIR, f'fold_{test_fold}_labels.npy'), np.asarray(fold_labels))

        fold_pred_df = test_df.copy()
        fold_pred_df['dual_prob_pred'] = fold_probs
        fold_pred_df['dual_correct_label'] = fold_labels
        fold_pred_df.to_csv(os.path.join(OUTPUT_DIR, f'fold_{test_fold}_predictions.csv'), index=False)

        fold_aucs.append(fold_auc)
        cv_results.append({
            'fold': test_fold,
            'auc': float(fold_auc),
            'best_auc': float(best_auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'y_prob': list(map(float, fold_probs)),
            'y_true': list(map(int, fold_labels)),
        })

        print(f"  Fold AUC: {fold_auc:.4f}")

    summary_df = pd.DataFrame([
        {'fold': row['fold'], 'auc': row['auc'], 'best_auc': row['best_auc']}
        for row in cv_results
    ])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'cv_summary.csv'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    print(f"\nMean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Outputs saved to {OUTPUT_DIR}")