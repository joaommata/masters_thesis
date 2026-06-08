"""
c2_cnn_cf_baseline.py
=====================
CNN baseline for C2 quality control using the CF image as input.
For each query sample, loads its nearest-opposite CF image and trains
a DenseNet121 to predict whether the query was correctly classified.

Cross-validation mirrors the existing C2 pipeline:
  - fold_i_predictions.csv is the test set for fold i
  - the remaining 4 folds are concatenated as training data

Requires: fold_*_predictions.csv files produced by c2_cv_pipeline_new_split.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

# ── Config ────────────────────────────────────────────────────────────────────
DISEASE    = 'effusion'
CF_COUNT   = 1
BASE_DIR   = "/zhome/d0/a/221493/thesis"
DATA_DIR   = os.path.join(BASE_DIR, "data")
CV_DIR     = os.path.join(BASE_DIR, f"results/C2_custom/{DISEASE}/cv_results/cf_{CF_COUNT}")
OUTPUT_DIR = os.path.join(BASE_DIR, f"results/C2_cnn_cf_baseline/{DISEASE}/cf_{CF_COUNT}")
N_FOLDS    = 5
N_EPOCHS   = 10
BATCH_SIZE = 32
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
class CFImageDataset(Dataset):
    """
    Loads the CF image for each sample and returns it with the correctness label.
    cf_paths column contains a single path (K=1 case).
    """
    def __init__(self, df, data_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        label   = float(row['correct'])
        # cf_paths is a single path string when K=1
        cf_path = str(row['cf_paths'])
        img     = Image.open(os.path.join(self.data_dir, cf_path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ── Transforms ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model

# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs).squeeze(1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs).squeeze(1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return auc, all_probs, all_labels

# ── Main CV loop ──────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Load all fold CSVs upfront
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

        # Test = this fold, train = all other folds concatenated
        test_df  = fold_dfs[test_fold]
        train_df = pd.concat([fold_dfs[i] for i in range(N_FOLDS) if i != test_fold],
                             ignore_index=True)

        print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

        train_loader = DataLoader(
            CFImageDataset(train_df, DATA_DIR, transform),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        test_loader = DataLoader(
            CFImageDataset(test_df, DATA_DIR, transform),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

        model    = build_model().to(DEVICE)
        # pos_weight handles class imbalance — same approach as query-image baseline
        pos_weight = torch.tensor(
            [(1 - train_df['correct'].mean()) / train_df['correct'].mean()]
        ).to(DEVICE)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(model.parameters(), lr=LR)

        best_auc = 0
        for epoch in range(N_EPOCHS):
            train_loss       = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_auc, _, _    = evaluate(model, test_loader, DEVICE)
            print(f"  Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(),
                           os.path.join(OUTPUT_DIR, f'fold_{test_fold}_best.pt'))

        fold_aucs.append(best_auc)
        print(f"  Best AUC fold {test_fold}: {best_auc:.4f}")

        fold_auc, fold_probs, fold_labels = evaluate(model, test_loader, DEVICE)
        fpr, tpr, _ = roc_curve(fold_labels, fold_probs)

        np.save(os.path.join(OUTPUT_DIR, f'fold_{test_fold}_fpr.npy'), fpr)
        np.save(os.path.join(OUTPUT_DIR, f'fold_{test_fold}_tpr.npy'), tpr)
        np.save(os.path.join(OUTPUT_DIR, f'fold_{test_fold}_probs.npy'), np.asarray(fold_probs))
        np.save(os.path.join(OUTPUT_DIR, f'fold_{test_fold}_labels.npy'), np.asarray(fold_labels))

        fold_pred_df = test_df.copy()
        fold_pred_df['cf_prob_pred'] = fold_probs
        fold_pred_df['cf_correct_label'] = fold_labels
        fold_pred_df.to_csv(
            os.path.join(OUTPUT_DIR, f'fold_{test_fold}_predictions.csv'),
            index=False
        )

        cv_results.append({
            'fold': test_fold,
            'auc': float(fold_auc),
            'best_auc': float(best_auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'y_prob': list(map(float, fold_probs)),
            'y_true': list(map(int, fold_labels)),
        })
        print(f"  Saved ROC data for fold {test_fold}")

    
    print(f"\n{'='*50}")
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")

    summary_df = pd.DataFrame([
        {
            'fold': row['fold'],
            'auc': row['auc'],
            'best_auc': row['best_auc'],
        }
        for row in cv_results
    ])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'cv_summary.csv'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    print(f"Saved ROC outputs to {OUTPUT_DIR}")