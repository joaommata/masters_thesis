"""
c2_cnn_baseline.py
==================
CNN baseline for C2 quality control.
Predicts correctness (correct=1/incorrect=0) directly from the image.
Uses the same DenseNet121 backbone as C0, but fine-tuned on the correctness label.
"""

import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

# ── Config ────────────────────────────────────────────────────────────────────
DISEASE    = 'effusion'
CF_COUNT   = 1
BASE_DIR   = "/zhome/d0/a/221493/thesis"
DATA_DIR   = os.path.join(BASE_DIR, "data")
C2_DATA    = os.path.join(BASE_DIR, f"results/C2_custom/{DISEASE}/c2_data.csv")  # same input as cv pipeline
CV_DIR     = os.path.join(BASE_DIR, f"results/C2_custom/{DISEASE}/cv_results/cf_{CF_COUNT}")
OUTPUT_DIR = os.path.join(BASE_DIR, f"results/C2_cnn_baseline/{DISEASE}/cf_{CF_COUNT}")
N_FOLDS    = 5
N_EPOCHS   = 10
BATCH_SIZE = 32
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
class CorrectnessDataset(Dataset):
    """
    Loads images and returns them with their correctness label.
    The target is 'correct' (1=C0 was right, 0=C0 was wrong).
    """
    def __init__(self, df, data_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = float(row['correct'])  # <-- key change: correctness, not disease label
        img   = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
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
    """DenseNet121 with final layer replaced for binary correctness prediction."""
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

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).squeeze(1)
            total_loss += criterion(logits, labels).item()
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), auc, all_probs, all_labels


def evaluate_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return auc, all_probs, all_labels

# ── Main ──────────────────────────────────────────────────────────────────────
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

        test_df  = fold_dfs[test_fold]
        train_df = pd.concat([fold_dfs[i] for i in range(N_FOLDS) if i != test_fold],
                             ignore_index=True)

        print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

        train_loader = DataLoader(CorrectnessDataset(train_df, DATA_DIR, transform),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        test_loader  = DataLoader(CorrectnessDataset(test_df, DATA_DIR, transform),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model = build_model().to(DEVICE)
        pos_weight = torch.tensor([(1 - train_df['correct'].mean()) / train_df['correct'].mean()]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_auc = 0
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
        fold_pred_df['cnn_prob_pred'] = fold_probs
        fold_pred_df['cnn_correct_label'] = fold_labels
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