"""
c2_KNN_baseline.py
=====================
KNN quality control baseline, evaluated with the same 5-fold CV
structure as c2_cv_pipeline_new_split.py, for direct comparison.

For each test sample xi:
  - Find k nearest neighbors in the training fold (pool: same_class or any)
  - Score = fraction of those neighbors that C0 predicted correctly
  - Evaluate with ROC AUC against ground truth 'correct' label

Outputs saved to:
  results/C2_custom/{disease}/knn_baseline/
    cv_summary.csv
    cv_detailed.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, roc_curve

# ── CONFIG ────────────────────────────────────────────────────────────────────
DISEASE     = 'effusion'
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
N_FOLDS     = 5
RANDOM_SEED = 42

K_VALUES    = [1, 3, 5, 10]
POOL_MODES  = ['same_class', 'any']

# ── LOAD DATA (same as cv pipeline) ──────────────────────────────────────────
results_base = os.path.join(RESULTS_DIR, '')
full_df = pd.read_csv(os.path.join(results_base, f'C2_custom/{DISEASE}/c2_data.csv'))

full_df.rename(columns={
    'prob': f'{DISEASE}_prob',
    'pred': f'{DISEASE}_pred',
    'true': f'{DISEASE}_true'
}, inplace=True)

print(f"Full dataset: {len(full_df):,} samples")
print(f"  Correct:   {(full_df['correct']==1).sum():,}")
print(f"  Incorrect: {(full_df['correct']==0).sum():,}\n")

# Identify attribute columns (same logic as rest of pipeline)
disease_pred_col = f'{DISEASE}_pred'
meta_cols = {f'{DISEASE}_prob', disease_pred_col, f'{DISEASE}_true',
             'correct', 'path', 'patient_id'}
attr_cols = [c for c in full_df.columns
             if c not in meta_cols
             and not c.startswith('emb_')]

print(f"Attribute features used for distance: {len(attr_cols)}\n")

# ── KNN SCORING FUNCTION ──────────────────────────────────────────────────────

def knn_quality_score(train_scaled, train_correct, train_preds,
                      query_scaled, query_preds, k, pool_mode):
    """
    Score each query sample by the fraction of its k nearest
    training neighbors that C0 predicted correctly.
    """
    n_queries = len(query_scaled)
    scores = np.zeros(n_queries)

    if pool_mode == 'any':
        nn = NearestNeighbors(n_neighbors=k, metric='manhattan', n_jobs=-1)
        nn.fit(train_scaled)
        _, idxs = nn.kneighbors(query_scaled)
        scores = train_correct[idxs].mean(axis=1)

    elif pool_mode == 'same_class':
        for cls in [0, 1]:
            # Build NN index only from training samples with same predicted class
            train_mask = (train_preds == cls)
            query_mask = (query_preds == cls)
            if not query_mask.any() or not train_mask.any():
                continue

            pool_size = train_mask.sum()
            nn = NearestNeighbors(
                n_neighbors=min(k, pool_size),  # pool may be smaller than k
                metric='manhattan', n_jobs=-1
            ).fit(train_scaled[train_mask])

            _, local_idxs = nn.kneighbors(query_scaled[query_mask])
            # Map local pool indices back to global training indices
            global_idxs = np.where(train_mask)[0][local_idxs]
            scores[query_mask] = train_correct[global_idxs].mean(axis=1)

    return scores

# ── CV LOOP ───────────────────────────────────────────────────────────────────

y = full_df['correct'].values
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# Storage mirrors cv_detailed.json format: {pool_mode: {k: [fold_results]}}
cv_results = {
    pool_mode: {str(k): [] for k in K_VALUES}
    for pool_mode in POOL_MODES
}

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, y)):
    print(f"\n{'-'*60}")
    print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"{'-'*60}")

    fold_train = full_df.iloc[train_idx].reset_index(drop=True)
    fold_test  = full_df.iloc[test_idx].reset_index(drop=True)

    # Scale fit on train fold only — same discipline as CF pipeline
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(fold_train[attr_cols].fillna(0).values)
    test_scaled  = scaler.transform(fold_test[attr_cols].fillna(0).values)

    train_preds   = fold_train[disease_pred_col].values.astype(int)
    train_correct = fold_train['correct'].values.astype(int)
    test_preds    = fold_test[disease_pred_col].values.astype(int)
    test_correct  = fold_test['correct'].values.astype(int)

    for pool_mode in POOL_MODES:
        for k in K_VALUES:
            scores = knn_quality_score(
                train_scaled, train_correct, train_preds,
                test_scaled,  test_preds,
                k=k, pool_mode=pool_mode
            )

            auc = roc_auc_score(test_correct, scores)
            fpr, tpr, _ = roc_curve(test_correct, scores)

            cv_results[pool_mode][str(k)].append({
                'auc': auc,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
            })

            print(f"  pool={pool_mode:<12} k={k:>2}  AUC={auc:.4f}")

# ── AGGREGATE ─────────────────────────────────────────────────────────────────

summary_rows = []
for pool_mode in POOL_MODES:
    for k in K_VALUES:
        aucs = [f['auc'] for f in cv_results[pool_mode][str(k)]]
        summary_rows.append({
            'pool_mode':  pool_mode,
            'k':          k,
            'mean_auc':   np.mean(aucs),
            'std_auc':    np.std(aucs),
            'fold_aucs':  ','.join([f'{a:.4f}' for a in aucs])
        })

summary_df = pd.DataFrame(summary_rows)

# ── SAVE ──────────────────────────────────────────────────────────────────────

out_dir = os.path.join(results_base, f'C2_custom/{DISEASE}/knn_baseline')
os.makedirs(out_dir, exist_ok=True)

summary_df.to_csv(os.path.join(out_dir, 'cv_summary.csv'), index=False)

with open(os.path.join(out_dir, 'cv_detailed.json'), 'w') as f:
    json.dump(cv_results, f, indent=2)

print(f"\n{'='*60}")
print(summary_df.to_string(index=False))
print(f"\nSaved to: {out_dir}")