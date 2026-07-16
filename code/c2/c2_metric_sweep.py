#!/usr/bin/env python3
"""
Metric sweep for CF neighbour retrieval.

Tests whether the distance metric used to SELECT CF neighbours affects downstream AUC.
Only the retrieval step changes; the ΔA computation (arithmetic diff in scaled attr
space) is identical across metrics — the metric determines WHICH neighbours are picked,
not HOW the diff vector is computed.

Metrics:  euclidean | manhattan | cosine | mahalanobis
Configs:  B1  (prob only — metric-agnostic baseline)
          M3  (prob + ΔA + attr — best non-emb config, depends on selected neighbours)
          MCF1 (prob + cf_prob)
          MCF2 (prob + cf_prob + attr + cf_attr)
          MCF3 (prob + cf_prob + attr + cf_attr + ΔA)
Regime:   unmatched only (same scope as MCF1-MCF5 in c2_cv_pipeline_new_split.py)

Usage:
    python code/c2_metric_sweep.py --disease effusion --cf_count 16 --backbone densenet
"""

import sys
import os
sys.path.insert(0, '/zhome/d0/a/221493/thesis/code')

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from c2_prepare_data_simulated_cf import (
    _get_disease_cols,
    add_clinical_ratios,
    attach_cf_features,
)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
N_FOLDS      = 5
METRICS      = ['euclidean', 'manhattan', 'cosine', 'mahalanobis']
EVAL_CONFIGS = ['B1', 'M3', 'MCF1', 'MCF2', 'MCF3']

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
BACKBONE_MAP = {
    'densenet': 'C2_custom',
    'resnet50': 'C2_resnet50',
    'vit':      'C2_vit',
}


# ── CF retrieval — metric-parameterised, unmatched regime ─────────────────────
def _compute_cf_unmatched(train_df, test_df, cf_count, disease, metric='euclidean'):
    """
    Mirrors compute_cf_for_split_unmatched() but accepts any sklearn metric string.
    Mahalanobis: VI computed from train_scaled with 1e-6 ridge regularisation.
    """
    disease_prob_col, disease_pred_col, disease_true_col = _get_disease_cols(disease)

    train_df = add_clinical_ratios(train_df.copy())
    test_df  = add_clinical_ratios(test_df.copy())

    meta_cols     = {disease_prob_col, disease_pred_col, disease_true_col,
                     'correct', 'path', 'patient_id'}
    emb_cols      = [c for c in train_df.columns if c.startswith('emb_')]
    relevant_cols = [c for c in train_df.columns
                     if c not in meta_cols and c not in emb_cols]

    train_clean = train_df.copy()
    test_clean  = test_df.copy()
    train_clean[relevant_cols] = train_clean[relevant_cols].fillna(0)
    test_clean[relevant_cols]  = test_clean[relevant_cols].fillna(0)

    attr_scaler  = StandardScaler()
    train_scaled = attr_scaler.fit_transform(train_clean[relevant_cols].values.astype(float))
    test_scaled  = attr_scaler.transform(test_clean[relevant_cols].values.astype(float))

    # Build NearestNeighbors kwargs; mahalanobis needs VI and brute algorithm
    if metric == 'mahalanobis':
        cov = np.cov(train_scaled.T)
        cov += np.eye(cov.shape[0]) * 1e-6
        VI  = np.linalg.inv(cov)
        nn_kw = dict(metric='mahalanobis', metric_params={'VI': VI},
                     algorithm='brute', n_jobs=1)
    else:
        nn_kw = dict(metric=metric, algorithm='brute', n_jobs=-1)

    train_preds = train_clean[disease_pred_col].values.astype(int)
    idx_pred1   = (train_preds == 1)
    idx_pred0   = (train_preds == 0)

    pool_idx = {'pred1': np.where(idx_pred1)[0], 'pred0': np.where(idx_pred0)[0]}
    train_scaled_pred1 = train_scaled[idx_pred1]
    train_scaled_pred0 = train_scaled[idx_pred0]
    train_prob_pred1   = train_clean[disease_prob_col].values[idx_pred1]
    train_prob_pred0   = train_clean[disease_prob_col].values[idx_pred0]

    nn_pred1 = NearestNeighbors(n_neighbors=cf_count, **nn_kw).fit(train_scaled_pred1)
    nn_pred0 = NearestNeighbors(n_neighbors=cf_count, **nn_kw).fit(train_scaled_pred0)

    def _diffs(df, query_scaled):
        query_preds = df[disease_pred_col].values.astype(int)
        n, d = query_scaled.shape
        diff_vecs = np.empty((n, d), dtype=np.float64)
        cf_probs  = np.empty(n, dtype=np.float64)
        cf_paths  = np.empty(n, dtype=object)

        mask = (query_preds == 1)
        if mask.any():
            _, idxs = nn_pred0.kneighbors(query_scaled[mask])
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred0[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_pred0[idxs].mean(axis=1)
            cf_paths[mask]  = np.array(['|'.join(train_clean['path'].values[pool_idx['pred0'][row]])
                                        for row in idxs])

        mask = (query_preds == 0)
        if mask.any():
            _, idxs = nn_pred1.kneighbors(query_scaled[mask])
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred1[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_pred1[idxs].mean(axis=1)
            cf_paths[mask]  = np.array(['|'.join(train_clean['path'].values[pool_idx['pred1'][row]])
                                        for row in idxs])

        return diff_vecs, cf_probs, cf_paths

    train_diff, train_cf_probs, train_cf_paths = _diffs(train_clean, train_scaled)
    test_diff,  test_cf_probs,  test_cf_paths  = _diffs(test_clean,  test_scaled)

    diff_col_names = [f'delta_{c}' for c in relevant_cols]
    train_out = pd.concat([train_clean,
                           pd.DataFrame(train_diff, columns=diff_col_names, index=train_clean.index),
                           pd.DataFrame({'cf_prob': train_cf_probs, 'cf_paths': train_cf_paths},
                                        index=train_clean.index)], axis=1)
    test_out  = pd.concat([test_clean,
                           pd.DataFrame(test_diff,  columns=diff_col_names, index=test_clean.index),
                           pd.DataFrame({'cf_prob': test_cf_probs, 'cf_paths': test_cf_paths},
                                        index=test_clean.index)], axis=1)

    return train_out, test_out


# ── Feature matrices for the eval subset ──────────────────────────────────────
def _build_eval_matrices(train_df, test_df, disease):
    meta_cols = {f'{disease}_prob', f'{disease}_pred', f'{disease}_true',
                 'correct', 'path', 'patient_id',
                 'cf_prob', 'cf_paths', 'delta_prob'}
    diff_cols    = [c for c in train_df.columns if c.startswith('delta_')]
    cf_attr_cols = [c for c in train_df.columns if c.startswith('cf_attr_')]
    attr_cols    = [c for c in train_df.columns
                    if c not in meta_cols
                    and not c.startswith('delta_')
                    and not c.startswith('emb_')
                    and not c.startswith('cf_attr_')
                    and not c.startswith('cf_emb_')]

    prob_tr,    prob_te    = train_df[[f'{disease}_prob']].values, test_df[[f'{disease}_prob']].values
    attr_tr,    attr_te    = train_df[attr_cols].values,           test_df[attr_cols].values
    diff_tr,    diff_te    = train_df[diff_cols].values,           test_df[diff_cols].values
    cf_prob_tr, cf_prob_te = train_df[['cf_prob']].values,         test_df[['cf_prob']].values

    configs = {
        'B1': (prob_tr, prob_te),
        'M3': (np.hstack([prob_tr, diff_tr, attr_tr]), np.hstack([prob_te, diff_te, attr_te])),
    }

    if cf_attr_cols:
        cf_attr_tr, cf_attr_te = train_df[cf_attr_cols].values, test_df[cf_attr_cols].values
        configs.update({
            'MCF1': (np.hstack([prob_tr, cf_prob_tr]),
                     np.hstack([prob_te, cf_prob_te])),
            'MCF2': (np.hstack([prob_tr, cf_prob_tr, attr_tr, cf_attr_tr]),
                     np.hstack([prob_te, cf_prob_te, attr_te, cf_attr_te])),
            'MCF3': (np.hstack([prob_tr, cf_prob_tr, attr_tr, cf_attr_tr, diff_tr]),
                     np.hstack([prob_te, cf_prob_te, attr_te, cf_attr_te, diff_te])),
        })

    return configs


# ── Single model train + AUC ──────────────────────────────────────────────────
def _score(X_tr, X_te, y_tr, y_te, model_type):
    scaler = StandardScaler()
    if model_type == 'LR':
        clf = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=RANDOM_SEED)
        clf.fit(scaler.fit_transform(X_tr), y_tr)
        return roc_auc_score(y_te, clf.predict_proba(scaler.transform(X_te))[:, 1])
    else:  # MLP
        clf = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                            early_stopping=True, validation_fraction=0.05, random_state=RANDOM_SEED)
        clf.fit(scaler.fit_transform(X_tr.astype(np.float32)), y_tr)
        return roc_auc_score(y_te, clf.predict_proba(scaler.transform(X_te.astype(np.float32)))[:, 1])


# ── Main ───────────────────────────────────────────────────────────────────────
def run_metric_sweep(disease, cf_count, backbone='densenet'):
    results_base = os.path.join(RESULTS_DIR, '')
    data_subdir  = BACKBONE_MAP[backbone]
    data_path    = os.path.join(results_base, f'{data_subdir}/{disease}/c2_data.csv')

    print(f"Loading: {data_path}")
    full_df = pd.read_csv(data_path)
    full_df.rename(columns={'prob': f'{disease}_prob',
                             'pred': f'{disease}_pred',
                             'true': f'{disease}_true'}, inplace=True)

    print(f"Dataset: {len(full_df):,} samples | disease={disease} | k={cf_count} | backbone={backbone}\n")

    y   = full_df['correct'].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    records = []

    for metric in METRICS:
        print(f"{'='*60}")
        print(f"Metric: {metric}")
        fold_aucs = {(cfg, mdl): [] for cfg in EVAL_CONFIGS for mdl in ['LR', 'MLP']}

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, y)):
            print(f"  fold {fold_idx + 1}/{N_FOLDS} ...", end=' ', flush=True)

            fold_train = full_df.iloc[train_idx].reset_index(drop=True)
            fold_test  = full_df.iloc[test_idx].reset_index(drop=True)
            y_tr = fold_train['correct'].values
            y_te = fold_test['correct'].values

            fold_train_cf, fold_test_cf = _compute_cf_unmatched(
                fold_train, fold_test, cf_count=cf_count, disease=disease, metric=metric)

            # Attach cf_attr_* / cf_emb_* (lookup always points into train fold)
            _meta_set = {f'{disease}_prob', f'{disease}_pred', f'{disease}_true',
                         'correct', 'path', 'patient_id', 'cf_prob', 'cf_paths'}
            _emb  = [c for c in fold_train_cf.columns if c.startswith('emb_')]
            _rel  = [c for c in fold_train_cf.columns
                     if c not in _meta_set
                     and not c.startswith('emb_')
                     and not c.startswith('delta_')]
            _lut          = fold_train_cf
            fold_train_cf = attach_cf_features(fold_train_cf, _lut, _rel, _emb, disease)
            fold_test_cf  = attach_cf_features(fold_test_cf,  _lut, _rel, _emb, disease)

            feature_sets = _build_eval_matrices(fold_train_cf, fold_test_cf, disease)

            for cfg, (X_tr, X_te) in feature_sets.items():
                for mdl in ['LR', 'MLP']:
                    fold_aucs[(cfg, mdl)].append(_score(X_tr, X_te, y_tr, y_te, mdl))

            print("done")

        for (cfg, mdl), aucs in fold_aucs.items():
            if aucs:
                records.append({'metric': metric, 'config': cfg, 'model': mdl,
                                 'mean_auc': round(np.mean(aucs), 4),
                                 'std_auc':  round(np.std(aucs),  4)})

    sweep_df = pd.DataFrame(records)

    # Print pivot: rows = config, cols = metric, for each model
    print(f"\n{'='*70}")
    print(f"RESULTS — {disease.upper()} | k={cf_count} | backbone={backbone}")
    print(f"{'='*70}")
    for mdl in ['LR', 'MLP']:
        sub = sweep_df[sweep_df['model'] == mdl].pivot(
            index='config', columns='metric', values='mean_auc')
        sub = sub.reindex(index=EVAL_CONFIGS, columns=METRICS)
        print(f"\nModel: {mdl}  (mean AUC across {N_FOLDS} folds)")
        print(sub.to_string())

    # Save
    out_dir  = os.path.join(results_base, f'{data_subdir}_corrected/{disease}/metric_sweep')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'metric_sweep_k{cf_count}.csv')
    sweep_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return sweep_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metric sweep for CF neighbour retrieval')
    parser.add_argument('--disease',  type=str, default='effusion')
    parser.add_argument('--cf_count', type=int, default=16)
    parser.add_argument('--backbone', type=str, default='densenet',
                        choices=['densenet', 'resnet50', 'vit'])
    args = parser.parse_args()
    run_metric_sweep(disease=args.disease, cf_count=args.cf_count, backbone=args.backbone)
