"""
c2_cv_pipeline_new_split.py

5-fold cross-validation pipeline for C2 quality control models with simulated counterfactuals (CFs).

Configs:
    Baseline (B1–B5):
        B1 : Model input = disease probability only
        B2 : Model input = attribute features only
        B3 : Model input = embedding features only
        B4 : disease probability + attribute features
        B5 : disease probability + embedding features

    Multi-modal (M1–M6):
        M1 : delta features only
        M2 : disease probability + delta features
        M3 : disease probability + delta + attribute features
        M4 : disease probability + delta + embedding features
        M5 : disease probability + delta + attribute + embedding features
        M6 : disease probability + delta + attribute + CF probability

    CF-enriched (MCF1–MCF5) — unmatched path only, progressive accumulation:
        MCF1 : prob + cf_prob
        MCF2 : prob + cf_prob + attr + cf_attr
        MCF3 : prob + cf_prob + attr + cf_attr + ΔA
        MCF4 : prob + cf_prob + attr + cf_attr + ΔA + emb + cf_emb
        MCF5 : prob + cf_prob + attr + cf_attr + ΔA + emb + cf_emb + Δemb

Usage:
    python c2_cv_pipeline_new_split.py --disease effusion --cf_count 16 --backbone resnet50 

Parameters:
    --disease   : Disease name (e.g., 'effusion')
    --cf_count  : Number of counterfactuals (k)
    --backbone  : C0 architecture ('densenet', 'resnet50', 'vit')
    --save_folds: Save individual fold CSVs to disk
    --unmatched  : Use unmatched CF pool (ablation)
    --correct_cf : CFs always drawn from correctly-classified examples (TP/TN pools)
    --k_offset   : Integer offset for CF selection (e.g., k_offset=3 uses the 4th-6th nearest CFs instead of 1st-3rd)
    --distance   : Distance metric for CF neighbour search ('l1', 'l2', 'cosine'); default 'l1'
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# Import the CF computation functions
sys.path.append('/zhome/d0/a/221493/thesis/code')
from c2_prepare_data_simulated_cf import (
    compute_cf_for_split_unmatched,
    compute_cf_for_split_further,
    compute_cf_for_split_matched_train_unmatched_test,
    compute_cf_for_split_correct_cf,
    compute_cf_for_split_gt_routing,
    attach_cf_features,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════════════════════
N_FOLDS     = 5
RANDOM_SEED = 42
CONFIGS     = ['B1', 'B2','B3','B4','B5',
               'M1','M2','M3','M4','M5','M6',
               'MCF1','MCF2','MCF3','MCF4','MCF5']
#CONFIGS = ['MCF1','MCF2','MCF3','MCF4','MCF5']  # For quick testing
# Map backbone name → results subdirectory
BACKBONE_MAP = {
    'densenet':  'C2_custom',
    'resnet50':  'C2_resnet50',
    'vit':       'C2_vit',
    'medmnist':  'C2_medmnist',
}

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
# ══════════════════════════════════════════════════════════════════════════════
# TRAIN-SPLIT BALANCING
# ══════════════════════════════════════════════════════════════════════════════

def balance_quadrants(train_df, disease, ratio=1.0, seed=RANDOM_SEED):
    """
    Subsample a training fold so C0's four outcome quadrants are balanced.

    Quadrants are (pred x correct): TP, FP, TN, FN. All error rows (FP, FN) are
    kept; the correct quadrants (TP, TN) are subsampled down to
    `ratio` * n_errors_per_quadrant, so the returned frame has roughly
    n_err : n_err : ratio*n_err : ratio*n_err across FN/FP/TN/TP.

    ratio=1.0 gives a fully balanced 1:1:1:1 pool. Larger values retain more
    correct examples (e.g. ratio=3.0 -> 3 correct per error, per prediction arm).
    A quadrant smaller than its target is kept whole (never upsampled), so no
    row is ever duplicated.

    Applied to the training partition only — test folds keep natural prevalence.
    """
    pred_col = f'{disease}_pred'
    rng = np.random.RandomState(seed)

    preds   = train_df[pred_col].values.astype(int)
    correct = train_df['correct'].values.astype(int)

    quadrants = {
        'TP': (preds == 1) & (correct == 1),
        'FP': (preds == 1) & (correct == 0),
        'TN': (preds == 0) & (correct == 1),
        'FN': (preds == 0) & (correct == 0),
    }

    n_before = {q: int(m.sum()) for q, m in quadrants.items()}

    # Errors are the scarce resource: they set the budget and are never dropped.
    n_err = min(n_before['FP'], n_before['FN'])
    if n_err == 0:
        print("    [balance] WARNING: empty error quadrant — skipping balancing for this fold")
        return train_df

    target_correct = int(round(ratio * n_err))

    keep_idx = []
    for q, mask in quadrants.items():
        idx = np.where(mask)[0]
        if q in ('FP', 'FN'):
            keep_idx.append(idx)                      # keep every error
        else:
            if len(idx) > target_correct:
                idx = rng.choice(idx, size=target_correct, replace=False)
            keep_idx.append(idx)

    keep_idx = np.sort(np.concatenate(keep_idx))
    balanced = train_df.iloc[keep_idx].reset_index(drop=True)

    b_preds   = balanced[pred_col].values.astype(int)
    b_correct = balanced['correct'].values.astype(int)
    n_after = {
        'TP': int(((b_preds == 1) & (b_correct == 1)).sum()),
        'FP': int(((b_preds == 1) & (b_correct == 0)).sum()),
        'TN': int(((b_preds == 0) & (b_correct == 1)).sum()),
        'FN': int(((b_preds == 0) & (b_correct == 0)).sum()),
    }
    print(f"    [balance] ratio={ratio}  {len(train_df):,} -> {len(balanced):,} samples")
    print(f"    [balance] before: {n_before}")
    print(f"    [balance] after:  {n_after}")

    return balanced


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, X_test, y_train, y_test, model_type='LR'):
    """Train a single model type and return test predictions + metrics + model/scaler."""

    if model_type == 'LR':
        scaler = StandardScaler()
        model  = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=RANDOM_SEED)
        model.fit(scaler.fit_transform(X_train), y_train)
        y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]
        return {
            'auc': float(roc_auc_score(y_test, y_prob)),
            'fpr': roc_curve(y_test, y_prob)[0].tolist(),
            'tpr': roc_curve(y_test, y_prob)[1].tolist(),
            'y_prob': y_prob.tolist(),
            'y_true': y_test.tolist(),
            'model': model,
            'scaler': scaler
        }

    elif model_type == 'RF':
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        return {
            'auc': float(roc_auc_score(y_test, y_prob)),
            'fpr': roc_curve(y_test, y_prob)[0].tolist(),
            'tpr': roc_curve(y_test, y_prob)[1].tolist(),
            'y_prob': y_prob.tolist(),
            'y_true': y_test.tolist(),
            'model': model
        }

    elif model_type == 'MLP':
        scaler = StandardScaler()
        model  = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, early_stopping=True, validation_fraction=0.05, random_state=RANDOM_SEED)
        X_train_f32 = X_train.astype(np.float32)
        X_test_f32  = X_test.astype(np.float32)
        model.fit(scaler.fit_transform(X_train_f32), y_train)
        y_prob = model.predict_proba(scaler.transform(X_test_f32))[:, 1]
        return {
            'auc': float(roc_auc_score(y_test, y_prob)),
            'fpr': roc_curve(y_test, y_prob)[0].tolist(),
            'tpr': roc_curve(y_test, y_prob)[1].tolist(),
            'y_prob': y_prob.tolist(),
            'y_true': y_test.tolist(),
            'model': model,
            'scaler': scaler
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")



# ══════════════════════════════════════════════════════════════════════════════
# Additional Baselines (Necessary Functions)
# ══════════════════════════════════════════════════════════════════════════════

def score_cf_anchor(fold_train_cf, fold_test_cf):
    """
    CF-anchor baseline: score each test sample by the mean correctness
    of training samples that share at least one CF anchor path with it.
    Falls back to 0.0 if no training sample shares an anchor.
    """
    # Build anchor -> [correctness] map from training samples
    anchor_correctness = {}
    for _, row in fold_train_cf.iterrows():
        if pd.isna(row['cf_paths']):
            continue
        for anchor in row['cf_paths'].split('|'):
            anchor_correctness.setdefault(anchor, []).append(row['correct'])

    # Score each test sample
    scores = np.zeros(len(fold_test_cf))
    for i, (_, row) in enumerate(fold_test_cf.iterrows()):
        if pd.isna(row['cf_paths']):
            continue
        shared = []
        for anchor in row['cf_paths'].split('|'):
            if anchor in anchor_correctness:
                shared.extend(anchor_correctness[anchor])
        if shared:
            scores[i] = np.mean(shared)
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrices(train_df, test_df, disease):
    """Extract all feature matrices for all configs from DataFrames."""

    meta_cols = {f'{disease}_prob', f'{disease}_pred', f'{disease}_true',
                 'correct', 'path', 'cam_path', 'patient_id',
                 'cf_prob', 'cf_paths', 'delta_prob'}
    diff_cols    = [c for c in train_df.columns if c.startswith('delta_')]
    emb_cols     = [c for c in train_df.columns if c.startswith('emb_')]
    cf_attr_cols = [c for c in train_df.columns if c.startswith('cf_attr_')]
    cf_emb_cols  = [c for c in train_df.columns if c.startswith('cf_emb_')]
    attr_cols    = [c for c in train_df.columns
                    if c not in meta_cols
                    and not c.startswith('delta_')
                    and not c.startswith('emb_')
                    and not c.startswith('cf_attr_')
                    and not c.startswith('cf_emb_')]

    prob_tr,    prob_te    = train_df[[f'{disease}_prob']].values, test_df[[f'{disease}_prob']].values
    
    # Theres no entropy column in the dataset, so we compute it here for B1.2
    prob_tr_entropy = -prob_tr * np.log(prob_tr + 1e-10) - (1 - prob_tr) * np.log(1 - prob_tr + 1e-10)
    prob_te_entropy = -prob_te * np.log(prob_te + 1e-10) - (1 - prob_te) * np.log(1 - prob_te + 1e-10)
    prob_tr_entropy = prob_tr_entropy.reshape(-1, 1)
    prob_te_entropy = prob_te_entropy.reshape(-1, 1)
    attr_tr,    attr_te    = train_df[attr_cols].values,           test_df[attr_cols].values
    diff_tr,    diff_te    = train_df[diff_cols].values,           test_df[diff_cols].values
    emb_tr,     emb_te     = train_df[emb_cols].values,            test_df[emb_cols].values
    cf_prob_tr, cf_prob_te = train_df[['cf_prob']].values,         test_df[['cf_prob']].values

    configs = {
        'B1': (prob_tr, prob_te),
        'B1.2': (prob_tr_entropy, prob_te_entropy),
        'B2': (attr_tr, attr_te),
        'B3': (emb_tr,  emb_te),
        'B4': (np.hstack([prob_tr, attr_tr]),              np.hstack([prob_te, attr_te])),
        'B5': (np.hstack([prob_tr, emb_tr]),               np.hstack([prob_te, emb_te])),
        'M1': (diff_tr, diff_te),
        'M2': (np.hstack([prob_tr, diff_tr]),              np.hstack([prob_te, diff_te])),
        'M3': (np.hstack([prob_tr, diff_tr, attr_tr]),     np.hstack([prob_te, diff_te, attr_te])),
        'M4': (np.hstack([prob_tr, diff_tr, emb_tr]),      np.hstack([prob_te, diff_te, emb_te])),
        'M5': (np.hstack([prob_tr, diff_tr, attr_tr, emb_tr]), np.hstack([prob_te, diff_te, attr_te, emb_te])),
        'M6': (np.hstack([prob_tr, diff_tr, attr_tr, cf_prob_tr]), np.hstack([prob_te, diff_te, attr_te, cf_prob_te])),
    }

    # MCF configs — require attach_cf_features() to have been called first
    if cf_attr_cols and cf_emb_cols:
        cf_attr_tr, cf_attr_te = train_df[cf_attr_cols].values, test_df[cf_attr_cols].values
        cf_emb_tr,  cf_emb_te  = train_df[cf_emb_cols].values,  test_df[cf_emb_cols].values
        # delta_emb computed on the fly: emb - cf_emb (aligned by construction)
        delta_emb_tr = train_df[emb_cols].values - train_df[cf_emb_cols].values
        delta_emb_te = test_df[emb_cols].values  - test_df[cf_emb_cols].values

        configs.update({
            'MCF1': (np.hstack([prob_tr, cf_prob_tr]),
                     np.hstack([prob_te, cf_prob_te])),
            'MCF2': (np.hstack([prob_tr, cf_prob_tr, attr_tr, cf_attr_tr]),
                     np.hstack([prob_te, cf_prob_te, attr_te, cf_attr_te])),
            'MCF3': (np.hstack([prob_tr, cf_prob_tr, attr_tr, cf_attr_tr, diff_tr]),
                     np.hstack([prob_te, cf_prob_te, attr_te, cf_attr_te, diff_te])),
            'MCF4': (np.hstack([prob_tr, cf_prob_tr, attr_tr, cf_attr_tr, diff_tr, emb_tr, cf_emb_tr]),
                     np.hstack([prob_te, cf_prob_te, attr_te, cf_attr_te, diff_te, emb_te, cf_emb_te])),
            'MCF5': (np.hstack([prob_tr, cf_prob_tr, attr_tr, cf_attr_tr, diff_tr, emb_tr, cf_emb_tr, delta_emb_tr]),
                     np.hstack([prob_te, cf_prob_te, attr_te, cf_attr_te, diff_te, emb_te, cf_emb_te, delta_emb_te])),
        })

    return configs


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CV PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(disease, cf_count, save_fold_data=False, unmatched=False, backbone='densenet', k_offset=0, correct_cf=False, extended=False, gt_routing=False, distance='l1', balance=False, balance_ratio=1.0):
    """
    Run complete 5-fold CV pipeline.

    Parameters
    ----------
    disease        : str  — Disease name (e.g., 'effusion')
    cf_count       : int  — Number of counterfactual neighbours (k)
    save_fold_data : bool — Save per-fold train/test CSVs to disk
    unmatched      : bool — Use unmatched CF pool (ablation)
    correct_cf     : bool — CFs always drawn from correctly-classified examples (TP/TN pools)
    backbone       : str  — C0 architecture ('densenet', 'resnet50', 'vit')
    k_offset       : int  — Skip k_offset nearest CFs for pred=0 queries (further strategy)
    extended       : bool — Use c2_data_extended.csv (includes extra radiomics features e.g. GLCM)
    gt_routing     : bool — CF pool partitioned by ground-truth label of training neighbours
    distance       : str  — Distance metric for CF neighbour search ('l1', 'l2', 'cosine')
    balance        : bool — Subsample each TRAIN fold to balance C0's TP/FP/TN/FN quadrants.
                            Test folds keep natural prevalence, so reported AUCs stay
                            comparable to unbalanced runs.
    balance_ratio  : float — Correct-per-error ratio kept when balancing (1.0 = 1:1:1:1)
    """

    print(f"\n{'='*70}")
    print(f"  {N_FOLDS}-FOLD CV: {disease.upper()} | CF={cf_count} | BACKBONE={backbone} | DISTANCE={distance}")
    print(f"{'='*70}\n")

    # ── Setup paths ───────────────────────────────────────────────────────
    results_base = os.path.join(RESULTS_DIR, '')
    data_subdir  = BACKBONE_MAP[backbone]                          # e.g. C2_resnet50
    if correct_cf and extended:
        cv_subdir = 'cv_results_extended_correct_cf'
    elif correct_cf:
        cv_subdir = 'cv_results_correct_cf'
    elif gt_routing:
        cv_subdir = 'cv_results_gt_routing'
    elif unmatched:
        cv_subdir = 'cv_results_unmatched'
    elif extended:
        cv_subdir = 'cv_results_extended'
    else:
        cv_subdir = 'cv_results'

    cv_dir       = os.path.join(results_base, f'{data_subdir}_corrected/{disease}/{cv_subdir}/cf_{cf_count}/test_entropy')
    cv_dir       = os.path.join(cv_dir, f'k_offset_{k_offset}') if k_offset > 0 else cv_dir
    cv_dir       = os.path.join(cv_dir, f'distance_{distance}') if distance != 'l1' else cv_dir
    # Balanced runs land in their own subtree so they never overwrite existing results
    cv_dir       = os.path.join(cv_dir, f'balanced_r{balance_ratio:g}') if balance else cv_dir

    fold_data_dir = os.path.join(cv_dir, 'fold_data')
    plots_dir    = os.path.join(cv_dir, 'cv_plots')

    os.makedirs(cv_dir, exist_ok=True)
    os.makedirs(fold_data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    data_csv = 'c2_data_extended.csv' if extended else 'c2_data.csv'
    full_df = pd.read_csv(os.path.join(results_base, f'{data_subdir}/{disease}/{data_csv}'))

    full_df.rename(columns={
        'prob': f'{disease}_prob',
        'pred': f'{disease}_pred',
        'true': f'{disease}_true'
    }, inplace=True)

    print(f"Backbone:  {backbone}  →  {data_subdir}/c2_data.csv")
    print(f"CV Subdir: {cv_subdir}")
    print(f"CF Count:  {cf_count}")
    print(f"Save Fold Data: {save_fold_data}")
    print(f"Use Unmatched CFs: {unmatched}")
    print(f"Use Correct-CF pools: {correct_cf}")
    print(f"Use GT-routing pools: {gt_routing}")
    print(f"K-offset for CFs: {k_offset}")
    print(f"Balance train folds: {balance}" + (f" (ratio={balance_ratio:g})" if balance else "") + "\n")
    print(f"Full dataset: {len(full_df):,} samples")
    print(f"  Correct:   {(full_df['correct']==1).sum():,}")
    print(f"  Incorrect: {(full_df['correct']==0).sum():,}\n")

    y = full_df['correct'].values

    # ── Initialize CV splitter ────────────────────────────────────────────
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    groups = full_df['patient_id'].values
    print(f"StratifiedGroupKFold: {N_FOLDS} folds, shuffle=True, random_state={RANDOM_SEED}\n")

    # ── Storage for results, this creates the column names ───────────────────────────────────────────────
    cv_results = {
        model_type: {config: [] for config in CONFIGS}
        for model_type in ['LR', 'RF', 'MLP']
    }
    cv_results['cf_anchor'] = []

    # ── FOLD LOOP ─────────────────────────────────────────────────────────
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, y, groups=groups)):

        print(f"\n{'-'*70}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'-'*70}")

        fold_train = full_df.iloc[train_idx].reset_index(drop=True)
        fold_test  = full_df.iloc[test_idx].reset_index(drop=True)

        print(f"  Train: {len(fold_train):,}  |  Test: {len(fold_test):,}")
        overlap = set(fold_train['patient_id']) & set(fold_test['patient_id'])
        print(f"  Patient overlap train/test: {len(overlap)} (should be 0)")

        # ── Balance the TRAIN partition only (test keeps natural prevalence) ──
        if balance:
            fold_train = balance_quadrants(
                fold_train, disease=disease, ratio=balance_ratio,
                seed=RANDOM_SEED + fold_idx
            )

        # ── Generate counterfactuals for this fold ────────────────────────
        if k_offset > 0:
            fold_train_cf, fold_test_cf, _ = compute_cf_for_split_further(
                train_df=fold_train, test_df=fold_test,
                cf_count=cf_count, disease=disease,
                distance=distance, k_offset=k_offset
            )
        elif correct_cf:
            fold_train_cf, fold_test_cf, _ = compute_cf_for_split_correct_cf(
                train_df=fold_train, test_df=fold_test,
                cf_count=cf_count, disease=disease, distance=distance
            )
        elif gt_routing:
            fold_train_cf, fold_test_cf, _ = compute_cf_for_split_gt_routing(
                train_df=fold_train, test_df=fold_test,
                cf_count=cf_count, disease=disease, distance=distance
            )
        elif unmatched:
            fold_train_cf, fold_test_cf, _ = compute_cf_for_split_unmatched(
                train_df=fold_train, test_df=fold_test,
                cf_count=cf_count, disease=disease, distance=distance
            )
        else:
            fold_train_cf, fold_test_cf, _ = compute_cf_for_split_matched_train_unmatched_test(
                train_df=fold_train, test_df=fold_test,
                cf_count=cf_count, disease=disease, distance=distance
            )
        print(f"  Generating {cf_count} counterfactuals...")


        # ── Attach CF neighbour attrs/emb ─────────────────────────────────
        if unmatched or correct_cf or gt_routing or k_offset > 0:
            _meta_cf = {f'{disease}_prob', f'{disease}_pred', f'{disease}_true',
                        'correct', 'path', 'cam_path', 'patient_id', 'cf_prob', 'cf_paths'}
            _emb_cols_cf = [c for c in fold_train_cf.columns if c.startswith('emb_')]
            _rel_cols_cf = [c for c in fold_train_cf.columns
                            if c not in _meta_cf
                            and not c.startswith('emb_')
                            and not c.startswith('delta_')]
            # CF paths always point into train; same lookup for both train and test
            _train_lookup = fold_train_cf
            fold_train_cf = attach_cf_features(fold_train_cf, _train_lookup, _rel_cols_cf, _emb_cols_cf, disease)
            fold_test_cf  = attach_cf_features(fold_test_cf,  _train_lookup, _rel_cols_cf, _emb_cols_cf, disease)

        if save_fold_data:
            fold_train_cf.to_csv(os.path.join(fold_data_dir, f'fold_{fold_idx}_train.csv'), index=False)
            fold_test_cf.to_csv( os.path.join(fold_data_dir, f'fold_{fold_idx}_test.csv'),  index=False)


        # ── Build feature matrices ────────────────────────────────────────
        feature_sets = build_feature_matrices(fold_train_cf, fold_test_cf, disease)

        y_train = fold_train_cf['correct'].values
        y_test  = fold_test_cf['correct'].values

        fold_pred_df = fold_test_cf.copy()

        # ── CF-anchor baseline ────────────────────────────────────────────
        cf_anchor_scores = score_cf_anchor(fold_train_cf, fold_test_cf)
        cf_anchor_auc = float(roc_auc_score(y_test, cf_anchor_scores))
        print(f"  CF-Anchor baseline AUC: {cf_anchor_auc:.4f}")
        cv_results['cf_anchor'].append(cf_anchor_auc)

        # ── Train all models on all configs ───────────────────────────────
        for model_type in ['LR', 'RF', 'MLP']:
        #for model_type in ['LR']:
            print(f"\n  {model_type}:")

            model_dir = os.path.join(cv_dir, 'models', model_type)
            os.makedirs(model_dir, exist_ok=True)

            for config in CONFIGS:
                if config not in feature_sets:
                    continue  # MCF configs absent when attach_cf_features was not called

                X_train, X_test = feature_sets[config]

                res = train_model(X_train, X_test, y_train, y_test, model_type=model_type)

                # Save model
                model_path = os.path.join(model_dir, f"{model_type}_{config}_fold{fold_idx}.pkl")
                if model_type in ['LR', 'MLP']:
                    joblib.dump({'model': res['model'], 'scaler': res['scaler']}, model_path)
                else:
                    joblib.dump(res['model'], model_path)

                cv_results[model_type][config].append({
                    'auc': res['auc'],
                    'fpr': res['fpr'],
                    'tpr': res['tpr'],
                    'y_prob': res['y_prob'],
                    'y_true': res['y_true']
                })
                print(f"    {config}: AUC = {res['auc']:.4f}")

                fold_pred_df[f"{model_type}_{config}_prob"] = res['y_prob']

        pred_csv_path = os.path.join(cv_dir, f'fold_{fold_idx}_predictions.csv')
        fold_pred_df.to_csv(pred_csv_path, index=False)
        print(f"Fold {fold_idx} predictions saved to: {pred_csv_path}")

    # ── AGGREGATE RESULTS ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATING RESULTS")
    print(f"{'='*70}\n")

    summary_rows = []
    for model_type in ['LR', 'RF', 'MLP']:
    #for model_type in ['LR']:
        for config in CONFIGS:
            aucs = [fold_res['auc'] for fold_res in cv_results[model_type][config]]
            if not aucs:
                continue  # config was skipped (e.g. MCF on matched path)
            summary_rows.append({
                'model':     model_type,
                'config':    config,
                'mean_auc':  np.mean(aucs),
                'std_auc':   np.std(aucs),
                'min_auc':   np.min(aucs),
                'max_auc':   np.max(aucs),
                'fold_aucs': ','.join([f'{a:.4f}' for a in aucs])
            })

    # Save the KNN baselines too
    aucs = cv_results['cf_anchor']
    summary_rows.append({
        'model': 'cf_anchor', 'config': 'cf_anchor',
        'mean_auc': np.mean(aucs), 'std_auc': np.std(aucs),
        'fold_aucs': ','.join([f'{a:.4f}' for a in aucs])
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(cv_dir, 'cv_summary.csv'), index=False)

    with open(os.path.join(cv_dir, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {cv_dir}")

    return cv_results, summary_df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    if any('jupyter' in arg or 'ipykernel' in arg for arg in sys.argv):
        run_cv(disease='effusion', cf_count=16, save_fold_data=False,
               unmatched=False, correct_cf=False, backbone='densenet', k_offset=0)
    else:
        parser = argparse.ArgumentParser(description='Run C2 cross-validation')
        parser.add_argument('--disease',  type=str, default='effusion')
        parser.add_argument('--cf_count', type=int, default=16)
        parser.add_argument('--backbone', type=str, default='densenet',
                            choices=['densenet', 'resnet50', 'vit', 'medmnist'])
        parser.add_argument('--save_folds', action='store_true')
        parser.add_argument('--unmatched',  action='store_true')
        parser.add_argument('--correct_cf', action='store_true')
        parser.add_argument('--gt_routing', action='store_true',
                            help='CF pool partitioned by ground-truth label of training neighbours')
        parser.add_argument('--k_offset', type=int, default=0)
        parser.add_argument('--extended',   action='store_true',
                            help='Use c2_data_extended.csv (adds GLCM radiomics features)')
        parser.add_argument('--distance', type=str, default='l1',
                            choices=['l1', 'l2', 'cosine'],
                            help='Distance metric for CF neighbour search')
        parser.add_argument('--balance', action='store_true',
                            help='Subsample each TRAIN fold to balance C0 TP/FP/TN/FN quadrants '
                                 '(test folds keep natural prevalence)')
        parser.add_argument('--balance_ratio', type=float, default=1.0,
                            help='Correct-per-error ratio when balancing (1.0 = fully balanced 1:1:1:1)')
        args = parser.parse_args()
        run_cv(
            disease=args.disease,
            cf_count=args.cf_count,
            save_fold_data=args.save_folds,
            unmatched=args.unmatched,
            correct_cf=args.correct_cf,
            gt_routing=args.gt_routing,
            backbone=args.backbone,
            k_offset=args.k_offset,
            extended=args.extended,
            distance=args.distance,
            balance=args.balance,
            balance_ratio=args.balance_ratio,
        )