#!/usr/bin/env python
# c2_cv_pipeline_subsampled.py
# ───────────────────────────────────────────────
# 5-fold cross-validation pipeline for C2 quality control models
# with support for simulated counterfactuals (CFs)
# and optional subsampling
# ───────────────────────────────────────────────

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# Import CF computation function
sys.path.append('/zhome/d0/a/221493/thesis/code')
from c2_prepare_data_simulated_cf import compute_cf_for_split, compute_cf_for_split_unmatched

# ────────────────────────────── CONFIG ──────────────────────────────
BASE_DIR    = '/zhome/d0/a/221493/thesis/'
N_FOLDS     = 5
RANDOM_SEED = 42
CONFIGS = ['B1','B2','B3','B4','B5','M1','M2','M3','M4','M5','M6']

# ────────────────────────────── TRAINING FUNCTIONS ──────────────────────────────
def train_model(X_train, X_test, y_train, y_test, model_type='LR'):
    if model_type == 'LR':
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=RANDOM_SEED)
        model.fit(scaler.fit_transform(X_train), y_train)
        y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]
        return {'auc': float(roc_auc_score(y_test, y_prob)),
                'fpr': roc_curve(y_test, y_prob)[0].tolist(),
                'tpr': roc_curve(y_test, y_prob)[1].tolist(),
                'y_prob': y_prob.tolist(),
                'y_true': y_test.tolist(),
                'model': model,
                'scaler': scaler}
    elif model_type == 'RF':
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        return {'auc': float(roc_auc_score(y_test, y_prob)),
                'fpr': roc_curve(y_test, y_prob)[0].tolist(),
                'tpr': roc_curve(y_test, y_prob)[1].tolist(),
                'y_prob': y_prob.tolist(),
                'y_true': y_test.tolist(),
                'model': model}
    elif model_type == 'MLP':
        scaler = StandardScaler()
        model = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500, early_stopping=True, validation_fraction=0.05, random_state=RANDOM_SEED)
        X_train_f32 = X_train.astype(np.float32)
        X_test_f32  = X_test.astype(np.float32)
        model.fit(scaler.fit_transform(X_train_f32), y_train)
        y_prob = model.predict_proba(scaler.transform(X_test_f32))[:, 1]
        return {'auc': float(roc_auc_score(y_test, y_prob)),
                'fpr': roc_curve(y_test, y_prob)[0].tolist(),
                'tpr': roc_curve(y_test, y_prob)[1].tolist(),
                'y_prob': y_prob.tolist(),
                'y_true': y_test.tolist(),
                'model': model,
                'scaler': scaler}
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ────────────────────────────── FEATURE MATRIX BUILDER ──────────────────────────────
def build_feature_matrices(train_df, test_df, disease):
    meta_cols = {f'{disease}_prob', f'{disease}_pred', f'{disease}_true', 'correct', 'path', 'patient_id', 'cf_prob', 'cf_paths'}
    diff_cols = [c for c in train_df.columns if c.startswith('delta_')]
    emb_cols  = [c for c in train_df.columns if c.startswith('emb_')]
    attr_cols = [c for c in train_df.columns if c not in meta_cols and not c.startswith('delta_') and not c.startswith('emb_')]

    prob_tr, prob_te = train_df[[f'{disease}_prob']].values, test_df[[f'{disease}_prob']].values
    attr_tr, attr_te = train_df[attr_cols].values, test_df[attr_cols].values
    diff_tr, diff_te = train_df[diff_cols].values, test_df[diff_cols].values
    emb_tr, emb_te   = train_df[emb_cols].values, test_df[emb_cols].values
    cf_prob_tr, cf_prob_te = train_df[['cf_prob']].values, test_df[['cf_prob']].values

    return {
        'B1': (prob_tr, prob_te),
        'B2': (attr_tr, attr_te),
        'B3': (emb_tr, emb_te),
        'B4': (np.hstack([prob_tr, attr_tr]), np.hstack([prob_te, attr_te])),
        'B5': (np.hstack([prob_tr, emb_tr]), np.hstack([prob_te, emb_te])),
        'M1': (diff_tr, diff_te),
        'M2': (np.hstack([prob_tr, diff_tr]), np.hstack([prob_te, diff_te])),
        'M3': (np.hstack([prob_tr, diff_tr, attr_tr]), np.hstack([prob_te, diff_te, attr_te])),
        'M4': (np.hstack([prob_tr, diff_tr, emb_tr]), np.hstack([prob_te, diff_te, emb_te])),
        'M5': (np.hstack([prob_tr, diff_tr, attr_tr, emb_tr]), np.hstack([prob_te, diff_te, attr_te, emb_te])),
        'M6': (np.hstack([prob_tr, diff_tr, attr_tr, cf_prob_tr]), np.hstack([prob_te, diff_te, attr_te, cf_prob_te]))
    }

# ────────────────────────────── MAIN CV PIPELINE ──────────────────────────────
def run_cv(disease, cf_count, save_fold_data=True, unmatched=False, subsample_size=None):
    print(f"\n{'='*70}")
    print(f"  {N_FOLDS}-FOLD CV: {disease.upper()} | CF={cf_count} | SUBSAMPLE={subsample_size}")
    print(f"{'='*70}\n")

    results_base = os.path.join(BASE_DIR, 'results')
    cv_subdir = 'cv_results_unmatched' if unmatched else 'cv_results'

    # Append subsample size to folder name
    subsample_tag = f"_sub{subsample_size}" if subsample_size else ""
    cv_dir = os.path.join(results_base, f'C2_sim_cf/{disease}/{cv_subdir}/cf_{cf_count}{subsample_tag}')
    fold_data_dir = os.path.join(cv_dir, 'fold_data')
    plots_dir     = os.path.join(cv_dir, 'cv_plots')
    os.makedirs(cv_dir, exist_ok=True)
    os.makedirs(fold_data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ── Load dataset ──
    c0_path = os.path.join(results_base, f'C0_baseline/{disease}/train_c0_{disease}.csv')
    c1_path = os.path.join(results_base, 'C1_attributes/train_c1_attribute_vector_rad.csv')
    c0_df = pd.read_csv(c0_path)
    c1_df = pd.read_csv(c1_path)
    full_df = c0_df.merge(c1_df, on='path', how='inner')

    # Subsample if requested
    if subsample_size:
        full_df = full_df.sample(n=subsample_size, random_state=RANDOM_SEED).reset_index(drop=True)

    # Rename columns
    full_df.rename(columns={'prob': f'{disease}_prob', 'pred': f'{disease}_pred', 'true': f'{disease}_true'}, inplace=True)

    print(f"Dataset: {len(full_df):,} samples (Correct: {(full_df['correct']==1).sum():,}, Incorrect: {(full_df['correct']==0).sum():,})\n")
    y = full_df['correct'].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = {m:{c:[] for c in CONFIGS} for m in ['LR','RF','MLP']}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, y)):
        print(f"\n{'-'*70}\nFOLD {fold_idx+1}/{N_FOLDS}\n{'-'*70}")
        fold_train = full_df.iloc[train_idx].reset_index(drop=True)
        fold_test  = full_df.iloc[test_idx].reset_index(drop=True)
        print(f"  Train: {len(fold_train):,} | Test: {len(fold_test):,}")

        # Generate counterfactuals
        print(f"  Generating {cf_count} counterfactuals...")
        cf_fn = compute_cf_for_split_unmatched if unmatched else compute_cf_for_split
        fold_train_cf, fold_test_cf, _ = cf_fn(train_df=fold_train, test_df=fold_test, cf_count=cf_count, disease=disease, distance='l1')

        if save_fold_data:
            fold_train_cf.to_csv(os.path.join(fold_data_dir, f'fold_{fold_idx}_train.csv'), index=False)
            fold_test_cf.to_csv(os.path.join(fold_data_dir, f'fold_{fold_idx}_test.csv'), index=False)

        feature_sets = build_feature_matrices(fold_train_cf, fold_test_cf, disease)
        y_train, y_test = fold_train_cf['correct'].values, fold_test_cf['correct'].values
        fold_pred_df = fold_test_cf.copy()

        for model_type in ['LR','RF','MLP']:
            print(f"\n  {model_type}:")
            model_dir = os.path.join(cv_dir, 'models', model_type)
            os.makedirs(model_dir, exist_ok=True)

            for config in CONFIGS:
                X_train, X_test = feature_sets[config]
                res = train_model(X_train, X_test, y_train, y_test, model_type=model_type)

                # Save model
                model_path = os.path.join(model_dir, f"{model_type}_{config}_fold{fold_idx}.pkl")
                if model_type in ['LR','MLP']:
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

                fold_pred_df[f"{model_type}_{config}_prob"] = res['y_prob']
                print(f"    {config}: AUC = {res['auc']:.4f}")

            # Save fold predictions
            fold_pred_df.to_csv(os.path.join(cv_dir, f'fold_{fold_idx}_predictions.csv'), index=False)
            print(f"Fold {fold_idx} predictions saved.")

    # Aggregate results
    summary_rows = []
    for model_type in ['LR','RF','MLP']:
        for config in CONFIGS:
            aucs = [f['auc'] for f in cv_results[model_type][config]]
            summary_rows.append({'model': model_type,
                                 'config': config,
                                 'mean_auc': np.mean(aucs),
                                 'std_auc': np.std(aucs),
                                 'min_auc': np.min(aucs),
                                 'max_auc': np.max(aucs),
                                 'fold_aucs': ','.join([f'{a:.4f}' for a in aucs])})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(cv_dir, 'cv_summary.csv'), index=False)
    with open(os.path.join(cv_dir, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {cv_dir}")
    return cv_results, summary_df

# ────────────────────────────── MAIN FUNCTION ──────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run C2 cross-validation')
    parser.add_argument('--disease', type=str, default='effusion', help='Disease to evaluate')
    parser.add_argument('--cf_count', type=int, default=1, help='Number of counterfactuals (k)')
    parser.add_argument('--no_save_folds', action='store_true', default=False, help='Skip saving individual fold CSVs')
    parser.add_argument('--unmatched', action='store_true', help='Use unmatched CF pools (ablation)')
    parser.add_argument('--subsample_size', type=int, default=None, help='Number of samples to use (optional)')
    args = parser.parse_args()

    run_cv(disease=args.disease,
           cf_count=args.cf_count,
           save_fold_data=not args.no_save_folds,
           unmatched=args.unmatched,
           subsample_size=args.subsample_size)