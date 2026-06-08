"""
c2_cv_pipeline_new_split.py

5-fold cross-validation pipeline for C2 quality control models
with support for simulated counterfactuals (CFs).

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

Usage:
    python c2_cv_pipeline_new_split.py --disease effusion --cf_count 16 --backbone resnet50

Parameters:
    --disease   : Disease name (e.g., 'effusion')
    --cf_count  : Number of counterfactuals (k)
    --backbone  : C0 architecture ('densenet', 'resnet50', 'vit')
    --save_folds: Save individual fold CSVs to disk
    --unmatched : Use unmatched CF pool (ablation)
    --k_offset   : Integer offset for CF selection (e.g., k_offset=3 uses the 4th-6th nearest CFs instead of 1st-3rd)
"""
import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# Import the refactored CF computation function
sys.path.append('/zhome/d0/a/221493/thesis/code')
from c2_prepare_data_simulated_cf import compute_cf_for_split, compute_cf_for_split_unmatched, compute_cf_for_split_further # added the new one for the "further" cfs on the disease absent
# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR    = '/zhome/d0/a/221493/thesis/'
N_FOLDS     = 5
RANDOM_SEED = 42
CONFIGS     = ['B1','B2','B3','B4','B5','M1','M2','M3','M4','M5','M6']

# Map backbone name → results subdirectory
BACKBONE_MAP = {
    'densenet': 'C2_custom',
    'resnet50': 'C2_resnet50',
    'vit':      'C2_vit',
}

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, X_test, y_train, y_test, model_type='LR'):
    """Train a single model type and return test predictions + metrics + model/scaler."""

    if model_type == 'LR':
        scaler = StandardScaler()
        model  = LogisticRegression(max_iter=5000, class_weight='balanced',
                                    random_state=RANDOM_SEED)
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
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                       random_state=RANDOM_SEED, n_jobs=-1)
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
        model  = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                               early_stopping=True, validation_fraction=0.05,
                               random_state=RANDOM_SEED)
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
# FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrices(train_df, test_df, disease):
    """Extract all feature matrices for all configs from DataFrames."""

    meta_cols = {f'{disease}_prob', f'{disease}_pred', f'{disease}_true',
                 'correct', 'path', 'patient_id', 'cf_prob', 'cf_paths'}
    diff_cols = [c for c in train_df.columns if c.startswith('delta_')]
    emb_cols  = [c for c in train_df.columns if c.startswith('emb_')]
    attr_cols = [c for c in train_df.columns
                 if c not in meta_cols
                 and not c.startswith('delta_')
                 and not c.startswith('emb_')]

    prob_tr,    prob_te    = train_df[[f'{disease}_prob']].values, test_df[[f'{disease}_prob']].values
    attr_tr,    attr_te    = train_df[attr_cols].values,           test_df[attr_cols].values
    diff_tr,    diff_te    = train_df[diff_cols].values,           test_df[diff_cols].values
    emb_tr,     emb_te     = train_df[emb_cols].values,            test_df[emb_cols].values
    cf_prob_tr, cf_prob_te = train_df[['cf_prob']].values,         test_df[['cf_prob']].values

    return {
        'B1': (prob_tr, prob_te),
        'B2': (attr_tr, attr_te),
        'B3': (emb_tr,  emb_te),
        'B4': (np.hstack([prob_tr, attr_tr]),              np.hstack([prob_te, attr_te])),
        'B5': (np.hstack([prob_tr, emb_tr]),               np.hstack([prob_te, emb_te])),
        'M1': (diff_tr, diff_te),
        'M2': (np.hstack([prob_tr, diff_tr]),              np.hstack([prob_te, diff_te])),
        'M3': (np.hstack([prob_tr, diff_tr, attr_tr]),     np.hstack([prob_te, diff_te, attr_te])),
        'M4': (np.hstack([prob_tr, diff_tr, emb_tr]),      np.hstack([prob_te, diff_te, emb_te])),
        'M5': (np.hstack([prob_tr, diff_tr, attr_tr, emb_tr]), np.hstack([prob_te, diff_te, attr_te, emb_te])),
        'M6': (np.hstack([prob_tr, diff_tr, attr_tr, cf_prob_tr]), np.hstack([prob_te, diff_te, attr_te, cf_prob_te]))
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CV PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(disease, cf_count, save_fold_data=False, unmatched=False, backbone='densenet', k_offset=0):
    """
    Run complete 5-fold CV pipeline.

    Parameters
    ----------
    disease      : str  — Disease name (e.g., 'effusion')
    cf_count     : int  — Number of counterfactual neighbours (k)
    save_fold_data: bool — Save per-fold train/test CSVs to disk
    unmatched    : bool — Use unmatched CF pool (ablation)
    backbone     : str  — C0 architecture ('densenet', 'resnet50', 'vit')
    """

    print(f"\n{'='*70}")
    print(f"  {N_FOLDS}-FOLD CV: {disease.upper()} | CF={cf_count} | BACKBONE={backbone}")
    print(f"{'='*70}\n")

    # ── Setup paths ───────────────────────────────────────────────────────
    results_base = os.path.join(BASE_DIR, 'results')
    data_subdir  = BACKBONE_MAP[backbone]                          # e.g. C2_resnet50
    cv_subdir    = 'cv_results_unmatched' if unmatched else 'cv_results'

    cv_dir       = os.path.join(results_base, f'{data_subdir}/{disease}/{cv_subdir}/cf_{cf_count}')
    cv_dir       = os.path.join(cv_dir, f'k_offset_{k_offset}') if k_offset > 0 else cv_dir
    
    fold_data_dir = os.path.join(cv_dir, 'fold_data')
    plots_dir    = os.path.join(cv_dir, 'cv_plots')

    os.makedirs(cv_dir, exist_ok=True)
    os.makedirs(fold_data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    full_df = pd.read_csv(os.path.join(results_base, f'{data_subdir}/{disease}/c2_data.csv'))

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
    print(f"K-offset for CFs: {k_offset}\n")
    print(f"Full dataset: {len(full_df):,} samples")
    print(f"  Correct:   {(full_df['correct']==1).sum():,}")
    print(f"  Incorrect: {(full_df['correct']==0).sum():,}\n")

    y = full_df['correct'].values

    # ── Initialize CV splitter ────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # ── Storage for results ───────────────────────────────────────────────
    cv_results = {
        model_type: {config: [] for config in CONFIGS}
        for model_type in ['LR', 'RF', 'MLP']
    }

    # ── FOLD LOOP ─────────────────────────────────────────────────────────
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, y)):

        print(f"\n{'-'*70}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'-'*70}")

        fold_train = full_df.iloc[train_idx].reset_index(drop=True)
        fold_test  = full_df.iloc[test_idx].reset_index(drop=True)

        print(f"  Train: {len(fold_train):,}  |  Test: {len(fold_test):,}")

        # ── Generate counterfactuals for this fold ────────────────────────
        if k_offset > 0:
            cf_fn = compute_cf_for_split_further
            fold_train_cf, fold_test_cf, _ = cf_fn(
                train_df=fold_train, test_df=fold_test,
                cf_count=cf_count, disease=disease,
                distance='l1', k_offset=k_offset
            )
        elif unmatched:
            cf_fn = compute_cf_for_split_unmatched
            fold_train_cf, fold_test_cf, _ = cf_fn(
                train_df=fold_train, test_df=fold_test,
                cf_count=cf_count, disease=disease, distance='l1'
            )
        else:
            cf_fn = compute_cf_for_split
            fold_train_cf, fold_test_cf, _ = cf_fn(
                train_df=fold_train, test_df=fold_test,
                cf_count=cf_count, disease=disease, distance='l1'
            )
        print(f"  Generating {cf_count} counterfactuals...")


        if save_fold_data:
            fold_train_cf.to_csv(os.path.join(fold_data_dir, f'fold_{fold_idx}_train.csv'), index=False)
            fold_test_cf.to_csv( os.path.join(fold_data_dir, f'fold_{fold_idx}_test.csv'),  index=False)

        # ── Build feature matrices ────────────────────────────────────────
        feature_sets = build_feature_matrices(fold_train_cf, fold_test_cf, disease)

        y_train = fold_train_cf['correct'].values
        y_test  = fold_test_cf['correct'].values

        fold_pred_df = fold_test_cf.copy()

        # ── Train all models on all configs ───────────────────────────────
        #for model_type in ['LR', 'RF', 'MLP']:
        for model_type in ['LR']:
            print(f"\n  {model_type}:")

            model_dir = os.path.join(cv_dir, 'models', model_type)
            os.makedirs(model_dir, exist_ok=True)

            for config in CONFIGS:
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
    #for model_type in ['LR', 'RF', 'MLP']:
    for model_type in ['LR']:
        for config in CONFIGS:
            aucs = [fold_res['auc'] for fold_res in cv_results[model_type][config]]
            summary_rows.append({
                'model':     model_type,
                'config':    config,
                'mean_auc':  np.mean(aucs),
                'std_auc':   np.std(aucs),
                'min_auc':   np.min(aucs),
                'max_auc':   np.max(aucs),
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
        run_cv(disease='effusion', cf_count=1, save_fold_data=False,
               unmatched=False, backbone='densenet', k_offset=0)
    else:
        parser = argparse.ArgumentParser(description='Run C2 cross-validation')
        parser.add_argument('--disease',  type=str, default='effusion')
        parser.add_argument('--cf_count', type=int, default=1)
        parser.add_argument('--backbone', type=str, default='densenet',
                            choices=['densenet', 'resnet50', 'vit'])
        parser.add_argument('--save_folds', action='store_true')
        parser.add_argument('--unmatched', action='store_true')
        parser.add_argument('--k_offset', type=int, default=0)
        args = parser.parse_args()
        run_cv(
            disease=args.disease,
            cf_count=args.cf_count,
            save_fold_data=args.save_folds,
            unmatched=args.unmatched,
            backbone=args.backbone,
            k_offset=args.k_offset
        )