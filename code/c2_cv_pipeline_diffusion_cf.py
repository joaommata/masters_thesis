"""
c2_cv_pipeline_diffusion_cf.py
==============================
Same as c2_cv_pipeline_new_split.py but uses real diffusion CFs
instead of simulated nearest-neighbour CFs.
"""
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

sys.path.append('/zhome/d0/a/221493/thesis/code')
from c2_prepare_data_simulated_cf import add_clinical_ratios  # NEW — reuse existing function

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR           = '/zhome/d0/a/221493/thesis/'
CF_ATTRIBUTES_PATH = '/zhome/d0/a/221493/thesis/results/diffusion_cf/cf_attributes.csv'  # NEW
MANIFEST_PATH      = '/zhome/d0/a/221493/thesis/results/diffusion_cf/cf_manifest.csv'    # NEW
N_FOLDS            = 5
RANDOM_SEED        = 42
CONFIGS = ['B1','B2','B3','B4','B5','M1','M2','M3','M4','M5','M6']
THRESHOLD = 0.5634  # C0 Youden-optimal threshold from training

# ══════════════════════════════════════════════════════════════════════════════
# NEW — CF INTEGRATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def attach_diffusion_cfs(df, cf_attrs, manifest, disease):
    """
    Merges pre-computed diffusion CF attributes onto a dataframe and
    computes delta vectors. Mirrors what compute_cf_for_split() does
    for the simulated case.
    
    Parameters
    ----------
    df         : original split dataframe (train or test fold)
    cf_attrs   : cf_attributes.csv loaded as DataFrame
    manifest   : cf_manifest.csv loaded as DataFrame (has path -> cf_path mapping)
    disease    : disease name string
    
    Returns
    -------
    df with delta_ columns and cf_prob column added
    """
    
    # Add clinical ratios to originals and CFs — same as simulated pipeline
    df       = add_clinical_ratios(df.copy())
    cf_attrs = add_clinical_ratios(cf_attrs.copy())
    
    # Identify attribute columns — exclude meta, embeddings, deltas
    meta_cols = {f'{disease}_prob', f'{disease}_pred', f'{disease}_true',
                 'correct', 'path', 'patient_id', 'cf_prob', 'cf_paths'}
    attr_cols = [c for c in df.columns
                 if c not in meta_cols
                 and not c.startswith('emb_')
                 and not c.startswith('delta_')]


    # Fill NaNs — same as compute_cf_for_split()
    df[attr_cols]       = df[attr_cols].fillna(0)
    cf_attrs[attr_cols] = cf_attrs[attr_cols].fillna(0)
    
    # Step 1: merge manifest to get cf_path and cf_prob for each original path
    df = df.merge(
        manifest[['path', 'cf_path', 'cf_prob']],
        on='path',
        how='left')
    
    # Step 2: merge CF attributes using cf_path as key
    df = df.merge(
        cf_attrs[['cf_path'] + attr_cols],
        on='cf_path',
        how='left',
        suffixes=('', '_cf')
    )
    
    # Step 3: compute delta = original_attr - cf_attr
    for col in attr_cols:
        cf_col = f'{col}_cf'
        if cf_col in df.columns:
            df[f'delta_{col}'] = df[col] - df[cf_col]
        else:
            df[f'delta_{col}'] = 0.0
    
    # Step 4: drop the raw CF attribute columns, keep only deltas
    cf_only_cols = [f'{col}_cf' for col in attr_cols] + ['cf_path']
    df = df.drop(columns=[c for c in cf_only_cols if c in df.columns])
    
    # Warn if any samples are missing a CF
    n_missing = df['cf_prob'].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} samples have no CF match — dropping them")
        df = df.dropna(subset=['cf_prob']).reset_index(drop=True)
    
    # Fill NaNs — same as simulated pipeline
    delta_cols = [c for c in df.columns if c.startswith('delta_')]
    df[delta_cols] = df[delta_cols].fillna(0)
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, X_test, y_train, y_test, model_type='LR'):
    if model_type == 'LR':
        scaler = StandardScaler()
        model  = LogisticRegression(max_iter=5000, class_weight='balanced',
                                    random_state=RANDOM_SEED)
        model.fit(scaler.fit_transform(X_train), y_train)
        y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]
        return {'auc': float(roc_auc_score(y_test, y_prob)),
                'fpr': roc_curve(y_test, y_prob)[0].tolist(),
                'tpr': roc_curve(y_test, y_prob)[1].tolist(),
                'y_prob': y_prob.tolist(), 'y_true': y_test.tolist(),
                'model': model, 'scaler': scaler}
    elif model_type == 'RF':
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                       random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        return {'auc': float(roc_auc_score(y_test, y_prob)),
                'fpr': roc_curve(y_test, y_prob)[0].tolist(),
                'tpr': roc_curve(y_test, y_prob)[1].tolist(),
                'y_prob': y_prob.tolist(), 'y_true': y_test.tolist(),
                'model': model}
    elif model_type == 'MLP':
        scaler = StandardScaler()
        model  = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                               early_stopping=True, validation_fraction=0.05,
                               random_state=RANDOM_SEED)
        model.fit(scaler.fit_transform(X_train.astype(np.float32)), y_train)
        y_prob = model.predict_proba(scaler.transform(X_test.astype(np.float32)))[:, 1]
        return {'auc': float(roc_auc_score(y_test, y_prob)),
                'fpr': roc_curve(y_test, y_prob)[0].tolist(),
                'tpr': roc_curve(y_test, y_prob)[1].tolist(),
                'y_prob': y_prob.tolist(), 'y_true': y_test.tolist(),
                'model': model, 'scaler': scaler}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE MATRIX BUILDER (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrices(train_df, test_df, disease):
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
        'B3': (emb_tr, emb_te),
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

def run_cv(disease):
    print(f"\n{'='*70}")
    print(f"  {N_FOLDS}-FOLD CV (DIFFUSION CF): {disease.upper()}")
    print(f"{'='*70}\n")

    results_base = os.path.join(BASE_DIR, 'results')
    cv_dir       = os.path.join(results_base, f'C2_diffusion/{disease}/cv_results')
    os.makedirs(cv_dir, exist_ok=True)

    # Load data — NEW: also load CF attributes and manifest once, outside fold loop
    full_df  = pd.read_csv(os.path.join(results_base, 'C2_custom/c2_data.csv'))
    cf_attrs = pd.read_csv(CF_ATTRIBUTES_PATH)   # NEW
    manifest = pd.read_csv(MANIFEST_PATH)         # NEW

    # Add margin to cf_attrs using cf_prob from manifest
    cf_attrs = cf_attrs.merge(manifest[['cf_path', 'cf_prob']], on='cf_path', how='left')
    cf_attrs['margin'] = np.abs(cf_attrs['cf_prob'] - THRESHOLD)
    
    # Rename C0 prediction columns to match expected format
    full_df.rename(columns={
        'prob': f'{disease}_prob',
        'pred': f'{disease}_pred',
        'true': f'{disease}_true'
    }, inplace=True)

    # Keep only samples that have a CF in the manifest  # NEW
    full_df = full_df[full_df['path'].isin(manifest['path'])].reset_index(drop=True)
    print(f"Samples with diffusion CF: {len(full_df):,}")
    print(f"  Correct:   {(full_df['correct']==1).sum():,}")
    print(f"  Incorrect: {(full_df['correct']==0).sum():,}\n")

    y   = full_df['correct'].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    cv_results = {m: {c: [] for c in CONFIGS} for m in ['LR', 'RF', 'MLP']}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, y)):
        print(f"\n{'-'*70}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'-'*70}")

        fold_train = full_df.iloc[train_idx].reset_index(drop=True)
        fold_test  = full_df.iloc[test_idx].reset_index(drop=True)

        # NEW — replace compute_cf_for_split() with diffusion version
        print("  Attaching diffusion CFs and computing deltas...")
        fold_train_cf = attach_diffusion_cfs(fold_train, cf_attrs, manifest, disease)
        fold_test_cf  = attach_diffusion_cfs(fold_test,  cf_attrs, manifest, disease)

        print(f"  Train: {len(fold_train_cf):,}  |  Test: {len(fold_test_cf):,}")

        print("NaN counts per column group:")
        print(f"  prob:    {fold_train_cf[f'{disease}_prob'].isna().sum()}")
        print(f"  cf_prob: {fold_train_cf['cf_prob'].isna().sum()}")
        print(f"  deltas:  {fold_train_cf[[c for c in fold_train_cf.columns if c.startswith('delta_')]].isna().sum().sum()}")
        print(f"  other:   {fold_train_cf.isna().sum().sum()}")

        feature_sets = build_feature_matrices(fold_train_cf, fold_test_cf, disease)
        y_train      = fold_train_cf['correct'].values
        y_test       = fold_test_cf['correct'].values
        fold_pred_df = fold_test_cf.copy()

        for model_type in ['LR', 'RF', 'MLP']:
            print(f"\n  {model_type}:")
            model_dir = os.path.join(cv_dir, 'models', model_type)
            os.makedirs(model_dir, exist_ok=True)

            for config in CONFIGS:
                X_train, X_test = feature_sets[config]
                res = train_model(X_train, X_test, y_train, y_test, model_type=model_type)

                joblib.dump(
                    {'model': res['model'], 'scaler': res.get('scaler')},
                    os.path.join(model_dir, f"{model_type}_{config}_fold{fold_idx}.pkl")
                )

                cv_results[model_type][config].append({
                    'auc': res['auc'], 'fpr': res['fpr'],
                    'tpr': res['tpr'], 'y_prob': res['y_prob'],
                    'y_true': res['y_true']
                })
                print(f"    {config}: AUC = {res['auc']:.4f}")

                fold_pred_df[f"{model_type}_{config}_prob"] = res['y_prob']

        fold_pred_df.to_csv(os.path.join(cv_dir, f'fold_{fold_idx}_predictions.csv'), index=False)

    # Aggregate and save
    summary_rows = []
    for model_type in ['LR', 'RF', 'MLP']:
        for config in CONFIGS:
            aucs = [f['auc'] for f in cv_results[model_type][config]]
            summary_rows.append({
                'model': model_type, 'config': config,
                'mean_auc': np.mean(aucs), 'std_auc': np.std(aucs),
                'fold_aucs': ','.join([f'{a:.4f}' for a in aucs])
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(cv_dir, 'cv_summary.csv'), index=False)
    with open(os.path.join(cv_dir, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    print(f"\n{'='*70}")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {cv_dir}")

    return cv_results, summary_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease', type=str, default='effusion')
    args = parser.parse_args()
    run_cv(disease=args.disease)