"""
c2_cv_pipeline_new_split.py

## CHANGED VERSION - 


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
    python c2_cv_pipeline_new_split.py --cf_count 5 --disease effusion

Parameters:
    --disease   : Disease name (e.g., 'effusion')
    --cf_count  : Number of counterfactuals (k)
    --no_save_folds : Skip saving individual fold CSVs (saves disk space)

Outputs:
    results/C2_custom/{disease}/cv_results/cf_{cf_count}/
        ├── fold_data/          # Cached fold splits with CFs per fold
        ├── models/             # Saved models for each config/fold
        ├── cv_summary.csv      # Mean ± std AUC per config/model
        ├── cv_detailed.json    # Full per-fold predictions and metrics
        └── cv_plots/           # Visualization of ROC curves etc. - TO DO
"""
import os
import sys
import json
import argparse
import joblib
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
from c2_prepare_data_simulated_cf import compute_cf_for_split, compute_cf_for_split_unmatched
# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR    = '/zhome/d0/a/221493/thesis/'
N_FOLDS     = 5
RANDOM_SEED = 42
CONFIGS = ['B1','B2','B3','B4','B5','M1','M2','M3','M4','M5','M6']

# Let's run only a few configs for testing: B1
#CONFIGS = ['B1', 'B2','B4', 'M3', 'M6']  # <-- TEMPORARY FOR TESTING
#N_FOLDS = 2  # <-- TEMPORARY FOR TESTING

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def train_model(X_train, X_test, y_train, y_test, model_type='LR'):
    """Train a single model type and return test predictions + metrics + model/scaler."""
    
    if model_type == 'LR':
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=5000, class_weight='balanced', 
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
            'model': model  # no scaler for RF
        }
    
    elif model_type == 'MLP':
        scaler = StandardScaler()
        model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                              early_stopping=True, validation_fraction=0.05, 
                              random_state=RANDOM_SEED)
        X_train_f32 = X_train.astype(np.float32)
        X_test_f32 = X_test.astype(np.float32)

        # Fit the MLP with sample weights
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
    """
    Extract all feature matrices for all configs from DataFrames.
    
    Returns:
        dict mapping config name → (X_train, X_test)
    """
    
    # Identify column groups
    meta_cols = {f'{disease}_prob', f'{disease}_pred', f'{disease}_true',
                 'correct', 'path', 'patient_id', 'cf_prob', 'cf_paths'}
    diff_cols = [c for c in train_df.columns if c.startswith('delta_')]
    emb_cols  = [c for c in train_df.columns if c.startswith('emb_')]
    attr_cols = [c for c in train_df.columns 
                 if c not in meta_cols 
                 and not c.startswith('delta_') 
                 and not c.startswith('emb_')]
    
    # Extract arrays
    prob_tr, prob_te = train_df[[f'{disease}_prob']].values, test_df[[f'{disease}_prob']].values
    attr_tr, attr_te = train_df[attr_cols].values, test_df[attr_cols].values
    diff_tr, diff_te = train_df[diff_cols].values, test_df[diff_cols].values
    emb_tr, emb_te   = train_df[emb_cols].values, test_df[emb_cols].values
    cf_prob_tr, cf_prob_te = train_df[['cf_prob']].values, test_df[['cf_prob']].values

    
    # Build all configs
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

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CV PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(disease, cf_count, save_fold_data=False, unmatched=False):    
    """
    Run complete 5-fold CV pipeline.
    
    Parameters
    ----------
    disease : str
        Disease name (e.g., 'effusion')
    cf_count : int
        Number of counterfactuals (k)
    save_fold_data : bool
        If True, save train/test CSVs per fold (useful for debugging)
    """
    
    print(f"\n{'='*70}")
    print(f"  {N_FOLDS}-FOLD CV: {disease.upper()} | CF={cf_count}")
    print(f"{'='*70}\n")
    
    
    # ── Setup paths ───────────────────────────────────────────────────────
    results_base = os.path.join(BASE_DIR, 'results')
    cv_subdir = 'cv_results_unmatched' if unmatched else 'cv_results'
    cv_dir = os.path.join(results_base, f'C2_custom/{disease}/{cv_subdir}/cf_{cf_count}')
    fold_data_dir = os.path.join(cv_dir, 'fold_data')
    plots_dir = os.path.join(cv_dir, 'cv_plots')

    os.makedirs(cv_dir, exist_ok=True)
    os.makedirs(fold_data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # NOW WITH HE NEW SPLIT, WE CAN LOAD THE FULL DATASET AND THEN SPLIT WITHIN EACH FOLD
    full_df = pd.read_csv(os.path.join(results_base, 'C2_custom/c2_data.csv'))
    
    # SUBSAMPLE TO 10K FOR FASTER TESTING (REMOVE THIS IN FINAL RUN)
    #if len(full_df) > 10000:
    #    full_df = full_df.sample(10000, random_state=RANDOM_SEED).reset_index(drop=True)
        
    # Rename columns to match expected format
    full_df.rename(columns={
        'prob': f'{disease}_prob',
        'pred': f'{disease}_pred',
        'true': f'{disease}_true'
    }, inplace=True)
    
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
        print(f"  Generating {cf_count} counterfactuals...")

        # This function computes CFs for the train/test split and returns new DataFrames
            # Replace the compute_cf_for_split call
        cf_fn = compute_cf_for_split_unmatched if unmatched else compute_cf_for_split
        fold_train_cf, fold_test_cf, _ = cf_fn(
            train_df=fold_train,
            test_df=fold_test,
            cf_count=cf_count,
            disease=disease,
            distance='l1'
        )
        
        # Optionally save fold data to disk - currenlty disabled
        if save_fold_data:
            fold_train_cf.to_csv(
                os.path.join(fold_data_dir, f'fold_{fold_idx}_train.csv'), 
                index=False
            )
            fold_test_cf.to_csv(
                os.path.join(fold_data_dir, f'fold_{fold_idx}_test.csv'), 
                index=False
            )
        
        # ── Build feature matrices ────────────────────────────────────────
        feature_sets = build_feature_matrices(fold_train_cf, fold_test_cf, disease)
        
        y_train = fold_train_cf['correct'].values
        y_test  = fold_test_cf['correct'].values
        
        # Prepare a DataFrame to store predictions for this fold
        fold_pred_df = fold_test_cf.copy()  # starts with all features, including path & correct

        # ── Train all models on all configs ───────────────────────────────
        for model_type in ['LR', 'RF', 'MLP']:
            print(f"\n  {model_type}:")

            # Folder to save models
            model_dir = os.path.join(cv_dir, 'models', model_type)
            os.makedirs(model_dir, exist_ok=True)

            for config in CONFIGS:
                X_train, X_test = feature_sets[config]

                # Train model and get metrics + trained model
                res = train_model(X_train, X_test, y_train, y_test, model_type=model_type)
                
                # Save model separately
                model_path = os.path.join(model_dir, f"{model_type}_{config}_fold{fold_idx}.pkl")
                if model_type in ['LR', 'MLP']:
                    joblib.dump({'model': res['model'], 'scaler': res['scaler']}, model_path)
                else:
                    joblib.dump(res['model'], model_path)
                    
                # Store results in cv_results dict
                # Only save metrics/predictions to JSON
                cv_results[model_type][config].append({
                    'auc': res['auc'],
                    'fpr': res['fpr'],
                    'tpr': res['tpr'],
                    'y_prob': res['y_prob'],
                    'y_true': res['y_true']
                })
                print(f"    {config}: AUC = {res['auc']:.4f}")
                
                # Save predicted probability in fold-wide CSV
                prob_col_name = f"{model_type}_{config}_prob"
                fold_pred_df[prob_col_name] = res['y_prob']
                
            # ── Save fold-wide predictions CSV
        pred_csv_path = os.path.join(cv_dir, f'fold_{fold_idx}_predictions.csv')
        fold_pred_df.to_csv(pred_csv_path, index=False)
        print(f"Fold {fold_idx} predictions saved to: {pred_csv_path}")
    
    # ── AGGREGATE RESULTS ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATING RESULTS")
    print(f"{'='*70}\n")
    
    summary_rows = []
    
    for model_type in ['LR', 'RF', 'MLP']:
        for config in CONFIGS:
            aucs = [fold_res['auc'] for fold_res in cv_results[model_type][config]]
            
            summary_rows.append({
                'model': model_type,
                'config': config,
                'mean_auc': np.mean(aucs),
                'std_auc': np.std(aucs),
                'min_auc': np.min(aucs),
                'max_auc': np.max(aucs),
                'fold_aucs': ','.join([f'{a:.4f}' for a in aucs])
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # ── SAVE RESULTS ──────────────────────────────────────────────────────
    summary_df.to_csv(os.path.join(cv_dir, 'cv_summary.csv'), index=False)
    
    with open(os.path.join(cv_dir, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {cv_dir}")
    
    return cv_results, summary_df

# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    
    # Jupyter passes kernel arguments that confuse argparse — detect and bypass
    if any('jupyter' in arg or 'ipykernel' in arg for arg in sys.argv):
        run_cv(disease='effusion', cf_count=1, save_fold_data=False, unmatched=False)
    else:
        parser = argparse.ArgumentParser(description='Run C2 cross-validation')
        parser.add_argument('--disease', type=str, default='effusion')
        parser.add_argument('--cf_count', type=int, default=1)
        parser.add_argument('--save_folds', action='store_true')
        parser.add_argument('--unmatched', action='store_true')
        args = parser.parse_args()
        run_cv(
            disease=args.disease,
            cf_count=args.cf_count,
            save_fold_data=args.save_folds,
            unmatched=args.unmatched
        )