"""
c2_train_simulated_cf.py
========================
Trains LR, RF and MLP models for C2 quality control, comparing all eleven input configurations for each model:
    B1 — prob only
    B2 — original attributes only
    B3 — embeddings only
    B4 — prob + attributes
    B5 — prob + embeddings
    M1 — ΔA only
    M2 — prob + ΔA
    M3 — prob + ΔA + attributes
    M4 — prob + ΔA + embeddings
    M5 — prob + ΔA + attributes + embeddings
    M6 — prob + ΔA + attributes + CF prob

Assumes c2_prepare_data_simulated_cf.py has already been run and saved:
    - train_with_diff_vectors_{cf_count}.csv
    - valid_with_diff_vectors_{cf_count}.csv

Outputs (saved to results/C2_sim_cf/{disease}/):
    - roc_data_{cf_count}.json
    - results_summary_{cf_count}.txt
    cf{cf_count}_lr/
        - roc_lr.png
        - lr_{config}_model.pkl
        - lr_{config}_scaler.pkl
    cf{cf_count}_rf/
        - roc_rf.png
        - rf_{config}_model.pkl
    cf{cf_count}_mlp/
        - roc_mlp.png
        - mlp_{config}_model.pkl
        - mlp_{config}_scaler.pkl
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from plot_config import PLOT_COLORS, PLOT_LS, PLOT_LW


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit this block only
# ══════════════════════════════════════════════════════════════════════════════

DISEASE  = 'effusion'   # effusion | pneumothorax | cardiomegaly | atelectasis
BASE_DIR = '/zhome/d0/a/221493/thesis/'
RUN_CV   = False        # set True to enable 5-fold cross-validation (slow)

cf_count = 1
UNMATCHED = True
# ══════════════════════════════════════════════════════════════════════════════

# Setup paths for use later
RESULTS_DIR  = os.path.join(BASE_DIR, 'results')
INPUT_DIR    = os.path.join(RESULTS_DIR, f'C2_sim_cf/{DISEASE}')
LR_DIR       = os.path.join(INPUT_DIR, f'cf{cf_count}_lr')
RF_DIR       = os.path.join(INPUT_DIR, f'cf{cf_count}_rf')
MLP_DIR      = os.path.join(INPUT_DIR, f'cf{cf_count}_mlp')

# If using unmatched CF examples, change the input and output dirs accordingly.
if UNMATCHED:
    LR_DIR       = os.path.join(INPUT_DIR, f'cf{cf_count}_unmatched_lr')
    RF_DIR       = os.path.join(INPUT_DIR, f'cf{cf_count}_unmatched_rf')
    MLP_DIR      = os.path.join(INPUT_DIR, f'cf{cf_count}_unmatched_mlp')

# Create output dirs if they don't exist
os.makedirs(LR_DIR,  exist_ok=True)
os.makedirs(RF_DIR,  exist_ok=True)
os.makedirs(MLP_DIR, exist_ok=True)

# Get the name of the disease column
disease_prob_col = f"{DISEASE.lower()}_prob"

# Define the name suffix for files based on whether we're using unmatched CF examples or not. 
# This is just to keep track of which results correspond to which CF generation method. 
# If UNMATCHED is True, it means we used the "unmatched" CF examples (these are generated without enforcing that the CF example is correctly assigned its class).
suffix = f"_{cf_count}_unmatched" if UNMATCHED else f"_{cf_count}"

# All configs — keys match plot_config.py
CONFIGS = ['B1', 'B2', 'B3', 'B4', 'B5', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']
CONFIG_LABELS = {
    'B1': 'B1 — Prob only',
    'B2': 'B2 — Attrs only',
    'B3': 'B3 — Embeddings only',
    'B4': 'B4 — Prob + Attrs',
    'B5': 'B5 — Prob + Embeddings',
    'M1': 'M1 — ΔA only',
    'M2': 'M2 — Prob + ΔA',
    'M3': 'M3 — Prob + ΔA + Attrs',
    'M4': 'M4 — Prob + ΔA + Embeddings',
    'M5': 'M5 — Prob + ΔA + Attrs + Embeddings',
    'M6': 'M6 — ΔA + Prob + CF Prob + Attrs',
}

# Helper function to organize feature sets for all configs
def _get_feature_sets(X_B1_tr, X_B1_va, X_B2_tr, X_B2_va,
                      X_B3_tr, X_B3_va, X_B4_tr, X_B4_va,
                      X_B5_tr, X_B5_va, X_M1_tr, X_M1_va,
                      X_M2_tr, X_M2_va, X_M3_tr, X_M3_va,
                      X_M4_tr, X_M4_va, X_M5_tr, X_M5_va,
                      X_M6_tr, X_M6_va):
    return {
        'B1': (X_B1_tr, X_B1_va),
        'B2': (X_B2_tr, X_B2_va),
        'B3': (X_B3_tr, X_B3_va),
        'B4': (X_B4_tr, X_B4_va),
        'B5': (X_B5_tr, X_B5_va),
        'M1': (X_M1_tr, X_M1_va),
        'M2': (X_M2_tr, X_M2_va),
        'M3': (X_M3_tr, X_M3_va),
        'M4': (X_M4_tr, X_M4_va),
        'M5': (X_M5_tr, X_M5_va),
        'M6': (X_M6_tr, X_M6_va),
    }


# ── Per-model run functions ───────────────────────────────────────────────────
# Each returns a results dict keyed by config, storing:
# auc, fpr, tpr, model, cv_mean, cv_std, scaler

def run_lr(fsets, y_train, y_valid):
    results = {}
    for tag, (Xtr, Xva) in fsets.items():
        scaler = StandardScaler()
        
         # MODEL 
        # Class Weight = weighting samples inversely proportional to class frequency.
        model  = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
        model.fit(scaler.fit_transform(Xtr), y_train)

        # Run prediction and calculate metrics
        y_prob      = model.predict_proba(scaler.transform(Xva))[:, 1]
        y_pred      = model.predict(scaler.transform(Xva))
        auc         = roc_auc_score(y_valid, y_prob)
        fpr, tpr, _ = roc_curve(y_valid, y_prob)

        if RUN_CV: # Only run CV if enabled, as it can be very slow
            pipe = Pipeline([('sc', StandardScaler()),
                             ('m',  LogisticRegression(max_iter=5000,
                                    class_weight='balanced', random_state=42))])
            cv = cross_val_score(pipe, Xtr, y_train,
                                 cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
            cv_mean, cv_std = cv.mean(), cv.std()
            
        else: # Standard test-set evaluation only
            cv_mean, cv_std = float('nan'), float('nan')


        print(f"  LR [{tag}] Valid AUC = {auc:.4f}" +
              (f" | 5-fold CV = {cv_mean:.4f} ± {cv_std:.4f}" if RUN_CV else ""))
        print(classification_report(y_valid, y_pred, target_names=['Incorrect', 'Correct']))

        # Store results for this config
        results[tag] = {'auc': auc, 'fpr': fpr, 'tpr': tpr, 'model': model,
                        'cv_mean': cv_mean, 'cv_std': cv_std, 'scaler': scaler}
    return results


def run_rf(fsets, y_train, y_valid):
    results = {}
    for tag, (Xtr, Xva) in fsets.items():
        
        # MODEL 
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                       random_state=42, n_jobs=-1)
        model.fit(Xtr, y_train)

        # Run prediction and calculate metrics
        y_prob      = model.predict_proba(Xva)[:, 1]
        y_pred      = model.predict(Xva)
        auc         = roc_auc_score(y_valid, y_prob)
        fpr, tpr, _ = roc_curve(y_valid, y_prob)

        if RUN_CV: # Only run CV if enabled, as it can be very slow
            pipe = Pipeline([('sc', StandardScaler()),
                             ('m',  RandomForestClassifier(n_estimators=200,
                                    class_weight='balanced', random_state=42, n_jobs=-1))])
            cv = cross_val_score(pipe, Xtr, y_train,
                                 cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
            cv_mean, cv_std = cv.mean(), cv.std()
            
        else: # Standard test-set evaluation only
            cv_mean, cv_std = float('nan'), float('nan')

        print(f"  RF [{tag}] Valid AUC = {auc:.4f}" +
              (f" | 5-fold CV = {cv_mean:.4f} ± {cv_std:.4f}" if RUN_CV else ""))
        print(classification_report(y_valid, y_pred, target_names=['Incorrect', 'Correct']))

        # Store results for this config
        results[tag] = {'auc': auc, 'fpr': fpr, 'tpr': tpr, 'model': model,
                        'cv_mean': cv_mean, 'cv_std': cv_std}
    return results


def run_mlp(fsets, y_train, y_valid):
    results = {}
    for tag, (Xtr, Xva) in fsets.items():
        Xtr32  = Xtr.astype(np.float32)
        Xva32  = Xva.astype(np.float32)
        scaler = StandardScaler()
        
         # MODEL 
        model  = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                               early_stopping=True, validation_fraction=0.05,
                               random_state=42)
        model.fit(scaler.fit_transform(Xtr32), y_train)

        # Run prediction and calculate metrics
        y_prob      = model.predict_proba(scaler.transform(Xva32))[:, 1]
        y_pred      = model.predict(scaler.transform(Xva32))
        auc         = roc_auc_score(y_valid, y_prob)
        fpr, tpr, _ = roc_curve(y_valid, y_prob)

        if RUN_CV: # Only run CV if enabled, as it can be very slow
            pipe = Pipeline([('sc', StandardScaler()),
                             ('m',  MLPClassifier(hidden_layer_sizes=(64, 32, 16),
                                    max_iter=500, early_stopping=True,
                                    validation_fraction=0.05, random_state=42))])
            cv = cross_val_score(pipe, Xtr32, y_train,
                                 cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
            cv_mean, cv_std = cv.mean(), cv.std()
            
        else: # Standard test-set evaluation only
            cv_mean, cv_std = float('nan'), float('nan')

        print(f"  MLP [{tag}] Valid AUC = {auc:.4f}" +
              (f" | 5-fold CV = {cv_mean:.4f} ± {cv_std:.4f}" if RUN_CV else ""))
        print(classification_report(y_valid, y_pred, target_names=['Incorrect', 'Correct']))

        # Store results for this config
        results[tag] = {'auc': auc, 'fpr': fpr, 'tpr': tpr, 'model': model,
                        'cv_mean': cv_mean, 'cv_std': cv_std, 'scaler': scaler}
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_roc_configs(results, model_name, disease, output_dir):
    """ROC curves for all configs of a single model type."""
    plt.figure(figsize=(8, 7))
    for tag in CONFIGS:
        plt.plot(results[tag]['fpr'], results[tag]['tpr'],
                 color=PLOT_COLORS[tag], lw=PLOT_LW[tag], ls=PLOT_LS[tag],
                 label=f"{CONFIG_LABELS[tag]:<35} AUC={results[tag]['auc']:.3f}")
    plt.plot([0, 1], [0, 1], 'k:', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'C2 — {model_name} — {disease}')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3) # alpha is the transparency of the grid lines
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_{model_name.lower()}.png'), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Print config info
    print(f"Disease    : {DISEASE}")
    print(f"Input dir  : {INPUT_DIR}")
    print(f"LR dir     : {LR_DIR}")
    print(f"RF dir     : {RF_DIR}")
    print(f"MLP dir    : {MLP_DIR}")
    print(f"Run CV     : {RUN_CV}\n")

    # ── Load pre-computed training and validation data ───────────────────────────
    # These were created with c2_prepare_data_simulated_cf.py, and contain:
    #   - Meta columns: path, patient_id, correct, {disease}_pred, {disease}_true, {disease}_prob
    #   - Attr columns: geometrics, ratios, radiomics and demographics
    #   - Embedding columns: emb_0, emb_1, ..., emb_1023
    #   - ΔA columns: delta_{attr} for each attr column (e.g. delta_lung_area) between original and CF
    #   - CF prob column: cf_prob = model's predicted probability for the found CF example
    
    train_df = pd.read_csv(os.path.join(INPUT_DIR, f'train_with_diff_vectors{suffix}.csv')) # suffix is either _{cf_count}_unmatched or _{cf_count} depending on whether we're using unmatched CF examples or not
    valid_df = pd.read_csv(os.path.join(INPUT_DIR, f'valid_with_diff_vectors{suffix}.csv'))

    # Identify column groups and divide so we can easily create feature matrices for each config. We use the following naming conventions:
    # - ΔA features start with 'delta_'
    # - Embedding features start with 'emb_'
    # - Meta features are explicitly listed (disease_prob_col, disease_pred, disease_true, correct, path, patient_id)
    
    diff_cols = [c for c in train_df.columns if c.startswith('delta_')]
    emb_cols  = [c for c in train_df.columns if c.startswith('emb_')]
    meta_cols = {disease_prob_col, f"{DISEASE.lower()}_pred", f"{DISEASE.lower()}_true",
                'correct', 'path', 'patient_id', 'cf_prob'}
    attr_cols = [c for c in train_df.columns
                 if c not in meta_cols
                 and not c.startswith('delta_')
                 and not c.startswith('emb_')]

    print(f"Train: {len(train_df):,}  Valid: {len(valid_df):,}") # Print number of samples in train and valid sets
    print(f"Attr features : {len(attr_cols)}") # Print number of attribute features (geometric, radiomic, demographic)
    print(f"ΔA features   : {len(diff_cols)}") # Print number of ΔA features (the "difference" features between original and CF examples)
    print(f"Emb features  : {len(emb_cols)}") # Print number of embedding features (the 1024-dim vector from the penultimate layer of the DenseNet121)

    # ── Extract labels for target ───────────────────────────────────────────────────────────
    y_train = train_df['correct'].values
    y_valid = valid_df['correct'].values

    print(f"Train — correct: {y_train.sum():,} ({y_train.mean():.1%})  "
          f"incorrect: {(1-y_train).sum():,} ({(1-y_train).mean():.1%})")
    print(f"Valid  — correct: {y_valid.sum():,} ({y_valid.mean():.1%})  "
          f"incorrect: {(1-y_valid).sum():,} ({(1-y_valid).mean():.1%})\n")

    # ── Feature matrices ──────────────────────────────────────────────────────
    # For each config, we create the corresponding feature matrices for train and valid sets. For example:
    prob_tr = train_df[[disease_prob_col]].values # Used in configs B1, B4, B5, M2, M3, M4, M5, M6
    prob_va = valid_df[[disease_prob_col]].values 
    attr_tr = train_df[attr_cols].values # Used in configs B2, B4, M3, M5, M6
    attr_va = valid_df[attr_cols].values 
    diff_tr = train_df[diff_cols].values # Used in configs M1, M2, M3, M4, M5, M6 
    diff_va = valid_df[diff_cols].values
    emb_tr  = train_df[emb_cols].values # Used in configs B3, B5, M4, M5
    emb_va  = valid_df[emb_cols].values
    cf_prob_tr = train_df[['cf_prob']].values # Used in configs M6
    cf_prob_va = valid_df[['cf_prob']].values

    fsets = _get_feature_sets(
        prob_tr,                              prob_va,                                  # B1
        attr_tr,                              attr_va,                                  # B2
        emb_tr,                               emb_va,                                   # B3
        np.hstack([prob_tr, attr_tr]),        np.hstack([prob_va, attr_va]),            # B4
        np.hstack([prob_tr, emb_tr]),         np.hstack([prob_va, emb_va]),             # B5
        diff_tr,                              diff_va,                                  # M1
        np.hstack([prob_tr, diff_tr]),        np.hstack([prob_va, diff_va]),            # M2
        np.hstack([prob_tr, diff_tr, attr_tr]),np.hstack([prob_va, diff_va, attr_va]),  # M3
        np.hstack([prob_tr, diff_tr, emb_tr]),np.hstack([prob_va, diff_va, emb_va]),    # M4
        np.hstack([prob_tr, diff_tr, attr_tr, emb_tr]),                                 # M5
        np.hstack([prob_va, diff_va, attr_va, emb_va]),
        np.hstack([prob_tr, diff_tr, attr_tr, cf_prob_tr]),                             # M6
        np.hstack([prob_va, diff_va, attr_va, cf_prob_va]),
    )

    # ── Train all models ──────────────────────────────────────────────────────

    print("="*60 + "\n  LOGISTIC REGRESSION\n" + "="*60)
    lr_res  = run_lr(fsets, y_train, y_valid)

    print("="*60 + "\n  RANDOM FOREST\n" + "="*60)
    rf_res  = run_rf(fsets, y_train, y_valid)

    print("="*60 + "\n  MLP\n" + "="*60)
    mlp_res = run_mlp(fsets, y_train, y_valid)

    # ── Per-model ROC plots ───────────────────────────────────────────────────

    plot_roc_configs(lr_res,  'LR',  DISEASE, LR_DIR)
    plot_roc_configs(rf_res,  'RF',  DISEASE, RF_DIR)
    plot_roc_configs(mlp_res, 'MLP', DISEASE, MLP_DIR)

    # ── Save ROC data for external plotting ──────────────────────────────────

    roc_data = {}
    for model_name, res in [('LR', lr_res), ('RF', rf_res), ('MLP', mlp_res)]:
        roc_data[model_name] = {}
        for tag in CONFIGS:
            roc_data[model_name][tag] = {
                'fpr':     res[tag]['fpr'].tolist(),
                'tpr':     res[tag]['tpr'].tolist(),
                'auc':     float(res[tag]['auc']),
                'cv_mean': float(res[tag]['cv_mean']),
                'cv_std':  float(res[tag]['cv_std']),
            }
    roc_save_path = os.path.join(INPUT_DIR, f'roc_data{suffix}.json')
    with open(roc_save_path, 'w') as f:
        json.dump(roc_data, f, indent=2)
    print(f"ROC data saved → {roc_save_path}")

    # ── Summary table ─────────────────────────────────────────────────────────

    summary = (
        f"=== C2 Model Results — {DISEASE} ===\n\n"
        f"Train : {len(train_df):,}  (correct: {y_train.sum():,} | incorrect: {(1-y_train).sum():,})\n"
        f"Valid : {len(valid_df):,}  (correct: {y_valid.sum():,} | incorrect: {(1-y_valid).sum():,})\n"
        f"ΔA features   : {len(diff_cols)}\n"
        f"Attr features : {len(attr_cols)}\n"
        f"Emb features  : {len(emb_cols)}\n"
        f"CV enabled    : {RUN_CV}\n\n"
        f"{'Model':<5} {'Config':<4} {'Valid AUC':>10}  {'CV mean':>10}  {'CV std':>10}\n"
        f"{'-'*50}\n"
    )
    for model_name, res in [('LR', lr_res), ('RF', rf_res), ('MLP', mlp_res)]:
        for tag in CONFIGS:
            summary += (f"{model_name:<5} {tag:<4} {res[tag]['auc']:>10.4f}  "
                        f"{res[tag]['cv_mean']:>10.4f}  {res[tag]['cv_std']:>10.4f}\n")
        summary += "\n"
        
    print("\n" + summary)
    with open(os.path.join(INPUT_DIR, f'results_summary{suffix}.txt'), 'w') as f:
        f.write(summary)

    # ── Save models ───────────────────────────────────────────────────────────

    # Iteratively goes to each config, and saves the 3 model and scalers
    for tag in CONFIGS:
        joblib.dump(lr_res[tag]['model'],   os.path.join(LR_DIR,  f'lr_{tag.lower()}_model.pkl'))
        joblib.dump(lr_res[tag]['scaler'],  os.path.join(LR_DIR,  f'lr_{tag.lower()}_scaler.pkl'))
        joblib.dump(rf_res[tag]['model'],   os.path.join(RF_DIR,  f'rf_{tag.lower()}_model.pkl'))
        joblib.dump(mlp_res[tag]['model'],  os.path.join(MLP_DIR, f'mlp_{tag.lower()}_model.pkl'))
        joblib.dump(mlp_res[tag]['scaler'], os.path.join(MLP_DIR, f'mlp_{tag.lower()}_scaler.pkl'))

    print(f"\nAll outputs saved → {INPUT_DIR}")


if __name__ == "__main__":
    main()
