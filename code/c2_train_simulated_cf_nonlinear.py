"""
c2_train_nonlinear.py
=====================
Extension of c2_train_simulated_cf.py that trains non-linear models (RF, MLP, GBM)
comparing three input configurations for each model:
    1. Baseline  — C0 probability only
    2. ΔA only   — simulated counterfactual difference vector only
    3. Combined  — C0 probability + ΔA

Assumes c2_train_simulated_cf.py has already been run and saved:
    - train_with_diff_vectors.csv
    - valid_with_diff_vectors.csv

Outputs (saved to results/C2_sim_cf/{disease}/nonlinear/):
    - roc_lr.png
    - roc_rf.png
    - roc_mlp.png
    - roc_gbm.png
    - roc_best_per_model.png
    - rf_feature_importances.png
    - results_summary_nonlinear.txt
    - rf_model.pkl  |  mlp_model.pkl  |  gbm_model.pkl

Usage:
    python c2_train_nonlinear.py
    Change DISEASE to match what you ran in c2_train_simulated_cf.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — match DISEASE to what you ran in c2_train_simulated_cf.py
# ══════════════════════════════════════════════════════════════════════════════

DISEASE  = 'effusion'   # effusion | pneumothorax | cardiomegaly | atelectasis
BASE_DIR = '/zhome/d0/a/221493/thesis/'

# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
INPUT_DIR   = os.path.join(RESULTS_DIR, f'C2_sim_cf/{DISEASE}')
OUTPUT_DIR  = os.path.join(INPUT_DIR, 'nonlinear')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Model training functions ──────────────────────────────────────────────────
# Each function trains on all three input configs and returns results dict

def run_lr(X_base_tr, X_base_va, X_diff_tr, X_diff_va,
           X_comb_tr, X_comb_va, y_train, y_valid):
    results = {}
    for tag, Xtr, Xva in [('base', X_base_tr, X_base_va),
                           ('diff', X_diff_tr, X_diff_va),
                           ('comb', X_comb_tr, X_comb_va)]:
        scaler  = StandardScaler()
        model   = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        model.fit(scaler.fit_transform(Xtr), y_train)
        y_prob  = model.predict_proba(scaler.transform(Xva))[:, 1]
        y_pred  = model.predict(scaler.transform(Xva))
        auc     = roc_auc_score(y_valid, y_prob)
        fpr, tpr, _ = roc_curve(y_valid, y_prob)
        pipe    = Pipeline([('sc', StandardScaler()),
                            ('m',  LogisticRegression(max_iter=1000,
                                   class_weight='balanced', random_state=42))])
        cv      = cross_val_score(pipe, Xtr, y_train,
                                  cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
        print(f"  LR [{tag:<4}] Valid AUC = {auc:.4f} | 5-fold CV = {cv.mean():.4f} ± {cv.std():.4f}")
        print(classification_report(y_valid, y_pred, target_names=['Incorrect', 'Correct']))
        results[tag] = {'auc': auc, 'fpr': fpr, 'tpr': tpr, 'model': model}
    return results


def run_rf(X_base_tr, X_base_va, X_diff_tr, X_diff_va,
           X_comb_tr, X_comb_va, y_train, y_valid):
    results = {}
    for tag, Xtr, Xva in [('base', X_base_tr, X_base_va),
                           ('diff', X_diff_tr, X_diff_va),
                           ('comb', X_comb_tr, X_comb_va)]:
        model   = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                         random_state=42, n_jobs=-1)
        model.fit(Xtr, y_train)
        y_prob  = model.predict_proba(Xva)[:, 1]
        y_pred  = model.predict(Xva)
        auc     = roc_auc_score(y_valid, y_prob)
        fpr, tpr, _ = roc_curve(y_valid, y_prob)
        pipe    = Pipeline([('sc', StandardScaler()),
                            ('m',  RandomForestClassifier(n_estimators=200,
                                   class_weight='balanced', random_state=42, n_jobs=-1))])
        cv      = cross_val_score(pipe, Xtr, y_train,
                                  cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
        print(f"  RF [{tag:<4}] Valid AUC = {auc:.4f} | 5-fold CV = {cv.mean():.4f} ± {cv.std():.4f}")
        print(classification_report(y_valid, y_pred, target_names=['Incorrect', 'Correct']))
        results[tag] = {'auc': auc, 'fpr': fpr, 'tpr': tpr, 'model': model}
    return results


def run_mlp(X_base_tr, X_base_va, X_diff_tr, X_diff_va,
            X_comb_tr, X_comb_va, y_train, y_valid):
    results = {}
    for tag, Xtr, Xva in [('base', X_base_tr, X_base_va),
                           ('diff', X_diff_tr, X_diff_va),
                           ('comb', X_comb_tr, X_comb_va)]:
        scaler  = StandardScaler()
        model   = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                                early_stopping=True, random_state=42)
        model.fit(scaler.fit_transform(Xtr), y_train)
        y_prob  = model.predict_proba(scaler.transform(Xva))[:, 1]
        y_pred  = model.predict(scaler.transform(Xva))
        auc     = roc_auc_score(y_valid, y_prob)
        fpr, tpr, _ = roc_curve(y_valid, y_prob)
        pipe    = Pipeline([('sc', StandardScaler()),
                            ('m',  MLPClassifier(hidden_layer_sizes=(64, 32, 16),
                                   max_iter=500, early_stopping=True, random_state=42))])
        cv      = cross_val_score(pipe, Xtr, y_train,
                                  cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
        print(f"  MLP [{tag:<4}] Valid AUC = {auc:.4f} | 5-fold CV = {cv.mean():.4f} ± {cv.std():.4f}")
        print(classification_report(y_valid, y_pred, target_names=['Incorrect', 'Correct']))
        results[tag] = {'auc': auc, 'fpr': fpr, 'tpr': tpr, 'model': model}
    return results


def plot_roc_trio(results, model_name, disease, output_dir):
    """Plot the three curves (base / diff / comb) for a single model."""
    plt.figure(figsize=(7, 6))
    plt.plot(results['base']['fpr'], results['base']['tpr'],
             color='gray',      lw=2, ls='--',
             label=f"Prob only   AUC={results['base']['auc']:.3f}")
    plt.plot(results['diff']['fpr'], results['diff']['tpr'],
             color='darkorange', lw=2,
             label=f"ΔA only     AUC={results['diff']['auc']:.3f}")
    plt.plot(results['comb']['fpr'], results['comb']['tpr'],
             color='steelblue',  lw=2,
             label=f"Prob + ΔA   AUC={results['comb']['auc']:.3f}")
    plt.plot([0, 1], [0, 1], 'k:', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'C2 — {model_name} — {disease}')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_{model_name.lower()}.png'), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Disease    : {DISEASE}")
    print(f"Input dir  : {INPUT_DIR}")
    print(f"Output dir : {OUTPUT_DIR}\n")

    # ── Load pre-computed diff vectors ────────────────────────────────────────

    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train_with_diff_vectors.csv'))
    valid_df = pd.read_csv(os.path.join(INPUT_DIR, 'valid_with_diff_vectors.csv'))

    diff_cols = [c for c in train_df.columns if c.startswith('delta_')]
    print(f"Train: {len(train_df):,}  Valid: {len(valid_df):,}  ΔA features: {len(diff_cols)}")

    y_train = train_df['correct'].values
    y_valid = valid_df['correct'].values

    print(f"Train — correct: {y_train.sum():,} ({y_train.mean():.1%})  incorrect: {(1-y_train).sum():,} ({(1-y_train).mean():.1%})")
    print(f"Valid  — correct: {y_valid.sum():,} ({y_valid.mean():.1%})  incorrect: {(1-y_valid).sum():,} ({(1-y_valid).mean():.1%})\n")

    # ── Feature matrices ──────────────────────────────────────────────────────

    X_base_tr = train_df[['prob']].values
    X_base_va = valid_df[['prob']].values
    X_diff_tr = train_df[diff_cols].values
    X_diff_va = valid_df[diff_cols].values
    X_comb_tr = np.hstack([X_base_tr, X_diff_tr])
    X_comb_va = np.hstack([X_base_va, X_diff_va])

    # ── Train all models ──────────────────────────────────────────────────────

    print("="*60 + "\n  LOGISTIC REGRESSION\n" + "="*60)
    lr_res  = run_lr( X_base_tr, X_base_va, X_diff_tr, X_diff_va, X_comb_tr, X_comb_va, y_train, y_valid)

    print("="*60 + "\n  RANDOM FOREST\n" + "="*60)
    rf_res  = run_rf( X_base_tr, X_base_va, X_diff_tr, X_diff_va, X_comb_tr, X_comb_va, y_train, y_valid)

    print("="*60 + "\n  MLP\n" + "="*60)
    mlp_res = run_mlp(X_base_tr, X_base_va, X_diff_tr, X_diff_va, X_comb_tr, X_comb_va, y_train, y_valid)

    # ── Per-model ROC plots (base / diff / comb) ──────────────────────────────

    plot_roc_trio(lr_res,  'LR',  DISEASE, OUTPUT_DIR)
    plot_roc_trio(rf_res,  'RF',  DISEASE, OUTPUT_DIR)
    plot_roc_trio(mlp_res, 'MLP', DISEASE, OUTPUT_DIR)

    # ── Summary ROC — best config per model (combined) ────────────────────────

    plt.figure(figsize=(8, 7))
    plt.plot(lr_res['base']['fpr'],  lr_res['base']['tpr'],
             color='gray',      lw=2, ls='--', label=f"Baseline (prob only)  AUC={lr_res['base']['auc']:.3f}")
    plt.plot(lr_res['comb']['fpr'],  lr_res['comb']['tpr'],
             color='steelblue',  lw=2, label=f"LR  prob + ΔA          AUC={lr_res['comb']['auc']:.3f}")
    plt.plot(rf_res['comb']['fpr'],  rf_res['comb']['tpr'],
             color='green',      lw=2, label=f"RF  prob + ΔA          AUC={rf_res['comb']['auc']:.3f}")
    plt.plot(mlp_res['comb']['fpr'], mlp_res['comb']['tpr'],
             color='darkorange', lw=2, label=f"MLP prob + ΔA          AUC={mlp_res['comb']['auc']:.3f}")
    plt.plot([0, 1], [0, 1], 'k:', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'C2 Quality Control — {DISEASE}\nAll Models — Combined Features (prob + ΔA)')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_best_per_model.png'), dpi=150)
    plt.close()

    # ── RF feature importances (combined model) ───────────────────────────────

    feature_names = ['C0_prob'] + diff_cols
    importances   = pd.Series(rf_res['comb']['model'].feature_importances_,
                               index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    sns.barplot(x=importances.index, y=importances.values)
    plt.xticks(rotation=90)
    plt.title(f'RF Feature Importances — Combined Model ({DISEASE})')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_feature_importances.png'), dpi=150)
    plt.close()

    # ── Summary table ─────────────────────────────────────────────────────────

    summary = (
        f"=== C2 Non-linear Model Results — {DISEASE} ===\n\n"
        f"Train : {len(train_df):,}  (correct: {y_train.sum():,} | incorrect: {(1-y_train).sum():,})\n"
        f"Valid : {len(valid_df):,}  (correct: {y_valid.sum():,} | incorrect: {(1-y_valid).sum():,})\n\n"
        f"| Model | Prob only | ΔA only | Prob + ΔA |\n"
        f"|-------|-----------|---------|----------|\n"
        f"| LR    | {lr_res['base']['auc']:.3f}     | {lr_res['diff']['auc']:.3f}   | {lr_res['comb']['auc']:.3f}     |\n"
        f"| RF    | {rf_res['base']['auc']:.3f}     | {rf_res['diff']['auc']:.3f}   | {rf_res['comb']['auc']:.3f}     |\n"
        f"| MLP   | {mlp_res['base']['auc']:.3f}     | {mlp_res['diff']['auc']:.3f}   | {mlp_res['comb']['auc']:.3f}     |\n"
        f"Top 10 RF feature importances (combined):\n{importances.head(10).to_string()}\n"
    )
    print("\n" + summary)
    with open(os.path.join(OUTPUT_DIR, 'results_summary_nonlinear.txt'), 'w') as f:
        f.write(summary)

    # ── Save models ───────────────────────────────────────────────────────────

    joblib.dump(rf_res['comb']['model'],  os.path.join(OUTPUT_DIR, 'rf_model.pkl'))
    joblib.dump(mlp_res['comb']['model'], os.path.join(OUTPUT_DIR, 'mlp_model.pkl'))

    print(f"\nAll outputs saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()