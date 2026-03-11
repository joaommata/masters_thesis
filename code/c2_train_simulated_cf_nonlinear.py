"""
c2_train_nonlinear.py
=====================
Extension of c2_train_simulated_cf.py that trains non-linear models (RF, MLP)
comparing five input configurations for each model:
    1. Baseline      — C0 probability only
    2. Attr only     — original attribute vector only
    3. ΔA only       — simulated counterfactual difference vector only
    4. Combined      — C0 probability + ΔA
    5. Extended      — C0 probability + ΔA + original attributes

Assumes c2_train_simulated_cf.py has already been run and saved:
    - train_with_diff_vectors.csv
    - valid_with_diff_vectors.csv

Outputs (saved to results/C2_sim_cf/{disease}/nonlinear/):
    - roc_lr.png
    - roc_rf.png
    - roc_mlp.png
    - roc_best_per_model.png
    - rf_feature_importances.png
    - results_summary_nonlinear.txt
    - rf_comb_model.pkl  |  mlp_comb_model.pkl

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — match DISEASE to what you ran in c2_train_simulated_cf.py
# ══════════════════════════════════════════════════════════════════════════════

DISEASE  = 'atelectasis'   # effusion | pneumothorax | cardiomegaly | atelectasis
BASE_DIR = '/zhome/d0/a/221493/thesis/'

# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR  = os.path.join(BASE_DIR, 'results')
INPUT_DIR    = os.path.join(RESULTS_DIR, f'C2_sim_cf/{DISEASE}')
OUTPUT_DIR   = os.path.join(INPUT_DIR, 'nonlinear')
os.makedirs(OUTPUT_DIR, exist_ok=True)

disease_prob_col = f"{DISEASE.lower()}_prob"


# ── Model training functions ──────────────────────────────────────────────────
# Each function trains on all five input configs and returns a results dict.

CONFIGS = ['base', 'attr', 'diff', 'comb', 'ext']
CONFIG_LABELS = {
    'base': 'Prob only',
    'attr': 'Attrs only',
    'diff': 'ΔA only',
    'comb': 'Prob + ΔA',
    'ext':  'Prob + ΔA + Attrs',
}
CONFIG_COLORS = {
    'base': 'gray',
    'attr': 'purple',
    'diff': 'darkorange',
    'comb': 'steelblue',
    'ext':  'green',
}
CONFIG_LS = {
    'base': '--',
    'attr': '--',
    'diff': '-',
    'comb': '-',
    'ext':  '-',
}


def _get_feature_sets(X_base_tr, X_base_va, X_attr_tr, X_attr_va,
                      X_diff_tr, X_diff_va, X_comb_tr, X_comb_va,
                      X_ext_tr,  X_ext_va):
    return {
        'base': (X_base_tr, X_base_va),
        'attr': (X_attr_tr, X_attr_va),
        'diff': (X_diff_tr, X_diff_va),
        'comb': (X_comb_tr, X_comb_va),
        'ext':  (X_ext_tr,  X_ext_va),
    }


def run_lr(X_base_tr, X_base_va, X_attr_tr, X_attr_va,
           X_diff_tr, X_diff_va, X_comb_tr, X_comb_va,
           X_ext_tr,  X_ext_va,  y_train, y_valid):
    results = {}
    fsets   = _get_feature_sets(X_base_tr, X_base_va, X_attr_tr, X_attr_va,
                                 X_diff_tr, X_diff_va, X_comb_tr, X_comb_va,
                                 X_ext_tr,  X_ext_va)
    for tag, (Xtr, Xva) in fsets.items():
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


def run_rf(X_base_tr, X_base_va, X_attr_tr, X_attr_va,
           X_diff_tr, X_diff_va, X_comb_tr, X_comb_va,
           X_ext_tr,  X_ext_va,  y_train, y_valid):
    results = {}
    fsets   = _get_feature_sets(X_base_tr, X_base_va, X_attr_tr, X_attr_va,
                                 X_diff_tr, X_diff_va, X_comb_tr, X_comb_va,
                                 X_ext_tr,  X_ext_va)
    for tag, (Xtr, Xva) in fsets.items():
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


def run_mlp(X_base_tr, X_base_va, X_attr_tr, X_attr_va,
            X_diff_tr, X_diff_va, X_comb_tr, X_comb_va,
            X_ext_tr,  X_ext_va,  y_train, y_valid):
    results = {}
    fsets   = _get_feature_sets(X_base_tr, X_base_va, X_attr_tr, X_attr_va,
                                 X_diff_tr, X_diff_va, X_comb_tr, X_comb_va,
                                 X_ext_tr,  X_ext_va)
    for tag, (Xtr, Xva) in fsets.items():
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


def plot_roc_five(results, model_name, disease, output_dir):
    """Plot all five config curves for a single model."""
    plt.figure(figsize=(7, 6))
    for tag in CONFIGS:
        plt.plot(results[tag]['fpr'], results[tag]['tpr'],
                 color=CONFIG_COLORS[tag], lw=2, ls=CONFIG_LS[tag],
                 label=f"{CONFIG_LABELS[tag]:<22} AUC={results[tag]['auc']:.3f}")
    plt.plot([0, 1], [0, 1], 'k:', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'C2 — {model_name} — {disease}\nDoes ΔA add signal beyond C0 Probs?')
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

    # Attribute columns = everything that's not meta, not disease cols, not delta cols
    meta_cols = {disease_prob_col, f"{DISEASE.lower()}_pred", f"{DISEASE.lower()}_true",
                 'correct', 'path', 'patient_id'}
    attr_cols = [c for c in train_df.columns
                 if c not in meta_cols and not c.startswith('delta_')]

    print(f"Train: {len(train_df):,}  Valid: {len(valid_df):,}")
    print(f"ΔA features   : {len(diff_cols)}")
    print(f"Attr features : {len(attr_cols)}")

    y_train = train_df['correct'].values
    y_valid = valid_df['correct'].values

    print(f"Train — correct: {y_train.sum():,} ({y_train.mean():.1%})  incorrect: {(1-y_train).sum():,} ({(1-y_train).mean():.1%})")
    print(f"Valid  — correct: {y_valid.sum():,} ({y_valid.mean():.1%})  incorrect: {(1-y_valid).sum():,} ({(1-y_valid).mean():.1%})\n")

    # ── Feature matrices ──────────────────────────────────────────────────────

    X_base_tr = train_df[[disease_prob_col]].values
    X_base_va = valid_df[[disease_prob_col]].values
    X_attr_tr = train_df[attr_cols].values
    X_attr_va = valid_df[attr_cols].values
    X_diff_tr = train_df[diff_cols].values
    X_diff_va = valid_df[diff_cols].values
    X_comb_tr = np.hstack([X_base_tr, X_diff_tr])
    X_comb_va = np.hstack([X_base_va, X_diff_va])
    X_ext_tr  = np.hstack([X_base_tr, X_diff_tr, X_attr_tr])
    X_ext_va  = np.hstack([X_base_va, X_diff_va, X_attr_va])

    # ── Train all models ──────────────────────────────────────────────────────

    print("="*60 + "\n  LOGISTIC REGRESSION\n" + "="*60)
    lr_res  = run_lr( X_base_tr, X_base_va, X_attr_tr, X_attr_va,
                      X_diff_tr, X_diff_va, X_comb_tr, X_comb_va,
                      X_ext_tr,  X_ext_va,  y_train, y_valid)

    print("="*60 + "\n  RANDOM FOREST\n" + "="*60)
    rf_res  = run_rf( X_base_tr, X_base_va, X_attr_tr, X_attr_va,
                      X_diff_tr, X_diff_va, X_comb_tr, X_comb_va,
                      X_ext_tr,  X_ext_va,  y_train, y_valid)

    print("="*60 + "\n  MLP\n" + "="*60)
    mlp_res = run_mlp(X_base_tr, X_base_va, X_attr_tr, X_attr_va,
                      X_diff_tr, X_diff_va, X_comb_tr, X_comb_va,
                      X_ext_tr,  X_ext_va,  y_train, y_valid)

    # ── Per-model ROC plots (all 5 configs) ───────────────────────────────────

    plot_roc_five(lr_res,  'LR',  DISEASE, OUTPUT_DIR)
    plot_roc_five(rf_res,  'RF',  DISEASE, OUTPUT_DIR)
    plot_roc_five(mlp_res, 'MLP', DISEASE, OUTPUT_DIR)

    # ── Summary ROC — best config (ext) per model ─────────────────────────────

    plt.figure(figsize=(8, 7))
    plt.plot(lr_res['base']['fpr'], lr_res['base']['tpr'],
             color='gray',      lw=2, ls='--',
             label=f"Baseline (prob only)       AUC={lr_res['base']['auc']:.3f}")
    plt.plot(lr_res['ext']['fpr'],  lr_res['ext']['tpr'],
             color='steelblue',  lw=2,
             label=f"LR  prob + ΔA + attrs      AUC={lr_res['ext']['auc']:.3f}")
    plt.plot(rf_res['ext']['fpr'],  rf_res['ext']['tpr'],
             color='green',      lw=2,
             label=f"RF  prob + ΔA + attrs      AUC={rf_res['ext']['auc']:.3f}")
    plt.plot(mlp_res['ext']['fpr'], mlp_res['ext']['tpr'],
             color='darkorange', lw=2,
             label=f"MLP prob + ΔA + attrs      AUC={mlp_res['ext']['auc']:.3f}")
    plt.plot([0, 1], [0, 1], 'k:', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'C2 Quality Control — {DISEASE}\nAll Models — Extended Features (prob + ΔA + attrs)')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_best_per_model.png'), dpi=150)
    plt.close()

    # ── RF feature importances (extended model) ───────────────────────────────

    ext_feature_names = [disease_prob_col] + diff_cols + attr_cols
    importances = pd.Series(rf_res['ext']['model'].feature_importances_,
                            index=ext_feature_names).sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    sns.barplot(x=importances.index, y=importances.values)
    plt.xticks(rotation=90)
    plt.title(f'RF Feature Importances — Extended Model ({DISEASE})')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_feature_importances.png'), dpi=150)
    plt.close()

    # ── Summary table ─────────────────────────────────────────────────────────

    summary = (
        f"=== C2 Non-linear Model Results — {DISEASE} ===\n\n"
        f"Train : {len(train_df):,}  (correct: {y_train.sum():,} | incorrect: {(1-y_train).sum():,})\n"
        f"Valid : {len(valid_df):,}  (correct: {y_valid.sum():,} | incorrect: {(1-y_valid).sum():,})\n"
        f"ΔA features   : {len(diff_cols)}\n"
        f"Attr features : {len(attr_cols)}\n\n"
        f"| Model | Prob only | Attrs only | ΔA only | Prob + ΔA | Prob + ΔA + Attrs |\n"
        f"|-------|-----------|------------|---------|-----------|------------------|\n"
        f"| LR    | {lr_res['base']['auc']:.3f}     | {lr_res['attr']['auc']:.3f}      | {lr_res['diff']['auc']:.3f}   | {lr_res['comb']['auc']:.3f}     | {lr_res['ext']['auc']:.3f}             |\n"
        f"| RF    | {rf_res['base']['auc']:.3f}     | {rf_res['attr']['auc']:.3f}      | {rf_res['diff']['auc']:.3f}   | {rf_res['comb']['auc']:.3f}     | {rf_res['ext']['auc']:.3f}             |\n"
        f"| MLP   | {mlp_res['base']['auc']:.3f}     | {mlp_res['attr']['auc']:.3f}      | {mlp_res['diff']['auc']:.3f}   | {mlp_res['comb']['auc']:.3f}     | {mlp_res['ext']['auc']:.3f}             |\n\n"
        f"Top 10 RF feature importances (extended):\n{importances.head(10).to_string()}\n"
    )
    print("\n" + summary)
    with open(os.path.join(OUTPUT_DIR, 'results_summary_nonlinear.txt'), 'w') as f:
        f.write(summary)

    # ── Save models ───────────────────────────────────────────────────────────

    joblib.dump(rf_res['comb']['model'],  os.path.join(OUTPUT_DIR, 'rf_comb_model.pkl'))
    joblib.dump(rf_res['ext']['model'],   os.path.join(OUTPUT_DIR, 'rf_ext_model.pkl'))
    joblib.dump(mlp_res['comb']['model'], os.path.join(OUTPUT_DIR, 'mlp_comb_model.pkl'))
    joblib.dump(mlp_res['ext']['model'],  os.path.join(OUTPUT_DIR, 'mlp_ext_model.pkl'))

    print(f"\nAll outputs saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()