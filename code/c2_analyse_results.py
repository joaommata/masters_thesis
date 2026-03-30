"""
c2_analyse_results.py
=====================
Loads trained C2 models and produces:
  1. C2 output distribution plots (by TP/FN/TN/FP) for the M6 configuration.
  2. ROC curves (from saved roc_data.json) for all configurations but only a selected subset of configs per model 
  3. Rejection analysis plots (accuracy and FPR/FNR vs threshold) for the M6 configuration.

All outputs are saved to:
  results/C2_sim_cf/{DISEASE}/analysis/cf_{CF_COUNT}/

The CF_COUNT parameter must match the suffix used when preparing the data
Usage:
    Edit the CONFIG block, then run:
        python c2_analyse_results.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.stats import gaussian_kde

sys.path.append('/zhome/d0/a/221493/thesis/code')
from plot_config import PLOT_COLORS


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit this block only
# ══════════════════════════════════════════════════════════════════════════════

DISEASE    = 'effusion'
CF_COUNT   = 1                            # number of CF neighbours used in data prep
BASE_DIR   = '/zhome/d0/a/221493/thesis'
THRESHOLDS = np.arange(0.1, 1.0, 0.1)    # operating thresholds for rejection analysis

# Which configs to show in ROC plots (per model)
CONFIGS_TO_SHOW = {
    'LR':  ['B1', 'B2', 'B4', 'M3', 'M6'],
    'RF':  ['B1', 'B2', 'B4', 'M3', 'M6'],
    'MLP': ['B1', 'B2', 'B4', 'M3', 'M6'],
}

# ══════════════════════════════════════════════════════════════════════════════

# Derived paths
INPUT_DIR  = os.path.join(BASE_DIR, 'results', 'C2_sim_cf', DISEASE)
OUTPUT_DIR = os.path.join(INPUT_DIR, 'analysis', f'cf_{CF_COUNT}')
ROC_PATH   = os.path.join(INPUT_DIR, f'roc_data_{CF_COUNT}.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'distributions'), exist_ok=True)

disease_prob_col = f'{DISEASE}_prob'

# Plot style shared across all figures
COLORS = {**PLOT_COLORS, 'M6': '#CC0000'}
LINESTYLES = {k: '--' if k.startswith('B') else '-' for k in COLORS}
LINEWIDTHS = {k:  2   if k.startswith('B') else 3   for k in COLORS}
LABELS = {
    'B1': 'B1 (Prob)',
    'B2': 'B2 (Attrs)',
    'B4': 'B4 (Prob + Attrs)',
    'M3': 'M3 (Prob+ΔA+Attrs)',
    'M6': 'M6 (Prob+ΔA+Attrs+CFProb)',
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    """Load validation split. CSV filename includes CF_COUNT suffix."""
    valid_path = os.path.join(INPUT_DIR, f'valid_with_diff_vectors_{CF_COUNT}.csv')
    train_path = os.path.join(INPUT_DIR, f'train_with_diff_vectors_{CF_COUNT}.csv')
    valid_df = pd.read_csv(valid_path)
    train_df = pd.read_csv(train_path)
    print(f'Loaded valid: {len(valid_df):,} rows  |  train: {len(train_df):,} rows')
    return valid_df, train_df


def get_feature_matrices(valid_df, train_df):
    """
    Identify column groups and build the M6 feature matrix.
    cf_prob is excluded from attr_cols to avoid double-counting.
    """
    meta_cols = {
        disease_prob_col, f'{DISEASE}_pred', f'{DISEASE}_true',
        'correct', 'path', 'patient_id', 'cf_prob', 'cf_paths'
    }
    diff_cols = [c for c in train_df.columns if c.startswith('delta_')]
    emb_cols  = [c for c in train_df.columns if c.startswith('emb_')]
    attr_cols = [c for c in train_df.columns
                 if c not in meta_cols
                 and not c.startswith('delta_')
                 and not c.startswith('emb_')]

    print(f'Attr features : {len(attr_cols)}')
    print(f'ΔA features   : {len(diff_cols)}')

    prob_va    = valid_df[[disease_prob_col]].values
    diff_va    = valid_df[diff_cols].values
    attr_va    = valid_df[attr_cols].values
    cf_prob_va = valid_df[['cf_prob']].values

    X_M6 = np.hstack([prob_va, diff_va, attr_va, cf_prob_va])
    print(f'M6 feature matrix: {X_M6.shape}')

    return X_M6


# ── Model inference ───────────────────────────────────────────────────────────

def run_inference(X_M6):
    """Load saved models and produce probability scores for the validation set."""
    lr_m6        = joblib.load(os.path.join(INPUT_DIR, f'cf{CF_COUNT}_lr',  'lr_m6_model.pkl'))
    lr_m6_scaler = joblib.load(os.path.join(INPUT_DIR, f'cf{CF_COUNT}_lr',  'lr_m6_scaler.pkl'))
    mlp_m6        = joblib.load(os.path.join(INPUT_DIR, f'cf{CF_COUNT}_mlp', 'mlp_m6_model.pkl'))
    mlp_m6_scaler = joblib.load(os.path.join(INPUT_DIR, f'cf{CF_COUNT}_mlp', 'mlp_m6_scaler.pkl'))
    rf_m6         = joblib.load(os.path.join(INPUT_DIR, f'cf{CF_COUNT}_rf',  'rf_m6_model.pkl'))
    
    lr_probs  = lr_m6.predict_proba(lr_m6_scaler.transform(X_M6))[:, 1]
    mlp_probs = mlp_m6.predict_proba(mlp_m6_scaler.transform(X_M6.astype(np.float32)))[:, 1]
    rf_probs  = rf_m6.predict_proba(X_M6)[:, 1]

    print(f'LR  probs — min={lr_probs.min():.3f}  max={lr_probs.max():.3f}  mean={lr_probs.mean():.3f}')
    print(f'MLP probs — min={mlp_probs.min():.3f}  max={mlp_probs.max():.3f}  mean={mlp_probs.mean():.3f}')
    print(f'RF  probs — min={rf_probs.min():.3f}  max={rf_probs.max():.3f}  mean={rf_probs.mean():.3f}')

    return {'LR': lr_probs, 'MLP': mlp_probs, 'RF': rf_probs}


# ── Plot 1: C2 output distributions by TP/FN/TN/FP ──────────────────────────

def plot_distributions(valid_df, model_probs):
    """KDE plots of C2 output scores split by ground truth × correctness."""
    true_label = valid_df[f'{DISEASE}_true'].values
    correct    = valid_df['correct'].values

    tp_mask = (true_label == 1) & (correct == 1)
    fn_mask = (true_label == 1) & (correct == 0)
    tn_mask = (true_label == 0) & (correct == 1)
    fp_mask = (true_label == 0) & (correct == 0)

    print(f'TP: {tp_mask.sum()}  FN: {fn_mask.sum()}  TN: {tn_mask.sum()}  FP: {fp_mask.sum()}')

    x_grid          = np.linspace(0, 1, 300)
    COLOR_CORRECT   = PLOT_COLORS['M3']
    COLOR_INCORRECT = PLOT_COLORS['M6']

    def kde_curve(probs, mask, min_samples=5):
        vals = probs[mask]
        if len(vals) < min_samples or np.std(vals) < 1e-6:
            return np.zeros_like(x_grid)
        return gaussian_kde(vals, bw_method='scott')(x_grid)

    plt.rcParams.update({
    'font.size': 17,
    'axes.linewidth': 1.2,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
})

    for model_name, c2_probs in model_probs.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Disease present panel
        ax = axes[0]
        tp_kde = kde_curve(c2_probs, tp_mask)
        fn_kde = kde_curve(c2_probs, fn_mask)
        ax.plot(x_grid, tp_kde, color=COLOR_CORRECT,   lw=1.8, label=f'TP (n={tp_mask.sum()})')
        ax.fill_between(x_grid, tp_kde, alpha=0.15, color=COLOR_CORRECT)
        ax.plot(x_grid, fn_kde, color=COLOR_INCORRECT, lw=1.8, label=f'FN (n={fn_mask.sum()})')
        ax.fill_between(x_grid, fn_kde, alpha=0.15, color=COLOR_INCORRECT)
        ax.set_xlabel('C2 output')
        ax.set_ylabel('Density')
        ax.set_title(f'Disease present — {model_name} M6 (CF={CF_COUNT})')
        ax.set_xlim(0, 1)
        ax.legend()

        # Disease absent panel
        ax = axes[1]
        tn_kde = kde_curve(c2_probs, tn_mask)
        fp_kde = kde_curve(c2_probs, fp_mask)
        ax.plot(x_grid, tn_kde, color=COLOR_CORRECT,   lw=1.8, label=f'TN (n={tn_mask.sum()})')
        ax.fill_between(x_grid, tn_kde, alpha=0.15, color=COLOR_CORRECT)
        ax.plot(x_grid, fp_kde, color=COLOR_INCORRECT, lw=1.8, label=f'FP (n={fp_mask.sum()})')
        ax.fill_between(x_grid, fp_kde, alpha=0.15, color=COLOR_INCORRECT)
        ax.set_xlabel('C2 output')
        ax.set_ylabel('Density')
        ax.set_title(f'Disease absent — {model_name} M6 (CF={CF_COUNT})')
        ax.set_xlim(0, 1)
        ax.legend()

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, 'distributions',
                                 f'distribution_{model_name.lower()}_m6_cf{CF_COUNT}.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f'Saved → {save_path}')


# ── Plot 2: ROC curves ────────────────────────────────────────────────────────

def plot_roc_curves():
    """Load roc_data.json and produce a 3-panel ROC figure."""
    with open(ROC_PATH) as f:
        roc_data = json.load(f)

    plt.rcParams.update({
        'font.size': 13, 'axes.linewidth': 1.2,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    plt.rcParams.update({
    'font.size': 17,
    'axes.linewidth': 1.2,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    model_full_names = {'LR': 'Logistic Regression', 'RF': 'Random Forest', 'MLP': 'Multilayer Perceptron'}

    for ax, model_name in zip(axes, ['LR', 'RF', 'MLP']):
        for tag in CONFIGS_TO_SHOW[model_name]:
            d = roc_data[model_name][tag]
            ax.plot(d['fpr'], d['tpr'],
                    color=COLORS[tag], lw=LINEWIDTHS[tag], ls=LINESTYLES[tag],
                    label=f"{LABELS.get(tag, tag)}\nAUC = {d['auc']:.3f}")

        ax.plot([0, 1], [0, 1], 'k:', lw=0.8, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'C2 — {model_full_names[model_name]}\n{DISEASE.capitalize()} CF={CF_COUNT}')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc='lower right', prop={'family': 'monospace', 'size': 11},
                  framealpha=0.95, handlelength=2, labelspacing=0.8)
        ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'roc_all_models_cf{CF_COUNT}.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'Saved → {save_path}')


# ── Plot 3: Rejection analysis ────────────────────────────────────────────────

def plot_rejection_analysis(valid_df, model_probs):
    """
    For each threshold, treat cases where C2 score < threshold as rejected.
    Plot two figures:
      (a) C0 accuracy on accepted cases vs threshold
      (b) FPR and FNR on accepted cases vs threshold
    """
    c0_preds = valid_df[f'{DISEASE}_pred'].values
    c0_true  = valid_df[f'{DISEASE}_true'].values
    correct  = valid_df['correct'].values

    c0_accuracy = (c0_preds == c0_true).mean() * 100

    # Baseline error rates (before any rejection)
    tp_total = ((c0_preds == 1) & (c0_true == 1)).sum()
    fp_total = ((c0_preds == 1) & (c0_true == 0)).sum()
    tn_total = ((c0_preds == 0) & (c0_true == 0)).sum()
    fn_total = ((c0_preds == 0) & (c0_true == 1)).sum()
    baseline_fpr = fp_total / (fp_total + tn_total) * 100
    baseline_fnr = fn_total / (fn_total + tp_total) * 100

    print(f'Baseline C0 accuracy : {c0_accuracy:.2f}%')
    print(f'Baseline FPR         : {baseline_fpr:.2f}%')
    print(f'Baseline FNR         : {baseline_fnr:.2f}%')

    for model_name, c2_scores in model_probs.items():

        acc_rows, err_rows = [], []
        for thresh in THRESHOLDS:
            flagged  = c2_scores < thresh
            accepted = ~flagged

            # Accuracy plot data
            acc_correct = ((c0_preds == c0_true) & accepted).sum()
            acc_total   = accepted.sum()
            acc_rows.append({
                'threshold':         round(thresh, 2),
                'rejected_pct':      flagged.mean() * 100,
                'rejected_n':        flagged.sum(),
                'accepted_n':        acc_total,
                'accuracy_correct_n': acc_correct,
                'accuracy_total_n':   acc_total,
                'accepted_accuracy': acc_correct / acc_total * 100 if acc_total > 0 else 0,
            })

            # FPR/FNR plot data
            tp = ((c0_preds == 1) & (c0_true == 1) & accepted).sum()
            fp = ((c0_preds == 1) & (c0_true == 0) & accepted).sum()
            tn = ((c0_preds == 0) & (c0_true == 0) & accepted).sum()
            fn = ((c0_preds == 0) & (c0_true == 1) & accepted).sum()
            err_rows.append({
            'threshold':    round(thresh, 2),
            'rejected_pct': flagged.mean() * 100,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'fpr': fp / (fp + tn) * 100 if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) * 100 if (fn + tp) > 0 else 0,
        })

        acc_df = pd.DataFrame(acc_rows)
        err_df = pd.DataFrame(err_rows)

        plt.rcParams.update({
    'font.size': 17,
    'axes.linewidth': 1.2,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
})
        # ── Figure (a): accuracy ──────────────────────────────────────────────
        fig, ax1 = plt.subplots(figsize=(7, 6))
        ax2 = ax1.twinx()
        ax1.axhline(c0_accuracy, color='#999999', lw=1.5, ls='--',
                    label=f'Baseline C0 ({c0_accuracy:.1f}%)')
        ax1.plot(acc_df['threshold'], acc_df['accepted_accuracy'],
                 color=COLORS['M2'], lw=2.2, marker='o', markersize=6,
                 label='C0 accuracy on accepted')
        ax2.plot(acc_df['threshold'], acc_df['rejected_pct'],
                 color=COLORS['M4'], lw=2.2, marker='s', markersize=6,
                 ls='--', label='Cases rejected (%)')
        ax1.set_xlabel('C2 Threshold')
        ax1.set_ylabel('C0 Accuracy on Accepted Cases (%)')
        ax2.set_ylabel('Cases Rejected (%)')
        ax1.set_ylim(70, 100); ax2.set_ylim(0, 50)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.set_title(f'C2 Rejection Analysis — {DISEASE.capitalize()} {model_name} CF={CF_COUNT}')
        ax1.grid(alpha=0.3, linewidth=0.5)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR,
                                 f'rejection_accuracy_{model_name.lower()}_cf{CF_COUNT}.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f'Saved → {save_path}')

        # ── Figure (b): FPR / FNR ────────────────────────────────────────────
        fig, ax1 = plt.subplots(figsize=(7, 6))
        ax2 = ax1.twinx()
        ax1.axhline(baseline_fpr, color=COLORS['M6'], lw=1.2, ls=':',
                    alpha=0.7, label=f'Baseline FPR ({baseline_fpr:.1f}%)')
        ax1.axhline(baseline_fnr, color=COLORS['M1'], lw=1.2, ls=':',
                    alpha=0.7, label=f'Baseline FNR ({baseline_fnr:.1f}%)')
        ax1.plot(err_df['threshold'], err_df['fpr'],
                 color=COLORS['M6'], lw=2.2, marker='o', markersize=6,
                 label='FPR after rejection')
        ax1.plot(err_df['threshold'], err_df['fnr'],
                 color=COLORS['M1'], lw=2.2, marker='o', markersize=6,
                 label='FNR after rejection')
        ax2.plot(err_df['threshold'], err_df['rejected_pct'],
                 color='#999999', lw=1.8, marker='s', markersize=5,
                 ls='--', label='Cases rejected (%)')
        ax1.set_xlabel('C2 Threshold')
        ax1.set_ylabel('Error Rate on Accepted Cases (%)')
        ax2.set_ylabel('Cases Rejected (%)')
        ax1.set_ylim(0, 35); ax2.set_ylim(0, 50)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
        ax1.set_title(f'C2 Clinical Rejection — {DISEASE.capitalize()} {model_name} CF={CF_COUNT}')
        ax1.grid(alpha=0.3, linewidth=0.5)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR,
                                 f'rejection_clinical_{model_name.lower()}_cf{CF_COUNT}.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f'Saved → {save_path}')

        # ── Save summary table ────────────────────────────────────────────────────────
        summary_df = acc_df.merge(err_df[['threshold', 'fpr', 'fnr']], on='threshold')
        summary_df.insert(0, 'model', model_name)
        summary_df.insert(1, 'cf_count', CF_COUNT)
        save_path = os.path.join(OUTPUT_DIR, f'rejection_summary_{model_name.lower()}_cf{CF_COUNT}.csv')
        summary_df.to_csv(save_path, index=False)
        print(f'Saved → {save_path}')

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f'Disease  : {DISEASE}')
    print(f'CF count : {CF_COUNT}')
    print(f'Input    : {INPUT_DIR}')
    print(f'Output   : {OUTPUT_DIR}\n')

    valid_df, train_df = load_data()
    X_M6               = get_feature_matrices(valid_df, train_df)
    model_probs        = run_inference(X_M6)

    print('\n── Distributions ────────────────────────────────────')
    plot_distributions(valid_df, model_probs)

    print('\n── ROC curves ───────────────────────────────────────')
    plot_roc_curves()

    print('\n── Rejection analysis ───────────────────────────────')
    plot_rejection_analysis(valid_df, model_probs)

    print(f'\nDone. All outputs saved → {OUTPUT_DIR}')


if __name__ == '__main__':
    main()