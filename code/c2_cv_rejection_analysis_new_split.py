"""
plot_risk_coverage.py
=====================
Generates a 3-panel Risk-Coverage curve figure (RF | LR | MLP)
for a given disease and CF count.

Requires: fold_*_predictions.csv files in the cv_results directory.
These contain per-sample C2 scores for every model/config combination.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from plot_config import PLOT_COLORS

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DISEASE   = 'effusion'
CF_COUNT  = 16
BASE_DIR  = '/zhome/d0/a/221493/thesis/results'

CONFIGS_TO_SHOW = ['B1', 'B2', 'B4', 'M3', 'M6']

COLORS = {**PLOT_COLORS, 'M6': '#CC0000'}

LINESTYLES = {k: '--' if k.startswith('B') else '-' for k in COLORS} # Baselines dashed, models solid
LINEWIDTHS = {k: 2 if k.startswith('B') else 3 for k in COLORS} # Baselines thinner, models thickerr

plt.rcParams.update({
    'font.size': 17,
    'axes.linewidth': 1.2,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
})


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTION: compute risk-coverage curve from per-sample scores
# ══════════════════════════════════════════════════════════════════════════════

def compute_risk_coverage(c2_scores, correct):
    """
    Given per-sample C2 confidence scores and binary correctness labels,
    compute the risk-coverage curve by sweeping the acceptance threshold.

    Strategy: sort samples by C2 score descending (most confident first).
    At each step k, accept the top-k samples and compute:
        coverage = k / n
        risk     = error rate on those k samples

    This gives a curve from coverage≈0 (accept only 1 sample) to coverage=1 (accept all).

    Parameters
    ----------
    c2_scores : np.array of shape (n,)
    correct   : np.array of shape (n,), binary (1=correct, 0=incorrect)

    Returns
    -------
    coverage : np.array
    risk     : np.array
    aurc     : float
    """
    n = len(c2_scores)

    # Sort by descending confidence — most confident samples accepted first
    order      = np.argsort(c2_scores)[::-1]
    correct_sorted = correct[order]

    # Cumulative error count as we add more samples
    cumulative_errors = np.cumsum(1 - correct_sorted)

    # At position k (1-indexed): accepted k samples
    k_values = np.arange(1, n + 1)
    coverage = k_values / n
    risk     = cumulative_errors / k_values  # error rate on accepted set

    aurc = auc(coverage, risk)
    return coverage, risk, aurc

def compute_selective_accuracy(c2_scores, correct):
    """Selective accuracy = accuracy on accepted set, sorted by descending confidence."""
    n = len(c2_scores)
    order = np.argsort(c2_scores)[::-1]
    correct_sorted = correct[order]
    
    cumulative_correct = np.cumsum(correct_sorted)
    k_values = np.arange(1, n + 1)
    
    coverage = k_values / n
    sel_accuracy = cumulative_correct / k_values  # accuracy on accepted set
    

    # AUSAC: Area Under Selective Accuracy-Coverage curve (higher = better)
    ausac = auc(coverage, sel_accuracy)
    return coverage, sel_accuracy, ausac

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── Load data ─────────────────────────────────────────────────────────
    cv_dir = os.path.join(BASE_DIR, f'C2_custom/{DISEASE}/cv_results/cf_{CF_COUNT}')
    fold_files = sorted([
        f for f in os.listdir(cv_dir)
        if f.startswith('fold_') and f.endswith('_predictions.csv')
    ])
    df = pd.concat(
        [pd.read_csv(os.path.join(cv_dir, f)) for f in fold_files],
        ignore_index=True
    )
    correct = df['correct'].values

    # ── Risk-Coverage plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, model_type in zip(axes, ['RF', 'LR', 'MLP']):
        for config in CONFIGS_TO_SHOW:
            prob_col = f'{model_type}_{config}_prob'
            if prob_col not in df.columns:
                continue
            coverage, risk, aurc = compute_risk_coverage(df[prob_col].values, correct)
            ax.plot(coverage, risk, color=COLORS[config], ls=LINESTYLES[config],
                    lw=2, label=f'{config} ({aurc:.4f})')
        ax.set_title(f'{model_type} (k={CF_COUNT})')
        ax.set_xlabel('Coverage %')
        ax.set_ylabel('Risk (Error Rate)')
        ax.set_xlim(0, 1)
        ax.legend(title='Config (AURC)', fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, f'risk_coverage_cf{CF_COUNT}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Risk-Coverage curves saved to {cv_dir}")
    
    # ── Selective Accuracy-Coverage plot ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, model_type in zip(axes, ['RF', 'LR', 'MLP']):
        for config in CONFIGS_TO_SHOW:
            prob_col = f'{model_type}_{config}_prob'
            if prob_col not in df.columns:
                continue
            coverage, sel_acc, ausac = compute_selective_accuracy(df[prob_col].values, correct)
            ax.plot(coverage, sel_acc, color=COLORS[config], ls=LINESTYLES[config],
                    lw=2, label=f'{config} ({ausac:.4f})')
        baseline_acc = correct.mean()
        ax.axhline(baseline_acc, color='black', lw=1, ls=':', alpha=0.5,
                   label=f'Baseline ({baseline_acc:.3f})')
        ax.set_title(f'{model_type} (k={CF_COUNT})')
        ax.set_xlabel('Coverage %')
        ax.set_ylabel('Selective Accuracy')
        ax.set_xlim(0, 1); ax.set_ylim(0.9, 1.0)
        ax.legend(title='Config (AUSAC)', fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, f'selective_accuracy_cf{CF_COUNT}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Selective Accuracy-Coverage curves saved to {cv_dir}")

if __name__ == "__main__":
    main()