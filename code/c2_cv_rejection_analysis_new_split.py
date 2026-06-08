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
COLORS = {config: PLOT_COLORS[config] for config in CONFIGS_TO_SHOW}

LINESTYLES = {k: '--' if k.startswith('B') else '-' for k in COLORS} # Baselines dashed, models solid
LINEWIDTHS = {k: 2 if k.startswith('B') else 3 for k in COLORS} # Baselines thinner, models thickerr

plt.style.use('seaborn-v0_8-white')
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 16,
    'axes.titlesize': 17, 'axes.labelsize': 16,
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 1.2, 'grid.alpha': 0.3,
    'grid.linewidth': 0.6,
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

def make_clinical_table(c2_scores, correct, n_total, thresholds, output_path):
    """
    For a set of thresholds, compute clinical deferral statistics.
    Assumes rejected cases are deferred to a specialist (always correct).
    
    Parameters
    ----------
    c2_scores  : np.array, C2 confidence scores
    correct    : np.array, binary (1=correct, 0=incorrect)
    n_total    : int, total number of cases
    thresholds : list of floats, C2 score cutoffs to evaluate
    output_path: str, where to save the CSV
    """
    rows = []
    baseline_errors = (correct == 0).sum()  # total errors C0 makes with no deferral

    for thresh in thresholds:
        accepted_mask = c2_scores >= thresh
        rejected_mask = ~accepted_mask

        n_accepted = accepted_mask.sum()
        n_deferred = rejected_mask.sum()

        errors_on_accepted  = (correct[accepted_mask] == 0).sum()  # C0 mistakes that slip through
        errors_avoided      = (correct[rejected_mask] == 0).sum()  # C0 mistakes caught by deferral
        correct_deferred    = (correct[rejected_mask] == 1).sum()  # unnecessary deferrals (cost)

        rows.append({
            'Threshold':            round(thresh, 2),
            'Cases Accepted':       n_accepted,
            'Cases Deferred (%)':   f"{n_deferred} ({n_deferred/n_total*100:.1f}%)",
            'Errors on Accepted':   errors_on_accepted,
            'Errors Avoided':       errors_avoided,
            'Unnecessary Deferrals': correct_deferred,
            'Error Rate Accepted':  f"{errors_on_accepted/n_accepted*100:.2f}%" if n_accepted > 0 else 'N/A',
        })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(output_path, index=False)
    print(table_df.to_string(index=False))
    print(f"\nClinical table saved to {output_path}")
    return table_df

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
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, model_type in zip(axes, ['RF', 'LR', 'MLP']):
        for config in CONFIGS_TO_SHOW:
            prob_col = f'{model_type}_{config}_prob'
            if prob_col not in df.columns:
                continue
            coverage, risk, aurc = compute_risk_coverage(df[prob_col].values, correct)
            ax.plot(coverage, risk, color=COLORS[config], ls=LINESTYLES[config],
                    lw=2, label=f'{config} ({aurc:.4f})')
        ax.set_title(f'Risk-Coverage {model_type}, k={CF_COUNT}')
        ax.set_xlabel('Coverage %')
        ax.set_ylabel('Risk (Error Rate)')  # risk plot
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        ax.spines['left'].set_position(('outward', 10))
        ax.legend(title='AURC',
                  loc='upper left',
                  prop={'family': 'monospace', 'size': 13},
                  framealpha=0.95, edgecolor='#cccccc', handlelength=2)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, f'risk_coverage_cf{CF_COUNT}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Risk-Coverage curves saved to {cv_dir}")

    # ── Selective Accuracy-Coverage plot ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
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
        ax.set_title(f'Selective Accuracy {model_type}, k={CF_COUNT}')
        ax.set_xlabel('Coverage %')
        ax.set_ylabel('Selective Accuracy')  # selective accuracy plot
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        ax.set_ylim(0.9, 1.0)
        ax.spines['left'].set_position(('outward', 10))
        ax.legend(title='AUSAC',
                  loc='lower left',
                  prop={'family': 'monospace', 'size': 13},
                  framealpha=0.95, edgecolor='#cccccc', handlelength=2)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, f'selective_accuracy_cf{CF_COUNT}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Selective Accuracy-Coverage curves saved to {cv_dir}")
    
    # ── Clinical deferral table (LR M6) ──────────────────────────────────
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    c2_scores  = df['LR_M6_prob'].values
    
    baseline_errors = (correct == 0).sum()
    baseline_total  = len(correct)
    baseline_rate   = baseline_errors / baseline_total * 100
    print(f"Baseline: {baseline_errors} errors / {baseline_total} cases ({baseline_rate:.2f}%)")
    
    make_clinical_table(
        c2_scores   = c2_scores,
        correct     = correct,
        n_total     = len(df),
        thresholds  = thresholds,
        output_path = os.path.join(cv_dir, f'clinical_table_LR_M6_cf{CF_COUNT}.csv')
    )
if __name__ == "__main__":
    main()