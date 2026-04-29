"""
cv_analyse_results.py
=====================
Comprehensive analysis of cross-validation results for C2 quality control.
This script essentially repeats the analysis of cf_rocs.py and c2_analyse_Results.py but automated to average results across folds and CF counts.

Produces:
  1. ROC curves with confidence bands (mean ± std across folds) (~ c2_analyse_results.py)
  2. C2 output distributions (TP/FN/TN/FP) for selected configs (~ c2_analyse_results.py)
  3. Rejection analysis (accuracy and error rates vs threshold) (~ c2_analyse_results.py)
  4. Comparison across different CF counts (~ c2_multiple_cf.py)

Usage:
    python cv_analyse_results.py --disease effusion --cf_counts 1 3 5 7 9

Outputs saved to:
    results/C2_sim_cf/{disease}/cv_analysis/
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy import interpolate

sys.path.append('/zhome/d0/a/221493/thesis/code')
from plot_config import PLOT_COLORS


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = '/zhome/d0/a/221493/thesis/'

# Which configs to show in ROC plots (per model), includes 3 baselines and 2 method
CONFIGS_TO_SHOW = {
    'LR':  ['B1', 'B2', 'B4', 'M3', 'M6'],
    'RF':  ['B1', 'B2', 'B4', 'M3', 'M6'],
    'MLP': ['B1', 'B2', 'B4', 'M3', 'M6'],
}

THRESHOLDS = np.arange(0.1, 1.0, 0.1)

# Plot styling
COLORS = {**PLOT_COLORS, 'M6': '#CC0000'}
LINESTYLES = {k: '--' if k.startswith('B') else '-' for k in COLORS} # Baselines dashed, models solid
LINEWIDTHS = {k: 2 if k.startswith('B') else 3 for k in COLORS} # Baselines thinner, models thicker
LABELS = { 
    'B1': 'B1 (Prob)',
    'B2': 'B2 (Attrs)',
    'B4': 'B4 (Prob + Attrs)',
    'M3': 'M3 (Prob+ΔA+Attrs)',
    'M6': 'M6 (Prob+ΔA+Attrs+CFProb)',
}

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
# ROC CURVE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def interpolate_roc_curve(fpr, tpr, common_fpr):
    """Interpolate ROC curve to common FPR points."""
    # I cannot average TPR values directly because they are from different FPR values.
    # Each fold produces ROC points at different FPR thresholds, so we need to interpolate TPR values at a common set of FPR points.
    f = interpolate.interp1d(fpr, tpr, kind='linear', 
                             bounds_error=False, fill_value=(0, 1))
    return f(common_fpr)


def compute_mean_roc(fold_results):
    """
    Compute mean ROC curve ± std across folds.
    
    Returns
    -------
    dict with keys: fpr, tpr_mean, tpr_std, auc_mean, auc_std
    """
    common_fpr = np.linspace(0, 1, 100)
    tpr_folds = []
    aucs = []
    
    # For each fold, get the FPR/TPR points, interpolate TPR at common FPR, and store AUC
    for fold_res in fold_results:
        fpr = np.array(fold_res['fpr'])
        tpr = np.array(fold_res['tpr'])
        tpr_interp = interpolate_roc_curve(fpr, tpr, common_fpr)
        
        tpr_folds.append(tpr_interp)
        aucs.append(fold_res['auc'])
    
    tpr_array = np.array(tpr_folds)
    
    # Then we can compute mean and std of TPR at each common FPR point, as well as mean and std of AUC across folds.
    return {
        'fpr': common_fpr,
        'tpr_mean': np.mean(tpr_array, axis=0),
        'tpr_std': np.std(tpr_array, axis=0),
        'auc_mean': np.mean(aucs),
        'auc_std': np.std(aucs),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: ROC CURVES PER MODEL (3-panel figure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(cv_results, disease, cf_count, output_dir):
    """
    3-panel ROC plot (LR | RF | MLP) with mean curves + confidence bands.
    """
    
    # The goal of thsi panel is to compare model architectures (LR vs RF vs MLP) while keeping the config fixed (e.g., M6) and CF count fixed.
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    model_names = {'LR': 'Logistic Regression', 'RF': 'Random Forest', 
                   'MLP': 'Multilayer Perceptron'}
    
    for ax, model_type in zip(axes, ['LR', 'RF', 'MLP']):
        
        # Access the fold results, compute mean ROC, and plot with confidence band for each config to show.
        for config in CONFIGS_TO_SHOW[model_type]:
            fold_results = cv_results[model_type][config]
            roc = compute_mean_roc(fold_results)
            
            # Plot mean curve
            ax.plot(roc['fpr'], roc['tpr_mean'],
                   color=COLORS[config], 
                   lw=LINEWIDTHS[config], 
                   ls=LINESTYLES[config],
                   label=f"{LABELS.get(config, config)}\n"
                         f"AUC = {roc['auc_mean']:.3f} ± {roc['auc_std']:.3f}")
            
            # Add confidence band (±1 std)
            ax.fill_between(roc['fpr'], 
                           roc['tpr_mean'] - roc['tpr_std'],
                           roc['tpr_mean'] + roc['tpr_std'],
                           color=COLORS[config], alpha=0.15)
        
        # Diagonal reference
        ax.plot([0, 1], [0, 1], 'k:', lw=0.8, alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_names[model_type]}\n{disease.capitalize()} CF={cf_count}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', 
                 prop={'family': 'monospace', 'size': 11},
                 framealpha=0.95, handlelength=2, labelspacing=0.8)
        ax.grid(alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'roc_all_models_cf{cf_count}.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved ROC curves → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: CF COUNT COMPARISON (for a single config, e.g., M6)
# ══════════════════════════════════════════════════════════════════════════════

def plot_cf_count_comparison(all_cv_results, disease, output_dir, 
                              config='M6', model_type='MLP'):
    """
    Compare ROC curves across different CF counts for one config/model.
    Here chosen to be M6 because it was the best performing in CF=1 across model types
    
    Parameters
    ----------
    all_cv_results : dict
        {cf_count: cv_results_dict}
    """
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    colors_cf = {
        1:  '#009E73',   # teal
        3:  '#0072B2',   # deep blue
        5:  '#56B4E9',   # sky blue
        7:  '#E69F00',   # amber
        9:  '#D55E00',   # vermillion
        11: '#CC0000',   # red
        13: '#CC79A7',   # mauve/pink
        15: '#6C01D7',   # purple
    }
    
    for cf_count in sorted(all_cv_results.keys()):
        cv_res = all_cv_results[cf_count]
        fold_results = cv_res[model_type][config]
        roc = compute_mean_roc(fold_results)
        
        color = colors_cf.get(cf_count, '#999999')
        
        ax.plot(roc['fpr'], roc['tpr_mean'],
               color=color, lw=2.5,
               label=f"{config} ({cf_count} CF)\n"
                     f"AUC = {roc['auc_mean']:.3f} ± {roc['auc_std']:.3f}")
        
        ax.fill_between(roc['fpr'],
                       roc['tpr_mean'] - roc['tpr_std'],
                       roc['tpr_mean'] + roc['tpr_std'],
                       color=color, alpha=0.15)
    
    ax.plot([0, 1], [0, 1], 'k:', lw=0.8, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Effect of CF count ({model_type} {config})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', prop={'family': 'monospace', 'size': 12},
             framealpha=0.95, handlelength=2, labelspacing=0.8)
    ax.grid(alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 
                            f'roc_cf_count_comparison_{model_type}_{config}.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved CF count comparison → {save_path}')

def plot_auc_vs_k(all_cv_results, disease, output_dir, config='M6', model_type='LR'):
    """
    Plot mean AUC vs CF count (k) as a simple line plot with error bars.
    Cleaner alternative to overlapping ROC curves for showing the k sweep trend.
    """
    
    ks, auc_means, auc_stds = [], [], []
    
    for cf_count in sorted(all_cv_results.keys()):
        fold_results = all_cv_results[cf_count][model_type][config]
        aucs = [f['auc'] for f in fold_results]
        ks.append(cf_count)
        auc_means.append(np.mean(aucs))
        auc_stds.append(np.std(aucs))
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    ax.plot(ks, auc_means, color=PLOT_COLORS['M6'], lw=2.5, 
            marker='o', markersize=7)
    
    ax.fill_between(ks,
                    np.array(auc_means) - np.array(auc_stds),
                    np.array(auc_means) + np.array(auc_stds),
                    color=PLOT_COLORS['M6'], alpha=0.15)
    
    ax.set_xlabel('Number of Counterfactuals (k)')
    ax.set_ylabel('Mean AUROC (5-fold CV)')
    ax.set_title(f'Effect of CF count on AUROC ({model_type} {config})')
    ax.set_xticks(ks)
    ax.grid(alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 
                             f'auc_vs_k_{model_type}_{config}.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved AUC vs k → {save_path}')
    
# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: C2 OUTPUT DISTRIBUTIONS (TP/FN/TN/FP)
# ══════════════════════════════════════════════════════════════════════════════

def plot_distributions(aggregated_df, disease, cf_count, output_dir, 
                       model_type='MLP', config='M6'):
    """
    KDE plots of C2 scores split by ground truth × correctness.
    Uses aggregated predictions across all folds.
    """
    
    prob_col = f'{model_type}_{config}_prob'
    
    if prob_col not in aggregated_df.columns:
        print(f"⚠ Column {prob_col} not found in aggregated predictions")
        return
    
    true_label = aggregated_df[f'{disease}_true'].values
    correct = aggregated_df['correct'].values
    c2_probs = aggregated_df[prob_col].values
    
    tp_mask = (true_label == 1) & (correct == 1)
    fn_mask = (true_label == 1) & (correct == 0)
    tn_mask = (true_label == 0) & (correct == 1)
    fp_mask = (true_label == 0) & (correct == 0)
    
    print(f'Distribution counts: TP={tp_mask.sum()} FN={fn_mask.sum()} '
          f'TN={tn_mask.sum()} FP={fp_mask.sum()}')
    
    x_grid = np.linspace(0, 1, 300)
    COLOR_CORRECT = PLOT_COLORS['M3']
    COLOR_INCORRECT = PLOT_COLORS['M6']
    
    def kde_curve(probs, mask, min_samples=10):
        vals = probs[mask]
        if len(vals) < min_samples or np.std(vals) < 1e-6:
            return np.zeros_like(x_grid)
        return gaussian_kde(vals, bw_method='scott')(x_grid)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Disease present
    ax = axes[0]
    tp_kde = kde_curve(c2_probs, tp_mask)
    fn_kde = kde_curve(c2_probs, fn_mask)
    ax.plot(x_grid, tp_kde, color=COLOR_CORRECT, lw=1.8, 
           label=f'TP (n={tp_mask.sum()})')
    ax.fill_between(x_grid, tp_kde, alpha=0.15, color=COLOR_CORRECT)
    ax.plot(x_grid, fn_kde, color=COLOR_INCORRECT, lw=1.8, 
           label=f'FN (n={fn_mask.sum()})')
    ax.fill_between(x_grid, fn_kde, alpha=0.15, color=COLOR_INCORRECT)
    ax.set_xlabel('C2 output')
    ax.set_ylabel('Density')
    ax.set_title(f'Disease present — {model_type} {config} (CF={cf_count})')
    ax.set_xlim(0, 1)
    ax.legend()
    
    # Disease absent
    ax = axes[1]
    tn_kde = kde_curve(c2_probs, tn_mask)
    fp_kde = kde_curve(c2_probs, fp_mask)
    ax.plot(x_grid, tn_kde, color=COLOR_CORRECT, lw=1.8, 
           label=f'TN (n={tn_mask.sum()})')
    ax.fill_between(x_grid, tn_kde, alpha=0.15, color=COLOR_CORRECT)
    ax.plot(x_grid, fp_kde, color=COLOR_INCORRECT, lw=1.8, 
           label=f'FP (n={fp_mask.sum()})')
    ax.fill_between(x_grid, fp_kde, alpha=0.15, color=COLOR_INCORRECT)
    ax.set_xlabel('C2 output')
    ax.set_ylabel('Density')
    ax.set_title(f'Disease absent — {model_type} {config} (CF={cf_count})')
    ax.set_xlim(0, 1)
    ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 
                            f'distribution_{model_type}_{config}_cf{cf_count}.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved distributions → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: REJECTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def plot_rejection_analysis(disease, cf_count, output_dir,
                            model_type='MLP', config='M6',
                            fold_files=None, cv_dir=None):
    """
    Accuracy and error rates vs C2 threshold, averaged across folds with shading.
    Each fold contributes one curve; we plot mean ± std.
    """

    prob_col = f'{model_type}_{config}_prob'

    # Storage: for each threshold, one value per fold
    acc_per_fold   = {round(t, 2): [] for t in THRESHOLDS}
    fpr_per_fold   = {round(t, 2): [] for t in THRESHOLDS}
    fnr_per_fold   = {round(t, 2): [] for t in THRESHOLDS}
    rej_per_fold   = {round(t, 2): [] for t in THRESHOLDS}
    baseline_accs, baseline_fprs, baseline_fnrs = [], [], []

    for fname in fold_files:
        df = pd.read_csv(os.path.join(cv_dir, fname))

        if prob_col not in df.columns:
            print(f"⚠ {prob_col} not found in {fname}, skipping")
            continue

        preds  = df[f'{disease}_pred'].values
        true   = df[f'{disease}_true'].values
        scores = df[prob_col].values

        # Baseline (no rejection) for this fold
        tp = ((preds==1)&(true==1)).sum(); fp = ((preds==1)&(true==0)).sum()
        tn = ((preds==0)&(true==0)).sum(); fn = ((preds==0)&(true==1)).sum()
        baseline_accs.append((preds==true).mean()*100)
        baseline_fprs.append(fp/(fp+tn)*100 if (fp+tn)>0 else 0)
        baseline_fnrs.append(fn/(fn+tp)*100 if (fn+tp)>0 else 0)

        # Rejection curve for this fold
        for thresh in THRESHOLDS:
            t = round(thresh, 2)
            accepted = scores >= thresh   # C2 accepts cases it's confident are correct

            tp = ((preds==1)&(true==1)&accepted).sum()
            fp = ((preds==1)&(true==0)&accepted).sum()
            tn = ((preds==0)&(true==0)&accepted).sum()
            fn = ((preds==0)&(true==1)&accepted).sum()
            n_acc = accepted.sum()

            acc_per_fold[t].append((preds==true)[accepted].mean()*100 if n_acc>0 else np.nan)
            fpr_per_fold[t].append(fp/(fp+tn)*100 if (fp+tn)>0 else np.nan)
            fnr_per_fold[t].append(fn/(fn+tp)*100 if (fn+tp)>0 else np.nan)
            rej_per_fold[t].append((~accepted).mean()*100)

    # Collapse to arrays indexed by threshold
    thresholds = [round(t, 2) for t in THRESHOLDS]
    acc_mean  = np.array([np.nanmean(acc_per_fold[t]) for t in thresholds])
    acc_std   = np.array([np.nanstd(acc_per_fold[t])  for t in thresholds])
    fpr_mean  = np.array([np.nanmean(fpr_per_fold[t]) for t in thresholds])
    fpr_std   = np.array([np.nanstd(fpr_per_fold[t])  for t in thresholds])
    fnr_mean  = np.array([np.nanmean(fnr_per_fold[t]) for t in thresholds])
    fnr_std   = np.array([np.nanstd(fnr_per_fold[t])  for t in thresholds])
    rej_mean  = np.array([np.nanmean(rej_per_fold[t]) for t in thresholds])

    b_acc = np.mean(baseline_accs); b_acc_std = np.std(baseline_accs)
    b_fpr = np.mean(baseline_fprs); b_fpr_std = np.std(baseline_fprs)
    b_fnr = np.mean(baseline_fnrs); b_fnr_std = np.std(baseline_fnrs)

    # ── Figure (a): Accuracy ──
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax2 = ax1.twinx()

    ax1.axhline(b_acc, color='#999999', lw=1.5, ls='--',
                label=f'Baseline C0 ({b_acc:.1f} ± {b_acc_std:.1f}%)')
    ax1.fill_between(thresholds, b_acc-b_acc_std, b_acc+b_acc_std,
                     color='#999999', alpha=0.15)

    ax1.plot(thresholds, acc_mean, color=COLORS['M2'], lw=2.2,
             marker='o', markersize=6, label='Accepted accuracy')
    ax1.fill_between(thresholds, acc_mean-acc_std, acc_mean+acc_std,
                     color=COLORS['M2'], alpha=0.2)

    ax2.plot(thresholds, rej_mean, color=COLORS['M4'], lw=2.2,
             marker='s', markersize=6, ls='--', label='Cases rejected (%)')

    ax1.set_xlabel('C2 Threshold')
    ax1.set_ylabel('C0 Accuracy on Accepted Cases (%)')
    ax2.set_ylabel('Cases Rejected (%)')
    ax1.set_ylim(70, 100); ax2.set_ylim(0, 50)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='center right')
    ax1.set_title(f'Rejection Analysis — {disease.capitalize()} {model_type} CF={cf_count}')
    ax1.grid(alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rejection_accuracy_{model_type}_{config}_cf{cf_count}.png'),
                dpi=600, bbox_inches='tight')
    plt.close()

    # ── Figure (b): FPR / FNR ──
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax2 = ax1.twinx()

    ax1.axhline(b_fpr, color=COLORS['M6'], lw=1.5, ls=':',
                label=f'Baseline FPR ({b_fpr:.1f} ± {b_fpr_std:.1f}%)')
    ax1.fill_between(thresholds, b_fpr-b_fpr_std, b_fpr+b_fpr_std,
                     color=COLORS['M6'], alpha=0.1)
    ax1.axhline(b_fnr, color=COLORS['M1'], lw=1.5, ls=':',
                label=f'Baseline FNR ({b_fnr:.1f} ± {b_fnr_std:.1f}%)')
    ax1.fill_between(thresholds, b_fnr-b_fnr_std, b_fnr+b_fnr_std,
                     color=COLORS['M1'], alpha=0.1)

    ax1.plot(thresholds, fpr_mean, color=COLORS['M6'], lw=2.2,
             marker='o', markersize=6, label='FPR after rejection')
    ax1.fill_between(thresholds, fpr_mean-fpr_std, fpr_mean+fpr_std,
                     color=COLORS['M6'], alpha=0.2)
    ax1.plot(thresholds, fnr_mean, color=COLORS['M1'], lw=2.2,
             marker='o', markersize=6, label='FNR after rejection')
    ax1.fill_between(thresholds, fnr_mean-fnr_std, fnr_mean+fnr_std,
                     color=COLORS['M1'], alpha=0.2)

    ax2.plot(thresholds, rej_mean, color='#999999', lw=1.8,
             marker='s', markersize=5, ls='--', label='Cases rejected (%)')

    ax1.set_xlabel('C2 Threshold')
    ax1.set_ylabel('Error Rate on Accepted Cases (%)')
    ax2.set_ylabel('Cases Rejected (%)')
    ax1.set_ylim(0, 35); ax2.set_ylim(0, 50)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper right', fontsize=10)
    ax1.set_title(f'Clinical Rejection — {disease.capitalize()} {model_type} CF={cf_count}')
    ax1.grid(alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rejection_clinical_{model_type}_{config}_cf{cf_count}.png'),
                dpi=600, bbox_inches='tight')
    plt.close()
    
        # Save summary table with mean ± std per threshold
    summary_rows = []
    for i, t in enumerate(thresholds):
        summary_rows.append({
            'threshold':   t,
            'rejected_pct': rej_mean[i],
            'acc_mean':    acc_mean[i],
            'acc_std':     acc_std[i],
            'fpr_mean':    fpr_mean[i],
            'fpr_std':     fpr_std[i],
            'fnr_mean':    fnr_mean[i],
            'fnr_std':     fnr_std[i],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.insert(0, 'model', model_type)
    summary_df.insert(1, 'config', config)
    summary_df.insert(2, 'cf_count', cf_count)

    # Add baseline as a separate row at the top for easy reference
    baseline_row = pd.DataFrame([{
        'model': model_type, 'config': config, 'cf_count': cf_count,
        'threshold': 0.0, 'rejected_pct': 0.0,
        'acc_mean': b_acc, 'acc_std': b_acc_std,
        'fpr_mean': b_fpr, 'fpr_std': b_fpr_std,
        'fnr_mean': b_fnr, 'fnr_std': b_fnr_std,
    }])
    summary_df = pd.concat([baseline_row, summary_df], ignore_index=True)

    summary_df.to_csv(
        os.path.join(output_dir, f'rejection_summary_{model_type}_{config}_cf{cf_count}.csv'),
        index=False
    )
    print(f'✓ Saved rejection plots for {model_type} {config} CF={cf_count}')
    
# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease', type=str, default='effusion')
    parser.add_argument('--cf_counts', type=int, nargs='+', required=True, default=[1, 3, 5, 7, 9],
                       help='List of CF counts to analyze (e.g., 1 3 5 7 9)')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['LR', 'RF', 'MLP'],
                       help='Which models to analyze')
    args = parser.parse_args()
    
    disease = args.disease
    cf_counts = args.cf_counts
    
    print(f"\n{'='*70}")
    print(f"  CV ANALYSIS: {disease.upper()}")
    print(f"  CF counts: {cf_counts}")
    print(f"{'='*70}\n")
    
    # Output directory
    output_base = os.path.join(BASE_DIR, f'results/C2_sim_cf/{disease}/cv_analysis')
    os.makedirs(output_base, exist_ok=True)
    
    # ── Load all CV results ───────────────────────────────────────────────
    all_cv_results = {}
    
    for cf_count in cf_counts:
        cv_dir = os.path.join(BASE_DIR, 
                             f'results/C2_sim_cf/{disease}/cv_results/cf_{cf_count}')
        json_path = os.path.join(cv_dir, 'cv_detailed.json')
        
        if not os.path.exists(json_path):
            print(f"⚠ Skipping CF={cf_count} — {json_path} not found")
            continue
        
        with open(json_path, 'r') as f:
            all_cv_results[cf_count] = json.load(f)
        
        print(f"✓ Loaded CV results for CF={cf_count}")
    
    if not all_cv_results:
        print("❌ No CV results found. Exiting.")
        return
    
    # ── Generate plots for each CF count ──────────────────────────────────
    for cf_count in sorted(all_cv_results.keys()):
        print(f"\n{'─'*70}")
        print(f"  Analyzing CF={cf_count}")
        print(f"{'─'*70}")
        
        cv_results = all_cv_results[cf_count]
        cf_output_dir = os.path.join(output_base, f'cf_{cf_count}')
        os.makedirs(cf_output_dir, exist_ok=True)
        
        # 1. ROC curves (3-panel)
        plot_roc_curves(cv_results, disease, cf_count, cf_output_dir)
        
        # Load aggregated predictions for distribution/rejection plots
        cv_dir = os.path.join(BASE_DIR, 
                             f'results/C2_sim_cf/{disease}/cv_results/cf_{cf_count}')
        
        # Aggregate fold predictions
        fold_files = sorted([f for f in os.listdir(cv_dir) 
                           if f.startswith('fold_') and f.endswith('_predictions.csv')])
        
         # ── Rewriting only the part that calls plot_rejection_analysis ──

        if fold_files:
            dfs = [pd.read_csv(os.path.join(cv_dir, f)) for f in fold_files]
            aggregated_df = pd.concat(dfs, ignore_index=True)
            
            # 2. Distributions (for each model)
            for model_type in args.models:
                plot_distributions(aggregated_df, disease, cf_count, 
                                cf_output_dir, model_type=model_type, config='M6')
            
            # 3. Rejection analysis (for each model) — pass fold info
            for model_type in args.models:
                plot_rejection_analysis(
                    disease=disease,
                    cf_count=cf_count,
                    output_dir=cf_output_dir,
                    model_type=model_type,
                    config='M6',
                    fold_files=fold_files,
                    cv_dir=cv_dir
                )
        else:
            print(f"⚠ No fold prediction files found for CF={cf_count}")
    # ── CF count comparison plot ──────────────────────────────────────────
    if len(all_cv_results) > 1:
        print(f"\n{'─'*70}")
        print("  Generating CF count comparison")
        print(f"{'─'*70}")
        
        comparison_dir = os.path.join(output_base, 'cf_comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        
        for model_type in args.models:
            plot_cf_count_comparison(all_cv_results, disease, comparison_dir,
                                   config='M6', model_type=model_type)
            plot_auc_vs_k(all_cv_results, disease, comparison_dir,
                         config='M6', model_type=model_type)
    
    print(f"\n{'='*70}")
    print(f"  ✓ All analyses complete!")
    print(f"  Output saved to: {output_base}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()