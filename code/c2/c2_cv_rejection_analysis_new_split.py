"""
c2_cv_rejection_analysis_refactored.py
=======================================
Generates risk-coverage curves and clinical deferral tables for C2.

C2 is treated as a binary ERROR DETECTOR:
    - Positive class = C0 made an error  (correct == 0)
    - Negative class = C0 was correct    (correct == 1)
    - C2 score = predicted probability of being correct
    - Rejection = low C2 score (c2_score < threshold)

Sensitivity = of all real errors, how many did C2 catch (reject)?
Specificity = of all correct predictions, how many did C2 pass (accept)?

Outputs:
    - risk_curve_cf{k}.png
    - roc_curve_cf{k}.png             (sanity check: AUC should match CV AUROC)
    - sens_spec_coverage_cf{k}.png
    - clinical_perfect_cf{k}.csv
    - clinical_kappa_cf{k}.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve
from plot_config import PLOT_COLORS

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DISEASE  = 'effusion'
CF_COUNT = 16
BASE_DIR = '/work3/s251710/thesis_results'
MODEL    = 'LR'
CONFIG   = 'M3'

plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 16,
    'axes.titlesize': 17,
    'axes.labelsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
})

COLOR = PLOT_COLORS[CONFIG]

# ══════════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_risk_coverage(scores, correct):
    """
    Sorts samples by descending C2 score (most confident first).
    At each step k, computes error rate on accepted set of size k.
    
    Returns: coverage, risk, AURC
    """
    n = len(scores)
    order = np.argsort(scores)[::-1]
    correct_sorted = correct[order]

    cumulative_errors = np.cumsum(1 - correct_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = cumulative_errors / k

    return coverage, risk, auc(coverage, risk)


def compute_sensitivity_specificity_vs_threshold(c2_scores, correct, n_thresholds=100):
    """
    Treats C2 as a binary error detector.
    
    At threshold t: samples with c2_score < t are REJECTED (flagged as errors).
    
    Sensitivity = of all real errors, how many did C2 reject?
    Specificity = of all correct predictions, how many did C2 accept?
    Coverage    = fraction of samples accepted (not rejected)
    
    Sanity check: ROC AUC of this curve should match CV pipeline AUROC.
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    actual_error = (correct == 0)

    sensitivities, specificities, coverages = [], [], []

    for t in thresholds:
        predicted_error = (c2_scores < t)  # C2 rejects this sample

        # Sensitivity: of all real errors, how many did C2 catch?
        sens = (predicted_error & actual_error).sum() / actual_error.sum()

        # Specificity: of all correct predictions, how many did C2 pass through?
        spec = (~predicted_error & ~actual_error).sum() / (~actual_error).sum()

        # Coverage: fraction of samples accepted
        coverage = (~predicted_error).sum() / len(correct)

        sensitivities.append(sens)
        specificities.append(spec)
        coverages.append(coverage)

    return thresholds, np.array(sensitivities), np.array(specificities), np.array(coverages)


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL TABLE
# ══════════════════════════════════════════════════════════════════════════════

def make_clinical_table(c2_scores, correct, thresholds, output_path, mode="perfect", kappa=None):
    """
    At each threshold, reports what happens to accepted and rejected samples.

    mode='perfect' : rejected cases assumed corrected by specialist (0 errors)
    mode='kappa'   : rejected cases assumed to have same error rate as baseline
    """
    rows = []
    n_total = len(correct)

    if mode == "kappa" and kappa is None:
        kappa = 1 - correct.mean()

    for t in thresholds:
        accepted = c2_scores >= t
        rejected = ~accepted

        n_acc = accepted.sum()
        acc_errors = (correct[accepted] == 0).sum()

        if mode == "perfect":
            rej_errors = 0
        elif mode == "kappa":
            rej_errors = rejected.sum() * kappa

        total_errors = acc_errors + rej_errors

        rows.append({
            "Threshold":            round(t, 4),
            "Coverage (%)":         round(n_acc / n_total * 100, 2),
            "Accepted":             int(n_acc),
            "Rejected":             int(rejected.sum()),
            "Errors on accepted":   int(acc_errors),
            "Errors on rejected":   float(rej_errors),
            "Total errors":         float(total_errors),
            "System error rate (%)": round(total_errors / n_total * 100, 2),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"\n[{mode.upper()}] Clinical table saved to: {output_path}")
    print(df_out.to_string(index=False))
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── Load predictions ──────────────────────────────────────────────────────
    cv_dir = os.path.join(BASE_DIR, f'C2_custom/{DISEASE}/cv_results/cf_{CF_COUNT}')
    files  = sorted([f for f in os.listdir(cv_dir)
                     if f.startswith("fold_") and f.endswith("_predictions.csv")])

    df      = pd.concat([pd.read_csv(os.path.join(cv_dir, f)) for f in files], ignore_index=True)
    correct = df["correct"].values
    scores  = df[f"{MODEL}_{CONFIG}_prob"].values
    base_error   = 1 - correct.mean()

    print(f"Loaded {len(df)} samples | baseline error rate κ = {base_error:.4f}")

    # ── 1. Risk-Coverage 
    fig, ax = plt.subplots(figsize=(6, 6))

    cov, risk, aurc       = compute_risk_coverage(scores, correct)

    ax.plot(cov, risk,     color=COLOR, lw=3, label=f"AURC={aurc:.4f}")

    ax.set_title(f"Risk-Coverage ({MODEL} {CONFIG}, k={CF_COUNT})")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, f"risk_curve_cf{CF_COUNT}.png"), dpi=300)
    plt.close()
    print(f"AURC={aurc:.4f}")


    # ── 2. ROC curve (sanity check) ───────────────────────────────────────────
    # C2 as error detector: positive = error (correct==0)
    # AUC here should match CV pipeline AUROC
    roc_auc = roc_auc_score(1 - correct, 1 - scores)  # flip: high score = correct
    fpr, tpr, _ = roc_curve(1 - correct, 1 - scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=COLOR, lw=3, label=f"AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label="Random")
    ax.set_title(f"ROC – C2 as Error Detector ({MODEL} {CONFIG}, k={CF_COUNT})")
    ax.set_xlabel("False Positive Rate (correct flagged as error)")
    ax.set_ylabel("True Positive Rate (errors caught)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, f"roc_curve_cf{CF_COUNT}.png"), dpi=300)
    plt.close()
    print(f"ROC AUC (error detector) = {roc_auc:.4f}  ← should match CV AUROC")

    # ── 3. Sensitivity & Specificity vs Coverage ──────────────────────────────
    thresholds, sens, spec, cov_t = compute_sensitivity_specificity_vs_threshold(scores, correct)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(cov_t, sens, color=COLOR,      lw=3, label="Sensitivity (errors caught)")
    ax.plot(cov_t, spec, color=COLOR,      lw=3, ls='--', label="Specificity (correct passed)")
    ax.set_title(f"Sensitivity & Specificity vs Coverage ({MODEL} {CONFIG}, k={CF_COUNT})")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Rate")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cv_dir, f"sens_spec_coverage_cf{CF_COUNT}.png"), dpi=300)
    plt.close()

    # ── 4. Clinical tables ────────────────────────────────────────────────────
    table_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    make_clinical_table(
        scores, correct, table_thresholds,
        output_path=os.path.join(cv_dir, f"clinical_perfect_cf{CF_COUNT}.csv"),
        mode="perfect"
    )

if __name__ == "__main__":
    main()