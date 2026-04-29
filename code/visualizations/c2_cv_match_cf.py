import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/zhome/d0/a/221493/thesis/code')
from plot_config import PLOT_COLORS
import json

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = '/zhome/d0/a/221493/thesis/'
DISEASE  = 'effusion'
MODEL    = 'LR'
CONFIGS  = ['M6']

matched_json   = BASE_DIR + f"results/C2_sim_cf/{DISEASE}/cv_results/cf_1/cv_detailed.json"
unmatched_json = BASE_DIR + f"results/C2_sim_cf/{DISEASE}/cv_results_unmatched/cf_1/cv_detailed.json"

CURVE_COLORS = {
    'M6': {'matched': '#009e73', 'unmatched': '#D55E00'},
}

roc_matched = json.load(open(matched_json, 'r'))
roc_unmatched = json.load(open(unmatched_json, 'r'))

# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.size': 17,
    'axes.linewidth': 1.2,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
})

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

mean_fpr = np.linspace(0, 1, 200)

for config_name in CONFIGS:
    color_matched   = CURVE_COLORS.get(config_name, {}).get('matched', '#0072B2')
    color_unmatched = CURVE_COLORS.get(config_name, {}).get('unmatched', '#0072B2')

    # ═════════════ Matched ═════════════
    tprs = []
    aucs = []

    for fold in roc_matched[MODEL][config_name]:
        fpr = np.array(fold['fpr'])
        tpr = np.array(fold['tpr'])

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(fold['auc'])

    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color=color_matched,
        lw=2.5,
        ls='-',
        label=f"{config_name} — Matched\nAUC = {mean_auc:.3f} ± {std_auc:.3f}"
    )

    ax.fill_between(
        mean_fpr,
        mean_tpr - std_tpr,
        mean_tpr + std_tpr,
        color=color_matched,
        alpha=0.15
    )

    # ═════════════ Unmatched ═════════════
    tprs = []
    aucs = []

    for fold in roc_unmatched[MODEL][config_name]:
        fpr = np.array(fold['fpr'])
        tpr = np.array(fold['tpr'])

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(fold['auc'])

    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color=color_unmatched,
        lw=2.5,
        ls='--',
        label=f"{config_name} — Unmatched\nAUC = {mean_auc:.3f} ± {std_auc:.3f}"
    )

    ax.fill_between(
        mean_fpr,
        mean_tpr - std_tpr,
        mean_tpr + std_tpr,
        color=color_unmatched,
        alpha=0.15
    )

# ══════════════════════════════════════════════════════════════════════════════

ax.plot([0, 1], [0, 1], 'k:', lw=0.8, alpha=0.5)

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'Effect of CF Matching (LR M6, k=1)')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.spines['left'].set_position(('outward', 10))

ax.legend(
    loc='lower right',
    prop={'family': 'monospace', 'size': 11},
    framealpha=0.95,
    edgecolor='#cccccc',
    handlelength=2,
    labelspacing=0.8
)

ax.grid(alpha=0.30, linewidth=0.5)

plt.tight_layout()
config_tag = '_vs_'.join(c.lower() for c in CONFIGS)

plt.savefig(
    os.path.join(BASE_DIR,
    f'results/C2_sim_cf/{DISEASE}/roc_matched_vs_unmatched_{MODEL.lower()}_{config_tag}.png'),
    dpi=150
)

plt.show()