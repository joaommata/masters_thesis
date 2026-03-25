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
BASE_DIR   = '/zhome/d0/a/221493/thesis/'

roc_data_1 = json.load(open(BASE_DIR + f"results/C2_sim_cf/effusion/roc_data_1.json"))
roc_data_3 = json.load(open(BASE_DIR + f"results/C2_sim_cf/effusion/roc_data_3.json"))
roc_data_5 = json.load(open(BASE_DIR + f"results/C2_sim_cf/effusion/roc_data_5.json"))

MODELS_TO_PLOT = ['LR', 'RF', 'MLP']
# Plot style shared across all figures
COLORS = {**PLOT_COLORS, 'M6': '#CC0000'}
LINESTYLES = {k: '--' if k.startswith('B') else '-' for k in COLORS}
LINEWIDTHS = {k:  2   if k.startswith('B') else 3   for k in COLORS}

colors_cf = {
    1: "#009E73",  # Red for 1 CF
    3: '#0072B2',  # Blue for 3 CFs
    5: '#6C01D7'}  # Green for 5 CFs
    
plt.rcParams.update({
    'font.size': 17,
    'axes.linewidth': 1.2,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
})
plt.rcParams.update({
        'font.size': 13, 'axes.linewidth': 1.2,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, model_name in zip(axes, MODELS_TO_PLOT):
    for roc_data, cf in zip([roc_data_1, roc_data_3, roc_data_5], [1, 3, 5]):
        d = roc_data[model_name]['M6']
        ax.plot(d['fpr'], d['tpr'],
                # Colours cant be M6 becasye they need to differ by CF count, so we use new colors that are similar but differ by shade
                color=colors_cf[cf],
                lw=LINEWIDTHS['M6'],
                ls=LINESTYLES['M6'],
                label=f"M6 ({cf} CF)\nAUC = {d['auc']:.3f}")   
    ax.plot([0, 1], [0, 1], 'k:', lw=0.8, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    name = "Linear Regression" if model_name == 'LR' else ("Random Forest" if model_name == 'RF' else "Multilayer MLP")
    ax.set_title(f'C2 | {name} | M6')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines['left'].set_position(('outward', 10))
    ax.legend(loc='lower right',
              bbox_to_anchor=(1.02, 0.0),
              prop={'family': 'monospace', 'size': 12},
              framealpha=0.95,
              edgecolor='#cccccc',
              handlelength=2,
              labelspacing=0.8)
    ax.grid(alpha=0.30, linewidth=0.5)
plt.tight_layout()
plt.show()
