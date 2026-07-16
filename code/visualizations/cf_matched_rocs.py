import os
import sys
import json
import matplotlib.pyplot as plt

sys.path.append('/zhome/d0/a/221493/thesis/code')
from plot_config import PLOT_COLORS

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
DISEASE  = 'effusion'
MODEL    = 'LR'
CONFIGS  = ['M6']  # compare two configs on one plot

# Load matched and unmatched ROC data (both at k=1)
roc_matched   = json.load(open(RESULTS_DIR + f"/C2_sim_cf/{DISEASE}/roc_data_1.json"))
roc_unmatched = json.load(open(RESULTS_DIR + f"/C2_sim_cf/{DISEASE}/roc_data_1_unmatched.json"))

# Explicit colors per config and matching strategy.
CURVE_COLORS = {
    'M6': {'matched': '#009e73', 'unmatched': '#D55E00'},  # green / vermillion
}

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

for config_name in CONFIGS:
    color_matched = CURVE_COLORS.get(config_name, {}).get('matched', '#0072B2')
    color_unmatched = CURVE_COLORS.get(config_name, {}).get('unmatched', '#0072B2')

    # Matched curve
    d = roc_matched[MODEL][config_name]
    ax.plot(d['fpr'], d['tpr'],
            color=color_matched,
        lw=2.5,
        ls='-',
        label=f"{config_name} — Matched\nAUC = {d['auc']:.3f}")

    # Unmatched curve
    d = roc_unmatched[MODEL][config_name]
    ax.plot(d['fpr'], d['tpr'],
            color=color_unmatched,
        lw=2.5,
        ls='--',
        label=f"{config_name} — Unmatched\nAUC = {d['auc']:.3f}")

ax.plot([0, 1], [0, 1], 'k:', lw=0.8, alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'CF matching strategy ({MODEL})')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.spines['left'].set_position(('outward', 10))
ax.legend(loc='lower right',
          prop={'family': 'monospace', 'size': 11},
          framealpha=0.95,
          edgecolor='#cccccc',
          handlelength=2,
          labelspacing=0.8)
ax.grid(alpha=0.30, linewidth=0.5)

plt.tight_layout()
config_tag = '_vs_'.join(c.lower() for c in CONFIGS)
plt.savefig(os.path.join(RESULTS_DIR, f'C2_sim_cf/{DISEASE}/roc_matched_vs_unmatched_{MODEL.lower()}_{config_tag}.png'), dpi=150)
plt.show()