"""KDE of C2 scores split by true C0 correct / incorrect, all diseases in a grid.

The first column is the untrained baseline, adjusted MSP (c2_msp.msp_threshold), scored
off C0's probability on the same rows the attribute model scores.

    python c2_plot_score_dist.py knn 5
    python c2_plot_score_dist.py diffusion 3
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score

from c2_msp import msp_threshold
from thesis_style import FAMILY, FS, FULL, MM, MUTED, apply

RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")

arm, k = sys.argv[1], sys.argv[2]
source = {"knn": "knn_correct_cf", "diffusion": "diffusion_hml_sub20k"}[arm]
diseases = ["atelectasis", "cardiomegaly", "consolidation", "edema", "effusion"]
# title -> (title colour, prediction files, score column, x label)
models = {
    # Not a model file: computed in the loop from C0's probability and Youden threshold.
    "Adjusted MSP": (MUTED, "attributes/fold_*_predictions.csv", "adj_msp",
                     "P(predicted class)"),
    "Attribute (I+C+D)": (FAMILY["attr"], "attributes/fold_*_predictions.csv",
                           "MLP_I+C+D_prob", "P(correct)"),
    "Image (dual + saliency)": (FAMILY["cnn"], "images/dual_sal/fold_*_predictions.csv",
                              "dual_sal_prob", "P(correct)"),
}
if arm == "diffusion":
    models["Trajectory (RNN)"] = (FAMILY["traj"], "trajectory/fold_*_predictions.csv",
                                  "RNN_TRAJ_prob", "P(correct)")


def load(pattern):
    """Pooled out-of-fold predictions, with a `fold` column for per-fold AUROC."""
    files = sorted(glob.glob(pattern))
    return pd.concat(pd.read_csv(f).assign(fold=i) if "fold_" in f else pd.read_csv(f)
                     for i, f in enumerate(files))


apply()
x = np.linspace(0, 1, 500)
fig, axes = plt.subplots(len(diseases), len(models),
                         figsize=(FULL, 26 * MM * len(diseases)), sharex=True, squeeze=False)
for i, disease in enumerate(diseases):
    root = os.path.join(RESULTS_DIR, "C2_final_results", disease, source, f"cf_{k}")
    for j, (title, (title_color, pattern, col, _)) in enumerate(models.items()):
        ax = axes[i, j]
        df = load(os.path.join(root, pattern))
        if col == "adj_msp":
            t = float(open(os.path.join(RESULTS_DIR, "C0_final", "multilabel_ignore",
                                        disease, "threshold.txt")).read())
            df[col] = msp_threshold(df[f"{disease}_prob"], t)
        for c, label, color in [(1, "correct", "green"), (0, "incorrect", "red")]:
            y = gaussian_kde(df.loc[df.correct == c, col])(x)
            ax.fill_between(x, y, color=color, alpha=0.2, lw=0)
            ax.plot(x, y, color=color, lw=1, label=label)
        auc = np.mean([roc_auc_score(g.correct, g[col]) for _, g in df.groupby("fold")])
        ax.text(0.5, 0.97, f"AUROC {auc:.3f}", transform=ax.transAxes, ha="center",
                va="top", fontsize=FS["tiny"], color=MUTED)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        if i == 0:
            ax.set_title(title, color=title_color)
        if j == 0:
            ax.set_ylabel(disease.capitalize(), fontweight="bold")
for ax, (_, _, _, xlabel) in zip(axes[-1], models.values()):
    ax.set_xlabel(f"C2 score, {xlabel}")
fig.suptitle(f"{'KNN' if arm == 'knn' else 'Diffusion'} counterfactuals, K = {k}",
             x=0.01, ha="left")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2,
           title="C0 prediction", title_fontsize=FS["small"])
fig.tight_layout()
# Dashed rule between the baseline column and the trained C2 models, after tight_layout
# so it sits midway in the final gap.
x_sep = (axes[0, 0].get_position().x1 + axes[0, 1].get_position().x0) / 2
fig.add_artist(plt.Line2D([x_sep, x_sep], [axes[-1, 0].get_position().y0,
                                           axes[0, 0].get_position().y1 + 0.02],
                          transform=fig.transFigure, color=MUTED, ls="--", lw=0.6))
out = os.path.join(RESULTS_DIR, "C2_final_results", f"score_dist_{arm}_k{k}.png")
fig.savefig(out, dpi=600)
print(out)
