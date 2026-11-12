"""Slide versions of figure 2d (Aim 2: baselines vs ad-MSP), one message per figure.

  2e  average over the five tasks: every method's AUROC gain over ad-MSP
  2f  our ensemble only, one row per task

Same numbers as 2d (BASE_GAINS in C2_evaluation.ipynb), read straight from the bootstrap
exports so the figures can be redrawn in seconds without running the notebook:

    python final_experiment/code/c2_aim2_slides.py         (from the repo root or final_experiment/)

The notebook calls slide_average / slide_ours with its own BASE_GAINS instead.
Style follows figure 2a (msp_forest): square plot area, filled = Holm p < alpha vs ad-MSP,
open = n.s., dashed zero line = ad-MSP, 95% bootstrap whiskers, gain printed above the point.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *  # noqa: E402,F403

RESULTS = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
C2 = os.path.join(RESULTS, "C2_final_results")
DISEASES = ["effusion", "atelectasis", "cardiomegaly", "consolidation", "edema"]
BASELINES = {"ConfidNet": "confidnet_vs_msp", "TrustScore": "trustscore_vs_msp", "SCCE": "scce_vs_msp"}
PANEL = 50 * MM            # plot-area side, as KEY_SIDE / the 2a panels


def _signed(x, dec=3):
    return f"{x:+.{dec}f}".replace("-", "−")


def load_gains(c2=C2):
    """AUROC gain over ad-MSP per (disease, method), as BASE_GAINS: the baselines from their
    comparison files, the ensemble from the bootstrap export's 'best overall - MSP' rows with
    the point estimate from the baselines summary (the notebook asserts the two agree)."""
    rows = []
    for m, f in BASELINES.items():
        t = pd.read_csv(os.path.join(c2, f"bootstrap_{f}.csv"))
        t = t[t.metric == "AUROC"]
        rows += [dict(disease=r.disease, method=m, gain=r.gain, lo=r.lo, hi=r.hi, p_adj=r.p_adj)
                 for r in t.itertuples()]
    s = pd.read_csv(os.path.join(c2, "bootstrap_baselines_summary.csv")).set_index(["disease", "method"])
    d = pd.read_csv(os.path.join(c2, "bootstrap_deltas.csv"))
    d = d[d.comparison == "best overall - MSP"].set_index("disease")
    for dis in DISEASES + ["AVERAGE"]:
        gain = s.loc[(dis, "Ensemble"), "AUROC"] - s.loc[(dis, "MSP"), "AUROC"]
        r = d.loc[dis]
        rows.append(dict(disease=dis, method="Ensemble", gain=gain, lo=r.lo, hi=r.hi, p_adj=r.p_adj))
    out = pd.DataFrame(rows)
    out["metric"] = "AUROC"
    return out


def _point(ax, y, r, color, alpha, dec, bold=False):
    """2a's point: whisker, dot filled when significant, gain printed above."""
    ax.plot([r.lo, r.hi], [y, y], color=INK, lw=LW["line"], solid_capstyle="butt", zorder=2)
    ax.scatter([r.gain], [y], s=55, zorder=3, c=[color if r.p_adj < alpha else "white"],
               edgecolors=[color], linewidths=1.3)
    # Beside the point, right of its whisker: keeps each row to one line of height.
    x0, x1 = ax.get_xlim()
    ax.text(max(r.hi, r.gain) + 0.045 * (x1 - x0), y, _signed(r.gain, dec), ha="left",
            va="center", fontsize=FS["base"], color=INK,
            fontweight="bold" if bold else "normal", zorder=4)


def _finish(fig, ax, color, title, subtitle, xlabel):
    ax.axvline(0.0, color=GREY, ls="--", lw=LW["ref"], zorder=1)
    ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_box_aspect(1)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=10)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FS["small"], color=MUTED)
    legend = [Line2D([], [], ls="none", marker="o", ms=5, color=color, label="Holm p < 0.05"),
              Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=color, mew=1.3,
                     label="not significant"),
              Line2D([], [], color=INK, lw=LW["ref"], label="95% bootstrap interval"),
              msp_handle("ad-MSP")]
    fig.legend(handles=legend, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    for _ in range(2):                 # second pass measures the squared axes
        fig.tight_layout(rect=(0, 0.12, 1, 1))


def _xticks(ax, lo, hi, step):
    ax.set_xlim(lo, hi)
    t = np.arange(np.ceil(lo / step) * step, hi + 1e-12, step)
    ax.set_xticks(t)
    dec = max(0, -int(np.floor(np.log10(step))))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: "0" if abs(x) < 1e-12 else _signed(x, dec)))


def slide_average(gains, alpha=0.05, figsize=(58 * MM, 62 * MM)):
    """Figure 2e: the five-task average, every method vs ad-MSP. SC-CE is an order of magnitude
    off the axis (2d's strip), so it is an arrow and its number instead of a point."""
    g = gains[(gains.metric == "AUROC") & (gains.disease == "AVERAGE")].set_index("method")
    rows = ["SCCE", "TrustScore", "ConfidNet", "Ensemble"]
    labels = ["SC-CE", "TrustScore", "ConfidNet", "Ours"]
    fig, ax = plt.subplots(figsize=figsize)
    main = g.loc[rows[1:]]
    _xticks(ax, min(main.lo.min(), 0) - 0.004, max(main.hi.max(), 0) + 0.009, 0.01)
    for y, m in enumerate(rows[1:], start=1):
        c = ENS if m == "Ensemble" else PRIOR
        _point(ax, y, g.loc[m], c, alpha, 3, bold=(m == "Ensemble"))
    x0, x1 = ax.get_xlim()
    ax.annotate("", xy=(x0, 0), xytext=(x0 + 0.12 * (x1 - x0), 0),
                arrowprops=dict(arrowstyle="-|>", color=PRIOR, lw=LW["ref"], mutation_scale=6))
    ax.text(x0 + 0.14 * (x1 - x0), 0, f"{_signed(g.loc['SCCE'].gain, 2)}  off-scale", ha="left",
            va="center", fontsize=FS["small"], color=MUTED,
            bbox=dict(fc="white", ec="none", pad=0.8), zorder=4)
    ax.set_yticks(range(len(rows)), labels)
    ours = ax.get_yticklabels()[-1]
    ours.set_fontweight("bold"); ours.set_color(ENS)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    _finish(fig, ax, PRIOR, "Only ours beats ad-MSP", "average over 5 findings",
            "AUROC gain over ad-MSP")
    return fig


def slide_ours(gains, alpha=0.05, figsize=(58 * MM, 62 * MM)):
    """Figure 2f: our ensemble's AUROC gain over ad-MSP per task, largest first."""
    g = gains[(gains.metric == "AUROC") & (gains.method == "Ensemble")
              & (gains.disease != "AVERAGE")].set_index("disease")
    order = g.sort_values("gain", ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=figsize)
    _xticks(ax, min(g.lo.min(), 0) - 0.0005, max(g.hi.max(), 0) + 0.0035, 0.004)
    for y, d in enumerate(order):
        _point(ax, y, g.loc[d], ENS, alpha, 4)
    ax.set_yticks(range(len(order)), [d.capitalize() for d in order])
    ax.set_ylim(len(order) - 0.45, -0.55)
    _finish(fig, ax, ENS, "Ours, per finding", "ensemble vs ad-MSP", "AUROC gain over ad-MSP")
    return fig


if __name__ == "__main__":
    matplotlib.use("Agg")
    apply()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # final_experiment/
    out = os.path.join(here, "figures", "c2_key")
    alpha = json.load(open(os.path.join(C2, "bootstrap_meta.json")))["alpha"]
    gains = load_gains()
    print(gains.round(4).to_string(index=False))
    for fig, name in ((slide_average(gains, alpha), "aim2e_average_slide"),
                      (slide_ours(gains, alpha), "aim2f_ours_slide")):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"knn_{name}.{ext}"))
    print("written to", out)
