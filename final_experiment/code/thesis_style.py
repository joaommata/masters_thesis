"""Thesis-wide figure style: one palette, one type scale, figures drawn at print size.

    import sys; sys.path.insert(0, "code")
    from thesis_style import *          # colours, sizes, helpers
    apply()                             # rcParams

Colour encodes the model FAMILY, and nothing else gets a hue:

    attributes  blue     FAMILY["attr"]
    CNNs        red      FAMILY["cnn"]
    trajectory  green    FAMILY["traj"]      (diffusion-CF levels arm)
    ensemble    purple   ENS                 (attributes + CNNs: blue + red)
    MSP         light grey, dashed           (the reference, never a series)

The hues follow the methods figure (attribute models blue, image models red, trajectory
models green). Red and green are the classic colour-blind confusion, so they are separated
by LIGHTNESS, not hue: the CNN red is dark (OKLCH L 0.50), the trajectory green light
(L 0.68). The ensemble purple (L 0.62, hue 300) is as dark and as blue as it can go while
staying apart from the attribute blue it is always drawn next to. Checked with the dataviz
palette validator's OKLab / Machado metrics, all four pairwise on a white surface:
colour-vision-deficiency separation dE >= 8.7 (attr-ensemble; every other pair >= 14),
normal-vision dE >= 16.2. "Below MSP" is still drawn hollow, not coloured.

Lightness encodes HOW MUCH COUNTERFACTUAL a model sees, within a family:

    TINT[f]       no counterfactual input   (B*, xi / xi_sal)
    RAMP[f][0-2]  light -> dark with more CF information: K = 1 / 5 / 10, or
                  M -> MCF, or CF-only -> both

so "darker = more counterfactual" reads the same way in every figure.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MM   = 1 / 25.4                    # figsize is in inches; write sizes as 60 * MM
FULL = 183 * MM                    # Nature double column
HALF = 89 * MM                     # Nature single column

FAMILY = {"attr": "#2863AB", "cnn": "#B02B27", "traj": "#54B05A"}
FAMILY_LABEL = {"attr": "attributes", "cnn": "CNN", "traj": "trajectory"}
# Same hues, OKLCH lightness 0.76 / 0.55 / 0.34 in every family, light -> dark. Wide steps because the lines
# they colour overlap; K_MARKER adds a shape so the steps never carry identity alone.
RAMP = {"attr": ["#7CB4FC", "#3772BB", "#02376F"],
        "cnn":  ["#F49287", "#C13C36", "#691211"],
        "traj": ["#81D584", "#3A9742", "#1D5522"]}
K_MARKER = ["o", "s", "D"]         # paired with the ramp steps, light -> dark
TINT = {"attr": "#D5E6FC", "cnn": "#FFE0DB", "traj": "#D9F4D9"}

ENS   = "#956ED2"                  # ensembles: attributes + CNNs, so blue + red = purple
GREY  = "#BFBFBF"                  # MSP reference line -- thin, light grey, dashed
INK   = "#1A1A1A"                  # text, axes, whiskers
MUTED = "#6B6B6B"                  # secondary text
HAIR  = "#E6E6E6"                  # gridlines, separators
SHADE = "#F2F2F2"                  # background band for no-counterfactual models
PRIOR = "#4D4D4D"                  # prior-art no-CF baselines: ConfidNet filled, TrustScore open

# Type scale (pt). Everything is drawn at final size, so these are the printed sizes.
FS = {"tiny": 5.5, "small": 6, "base": 7, "title": 7.5, "head": 8}
LW = {"hair": 0.5, "axis": 0.6, "ref": 0.8, "line": 1.5}


def _k_steps(ks, n):
    ks = sorted(ks)
    return ks, [round(i * (n - 1) / max(len(ks) - 1, 1)) for i in range(len(ks))]


def k_colors(family, ks):
    """K -> colour: the family ramp, light to dark as K grows."""
    ks, idx = _k_steps(ks, len(RAMP[family]))
    return {k: RAMP[family][i] for k, i in zip(ks, idx)}


def k_markers(ks):
    """K -> marker, paired with k_colors so the shade is never the only cue."""
    ks, idx = _k_steps(ks, len(K_MARKER))
    return {k: K_MARKER[i] for k, i in zip(ks, idx)}


def msp_line(ax, y, **kw):
    """The MSP reference: horizontal, light grey, dashed, behind the data."""
    return ax.axhline(y, color=GREY, ls="--", lw=LW["ref"], zorder=1, **kw)


def msp_handle(label="MSP"):
    return Line2D([], [], color=GREY, ls="--", lw=LW["ref"], label=label)


def line_handle(color, label, marker="o"):
    # No white ring on the legend marker: at legend size it cuts the line into dashes.
    return Line2D([], [], color=color, lw=LW["line"], marker=marker, ms=4, label=label)


def patch_handle(color, label):
    return Line2D([], [], color=color, lw=0, marker="s", ms=6, label=label)


def panel_letter(ax, letter, x=-0.16, y=1.06):
    """Nature-style bold lowercase panel label, outside the top-left corner."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS["head"] + 1,
            fontweight="bold", va="bottom", ha="left", color=INK)


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS["base"], "axes.titlesize": FS["title"], "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 4,
        "axes.labelsize": FS["base"], "xtick.labelsize": FS["small"],
        "ytick.labelsize": FS["small"], "legend.fontsize": FS["small"],
        "figure.titlesize": FS["head"], "figure.titleweight": "bold",
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": LW["axis"], "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": LW["axis"], "ytick.major.width": LW["axis"],
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 3,
        "lines.linewidth": LW["line"], "lines.markersize": 4,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "patch.linewidth": 0, "errorbar.capsize": 0,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2, "legend.borderaxespad": 0.2,
        "axes.grid": False, "grid.color": HAIR, "grid.linewidth": LW["hair"],
        "grid.linestyle": "-",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "svg.fonttype": "none",
    })


__all__ = ["MM", "FULL", "HALF", "FAMILY", "FAMILY_LABEL", "RAMP", "K_MARKER", "TINT", "ENS", "GREY", "PRIOR",
           "INK", "MUTED", "HAIR", "SHADE", "FS", "LW", "k_colors", "k_markers", "msp_line", "msp_handle",
           "line_handle", "patch_handle", "panel_letter", "apply"]
