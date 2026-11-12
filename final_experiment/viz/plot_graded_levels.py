"""Schematic for §3.5.2.C: placement of the graded soft targets (no data)."""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from thesis_style import *

apply()
TAU = 0.40
NEG_LV = [0.30, 0.175, 0.05]     # low, mid, high  (q95 -> q5 of negatives)
POS_LV = [0.50, 0.715, 0.93]     # low, mid, high  (q5 -> q95 of positives)
# negatives green, positives red; level markers light -> dark = low -> high within each class
C_NEG, C_POS = FAMILY["traj"], FAMILY["cnn"]
LV_NEG, LV_POS = RAMP["traj"], RAMP["cnn"]

fig, ax = plt.subplots(figsize=(65 * MM, 42 * MM))

x = np.linspace(0, 1, 600)
# smooth class-conditional densities (Beta shapes), cut at tau: correctly classified cases only
neg = np.where(x < TAU, x**1.2 * (1 - x)**6.5, np.nan)
pos = np.where(x > TAU, x**3.5 * (1 - x)**1.6, np.nan)
for d, c in ((neg / np.nanmax(neg), C_NEG), (pos / np.nanmax(pos), C_POS)):
    ax.fill_between(x, 0, d, color=c, alpha=0.15, lw=0)
    ax.plot(x, d, color=c, lw=LW["line"])

ax.plot([TAU, TAU], [-0.2, 1.05], color=INK, lw=LW["ref"], ls=(0, (3, 2)))
ax.text(TAU, 1.09, r"$\tau$", ha="center", va="bottom", fontsize=FS["base"])

y0 = -0.2
for lv, cols in ((NEG_LV, LV_NEG), (POS_LV, LV_POS)):
    ax.plot([lv[0], lv[2]], [y0] * 2, color=MUTED, lw=LW["ref"], zorder=1)
    for name, xv, c in zip(("low", "mid", "high"), lv, cols):
        ax.scatter(xv, y0, s=22, color=c, edgecolor=INK, lw=LW["hair"], zorder=3)
        ax.text(xv, y0 - 0.1, name, ha="center", va="top", fontsize=FS["small"])

ax.text(0.19, 0.38, "true negatives", color=C_NEG, fontsize=FS["small"], ha="center")
ax.text(0.72, 0.38, "true positives", color=C_POS, fontsize=FS["small"], ha="center")

ax.set_xlim(0, 1)
ax.set_ylim(-0.42, 1.25)
ax.set_xticks([0, 1])
ax.set_yticks([])
ax.set_xlabel(r"$C_0$ probability")
ax.xaxis.set_label_coords(0.5, -0.07)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_position(("data", 0))

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "graded_levels_schematic.pdf"))
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "graded_levels_schematic.png"), dpi=400)
