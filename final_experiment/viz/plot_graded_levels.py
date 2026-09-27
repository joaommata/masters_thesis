"""Schematic for §3.5.2.C: placement of the graded soft targets (no data)."""
import numpy as np
import matplotlib.pyplot as plt

C_POS, C_NEG, INK, INK_2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
TAU = 0.40
NEG_LV = [0.30, 0.175, 0.05]     # low, mid, high  (q95 -> q5 of negatives)
POS_LV = [0.50, 0.715, 0.93]     # low, mid, high  (q5 -> q95 of positives)

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(5.6, 2.3))

x = np.linspace(0, 1, 500)
neg = np.where(x < TAU, np.exp(-x / 0.09), np.nan)
pos = np.where(x > TAU, np.exp((x - 1) / 0.25), np.nan)
for d, c in ((neg / np.nanmax(neg), C_NEG), (pos / np.nanmax(pos), C_POS)):
    ax.fill_between(x, 0, d, color=c, alpha=0.15, lw=0)
    ax.plot(x, d, color=c, lw=1.6)

ax.plot([TAU, TAU], [-0.12, 1.02], color=INK, lw=1, ls=(0, (4, 3)))
ax.text(TAU, 1.06, r"$\tau$", ha="center", va="bottom", color=INK)

for lv, c in ((NEG_LV, C_NEG), (POS_LV, C_POS)):
    ax.plot(lv, [-0.12] * 3, color=c, lw=2, solid_capstyle="round")
    ax.scatter(lv, [-0.12] * 3, s=50, color=c, edgecolor="white", lw=1.5, zorder=3)
    for i, xv in enumerate(lv, 1):
        ax.text(xv, -0.2, rf"$\ell_{i}$", ha="center", va="top", color=INK)

ax.text(0.20, 0.55, "true negatives", color=C_NEG, fontsize=8, ha="center")
ax.text(0.70, 0.55, "true positives", color=C_POS, fontsize=8, ha="center")

ax.set_xlim(0, 1)
ax.set_ylim(-0.4, 1.2)
ax.set_xticks([0, 1])
ax.set_yticks([])
ax.set_xlabel(r"$C_0$ probability", color=INK)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_position(("data", 0))
ax.xaxis.set_label_coords(0.5, -0.1)

fig.savefig("graded_levels_schematic.pdf", bbox_inches="tight")
fig.savefig("graded_levels_schematic.png", dpi=300, bbox_inches="tight",
            facecolor="white")