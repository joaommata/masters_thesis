"""
plot_config.py
==============
Shared palette for all C2 scripts.
Import with:
    from plot_config import PLOT_COLORS, PLOT_LS, PLOT_LW

Mapping:
    B* = baselines  → muted colors, dashed, thin
    M* = ΔA models  → vivid colors, solid, thick
"""

PLOT_COLORS = {
    'B1': '#999999',   # gray
    'B2': '#CC79A7',   # mauve/pink
    'B3': '#F0E442',   # yellow
    'B4': '#E69F00',   # amber
    'B5': '#D55E00',   # vermillion
    'M1': '#56B4E9',   # sky blue
    'M2': '#0072B2',   # deep blue
    'M3': '#009E73',   # teal-green
    'M4': '#9370DB',   # medium purple
    'M5': '#2D2D2D',   # black (full model)
    'M6': '#CC0000',   # red
}

PLOT_LS = {k: '--' if k.startswith('B') else '-' for k in PLOT_COLORS}
PLOT_LW = {k: 1.5  if k.startswith('B') else 2.2  for k in PLOT_COLORS}

