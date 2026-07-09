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
    'B1': '#BFBFBF',   # light gray — always shown as a thin reference dashed line
    'B2': '#CC79A7',   # mauve/pink
    'B3': '#F0E442',   # yellow
    'B4': '#E69F00',   # amber
    'B5': '#D55E00',   # vermillion
    'M1': '#56B4E9',   # sky blue
    'M2': '#009E73',   # teal-green
    'M3': '#0072B2',   # deep blue
    'M4': '#9370DB',   # medium purple
    'M5': '#2D2D2D',   # near-black
    'M6': '#CC0000',   # red
    # CF-enriched configs (MCF*) — include raw CF features alongside ΔA
    'MCF1': '#FF6B6B',  # coral
    'MCF2': '#FF9500',  # orange
    'MCF3': '#2ECC71',  # emerald
    'MCF4': '#3498DB',  # cornflower blue
    'MCF5': '#8E44AD',  # violet
}

PLOT_LS = {k: '--' if k.startswith('B') else '-' for k in PLOT_COLORS}
PLOT_LW = {k: 1.5  if k.startswith('B') else 2.2  for k in PLOT_COLORS}
PLOT_LW['B1'] = 1.0   # thinner than the other baselines

