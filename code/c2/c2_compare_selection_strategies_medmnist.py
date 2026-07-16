"""
c2_compare_selection_strategies_medmnist.py
============================================
MedMNIST counterpart of c2_compare_selection_strategies.py.

Compares CF *selection strategies* (how the counterfactual neighbour is chosen)
at a fixed CF count, holding model/config fixed.

Strategies compared (subdirs under results/C2_medmnist_corrected/{disease}/):
  correct_cf           - nearest opposite-prediction neighbour, attr space (L1), restricted to correct CFs
  correct_cf_cosine    - same correctness restriction, neighbour chosen by cosine similarity in attr space
  correct_cf_emb       - same correctness restriction, neighbour chosen in embedding space
  correct_train_only   - samples are chosen of the same correctness in train (NOT TEST)
  extended_correct_cf  - correct_cf but with the extended radiomics feature set
  gt_routing           - oracle: chooses the ground truth opposite of the prediction
  unmatched            - naive nearest opposite-prediction neighbour, no correctness restriction

Only configs that actually depend on the CF (B5, M1-M6, MCF1-5) are compared — B1-B4 use no
CF information and are identical across all strategies.

Usage:
    python c2_compare_selection_strategies_medmnist.py --disease effusion --cf_count 1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('/zhome/d0/a/221493/thesis/code')
from plot_config import PLOT_COLORS, PLOT_LS, PLOT_LW

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"

STRATEGIES = {
    'correct_cf':          ('cv_results_correct_cf',          'l1',     'Correct CF (L1)'),
    'correct_cf_cosine':   ('cv_results_correct_cf',          'cosine', 'Correct CF (Cosine)'),
    'unmatched':           ('cv_results_unmatched',           'l1',     'Unmatched (L1)'),
    'unmatched_cf_cosine':   ('cv_results_unmatched',           'cosine', 'Unmatched (Cosine)'),
    'correct_cf_emb':      ('cv_results_correct_cf_emb',      'l1',     'Correct CF (Emb. Space)'),
    #'correct_train_only':  ('cv_results',                     'l1',     'Matched (L1, Train Only)'),
    'gt_routing':          ('cv_results_gt_routing',          'l1',     'GT Routing (L1)'),
    'gt_routing_cosine':   ('cv_results_gt_routing',          'cosine', 'GT Routing (Cosine)'),
    'extended_correct_cf': ('cv_results_extended_correct_cf', 'l1',     'Correct CF (Extra Texture)'),
}

CONFIG_ORDER = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'MCF1', 'MCF2', 'MCF3', 'MCF4', 'MCF5']

# Configs that use no CF information — identical across strategies, used as a reference baseline.
BASELINE_CONFIGS = ['B1', 'B2', 'B3', 'B4', 'B5']

STRATEGY_COLORS = {
    'correct_cf':          '#0072B2',   # blue
    'correct_cf_cosine':   '#CC79A7',   # pink
    'unmatched_cf_cosine': '#F0E442',   # yellow
    'correct_cf_emb':      '#56B4E9',   # sky blue
    'extended_correct_cf': '#009E73',   # teal
    'unmatched':           '#D55E00',   # vermillion (naive baseline)
    #'correct_train_only':  "#E600C7A4",   # orange
    'gt_routing':          '#E69F00',   # orange (oracle)
    'gt_routing_cosine':   '#000000',   # black (oracle, cosine)
}

plt.rcParams.update({
    'font.size': 17,
    'axes.linewidth': 1.2,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
})


# ══════════════════════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_strategy_data(disease, cf_count, models):
    """Load cv_summary.csv for each strategy, skipping missing ones. Returns long DataFrame."""
    input_root = os.path.join(RESULTS_DIR, '', 'C2_medmnist_corrected', disease)

    rows = []
    for strategy, (subdir, distance, _) in STRATEGIES.items():
        cf_dir = os.path.join(input_root, subdir, f'cf_{cf_count}')
        if distance != 'l1':
            cf_dir = os.path.join(cf_dir, f'distance_{distance}')
        csv_path = os.path.join(cf_dir, 'cv_summary.csv')
        if not os.path.exists(csv_path):
            print(f"⚠ Skipping {strategy} — {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        df = df[df['model'].isin(models) & df['config'].isin(CONFIG_ORDER + BASELINE_CONFIGS)].copy()
        df.insert(0, 'strategy', strategy)
        rows.append(df[['strategy', 'model', 'config', 'mean_auc', 'std_auc']])
        print(f"✓ Loaded {strategy} — {len(df)} rows")

    if not rows:
        raise SystemExit(f"No strategy data found for disease={disease} cf_count={cf_count}")

    return pd.concat(rows, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: one panel per model, x=config, line=strategy
# ══════════════════════════════════════════════════════════════════════════════

def plot_strategy_comparison(long_df, model_type, output_dir, cf_count, disease):
    sub = long_df[long_df['model'] == model_type]
    if sub.empty:
        print(f"⚠ No data for model={model_type}, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for strategy, (_, _, label) in STRATEGIES.items():
        strat_df = sub[sub['strategy'] == strategy]
        if strat_df.empty:
            continue

        # Keep only configs present for this strategy, in canonical order
        present_configs = [c for c in CONFIG_ORDER if c in strat_df['config'].values]
        strat_df = strat_df.set_index('config').loc[present_configs].reset_index()

        x = [CONFIG_ORDER.index(c) for c in present_configs]
        y = strat_df['mean_auc'].values
        yerr = strat_df['std_auc'].values
        color = STRATEGY_COLORS[strategy]

        ax.plot(x, y, color=color, lw=2.4, marker='o', markersize=7, label=label)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.15)

    # Best no-CF baseline (B1-B4), identical across strategies — shown for reference.
    baseline_df = sub[sub['config'].isin(BASELINE_CONFIGS)]
    if not baseline_df.empty:
        best_row = baseline_df.loc[baseline_df['mean_auc'].idxmax()]
        ax.axhline(best_row['mean_auc'], color='black', linestyle='--', lw=1.6,
                   label=f"Best baseline ({best_row['config']})")

    # B1 (prob-only) reference line, always shown alongside the best baseline.
    b1_df = sub[sub['config'] == 'B1']
    if not b1_df.empty:
        b1_auc = b1_df['mean_auc'].iloc[0]
        ax.axhline(b1_auc, color=PLOT_COLORS['B1'], linestyle=PLOT_LS['B1'], lw=PLOT_LW['B1'],
                   label='B1 (Prob only)')

    ax.set_xticks(range(len(CONFIG_ORDER)))
    ax.set_xticklabels(CONFIG_ORDER, rotation=45, ha='right')
    ax.set_xlabel('Config')
    ax.set_ylabel('Mean AUROC (5-fold CV)')
    ax.set_ylim(0.85, 0.95)
    ax.set_title(f'CF Selection Strategy Comparison — MedMNIST {disease.capitalize()} {model_type} (CF={cf_count})')
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'strategy_comparison_{model_type}_cf{cf_count}.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════

def save_leaderboard(long_df, model_type, output_dir, cf_count):
    sub = long_df[long_df['model'] == model_type]
    if sub.empty:
        return

    pivot = sub.pivot(index='strategy', columns='config', values='mean_auc')
    pivot = pivot.reindex(columns=[c for c in CONFIG_ORDER if c in pivot.columns])

    print(f"\nLeaderboard — {model_type} (mean AUC across 5 folds)")
    print(pivot.to_string(float_format=lambda v: f'{v:.4f}'))

    save_path = os.path.join(output_dir, f'strategy_leaderboard_{model_type}_cf{cf_count}.csv')
    pivot.to_csv(save_path)
    print(f'✓ Saved leaderboard → {save_path}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease', type=str, default='effusion')
    parser.add_argument('--cf_count', type=int, default=1)
    parser.add_argument('--models', type=str, nargs='+', default=['LR', 'RF', 'MLP'])
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  CF SELECTION STRATEGY COMPARISON (MedMNIST): {args.disease.upper()} — CF={args.cf_count}")
    print(f"{'='*70}\n")

    long_df = load_strategy_data(args.disease, args.cf_count, args.models)

    output_dir = os.path.join(RESULTS_DIR, '', 'C2_medmnist_corrected', args.disease,
                               'strategy_comparison', f'cf_{args.cf_count}')
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f'strategy_comparison_cf{args.cf_count}.csv')
    long_df.to_csv(csv_path, index=False)
    print(f'\n✓ Saved combined table → {csv_path}')

    print(f"\n{'─'*70}")
    print("  Plots + leaderboards")
    print(f"{'─'*70}")
    for model_type in args.models:
        plot_strategy_comparison(long_df, model_type, output_dir, args.cf_count, args.disease)
        save_leaderboard(long_df, model_type, output_dir, args.cf_count)

    print(f"\n{'─'*70}")
    print("  Top 5 overall (strategy, model, config, mean_auc)")
    print(f"{'─'*70}")
    top5 = long_df.sort_values('mean_auc', ascending=False).head(5)
    print(top5[['strategy', 'model', 'config', 'mean_auc', 'std_auc']].to_string(index=False))

    print(f"\n{'='*70}")
    print(f"  ✓ Done. Output saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
