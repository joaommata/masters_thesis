"""
ablation_delta_categories_fast.py
==================================
Ablation study for delta-A feature categories in M3.
Reads pre-saved fold CSVs instead of recomputing CFs.

To test with one fold:  change range(N_FOLDS) to range(1) in the loading section
To run fully:           keep range(N_FOLDS)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# ── CONFIG ────────────────────────────────────────────────────────────────────
DISEASE  = 'effusion'
CF_COUNT = 16
N_FOLDS  = 5
SEED     = 42
MODEL   = 'mlp'
CV_DIR   = f'/zhome/d0/a/221493/thesis/results/C2_custom/{DISEASE}/cv_results/cf_{CF_COUNT}'
OUT_DIR  = f'/zhome/d0/a/221493/thesis/results/ablation_delta/{DISEASE}_cf{CF_COUNT}_{MODEL}'
os.makedirs(OUT_DIR, exist_ok=True)

META_COLS = {
    f'{DISEASE}_prob', f'{DISEASE}_pred', f'{DISEASE}_true',
    'correct', 'path', 'patient_id', 'cf_prob', 'cf_paths', 'margin'
}

ORGANS = [
    "Left Lung", "Right Lung", "Heart", "Mediastinum",
    "Facies Diaphragmatica", "Aorta", "Left Hilus Pulmonis",
    "Right Hilus Pulmonis", "Left Scapula", "Right Scapula",
    "Left Clavicle", "Right Clavicle", "Weasand", "Spine"
]
FEATURE_TYPES = {
    'shape2D':    lambda col: 'shape2D'    in col,
    'firstorder': lambda col: 'firstorder' in col,
    'geometric':  lambda col: any(t in col for t in ['area_pixels', 'bbox', 'perimeter']),
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def build_categories(delta_cols):
    """Group delta columns by (organ, feature_type), plus demographics and ratios."""
    categories = {}

    for organ in ORGANS:
        for ftype, match_fn in FEATURE_TYPES.items():
            cols = [c for c in delta_cols if organ in c and match_fn(c)]
            if cols:
                categories[f"{organ} | {ftype}"] = cols

    demo = [c for c in delta_cols if any(t in c for t in ['age', 'sex', 'race'])]
    if demo:
        categories['Global | demographic'] = demo

    ratios = [c for c in delta_cols if 'ratio' in c or 'fraction' in c]
    if ratios:
        categories['Global | ratio'] = ratios

    return categories


def get_m3_cols(df, drop_delta_cols=None):
    """Return column names for M3 = prob + delta + attr, optionally dropping some delta cols."""
    drop_delta_cols = set(drop_delta_cols or [])

    # Exclude model output probability columns (LR_B1_prob etc.)
    model_prob_cols = {c for c in df.columns if c.endswith('_prob') and
                       any(m in c for m in ['LR_', 'RF_', 'MLP_'])}

    delta_cols = [c for c in df.columns
                  if c.startswith('delta_') and c not in drop_delta_cols]

    attr_cols = [c for c in df.columns
                 if c not in META_COLS
                 and c not in model_prob_cols
                 and not c.startswith('delta_')
                 and not c.startswith('emb_')]

    return [f'{DISEASE}_prob'] + delta_cols + attr_cols


def get_train_test(fold_dfs, test_fold_idx):
    """
    Return train and test DataFrames for a given fold.
    If only one fold is loaded (smoke test), train and test on the same fold.
    """
    test_df = fold_dfs[test_fold_idx]

    if len(fold_dfs) == 1:
        # Smoke test: no real train/test split, just checking the script runs
        train_df = fold_dfs[0]
    else:
        train_df = pd.concat(
            [fold_dfs[i] for i in range(len(fold_dfs)) if i != test_fold_idx],
            ignore_index=True
        )

    return train_df, test_df


def train_lr(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    model  = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=SEED)
    model.fit(scaler.fit_transform(X_train), y_train)
    y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]
    return float(roc_auc_score(y_test, y_prob))

def train_mlp(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                              early_stopping=True, validation_fraction=0.05, 
                              random_state=SEED)
    X_train_f32 = X_train.astype(np.float32)
    X_test_f32 = X_test.astype(np.float32)

        # Fit the MLP with sample weights
    model.fit(scaler.fit_transform(X_train_f32), y_train)
    y_prob = model.predict_proba(scaler.transform(X_test_f32))[:, 1]
    return float(roc_auc_score(y_test, y_prob))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():

    # ── Load fold CSVs ────────────────────────────────────────────────────
    # Change range(1) to range(N_FOLDS) for the full run
    fold_dfs = []
    for i in range(N_FOLDS):
        path = os.path.join(CV_DIR, f'fold_{i}_predictions.csv')
        fold_dfs.append(pd.read_csv(path))
        print(f"Loaded fold {i}: {len(fold_dfs[-1]):,} rows")

    if len(fold_dfs) == 1:
        print("NOTE: Running smoke test with 1 fold — AUC values are not meaningful")

    # ── Build delta categories from fold 0 ────────────────────────────────
    delta_cols = [c for c in fold_dfs[0].columns if c.startswith('delta_')]
    categories = build_categories(delta_cols)
    print(f"\nFound {len(categories)} delta categories")
    for k, v in categories.items():
        print(f"  {k}: {len(v)} features")

    results = []

    # ── Baseline: full M3 ─────────────────────────────────────────────────
    print("\nBaseline (full M3) MLP cf16...")
    baseline_aucs = []

    for test_fold_idx in range(len(fold_dfs)):
        train_df, test_df = get_train_test(fold_dfs, test_fold_idx)
        cols = get_m3_cols(train_df)
#        auc  = train_lr(train_df[cols].values, test_df[cols].values,
#                        train_df['correct'].values, test_df['correct'].values)
        auc  = train_mlp(train_df[cols].values, test_df[cols].values,
                        train_df['correct'].values, test_df['correct'].values)
        baseline_aucs.append(auc)
        print(f"  Fold {test_fold_idx}: {auc:.4f}")

    baseline_mean = np.mean(baseline_aucs)
    baseline_std  = np.std(baseline_aucs)
    print(f"  Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
    results.append({'category': 'FULL M3 (MLP)', 'n_dropped': 0,
                    'mean_auc': baseline_mean, 'std_auc': baseline_std, 'auc_drop': 0.0})

    # ── Ablation: drop one category at a time ─────────────────────────────
    for cat_name, drop_cols in categories.items():
        print(f"\nAblating: {cat_name} ({len(drop_cols)} features)...")
        fold_aucs = []

        for test_fold_idx in range(len(fold_dfs)):
            train_df, test_df = get_train_test(fold_dfs, test_fold_idx)
            cols = get_m3_cols(train_df, drop_delta_cols=drop_cols)
#            auc  = train_lr(train_df[cols].values, test_df[cols].values,
#                            train_df['correct'].values, test_df['correct'].values)
            auc  = train_mlp(train_df[cols].values, test_df[cols].values,
                            train_df['correct'].values, test_df['correct'].values)
            fold_aucs.append(auc)

        mean_auc = np.mean(fold_aucs)
        std_auc  = np.std(fold_aucs)
        drop     = baseline_mean - mean_auc
        print(f"  AUC: {mean_auc:.4f} ± {std_auc:.4f}  (drop: {drop:+.4f})")
        results.append({'category': cat_name, 'n_dropped': len(drop_cols),
                        'mean_auc': mean_auc, 'std_auc': std_auc, 'auc_drop': drop})

    # ── Save results ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values('auc_drop', ascending=False)
    results_df.to_csv(os.path.join(OUT_DIR, 'ablation_results_mlp_16cf.csv'), index=False)

    # ── Plot ──────────────────────────────────────────────────────────────
    ablation_only = results_df[results_df['category'] != 'FULL M3 (MLP)']
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#CC0000' if d > 0 else '#009E73' for d in ablation_only['auc_drop']]
    ax.barh(ablation_only['category'], ablation_only['auc_drop'], color=colors)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('AUC drop (positive = removing this hurts performance)')
    ax.set_title(f'ΔA Category Ablation — {DISEASE.capitalize()} M3 (MLP) (CF={CF_COUNT})')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ablation_delta_categories_mlp_16cf.png'), dpi=300)
    plt.close()

    print(f"\nDone. Results saved to {OUT_DIR}")


if __name__ == '__main__':
    main()