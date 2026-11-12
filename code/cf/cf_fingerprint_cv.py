"""
cf_fingerprint_cv.py
====================
5-fold CV for C2 correctness prediction on the diffusion-dynamics FINGERPRINT.

The fingerprint is the 15-cell (region x t_start) trajectory/pixel feature block built by
cf_extract_diverse_features.py -> cf_features.parquet, pivoted wide to one row per image.
It is a different model from every other C2 pipeline: no CF attributes, no delta-A, no
retrieval — the features describe how the diffusion counterfactual MOVED, not what it
looks like.

So far only HistGradientBoosting has been run on it (notebooks/NewDiffusionCFAttributes.ipynb
and code/cf/cf_cell_ablation.py). This script runs the SAME LR / RF / MLP estimators used by
the c2 CV pipelines (c2_cv_pipeline_cf_ensemble.py, c2_cv_pipeline_diverse_cf.py) on the same
folds, so the fingerprint numbers are comparable across model families as well as across
feature sets.

Feature sets mirror the notebook so the HGB reference carries over unchanged:
    prob_only     orig_prob alone (this is B1, restricted to the 5,431-image grid)
    fingerprint   the 669 pivoted cell columns
    both          orig_prob + fingerprint

Missing values: the fingerprint is structurally sparse — steps_to_flip / first_cross_step are
NaN when a trajectory never crosses the decision boundary, and the align_* features exist only
for the unmasked 'general' cells. HGB reads NaN natively, so it gets the raw matrix. LR/RF/MLP
cannot, so they get a per-fold median imputation WITH missingness indicators, which preserves
"never crossed" as information instead of silently recoding it as a typical crossing time. The
imputer is fit on the training fold only.

Folds are StratifiedGroupKFold on patient_id (seed 42), identical to cf_cell_ablation.py.

Usage:
    python cf_fingerprint_cv.py
    python cf_fingerprint_cv.py --models LR RF MLP HGB
    python cf_fingerprint_cv.py --sets fingerprint both --save_models
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cf_cell_ablation as ablation

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

N_FOLDS      = 5
RANDOM_SEED  = 42
HGB_MAX_ITER = 300          # matches cf_cell_ablation.py, so its FULL row is comparable

MODEL_TYPES  = ['LR', 'RF', 'MLP', 'HGB']
SET_NAMES    = ['prob_only', 'fingerprint', 'both']


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

def make_model(model_type):
    """
    Same estimators and hyper-parameters as train_model() in the c2 CV pipelines, wrapped in
    a Pipeline so the per-fold imputer and scaler are fit on training data only.
    """
    impute = SimpleImputer(strategy='median', add_indicator=True)

    if model_type == 'LR':
        return Pipeline([
            ('impute', impute),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, class_weight='balanced',
                                       random_state=RANDOM_SEED)),
        ])

    if model_type == 'RF':
        return Pipeline([
            ('impute', impute),
            ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                           random_state=RANDOM_SEED, n_jobs=-1)),
        ])

    if model_type == 'MLP':
        return Pipeline([
            ('impute', impute),
            ('scale', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                                  early_stopping=True, validation_fraction=0.05,
                                  random_state=RANDOM_SEED)),
        ])

    if model_type == 'HGB':
        # No imputer: HGB routes NaN down its own branch, which is how the notebook and
        # cf_cell_ablation.py produced the ~0.79 reference.
        return Pipeline([
            ('clf', HistGradientBoostingClassifier(max_iter=HGB_MAX_ITER,
                                                   random_state=0)),
        ])

    raise ValueError(f"Unknown model type: {model_type}")


# ══════════════════════════════════════════════════════════════════════════════
# CV
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(models, set_names, out_dir, save_models=False):
    config, ok, X_all = ablation.build_matrix()

    fp_cols = [c for c in X_all.columns if '__' in c]
    feature_sets = {
        'prob_only':   ['orig_prob'],
        'fingerprint': fp_cols,
        'both':        ['orig_prob'] + fp_cols,
    }
    sets = {k: feature_sets[k] for k in set_names}

    y      = X_all['correct'].values
    groups = X_all['patient_id'].values
    paths  = X_all.index.values

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 74}")
    print(f"  {N_FOLDS}-FOLD CV — DIFFUSION-DYNAMICS FINGERPRINT")
    print(f"  images: {len(X_all):,}   patients: {X_all['patient_id'].nunique():,}   "
          f"correct rate: {y.mean():.3f}")
    print(f"  fingerprint columns: {len(fp_cols)}   "
          f"(NaN rate {X_all[fp_cols].isna().values.mean():.3%})")
    print(f"  MODELS={models}   SETS={list(sets)}")
    print(f"  out: {out_dir}")
    print(f"{'=' * 74}\n", flush=True)

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = {m: {s: [] for s in sets} for m in models}

    for fold_idx, (tr, te) in enumerate(cv.split(X_all, y, groups)):
        overlap = set(groups[tr]) & set(groups[te])
        print(f"\n{'-' * 74}\nFOLD {fold_idx + 1}/{N_FOLDS}   "
              f"train {len(tr):,} | test {len(te):,} | "
              f"patient overlap {len(overlap)} (should be 0)\n{'-' * 74}", flush=True)

        y_train, y_test = y[tr], y[te]
        fold_pred = pd.DataFrame({'path': paths[te], 'patient_id': groups[te],
                                  'orig_prob': X_all['orig_prob'].values[te],
                                  'correct': y_test})

        for model_type in models:
            print(f"\n  {model_type}:", flush=True)
            for set_name, cols in sets.items():
                X = X_all[cols].values.astype(float)
                model = make_model(model_type)
                model.fit(X[tr], y_train)
                y_prob = model.predict_proba(X[te])[:, 1]

                auc = float(roc_auc_score(y_test, y_prob))
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                cv_results[model_type][set_name].append({
                    'auc': auc, 'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
                    'y_prob': y_prob.tolist(), 'y_true': y_test.tolist(),
                })
                fold_pred[f"{model_type}_{set_name}_prob"] = y_prob
                print(f"    {set_name:<13} ({len(cols):>3} cols): AUC = {auc:.4f}",
                      flush=True)

                if save_models:
                    model_dir = os.path.join(out_dir, 'models', model_type)
                    os.makedirs(model_dir, exist_ok=True)
                    joblib.dump(model, os.path.join(
                        model_dir, f"{model_type}_{set_name}_fold{fold_idx}.pkl"))

        fold_pred.to_csv(os.path.join(out_dir, f'fold_{fold_idx}_predictions.csv'),
                         index=False)

    # ── aggregate ─────────────────────────────────────────────────────────────
    rows = []
    for model_type in models:
        for set_name in sets:
            aucs = [f['auc'] for f in cv_results[model_type][set_name]]
            rows.append({
                'model': model_type, 'feature_set': set_name,
                'n_features': len(sets[set_name]),
                'mean_auc': np.mean(aucs), 'std_auc': np.std(aucs),
                'min_auc': np.min(aucs), 'max_auc': np.max(aucs),
                'fold_aucs': ','.join(f'{a:.4f}' for a in aucs),
            })

    summary = pd.DataFrame(rows).sort_values('mean_auc', ascending=False)
    summary.to_csv(os.path.join(out_dir, 'cv_summary.csv'), index=False)
    with open(os.path.join(out_dir, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    with open(os.path.join(out_dir, 'run_config.json'), 'w') as f:
        json.dump({'models': models, 'sets': list(sets), 'n_folds': N_FOLDS,
                   'seed': RANDOM_SEED, 'hgb_max_iter': HGB_MAX_ITER,
                   'n_fingerprint_cols': len(fp_cols),
                   'n_images': int(len(X_all)),
                   'regions': config.get('regions')}, f, indent=2)

    print(f"\n{'=' * 74}")
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {out_dir}", flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', default=MODEL_TYPES, choices=MODEL_TYPES)
    p.add_argument('--sets', nargs='+', default=SET_NAMES, choices=SET_NAMES)
    p.add_argument('--features_dir', default=ablation.RESULTS_DIR,
                   help="Directory holding cf_features.parquet.")
    p.add_argument('--out_dir', default=None,
                   help="Default: <features_dir>/fingerprint_cv")
    p.add_argument('--hgb_max_iter', type=int, default=HGB_MAX_ITER)
    p.add_argument('--save_models', action='store_true',
                   help="Pickle every fitted fold model (RF pickles are large).")
    args = p.parse_args()

    HGB_MAX_ITER = args.hgb_max_iter
    ablation.RESULTS_DIR = args.features_dir
    ablation.FEATURES_PARQUET = os.path.join(args.features_dir, 'cf_features.parquet')

    out_dir = args.out_dir or os.path.join(args.features_dir, 'fingerprint_cv')
    run_cv(args.models, args.sets, out_dir, args.save_models)
