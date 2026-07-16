"""
c2_cv_pipeline_emb_cf.py
========================
Standalone experiment: same 5-fold CV pipeline / model configs as
c2_cv_pipeline_new_split.py, but the counterfactual neighbour is selected by
embedding-space similarity (cosine by default) instead of attribute-space
distance. CF strategy: correct-CF routing (pred=1 -> nearest TN, pred=0 -> nearest TP).

Does not modify c2_cv_pipeline_new_split.py or c2_prepare_data_simulated_cf.py —
reuses their training/feature-matrix helpers as-is.

Usage:
    python c2_cv_pipeline_emb_cf.py --disease effusion --cf_count 1 --backbone densenet --distance cosine
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.append('/zhome/d0/a/221493/thesis/code/c2')
from c2_prepare_data_simulated_cf import attach_cf_features
from c2_prepare_data_simulated_cf_emb import compute_cf_for_split_correct_cf_emb
from c2_cv_pipeline_new_split import (
    train_model, score_cf_anchor, build_feature_matrices,
    CONFIGS, BACKBONE_MAP, RESULTS_DIR, N_FOLDS, RANDOM_SEED,
)


def run_cv_emb(disease, cf_count, backbone='densenet', distance='cosine', save_fold_data=False):
    print(f"\n{'='*70}")
    print(f"  {N_FOLDS}-FOLD CV (EMBEDDING-SPACE CF): {disease.upper()} | CF={cf_count} | "
          f"BACKBONE={backbone} | DISTANCE={distance}")
    print(f"{'='*70}\n")

    results_base = os.path.join(RESULTS_DIR, '')
    data_subdir  = BACKBONE_MAP[backbone]

    cv_subdir = 'cv_results_correct_cf_emb'
    cv_dir    = os.path.join(results_base, f'{data_subdir}_corrected/{disease}/{cv_subdir}/cf_{cf_count}')
    cv_dir    = os.path.join(cv_dir, f'distance_{distance}') if distance != 'cosine' else cv_dir

    fold_data_dir = os.path.join(cv_dir, 'fold_data')
    os.makedirs(cv_dir, exist_ok=True)
    os.makedirs(fold_data_dir, exist_ok=True)

    full_df = pd.read_csv(os.path.join(results_base, f'{data_subdir}/{disease}/c2_data.csv'))
    full_df.rename(columns={
        'prob': f'{disease}_prob',
        'pred': f'{disease}_pred',
        'true': f'{disease}_true'
    }, inplace=True)

    print(f"Backbone:  {backbone}  ->  {data_subdir}/c2_data.csv")
    print(f"CV Subdir: {cv_subdir}")
    print(f"CF Count:  {cf_count}")
    print(f"Distance:  {distance}\n")
    print(f"Full dataset: {len(full_df):,} samples")
    print(f"  Correct:   {(full_df['correct']==1).sum():,}")
    print(f"  Incorrect: {(full_df['correct']==0).sum():,}\n")

    y = full_df['correct'].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    cv_results = {model_type: {config: [] for config in CONFIGS} for model_type in ['LR', 'RF', 'MLP']}
    cv_results['cf_anchor'] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, y)):
        print(f"\n{'-'*70}\nFOLD {fold_idx + 1}/{N_FOLDS}\n{'-'*70}")

        fold_train = full_df.iloc[train_idx].reset_index(drop=True)
        fold_test  = full_df.iloc[test_idx].reset_index(drop=True)
        print(f"  Train: {len(fold_train):,}  |  Test: {len(fold_test):,}")

        fold_train_cf, fold_test_cf, _ = compute_cf_for_split_correct_cf_emb(
            train_df=fold_train, test_df=fold_test,
            cf_count=cf_count, disease=disease, distance=distance
        )
        print(f"  Generating {cf_count} embedding-space counterfactuals...")

        _meta_cf = {f'{disease}_prob', f'{disease}_pred', f'{disease}_true',
                    'correct', 'path', 'patient_id', 'cf_prob', 'cf_paths'}
        _emb_cols_cf = [c for c in fold_train_cf.columns if c.startswith('emb_')]
        _rel_cols_cf = [c for c in fold_train_cf.columns
                        if c not in _meta_cf and not c.startswith('emb_') and not c.startswith('delta_')]
        _train_lookup = fold_train_cf
        fold_train_cf = attach_cf_features(fold_train_cf, _train_lookup, _rel_cols_cf, _emb_cols_cf, disease)
        fold_test_cf  = attach_cf_features(fold_test_cf,  _train_lookup, _rel_cols_cf, _emb_cols_cf, disease)

        if save_fold_data:
            fold_train_cf.to_csv(os.path.join(fold_data_dir, f'fold_{fold_idx}_train.csv'), index=False)
            fold_test_cf.to_csv( os.path.join(fold_data_dir, f'fold_{fold_idx}_test.csv'),  index=False)

        feature_sets = build_feature_matrices(fold_train_cf, fold_test_cf, disease)
        y_train = fold_train_cf['correct'].values
        y_test  = fold_test_cf['correct'].values
        fold_pred_df = fold_test_cf.copy()

        cf_anchor_scores = score_cf_anchor(fold_train_cf, fold_test_cf)
        cf_anchor_auc = float(roc_auc_score(y_test, cf_anchor_scores))
        print(f"  CF-Anchor baseline AUC: {cf_anchor_auc:.4f}")
        cv_results['cf_anchor'].append(cf_anchor_auc)

        for model_type in ['LR', 'RF', 'MLP']:
            print(f"\n  {model_type}:")
            model_dir = os.path.join(cv_dir, 'models', model_type)
            os.makedirs(model_dir, exist_ok=True)

            for config in CONFIGS:
                if config not in feature_sets:
                    continue

                X_train, X_test = feature_sets[config]
                res = train_model(X_train, X_test, y_train, y_test, model_type=model_type)

                model_path = os.path.join(model_dir, f"{model_type}_{config}_fold{fold_idx}.pkl")
                if model_type in ['LR', 'MLP']:
                    joblib.dump({'model': res['model'], 'scaler': res['scaler']}, model_path)
                else:
                    joblib.dump(res['model'], model_path)

                cv_results[model_type][config].append({
                    'auc': res['auc'], 'fpr': res['fpr'], 'tpr': res['tpr'],
                    'y_prob': res['y_prob'], 'y_true': res['y_true']
                })
                print(f"    {config}: AUC = {res['auc']:.4f}")
                fold_pred_df[f"{model_type}_{config}_prob"] = res['y_prob']

        pred_csv_path = os.path.join(cv_dir, f'fold_{fold_idx}_predictions.csv')
        fold_pred_df.to_csv(pred_csv_path, index=False)
        print(f"Fold {fold_idx} predictions saved to: {pred_csv_path}")

    print(f"\n{'='*70}\nAGGREGATING RESULTS\n{'='*70}\n")

    summary_rows = []
    for model_type in ['LR', 'RF', 'MLP']:
        for config in CONFIGS:
            aucs = [fold_res['auc'] for fold_res in cv_results[model_type][config]]
            if not aucs:
                continue
            summary_rows.append({
                'model': model_type, 'config': config,
                'mean_auc': np.mean(aucs), 'std_auc': np.std(aucs),
                'min_auc': np.min(aucs), 'max_auc': np.max(aucs),
                'fold_aucs': ','.join([f'{a:.4f}' for a in aucs])
            })

    aucs = cv_results['cf_anchor']
    summary_rows.append({
        'model': 'cf_anchor', 'config': 'cf_anchor',
        'mean_auc': np.mean(aucs), 'std_auc': np.std(aucs),
        'fold_aucs': ','.join([f'{a:.4f}' for a in aucs])
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(cv_dir, 'cv_summary.csv'), index=False)

    with open(os.path.join(cv_dir, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {cv_dir}")

    return cv_results, summary_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run C2 CV with embedding-space CF selection')
    parser.add_argument('--disease',  type=str, default='effusion')
    parser.add_argument('--cf_count', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='densenet',
                        choices=['densenet', 'resnet50', 'vit'])
    parser.add_argument('--distance', type=str, default='cosine',
                        choices=['l1', 'l2', 'cosine'])
    parser.add_argument('--save_folds', action='store_true')
    args = parser.parse_args()

    run_cv_emb(
        disease=args.disease,
        cf_count=args.cf_count,
        backbone=args.backbone,
        distance=args.distance,
        save_fold_data=args.save_folds,
    )
