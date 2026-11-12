"""Backfill the `cf_paths` column into existing fold_*_predictions.csv files.

The cf_ensemble pipeline used to keep CF assignments only as in-memory index
arrays, so the fold prediction CSVs it wrote have no `cf_paths` column — but
the CNN baselines (c2_cnn_all.py) require one. The fold split
(StratifiedGroupKFold, fixed seed) and KNN retrieval are deterministic, so the
assignment is reconstructed exactly without retraining. Row alignment with each
existing CSV is asserted on `path` before anything is written, and files are
replaced atomically.

Only unbalanced runs are supported: --balance subsamples the train fold with a
per-fold seed, which changes the CF pool, and those runs live in balanced_*
subdirs this script never touches.

Usage:
    python c2_backfill_cf_paths.py --disease effusion --cf_count 16 --entropy
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_cv_pipeline_cf_ensemble as pipe


def backfill(disease, cf_count, backbone='densenet', distance='l1',
             entropy=False, extended=False, overwrite=False):
    data_subdir = pipe.BACKBONE_MAP[backbone]
    cv_subdir = 'cv_results_extended_cf_ensemble' if extended else 'cv_results_cf_ensemble'
    cv_dir = os.path.join(pipe.RESULTS_DIR,
                          f'{data_subdir}_corrected/{disease}/{cv_subdir}/cf_{cf_count}')
    if distance != 'l1':
        cv_dir = os.path.join(cv_dir, f'distance_{distance}')
    if entropy:
        cv_dir = os.path.join(cv_dir, 'TEST_entropy')
    print(f"Target: {cv_dir}")

    data_csv = 'c2_data_extended.csv' if extended else 'c2_data.csv'
    full_df = pd.read_csv(os.path.join(pipe.RESULTS_DIR, f'{data_subdir}/{disease}/{data_csv}'))
    full_df.rename(columns={
        'prob': f'{disease}_prob',
        'pred': f'{disease}_pred',
        'true': f'{disease}_true',
    }, inplace=True)

    y = full_df['correct'].values
    groups = full_df['patient_id'].values
    skf = StratifiedGroupKFold(n_splits=pipe.N_FOLDS, shuffle=True,
                               random_state=pipe.RANDOM_SEED)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, y, groups=groups)):
        csv_path = os.path.join(cv_dir, f'fold_{fold_idx}_predictions.csv')
        existing = pd.read_csv(csv_path)
        if 'cf_paths' in existing.columns and not overwrite:
            print(f"Fold {fold_idx}: cf_paths already present, skipping")
            continue

        fold_train = full_df.iloc[train_idx].reset_index(drop=True)
        fold_test = full_df.iloc[test_idx].reset_index(drop=True)

        (train_clean, test_clean, _train_cf_idx, test_cf_idx,
         _ts, _qs, _rc, _ec) = pipe.compute_ranked_cf_correct_cf(
            train_df=fold_train, test_df=fold_test,
            cf_count=cf_count, disease=disease, distance=distance,
        )

        # The saved CSV must be the same rows in the same order as the
        # reconstructed test split, otherwise the run used different data.
        if len(existing) != len(test_clean) or \
                not (existing['path'].values == test_clean['path'].values).all():
            raise RuntimeError(
                f"Fold {fold_idx}: row mismatch between {csv_path} and the "
                f"reconstructed split — was the run made from a different "
                f"{data_csv} or with --balance?")

        train_paths = train_clean['path'].values
        existing['cf_paths'] = [
            '|'.join(train_paths[test_cf_idx[i]]) for i in range(len(test_cf_idx))
        ]

        tmp_path = csv_path + '.tmp'
        existing.to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)
        print(f"Fold {fold_idx}: wrote cf_paths ({len(existing):,} rows, k={cf_count})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--disease', type=str, default='effusion')
    parser.add_argument('--cf_count', type=int, required=True)
    parser.add_argument('--backbone', type=str, default='densenet',
                        choices=list(pipe.BACKBONE_MAP))
    parser.add_argument('--distance', type=str, default='l1',
                        choices=['l1', 'l2', 'cosine'])
    parser.add_argument('--entropy', action='store_true',
                        help='Target the TEST_entropy/ subdirectory')
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute even if cf_paths is already present')
    args = parser.parse_args()

    backfill(disease=args.disease, cf_count=args.cf_count, backbone=args.backbone,
             distance=args.distance, entropy=args.entropy, extended=args.extended,
             overwrite=args.overwrite)
