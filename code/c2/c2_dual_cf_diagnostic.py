#!/usr/bin/env python3
"""
Diagnostic: does splitting CF neighbours by their correctness add signal?

For each query xᵢ we compute TWO ΔA vectors from the OPPOSITE-prediction pool:
  delta_corrnb_*  — diff to k nearest CORRECT    neighbours (TN for pred=1, TP for pred=0)
  delta_incnb_*   — diff to k nearest INCORRECT  neighbours (FN for pred=1, FP for pred=0)

Routing is pred-only (unmatched). Neighbour correctness is used only to sub-split the
search pool — legal at train time. No test-set routing, no model training.

The separation check: within each neighbour-type, does the mean ΔA of correct queries
differ from the mean ΔA of incorrect queries? Measured as centroid separation relative
to the overall mean ΔA magnitude.

Usage:
    python code/c2_dual_cf_diagnostic.py --disease effusion --cf_count 1 5 16
"""

import sys
import os
sys.path.insert(0, '/zhome/d0/a/221493/thesis/code')

import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from c2_prepare_data_simulated_cf import _get_disease_cols, add_clinical_ratios

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
BACKBONE_MAP = {
    'densenet': 'C2_custom',
    'resnet50': 'C2_resnet50',
    'vit':      'C2_vit',
}
QUADRANT_ORDER = ['TP', 'TN', 'FP', 'FN']


# ── Core: compute two ΔA vectors per sample ───────────────────────────────────
def _compute_dual_cf(df, cf_count, disease, metric='manhattan'):
    """
    Returns (df_aug, relevant_cols).
    df_aug has delta_corrnb_* and delta_incnb_* columns appended.
    Operates on the full dataset — no fold split.
    """
    disease_prob_col, disease_pred_col, disease_true_col = _get_disease_cols(disease)

    df = add_clinical_ratios(df.copy())

    meta_cols     = {disease_prob_col, disease_pred_col, disease_true_col,
                     'correct', 'path', 'patient_id'}
    emb_cols      = [c for c in df.columns if c.startswith('emb_')]
    relevant_cols = [c for c in df.columns if c not in meta_cols and c not in emb_cols]

    df_clean = df.copy()
    df_clean[relevant_cols] = df_clean[relevant_cols].fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clean[relevant_cols].values.astype(float))

    preds = df_clean[disease_pred_col].values.astype(int)
    trues = df_clean[disease_true_col].values.astype(int)

    # 4 sub-pools: pred-routing first, correctness split within
    masks = {
        'pred0_corr':   (preds == 0) & (trues == 0),   # TN — corr-nb for pred=1 queries
        'pred0_incorr': (preds == 0) & (trues == 1),   # FN — inc-nb  for pred=1 queries
        'pred1_corr':   (preds == 1) & (trues == 1),   # TP — corr-nb for pred=0 queries
        'pred1_incorr': (preds == 1) & (trues == 0),   # FP — inc-nb  for pred=0 queries
    }
    pool_sizes = {k: int(v.sum()) for k, v in masks.items()}
    print(f"  Pool sizes: {pool_sizes}")

    nn_kw = {'metric': metric, 'algorithm': 'brute', 'n_jobs': -1}

    def _fit(mask_key):
        k = min(cf_count, pool_sizes[mask_key])
        if k < cf_count:
            print(f"  WARNING: pool '{mask_key}' has {pool_sizes[mask_key]} samples "
                  f"< cf_count={cf_count}; using k={k}")
        return NearestNeighbors(n_neighbors=k, **nn_kw).fit(scaled[masks[mask_key]]), k

    nn_p0c, _ = _fit('pred0_corr')
    nn_p0i, _ = _fit('pred0_incorr')
    nn_p1c, _ = _fit('pred1_corr')
    nn_p1i, _ = _fit('pred1_incorr')

    n, d = scaled.shape
    diff_corrnb = np.full((n, d), np.nan, dtype=np.float64)
    diff_incnb  = np.full((n, d), np.nan, dtype=np.float64)

    # pred=1 queries → opposite (pred=0) pools
    mask1 = (preds == 1)
    if mask1.any():
        q      = scaled[mask1]
        pool0c = scaled[masks['pred0_corr']]
        pool0i = scaled[masks['pred0_incorr']]
        _, ic  = nn_p0c.kneighbors(q)
        _, ii  = nn_p0i.kneighbors(q)
        diff_corrnb[mask1] = (q[:, np.newaxis, :] - pool0c[ic]).mean(axis=1)
        diff_incnb[mask1]  = (q[:, np.newaxis, :] - pool0i[ii]).mean(axis=1)

    # pred=0 queries → opposite (pred=1) pools
    mask0 = (preds == 0)
    if mask0.any():
        q      = scaled[mask0]
        pool1c = scaled[masks['pred1_corr']]
        pool1i = scaled[masks['pred1_incorr']]
        _, ic  = nn_p1c.kneighbors(q)
        _, ii  = nn_p1i.kneighbors(q)
        diff_corrnb[mask0] = (q[:, np.newaxis, :] - pool1c[ic]).mean(axis=1)
        diff_incnb[mask0]  = (q[:, np.newaxis, :] - pool1i[ii]).mean(axis=1)

    corrnb_cols = [f'delta_corrnb_{c}' for c in relevant_cols]
    incnb_cols  = [f'delta_incnb_{c}'  for c in relevant_cols]

    df_aug = pd.concat([
        df_clean,
        pd.DataFrame(diff_corrnb, columns=corrnb_cols, index=df_clean.index),
        pd.DataFrame(diff_incnb,  columns=incnb_cols,  index=df_clean.index),
    ], axis=1)

    assert not df_aug[corrnb_cols].isnull().any().any(), "NaN in delta_corrnb_*"
    assert not df_aug[incnb_cols].isnull().any().any(),  "NaN in delta_incnb_*"

    return df_aug, relevant_cols


# ── Quadrant label ─────────────────────────────────────────────────────────────
def _assign_quadrant(df, disease_pred_col, disease_true_col):
    p = df[disease_pred_col].values.astype(int)
    t = df[disease_true_col].values.astype(int)
    return np.where((p==1)&(t==1), 'TP',
           np.where((p==0)&(t==0), 'TN',
           np.where((p==1)&(t==0), 'FP', 'FN')))


# ── Main analysis for one cf_count ────────────────────────────────────────────
def run_diagnostic(df_raw, disease, cf_count, backbone, out_dir):
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC  disease={disease}  k={cf_count}  backbone={backbone}")
    print(f"{'='*70}")

    _, disease_pred_col, disease_true_col = _get_disease_cols(disease)

    # ── 1. Compute dual ΔA ───────────────────────────────────────────────────
    print(f"\n[1] Computing dual ΔA vectors (k={cf_count}) ...")
    df_aug, relevant_cols = _compute_dual_cf(df_raw.copy(), cf_count, disease)
    corrnb_cols = [f'delta_corrnb_{c}' for c in relevant_cols]
    incnb_cols  = [f'delta_incnb_{c}'  for c in relevant_cols]
    print(f"  {len(relevant_cols)} attribute dimensions")

    # ── 2. Quadrant labels + mean tables ─────────────────────────────────────
    print("\n[2] Quadrant mean tables ...")
    df_aug['_quadrant'] = _assign_quadrant(df_aug, disease_pred_col, disease_true_col)

    mean_corrnb = (df_aug.groupby('_quadrant')[corrnb_cols]
                         .mean().reindex(QUADRANT_ORDER))
    mean_incnb  = (df_aug.groupby('_quadrant')[incnb_cols]
                         .mean().reindex(QUADRANT_ORDER))
    mean_corrnb.index.name = 'quadrant'
    mean_incnb.index.name  = 'quadrant'

    mean_corrnb.to_csv(os.path.join(out_dir, f'dual_cf_mean_corrnb_k{cf_count}.csv'))
    mean_incnb.to_csv(os.path.join(out_dir,  f'dual_cf_mean_incnb_k{cf_count}.csv'))
    print(f"  Saved mean tables  (shape {mean_corrnb.shape})")

    # ── 3. Separation check ───────────────────────────────────────────────────
    print("\n[3] Separation check ...")

    # 3a — L2 norm of mean ΔA per quadrant
    corrnb_norms = mean_corrnb.apply(np.linalg.norm, axis=1)
    incnb_norms  = mean_incnb.apply(np.linalg.norm,  axis=1)
    norm_table   = pd.DataFrame({'||ΔA||  corrnb': corrnb_norms,
                                 '||ΔA||  incnb':  incnb_norms})
    print("\n  L2 norm of mean ΔA per quadrant:")
    print(norm_table.to_string(float_format='{:.4f}'.format))

    # 3b — Centroid separation: ||mean(correct ΔA) - mean(incorrect ΔA)||₂
    corr_mask   = df_aug['correct'].values == 1
    incorr_mask = ~corr_mask

    sep_corrnb = np.linalg.norm(
        df_aug.loc[corr_mask,   corrnb_cols].mean().values -
        df_aug.loc[incorr_mask, corrnb_cols].mean().values
    )
    sep_incnb = np.linalg.norm(
        df_aug.loc[corr_mask,   incnb_cols].mean().values -
        df_aug.loc[incorr_mask, incnb_cols].mean().values
    )

    # Normalise by overall mean ΔA magnitude (reference scale)
    ref_corrnb = np.linalg.norm(df_aug[corrnb_cols].mean().values) + 1e-9
    ref_incnb  = np.linalg.norm(df_aug[incnb_cols].mean().values)  + 1e-9
    rel_corrnb = sep_corrnb / ref_corrnb
    rel_incnb  = sep_incnb  / ref_incnb

    print(f"\n  Centroid separation  (correct vs incorrect queries):")
    print(f"    corrnb  raw={sep_corrnb:.4f}  ref={ref_corrnb:.4f}  "
          f"rel={rel_corrnb:.4f}  ({rel_corrnb*100:.1f}% of mean ΔA magnitude)")
    print(f"    incnb   raw={sep_incnb:.4f}   ref={ref_incnb:.4f}  "
          f"rel={rel_incnb:.4f}  ({rel_incnb*100:.1f}% of mean ΔA magnitude)")

    # ── 4. Verdict (threshold: rel separation > 10%) ─────────────────────────
    THRESH = 0.10
    corrnb_sep = rel_corrnb > THRESH
    incnb_sep  = rel_incnb  > THRESH

    if corrnb_sep and incnb_sep:
        verdict = ("BOTH views separate — augmentation has legs; "
                   "build dual-row training.")
    elif corrnb_sep and not incnb_sep:
        verdict = ("ONLY corr-neighbour view separates — "
                   "incorr-neighbour adds noise; do NOT build augmentation.")
    else:
        verdict = ("NEITHER view separates — the correctness split adds no signal; "
                   "reconsider direction.")

    print(f"\n{'='*70}")
    print(f"VERDICT (k={cf_count}): {verdict}")
    print(f"  threshold: rel separation > {THRESH*100:.0f}%")
    print(f"{'='*70}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Dual-CF diagnostic: separation check')
    parser.add_argument('--disease',  type=str, default='effusion')
    parser.add_argument('--cf_count', type=int, nargs='+', default=[1, 5, 16],
                        help='One or more cf_count values (default: 1 5 16)')
    parser.add_argument('--backbone', type=str, default='densenet',
                        choices=['densenet', 'resnet50', 'vit'])
    args = parser.parse_args()

    results_base = os.path.join(RESULTS_DIR, '')
    data_subdir  = BACKBONE_MAP[args.backbone]
    data_path    = os.path.join(results_base, f'{data_subdir}/{args.disease}/c2_data.csv')

    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path)
    df.rename(columns={'prob': f'{args.disease}_prob',
                       'pred': f'{args.disease}_pred',
                       'true': f'{args.disease}_true'}, inplace=True)
    assert df['path'].is_unique, "duplicate paths in c2_data.csv"
    print(f"Dataset: {len(df):,} samples")

    out_dir = os.path.join(results_base,
                           f'{data_subdir}_corrected/{args.disease}/dual_cf_diagnostic')
    os.makedirs(out_dir, exist_ok=True)

    for k in args.cf_count:
        run_diagnostic(df, args.disease, k, args.backbone, out_dir)


if __name__ == '__main__':
    main()
