"""
c2_dual_cf_isolation.py — Dual-CF diagnostic (standalone, do not modify production pipeline).

Question: does providing BOTH a correct-neighbour CF AND an incorrect-neighbour CF
(dual) improve correctness prediction over a single nearest unmatched CF (single)?

5 configs
---------
B1         : prob only (sanity control — must reproduce existing B1 AUC)
B4_single  : prob + attr + cf_attr          (single mixed opposite pool)
B4_dual    : prob + attr + cf_attr_corr + cf_attr_inc
M3_single  : prob + attr + ΔA              (single mixed opposite pool)
M3_dual    : prob + attr + ΔA_corr + ΔA_inc

Fold rule
---------
All CF machinery is fit on TRAIN fold only.  Test samples are routed by their
predicted label into train-pool sub-pools (never by their own correctness/true label).

Usage
-----
    python code/c2_dual_cf_isolation.py --disease effusion --cf_count 16
    python code/c2_dual_cf_isolation.py --disease cardiomegaly --cf_count 8 --backbone resnet50
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/zhome/d0/a/221493/thesis/code')
from c2_prepare_data_simulated_cf import _get_disease_cols, add_clinical_ratios

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
BACKBONE_MAP = {
    'densenet': 'C2_custom',
    'resnet50': 'C2_resnet50',
    'vit':      'C2_vit',
}
RANDOM_SEED = 42
N_FOLDS     = 5
CONFIGS     = ['B1', 'B4_single', 'B4_dual', 'M3_single', 'M3_dual']
PAIRED      = [('B4_dual', 'B4_single'), ('M3_dual', 'M3_single')]


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_attr_cols(df, disease):
    prob_col, pred_col, true_col = _get_disease_cols(disease)
    meta = {prob_col, pred_col, true_col, 'correct', 'path', 'patient_id'}
    emb  = {c for c in df.columns if c.startswith('emb_')}
    # also exclude any delta_ columns that may have been added upstream
    delta = {c for c in df.columns if c.startswith('delta_')}
    return [c for c in df.columns if c not in meta and c not in emb and c not in delta]


# ──────────────────────────────────────────────────────────────────────────────
# _fit_pools
# ──────────────────────────────────────────────────────────────────────────────

def _fit_pools(train_df, cf_count, disease, metric='manhattan'):
    """
    Fit scaler + six NearestNeighbors pools on the training fold.

    Pools built
    -----------
    pred0, pred1          — mixed opposite pools (for single-CF configs)
    pred0_corr, pred0_inc — correctness sub-pools of pred0 (for dual-CF)
    pred1_corr, pred1_inc — correctness sub-pools of pred1 (for dual-CF)

    Returns
    -------
    dict with keys: scaler, attr_cols, pred_col, pools
    pools[name] = {'nn': NearestNeighbors, 'scaled': np.ndarray, 'k': int}
    """
    _, pred_col, _ = _get_disease_cols(disease)

    train_df = add_clinical_ratios(train_df.copy())
    acols    = _get_attr_cols(train_df, disease)

    train_df[acols] = train_df[acols].fillna(0)

    scaler       = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[acols].values.astype(float))

    preds   = train_df[pred_col].values.astype(int)
    correct = train_df['correct'].values.astype(int)

    masks = {
        'pred0':      preds == 0,
        'pred1':      preds == 1,
        'pred0_corr': (preds == 0) & (correct == 1),
        'pred0_inc':  (preds == 0) & (correct == 0),
        'pred1_corr': (preds == 1) & (correct == 1),
        'pred1_inc':  (preds == 1) & (correct == 0),
    }

    pools = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        k = min(cf_count, n)
        if n < cf_count:
            warnings.warn(
                f"[dual-cf] Pool '{name}' has only {n} train samples "
                f"(cf_count={cf_count}); capping at k={k}.",
                stacklevel=2,
            )
        pool_scaled = train_scaled[mask]
        nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
        nn.fit(pool_scaled)
        pools[name] = {'nn': nn, 'scaled': pool_scaled, 'k': k}

    return {
        'scaler':    scaler,
        'attr_cols': acols,
        'pred_col':  pred_col,
        'pools':     pools,
    }


# ──────────────────────────────────────────────────────────────────────────────
# _apply
# ──────────────────────────────────────────────────────────────────────────────

def _apply(fitted, query_df, disease):
    """
    Retrieve CF features for every query sample.

    Routing rule (fold-clean)
    -------------------------
    - pred=1 queries  →  opposite pools: pred0, pred0_corr, pred0_inc
    - pred=0 queries  →  opposite pools: pred1, pred1_corr, pred1_inc
    Test samples are NEVER routed by their own correctness label.

    Returns dict of np.ndarrays
    ---------------------------
    prob          (n,1)   disease probability
    attrs         (n,d)   query's own scaled attributes
    cf_attr       (n,d)   mean attrs of k nearest mixed-pool CFs      (single)
    cf_attr_corr  (n,d)   mean attrs of k nearest correct-pool CFs    (dual)
    cf_attr_inc   (n,d)   mean attrs of k nearest incorrect-pool CFs  (dual)
    delta         (n,d)   query_scaled − cf_attr                      (single)
    delta_corr    (n,d)   query_scaled − cf_attr_corr                 (dual)
    delta_inc     (n,d)   query_scaled − cf_attr_inc                  (dual)
    y             (n,)    correctness labels
    """
    scaler   = fitted['scaler']
    acols    = fitted['attr_cols']
    pools    = fitted['pools']
    pred_col = fitted['pred_col']

    prob_col, _, _ = _get_disease_cols(disease)

    query_df = add_clinical_ratios(query_df.copy())
    query_df[acols] = query_df[acols].fillna(0)

    query_scaled = scaler.transform(query_df[acols].values.astype(float))
    n, d = query_scaled.shape
    preds = query_df[pred_col].values.astype(int)

    cf_attr      = np.full((n, d), np.nan)
    cf_attr_corr = np.full((n, d), np.nan)
    cf_attr_inc  = np.full((n, d), np.nan)

    for pred_val in [0, 1]:
        opp  = 1 - pred_val
        mask = preds == pred_val
        if not mask.any():
            continue
        q = query_scaled[mask]

        # single: mixed opposite pool
        _, ix  = pools[f'pred{opp}']['nn'].kneighbors(q)
        cf_attr[mask] = pools[f'pred{opp}']['scaled'][ix].mean(axis=1)

        # dual: correct sub-pool
        _, ix_c = pools[f'pred{opp}_corr']['nn'].kneighbors(q)
        cf_attr_corr[mask] = pools[f'pred{opp}_corr']['scaled'][ix_c].mean(axis=1)

        # dual: incorrect sub-pool
        _, ix_i = pools[f'pred{opp}_inc']['nn'].kneighbors(q)
        cf_attr_inc[mask] = pools[f'pred{opp}_inc']['scaled'][ix_i].mean(axis=1)

    assert not np.isnan(cf_attr).any(),      "NaN in cf_attr — routing missed some samples"
    assert not np.isnan(cf_attr_corr).any(), "NaN in cf_attr_corr"
    assert not np.isnan(cf_attr_inc).any(),  "NaN in cf_attr_inc"

    delta      = query_scaled - cf_attr
    delta_corr = query_scaled - cf_attr_corr
    delta_inc  = query_scaled - cf_attr_inc

    return {
        'prob':         query_df[[prob_col]].values.astype(float),
        'attrs':        query_scaled,
        'cf_attr':      cf_attr,
        'cf_attr_corr': cf_attr_corr,
        'cf_attr_inc':  cf_attr_inc,
        'delta':        delta,
        'delta_corr':   delta_corr,
        'delta_inc':    delta_inc,
        'y':            query_df['correct'].values.astype(int),
    }


# ──────────────────────────────────────────────────────────────────────────────
# build_configs
# ──────────────────────────────────────────────────────────────────────────────

def build_configs(tr, te):
    """
    Build (X_train, X_test) for every config from the _apply output dicts.

    Config definitions
    ------------------
    B1         : [prob]
    B4_single  : [prob, attrs, cf_attr]
    B4_dual    : [prob, attrs, cf_attr_corr, cf_attr_inc]
    M3_single  : [prob, attrs, delta]
    M3_dual    : [prob, attrs, delta_corr, delta_inc]
    """
    def stack(d, *keys):
        return np.hstack([d[k] for k in keys])

    return {
        'B1':        (tr['prob'],                                             te['prob']),
        'B4_single': (stack(tr, 'prob', 'attrs', 'cf_attr'),                 stack(te, 'prob', 'attrs', 'cf_attr')),
        'B4_dual':   (stack(tr, 'prob', 'attrs', 'cf_attr_corr', 'cf_attr_inc'),
                      stack(te, 'prob', 'attrs', 'cf_attr_corr', 'cf_attr_inc')),
        'M3_single': (stack(tr, 'prob', 'attrs', 'delta'),                   stack(te, 'prob', 'attrs', 'delta')),
        'M3_dual':   (stack(tr, 'prob', 'attrs', 'delta_corr', 'delta_inc'), stack(te, 'prob', 'attrs', 'delta_corr', 'delta_inc')),
    }


# ──────────────────────────────────────────────────────────────────────────────
# model training
# ──────────────────────────────────────────────────────────────────────────────

def _train_lr(X_tr, X_te, y_tr, y_te):
    sc = StandardScaler()
    m  = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=RANDOM_SEED)
    m.fit(sc.fit_transform(X_tr), y_tr)
    return roc_auc_score(y_te, m.predict_proba(sc.transform(X_te))[:, 1])


def _train_mlp(X_tr, X_te, y_tr, y_te):
    sc = StandardScaler()
    m  = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.05,
        random_state=RANDOM_SEED,
    )
    m.fit(sc.fit_transform(X_tr.astype(np.float32)), y_tr)
    return roc_auc_score(y_te, m.predict_proba(sc.transform(X_te.astype(np.float32)))[:, 1])


def _train_rf(X_tr, X_te, y_tr, y_te):
    m = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                               random_state=RANDOM_SEED, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])


# ──────────────────────────────────────────────────────────────────────────────
# run
# ──────────────────────────────────────────────────────────────────────────────

def run(disease, cf_count, backbone):
    print(f"\n{'='*70}")
    print(f"  DUAL-CF ISOLATION  |  disease={disease}  cf_count={cf_count}  backbone={backbone}")
    print(f"{'='*70}\n")

    data_subdir = BACKBONE_MAP[backbone]
    data_path   = os.path.join(RESULTS_DIR, '', data_subdir, disease, 'c2_data.csv')
    full_df = pd.read_csv(data_path)

    full_df.rename(columns={
        'prob': f'{disease}_prob',
        'pred': f'{disease}_pred',
        'true': f'{disease}_true',
    }, inplace=True)

    assert full_df['path'].is_unique, "Duplicate paths in full dataset"

    print(f"Loaded {len(full_df):,} samples  "
          f"(correct={( full_df['correct']==1).sum():,}, "
          f"incorrect={(full_df['correct']==0).sum():,})\n")

    y = full_df['correct'].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # fold_aucs[model][config] = list of per-fold AUCs
    fold_aucs = {
        model: {cfg: [] for cfg in CONFIGS}
        for model in ('LR', 'RF', 'MLP')
    }

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(full_df, y)):
        fold_train = full_df.iloc[tr_idx].reset_index(drop=True)
        fold_test  = full_df.iloc[te_idx].reset_index(drop=True)

        assert fold_train['path'].is_unique, f"Duplicate paths in fold {fold_idx} train"

        print(f"{'─'*60}")
        print(f"FOLD {fold_idx+1}/{N_FOLDS}  train={len(fold_train):,}  test={len(fold_test):,}")
        print(f"{'─'*60}")

        fitted   = _fit_pools(fold_train, cf_count, disease)
        tr_feats = _apply(fitted, fold_train, disease)
        te_feats = _apply(fitted, fold_test,  disease)

        configs = build_configs(tr_feats, te_feats)

        y_tr = tr_feats['y']
        y_te = te_feats['y']

        for model_name, train_fn in [('LR', _train_lr), ('RF', _train_rf), ('MLP', _train_mlp)]:
            print(f"  {model_name}:")
            for cfg in CONFIGS:
                X_tr, X_te = configs[cfg]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    auc = train_fn(X_tr, X_te, y_tr, y_te)
                fold_aucs[model_name][cfg].append(auc)
                print(f"    {cfg:12s}: {auc:.4f}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY  (mean ± std AUC across 5 folds)")
    print(f"{'='*70}")

    rows = []
    for model_name in ('LR', 'RF', 'MLP'):
        print(f"\n  {model_name}:")
        for cfg in CONFIGS:
            aucs = fold_aucs[model_name][cfg]
            m, s = np.mean(aucs), np.std(aucs)
            fold_str = '  '.join(f'{a:.4f}' for a in aucs)
            print(f"    {cfg:12s}: {m:.4f} ± {s:.4f}   [{fold_str}]")
            rows.append({
                'model':     model_name,
                'config':    cfg,
                'mean_auc':  round(m, 4),
                'std_auc':   round(s, 4),
                'fold_aucs': ','.join(f'{a:.4f}' for a in aucs),
            })

    summary_df = pd.DataFrame(rows)

    # ── Paired comparison table ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PAIRED COMPARISONS  (dual vs single)")
    print(f"{'='*70}")

    paired_rows = []
    for new_cfg, base_cfg in PAIRED:
        for model_name in ('LR', 'RF', 'MLP'):
            r_new  = summary_df[(summary_df['model'] == model_name) & (summary_df['config'] == new_cfg)].iloc[0]
            r_base = summary_df[(summary_df['model'] == model_name) & (summary_df['config'] == base_cfg)].iloc[0]
            delta  = round(r_new['mean_auc'] - r_base['mean_auc'], 4)
            paired_rows.append({
                'model':        model_name,
                'config':       new_cfg,
                'baseline':     base_cfg,
                'mean_auc':     r_new['mean_auc'],
                'baseline_auc': r_base['mean_auc'],
                'delta_auc':    delta,
                'std':          r_new['std_auc'],
            })

    paired_df = pd.DataFrame(paired_rows)
    print(paired_df.to_string(index=False))

    # ── Verdicts ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICTS")
    print(f"{'='*70}")
    for new_cfg, base_cfg in PAIRED:
        print(f"\n  {new_cfg} vs {base_cfg}:")
        for model_name in ('LR', 'RF', 'MLP'):
            pr = paired_df[(paired_df['model'] == model_name) & (paired_df['config'] == new_cfg)].iloc[0]
            delta = pr['delta_auc']
            std   = pr['std']
            if abs(delta) <= std:
                verdict = f"NO DIFFERENCE (|Δ|={delta:+.4f} ≤ std={std:.4f})"
            elif delta > std:
                verdict = f"DUAL WINS     (Δ={delta:+.4f} > std={std:.4f})"
            else:
                verdict = f"DUAL WORSE    (Δ={delta:+.4f}, |Δ|={abs(delta):.4f} > std={std:.4f})"
            print(f"    {model_name}: {verdict}")

    # ── Save CSV ───────────────────────────────────────────────────────────
    out_dir  = os.path.join(RESULTS_DIR, '', 'dual_cf_isolation', disease)
    os.makedirs(out_dir, exist_ok=True)

    tag     = f'{backbone}_cf{cf_count}'
    sum_csv  = os.path.join(out_dir, f'summary_{tag}.csv')
    pair_csv = os.path.join(out_dir, f'paired_{tag}.csv')
    summary_df.to_csv(sum_csv, index=False)
    paired_df.to_csv(pair_csv, index=False)
    print(f"\nSaved: {sum_csv}")
    print(f"Saved: {pair_csv}\n")

    return summary_df, paired_df


# ──────────────────────────────────────────────────────────────────────────────
# entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual-CF isolation experiment')
    parser.add_argument('--disease',   type=str, default='effusion')
    parser.add_argument('--cf_count',  type=int, default=16)
    parser.add_argument('--backbone',  type=str, default='densenet',
                        choices=list(BACKBONE_MAP.keys()))
    args = parser.parse_args()
    run(disease=args.disease, cf_count=args.cf_count, backbone=args.backbone)
