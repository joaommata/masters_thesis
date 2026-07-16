"""
c2_cv_pipeline_dispersion.py

5-fold cross-validation pipeline for C2 quality control models with CF
DISPERSION features.

Motivation
----------
The standard pipeline (c2_cv_pipeline_new_split.py) retrieves k counterfactual
neighbours per sample and collapses them with a MEAN — every CF-derived column
(`cf_prob`, `cf_attr_*`, `cf_emb_*`, `delta_*`) is a first moment over the k
neighbours. Two facts about that collapse:

  1. The SECOND moment is discarded. Two samples whose k CFs have disease
     probabilities [0.5, 0.5, ..., 0.5] and [0.05, 0.95, 0.05, 0.95, ...] both
     get `cf_prob = 0.5`, and no existing config can tell them apart. Yet the
     first sits in a region where every counterfactual agrees, and the second on
     a knife-edge where they contradict each other. The latter is exactly where
     C0 is likely to be wrong — which is what C2 exists to predict.

  2. The DISTANCES are thrown away. Every retrieval in the standard pipeline
     reads `_, idxs = nn.kneighbors(...)`; sklearn already computed the
     distances and they are dropped on the spot. How far the nearest
     counterfactual is, and how spread the neighbourhood is, are free signals
     about whether the query sits in a dense or sparse part of attribute space.

This script keeps the one-row-per-sample structure of the standard pipeline (one
model, one prediction per sample — no ensembling; see c2_cv_pipeline_cf_ensemble.py
for that orthogonal idea) but enriches the collapse with a small dispersion block:

    cf_prob_std : std over the k CFs' disease probabilities
    dist_1      : distance to the nearest CF
    dist_mean   : mean distance over the k CFs
    dist_range  : d_k - d_1, the spread of the neighbourhood

Only 4 scalars, so config width barely moves and any AUC change is attributable
to the dispersion signal rather than to added model capacity.

NOTE: dispersion is identically zero at k=1 (a single neighbour has no spread),
so MCF6/MCF7 collapse onto MCF5/MCF3 there. Run with cf_count > 1.

CF routing is fixed to the `correct_cf` strategy (the counterfactual is always a
correctly-classified training example, routed by prediction only):
    pred=1 (TP or FP) -> k nearest TN  (train pred=0, correct=1)
    pred=0 (TN or FN) -> k nearest TP  (train pred=1, correct=1)

Configs (B1-MCF5 reproduce the standard pipeline exactly; MCF6/MCF7 are new).
B3/B5 are omitted — they carry no CF-derived features, so the standard run
(cv_results_correct_cf) already reproduces them, and they are the two 1024-dim
embedding fits that dominate the baseline cost:
    B1   : disease probability only
    B2   : attribute features only
    B4   : disease probability + attribute features
    M1   : delta features only
    M2   : prob + delta
    M3   : prob + delta + attr
    M4   : prob + delta + emb
    M5   : prob + delta + attr + emb
    M6   : prob + delta + attr + cf_prob
    MCF1 : prob + cf_prob
    MCF2 : prob + cf_prob + attr + cf_attr
    MCF3 : prob + cf_prob + attr + cf_attr + delta
    MCF4 : prob + cf_prob + attr + cf_attr + delta + emb + cf_emb
    MCF5 : prob + cf_prob + attr + cf_attr + delta + emb + cf_emb + delta_emb
    MCF6 : MCF5 + dispersion      <- most complete config + second moment
    MCF7 : MCF3 + dispersion      <- compact control, readable under LR

The MCF5 -> MCF6 and MCF3 -> MCF7 pairs are controlled one-variable comparisons:
identical neighbours, identical blocks, differing only in whether the k CFs are
summarised by the mean alone or by the mean plus the spread.

Usage:
    python c2_cv_pipeline_dispersion.py --disease effusion --cf_count 16 --backbone densenet

Parameters:
    --disease    : Disease name (e.g., 'effusion')
    --cf_count   : Number of counterfactuals (k); must be > 1 for dispersion
    --backbone   : C0 architecture ('densenet', 'resnet50', 'vit', 'medmnist')
    --save_folds : Save per-fold train/test CSVs to disk
    --extended   : Use c2_data_extended.csv (adds GLCM radiomics features)
    --distance   : Distance metric for CF neighbour search ('l1', 'l2', 'cosine')
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
N_FOLDS     = 5
RANDOM_SEED = 42
MODEL_TYPES = ['LR', 'RF', 'MLP']
# B3/B5 are omitted: no CF-derived feature enters any B* design matrix, so their
# AUCs are reproduced exactly by the standard run (cv_results_correct_cf), which
# shares this one's folds, input CSV, models and seed. B3/B5 are the two 1024-dim
# embedding fits, i.e. nearly all of the B* cost. B1/B2/B4 are retained because
# c2_analyse_results.py plots them. Merge B3/B5 from the standard run if needed.
CONFIGS     = ['B1', 'B2', 'B4',
               'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
               'MCF1', 'MCF2', 'MCF3', 'MCF4', 'MCF5',
               'MCF6', 'MCF7']

# Dispersion pairs: (config_with_dispersion, its base without it)
DISPERSION_PAIRS = [('MCF6', 'MCF5'), ('MCF7', 'MCF3')]

BACKBONE_MAP = {
    'densenet':  'C2_custom',
    'resnet50':  'C2_resnet50',
    'vit':       'C2_vit',
    'medmnist':  'C2_medmnist',
}

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_metric(distance):
    """Map a --distance flag to a sklearn NearestNeighbors metric name."""
    if distance == 'l1':
        return 'manhattan'
    elif distance == 'cosine':
        return 'cosine'
    else:
        return 'euclidean'


def add_clinical_ratios(df):
    df = df.copy()

    # Cardiothoracic ratio - most important for cardiomegaly. Normal < 0.5
    df["cardiothoracic_ratio"] = df["Heart_bbox_width"] / (
        df["Left Lung_bbox_width"] + df["Right Lung_bbox_width"]
    )

    # Lung symmetry - asymmetry can indicate effusion or pneumothorax
    df["lung_area_ratio"] = df["Left Lung_area_pixels"] / (df["Right Lung_area_pixels"] + 1e-6)
    df["lung_height_ratio"] = df["Left Lung_bbox_height"] / (df["Right Lung_bbox_height"] + 1e-6)
    df["lung_width_ratio"] = df["Left Lung_bbox_width"] / (df["Right Lung_bbox_width"] + 1e-6)

    # Lung area relative to total - captures hyperinflation/collapse
    total_area = df["Left Lung_area_pixels"] + df["Right Lung_area_pixels"]
    df["left_lung_fraction"] = df["Left Lung_area_pixels"] / (total_area + 1e-6)
    df["right_lung_fraction"] = df["Right Lung_area_pixels"] / (total_area + 1e-6)

    # Mediastinal width relative to lung width - widens in effusion/cardiomegaly
    df["mediastinal_ratio"] = df["Mediastinum_bbox_width"] / (
        df["Left Lung_bbox_width"] + df["Right Lung_bbox_width"] + 1e-6
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CF RETRIEVAL — keeps distances, computes both moments over the k neighbours
# ══════════════════════════════════════════════════════════════════════════════

def compute_cf_with_dispersion(train_df, test_df, cf_count, disease, distance='l1'):
    """
    Retrieve k counterfactuals per query (correct_cf routing) and summarise them
    by BOTH their mean (as the standard pipeline does) and their spread.

    Unlike the standard pipeline this keeps the distances returned by
    kneighbors() rather than discarding them.

    Returns a dict of feature blocks for train and test, each with:
        prob, attr, emb                     — query side (raw / scaled as noted)
        cf_prob, cf_attr, cf_emb            — mean over the k CFs
        delta, delta_emb                    — mean contrast over the k CFs
        disp                                — (n, 4) [cf_prob_std, d1, dmean, drange]
    """
    disease_lower = disease.lower()
    prob_col = f'{disease_lower}_prob'
    pred_col = f'{disease_lower}_pred'
    true_col = f'{disease_lower}_true'

    train_df = add_clinical_ratios(train_df.copy())
    test_df  = add_clinical_ratios(test_df.copy())

    meta_cols = {prob_col, pred_col, true_col, 'correct', 'path', 'patient_id'}
    emb_cols  = [c for c in train_df.columns if c.startswith('emb_')]
    relevant_cols = [c for c in train_df.columns if c not in meta_cols and c not in emb_cols]

    train_clean = train_df.copy()
    test_clean  = test_df.copy()
    train_clean[relevant_cols] = train_clean[relevant_cols].fillna(0)
    test_clean[relevant_cols]  = test_clean[relevant_cols].fillna(0)

    # Scaler fit on train only
    attr_scaler  = StandardScaler()
    train_scaled = attr_scaler.fit_transform(train_clean[relevant_cols].values.astype(float))
    test_scaled  = attr_scaler.transform(test_clean[relevant_cols].values.astype(float))

    metric = _resolve_metric(distance)
    train_preds   = train_clean[pred_col].values.astype(int)
    train_correct = train_clean['correct'].values.astype(int)

    # TP pool: pred=1 correct=1;  TN pool: pred=0 correct=1
    idx_tp = (train_preds == 1) & (train_correct == 1)
    idx_tn = (train_preds == 0) & (train_correct == 1)

    for name, idx in [('TP', idx_tp), ('TN', idx_tn)]:
        if idx.sum() < cf_count:
            raise ValueError(f"{name} pool has {idx.sum()} samples, need >= cf_count={cf_count}")

    train_prob_all = train_clean[prob_col].values.astype(float)
    train_attr_all = train_clean[relevant_cols].values.astype(float)
    train_emb_all  = train_clean[emb_cols].values.astype(float)

    pools = {
        'tp': (np.where(idx_tp)[0],
               NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled[idx_tp])),
        'tn': (np.where(idx_tn)[0],
               NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled[idx_tn])),
    }

    def _summarise(df, query_scaled):
        """Collapse the k CFs of every query into mean blocks + a dispersion block."""
        query_preds = df[pred_col].values.astype(int)
        n = len(df)

        cf_idx  = np.empty((n, cf_count), dtype=np.int64)   # into train_clean
        cf_dist = np.empty((n, cf_count), dtype=np.float64)

        # pred=1 (TP or FP) -> TN pool ;  pred=0 (TN or FN) -> TP pool
        for pred_val, pool_key in [(1, 'tn'), (0, 'tp')]:
            mask = (query_preds == pred_val)
            if not mask.any():
                continue
            pool_global, nn = pools[pool_key]
            dists, idxs = nn.kneighbors(query_scaled[mask])   # distances KEPT
            cf_idx[mask]  = pool_global[idxs]
            cf_dist[mask] = dists

        # ── First moments (identical to the standard pipeline) ────────────
        # Feature spaces follow the original: attr / cf_attr / emb / cf_emb are
        # RAW, while delta is the contrast in STANDARDISED attribute space.
        # (mean_j(query - cf_j) == query - mean_j(cf_j), the query term being
        #  constant over j, so this reproduces compute_diff_vectors exactly.)
        cf_prob_k      = train_prob_all[cf_idx]                     # (n, k)
        cf_prob        = cf_prob_k.mean(axis=1, keepdims=True)
        cf_attr_scaled = train_scaled[cf_idx].mean(axis=1)          # scaled, for delta
        cf_attr_raw    = train_attr_all[cf_idx].mean(axis=1)        # raw, for the blocks
        cf_emb         = train_emb_all[cf_idx].mean(axis=1)
        delta          = query_scaled - cf_attr_scaled
        delta_emb      = df[emb_cols].values.astype(float) - cf_emb

        # ── Second moment + neighbourhood geometry (NEW) ──────────────────
        # kneighbors returns distances already sorted ascending
        cf_prob_std = cf_prob_k.std(axis=1)
        dist_1      = cf_dist[:, 0]
        dist_mean   = cf_dist.mean(axis=1)
        dist_range  = cf_dist[:, -1] - cf_dist[:, 0]
        disp = np.column_stack([cf_prob_std, dist_1, dist_mean, dist_range])

        return {
            'prob':      df[[prob_col]].values.astype(float),
            'attr':      df[relevant_cols].values.astype(float),   # raw, as in original
            'emb':       df[emb_cols].values.astype(float),
            'cf_prob':   cf_prob,
            'cf_attr':   cf_attr_raw,
            'cf_emb':    cf_emb,
            'delta':     delta,
            'delta_emb': delta_emb,
            'disp':      disp,
        }

    return (_summarise(train_clean, train_scaled),
            _summarise(test_clean,  test_scaled),
            train_clean, test_clean)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def assemble_configs(b):
    """Map config name -> design matrix, given the feature blocks `b`."""
    mcf3 = np.hstack([b['prob'], b['cf_prob'], b['attr'], b['cf_attr'], b['delta']])
    mcf5 = np.hstack([b['prob'], b['cf_prob'], b['attr'], b['cf_attr'], b['delta'],
                      b['emb'], b['cf_emb'], b['delta_emb']])
    return {
        'B1':   b['prob'],
        'B2':   b['attr'],
        'B3':   b['emb'],
        'B4':   np.hstack([b['prob'], b['attr']]),
        'B5':   np.hstack([b['prob'], b['emb']]),
        'M1':   b['delta'],
        'M2':   np.hstack([b['prob'], b['delta']]),
        'M3':   np.hstack([b['prob'], b['delta'], b['attr']]),
        'M4':   np.hstack([b['prob'], b['delta'], b['emb']]),
        'M5':   np.hstack([b['prob'], b['delta'], b['attr'], b['emb']]),
        'M6':   np.hstack([b['prob'], b['delta'], b['attr'], b['cf_prob']]),
        'MCF1': np.hstack([b['prob'], b['cf_prob']]),
        'MCF2': np.hstack([b['prob'], b['cf_prob'], b['attr'], b['cf_attr']]),
        'MCF3': mcf3,
        'MCF4': np.hstack([b['prob'], b['cf_prob'], b['attr'], b['cf_attr'], b['delta'],
                           b['emb'], b['cf_emb']]),
        'MCF5': mcf5,
        # ── dispersion variants: same blocks, plus the second moment ──────
        'MCF6': np.hstack([mcf5, b['disp']]),
        'MCF7': np.hstack([mcf3, b['disp']]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, X_test, y_train, model_type):
    """Train one model and return test probabilities + the fitted objects."""

    if model_type == 'LR':
        scaler = StandardScaler()
        model  = LogisticRegression(max_iter=5000, class_weight='balanced',
                                    random_state=RANDOM_SEED)
        model.fit(scaler.fit_transform(X_train), y_train)
        y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]
        return y_prob, {'model': model, 'scaler': scaler}

    elif model_type == 'RF':
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                       random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        return y_prob, model

    elif model_type == 'MLP':
        scaler = StandardScaler()
        model  = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                               early_stopping=True, validation_fraction=0.05,
                               random_state=RANDOM_SEED)
        model.fit(scaler.fit_transform(X_train.astype(np.float32)), y_train)
        y_prob = model.predict_proba(scaler.transform(X_test.astype(np.float32)))[:, 1]
        return y_prob, {'model': model, 'scaler': scaler}

    raise ValueError(f"Unknown model type: {model_type}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CV PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(disease, cf_count, save_fold_data=False, backbone='densenet',
           extended=False, distance='l1', save_models=False):
    """Run the 5-fold CV pipeline with CF dispersion features (MCF6/MCF7)."""

    if cf_count < 2:
        raise ValueError(
            f"cf_count={cf_count}: dispersion is identically zero with a single "
            "neighbour (MCF6==MCF5, MCF7==MCF3). Use cf_count > 1."
        )

    print(f"\n{'='*70}")
    print(f"  {N_FOLDS}-FOLD CV (CF DISPERSION): {disease.upper()} | k={cf_count} "
          f"| BACKBONE={backbone} | DISTANCE={distance}")
    print(f"{'='*70}\n")

    results_base = os.path.join(RESULTS_DIR, '')
    data_subdir  = BACKBONE_MAP[backbone]
    cv_subdir    = 'cv_results_extended_dispersion' if extended else 'cv_results_dispersion'

    cv_dir = os.path.join(results_base, f'{data_subdir}_corrected/{disease}/{cv_subdir}/cf_{cf_count}')
    cv_dir = os.path.join(cv_dir, f'distance_{distance}') if distance != 'l1' else cv_dir

    fold_data_dir = os.path.join(cv_dir, 'fold_data')
    os.makedirs(cv_dir, exist_ok=True)
    os.makedirs(fold_data_dir, exist_ok=True)

    data_csv = 'c2_data_extended.csv' if extended else 'c2_data.csv'
    full_df = pd.read_csv(os.path.join(results_base, f'{data_subdir}/{disease}/{data_csv}'))
    full_df.rename(columns={'prob': f'{disease}_prob',
                            'pred': f'{disease}_pred',
                            'true': f'{disease}_true'}, inplace=True)

    print(f"Backbone:   {backbone}  ->  {data_subdir}/{data_csv}")
    print(f"CV Subdir:  {cv_subdir}")
    print(f"CF Count:   {cf_count}")
    print(f"CF Routing: correct_cf (pred=1 -> TN pool, pred=0 -> TP pool)")
    print(f"Dispersion: cf_prob_std, dist_1, dist_mean, dist_range (4 scalars)")
    print(f"            MCF6 = MCF5 + disp   |   MCF7 = MCF3 + disp\n")
    print(f"Full dataset: {len(full_df):,} samples")
    print(f"  Correct:   {(full_df['correct']==1).sum():,}")
    print(f"  Incorrect: {(full_df['correct']==0).sum():,}\n")

    y = full_df['correct'].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    cv_results = {mt: {cfg: [] for cfg in CONFIGS} for mt in MODEL_TYPES}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, y)):

        print(f"\n{'-'*70}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'-'*70}")

        fold_train = full_df.iloc[train_idx].reset_index(drop=True)
        fold_test  = full_df.iloc[test_idx].reset_index(drop=True)
        print(f"  Train: {len(fold_train):,}  |  Test: {len(fold_test):,}")

        print(f"  Retrieving {cf_count} CFs and computing dispersion...")
        b_tr, b_te, train_clean, test_clean = compute_cf_with_dispersion(
            train_df=fold_train, test_df=fold_test,
            cf_count=cf_count, disease=disease, distance=distance,
        )

        # Sanity: dispersion must be non-degenerate for k>1
        if fold_idx == 0:
            d = b_tr['disp']
            print(f"    dispersion block {d.shape}: "
                  f"cf_prob_std mean={d[:,0].mean():.4f}  "
                  f"dist_1 mean={d[:,1].mean():.3f}  "
                  f"dist_range mean={d[:,3].mean():.3f}")
            if np.allclose(d[:, 0], 0):
                raise RuntimeError("cf_prob_std is all-zero — dispersion is degenerate")

        feats_tr = assemble_configs(b_tr)
        feats_te = assemble_configs(b_te)

        y_train = train_clean['correct'].values
        y_test  = test_clean['correct'].values

        if save_fold_data:
            for name, clean, b in [('train', train_clean, b_tr), ('test', test_clean, b_te)]:
                out = clean.copy()
                out['cf_prob']     = b['cf_prob'][:, 0]
                out['cf_prob_std'] = b['disp'][:, 0]
                out['dist_1']      = b['disp'][:, 1]
                out['dist_mean']   = b['disp'][:, 2]
                out['dist_range']  = b['disp'][:, 3]
                out.to_csv(os.path.join(fold_data_dir, f'fold_{fold_idx}_{name}.csv'), index=False)

        fold_pred_df = test_clean.copy()

        for model_type in MODEL_TYPES:
            print(f"\n  {model_type}:")
            if save_models:
                model_dir = os.path.join(cv_dir, 'models', model_type)
                os.makedirs(model_dir, exist_ok=True)

            fold_aucs = {}
            for config in CONFIGS:
                y_prob, fitted = train_model(feats_tr[config], feats_te[config],
                                             y_train, model_type)
                auc = float(roc_auc_score(y_test, y_prob))
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                fold_aucs[config] = auc

                if save_models:
                    joblib.dump(fitted, os.path.join(
                        model_dir, f"{model_type}_{config}_fold{fold_idx}.pkl"))

                cv_results[model_type][config].append({
                    'auc': auc, 'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
                    'y_prob': y_prob.tolist(), 'y_true': y_test.tolist(),
                })
                fold_pred_df[f"{model_type}_{config}_prob"] = y_prob
                print(f"    {config}: AUC = {auc:.4f}")

            # The controlled comparison this script exists for
            for disp_cfg, base_cfg in DISPERSION_PAIRS:
                gain = fold_aucs[disp_cfg] - fold_aucs[base_cfg]
                print(f"    -> {disp_cfg} - {base_cfg} = {gain:+.4f}  (dispersion effect)")

        pred_csv_path = os.path.join(cv_dir, f'fold_{fold_idx}_predictions.csv')
        fold_pred_df.to_csv(pred_csv_path, index=False)
        print(f"\nFold {fold_idx} predictions saved to: {pred_csv_path}")

    # ── AGGREGATE ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATING RESULTS")
    print(f"{'='*70}\n")

    summary_rows = []
    for model_type in MODEL_TYPES:
        for config in CONFIGS:
            aucs = [f['auc'] for f in cv_results[model_type][config]]
            if not aucs:
                continue
            summary_rows.append({
                'model': model_type, 'config': config,
                'mean_auc': np.mean(aucs), 'std_auc': np.std(aucs),
                'min_auc': np.min(aucs), 'max_auc': np.max(aucs),
                'fold_aucs': ','.join(f'{a:.4f}' for a in aucs),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(cv_dir, 'cv_summary.csv'), index=False)

    with open(os.path.join(cv_dir, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    print(summary_df.to_string(index=False))

    # ── The headline: paired dispersion effect, per model ─────────────────
    print(f"\n{'='*70}")
    print("DISPERSION EFFECT (paired, same folds)")
    print(f"{'='*70}")
    disp_rows = []
    for model_type in MODEL_TYPES:
        for disp_cfg, base_cfg in DISPERSION_PAIRS:
            a = np.array([f['auc'] for f in cv_results[model_type][disp_cfg]])
            b = np.array([f['auc'] for f in cv_results[model_type][base_cfg]])
            d = a - b
            disp_rows.append({
                'model': model_type,
                'comparison': f'{disp_cfg} - {base_cfg}',
                'base_auc': b.mean(), 'disp_auc': a.mean(),
                'gain': d.mean(), 'gain_std': d.std(),
                'folds_improved': int((d > 0).sum()), 'n_folds': len(d),
            })
    disp_df = pd.DataFrame(disp_rows)
    disp_df.to_csv(os.path.join(cv_dir, 'dispersion_effect.csv'), index=False)
    print(disp_df.to_string(index=False))

    print(f"\nResults saved to: {cv_dir}")
    return cv_results, summary_df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    if any('jupyter' in arg or 'ipykernel' in arg for arg in sys.argv):
        run_cv(disease='effusion', cf_count=16, backbone='densenet')
    else:
        parser = argparse.ArgumentParser(
            description='Run C2 cross-validation with CF dispersion features')
        parser.add_argument('--disease',  type=str, default='effusion')
        parser.add_argument('--cf_count', type=int, default=16,
                            help='Number of counterfactuals (k); must be > 1')
        parser.add_argument('--backbone', type=str, default='densenet',
                            choices=['densenet', 'resnet50', 'vit', 'medmnist'])
        parser.add_argument('--save_folds', action='store_true')
        parser.add_argument('--extended',   action='store_true',
                            help='Use c2_data_extended.csv (adds GLCM radiomics features)')
        parser.add_argument('--distance', type=str, default='l1',
                            choices=['l1', 'l2', 'cosine'],
                            help='Distance metric for CF neighbour search')
        parser.add_argument('--save_models', action='store_true',
                            help='Persist per-fold fitted models to models/ '
                                 '(large .pkl dumps; off by default to save disk quota)')
        args = parser.parse_args()

        run_cv(
            disease=args.disease,
            cf_count=args.cf_count,
            save_fold_data=args.save_folds,
            backbone=args.backbone,
            extended=args.extended,
            distance=args.distance,
            save_models=args.save_models,
        )
