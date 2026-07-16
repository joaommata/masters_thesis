"""
c2_cv_pipeline_cf_ensemble.py

5-fold cross-validation pipeline for C2 quality control models, using a
per-counterfactual ENSEMBLE instead of averaging the k counterfactuals into a
single attribute vector.

Difference from c2_cv_pipeline_new_split.py
-------------------------------------------
The original pipeline retrieves k counterfactual neighbours per sample and
collapses them with a mean (delta vector, cf_prob, cf_attr_*, cf_emb_* are all
averaged over the k CFs), producing one feature row per sample.

Here, each of the k counterfactuals is kept separate:
    * Rank j (j = 0..k-1) yields its own feature row per sample, in which the
      query-side features (prob, attr, emb) are repeated and only the
      CF-derived features (delta_*, cf_prob, cf_attr_*, cf_emb_*, delta_emb)
      vary.
    * ONE model per (model_type, config) is trained on N rows, each sample paired
      with its NEAREST counterfactual (rank 0). The training set is therefore the
      same size as the original pipeline's and does not grow with k.
    * At test time that model scores all k expansions of each test sample and the
      k probabilities are averaged into one score per sample.

So the averaging moves from the feature space (before the model) to the score
space (after the model): pure test-time ensembling over counterfactuals rather
than feature-averaging. Holding the training set fixed across k means that
comparing k=1 / 3 / 16 varies exactly one thing — how many counterfactuals are
averaged at prediction time — so `ensemble_gain` isolates that effect alone.

For the baseline configs no CF-derived features enter the design matrix, so the k
test expansions of a sample are identical and the scheme reduces to the original.
(This makes them a useful invariant: for any B* config, mean_auc and
mean_single_cf_auc must agree exactly.)

Because of that invariance the B* AUCs are reproduced exactly by the non-ensemble
pipeline, which shares this one's folds, input CSV, models and seed. Only B1, B2
and B4 are therefore trained here -- enough to keep the invariant and to satisfy
c2_analyse_results.py. B3 and B5 (the 1024-dim embedding baselines, and nearly all
of the B* compute) are skipped; take them from the non-ensemble run's
cv_detailed.json if a plot needs them.

CF routing is fixed to the `correct_cf` strategy (counterfactual is always a
correctly-classified training example, routed by prediction only):
    pred=1 (TP or FP) -> nearest TN  (train pred=0, correct=1)
    pred=0 (TN or FN) -> nearest TP  (train pred=1, correct=1)

Configs:
    Baseline (B1, B2, B4 -- see note above on the omitted B3/B5):
        B1 : disease probability only
        B2 : attribute features only
        B4 : disease probability + attribute features

    Multi-modal (M1-M6):
        M1 : delta features only
        M2 : disease probability + delta features
        M3 : disease probability + delta + attribute features
        M4 : disease probability + delta + embedding features
        M5 : disease probability + delta + attribute + embedding features
        M6 : disease probability + delta + attribute + CF probability

    CF-enriched (MCF1-MCF5), progressive accumulation:
        MCF1 : prob + cf_prob
        MCF2 : prob + cf_prob + attr + cf_attr
        MCF3 : prob + cf_prob + attr + cf_attr + delta_attr
        MCF4 : prob + cf_prob + attr + cf_attr + delta_attr + emb + cf_emb
        MCF5 : prob + cf_prob + attr + cf_attr + delta_attr + emb + cf_emb + delta_emb

Usage:
    python c2_cv_pipeline_cf_ensemble.py --disease effusion --cf_count 16 --backbone resnet50

Parameters:
    --disease    : Disease name (e.g., 'effusion')
    --cf_count   : Number of counterfactuals (k) averaged at test time
    --backbone   : C0 architecture ('densenet', 'resnet50', 'vit', 'medmnist')
    --save_folds : Save individual fold train/test CSVs (rank-0 expansion) to disk
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
from sklearn.model_selection import StratifiedGroupKFold

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
N_FOLDS     = 5
RANDOM_SEED = 42
MODEL_TYPES = ['LR', 'RF', 'MLP']

# ── Entropy / high-capacity-MLP switches (set from CLI in run_cv) ──────────────
# When ENTROPY is on, the raw C0 probability feature (and its CF counterpart) is
# replaced everywhere by its binary entropy H(p) = -p logp - (1-p) log(1-p),
# matching c2_cv_pipeline_new_split_entropy.py. This also flips the MLP to a
# larger, better-regularised network to amplify the mid-confidence signal.
ENTROPY = False

def _binary_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


# B3/B5 are omitted: no CF-derived feature enters any B* design matrix, so their
# k test expansions are identical and their AUCs reproduce the non-ensemble run
# (cv_results_correct_cf) exactly. B1/B2/B4 are retained because
# c2_analyse_results.py plots them and B1 anchors the ensemble invariant
# (mean_auc == mean_single_cf_auc). B3/B5 are the two 1024-dim embedding fits,
# i.e. nearly all of the B* cost; merge them from cv_detailed.json if needed.
CONFIGS     = ['B1', 'B2', 'B4',
               'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
               'MCF1', 'MCF2', 'MCF3', 'MCF4', 'MCF5']

BACKBONE_MAP = {
    'densenet':  'C2_custom',
    'resnet50':  'C2_resnet50',
    'vit':       'C2_vit',
    'medmnist':  'C2_medmnist',
}

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (mirrors c2_prepare_data_simulated_cf.add_clinical_ratios)
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
# TRAIN-SPLIT BALANCING
# ══════════════════════════════════════════════════════════════════════════════

def balance_quadrants(train_df, disease, ratio=1.0, seed=RANDOM_SEED, strict=False):
    """
    Subsample a training fold so C0's four outcome quadrants are balanced.

    Quadrants are (pred x correct): TP, FP, TN, FN. Every error row (FP, FN) is
    kept by default; the correct quadrants (TP, TN) are subsampled down to
    `ratio` * n_err, where n_err = min(n_FP, n_FN).

    `ratio` is therefore "how many TPs (and TNs) to keep per FP", NOT the
    resulting corrects-per-error ratio: the surplus of the larger error quadrant
    inflates the error side without entering the target. With strict=True the
    error quadrants are also capped at n_err, giving an exact 1:1:1:1 pool at
    ratio=1.0 at the cost of discarding real errors.

    Quadrants smaller than their target are kept whole — nothing is ever
    upsampled or duplicated.

    Applied to the training partition only; test folds keep natural prevalence.
    """
    pred_col = f'{disease}_pred'
    rng = np.random.RandomState(seed)

    preds   = train_df[pred_col].values.astype(int)
    correct = train_df['correct'].values.astype(int)

    quadrants = {
        'TP': (preds == 1) & (correct == 1),
        'FP': (preds == 1) & (correct == 0),
        'TN': (preds == 0) & (correct == 1),
        'FN': (preds == 0) & (correct == 0),
    }
    n_before = {q: int(m.sum()) for q, m in quadrants.items()}

    # Errors are the scarce resource: they set the budget.
    n_err = min(n_before['FP'], n_before['FN'])
    if n_err == 0:
        print("    [balance] WARNING: empty error quadrant — skipping balancing for this fold")
        return train_df

    target_correct = int(round(ratio * n_err))

    keep_idx = []
    for q, mask in quadrants.items():
        idx = np.where(mask)[0]
        target = n_err if q in ('FP', 'FN') else target_correct
        if q in ('FP', 'FN') and not strict:
            keep_idx.append(idx)              # keep every error
            continue
        if len(idx) > target:
            idx = rng.choice(idx, size=target, replace=False)
        keep_idx.append(idx)

    keep_idx = np.sort(np.concatenate(keep_idx))
    balanced = train_df.iloc[keep_idx].reset_index(drop=True)

    b_preds   = balanced[pred_col].values.astype(int)
    b_correct = balanced['correct'].values.astype(int)
    n_after = {
        'TP': int(((b_preds == 1) & (b_correct == 1)).sum()),
        'FP': int(((b_preds == 1) & (b_correct == 0)).sum()),
        'TN': int(((b_preds == 0) & (b_correct == 1)).sum()),
        'FN': int(((b_preds == 0) & (b_correct == 0)).sum()),
    }
    n_corr = n_after['TP'] + n_after['TN']
    n_errs = n_after['FP'] + n_after['FN']
    print(f"    [balance] ratio={ratio:g}{' strict' if strict else ''}  "
          f"{len(train_df):,} -> {len(balanced):,} samples")
    print(f"    [balance] before: {n_before}")
    print(f"    [balance] after:  {n_after}  "
          f"(corrects:errors = {n_corr/max(n_errs,1):.2f}:1)")

    return balanced


# ══════════════════════════════════════════════════════════════════════════════
# RANKED CF RETRIEVAL (correct_cf routing, k CFs kept separate)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ranked_cf_correct_cf(train_df, test_df, cf_count, disease, distance='l1'):
    """
    Retrieve the k nearest counterfactuals per query WITHOUT averaging them.

    Routing (correct_cf strategy, prediction-only so it is not leaky at test time):
        pred=1 (TP or FP) -> k nearest TN  (train pred=0, correct=1)
        pred=0 (TN or FN) -> k nearest TP  (train pred=1, correct=1)

    Returns
    -------
    train_clean, test_clean : DataFrame
        Query frames with clinical ratios added and attribute NaNs filled.
    train_cf_idx, test_cf_idx : (n, k) int array
        Row positions into `train_clean` of the j-th nearest CF for each query.
    train_scaled, test_scaled : (n, d) float array
        Standardised attribute matrices for the queries (scaler fit on train).
    relevant_cols, emb_cols : list[str]
    """
    disease_lower = disease.lower()
    disease_prob_col = f'{disease_lower}_prob'
    disease_pred_col = f'{disease_lower}_pred'
    disease_true_col = f'{disease_lower}_true'

    train_df = add_clinical_ratios(train_df.copy())
    test_df  = add_clinical_ratios(test_df.copy())

    meta_cols = {disease_prob_col, disease_pred_col, disease_true_col,
                 'correct', 'path', 'cam_path', 'patient_id'}
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
    train_preds   = train_clean[disease_pred_col].values.astype(int)
    train_correct = train_clean['correct'].values.astype(int)

    # TP pool: pred=1 correct=1;  TN pool: pred=0 correct=1
    idx_tp = (train_preds == 1) & (train_correct == 1)
    idx_tn = (train_preds == 0) & (train_correct == 1)

    pool_global_idx = {'tp': np.where(idx_tp)[0], 'tn': np.where(idx_tn)[0]}

    for key, pool in pool_global_idx.items():
        if len(pool) < cf_count:
            raise ValueError(
                f"{key.upper()} pool has only {len(pool)} samples, need >= cf_count={cf_count}"
            )

    nn_tp = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled[idx_tp])
    nn_tn = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled[idx_tn])

    def _rank_cfs(df, query_scaled):
        """Return an (n, cf_count) array of train_clean row positions, rank-ordered."""
        query_preds = df[disease_pred_col].values.astype(int)
        cf_idx = np.empty((len(df), cf_count), dtype=np.int64)

        # pred=1 (TP or FP) -> TN pool
        mask = (query_preds == 1)
        if mask.any():
            _, idxs = nn_tn.kneighbors(query_scaled[mask])
            cf_idx[mask] = pool_global_idx['tn'][idxs]

        # pred=0 (TN or FN) -> TP pool
        mask = (query_preds == 0)
        if mask.any():
            _, idxs = nn_tp.kneighbors(query_scaled[mask])
            cf_idx[mask] = pool_global_idx['tp'][idxs]

        return cf_idx

    train_cf_idx = _rank_cfs(train_clean, train_scaled)
    test_cf_idx  = _rank_cfs(test_clean,  test_scaled)

    return (train_clean, test_clean, train_cf_idx, test_cf_idx,
            train_scaled, test_scaled, relevant_cols, emb_cols)


# ══════════════════════════════════════════════════════════════════════════════
# PER-RANK FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_rank_blocks(query_clean, query_scaled, train_clean, train_scaled,
                      cf_idx, rank, disease, relevant_cols, emb_cols):
    """
    Build the raw feature blocks for a single CF rank.

    Query-side blocks (prob, attr, emb) do not depend on `rank`; the CF-side
    blocks (delta, cf_prob, cf_attr, cf_emb, delta_emb) are taken from the
    rank-th nearest counterfactual of each query.

    Feature spaces follow the original pipeline exactly: attr / cf_attr / emb /
    cf_emb are RAW values, while delta is computed in STANDARDISED attribute
    space (as compute_diff_vectors does) and delta_emb in raw embedding space.
    """
    disease_lower = disease.lower()
    j = cf_idx[:, rank]                                   # train row positions of rank-j CF

    prob    = query_clean[[f'{disease_lower}_prob']].values.astype(float)
    attr    = query_clean[relevant_cols].values.astype(float)
    emb     = query_clean[emb_cols].values.astype(float)

    cf_prob = train_clean[f'{disease_lower}_prob'].values[j].reshape(-1, 1)
    cf_attr = train_clean[relevant_cols].values.astype(float)[j]
    cf_emb  = train_clean[emb_cols].values.astype(float)[j]

    # Entropy mode: swap the C0 probability feature (query + CF) for its binary
    # entropy, so every config that uses b['prob'] / b['cf_prob'] gets it.
    if ENTROPY:
        prob    = _binary_entropy(prob)
        cf_prob = _binary_entropy(cf_prob)

    delta     = query_scaled - train_scaled[j]            # scaled space, as in the original
    delta_emb = emb - cf_emb

    return {
        'prob': prob, 'attr': attr, 'emb': emb,
        'cf_prob': cf_prob, 'cf_attr': cf_attr, 'cf_emb': cf_emb,
        'delta': delta, 'delta_emb': delta_emb,
    }


def assemble_configs(b):
    """Map config name -> design matrix, given the feature blocks `b` for one rank."""
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
        'MCF3': np.hstack([b['prob'], b['cf_prob'], b['attr'], b['cf_attr'], b['delta']]),
        'MCF4': np.hstack([b['prob'], b['cf_prob'], b['attr'], b['cf_attr'], b['delta'],
                           b['emb'], b['cf_emb']]),
        'MCF5': np.hstack([b['prob'], b['cf_prob'], b['attr'], b['cf_attr'], b['delta'],
                           b['emb'], b['cf_emb'], b['delta_emb']]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, X_test_ranks, y_train, model_type):
    """
    Train ONE model on the nearest-CF training rows, then score every CF
    expansion of the test set with it.

    Parameters
    ----------
    X_train      : (N, d) training design matrix, built from the rank-0 CF only
    X_test_ranks : list of k arrays, each (n_test, d) — the rank-j test expansion
    y_train      : (N,) labels

    Returns
    -------
    member_probs : (k, n_test) per-rank test probabilities
    fitted       : the fitted model (plus scaler where applicable)
    """

    if model_type == 'LR':
        scaler = StandardScaler()
        model  = LogisticRegression(max_iter=5000, class_weight='balanced',
                                    random_state=RANDOM_SEED)
        model.fit(scaler.fit_transform(X_train), y_train)
        member_probs = np.stack([
            model.predict_proba(scaler.transform(X_te))[:, 1] for X_te in X_test_ranks
        ])
        return member_probs, {'model': model, 'scaler': scaler}

    elif model_type == 'RF':
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                       random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        member_probs = np.stack([
            model.predict_proba(X_te)[:, 1] for X_te in X_test_ranks
        ])
        return member_probs, model

    elif model_type == 'MLP':
        scaler = StandardScaler()
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.05,
            random_state=RANDOM_SEED,
        )
        model.fit(scaler.fit_transform(X_train.astype(np.float32)), y_train)
        member_probs = np.stack([
            model.predict_proba(scaler.transform(X_te.astype(np.float32)))[:, 1]
            for X_te in X_test_ranks
        ])
        return member_probs, {'model': model, 'scaler': scaler}

    raise ValueError(f"Unknown model type: {model_type}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CV PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(disease, cf_count, save_fold_data=False, backbone='densenet',
           extended=False, distance='l1', entropy=False,
           models=None, configs=None, balance=False, balance_ratio=1.0,
           strict_balance=False):
    """
    Run the 5-fold CV pipeline with test-time averaging over k counterfactuals.

    One model per (model_type, config) is trained on N rows (each sample paired
    with its nearest CF); each test sample is then scored k times (once per CF)
    and the k probabilities are averaged.

    Parameters
    ----------
    disease        : str  - Disease name (e.g., 'effusion')
    cf_count       : int  - Number of counterfactual neighbours (k)
    save_fold_data : bool - Save per-fold train/test CSVs (rank-0 expansion)
    backbone       : str  - C0 architecture ('densenet', 'resnet50', 'vit', 'medmnist')
    extended       : bool - Use c2_data_extended.csv (adds GLCM radiomics features)
    distance       : str  - Distance metric for CF neighbour search
    balance        : bool - Subsample each TRAIN fold to balance C0's TP/FP/TN/FN
                            quadrants. Test folds keep natural prevalence, so AUCs
                            stay comparable to unbalanced runs. NOTE: this also
                            shrinks the TP/TN pools the CFs are drawn from.
    balance_ratio  : float - TPs (and TNs) kept per FP; see balance_quadrants
    strict_balance : bool - Also cap FP/FN at min(n_FP, n_FN) for an exact 1:1:1:1
    """

    # big_mlp defaults to following entropy (the entropy+ensemble run we want
    # is the one that uses the high-capacity net), but can be forced independently.
    global ENTROPY, MODEL_TYPES, CONFIGS
    ENTROPY = entropy

    # Optional restriction to a subset of models / configs (skip wasted compute).
    if models is not None:
        MODEL_TYPES = list(models)
    if configs is not None:
        CONFIGS = list(configs)

    print(f"\n{'='*70}")
    print(f"  {N_FOLDS}-FOLD CV (CF SCORE-AVERAGING): {disease.upper()} | k={cf_count} "
          f"| BACKBONE={backbone} | DISTANCE={distance}")
    print(f"  ENTROPY={ENTROPY}")
    print(f"  MODELS={MODEL_TYPES}  CONFIGS={CONFIGS}")
    print(f"{'='*70}\n")

    # ── Setup paths ───────────────────────────────────────────────────────
    results_base = os.path.join(RESULTS_DIR, '')
    data_subdir  = BACKBONE_MAP[backbone]
    cv_subdir    = 'cv_results_extended_cf_ensemble' if extended else 'cv_results_cf_ensemble'

    cv_dir = os.path.join(results_base, f'{data_subdir}_corrected/{disease}/{cv_subdir}/cf_{cf_count}')
    cv_dir = os.path.join(cv_dir, f'distance_{distance}') if distance != 'l1' else cv_dir
    # Keep entropy (+ big-MLP) runs in a sibling dir so they don't clobber the
    # existing raw-prob ensemble results.
    if ENTROPY:
        cv_dir = os.path.join(cv_dir, 'TEST_entropy')
    # Balanced runs land in their own subtree so they never overwrite existing results.
    if balance:
        cv_dir = os.path.join(
            cv_dir, f'balanced_r{balance_ratio:g}{"_strict" if strict_balance else ""}')

    fold_data_dir = os.path.join(cv_dir, 'fold_data')
    os.makedirs(cv_dir, exist_ok=True)
    os.makedirs(fold_data_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    data_csv = 'c2_data_extended.csv' if extended else 'c2_data.csv'
    full_df = pd.read_csv(os.path.join(results_base, f'{data_subdir}/{disease}/{data_csv}'))

    full_df.rename(columns={
        'prob': f'{disease}_prob',
        'pred': f'{disease}_pred',
        'true': f'{disease}_true',
    }, inplace=True)

    print(f"Backbone:   {backbone}  ->  {data_subdir}/{data_csv}")
    print(f"CV Subdir:  {cv_subdir}")
    print(f"CF Count:   {cf_count}  (train on nearest CF; test scores averaged over k)")
    print(f"CF Routing: correct_cf (pred=1 -> TN pool, pred=0 -> TP pool)")
    print(f"Save Fold Data: {save_fold_data}")
    print(f"Balance train folds: {balance}"
          + (f" (ratio={balance_ratio:g}{', strict' if strict_balance else ''})" if balance else "")
          + "\n")
    print(f"Full dataset: {len(full_df):,} samples")
    print(f"  Correct:   {(full_df['correct']==1).sum():,}")
    print(f"  Incorrect: {(full_df['correct']==0).sum():,}\n")

    y = full_df['correct'].values
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    groups = full_df['patient_id'].values

    cv_results = {mt: {cfg: [] for cfg in CONFIGS} for mt in MODEL_TYPES}

    # ── FOLD LOOP ─────────────────────────────────────────────────────────
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, y, groups=groups)):

        print(f"\n{'-'*70}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'-'*70}")

        fold_train = full_df.iloc[train_idx].reset_index(drop=True)
        fold_test  = full_df.iloc[test_idx].reset_index(drop=True)
        print(f"  Train: {len(fold_train):,}  |  Test: {len(fold_test):,}")
        overlap = set(fold_train['patient_id']) & set(fold_test['patient_id'])
        print(f"  Patient overlap train/test: {len(overlap)} (should be 0)")

        # ── Balance the TRAIN partition only (test keeps natural prevalence) ──
        if balance:
            fold_train = balance_quadrants(
                fold_train, disease=disease, ratio=balance_ratio,
                seed=RANDOM_SEED + fold_idx, strict=strict_balance,
            )

        # ── Retrieve k ranked counterfactuals (no averaging) ──────────────
        print(f"  Retrieving {cf_count} ranked counterfactuals...")
        (train_clean, test_clean, train_cf_idx, test_cf_idx,
         train_scaled, test_scaled, relevant_cols, emb_cols) = compute_ranked_cf_correct_cf(
            train_df=fold_train, test_df=fold_test,
            cf_count=cf_count, disease=disease, distance=distance,
        )

        y_train = train_clean['correct'].values
        y_test  = test_clean['correct'].values

        # ── Design matrices ───────────────────────────────────────────────
        # Training uses the nearest CF only (rank 0), so its size is independent
        # of k. The test set is expanded once per rank; those k score vectors are
        # averaged after prediction.
        b_tr = build_rank_blocks(train_clean, train_scaled, train_clean, train_scaled,
                                 train_cf_idx, 0, disease, relevant_cols, emb_cols)
        rank_train = [assemble_configs(b_tr)]

        rank_test = []
        for rank in range(cf_count):
            b_te = build_rank_blocks(test_clean, test_scaled, train_clean, train_scaled,
                                     test_cf_idx, rank, disease, relevant_cols, emb_cols)
            rank_test.append(assemble_configs(b_te))

        if save_fold_data:
            # rank-0 expansion, for inspection
            _dump_rank0(train_clean, test_clean, train_cf_idx, test_cf_idx,
                        train_clean, disease, fold_data_dir, fold_idx)

        fold_pred_df = test_clean.copy()

        # ── One model per (model_type, config), scored over k CF expansions ──
        for model_type in MODEL_TYPES:
            print(f"\n  {model_type}:")

            model_dir = os.path.join(cv_dir, 'models', model_type)
            os.makedirs(model_dir, exist_ok=True)

            for config in CONFIGS:
                # Train on the nearest-CF rows only (N rows, independent of k)
                X_train_r0   = rank_train[0][config]
                X_test_ranks = [rank_test[r][config] for r in range(cf_count)]

                member_probs, fitted = train_model(X_train_r0, X_test_ranks,
                                                   y_train, model_type)

                # Ensemble = mean of the k per-CF test probabilities
                y_prob = member_probs.mean(axis=0)
                auc = float(roc_auc_score(y_test, y_prob))
                fpr, tpr, _ = roc_curve(y_test, y_prob)

                joblib.dump(fitted, os.path.join(
                    model_dir, f"{model_type}_{config}_fold{fold_idx}.pkl"))

                member_aucs = [float(roc_auc_score(y_test, member_probs[r]))
                               for r in range(cf_count)]

                cv_results[model_type][config].append({
                    'auc': auc,
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'y_prob': y_prob.tolist(),
                    'y_true': y_test.tolist(),
                    'member_aucs': member_aucs,
                })
                print(f"    {config}: ensemble AUC = {auc:.4f}  "
                      f"(single-CF mean {np.mean(member_aucs):.4f}, "
                      f"best {np.max(member_aucs):.4f})")

                fold_pred_df[f"{model_type}_{config}_prob"] = y_prob

        pred_csv_path = os.path.join(cv_dir, f'fold_{fold_idx}_predictions.csv')
        fold_pred_df.to_csv(pred_csv_path, index=False)
        print(f"\nFold {fold_idx} predictions saved to: {pred_csv_path}")

    # ── AGGREGATE RESULTS ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATING RESULTS")
    print(f"{'='*70}\n")

    summary_rows = []
    for model_type in MODEL_TYPES:
        for config in CONFIGS:
            folds = cv_results[model_type][config]
            if not folds:
                continue
            aucs = [f['auc'] for f in folds]
            # AUC if a single (rank-j) CF were used instead of averaging the k scores
            single_cf_means = [np.mean(f['member_aucs']) for f in folds]
            summary_rows.append({
                'model':             model_type,
                'config':            config,
                'mean_auc':          np.mean(aucs),
                'std_auc':           np.std(aucs),
                'min_auc':           np.min(aucs),
                'max_auc':           np.max(aucs),
                'mean_single_cf_auc': np.mean(single_cf_means),
                'ensemble_gain':     np.mean(aucs) - np.mean(single_cf_means),
                'fold_aucs':         ','.join(f'{a:.4f}' for a in aucs),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(cv_dir, 'cv_summary.csv'), index=False)

    with open(os.path.join(cv_dir, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {cv_dir}")

    return cv_results, summary_df


def _dump_rank0(train_clean, test_clean, train_cf_idx, test_cf_idx,
                train_lookup, disease, fold_data_dir, fold_idx):
    """Write the rank-0 (nearest) CF expansion of each split, for inspection."""
    for name, df, cf_idx in [('train', train_clean, train_cf_idx),
                             ('test',  test_clean,  test_cf_idx)]:
        out = df.copy()
        j = cf_idx[:, 0]
        out['cf_path'] = train_lookup['path'].values[j]
        out['cf_prob'] = train_lookup[f'{disease}_prob'].values[j]
        out['delta_prob'] = out[f'{disease}_prob'].values - out['cf_prob'].values
        out.to_csv(os.path.join(fold_data_dir, f'fold_{fold_idx}_{name}.csv'), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    if any('jupyter' in arg or 'ipykernel' in arg for arg in sys.argv):
        run_cv(disease='effusion', cf_count=16, backbone='densenet')
    else:
        parser = argparse.ArgumentParser(
            description='Run C2 cross-validation, averaging scores over k counterfactuals')
        parser.add_argument('--disease',  type=str, default='effusion')
        parser.add_argument('--cf_count', type=int, default=16,
                            help='Number of counterfactuals (k) averaged at test time')
        parser.add_argument('--backbone', type=str, default='densenet',
                            choices=['densenet', 'resnet50', 'vit', 'medmnist'])
        parser.add_argument('--save_folds', action='store_true')
        parser.add_argument('--extended',   action='store_true',
                            help='Use c2_data_extended.csv (adds GLCM radiomics features)')
        parser.add_argument('--distance', type=str, default='l1',
                            choices=['l1', 'l2', 'cosine'],
                            help='Distance metric for CF neighbour search')
        parser.add_argument('--entropy', action='store_true',
                            help='Replace the C0 probability feature with its binary entropy; results are written to a TEST_entropy/ subdirectory.')
        parser.add_argument('--models', nargs='+', default=None,
                            choices=['LR', 'RF', 'MLP'],
                            help='Restrict to a subset of model types (default: all)')
        parser.add_argument('--configs', nargs='+', default=None,
                            help='Restrict to a subset of configs, e.g. MCF4 MCF5 '
                                 '(default: all)')
        parser.add_argument('--balance', action='store_true',
                            help='Subsample each TRAIN fold to balance C0 TP/FP/TN/FN '
                                 'quadrants (test folds keep natural prevalence)')
        parser.add_argument('--balance_ratio', type=float, default=1.0,
                            help='TPs (and TNs) kept per FP. NOT the resulting '
                                 'corrects-per-error ratio — the larger error quadrant '
                                 'is kept whole unless --strict_balance. Single value; '
                                 'run once per ratio to sweep.')
        parser.add_argument('--strict_balance', action='store_true',
                            help='Also cap FP/FN at min(n_FP, n_FN) for an exact 1:1:1:1 '
                                 'pool (discards real errors)')
        args = parser.parse_args()

        run_cv(
            disease=args.disease,
            cf_count=args.cf_count,
            save_fold_data=args.save_folds,
            backbone=args.backbone,
            extended=args.extended,
            distance=args.distance,
            entropy=args.entropy,
            models=args.models,
            configs=args.configs,
            balance=args.balance,
            balance_ratio=args.balance_ratio,
            strict_balance=args.strict_balance,
        )
