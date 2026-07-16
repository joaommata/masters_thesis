"""
c2_cv_pipeline_cf_stack.py

5-fold cross-validation pipeline for C2 quality control models, using a LEARNED
STACKING meta-layer over the k per-counterfactual logits instead of a plain
score-average (cf. c2_cv_pipeline_cf_ensemble.py, which averages the k scores).

Difference from c2_cv_pipeline_cf_ensemble.py
---------------------------------------------
The ensemble pipeline keeps the k counterfactuals separate, trains one base model
on the nearest-CF (rank-0) rows, scores all k test expansions, and combines the
k probabilities with a fixed mean.

Here the base model is IDENTICAL (same rank-0 fit, same folds, same seed), but the
fixed mean combiner is replaced by a learned one:
    * The base model produces OUT-OF-FOLD (k, N) train logits via an inner CV
      over the train fold: for each inner split a fresh base is fit on the
      inner-train rows and scores the held-out inner-val rows, so every train
      row is scored by a base that never saw it (unbiased logits).
    * A meta logistic regression is fit on those OOF train logits (input = k
      logits, output = P(correct)). This is the "final classification layer"
      over the per-CF logits, trained on the 80% train fold only.
    * At test time the (full-train-fold) base model produces the k test logits
      and the meta-LR combines them into one score per sample.

Because the base model is unchanged, stack-vs-average is a within-run,
same-base-model comparison: each fold stores both the stacked AUC (`auc`, the
headline `mean_auc`) and the plain-average AUC (`avg_auc` -> `mean_avg_auc`), and
`stack_vs_avg` isolates the effect of learning the combiner. `stack_gain`
(stack AUC - mean single-CF AUC) mirrors the ensemble's `ensemble_gain`, which is
also emitted (as an alias) so the analysis notebook's merge keeps working.

Note on leakage: the meta-LR is fit on OUT-OF-FOLD train logits (inner CV, see
oof_train_member_probs), so the combiner never sees a logit produced by a base
model that was trained on that same row. This removes the optimistic in-sample
bias a plain 80/20 stack would have, so the learned combiner generalises to the
honest test logits rather than overfitting the train fold. (The in-sample train
logits are still computed in train_model, but only for diagnostics.) The
test-fold AUC is honest either way — the 20% test is untouched until the final
score.

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
import time
import errno
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


def robust_dump(obj, path, retries=5, delay=0.5):
    """joblib.dump that tolerates BeeGFS metadata lag.

    On BeeGFS a freshly-created directory can still be invisible to the node
    doing the subsequent open(), so the write fails with ENOENT even though
    makedirs() just succeeded. Re-create the parent and retry with backoff.
    """
    for attempt in range(retries):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(obj, path)
            return
        except OSError as e:
            if e.errno == errno.ENOENT and attempt < retries - 1:
                time.sleep(delay * (attempt + 1))   # let BeeGFS metadata settle
                continue
            raise


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
                 'correct', 'path', 'patient_id'}
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

    delta     = query_scaled - train_scaled[j]            # scaled space, as in the original
    delta_emb = emb - cf_emb

    return {
        'prob': prob, 'attr': attr, 'emb': emb,
        'cf_prob': cf_prob, 'cf_attr': cf_attr, 'cf_emb': cf_emb,
        'delta': delta, 'delta_emb': delta_emb,
    }


# config name -> ordered list of block keys whose column-stack is that design matrix
_CONFIG_RECIPE = {
    'B1':   ['prob'],
    'B2':   ['attr'],
    'B3':   ['emb'],
    'B4':   ['prob', 'attr'],
    'B5':   ['prob', 'emb'],
    'M1':   ['delta'],
    'M2':   ['prob', 'delta'],
    'M3':   ['prob', 'delta', 'attr'],
    'M4':   ['prob', 'delta', 'emb'],
    'M5':   ['prob', 'delta', 'attr', 'emb'],
    'M6':   ['prob', 'delta', 'attr', 'cf_prob'],
    'MCF1': ['prob', 'cf_prob'],
    'MCF2': ['prob', 'cf_prob', 'attr', 'cf_attr'],
    'MCF3': ['prob', 'cf_prob', 'attr', 'cf_attr', 'delta'],
    'MCF4': ['prob', 'cf_prob', 'attr', 'cf_attr', 'delta', 'emb', 'cf_emb'],
    'MCF5': ['prob', 'cf_prob', 'attr', 'cf_attr', 'delta', 'emb', 'cf_emb', 'delta_emb'],
}


def assemble_one(b, config):
    """Design matrix for a single config from the feature blocks `b` (one rank).

    Used in the stacking loop to materialise one config's k expansions at a time,
    so peak memory is k × (this config) rather than k × (all configs) — the wide
    MCF4/MCF5 matrices otherwise OOM when stacked across all k ranks.
    """
    keys = _CONFIG_RECIPE[config]
    return b[keys[0]] if len(keys) == 1 else np.hstack([b[k] for k in keys])


def assemble_configs(b):
    """Map config name -> design matrix, given the feature blocks `b` for one rank."""
    return {cfg: assemble_one(b, cfg) for cfg in _CONFIG_RECIPE}


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _make_estimator(model_type):
    """Return a fresh (model, scaler) pair for the given base model type.

    scaler is None for RF (tree models are scale-invariant). LR/MLP get a
    StandardScaler. Factored out so the inner-CV OOF loop rebuilds the exact
    same estimator on each inner-train split.
    """
    if model_type == 'LR':
        scaler = StandardScaler()
        model  = LogisticRegression(max_iter=5000, class_weight='balanced',
                                    random_state=RANDOM_SEED)
    elif model_type == 'RF':
        scaler = None
        model  = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                        random_state=RANDOM_SEED, n_jobs=-1)
    elif model_type == 'MLP':
        scaler = StandardScaler()
        model  = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                               early_stopping=True, validation_fraction=0.05,
                               random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model, scaler


def _prep_factory(scaler, is_mlp):
    """Build a `_prep(X, fit)` closure that casts (MLP) and scales (LR/MLP)."""
    def _prep(X, fit):
        if is_mlp:
            X = X.astype(np.float32)
        if scaler is None:
            return X
        return scaler.fit_transform(X) if fit else scaler.transform(X)
    return _prep


def train_model(make_train, make_test, cf_count, y_train, model_type):
    """
    Train ONE base model on the nearest-CF (rank-0) training rows, then score
    every CF expansion of BOTH the train and test sets with it.

    The k test-side logits are what the stacking meta-layer combines at
    prediction time. The base model itself is identical to the averaging
    ensemble's (fit on the rank-0 expansion only), so stack-vs-average differs
    in exactly one thing: how the k logits are combined.

    The k train-side logits returned here are IN-SAMPLE (base scored on the same
    rank-0 rows it was fit on) and are optimistically biased; they are kept only
    for diagnostics. The meta-layer is instead fit on out-of-fold train logits
    from `oof_train_member_probs`, which are unbiased -> the learned combiner
    generalises to the honest test logits.

    Rank expansions are built on demand — one at a time via `make_train(rank)` /
    `make_test(rank)` — and discarded after scoring, so peak memory holds a single
    wide (e.g. MCF5, ~3.4k-col) matrix rather than all k at once. Materialising all
    k up front OOMs at k=16.

    Parameters
    ----------
    make_train : callable rank -> (N, d) train design matrix for that rank
    make_test  : callable rank -> (n_test, d) test design matrix for that rank
    cf_count   : int  number of ranks (k)
    y_train    : (N,) labels

    Returns
    -------
    train_member_probs : (k, N)      per-rank IN-SAMPLE train probabilities (diagnostic)
    test_member_probs  : (k, n_test) per-rank test probabilities
    fitted             : the fitted base model (plus scaler where applicable)
    """
    is_mlp = (model_type == 'MLP')
    model, scaler = _make_estimator(model_type)
    _prep = _prep_factory(scaler, is_mlp)

    # Fit on the rank-0 (nearest CF) expansion only, then score each rank in turn.
    X0 = make_train(0)
    model.fit(_prep(X0, fit=True), y_train)
    del X0

    train_probs, test_probs = [], []
    for rank in range(cf_count):
        Xtr = make_train(rank)
        train_probs.append(model.predict_proba(_prep(Xtr, fit=False))[:, 1])
        del Xtr
        Xte = make_test(rank)
        test_probs.append(model.predict_proba(_prep(Xte, fit=False))[:, 1])
        del Xte

    train_member_probs = np.stack(train_probs)
    test_member_probs  = np.stack(test_probs)
    fitted = model if scaler is None else {'model': model, 'scaler': scaler}
    return train_member_probs, test_member_probs, fitted


def oof_train_member_probs(make_train, cf_count, y_train, model_type,
                           n_inner_folds=5):
    """
    Out-of-fold per-rank train probabilities for the stacking meta-layer.

    The in-sample train logits (`train_member_probs` in `train_model`) are
    optimistically biased because the base model scores the same rank-0 rows it
    was fit on, so a meta-LR trained on them overfits and fails to generalise to
    the honest test logits. This routine removes that leakage: it runs an inner
    StratifiedKFold over the train fold and, for each inner split, fits a fresh
    base model on the inner-train rows (rank-0) and scores every rank of the
    held-out inner-validation rows. Each train row is therefore scored by a base
    model that never saw it -> unbiased (k, N) logits to fit the meta-LR on.

    The base estimator, rank-0-only fit, and per-rank scoring exactly mirror
    `train_model`, so the OOF logits are drawn from the same distribution as the
    test logits the meta-LR will ultimately combine.

    Returns
    -------
    oof_probs : (k, N) out-of-fold per-rank train probabilities.
    """
    is_mlp = (model_type == 'MLP')
    N = len(y_train)
    oof_probs = np.empty((cf_count, N), dtype=float)

    inner_skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True,
                                random_state=RANDOM_SEED)
    # Placeholder split matrix (only rank-0 needed to define the split geometry).
    X0_full = make_train(0)

    for inner_tr, inner_val in inner_skf.split(X0_full, y_train):
        model, scaler = _make_estimator(model_type)
        _prep = _prep_factory(scaler, is_mlp)

        # Fit on the inner-train rank-0 rows only (mirrors train_model's fit).
        model.fit(_prep(X0_full[inner_tr], fit=True), y_train[inner_tr])

        # Score every rank of the held-out inner-validation rows.
        for rank in range(cf_count):
            Xtr = make_train(rank)
            oof_probs[rank, inner_val] = model.predict_proba(
                _prep(Xtr[inner_val], fit=False))[:, 1]
            del Xtr

    del X0_full
    return oof_probs


def fit_stacker(train_member_probs, y_train):
    """
    Meta-layer: a logistic regression over the k base-model logits.

    train_member_probs : (k, N) train-fold per-CF probabilities
    Returns the fitted meta-LR (input = k logits, output = combined P(correct)).
    Falls back to a mean-combiner sentinel when k == 1 (nothing to stack).
    """
    meta = LogisticRegression(max_iter=5000, class_weight='balanced',
                              random_state=RANDOM_SEED)
    meta.fit(train_member_probs.T, y_train)   # (N, k) -> P(correct)
    return meta


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CV PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(disease, cf_count, save_fold_data=False, backbone='densenet',
           extended=False, distance='l1'):
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
    """

    print(f"\n{'='*70}")
    print(f"  {N_FOLDS}-FOLD CV (CF SCORE-STACKING): {disease.upper()} | k={cf_count} "
          f"| BACKBONE={backbone} | DISTANCE={distance}")
    print(f"{'='*70}\n")

    # ── Setup paths ───────────────────────────────────────────────────────
    results_base = os.path.join(RESULTS_DIR, '')
    data_subdir  = BACKBONE_MAP[backbone]
    cv_subdir    = 'cv_results_extended_cf_stack' if extended else 'cv_results_cf_stack'

    cv_dir = os.path.join(results_base, f'{data_subdir}_corrected/{disease}/{cv_subdir}/cf_{cf_count}')
    cv_dir = os.path.join(cv_dir, f'distance_{distance}') if distance != 'l1' else cv_dir

    fold_data_dir = os.path.join(cv_dir, 'fold_data')
    os.makedirs(cv_dir, exist_ok=True)
    os.makedirs(fold_data_dir, exist_ok=True)

    # Pre-create every model output folder up front, so BeeGFS has the whole run
    # to make them visible before the first joblib.dump (see robust_dump).
    for _mt in MODEL_TYPES:
        os.makedirs(os.path.join(cv_dir, 'models', _mt), exist_ok=True)

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
    print(f"CF Count:   {cf_count}  (train base on nearest CF; k test logits stacked by meta-LR)")
    print(f"CF Routing: correct_cf (pred=1 -> TN pool, pred=0 -> TP pool)")
    print(f"Save Fold Data: {save_fold_data}\n")
    print(f"Full dataset: {len(full_df):,} samples")
    print(f"  Correct:   {(full_df['correct']==1).sum():,}")
    print(f"  Incorrect: {(full_df['correct']==0).sum():,}\n")

    y = full_df['correct'].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    cv_results = {mt: {cfg: [] for cfg in CONFIGS} for mt in MODEL_TYPES}

    # ── FOLD LOOP ─────────────────────────────────────────────────────────
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, y)):

        print(f"\n{'-'*70}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'-'*70}")

        fold_train = full_df.iloc[train_idx].reset_index(drop=True)
        fold_test  = full_df.iloc[test_idx].reset_index(drop=True)
        print(f"  Train: {len(fold_train):,}  |  Test: {len(fold_test):,}")

        # ── Retrieve k ranked counterfactuals (no averaging) ──────────────
        print(f"  Retrieving {cf_count} ranked counterfactuals...")
        (train_clean, test_clean, train_cf_idx, test_cf_idx,
         train_scaled, test_scaled, relevant_cols, emb_cols) = compute_ranked_cf_correct_cf(
            train_df=fold_train, test_df=fold_test,
            cf_count=cf_count, disease=disease, distance=distance,
        )

        y_train = train_clean['correct'].values
        y_test  = test_clean['correct'].values

        # ── Design blocks (per rank) ──────────────────────────────────────
        # The base model is fit on the rank-0 (nearest CF) train rows, so its
        # training size is independent of k. Both train and test are expanded
        # once per rank: the k train logits fit the stacking meta-layer; the k
        # test logits are what the meta-layer combines at prediction time.
        #
        # We hold only the RAW blocks (prob/attr/emb/delta/cf_*) for the k ranks
        # here, and assemble the wide per-config matrices lazily inside the config
        # loop — one config at a time — so memory peaks at k × (one config) rather
        # than k × (all 14 configs); the MCF4/MCF5 matrices are ~3.4k-wide and
        # materialising all of them for every rank at once would OOM.
        blocks_train = [
            build_rank_blocks(train_clean, train_scaled, train_clean, train_scaled,
                              train_cf_idx, rank, disease, relevant_cols, emb_cols)
            for rank in range(cf_count)
        ]
        blocks_test = [
            build_rank_blocks(test_clean, test_scaled, train_clean, train_scaled,
                              test_cf_idx, rank, disease, relevant_cols, emb_cols)
            for rank in range(cf_count)
        ]

        if save_fold_data:
            # rank-0 expansion, for inspection
            _dump_rank0(train_clean, test_clean, train_cf_idx, test_cf_idx,
                        train_clean, disease, fold_data_dir, fold_idx)

        fold_pred_df = test_clean.copy()

        # ── One model per (model_type, config), scored over k CF expansions ──
        for model_type in MODEL_TYPES:
            print(f"\n  {model_type}:")

            model_dir = os.path.join(cv_dir, 'models', model_type)   # pre-created above

            for config in CONFIGS:
                # Build this config's rank-j expansion on demand (one at a time),
                # so we never hold all k wide matrices simultaneously.
                make_train = lambda r, c=config: assemble_one(blocks_train[r], c)
                make_test  = lambda r, c=config: assemble_one(blocks_test[r],  c)

                train_member_probs, test_member_probs, fitted = train_model(
                    make_train, make_test, cf_count, y_train, model_type)

                # Meta-layer: LR over the k base logits. Fit on OUT-OF-FOLD train
                # logits (inner CV, base never saw the row it scores) so the
                # combiner is trained on unbiased inputs and generalises to the
                # honest test logits, then applied to the test fold. With k==1
                # there is nothing to combine, so it degenerates to the single
                # logit (== average == stack).
                if cf_count > 1:
                    oof_probs = oof_train_member_probs(
                        make_train, cf_count, y_train, model_type)
                    meta   = fit_stacker(oof_probs, y_train)
                    y_prob = meta.predict_proba(test_member_probs.T)[:, 1]
                else:
                    meta   = None
                    y_prob = test_member_probs[0]

                auc = float(roc_auc_score(y_test, y_prob))
                fpr, tpr, _ = roc_curve(y_test, y_prob)

                # Plain score-average (the cf_ensemble baseline) on the SAME logits,
                # so stack-vs-average is a within-run, same-base-model comparison.
                y_prob_avg = test_member_probs.mean(axis=0)
                avg_auc    = float(roc_auc_score(y_test, y_prob_avg))

                robust_dump({'base': fitted, 'meta': meta}, os.path.join(
                    model_dir, f"{model_type}_{config}_fold{fold_idx}.pkl"))

                member_aucs = [float(roc_auc_score(y_test, test_member_probs[r]))
                               for r in range(cf_count)]

                cv_results[model_type][config].append({
                    'auc': auc,
                    'avg_auc': avg_auc,
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'y_prob': y_prob.tolist(),
                    'y_true': y_test.tolist(),
                    'member_aucs': member_aucs,
                })
                print(f"    {config}: stack AUC = {auc:.4f}  "
                      f"(avg {avg_auc:.4f}, single-CF mean {np.mean(member_aucs):.4f})")

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
            aucs     = [f['auc'] for f in folds]        # stacked combiner (headline)
            avg_aucs = [f['avg_auc'] for f in folds]     # plain score-average, same base
            # AUC if a single (rank-j) CF were used, averaged over the k members
            single_cf_means = [np.mean(f['member_aucs']) for f in folds]
            summary_rows.append({
                'model':             model_type,
                'config':            config,
                'mean_auc':          np.mean(aucs),          # = the STACK AUC
                'std_auc':           np.std(aucs),
                'min_auc':           np.min(aucs),
                'max_auc':           np.max(aucs),
                'mean_avg_auc':      np.mean(avg_aucs),      # plain average combiner
                'stack_vs_avg':      np.mean(aucs) - np.mean(avg_aucs),
                'mean_single_cf_auc': np.mean(single_cf_means),
                'stack_gain':        np.mean(aucs) - np.mean(single_cf_means),
                # alias so the notebook's cf_ensemble merge (expects ensemble_gain) works
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
        args = parser.parse_args()

        run_cv(
            disease=args.disease,
            cf_count=args.cf_count,
            save_fold_data=args.save_folds,
            backbone=args.backbone,
            extended=args.extended,
            distance=args.distance,
        )
