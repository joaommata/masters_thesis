"""
c2_cv_pipeline_diffusion_cf.py
==============================
Same as c2_cv_pipeline_new_split.py but uses real diffusion CFs
instead of simulated nearest-neighbour CFs.
"""
import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.append('/zhome/d0/a/221493/thesis/code')
sys.path.append('/zhome/d0/a/221493/thesis/code/c2')  # c2_prepare_data_simulated_cf lives here
from c2_prepare_data_simulated_cf import add_clinical_ratios  # NEW — reuse existing function

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
CF_ATTRIBUTES_PATH = '/work3/s251710/thesis_results/diffusion_cf/cf_attributes_10_0.25_n3.csv'
MANIFEST_PATH      = '/work3/s251710/thesis_results/diffusion_cf/cf_manifest_10_0.25_n3.csv'
N_FOLDS            = 5
RANDOM_SEED        = 42
CONFIGS = ['B1','B2','B3','B4','B5','M1','M2','M3','M4','M5','M6']
with open('/work3/s251710/thesis_results/C0_custom/effusion/threshold.txt') as _f:
    THRESHOLD = float(_f.read().strip())  # C0 Youden-optimal threshold from training

# ── Entropy / high-capacity-MLP switches (set from CLI in run_cv) ──────────────
# When ENTROPY is on, the raw C0 probability feature (and its CF counterpart
# cf_prob) is replaced everywhere by its binary entropy
#     H(p) = -p log p - (1-p) log(1-p),
# matching c2_cv_pipeline_cf_ensemble.py. --entropy also flips the MLP to the
# high-capacity BIG_MLP by default (override with --no_big_mlp).
ENTROPY = False   # replace prob (+ cf_prob) feature with H(prob); set by --entropy
BIG_MLP = False   # use the high-capacity MLP; enabled alongside --entropy


def _binary_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


# High-capacity MLP used when BIG_MLP is on (mirrors c2_cv_pipeline_cf_ensemble.py).
BIG_MLP_KWARGS = dict(
    hidden_layer_sizes=(256, 128, 64), activation='relu', alpha=1e-4,
    batch_size=256, learning_rate_init=1e-3, max_iter=1000,
    early_stopping=True, validation_fraction=0.10, n_iter_no_change=25,
    random_state=RANDOM_SEED,
)

# ══════════════════════════════════════════════════════════════════════════════
# NEW — CF INTEGRATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def _attr_cols_for(df, disease):
    """
    Attribute columns: everything that is not meta / embedding / delta.

    meta_cols matches c2_cv_pipeline_cf_ensemble.py exactly (which does NOT
    exclude `margin`, so margin stays an attribute feature for parity with the
    tabular ensemble). cf_path / cf_idx / cf_prob are also excluded because the
    per-rank CF frame carries them, but they never appear in the query frame.
    """
    meta_cols = {f'{disease}_prob', f'{disease}_pred', f'{disease}_true',
                 'correct', 'path', 'patient_id', 'cf_prob', 'cf_path', 'cf_idx', 'cf_paths'}
    return [c for c in df.columns
            if c not in meta_cols
            and not c.startswith('emb_')
            and not c.startswith('delta_')]


def attach_diffusion_cfs_ranked(df, cf_attrs, manifest, disease):
    """
    Attach the k diffusion CFs of each patient WITHOUT averaging them — one
    rank per CF (manifest column cf_idx = 0..k-1), so the k CFs can be score-
    ensembled at test time exactly like c2_cv_pipeline_cf_ensemble.py.

    Parameters
    ----------
    df         : original split dataframe (train or test fold)
    cf_attrs   : cf_attributes CSV as DataFrame, already merged with clinical
                 ratios and (cf_idx, cf_prob) from the manifest.
    manifest   : cf_manifest CSV as DataFrame (must contain cf_idx).
    disease    : disease name string.

    Returns
    -------
    q             : query frame (clinical ratios added, attr NaNs filled).
    delta_ranks   : list[k] of (n, |attr|) delta_attr arrays (query - rank-j CF).
    cf_prob_ranks : list[k] of (n,) rank-j CF probability arrays.
    attr_cols     : attribute column names.
    """
    q = add_clinical_ratios(df.copy())
    attr_cols = _attr_cols_for(q, disease)
    q[attr_cols] = q[attr_cols].fillna(0)

    k = int(manifest['cf_idx'].max()) + 1
    q_attr = q[attr_cols].values.astype(float)

    delta_ranks, cf_prob_ranks = [], []
    for rank in range(k):
        sub = (cf_attrs[cf_attrs['cf_idx'] == rank][['path'] + attr_cols + ['cf_prob']]
               .drop_duplicates('path'))
        merged = q[['path']].merge(sub, on='path', how='left')

        cf_attr = merged[attr_cols].values.astype(float)
        # Patients with no CF at this rank -> use their own attrs (delta == 0),
        # mirroring the non-ensemble diffusion pipeline's zero-fill of deltas.
        missing = np.isnan(cf_attr).any(axis=1)
        if missing.any():
            cf_attr[missing] = q_attr[missing]

        delta_ranks.append(q_attr - cf_attr)
        cf_prob_ranks.append(merged['cf_prob'].fillna(0.0).values.astype(float))

    return q, delta_ranks, cf_prob_ranks, attr_cols


def attach_diffusion_cfs(df, cf_attrs, manifest, disease):
    """
    Merges pre-computed diffusion CF attributes onto a dataframe and
    computes delta vectors. Supports multi-CF manifests (cf_idx column).

    When multiple CFs exist per patient, delta features are averaged across
    all CFs that successfully flipped the classifier (flipped==1). Averaging
    over diverse CFs reduces noise in the delta signal, analogous to what
    the matched approach gains from K=16 neighbours.

    Retained for reference / the legacy averaging path; the CV pipeline now
    uses attach_diffusion_cfs_ranked() so the k CFs can be score-ensembled.

    Parameters
    ----------
    df         : original split dataframe (train or test fold)
    cf_attrs   : cf_attributes CSV loaded as DataFrame (indexed by cf_path)
    manifest   : cf_manifest CSV loaded as DataFrame
    disease    : disease name string

    Returns
    -------
    df with delta_ columns and cf_prob column added
    """

    # Add clinical ratios to originals and CFs — same as simulated pipeline
    df       = add_clinical_ratios(df.copy())
    cf_attrs = add_clinical_ratios(cf_attrs.copy())

    # Identify attribute columns — exclude meta, embeddings, deltas
    meta_cols = {f'{disease}_prob', f'{disease}_pred', f'{disease}_true',
                 'correct', 'path', 'patient_id', 'cf_prob', 'cf_paths'}
    attr_cols = [c for c in df.columns
                 if c not in meta_cols
                 and not c.startswith('emb_')
                 and not c.startswith('delta_')]

    # Fill NaNs — same as compute_cf_for_split()
    df[attr_cols]       = df[attr_cols].fillna(0)
    cf_attrs[attr_cols] = cf_attrs[attr_cols].fillna(0)

    # -- Handle both single-CF (old) and multi-CF (new) manifests ----------
    multi_cf = 'cf_idx' in manifest.columns

    if multi_cf:
        n_total = len(manifest)
        n_flip  = manifest['flipped'].sum()
        print(f"  Manifest: {n_total} CFs, {n_flip} flipped ({n_flip/max(n_total,1):.1%})")

        # Merge CF attributes into manifest
        manifest_with_attrs = manifest.merge(
            cf_attrs[['cf_path'] + attr_cols], on='cf_path', how='left'
        )

        agg_dict = {col: 'mean' for col in attr_cols}
        agg_dict['cf_prob'] = 'mean'

        # Primary: average over flipped CFs
        flipped = manifest_with_attrs[manifest_with_attrs['flipped'] == 1]
        cf_agg_flipped = flipped.groupby('path').agg(agg_dict).reset_index()

        # Fallback: for patients with no flipped CF, use the best-attempt CF —
        # the one that moved furthest toward the target (min cf_prob for positives,
        # max for negatives). We don't know original pred here so we pick the CF
        # that is most extreme in either direction (closest to 0 or 1).
        not_flipped_paths = set(manifest['path']) - set(cf_agg_flipped['path'])
        if not_flipped_paths:
            print(f"  Fallback: {len(not_flipped_paths)} patients had no flipped CF — using best-attempt CF")
            unflipped = manifest_with_attrs[manifest_with_attrs['path'].isin(not_flipped_paths)].copy()
            # "Best attempt" = CF whose cf_prob is furthest from 0.5 (most committed in either direction)
            unflipped['dist_from_mid'] = (unflipped['cf_prob'] - 0.5).abs()
            best_idx = unflipped.groupby('path')['dist_from_mid'].idxmax()
            cf_agg_fallback = unflipped.loc[best_idx].groupby('path').agg(agg_dict).reset_index()
            cf_agg = pd.concat([cf_agg_flipped, cf_agg_fallback], ignore_index=True)
        else:
            cf_agg = cf_agg_flipped

        cf_agg.columns = ['path'] + [f'{c}_cf' for c in attr_cols] + ['cf_prob']
        df = df.merge(cf_agg, on='path', how='left')

    else:
        # Legacy single-CF path
        df = df.merge(manifest[['path', 'cf_path', 'cf_prob']], on='path', how='left')
        df = df.merge(
            cf_attrs[['cf_path'] + attr_cols], on='cf_path', how='left',
            suffixes=('', '_cf')
        )
        df = df.drop(columns=[c for c in ['cf_path'] if c in df.columns])

    # Compute delta = original_attr - cf_attr
    for col in attr_cols:
        cf_col = f'{col}_cf'
        if cf_col in df.columns:
            df[f'delta_{col}'] = df[col] - df[cf_col]
        else:
            df[f'delta_{col}'] = 0.0

    # Drop raw CF attribute columns, keep only deltas
    cf_only_cols = [f'{col}_cf' for col in attr_cols]
    df = df.drop(columns=[c for c in cf_only_cols if c in df.columns])

    # Any remaining NaN deltas (patient not in manifest at all) → zero
    n_missing = df['cf_prob'].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} samples have no CF in manifest at all — delta set to zero")

    delta_cols = [c for c in df.columns if c.startswith('delta_')]
    df[delta_cols] = df[delta_cols].fillna(0)
    df['cf_prob']  = df['cf_prob'].fillna(0)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, X_test_ranks, y_train, model_type='LR'):
    """
    Train ONE model on the rank-0 training rows, then score every CF expansion
    of the test set. Returns (member_probs (k, n_test), fitted).

    Mirrors c2_cv_pipeline_cf_ensemble.train_model: the training set is the
    rank-0 design matrix only (size independent of k), and the k score vectors
    are averaged after prediction. BIG_MLP swaps in the high-capacity MLP.
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
        if BIG_MLP:
            model = MLPClassifier(**BIG_MLP_KWARGS)
        else:
            model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                                  early_stopping=True, validation_fraction=0.05,
                                  random_state=RANDOM_SEED)
        model.fit(scaler.fit_transform(X_train.astype(np.float32)), y_train)
        member_probs = np.stack([
            model.predict_proba(scaler.transform(X_te.astype(np.float32)))[:, 1]
            for X_te in X_test_ranks
        ])
        return member_probs, {'model': model, 'scaler': scaler}
    raise ValueError(f"Unknown model type: {model_type}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE MATRIX BUILDER (per-CF rank; entropy-aware)
# ══════════════════════════════════════════════════════════════════════════════

def assemble_configs_ranked(q, delta, cf_prob, disease, attr_cols, emb_cols):
    """
    Build the {config: design-matrix} dict for ONE CF rank.

    Query-side blocks (prob, attr, emb) are shared across ranks; the CF-side
    blocks (delta_attr, cf_prob) come from the rank-th diffusion CF. When
    ENTROPY is on, prob and cf_prob are replaced by their binary entropy so
    every config that uses them picks it up.

    The diffusion cf_attrs carry no embeddings, so emb is query-side only and
    the emb-CF configs (MCF*) are not available; CONFIGS stays B1-B5,M1-M6.
    """
    prob    = q[[f'{disease}_prob']].values.astype(float)
    attr    = q[attr_cols].values.astype(float)
    emb     = q[emb_cols].values.astype(float)
    cf_prob = cf_prob.reshape(-1, 1).astype(float)

    if ENTROPY:
        prob    = _binary_entropy(prob)
        cf_prob = _binary_entropy(cf_prob)

    return {
        'B1': prob,
        'B2': attr,
        'B3': emb,
        'B4': np.hstack([prob, attr]),
        'B5': np.hstack([prob, emb]),
        'M1': delta,
        'M2': np.hstack([prob, delta]),
        'M3': np.hstack([prob, delta, attr]),
        'M4': np.hstack([prob, delta, emb]),
        'M5': np.hstack([prob, delta, attr, emb]),
        'M6': np.hstack([prob, delta, attr, cf_prob]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CV PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(disease, entropy=False, big_mlp=None):
    global ENTROPY, BIG_MLP
    ENTROPY = entropy
    BIG_MLP = entropy if big_mlp is None else big_mlp

    print(f"\n{'='*70}")
    print(f"  {N_FOLDS}-FOLD CV (DIFFUSION CF, per-CF ENSEMBLE): {disease.upper()}")
    print(f"  ENTROPY={ENTROPY}  BIG_MLP={BIG_MLP}")
    print(f"{'='*70}\n")

    results_base = os.path.join(RESULTS_DIR, '')
    cv_dir       = os.path.join(results_base, f'C2_diffusion/{disease}/cv_results')
    if ENTROPY:
        cv_dir = os.path.join(cv_dir, 'entropy')
    os.makedirs(cv_dir, exist_ok=True)

    # Load data — also load CF attributes and manifest once, outside fold loop
    full_df  = pd.read_csv(os.path.join(results_base, f'C2_custom/{disease}/c2_data.csv'))
    cf_attrs = pd.read_csv(CF_ATTRIBUTES_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    # Prepare CF attributes once: clinical ratios, then attach cf_idx + cf_prob
    # from the manifest so attach_diffusion_cfs_ranked() can slice per rank.
    cf_attrs = add_clinical_ratios(cf_attrs.copy())
    cf_attrs = cf_attrs.merge(
        manifest[['path', 'cf_idx', 'cf_path', 'cf_prob']], on='cf_path', how='left')
    cf_attrs['margin'] = np.abs(cf_attrs['cf_prob'] - THRESHOLD)

    # Rename C0 prediction columns to match expected format
    full_df.rename(columns={
        'prob': f'{disease}_prob',
        'pred': f'{disease}_pred',
        'true': f'{disease}_true'
    }, inplace=True)

    # Keep only samples that have a CF in the manifest
    full_df = full_df[full_df['path'].isin(manifest['path'])].reset_index(drop=True)
    emb_cols = [c for c in full_df.columns if c.startswith('emb_')]
    k = int(manifest['cf_idx'].max()) + 1
    print(f"Samples with diffusion CF: {len(full_df):,}  (k={k} CFs/patient, score-ensembled)")
    print(f"  Correct:   {(full_df['correct']==1).sum():,}")
    print(f"  Incorrect: {(full_df['correct']==0).sum():,}\n")

    y   = full_df['correct'].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    cv_results = {m: {c: [] for c in CONFIGS} for m in ['LR', 'RF', 'MLP']}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, y)):
        print(f"\n{'-'*70}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'-'*70}")

        fold_train = full_df.iloc[train_idx].reset_index(drop=True)
        fold_test  = full_df.iloc[test_idx].reset_index(drop=True)

        # Attach the k diffusion CFs as separate ranks (no feature averaging).
        print("  Attaching ranked diffusion CFs and computing per-CF deltas...")
        train_q, tr_delta, tr_cf_prob, attr_cols = attach_diffusion_cfs_ranked(
            fold_train, cf_attrs, manifest, disease)
        test_q,  te_delta, te_cf_prob, _         = attach_diffusion_cfs_ranked(
            fold_test,  cf_attrs, manifest, disease)

        y_train = train_q['correct'].values
        y_test  = test_q['correct'].values
        print(f"  Train: {len(train_q):,}  |  Test: {len(test_q):,}")

        # Train on rank-0 only; expand the test set over all k ranks.
        rank_train = assemble_configs_ranked(
            train_q, tr_delta[0], tr_cf_prob[0], disease, attr_cols, emb_cols)
        rank_test = [
            assemble_configs_ranked(test_q, te_delta[r], te_cf_prob[r], disease, attr_cols, emb_cols)
            for r in range(k)
        ]

        fold_pred_df = test_q.copy()

        for model_type in ['LR', 'RF', 'MLP']:
            print(f"\n  {model_type}:")
            model_dir = os.path.join(cv_dir, 'models', model_type)
            os.makedirs(model_dir, exist_ok=True)

            for config in CONFIGS:
                X_train_r0   = rank_train[config]
                X_test_ranks = [rank_test[r][config] for r in range(k)]

                member_probs, fitted = train_model(
                    X_train_r0, X_test_ranks, y_train, model_type=model_type)
                y_prob = member_probs.mean(axis=0)
                auc = float(roc_auc_score(y_test, y_prob))
                fpr, tpr, _ = roc_curve(y_test, y_prob)

                joblib.dump(fitted, os.path.join(
                    model_dir, f"{model_type}_{config}_fold{fold_idx}.pkl"))

                member_aucs = [float(roc_auc_score(y_test, member_probs[r])) for r in range(k)]

                cv_results[model_type][config].append({
                    'auc': auc, 'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
                    'y_prob': y_prob.tolist(), 'y_true': y_test.tolist(),
                    'member_aucs': member_aucs,
                })
                print(f"    {config}: ensemble AUC = {auc:.4f}  "
                      f"(single-CF mean {np.mean(member_aucs):.4f})")

                fold_pred_df[f"{model_type}_{config}_prob"] = y_prob

        fold_pred_df.to_csv(os.path.join(cv_dir, f'fold_{fold_idx}_predictions.csv'), index=False)

    # Aggregate and save
    summary_rows = []
    for model_type in ['LR', 'RF', 'MLP']:
        for config in CONFIGS:
            folds = cv_results[model_type][config]
            aucs = [f['auc'] for f in folds]
            single_cf_means = [np.mean(f['member_aucs']) for f in folds]
            summary_rows.append({
                'model': model_type, 'config': config,
                'mean_auc': np.mean(aucs), 'std_auc': np.std(aucs),
                'min_auc': np.min(aucs), 'max_auc': np.max(aucs),
                'mean_single_cf_auc': np.mean(single_cf_means),
                'ensemble_gain': np.mean(aucs) - np.mean(single_cf_means),
                'fold_aucs': ','.join([f'{a:.4f}' for a in aucs])
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(cv_dir, 'cv_summary.csv'), index=False)
    with open(os.path.join(cv_dir, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    print(f"\n{'='*70}")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {cv_dir}")

    return cv_results, summary_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='C2 CV over diffusion CFs (per-CF score ensemble, hardcoded to '
                    'match c2_cv_pipeline_cf_ensemble.py); optional entropy transform.')
    parser.add_argument('--disease', type=str, default='effusion')
    parser.add_argument('--entropy', action='store_true',
                        help='Replace C0 prob (+ cf_prob) with binary entropy H(p); '
                             'also enables the high-capacity MLP unless --no_big_mlp.')
    parser.add_argument('--big_mlp', dest='big_mlp', action='store_true', default=None,
                        help='Force the high-capacity MLP independently of --entropy.')
    parser.add_argument('--no_big_mlp', dest='big_mlp', action='store_false',
                        help='Force the small MLP even with --entropy.')
    args = parser.parse_args()
    run_cv(disease=args.disease, entropy=args.entropy, big_mlp=args.big_mlp)