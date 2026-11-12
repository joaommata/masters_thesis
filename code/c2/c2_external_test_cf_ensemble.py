"""
c2_external_test_cf_ensemble.py
===============================
Trains the Family-A (attribute-based) C2 ladder on the FULL C0 held-out cohort
and evaluates it once on the OFFICIAL CheXpert test set (valid.csv, 202 frontal
images, labelled by three board-certified radiologists).

    train : C2_<backbone>/<disease>/c2_data.csv                 (45,155 rows, NLP labels)
    test  : C2_<backbone>/<disease>/c2_data_chexpert_test.csv   (202 rows, expert labels)
    CF pool: the train frame, exactly as in cross-validation

This is the same experiment as `c2_cv_pipeline_cf_ensemble.py` with the 5-fold
loop replaced by a single external split, so the retrieval, the feature blocks,
the config definitions and the model hyperparameters are IMPORTED from that
module rather than restated here. Nothing about how a design matrix is built is
redefined in this file -- if it were, a difference in the external result could
not be attributed to the test set.

What the split buys, and what it does not
----------------------------------------
It buys clean labels. C2 is trained to predict correctness against the CheXpert
NLP labeller and tested against correctness under radiologist labels, so a result
that holds here is a result that survives the label noise the thesis names as a
limitation.

It does not buy resolution. C0 makes on the order of 20 errors out of 202, so the
positive class for error detection is tiny and a 95% CI on AUROC is roughly
+/-0.13 wide. Every AUROC below is therefore reported with a patient-level
bootstrap interval, and the summary is ordered so that the ladder's SHAPE -- which
configs sit above the confidence baseline -- is what the reader compares, not
differences of 0.01.

The MSP row
-----------
A `MSP` pseudo-config is included alongside the fitted models. It is not trained:
it scores correctness directly by |prob - threshold|, the plain softmax baseline
that [Bernhardt 2022] and [Bungert 2023] found no learned rejector beats. On a
202-row test set a training-free baseline is the most trustworthy reference
available, because it cannot be affected by how much training data the fitted
models got.

Usage:
    python c2_external_test_cf_ensemble.py --disease effusion --cf_count 16 --entropy
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import c2_cv_pipeline_cf_ensemble as cvp

RESULTS_DIR  = cvp.RESULTS_DIR
BACKBONE_MAP = cvp.BACKBONE_MAP
N_BOOTSTRAP  = 2000
BOOT_SEED    = 42


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_auc_ci(y_true, y_score, groups, n=N_BOOTSTRAP, seed=BOOT_SEED):
    """Percentile 95% CI for AUROC, resampling PATIENTS rather than images.

    Images from one patient are not independent, so a row-level bootstrap would
    understate the interval. With ~200 patients and ~1 image each the two are
    nearly the same here, but the patient-level version is the one that stays
    correct if this is ever run on a multi-image-per-patient cohort.
    """
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}

    aucs = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in pick])
        yt = y_true[rows]
        if yt.min() == yt.max():          # a resample with no errors has no AUROC
            continue
        aucs.append(roc_auc_score(yt, y_score[rows]))

    if not aucs:
        return float('nan'), float('nan'), 0
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), len(aucs)


def risk_coverage(y_true_correct, y_score):
    """AURC over the full coverage range. Higher score = more likely correct."""
    order = np.argsort(-np.asarray(y_score))          # most confident first
    errs  = 1 - np.asarray(y_true_correct)[order]
    n     = len(errs)
    risks = np.cumsum(errs) / np.arange(1, n + 1)
    return float(np.mean(risks)), risks, np.arange(1, n + 1) / n


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_external_test(disease, cf_count, backbone='densenet', distance='l1',
                      entropy=False, models=None, configs=None,
                      strategy='correct_cf', search_space='attr',
                      test_file='c2_data_chexpert_test.csv', tag='chexpert_test'):

    # The imported module reads ENTROPY as a module global inside
    # build_rank_blocks, so it is set there, not here.
    cvp.ENTROPY = entropy
    model_types = list(models)  if models  is not None else list(cvp.MODEL_TYPES)
    config_list = list(configs) if configs is not None else list(cvp.CONFIGS)

    print(f"\n{'='*74}")
    print(f"  EXTERNAL TEST: {disease.upper()} | k={cf_count} | BACKBONE={backbone}")
    print(f"  DISTANCE={distance}  STRATEGY={strategy}  SEARCH_SPACE={search_space}")
    print(f"  ENTROPY={entropy}")
    print(f"  MODELS={model_types}")
    print(f"  CONFIGS={config_list}")
    print(f"{'='*74}\n")

    data_subdir = BACKBONE_MAP[backbone]
    data_dir    = os.path.join(RESULTS_DIR, data_subdir, disease)

    # Mirrors the CV pipeline's directory grammar, under an external_test/ root so
    # nothing can land on top of a cross-validation result.
    out_dir = os.path.join(RESULTS_DIR, f'{data_subdir}_corrected', disease,
                           'external_test', tag,
                           f'cf_{cf_count}', f'distance_{distance}',
                           f'strategy_{strategy}')
    if search_space != 'attr':
        out_dir = os.path.join(out_dir, f'search_{search_space}')
    if entropy:
        out_dir = os.path.join(out_dir, 'TEST_entropy_added(M-only)')
    os.makedirs(out_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    rename = {'prob': f'{disease}_prob', 'pred': f'{disease}_pred', 'true': f'{disease}_true'}

    train_df = pd.read_csv(os.path.join(data_dir, 'c2_data.csv')).rename(columns=rename)
    test_df  = pd.read_csv(os.path.join(data_dir, test_file)).rename(columns=rename)

    print(f"Train (C0 held-out, NLP labels): {len(train_df):,} rows, "
          f"{train_df['patient_id'].nunique():,} patients")
    print(f"  correct=1 {int((train_df['correct']==1).sum()):,} | "
          f"correct=0 {int((train_df['correct']==0).sum()):,} "
          f"(acc {train_df['correct'].mean():.4f})")
    print(f"Test  (official CheXpert, expert labels): {len(test_df):,} rows, "
          f"{test_df['patient_id'].nunique():,} patients")
    n_err = int((test_df['correct'] == 0).sum())
    print(f"  correct=1 {int((test_df['correct']==1).sum()):,} | "
          f"correct=0 {n_err:,} (acc {test_df['correct'].mean():.4f})")

    # C0 was trained on a patient-level split of train.csv; valid patients are
    # disjoint by construction. Assert it rather than assume it.
    overlap = set(train_df['patient_id']) & set(test_df['patient_id'])
    if overlap:
        raise ValueError(f"{len(overlap)} patients appear in BOTH the training cohort "
                         f"and the external test set: {sorted(overlap)[:5]}")
    print(f"  patient overlap with train: 0 (verified)\n")

    if n_err < 25:
        print(f"  ! Only {n_err} errors. Treat every interval below as wide and "
              f"compare the SHAPE of the ladder, not individual 0.01 gaps.\n")

    missing = [c for c in train_df.columns if c not in test_df.columns]
    if missing:
        raise ValueError(f"Test frame is missing {len(missing)} train columns, e.g. "
                         f"{missing[:6]} -- rerun c1_match_chexpert_test.py")

    # ── Retrieve k ranked counterfactuals from the FULL training pool ─────────
    print(f"Retrieving {cf_count} ranked counterfactuals "
          f"(strategy={strategy}, search space={search_space})...")
    print(f"  (this is the slow step: {len(train_df):,} train queries against the pool)")
    (train_clean, test_clean, train_cf_idx, test_cf_idx,
     train_scaled, test_scaled, relevant_cols, emb_cols) = cvp.CF_STRATEGIES[strategy](
        train_df=train_df, test_df=test_df,
        cf_count=cf_count, disease=disease, distance=distance,
        search_space=search_space,
    )
    print(f"  attribute space: {len(relevant_cols)} cols | embeddings: {len(emb_cols)} cols")

    y_train = train_clean['correct'].values
    y_test  = test_clean['correct'].values
    groups_test = test_clean['patient_id'].values

    # ── Design matrices (built by the imported, canonical code) ──────────────
    print("\nBuilding design matrices...")
    b_tr = cvp.build_rank_blocks(train_clean, train_scaled, train_clean, train_scaled,
                                 train_cf_idx, 0, disease, relevant_cols, emb_cols)
    train_designs = cvp.assemble_configs(b_tr)

    test_designs = []
    for rank in range(cf_count):
        b_te = cvp.build_rank_blocks(test_clean, test_scaled, train_clean, train_scaled,
                                     test_cf_idx, rank, disease, relevant_cols, emb_cols)
        test_designs.append(cvp.assemble_configs(b_te))

    pred_df = test_clean[['path', 'patient_id', f'{disease}_prob',
                          f'{disease}_pred', f'{disease}_true', 'correct']].copy()
    train_paths = train_clean['path'].values
    pred_df['cf_paths'] = ['|'.join(train_paths[test_cf_idx[i]]) for i in range(len(test_cf_idx))]

    # ── The training-free confidence baseline ────────────────────────────────
    # |prob - t| recovered from the margin column of the same frame the pipeline
    # reads, so it sits at exactly the operating point `correct` was defined at.
    msp_score = test_clean['margin'].values.astype(float) \
        if 'margin' in test_clean.columns \
        else np.abs(test_clean[f'{disease}_prob'].values - 0.5)

    results = {mt: {} for mt in model_types}
    rows = []

    def _record(model_type, config, y_prob, member_aucs=None, trained=True):
        auc = float(roc_auc_score(y_test, y_prob))
        lo, hi, n_ok = bootstrap_auc_ci(y_test, y_prob, groups_test)
        aurc, _, _ = risk_coverage(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results.setdefault(model_type, {})[config] = [{
            'auc': auc, 'auc_lo': lo, 'auc_hi': hi, 'aurc': aurc,
            'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
            'y_prob': np.asarray(y_prob).tolist(), 'y_true': y_test.tolist(),
            'member_aucs': member_aucs if member_aucs is not None else [auc],
        }]
        rows.append({
            'model': model_type, 'config': config,
            'auc': auc, 'auc_lo': lo, 'auc_hi': hi, 'ci_width': hi - lo,
            'aurc': aurc,
            'single_cf_auc': float(member_aucs[0]) if member_aucs else auc,
            'mean_single_cf_auc': float(np.mean(member_aucs)) if member_aucs else auc,
            'n_test': len(y_test), 'n_errors': int((y_test == 0).sum()),
            'trained': trained,
        })
        pred_df[f'{model_type}_{config}_prob'] = y_prob
        star = '' if trained else '  (training-free)'
        print(f"    {config:<6} AUC = {auc:.4f}  [{lo:.4f}, {hi:.4f}]  "
              f"AURC = {aurc:.4f}{star}")

    # MSP is model-independent; recorded once under a pseudo-model so the
    # notebook can pick it up through the same loaders as everything else.
    print("\n  BASELINE (not trained):")
    _record('NONE', 'MSP', msp_score, trained=False)

    # ── Fit and score ────────────────────────────────────────────────────────
    for model_type in model_types:
        print(f"\n  {model_type}:")
        model_dir = os.path.join(out_dir, 'models', model_type)
        os.makedirs(model_dir, exist_ok=True)

        for config in config_list:
            X_train = train_designs[config]
            X_tests = [test_designs[r][config] for r in range(cf_count)]

            member_probs, fitted = cvp.train_model(X_train, X_tests, y_train, model_type)
            y_prob = member_probs.mean(axis=0)      # ensemble over the k CFs
            member_aucs = [float(roc_auc_score(y_test, member_probs[r]))
                           for r in range(cf_count)]

            joblib.dump(fitted, os.path.join(model_dir, f"{model_type}_{config}.pkl"))
            _record(model_type, config, y_prob, member_aucs=member_aucs)

    # ── Write ────────────────────────────────────────────────────────────────
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, 'cv_summary.csv'), index=False)
    with open(os.path.join(out_dir, 'cv_detailed.json'), 'w') as f:
        json.dump(results, f, indent=2)
    pred_df.to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)

    with open(os.path.join(out_dir, 'run_config.json'), 'w') as f:
        json.dump({
            'disease': disease, 'cf_count': cf_count, 'backbone': backbone,
            'distance': distance, 'strategy': strategy, 'search_space': search_space,
            'entropy': entropy, 'models': model_types, 'configs': config_list,
            'train_csv': os.path.join(data_dir, 'c2_data.csv'),
            'test_csv': os.path.join(data_dir, test_file),
            'n_train': int(len(train_clean)), 'n_test': int(len(test_clean)),
            'n_test_errors': int((y_test == 0).sum()),
            'n_bootstrap': N_BOOTSTRAP,
        }, f, indent=2)

    print(f"\n{'='*74}")
    print("EXTERNAL TEST SUMMARY  (sorted by AUROC)")
    print(f"{'='*74}\n")
    show = summary.sort_values('auc', ascending=False)[
        ['model', 'config', 'auc', 'auc_lo', 'auc_hi', 'ci_width', 'aurc', 'single_cf_auc']]
    print(show.to_string(index=False))

    msp_auc = summary.loc[summary['config'] == 'MSP', 'auc'].iloc[0]
    beat = summary[(summary['auc'] > msp_auc) & (summary['config'] != 'MSP')]
    print(f"\nMSP (training-free) = {msp_auc:.4f}")
    print(f"{len(beat)} of {len(summary)-1} fitted configs score above it. "
          f"With {int((y_test==0).sum())} errors, none of these differences is "
          f"individually significant -- read the ranking, not the gaps.")
    print(f"\nResults saved to: {out_dir}")

    return results, summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Family-A C2 ladder on an external test set')
    ap.add_argument('--disease',  type=str, default='effusion')
    ap.add_argument('--cf_count', type=int, default=16,
                    help='k counterfactuals averaged at test time (rank 0 = the k=1 result)')
    ap.add_argument('--backbone', type=str, default='densenet',
                    choices=list(BACKBONE_MAP.keys()))
    ap.add_argument('--distance', type=str, default='l1', choices=['l1', 'l2', 'cosine'])
    ap.add_argument('--entropy', action='store_true',
                    help='ADD H(p) to the M/MCF configs, baselines untouched — '
                         'matches the canonical CV run')
    ap.add_argument('--models', nargs='+', default=None, choices=['LR', 'RF', 'MLP'])
    ap.add_argument('--configs', nargs='+', default=None)
    ap.add_argument('--strategy', type=str, default='correct_cf',
                    choices=['correct_cf', 'gt_routing'])
    ap.add_argument('--search_space', type=str, default='attr', choices=['attr', 'emb'])
    ap.add_argument('--test_file', type=str, default='c2_data_chexpert_test.csv')
    ap.add_argument('--tag', type=str, default='chexpert_test',
                    help='Sub-directory name identifying the external set')
    a = ap.parse_args()

    run_external_test(disease=a.disease, cf_count=a.cf_count, backbone=a.backbone,
                      distance=a.distance, entropy=a.entropy, models=a.models,
                      configs=a.configs, strategy=a.strategy,
                      search_space=a.search_space, test_file=a.test_file, tag=a.tag)
