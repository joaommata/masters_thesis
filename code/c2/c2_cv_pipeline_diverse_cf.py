"""
c2_cv_pipeline_diverse_cf.py
============================
5-fold CV for C2 correctness prediction on the diverse-CF grid.

Unlike c2_cv_pipeline_cf_ensemble.py, NO counterfactual is retrieved: every query image
already has its own diffusion-generated counterfactuals, one per (region, t_start) cell of
the grid. Those are the CFs, and delta-A = attr(query) - attr(CF) is computed against them
directly. There is no KNN, no CF pool and no routing strategy.

The CF attributes are the CLASSIC ones (demographics + geometry + firstorder/shape2D
radiomics), produced by c1_diverse_build_attribute_vector.py. The diffusion-dynamics
fingerprint (cf_extract_diverse_features.py -> cf_features.parquet) is a SEPARATE model and
is deliberately not used here.

Cells act as ensemble members in place of KNN ranks: one model is trained on the
--train_cell expansion, then each test image is scored once per selected cell and the
scores are averaged, exactly as the k CF ranks are averaged in the ensemble pipeline.

MCF4/MCF5 are unavailable: they need cf_emb, and the CFs have no C0 embedding (that would
require a separate C0 forward pass over the CF tensors).

Usage:
    python c2_cv_pipeline_diverse_cf.py --disease effusion
    python c2_cv_pipeline_diverse_cf.py --disease effusion --cells general
    python c2_cv_pipeline_diverse_cf.py --disease effusion --per_cell
"""

import os
import sys
import json
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from c2_feature_spec import C0_DERIVED_COLS, RSNA_META_COLS
from c2_cv_pipeline_cf_ensemble import add_clinical_ratios, _binary_entropy

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
N_FOLDS     = 5
RANDOM_SEED = 42

MODEL_TYPES = ['LR', 'RF', 'MLP', 'HGB']

# MCF4/MCF5 need cf_emb — see module docstring.
CONFIGS = ['B1', 'B2', 'B3', 'B4', 'B5',
           'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
           'MCF1', 'MCF2', 'MCF3']

BACKBONE_MAP = {
    'final':     'C2_final',   # final_experiment rebuild (c2_build_dataset.py)
    'densenet': ('C2_custom',   f"{RESULTS_DIR}/diverse_cf"),
    'medmnist': ('C2_medmnist', f"{RESULTS_DIR}/diverse_cf_medmnist"),
}

ENTROPY = False   # set by --entropy; swaps prob/cf_prob for their binary entropy


# ══════════════════════════════════════════════════════════════════════════════
# CELL SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def cell_key(region, t_start):
    return f"{region}:{int(t_start)}"


def resolve_cells(cf_attrs, spec):
    """
    Map a --cells spec to an ordered list of (region, t_start) cells.

    'all'      every cell in the grid
    'general'  the unmasked CF only (the closest analogue of a classic diffusion CF)
    'masked'   the four anatomical regions, i.e. everything except 'general'
    otherwise  an explicit list of 'Region:t' keys
    """
    available = (cf_attrs[['region', 't_start']].drop_duplicates()
                 .sort_values(['region', 't_start']))
    cells = [(r, int(t)) for r, t in available.itertuples(index=False)]

    if spec == ['all']:
        chosen = cells
    elif spec == ['general']:
        chosen = [c for c in cells if c[0] == 'general']
    elif spec == ['masked']:
        chosen = [c for c in cells if c[0] != 'general']
    else:
        lookup = {cell_key(r, t): (r, t) for r, t in cells}
        missing = [s for s in spec if s not in lookup]
        if missing:
            raise SystemExit(f"Unknown cells {missing}. Available: {sorted(lookup)}")
        chosen = [lookup[s] for s in spec]

    if not chosen:
        raise SystemExit(f"--cells {spec} selected no cells. Available: {cells}")
    # 'general' first so the default --train_cell is a real counterfactual (it has the
    # highest flip rate); the rest in a stable order.
    return sorted(chosen, key=lambda c: (c[0] != 'general', c[0], c[1]))


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

def build_cell_blocks(q_clean, q_scaled, cf_by_cell, cell, attr_cols, emb_cols,
                      disease, scaler):
    """
    Feature blocks for one query frame paired with ONE grid cell's counterfactuals.

    Query-side blocks (prob, attr, emb) do not depend on the cell. The CF-side blocks
    come from that cell's CF of each query — a fixed pairing, not a retrieval.

    delta is taken in STANDARDISED attribute space with the scaler fit on the training
    queries, matching build_rank_blocks() in c2_cv_pipeline_cf_ensemble.py.
    """
    disease_lower = disease.lower()
    cf = cf_by_cell[cell].reindex(q_clean['path'].values)

    prob    = q_clean[[f'{disease_lower}_prob']].values.astype(float)
    attr    = q_clean[attr_cols].values.astype(float)
    emb     = q_clean[emb_cols].values.astype(float)

    cf_prob = cf[['cf_prob']].values.astype(float)
    cf_attr = cf[attr_cols].values.astype(float)
    cf_scaled = scaler.transform(cf_attr)

    if ENTROPY:
        ent_prob, ent_cf_prob = _binary_entropy(prob), _binary_entropy(cf_prob)
    else:
        ent_prob    = np.empty((prob.shape[0], 0))
        ent_cf_prob = np.empty((cf_prob.shape[0], 0))

    return {
        'prob': prob, 'attr': attr, 'emb': emb,
        'cf_prob': cf_prob, 'cf_attr': cf_attr,
        'delta': q_scaled - cf_scaled,
        'ent_prob': ent_prob, 'ent_cf_prob': ent_cf_prob,
    }


def assemble_configs(b):
    """
    config -> design matrix. Same definitions as c2_cv_pipeline_cf_ensemble.py for every
    config that does not need an embedding of the CF image.
    """
    return {
        'B1':   b['prob'],
        'B2':   b['attr'],
        'B3':   b['emb'],
        'B4':   np.hstack([b['prob'], b['attr']]),
        'B5':   np.hstack([b['prob'], b['emb']]),
        'M1':   np.hstack([b['delta'], b['ent_prob']]),
        'M2':   np.hstack([b['prob'], b['delta'], b['ent_prob']]),
        'M3':   np.hstack([b['prob'], b['delta'], b['attr'], b['ent_prob']]),
        'M4':   np.hstack([b['prob'], b['delta'], b['emb'], b['ent_prob']]),
        'M5':   np.hstack([b['prob'], b['delta'], b['attr'], b['emb'], b['ent_prob']]),
        'M6':   np.hstack([b['prob'], b['delta'], b['attr'], b['cf_prob'],
                           b['ent_prob'], b['ent_cf_prob']]),
        'MCF1': np.hstack([b['prob'], b['cf_prob'], b['ent_prob'], b['ent_cf_prob']]),
        'MCF2': np.hstack([b['prob'], b['cf_prob'], b['attr'], b['cf_attr'],
                           b['ent_prob'], b['ent_cf_prob']]),
        'MCF3': np.hstack([b['prob'], b['cf_prob'], b['attr'], b['cf_attr'], b['delta'],
                           b['ent_prob'], b['ent_cf_prob']]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, X_test_cells, y_train, model_type):
    """Train once on the --train_cell rows, score every selected cell's test expansion."""
    if model_type == 'LR':
        scaler = StandardScaler()
        model  = LogisticRegression(max_iter=5000, class_weight='balanced',
                                    random_state=RANDOM_SEED)
        model.fit(scaler.fit_transform(X_train), y_train)
        probs = np.stack([model.predict_proba(scaler.transform(X))[:, 1]
                          for X in X_test_cells])
        return probs, {'model': model, 'scaler': scaler}

    if model_type == 'RF':
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                       random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        probs = np.stack([model.predict_proba(X)[:, 1] for X in X_test_cells])
        return probs, model

    if model_type == 'MLP':
        scaler = StandardScaler()
        model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                              early_stopping=True, validation_fraction=0.05,
                              random_state=RANDOM_SEED)
        model.fit(scaler.fit_transform(X_train.astype(np.float32)), y_train)
        probs = np.stack([model.predict_proba(scaler.transform(X.astype(np.float32)))[:, 1]
                          for X in X_test_cells])
        return probs, {'model': model, 'scaler': scaler}

    if model_type == 'HGB':
        # Added for this pipeline: it is what scores ~0.79 on the dynamics fingerprint in
        # cf_cell_ablation.py, and delta-A over ~450 correlated columns is the same shape
        # of problem. No scaler — trees are scale-free.
        model = HistGradientBoostingClassifier(max_iter=300, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        probs = np.stack([model.predict_proba(X)[:, 1] for X in X_test_cells])
        return probs, model

    raise ValueError(f"Unknown model type: {model_type}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data(disease, backbone):
    """Query frame restricted to the grid, plus the CF attributes keyed by cell."""
    data_subdir, grid_dir = BACKBONE_MAP[backbone]

    attrs_path = os.path.join(grid_dir, 'cf_attributes.parquet')
    if not os.path.exists(attrs_path):
        raise SystemExit(
            f"No CF attributes at {attrs_path}.\n"
            f"Run c1_diverse_build_attribute_vector.py "
            f"(jobs/analysis/build_diverse_cf_attributes.sh) first.")

    cf_attrs = pd.read_parquet(attrs_path)
    full_df  = pd.read_csv(os.path.join(RESULTS_DIR, f'{data_subdir}/{disease}/c2_data.csv'))
    full_df.rename(columns={'prob':  f'{disease}_prob',
                            'pred':  f'{disease}_pred',
                            'true':  f'{disease}_true'}, inplace=True)

    grid_paths = set(cf_attrs['path'].unique())
    before = len(full_df)
    full_df = full_df[full_df['path'].isin(grid_paths)].reset_index(drop=True)
    print(f"c2_data.csv: {before:,} rows -> {len(full_df):,} on the grid "
          f"({full_df['patient_id'].nunique():,} patients)")

    missing = grid_paths - set(full_df['path'])
    if missing:
        print(f"[warn] {len(missing):,} grid images are absent from c2_data.csv "
              f"and are dropped (c2_data.csv may have been regenerated since the grid).")

    return full_df, cf_attrs


def prepare_features(full_df, cf_attrs, disease, cells):
    """
    Returns the query frame with clinical ratios, the CF frame per cell (indexed by the
    query path), and the attribute columns shared by both sides.
    """
    disease_lower = disease.lower()
    q = add_clinical_ratios(full_df.copy())

    meta_cols = {f'{disease_lower}_prob', f'{disease_lower}_pred', f'{disease_lower}_true',
                 'correct', 'path', 'cam_path', 'patient_id',
                 *C0_DERIVED_COLS, *RSNA_META_COLS}
    emb_cols  = [c for c in q.columns if c.startswith('emb_')]
    q_attr    = [c for c in q.columns if c not in meta_cols and c not in emb_cols]

    cf = add_clinical_ratios(cf_attrs.copy())
    attr_cols = [c for c in q_attr if c in cf.columns]
    dropped = [c for c in q_attr if c not in cf.columns]
    print(f"Attribute columns: {len(attr_cols)} shared "
          f"({len(dropped)} query-only, no delta possible: {dropped[:8]}"
          f"{' ...' if len(dropped) > 8 else ''})")

    q[attr_cols] = q[attr_cols].fillna(0)
    cf[attr_cols] = cf[attr_cols].fillna(0)

    cf_by_cell = {}
    for region, t in cells:
        sub = cf[(cf['region'] == region) & (cf['t_start'] == t)]
        sub = sub.drop_duplicates('path').set_index('path')
        cf_by_cell[(region, t)] = sub

    # Only images with a CF in EVERY selected cell can be scored by every ensemble
    # member; keeping partial images would silently make the members' test sets differ.
    complete = set(q['path'])
    for cell, sub in cf_by_cell.items():
        complete &= set(sub.index)
    n_before = len(q)
    q = q[q['path'].isin(complete)].reset_index(drop=True)
    if len(q) < n_before:
        print(f"[warn] {n_before - len(q):,} images dropped: no CF in every selected cell "
              f"(usually an empty mask).")

    return q, cf_by_cell, attr_cols, emb_cols


# ══════════════════════════════════════════════════════════════════════════════
# CV
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(disease, backbone='densenet', cells_spec=('all',), train_cell=None,
           entropy=False, models=None, configs=None, tag=None):
    global ENTROPY, MODEL_TYPES, CONFIGS
    ENTROPY = entropy
    if models is not None:
        MODEL_TYPES = list(models)
    if configs is not None:
        CONFIGS = list(configs)

    full_df, cf_attrs = load_data(disease, backbone)
    cells = resolve_cells(cf_attrs, list(cells_spec))

    if train_cell is None:
        train_idx = 0
    else:
        keys = [cell_key(*c) for c in cells]
        if train_cell not in keys:
            raise SystemExit(f"--train_cell {train_cell} is not among the selected "
                             f"cells: {keys}")
        train_idx = keys.index(train_cell)

    data_subdir, _ = BACKBONE_MAP[backbone]
    cv_dir = os.path.join(RESULTS_DIR, f'{data_subdir}_corrected/{disease}',
                          'cv_results_diverse_cf', tag or '_'.join(cells_spec))
    if ENTROPY:
        cv_dir = os.path.join(cv_dir, 'TEST_entropy')
    os.makedirs(cv_dir, exist_ok=True)

    q, cf_by_cell, attr_cols, emb_cols = prepare_features(full_df, cf_attrs, disease, cells)

    print(f"\n{'='*70}")
    print(f"  {N_FOLDS}-FOLD CV (DIVERSE CF, PREDETERMINED PAIRS): {disease.upper()}")
    print(f"  cells ({len(cells)}): {[cell_key(*c) for c in cells]}")
    print(f"  train cell: {cell_key(*cells[train_idx])}   ENTROPY={ENTROPY}")
    print(f"  MODELS={MODEL_TYPES}  CONFIGS={CONFIGS}")
    print(f"  images: {len(q):,}  patients: {q['patient_id'].nunique():,}  "
          f"correct rate: {q['correct'].mean():.3f}")
    print(f"{'='*70}\n", flush=True)

    y = q['correct'].values
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = {mt: {cfg: [] for cfg in CONFIGS} for mt in MODEL_TYPES}

    for fold_idx, (tr, te) in enumerate(skf.split(q, y, groups=q['patient_id'].values)):
        fold_train = q.iloc[tr].reset_index(drop=True)
        fold_test  = q.iloc[te].reset_index(drop=True)
        overlap = set(fold_train['patient_id']) & set(fold_test['patient_id'])
        print(f"\n{'-'*70}\nFOLD {fold_idx + 1}/{N_FOLDS}   "
              f"train {len(fold_train):,} | test {len(fold_test):,} | "
              f"patient overlap {len(overlap)} (should be 0)\n{'-'*70}", flush=True)

        # Scaler fit on the TRAINING queries only; both sides of every delta use it.
        scaler = StandardScaler().fit(fold_train[attr_cols].values.astype(float))
        tr_scaled = scaler.transform(fold_train[attr_cols].values.astype(float))
        te_scaled = scaler.transform(fold_test[attr_cols].values.astype(float))

        b_tr = build_cell_blocks(fold_train, tr_scaled, cf_by_cell, cells[train_idx],
                                 attr_cols, emb_cols, disease, scaler)
        X_train_all = assemble_configs(b_tr)

        X_test_cells = []
        for cell in cells:
            b_te = build_cell_blocks(fold_test, te_scaled, cf_by_cell, cell,
                                     attr_cols, emb_cols, disease, scaler)
            X_test_cells.append(assemble_configs(b_te))

        y_train, y_test = fold_train['correct'].values, fold_test['correct'].values
        fold_pred_df = fold_test[['path', 'patient_id', f'{disease}_prob', 'correct']].copy()

        for model_type in MODEL_TYPES:
            print(f"\n  {model_type}:", flush=True)
            model_dir = os.path.join(cv_dir, 'models', model_type)
            os.makedirs(model_dir, exist_ok=True)

            for config in CONFIGS:
                member_probs, fitted = train_model(
                    X_train_all[config], [X[config] for X in X_test_cells],
                    y_train, model_type)

                y_prob = member_probs.mean(axis=0)
                auc = float(roc_auc_score(y_test, y_prob))
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                joblib.dump(fitted, os.path.join(
                    model_dir, f"{model_type}_{config}_fold{fold_idx}.pkl"))

                member_aucs = [float(roc_auc_score(y_test, member_probs[i]))
                               for i in range(len(cells))]
                cv_results[model_type][config].append({
                    'auc': auc, 'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
                    'y_prob': y_prob.tolist(), 'y_true': y_test.tolist(),
                    'member_aucs': member_aucs,
                })
                best = int(np.argmax(member_aucs))
                print(f"    {config}: ensemble AUC = {auc:.4f}  "
                      f"(single-cell mean {np.mean(member_aucs):.4f}, "
                      f"best {member_aucs[best]:.4f} @ {cell_key(*cells[best])})",
                      flush=True)
                fold_pred_df[f"{model_type}_{config}_prob"] = y_prob

        fold_pred_df.to_csv(os.path.join(cv_dir, f'fold_{fold_idx}_predictions.csv'),
                            index=False)

    # ── aggregate ─────────────────────────────────────────────────────────────
    rows = []
    for model_type in MODEL_TYPES:
        for config in CONFIGS:
            folds = cv_results[model_type][config]
            if not folds:
                continue
            aucs = [f['auc'] for f in folds]
            singles = [np.mean(f['member_aucs']) for f in folds]
            per_cell = np.mean([f['member_aucs'] for f in folds], axis=0)
            rows.append({
                'model': model_type, 'config': config,
                'mean_auc': np.mean(aucs), 'std_auc': np.std(aucs),
                'min_auc': np.min(aucs), 'max_auc': np.max(aucs),
                'mean_single_cell_auc': np.mean(singles),
                'ensemble_gain': np.mean(aucs) - np.mean(singles),
                'best_cell': cell_key(*cells[int(np.argmax(per_cell))]),
                'best_cell_auc': float(np.max(per_cell)),
                'fold_aucs': ','.join(f'{a:.4f}' for a in aucs),
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(cv_dir, 'cv_summary.csv'), index=False)
    with open(os.path.join(cv_dir, 'cv_detailed.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    with open(os.path.join(cv_dir, 'cells.json'), 'w') as f:
        json.dump({'cells': [cell_key(*c) for c in cells],
                   'train_cell': cell_key(*cells[train_idx])}, f, indent=2)

    print(f"\n{'='*70}\n{summary_df.to_string(index=False)}")
    print(f"\nResults saved to: {cv_dir}")
    return cv_results, summary_df


def run_per_cell(disease, backbone, entropy, models, configs):
    """One k=1 CV per grid cell — which region/t_start gives the most useful delta-A."""
    _, cf_attrs = load_data(disease, backbone)
    all_cells = resolve_cells(cf_attrs, ['all'])

    frames = []
    for region, t in all_cells:
        key = cell_key(region, t)
        print(f"\n\n########## CELL {key} ##########\n", flush=True)
        _, summary = run_cv(disease, backbone, cells_spec=(key,), train_cell=key,
                            entropy=entropy, models=models, configs=configs,
                            tag=f"per_cell/{region.replace(' ', '_')}_t{t}")
        summary.insert(0, 'cell', key)
        frames.append(summary)

    out = pd.concat(frames, ignore_index=True)
    data_subdir, _ = BACKBONE_MAP[backbone]
    path = os.path.join(RESULTS_DIR, f'{data_subdir}_corrected/{disease}',
                        'cv_results_diverse_cf', 'per_cell', 'per_cell_summary.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\n\nPer-cell summary -> {path}")
    best = out.sort_values('mean_auc', ascending=False).head(15)
    print(best[['cell', 'model', 'config', 'mean_auc', 'std_auc']].to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--disease', default='effusion')
    p.add_argument('--backbone', default='densenet', choices=list(BACKBONE_MAP))
    p.add_argument('--cells', nargs='+', default=['all'],
                   help="'all', 'general', 'masked', or explicit 'Region:t' keys.")
    p.add_argument('--train_cell', default=None,
                   help="Cell used to build the training rows (default: first selected, "
                        "which is a 'general' cell when one is selected).")
    p.add_argument('--per_cell', action='store_true',
                   help="Run one k=1 CV per grid cell and write a per-cell AUC table.")
    p.add_argument('--entropy', action='store_true',
                   help="Replace prob/cf_prob with their binary entropy in the M configs.")
    p.add_argument('--models', nargs='+', default=None)
    p.add_argument('--configs', nargs='+', default=None)
    p.add_argument('--tag', default=None, help="Result subdirectory name.")
    args = p.parse_args()

    if args.per_cell:
        run_per_cell(args.disease, args.backbone, args.entropy, args.models, args.configs)
    else:
        run_cv(args.disease, args.backbone, tuple(args.cells), args.train_cell,
               args.entropy, args.models, args.configs, args.tag)
