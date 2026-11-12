"""
Cell ablation over the 15-cell diffusion-CF fingerprint.

For each (region, t_start) cell, drop that cell's features from the fingerprint and
re-run the 5-fold CV to see how the AUC changes. Mirrors the notebook cells in
notebooks/NewDiffusionCFAttributes.ipynb (feature build -> wide pivot -> CV), but
runs headless so it can go to the cluster.

Folds are StratifiedGroupKFold on patient_id, same seed as the notebook, so the
full-fingerprint reference AUC is directly comparable to the ablated ones.
"""

import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

RESULTS_DIR = "/work3/s251710/thesis_results/diverse_cf"
CONFIG_JSON = "/zhome/d0/a/221493/thesis/results/diverse_cf/config.json"
FEATURES_PARQUET = os.path.join(RESULTS_DIR, "cf_features.parquet")
OUT_CSV = os.path.join(RESULTS_DIR, "cell_ablation_aucs.csv")

N_SPLITS = 5
MAX_ITER = 300
SEED = 42
N_JOBS = int(os.environ.get("LSB_DJOB_NUMPROC", "8"))

ID_COLS = {'path', 'cf_id', 'patient_id', 'region', 't_start', 'quadrant',
           'pred', 'true', 'correct', 'mask_empty'}
IMG_LEVEL = ['boundary_dist', 'orig_prob']


def build_matrix():
    with open(CONFIG_JSON) as f:
        config = json.load(f)

    cf_all = pd.read_parquet(FEATURES_PARQUET)
    ok = cf_all[cf_all['mask_empty'] == 0]

    feats = [c for c in ok.columns
             if c not in ID_COLS and c != 'boundary_dist'
             and pd.api.types.is_numeric_dtype(ok[c])]
    cell_feats = [c for c in feats if c not in IMG_LEVEL]

    wide = ok.pivot_table(index='path', columns=['region', 't_start'],
                          values=cell_feats, aggfunc='first')
    wide.columns = [f"{f}__{r.replace(' ', '_')}__t{t}" for f, r, t in wide.columns]

    meta = (ok.groupby('path')
              .agg(patient_id=('patient_id', 'first'), correct=('correct', 'first'),
                   boundary_dist=('boundary_dist', 'first'),
                   orig_prob=('orig_prob', 'first'), quadrant=('quadrant', 'first')))
    X_all = meta.join(wide)
    return config, ok, X_all


def cv_auc(X, y, groups, cv):
    aucs = []
    for tr, te in cv.split(X, y, groups):
        m = HistGradientBoostingClassifier(max_iter=MAX_ITER,
                                           random_state=0).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return aucs


def main():
    config, ok, X_all = build_matrix()
    print(f"{X_all.shape} | images: {len(X_all)} | patients: {X_all.patient_id.nunique()}",
          flush=True)

    y = X_all['correct'].values
    groups = X_all['patient_id'].values
    fp_cols = [c for c in X_all.columns if '__' in c]
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Cells enumerated from the DATA, not config['regions'] — config lists only the
    # four masked regions, so driving the loop off it silently skips the three
    # `general` cells (the notebook version of this loop has that gap).
    regions = [r for r in config['regions'] if r in set(ok['region'])]
    regions += [r for r in sorted(ok['region'].unique()) if r not in regions]
    cells = []
    for region in regions:
        for t in sorted(ok.loc[ok['region'] == region, 't_start'].unique()):
            cells.append(f"__{region.replace(' ', '_')}__t{int(t)}")

    full = cv_auc(X_all[fp_cols].values, y, groups, cv)
    print(f"full fingerprint ({len(fp_cols)} cols)  AUC {np.mean(full):.4f} "
          f"± {np.std(full):.4f}   {np.round(full, 4)}", flush=True)

    def run(cell):
        cols = [c for c in fp_cols if not c.endswith(cell)]
        return cell, len(cols), cv_auc(X_all[cols].values, y, groups, cv)

    print(f"ablating {len(cells)} cells on {N_JOBS} workers ...", flush=True)
    out = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(run)(c) for c in cells)

    rows = [{'cell': 'FULL', 'n_cols': len(fp_cols), 'auc_mean': np.mean(full),
             'auc_std': np.std(full),
             **{f'fold{i}': a for i, a in enumerate(full)}}]
    for cell, n_cols, aucs in out:
        rows.append({'cell': cell, 'n_cols': n_cols, 'auc_mean': np.mean(aucs),
                     'auc_std': np.std(aucs),
                     **{f'fold{i}': a for i, a in enumerate(aucs)}})
    df = pd.DataFrame(rows)
    df['delta_vs_full'] = df['auc_mean'] - np.mean(full)

    print("\nAblation results (removing one cell at a time):")
    for r in df.itertuples():
        print(f"{r.cell:<30} AUC {r.auc_mean:.4f} ± {r.auc_std:.4f}"
              f"   delta {r.delta_vs_full:+.4f}")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
