"""
c2_reduce_mixed_data.py
=======================
Build reduced "mixed" feature files that fuse the C0 embeddings (already inline
in the cf_ensemble prediction CSVs) with the CNN pair embeddings, compressing
both embedding blocks with PCA so a downstream fusion model has a sane number
of inputs.

What it consumes
----------------
Per outer fold i (0..N_FOLDS-1):

  * fold_{i}_predictions.csv   (cv_results_cf_ensemble/cf_{K}/TEST_entropy/)
        One row PER QUERY (8,816 rows for effusion cf_16). Carries:
          - meta:   path, patient_id, *_true, *_pred, correct, cam_path,
                    margin, cf_paths, and the C0-derived prob_* taps
          - attrs:  demographics + geometry + radiomics (the ~450 attribute cols)
          - C0 emb: emb_0 .. emb_1023   (1024-d, inline)
          - LR_/RF_/MLP_*_prob          (existing config predictions; passthrough)

  * fold_{i}_pair_embeddings.npz   (C2_cnn/{disease}/late/{config}/cf_{K}/)
        One row PER (query, cf) PAIR (141,056 rows = 8,816 x 16). Carries:
          - emb          (N_pairs x 4096 float16)
          - sample_paths (query image path, repeated K times, CF-order = CSV order)
          - cf_paths     (the specific CF path for that pair)
        The K pairs for a query are contiguous and nearest-CF-first, identical
        to the '|'-split order of the CSV's cf_paths column (verified).

Collapsing CNN pairs -> one vector per query
--------------------------------------------
The CNN pipeline TRAINS on the nearest CF (one pair per query) and only
aggregates at EVAL time, on the K *scores* (post-model), not on the embeddings.
So the object a new fusion model actually consumes per forward pass is the
NEAREST-CF embedding. We therefore keep the nearest-CF pair (the first of the 16
contiguous rows) and drop the rest. Averaging embeddings would pre-fuse in
feature space, which corresponds to nothing the CNN computes.

PCA, train/test discipline
---------------------------
Each fold_{i} file is the HELD-OUT TEST set of outer fold i; its training data
is the union of the OTHER folds. To avoid the same class of leakage fixed on
2026-07-16, PCA (and its StandardScaler) for fold i is FIT on the concatenation
of all other folds and only TRANSFORMS fold i. Nothing from fold i touches the
fitted components.

Two embedding blocks are reduced independently:
  * C0 emb   (1024-d)  -> c0pca_0 ..  c0pca_{k0-1}
  * CNN emb  (4096-d)  -> cnnpca_0 .. cnnpca_{k1-1}
PCA(n_components=0.95) chooses k per block per fold; the realised k is printed
and written into a sidecar meta JSON so the write-up can report it.

Output
------
  {out_dir}/fold_{i}_reduced.csv     one row per query:
      [all non-emb columns from the prediction CSV, verbatim]
      + c0pca_0..   + cnnpca_0..
  {out_dir}/reduce_meta.json         per-fold k0/k1, variance, row counts, paths

Everything that is NOT a C0/CNN embedding is passed through untouched, so the
attribute space, deltas-source columns, prob, cf_paths and the existing
LR/RF/MLP config probabilities remain available downstream.

Usage
-----
    python c2_reduce_mixed_data.py --disease effusion --cf_count 16 \
        --cnn_config late/dual_sal_scalars --var 0.95

    # fixed k instead of a variance target:
    python c2_reduce_mixed_data.py --disease effusion --pca_dims 128
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = "/work3/s251710/thesis_results"
N_FOLDS = 5
RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
def ensemble_dir(disease, cf_count):
    return os.path.join(
        RESULTS_DIR,
        f"C2_custom_corrected/{disease}/cv_results_cf_ensemble/cf_{cf_count}/TEST_entropy",
    )


def cnn_dir(disease, cnn_config, cf_count):
    return os.path.join(RESULTS_DIR, f"C2_cnn/{disease}/{cnn_config}/cf_{cf_count}")


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────
def load_fold_csv(disease, cf_count, fold):
    path = os.path.join(ensemble_dir(disease, cf_count), f"fold_{fold}_predictions.csv")
    return pd.read_csv(path)


def emb_column_names(df):
    """The inline C0 embedding columns, in numeric order (emb_0 .. emb_N)."""
    cols = [c for c in df.columns if c.startswith("emb_")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def load_nearest_cnn_emb(disease, cnn_config, cf_count, fold, query_paths):
    """
    Return an (n_queries x 4096) float32 array of NEAREST-CF CNN embeddings,
    row-aligned to `query_paths` (the prediction CSV's `path` order).

    The npz stores K contiguous, nearest-first pairs per query. We take the
    first pair of each query's block, then reindex to the CSV order so the
    returned matrix lines up row-for-row with the CSV.
    """
    npz_path = os.path.join(
        cnn_dir(disease, cnn_config, cf_count), f"fold_{fold}_pair_embeddings.npz"
    )
    z = np.load(npz_path, allow_pickle=True)
    emb = z["emb"]                       # (N_pairs, 4096) float16
    sample_paths = z["sample_paths"]     # (N_pairs,) object

    # First occurrence of each query = its nearest CF (blocks are contiguous &
    # nearest-first). np.unique(return_index=True) gives the first index per key.
    uniq, first_idx = np.unique(sample_paths, return_index=True)
    nearest = {p: emb[i].astype(np.float32) for p, i in zip(uniq, first_idx)}

    missing = [p for p in query_paths if p not in nearest]
    if missing:
        raise KeyError(
            f"fold {fold}: {len(missing)} query paths in the CSV have no CNN "
            f"embedding (e.g. {missing[0]}). CSV and npz are out of sync."
        )
    return np.vstack([nearest[p] for p in query_paths])


# ──────────────────────────────────────────────────────────────────────────────
# PCA fit-on-others / transform-this discipline
# ──────────────────────────────────────────────────────────────────────────────
def fit_pca_on_others(block_by_fold, test_fold, n_components):
    """
    Fit StandardScaler + PCA on every fold EXCEPT `test_fold`, return the fitted
    (scaler, pca). `n_components` is a float (variance target) or int (fixed k).
    """
    train = np.vstack([m for f, m in block_by_fold.items() if f != test_fold])
    scaler = StandardScaler().fit(train)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(scaler.transform(train))
    return scaler, pca


def transform_block(scaler, pca, mat):
    return pca.transform(scaler.transform(mat))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", default="effusion")
    ap.add_argument("--cf_count", type=int, default=16)
    ap.add_argument("--cnn_config", default="late/dual_sal_scalars",
                    help="Sub-path under C2_cnn/{disease}/ holding the npz files.")
    ap.add_argument("--var", type=float, default=0.95,
                    help="PCA variance target (used unless --pca_dims is set).")
    ap.add_argument("--pca_dims", type=int, default=None,
                    help="Fixed number of PCA components per block; overrides --var.")
    ap.add_argument("--out_dir", default=None,
                    help="Output dir (default: <ensemble_dir>/reduced_mixed_data).")
    args = ap.parse_args()

    n_components = args.pca_dims if args.pca_dims is not None else args.var
    out_dir = args.out_dir or os.path.join(
        ensemble_dir(args.disease, args.cf_count), "reduced_mixed_data"
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"disease={args.disease} cf={args.cf_count} cnn={args.cnn_config}")
    print(f"PCA n_components={n_components}  ->  {out_dir}\n")

    # 1) Load every fold once: CSV, its C0-emb matrix, its nearest CNN-emb matrix.
    csvs, c0_by_fold, cnn_by_fold, emb_cols = {}, {}, {}, None
    for i in range(N_FOLDS):
        df = load_fold_csv(args.disease, args.cf_count, i)
        if emb_cols is None:
            emb_cols = emb_column_names(df)
        c0 = df[emb_cols].values.astype(np.float32)
        cnn = load_nearest_cnn_emb(
            args.disease, args.cnn_config, args.cf_count, i, df["path"].tolist()
        )
        csvs[i], c0_by_fold[i], cnn_by_fold[i] = df, c0, cnn
        print(f"  fold {i}: {len(df)} queries | C0 {c0.shape} | CNN {cnn.shape}")

    keep_cols = [c for c in csvs[0].columns if c not in set(emb_cols)]
    print(f"\nC0 emb block: {len(emb_cols)} dims | passthrough columns: {len(keep_cols)}\n")

    # 2) For each fold-as-test: fit PCA on the other folds, transform this fold.
    meta = {"disease": args.disease, "cf_count": args.cf_count,
            "cnn_config": args.cnn_config, "n_components": n_components,
            "c0_emb_dims": len(emb_cols), "cnn_emb_dims": cnn_by_fold[0].shape[1],
            "folds": {}}

    for i in range(N_FOLDS):
        c0_scaler, c0_pca = fit_pca_on_others(c0_by_fold, i, n_components)
        cnn_scaler, cnn_pca = fit_pca_on_others(cnn_by_fold, i, n_components)

        c0_red = transform_block(c0_scaler, c0_pca, c0_by_fold[i])
        cnn_red = transform_block(cnn_scaler, cnn_pca, cnn_by_fold[i])
        k0, k1 = c0_red.shape[1], cnn_red.shape[1]

        out = csvs[i][keep_cols].reset_index(drop=True)
        c0_df = pd.DataFrame(c0_red, columns=[f"c0pca_{j}" for j in range(k0)])
        cnn_df = pd.DataFrame(cnn_red, columns=[f"cnnpca_{j}" for j in range(k1)])
        out = pd.concat([out, c0_df, cnn_df], axis=1)

        out_path = os.path.join(out_dir, f"fold_{i}_reduced.csv")
        out.to_csv(out_path, index=False)

        meta["folds"][i] = {
            "rows": int(len(out)),
            "c0_k": int(k0), "c0_var_retained": float(c0_pca.explained_variance_ratio_.sum()),
            "cnn_k": int(k1), "cnn_var_retained": float(cnn_pca.explained_variance_ratio_.sum()),
            "out_path": out_path,
        }
        print(f"  fold {i}: C0 {len(emb_cols)}->{k0} "
              f"({c0_pca.explained_variance_ratio_.sum():.3f} var) | "
              f"CNN {cnn_by_fold[i].shape[1]}->{k1} "
              f"({cnn_pca.explained_variance_ratio_.sum():.3f} var) | "
              f"{out.shape[1]} cols -> {os.path.basename(out_path)}")

    with open(os.path.join(out_dir, "reduce_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nWrote {N_FOLDS} reduced CSVs + reduce_meta.json to {out_dir}")


if __name__ == "__main__":
    main()
