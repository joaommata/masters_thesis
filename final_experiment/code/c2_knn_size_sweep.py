# c2_knn_size_sweep.py
"""
How much data does KNN retrieval need to match ONE diffusion counterfactual?

The deployment question: a site with N images of its own can either retrieve
counterfactuals from those N (KNN), or generate one per query (diffusion). KNN
gets better with N -- closer neighbours -- and with K -- averaging more of them.
Diffusion does not depend on N at all. This sweeps N x K for the KNN arm and
puts the diffusion K=1 result beside it as the target line.

Design: CV for the test side, subsampling for the training side
-----------------------------------------------------------------
    test    the 5 test folds of the 20k diffusion subset, read from the
            diffusion_hml_sub20k pairing -- identical to knn_correct_cf_sub20k's.
            Every N, K, seed and arm is scored on exactly these ~20k rows, so all
            comparisons are paired and the test set does not shrink with N.
    train   for fold i, the pool is ALL of c2_data.csv minus every patient of
            test fold i (~90k rows). A seed shuffles the pool's PATIENTS once and
            N takes a prefix of that order, so the subsamples are patient-whole
            and nested (N=1k is inside N=2k is inside ...).
    KNN     the N training rows are both the C2 training set and the only rows
            counterfactuals can be retrieved from -- what a site of size N has.

Inside one (fold, seed, N) everything is what c2_train_attributes.py does on a
cached fold, with the pairing computed here instead of read from disk:

  * FoldContext on (train rows + test fold): ratios, NaN fill, scaler fit on
    the N train rows only;
  * correct_cf routing: pred=1 -> nearest correctly-classified pred=0 train row,
    pred=0 -> nearest correctly-classified pred=1 train row; L1 in the SCALED
    attribute space -- the same space the delta features are built in, so the
    retrieval cannot drift from the features (see c2_scce.RetrievalSpace for why
    the cached 20k pairing does not have this property);
  * train on the nearest counterfactual only, score each of the K test slots,
    average the first K probabilities. One run at --k-max gives every K <= k-max.

The neighbour search runs on the GPU (torch.cdist, p=1). The sklearn brute-force
search the cached folds were built with is ~2.3 h per fold at full size.

Diffusion arm (--diffusion-ref): the same MLPs, same seeds, on the cached
diffusion_hml_sub20k cf_1 folds -- the full 16k train side, plus nested patient-prefix
subsamples of it at --diffusion-sizes, so diffusion is a curve over N too. At the
full train side, seed 0 uses RANDOM_SEED and so reproduces
C2_final_results/<d>/diffusion_hml_sub20k/cf_1/attributes up to float nondeterminism (effusion fold 0 I+C: 0.7717 on CPU vs 0.7709 on GPU).

    python c2_knn_size_sweep.py --disease effusion --folds 0
    python c2_knn_size_sweep.py --disease effusion --diffusion-ref
    python c2_knn_size_sweep.py --disease effusion --summarise

Output: C2_final_results/<disease>/knn_size_sweep/
    knn/fold_<i>/seed_<s>/n_<N>.npz|json    per-run test probs, distances, AUCs
    diffusion_ref/fold_<i>/seed_<s>.npz|json         full 16k train side
    diffusion_ref/fold_<i>/seed_<s>/n_<N>.npz|json   subsampled
    sweep_pooled.csv                         pooled out-of-fold AUROC per
                                             (arm, seed, N, config, K)
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import (                                          # noqa: E402
    RESULTS_DIR, RANDOM_SEED, N_FOLDS, load_c2_data, load_folds, pairing_path,
    FoldContext, materialize, slot_pairing, _row_ids,
)
from c2_cf_sources import build_cf_source                       # noqa: E402
from c2_train_attributes import (                               # noqa: E402
    feature_columns, config_matrix, fit, CF_DEPENDENT, MODEL, DEVICE,
)
from c2_msp import msp_score                                    # noqa: E402

DIFFUSION_SOURCE = "diffusion_hml_sub20k"
CONFIGS = ["I", "I+C", "I+C+D"]
DEFAULT_SIZES = "1000,2000,4000,8000,16000,32000,all"
K_LIST = [1, 2, 3, 5, 10, 15, 20]


def out_root(disease):
    return os.path.join(RESULTS_DIR, "C2_final_results", disease, "knn_size_sweep")


def c0_threshold(disease):
    p = os.path.join(RESULTS_DIR, "C0_final", "multilabel_ignore", disease, "threshold.txt")
    return float(open(p).read().strip())


# ══════════════════════════════════════════════════════════════════════════════
# SPLITS
# ══════════════════════════════════════════════════════════════════════════════

def diffusion_test_paths(disease, fold):
    """Test paths of one fold of the 20k subset -- the fixed evaluation set."""
    p = pairing_path(disease, DIFFUSION_SOURCE, 1, fold)
    pair = pd.read_parquet(p, columns=["split", "path"])
    return pair.loc[pair["split"] == "test", "path"].values


def patient_order(pool, seed):
    """The pool sorted into one random patient order, and the cumulative row count
    at each patient boundary. N rows = the shortest patient prefix holding >= N."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool["patient_id"].unique())
    rank = pd.Series(np.arange(len(perm)), index=perm)
    ordered = pool.iloc[np.argsort(rank.loc[pool["patient_id"]].values,
                                   kind="stable")].reset_index(drop=True)
    cum = np.cumsum(ordered.groupby("patient_id", sort=False).size().values)
    return ordered, cum


def resolve_sizes(spec, pool_rows):
    """'1000,...,all' -> sorted unique row targets, capped at the pool."""
    out = []
    for s in spec.split(","):
        s = s.strip()
        out.append(pool_rows if s == "all" else min(int(s), pool_rows))
    return sorted(set(out))


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def knn_l1(Q, P, k, chunk_elems=200_000_000):
    """k nearest rows of P for each row of Q under L1, nearest first.

    Chunked over Q so the (chunk x |P|) distance block stays ~800 MB in float32.
    """
    P_t = torch.as_tensor(P, dtype=torch.float32, device=DEVICE)
    rows = max(1, chunk_elems // max(len(P), 1))
    dist = np.empty((len(Q), k), dtype=np.float32)
    idx = np.empty((len(Q), k), dtype=np.int64)
    for s0 in range(0, len(Q), rows):
        q = torch.as_tensor(Q[s0:s0 + rows], dtype=torch.float32, device=DEVICE)
        d, i = torch.topk(torch.cdist(q, P_t, p=1), k, dim=1, largest=False, sorted=True)
        dist[s0:s0 + rows] = d.cpu().numpy()
        idx[s0:s0 + rows] = i.cpu().numpy()
    del P_t
    return dist, idx


def pair_correct_cf(ctx, q_rows, tr_rows, disease, k):
    """correct_cf pairing of q_rows against the train rows, in ctx's scaled space.

    Returns (pairing frame in c2_folds' PAIRING shape, L1 distances [n, k]).
    """
    lut = ctx.lut
    pred = lut[f"{disease}_pred"].values.astype(int)
    correct = lut["correct"].values.astype(int)
    prob = lut[f"{disease}_prob"].values.astype(float)
    paths = lut.index.values
    X = ctx.attr_scaled_all

    pools = {1: tr_rows[(pred[tr_rows] == 0) & (correct[tr_rows] == 1)],   # pred=1 -> TN
             0: tr_rows[(pred[tr_rows] == 1) & (correct[tr_rows] == 1)]}   # pred=0 -> TP
    cf = np.empty((len(q_rows), k), dtype=np.int64)
    dist = np.empty((len(q_rows), k), dtype=np.float32)
    q_pred = pred[q_rows]
    for p, pool in pools.items():
        sel = q_pred == p
        if sel.any():
            d, i = knn_l1(X[q_rows[sel]], X[pool], k)
            cf[sel], dist[sel] = pool[i], d

    pair = pd.DataFrame({
        "path": paths[q_rows],
        "cf_paths": ["|".join(paths[r]) for r in cf],
        "cf_probs": ["|".join(repr(float(v)) for v in prob[r]) for r in cf],
        "cf_prob": prob[cf].mean(axis=1),
    })
    return pair, dist


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / SCORE
# ══════════════════════════════════════════════════════════════════════════════

def predict(model, X):
    net, scaler = model
    with torch.no_grad():
        x = torch.tensor(scaler.transform(X), dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(net(x).squeeze(1)).cpu().numpy().astype(np.float64)


def fit_configs(train_df, disease, configs, seed):
    cols = feature_columns(train_df, disease)
    y = train_df["correct"].values
    groups = train_df["patient_id"].values
    models = {}
    for cfg in configs:
        models[cfg] = fit(config_matrix(train_df, cols, cfg), y, groups,
                          seed=RANDOM_SEED + seed)
    if DEVICE != "cpu":
        torch.cuda.empty_cache()
    return cols, models


def score_slots(slot_frames, cols, models):
    """{config: [n_slots_used, n_test]} -- a K-invariant config is scored on slot 0 only."""
    out = {cfg: [] for cfg in models}
    for j, sl in enumerate(slot_frames):
        for cfg, m in models.items():
            if j == 0 or cfg in CF_DEPENDENT:
                out[cfg].append(predict(m, config_matrix(sl, cols, cfg)))
    return {cfg: np.stack(v) for cfg, v in out.items()}


def auc_by_k(y, probs, k_max):
    rows = []
    for cfg, P in probs.items():
        for k in K_LIST:
            if k > k_max or (k > 1 and cfg not in CF_DEPENDENT):
                continue
            rows.append({"config": cfg, "k": k,
                         "auc": float(roc_auc_score(y, P[:k].mean(axis=0)))})
    return rows


def save(base, y, test_df, disease, t, probs, extra_arrays, meta):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    np.savez_compressed(base + ".npz",
                        paths=test_df["path"].values.astype(str),
                        labels=y.astype(np.int8),
                        c0_prob=test_df[f"{disease}_prob"].values.astype(np.float64),
                        **{f"{MODEL}_{c}": P.astype(np.float32) for c, P in probs.items()},
                        **extra_arrays)
    meta["msp_auc"] = float(roc_auc_score(
        y, msp_score(test_df[f"{disease}_prob"].values, t, "threshold")))
    # json last: its presence marks the run complete.
    with open(base + ".json", "w") as fh:
        json.dump(meta, fh, indent=1)


# ══════════════════════════════════════════════════════════════════════════════
# KNN ARM
# ══════════════════════════════════════════════════════════════════════════════

def run_knn_fold(full, disease, fold, args, t):
    test_paths = diffusion_test_paths(disease, fold)
    test_df = full[full["path"].isin(set(test_paths))].reset_index(drop=True)
    assert len(test_df) == len(test_paths), f"fold {fold}: test paths missing from c2_data"
    test_pat = set(test_df["patient_id"])
    pool = full[~full["patient_id"].isin(test_pat)].reset_index(drop=True)
    sizes = resolve_sizes(args.sizes, len(pool))
    print(f"\n{'='*72}\n  FOLD {fold}: test {len(test_df):,} | pool {len(pool):,} rows, "
          f"{pool['patient_id'].nunique():,} patients\n  sizes: {sizes}\n{'='*72}")

    for seed in range(args.seeds):
        ordered, cum = patient_order(pool, seed=1000 * fold + seed)
        for n in sizes:
            full_pool = n >= len(pool)
            # The full pool is the same rows for every seed; only the MLP seed would
            # differ, which --seeds-full controls separately.
            if full_pool and seed >= args.seeds_full:
                continue
            tag = "all" if full_pool else str(n)
            base = os.path.join(out_root(disease), "knn", f"fold_{fold}",
                                f"seed_{seed}", f"n_{tag}")
            if os.path.exists(base + ".json") and not args.overwrite:
                print(f"  seed {seed} n={tag}: done, skipping")
                continue
            t0 = time.time()
            n_rows = int(cum[min(np.searchsorted(cum, n), len(cum) - 1)])
            train_src = ordered.iloc[:n_rows]

            ctx = FoldContext(pd.concat([train_src, test_df], ignore_index=True),
                              train_src[["path"]], disease)
            tr_rows = _row_ids(train_src["path"].values, ctx.row_of, "train")
            te_rows = _row_ids(test_df["path"].values, ctx.row_of, "test")
            pred, corr = (ctx.lut[f"{disease}_pred"].values[tr_rows].astype(int),
                          ctx.lut["correct"].values[tr_rows].astype(int))
            n_tp = int(((pred == 1) & (corr == 1)).sum())
            n_tn = int(((pred == 0) & (corr == 1)).sum())
            k_test = min(args.k_max, n_tp, n_tn)
            if k_test < 1:
                print(f"  seed {seed} n={tag}: empty TP or TN pool, skipping")
                continue

            pair_tr, d_tr = pair_correct_cf(ctx, tr_rows, tr_rows, disease, 1)
            pair_te, d_te = pair_correct_cf(ctx, te_rows, tr_rows, disease, k_test)
            t_pair = time.time() - t0

            train_df = materialize(ctx, slot_pairing(pair_tr, 0))
            cols, models = fit_configs(train_df, disease, args.configs, seed)
            slots = (materialize(ctx, slot_pairing(pair_te, j)) for j in range(k_test))
            probs = score_slots(slots, cols, models)
            y = test_df.set_index("path").loc[pair_te["path"], "correct"].values
            aucs = auc_by_k(y, probs, k_test)

            n_dim = len(ctx.rel)
            meta = {"arm": "knn", "disease": disease, "fold": fold, "seed": seed,
                    "n_target": tag, "n_train": int(n_rows),
                    "n_train_errors": int((train_df["correct"] == 0).sum()),
                    "pool_tp": n_tp, "pool_tn": n_tn, "k_test": int(k_test),
                    "n_test": int(len(y)), "n_dim": n_dim,
                    "train_nn1_l1_per_dim": float(d_tr[:, 0].mean() / n_dim),
                    "test_nn1_l1_per_dim": float(d_te[:, 0].mean() / n_dim),
                    "test_nnK_l1_per_dim": float(d_te[:, -1].mean() / n_dim),
                    "aucs": aucs, "seconds_pairing": round(t_pair, 1),
                    "seconds_total": round(time.time() - t0, 1)}
            test_out = test_df.set_index("path").loc[pair_te["path"]].reset_index()
            save(base, y, test_out, disease, t, probs,
                 {"test_l1": d_te.astype(np.float32),
                  "cf_paths": pair_te["cf_paths"].values.astype(str)}, meta)

            best = {r["config"]: r["auc"] for r in aucs if r["k"] == 1}
            print(f"  seed {seed} n={tag:>6} ({n_rows:,} rows, TP {n_tp:,} / TN {n_tn:,}) "
                  f"| K1 " + " ".join(f"{c}={a:.4f}" for c, a in best.items())
                  + f" | MSP={meta['msp_auc']:.4f} | nn1/dim={meta['test_nn1_l1_per_dim']:.3f}"
                  + f" [{meta['seconds_total']:.0f}s]", flush=True)
            del ctx, train_df, models


# ══════════════════════════════════════════════════════════════════════════════
# DIFFUSION REFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_diffusion_ref(disease, args, t):
    src = build_cf_source("diffusion", subset="20k")
    for fold in load_folds(disease, src, 1, build_if_missing=False):
        train_df = fold.train
        slot0 = fold.test_slot(0)
        y = slot0["correct"].values
        n_dim = len(fold.ctx.rel)
        delta_cols = [f"delta_{c}" for c in fold.ctx.rel]
        l1_test = np.abs(slot0[delta_cols].values).sum(axis=1)
        train_df = train_df.assign(_l1=np.abs(train_df[delta_cols].values).sum(axis=1))
        sizes = [n for n in resolve_sizes(args.diffusion_sizes, len(train_df))
                 if n < len(train_df)]
        for seed in range(args.seeds):
            # Same nested patient-prefix subsampling as the KNN arm, drawn from the
            # diffusion train folds. The full train fold keeps its original path and
            # n_target "20k" (the 16k train side of the 20k subset).
            ordered, cum = patient_order(train_df, seed=1000 * fold.index + seed)
            plan = [(None, os.path.join(out_root(disease), "diffusion_ref",
                                        f"fold_{fold.index}", f"seed_{seed}"))]
            plan += [(n, os.path.join(out_root(disease), "diffusion_ref",
                                      f"fold_{fold.index}", f"seed_{seed}", f"n_{n}"))
                     for n in sizes]
            for n, base in plan:
                tag = "20k" if n is None else str(n)
                if os.path.exists(base + ".json") and not args.overwrite:
                    print(f"  diffusion fold {fold.index} seed {seed} n={tag}: done, skipping")
                    continue
                t0 = time.time()
                sub = (train_df if n is None else
                       ordered.iloc[:int(cum[min(np.searchsorted(cum, n), len(cum) - 1)])])
                cols, models = fit_configs(sub.drop(columns="_l1"), disease,
                                           args.configs, seed)
                probs = score_slots([slot0], cols, models)
                aucs = auc_by_k(y, probs, 1)
                meta = {"arm": "diffusion", "disease": disease, "fold": fold.index,
                        "seed": seed, "n_target": tag, "n_train": int(len(sub)),
                        "n_train_errors": int((sub["correct"] == 0).sum()),
                        "k_test": 1, "n_test": int(len(y)), "n_dim": n_dim,
                        "train_nn1_l1_per_dim": float(sub["_l1"].mean() / n_dim),
                        "test_nn1_l1_per_dim": float(l1_test.mean() / n_dim),
                        "aucs": aucs, "seconds_total": round(time.time() - t0, 1)}
                save(base, y, slot0, disease, t, probs,
                     {"test_l1": l1_test.astype(np.float32)[:, None]}, meta)
                print(f"  diffusion fold {fold.index} seed {seed} n={tag:>5} "
                      f"({len(sub):,} rows) | "
                      + " ".join(f"{r['config']}={r['auc']:.4f}" for r in aucs)
                      + f" | MSP={meta['msp_auc']:.4f} [{meta['seconds_total']:.0f}s]",
                      flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def summarise(disease):
    """Pool the 5 test folds per (arm, seed, N) into one out-of-fold AUROC.

    Only (arm, seed, N) cells with all folds present are pooled; anything partial
    is reported and left out rather than scored on a smaller test set.
    """
    root = out_root(disease)
    runs = []
    for j in glob.glob(os.path.join(root, "*", "fold_*", "**", "*.json"), recursive=True):
        meta = json.load(open(j))
        meta["npz"] = j[:-5] + ".npz"
        runs.append(meta)
    if not runs:
        raise SystemExit(f"no runs under {root}")
    meta = pd.DataFrame(runs)

    rows = []
    for (arm, seed, tag), g in meta.groupby(["arm", "seed", "n_target"]):
        if g["fold"].nunique() < N_FOLDS:
            print(f"  incomplete: {arm} seed {seed} n={tag} has folds "
                  f"{sorted(g['fold'])}")
            continue
        data = [np.load(p) for p in g["npz"]]
        y = np.concatenate([d["labels"] for d in data])
        k_max = int(g["k_test"].min())
        t = c0_threshold(disease)
        msp = msp_score(np.concatenate([d["c0_prob"] for d in data]), t, "threshold")
        common = {"arm": arm, "seed": seed, "n_target": tag,
                  "n_train_mean": float(g["n_train"].mean()),
                  "n_train_errors_mean": float(g["n_train_errors"].mean()),
                  "test_nn1_l1_per_dim": float(g["test_nn1_l1_per_dim"].mean()),
                  "msp_auc": float(roc_auc_score(y, msp)), "n_test": int(len(y))}
        for cfg in CONFIGS:
            key = f"{MODEL}_{cfg}"
            if key not in data[0]:
                continue
            P = np.concatenate([d[key] for d in data], axis=1)
            for k in K_LIST:
                if k > min(k_max, P.shape[0]):
                    continue
                rows.append({**common, "config": cfg, "k": k,
                             "auc": float(roc_auc_score(y, P[:k].mean(axis=0)))})
    res = pd.DataFrame(rows)
    out = os.path.join(root, "sweep_pooled.csv")
    res.to_csv(out, index=False)

    # Console view: mean over seeds, the full-K grid for the CF configs.
    agg = (res.groupby(["arm", "n_target", "config", "k"])
              .agg(auc=("auc", "mean"), sd=("auc", "std"), n_seeds=("seed", "nunique"),
                   n_train=("n_train_mean", "mean"))
              .reset_index())
    agg["n_sort"] = agg["n_train"]
    print(f"\n{'='*72}\n  {disease}: pooled OOF AUROC, mean over seeds\n{'='*72}")
    print(f"  MSP (threshold): {res['msp_auc'].iloc[0]:.4f}")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(agg.sort_values(["arm", "n_sort", "config", "k"])
                 .drop(columns="n_sort").to_string(index=False, float_format="%.4f"))
    print(f"\n  wrote {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--sizes", default=DEFAULT_SIZES,
                    help="comma-separated training-set sizes; 'all' = the whole pool")
    ap.add_argument("--seeds", type=int, default=5,
                    help="subsample (and MLP) seeds per size")
    ap.add_argument("--seeds-full", type=int, default=1,
                    help="seeds at the full pool, where only the MLP seed varies")
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--diffusion-ref", action="store_true",
                    help="train the diffusion K=1 reference instead of the KNN sweep")
    ap.add_argument("--diffusion-sizes", default="1000,2000,4000,8000",
                    help="subsampled train sizes for the diffusion arm; the full "
                         "train fold (~16k) is always run as well")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.configs = [c.strip() for c in args.configs.split(",")]
    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; expected a subset of {CONFIGS}")

    if args.summarise:
        summarise(args.disease)
        return

    t = c0_threshold(args.disease)
    print(f"  {args.disease} | device {DEVICE} | configs {args.configs} | "
          f"C0 threshold {t:.4f}")
    if args.diffusion_ref:
        run_diffusion_ref(args.disease, args, t)
        return

    full = load_c2_data(args.disease)
    print(f"  c2_data: {len(full):,} rows, {full['patient_id'].nunique():,} patients")
    for fold in [int(f) for f in args.folds.split(",")]:
        run_knn_fold(full, args.disease, fold, args, t)


if __name__ == "__main__":
    main()
