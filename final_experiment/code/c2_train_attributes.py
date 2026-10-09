# c2_train_attributes.py
"""
Stage 5b: attribute-based C2 models on the cached folds -- the block ablation.

Every feature belongs to one of three blocks (the design of c2_block_ablation.ipynb):

    I   Input   attrs of x, C0 emb of x
    C   CF      attrs of the CF, C0 emb of the CF
    D   Delta   scaled attrs(x) - scaled attrs(CF), emb(x) - emb(CF)

and the configs are combinations of blocks:

    I       query only -- no counterfactual, K-invariant
    C       counterfactual only
    D       difference only
    I+C
    I+C+D

Probability-free: C0's prob(x), prob(CF) and their difference are in NO config, so
what is measured is what the attributes and embeddings carry on their own. MSP is
the baseline, and it is scored downstream (c2_msp.py), not here.

With probability (--with-prob): the final-system stage, run after the ablation has
picked its winner. A fourth block

    P   Prob    [p_xi, H(p_xi), p_cf, H(p_cf), p_xi - p_cf]

-- exactly the five C0 scalars dual_sal_scalars gets in c2_train_images.py, per CF
slot -- is appended to the winning config, giving I+C+D+P. It writes to
.../cf_<K>/attributes_prob, never to .../attributes, so the probability-free
ablation and everything that reads it are untouched.

    python c2_train_attributes.py --disease effusion --cf-source knn --cf-count 10 --with-prob

One model: a torch MLP, 64-32-16 ReLU, Adam 1e-3, L2 1e-4, batch 200,
class-balanced BCE (pos_weight = n_neg / n_pos), early stopping on the AUC of a
patient-grouped 10% val slice of train (patience 10, max 500 epochs). Runs on the
GPU when there is one.

USAGE:

    python c2_train_attributes.py --disease effusion --cf-source knn --cf-count 10

More than one counterfactual per query
--------------------------------------
Train on the NEAREST counterfactual only (slot 0), predict each of the K test
slots, average the K probabilities. Slot j's score does not depend on K, so one
K=10 run also holds K=5 and K=1 exactly: c2_derive_k.py averages a prefix of the
per-slot probabilities written to fold_<i>_slot_probs.npz. `I` has no CF block and
is scored once (k_used=1).

Results land in .../<source>/cf_<K>/attributes, in the same layout the ensemble,
derive and bootstrap code read. The B/M/MCF ladder this replaced is in
archive/c2_train_attributes_ladder.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import (                                          # noqa: E402
    RESULTS_DIR, RANDOM_SEED, load_folds, add_cf_args, cf_source_from_args, fold_dir,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code", "c2"))
from c2_feature_spec import C0_DERIVED_COLS, RSNA_META_COLS     # noqa: E402

CONFIGS = ["I", "C", "D", "I+C", "I+C+D"]
# --with-prob configs: the ablation winner plus the C0 scalars. Kept out of CONFIGS so
# the probability-free ablation can never pick one up by default.
PROB_CONFIGS = ["I+C+D+P"]
MODEL = "MLP"
# Configs whose values depend on WHICH counterfactual the query was paired with.
CF_DEPENDENT = {c for c in CONFIGS + PROB_CONFIGS if c != "I"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

def feature_columns(df, disease):
    """The column list behind each block, derived once per fold from the train frame.

    The attribute space is whatever is left after removing the named metadata and
    the prefixed groups. That makes `meta_cols` load-bearing: a column that
    belongs to C0 but is not named here silently becomes an attribute and inflates
    every attribute config. C0_DERIVED_COLS exists because `margin` and the seven
    SDN prob_* columns did exactly that.

    Derived from the TRAIN frame and then applied by name to every test slot, so
    a slot cannot silently reorder or drop a column.
    """
    meta_cols = {f"{disease}_prob", f"{disease}_pred", f"{disease}_true",
                 "correct", "path", "cam_path", "patient_id",
                 "cf_prob", "cf_paths", "cf_probs", "delta_prob",
                 "label_raw", "certain",
                 *C0_DERIVED_COLS, *RSNA_META_COLS}

    cols = {
        "delta":   [c for c in df.columns
                    if c.startswith("delta_") and c != "delta_prob"],
        "emb":     [c for c in df.columns if c.startswith("emb_")],
        "cf_attr": [c for c in df.columns if c.startswith("cf_attr_")],
        "cf_emb":  [c for c in df.columns if c.startswith("cf_emb_")],
        "attr":    [c for c in df.columns
                    if c not in meta_cols
                    and not c.startswith("delta_")
                    and not c.startswith("emb_")
                    and not c.startswith("cf_attr_")
                    and not c.startswith("cf_emb_")],
    }

    non_numeric = [c for c in cols["attr"]
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"non-numeric columns reached the attribute space: {non_numeric[:5]}\n"
            f"They must be added to meta_cols -- a string column crashes "
            f"StandardScaler the way cam_path did.")
    # Not a feature list: the query-probability column the P block reads.
    cols["prob"] = f"{disease}_prob"
    return cols


def _entropy(p):
    """Binary entropy, clipped as in c2_train_images._entropy."""
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def block(df, cols, b):
    """The features of one block. Only P carries C0's probability."""
    if b == "P":
        # Same five scalars, same order, as c2_train_images._make_scalars. A slot
        # frame holds ONE counterfactual, so cf_prob is that CF's own probability.
        p_xi = df[cols["prob"]].values.astype(np.float64)
        p_cf = df["cf_prob"].values.astype(np.float64)
        return [np.stack([p_xi, _entropy(p_xi), p_cf, _entropy(p_cf), p_xi - p_cf],
                         axis=1)]
    if b == "I":
        return [df[cols["attr"]].values, df[cols["emb"]].values]
    if b == "C":
        return [df[cols["cf_attr"]].values, df[cols["cf_emb"]].values]
    if b == "D":
        return [df[cols["delta"]].values,
                df[cols["emb"]].values - df[cols["cf_emb"]].values]
    raise ValueError(f"unknown block {b!r}")


def config_matrix(df, cols, config):
    """The design matrix of one config, e.g. "I+C" -> [attr, emb, cf_attr, cf_emb]."""
    return np.hstack([f for b in config.split("+") for f in block(df, cols, b)]
                     ).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def fit_predict(X_tr, y_tr, groups, X_tests, epochs=500, patience=10, bs=200,
                seed=RANDOM_SEED):
    """Fit the MLP on X_tr and return P(correct) for each matrix in X_tests."""
    net, scaler = fit(X_tr, y_tr, groups, epochs, patience, bs, seed)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return [torch.sigmoid(net(to_t(scaler.transform(Xt))).squeeze(1))
                .cpu().numpy().astype(np.float64) for Xt in X_tests]


def fit(X_tr, y_tr, groups, epochs=500, patience=10, bs=200, seed=RANDOM_SEED):
    """Fit the MLP on X_tr; return the best net (eval mode) and its input scaler.

    The val split is 10% of the training rows, grouped by patient, so no patient
    is on both sides of it.
    """
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X_tr)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    X, y = to_t(scaler.transform(X_tr)), to_t(y_tr)
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
                    .split(X_tr, y_tr, groups))
    y_val = y_tr[val]
    fit, val = torch.tensor(fit, device=DEVICE), torch.tensor(val, device=DEVICE)

    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor((y_tr == 0).sum() / (y_tr == 1).sum(),
                              dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best, best_state, wait = -1, None, 0
    for _ in range(epochs):
        net.train()
        for idx in fit[torch.randperm(len(fit), device=DEVICE)].split(bs):
            opt.zero_grad()
            loss_fn(net(X[idx]).squeeze(1), y[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            score = roc_auc_score(y_val, net(X[val]).squeeze(1).cpu().numpy())
        if score > best + 1e-4:
            best, wait = score, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, scaler


def score_cf_anchor(train_df, test_slots):
    """Baseline: score a (query, CF) pair by the mean correctness of training rows
    that retrieved the same counterfactual. Returns the per-slot matrix, or None
    if the counterfactuals are not shared between queries (a generated source:
    every anchor group would have size one and the AUC would be a silent 0.5).
    """
    anchor = {}
    for a, corr in zip(train_df["cf_paths"], train_df["correct"]):
        anchor.setdefault(str(a), []).append(corr)
    if len(anchor) == len(train_df) and len(train_df) > 1:
        return None
    anchor = {a: float(np.mean(v)) for a, v in anchor.items()}
    return np.stack([np.array([anchor.get(str(a), 0.0) for a in slot["cf_paths"]])
                     for slot in test_slots])


# ══════════════════════════════════════════════════════════════════════════════
# CV
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    src = cf_source_from_args(args)
    allowed = PROB_CONFIGS if args.with_prob else CONFIGS
    configs = [c.strip() for c in args.configs.split(",")] if args.configs else allowed
    unknown = [c for c in configs if c not in allowed]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; expected {allowed}"
                         f"{'' if args.with_prob else ' (P configs need --with-prob)'}")

    out_dir = os.path.join(RESULTS_DIR, "C2_final_results", args.disease,
                           src.name, f"cf_{args.cf_count}",
                           "attributes_prob" if args.with_prob else "attributes")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "cv_summary.csv")) and not args.overwrite:
        print(f"  cv_summary.csv already present in {out_dir}  (--overwrite to rebuild)")
        return

    print(f"\n{'='*72}")
    print(f"  ATTRIBUTE MODELS: {args.disease} | {src.describe()} | K={args.cf_count}")
    print(f"{'='*72}")
    print(f"  folds:   {fold_dir(args.disease, src.name, args.cf_count)}")
    print(f"  results: {out_dir}")
    print(f"  configs: {', '.join(configs)}  "
          f"({'with C0 scalars' if args.with_prob else 'no prob'})   device: {DEVICE}")

    results = {c: [] for c in configs}
    results["cf_anchor"] = []
    dims_seen = {}

    for fold in load_folds(args.disease, src, args.cf_count, n_folds=args.n_folds):
        train_df = fold.train
        print(f"\n{'-'*72}\n  FOLD {fold.index + 1}/{args.n_folds}"
              f"  (train {len(train_df):,} | test {len(fold):,} x {fold.n_slots} CF)"
              f"\n{'-'*72}")

        y_train = train_df["correct"].values
        groups = train_df["patient_id"].values
        cols = feature_columns(train_df, args.disease)
        print(f"  feature groups: attrs={len(cols['attr'])} delta={len(cols['delta'])} "
              f"emb={len(cols['emb'])} cf_attr={len(cols['cf_attr'])} "
              f"cf_emb={len(cols['cf_emb'])}")

        slots = [fold.test_slot(j) for j in range(fold.n_slots)]
        y_test = slots[0]["correct"].values
        for j, sl in enumerate(slots[1:], 1):
            assert (sl["path"].values == slots[0]["path"].values).all(), (
                f"fold {fold.index}: slot {j} has different query rows than slot 0")

        pred_df = slots[0][["path", "patient_id", f"{args.disease}_prob", "correct"]].copy()
        pred_df["cf_paths"] = fold.test_cf_paths
        pred_df["cf_prob"] = np.mean([sl["cf_prob"].values for sl in slots], axis=0)

        anchor_slots = score_cf_anchor(train_df, slots)
        slot_probs = {}
        if anchor_slots is not None:
            anchor_auc = float(roc_auc_score(y_test, anchor_slots.mean(axis=0)))
            results["cf_anchor"].append(anchor_auc)
            pred_df["cf_anchor_prob"] = anchor_slots.mean(axis=0)
            slot_probs["cf_anchor"] = anchor_slots
            print(f"  CF-anchor baseline AUC: {anchor_auc:.4f}")

        for cfg in configs:
            t0 = time.time()
            X_tr = config_matrix(train_df, cols, cfg)
            dims_seen.setdefault(cfg, int(X_tr.shape[1]))
            # A config with no CF block scores the same on every slot: evaluate once.
            use = slots if cfg in CF_DEPENDENT else slots[:1]
            per_slot = np.stack(fit_predict(X_tr, y_train, groups,
                                            [config_matrix(sl, cols, cfg) for sl in use]))
            if DEVICE != "cpu":
                torch.cuda.empty_cache()
            slot_probs[f"{MODEL}_{cfg}"] = per_slot
            y_prob = per_slot.mean(axis=0)
            auc = float(roc_auc_score(y_test, y_prob))
            results[cfg].append(auc)
            pred_df[f"{MODEL}_{cfg}_prob"] = y_prob
            print(f"  {MODEL} {cfg:<6} dim={X_tr.shape[1]:<5} x{len(use)} CF   "
                  f"AUC = {auc:.4f}   [{time.time()-t0:.0f}s]", flush=True)
            del X_tr

        # Written per fold: a wall-clock kill after fold 3 still leaves 3 usable folds.
        pred_df.to_csv(os.path.join(out_dir, f"fold_{fold.index}_predictions.csv"),
                       index=False)
        np.savez_compressed(
            os.path.join(out_dir, f"fold_{fold.index}_slot_probs.npz"),
            paths=slots[0]["path"].values.astype(str),
            patient_id=slots[0]["patient_id"].values.astype(str),
            labels=y_test.astype(np.int8),
            cf_paths=fold.test_cf_paths.astype(str),
            **slot_probs)

    # ── summary ───────────────────────────────────────────────────────────────
    rows = [{"model": MODEL, "config": cfg, "n_folds": len(a),
             "k_used": args.cf_count if cfg in CF_DEPENDENT else 1,
             "auc_mean": float(np.mean(a)), "auc_std": float(np.std(a)),
             "auc_folds": ";".join(f"{x:.4f}" for x in a)}
            for cfg, a in results.items() if cfg != "cf_anchor" and a]
    if results["cf_anchor"]:
        a = results["cf_anchor"]
        rows.append({"model": "baseline", "config": "cf_anchor", "n_folds": len(a),
                     "k_used": args.cf_count,
                     "auc_mean": float(np.mean(a)), "auc_std": float(np.std(a)),
                     "auc_folds": ";".join(f"{x:.4f}" for x in a)})

    summary = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    summary_path = os.path.join(out_dir, "cv_summary.csv")
    summary.to_csv(summary_path, index=False)

    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump({"disease": args.disease, "cf_source": src.name,
                   "cf_source_description": src.describe(),
                   "cf_count": args.cf_count, "train_k": 1,
                   "k_combination": "mean of the K predicted probabilities",
                   "n_folds": args.n_folds,
                   "design": ("ablation winner + C0 scalars [p_xi, H(p_xi), p_cf, "
                              "H(p_cf), p_xi - p_cf]" if args.with_prob
                              else "block ablation, no prob"),
                   "configs": configs, "models": [MODEL],
                   "cf_dependent_configs": sorted(CF_DEPENDENT),
                   "seed": RANDOM_SEED, "feature_dims": dims_seen}, fh, indent=2)

    print(f"\n{'='*72}\n  SUMMARY\n{'='*72}")
    for _, r in summary.iterrows():
        print(f"  {r['model']:<9} {r['config']:<9} {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")
    print(f"\n  wrote {summary_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--configs", default="",
                    help=f"comma-separated subset; default all ({','.join(CONFIGS)})")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if cv_summary.csv already exists")
    ap.add_argument("--with-prob", action="store_true",
                    help=f"train {','.join(PROB_CONFIGS)} (the C0 scalars of "
                         f"dual_sal_scalars added) into attributes_prob/")
    add_cf_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
