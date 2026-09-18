# c2_ensemble.py
"""
Stage 5d: combine the best attribute model with the best CNN.

Both trainers write per-fold predictions keyed by `path`, off the same cached
pairing, so fold membership and counterfactual assignment are identical on both
sides. That is what makes this join meaningful: a row's attribute prediction and
its CNN prediction come from models that saw the same 5-fold split and the same
counterfactual, and neither saw that row in training.

    python c2_ensemble.py --disease effusion --cf-source knn --cf-count 10

The two are picked by cross-validated AUC from each trainer's own cv_summary.csv;
--attr-model and --cnn-model override.

Choosing K
----------
K is a knob of each family separately, not a shared condition of the experiment:
the attribute models pool K counterfactual feature vectors, the CNN averages K
per-slot probabilities, and nothing makes them saturate at the same K. Pinning
both to one K means the family that wanted a different one enters the ensemble
handicapped, and the comparison then measures that handicap rather than
complementarity.

--search-k lets each side choose its own K by its own cross-validated AUC:

    python c2_ensemble.py --disease effusion --search-k 1,5,10

This is sound because the folds are built once per disease and cached, so every
K carries identical fold membership -- load_predictions asserts it rather than
trusting it. Each row is therefore out-of-fold on both sides regardless of which
K each side came from. Selection is still by cross-validated AUC on those same
folds, so the protocol is unchanged; only the search space is wider.

Combiners
---------
mean   arithmetic mean of the two probabilities. No fitting, so no leakage, but it
       assumes the two are comparably calibrated -- an over-confident model
       dominates the average regardless of which is more accurate.
rank   mean of within-fold percentile ranks. Calibration-free, so it asks "do
       these two ORDER cases differently" rather than "are their numbers on the
       same scale". Usually the honest first thing to look at.
lr     logistic regression on the two probabilities, fitted leave-one-fold-out:
       for fold i the meta-learner trains on the other four and predicts fold i,
       so no row's own prediction trains the model that scores it.

A caveat on `lr` worth stating plainly: the predictions used to TRAIN the
meta-learner come from base models that did train on fold i's rows. That is
ordinary stacking and carries the usual mild optimism. Treat an `lr` that beats
`rank` by a hair as noise, not a result.

Each ensemble is reported against the better of its two components. The deltas are
thousandths of AUROC and this script does no significance testing, so read them as
descriptive: a gain smaller than the fold-to-fold std next to it is not a result.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import (                                          # noqa: E402
    RESULTS_DIR, RANDOM_SEED, add_cf_args, cf_source_from_args,
)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

def results_root(disease, source_name, cf_count):
    return os.path.join(RESULTS_DIR, "C2_final_results", disease, source_name,
                        f"cf_{cf_count}")


def load_predictions(attr_root, cnn_root, n_folds, cnn_key):
    """Out-of-fold attribute and CNN predictions, joined on `path`.

    The two roots may be DIFFERENT K. That is safe, and the reason is that K
    changes only how many counterfactuals a prediction is pooled over -- it never
    changes the split. Folds are built once per disease and cached, so cf_1 and
    cf_10 carry byte-identical fold membership; this is asserted below rather
    than assumed. Every row therefore stays out-of-fold on both sides, which is
    the only property the join actually needs.
    """
    parts = []
    for i in range(n_folds):
        p = os.path.join(attr_root, "attributes", f"fold_{i}_predictions.csv")
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found -- run c2_train_attributes.py first")
        f = pd.read_csv(p)
        f["fold"] = i
        parts.append(f)
    attr = pd.concat(parts, ignore_index=True)

    parts = []
    for i in range(n_folds):
        p = os.path.join(cnn_root, "images", cnn_key, f"fold_{i}_predictions.csv")
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found -- run c2_train_images.py first")
        f = pd.read_csv(p)
        f["cnn_fold"] = i
        parts.append(f)
    cnn = pd.concat(parts, ignore_index=True)[["path", "cnn_fold", f"{cnn_key}_prob"]]

    df = attr.merge(cnn, on="path", how="inner")
    if len(df) != len(attr):
        print(f"  WARNING: {len(attr) - len(df):,} rows dropped joining CNN to "
              f"attribute predictions -- the two ladders disagree on membership")

    # A cross-K join is only out-of-fold if both sides put every row in the same
    # fold. If they ever diverge, a row's CNN prediction could come from a model
    # that trained on it, so this is a hard failure rather than a warning.
    bad = int((df["fold"] != df["cnn_fold"]).sum())
    if bad:
        raise SystemExit(
            f"{bad:,} rows have different fold assignments in the attribute and "
            f"CNN trees -- the two Ks do not share a split, so the ensemble would "
            f"not be out-of-fold. Rebuild the folds before ensembling.")
    df = df.drop(columns="cnn_fold")
    return df.reset_index(drop=True)


def pick_best_over_k(disease, source_name, kind, ks):
    """Best (model, K) for one family, searched over `ks` independently.

    K is a knob of each family, not a shared experimental condition: the
    attributes pool K counterfactual feature vectors, the CNN averages K
    per-slot probabilities, and the two saturate at different K. Tying both to
    one K therefore hands the ensemble a handicapped component whenever the
    optima differ -- which they do, e.g. effusion attributes still gain from
    K=10 while its CNN is flat past K=5.

    Both families are still selected by their own cross-validated AUC on the
    same folds, so this widens the search space without changing the protocol.
    Returns (key, K, auc).
    """
    best = None
    for k in ks:
        root = results_root(disease, source_name, k)
        if not os.path.isdir(os.path.join(root, kind)):
            continue
        try:
            key = pick_best(root, kind)
        except SystemExit:
            continue
        auc = score_of(root, kind, key)
        if auc is None:
            continue
        if best is None or auc > best[2]:
            best = (key, k, auc)
    if best is None:
        raise SystemExit(
            f"no finished {kind} run for {disease} at any of K={list(ks)}")
    return best


def score_of(root, kind, key):
    """Mean CV AUC of one already-chosen model, read from its own trainer's
    output. Kept separate from pick_best so the K search compares like with
    like: the same statistic, sourced the same way, at every K."""
    p = os.path.join(root, kind, "cv_summary.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        if kind == "attributes":
            s = s[s["model"] != "baseline"]
            m = s["model"].astype(str) + "_" + s["config"].astype(str)
        else:
            m = s["model"].astype(str)
        hit = s[m == key]
        if len(hit):
            return float(hit.iloc[0]["auc_mean"])
        return None
    f = os.path.join(root, kind, key, "fold_aucs.csv")
    if os.path.exists(f):
        return float(pd.read_csv(f)["auc"].mean())
    return None


def pick_best(root, kind):
    """Best config/model by mean CV AUC, from the trainer's own cv_summary.csv.

    cv_summary.csv is only written once a whole sweep finishes, so for the images
    ladder we fall back to scanning each model's fold_aucs.csv. That keeps this
    usable mid-run, when some CNN variants have finished and others have not.
    """
    p = os.path.join(root, kind, "cv_summary.csv")
    if os.path.exists(p):
        s = pd.read_csv(p).sort_values("auc_mean", ascending=False)
        if kind == "attributes":
            s = s[s["model"] != "baseline"]
            r = s.iloc[0]
            return f"{r['model']}_{r['config']}"
        return s.iloc[0]["model"]

    if kind == "attributes":
        raise SystemExit(f"{p} not found -- run c2_train_attributes.py first")

    d = os.path.join(root, "images")
    scored = {}
    for key in (sorted(os.listdir(d)) if os.path.isdir(d) else []):
        f = os.path.join(d, key, "fold_aucs.csv")
        if os.path.exists(f):
            scored[key] = pd.read_csv(f)["auc"].mean()
    if not scored:
        raise SystemExit(f"no finished CNN models under {d}")
    return max(scored, key=scored.get)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINERS
# ══════════════════════════════════════════════════════════════════════════════

def combine_mean(df, cols, n_folds):
    return df[cols].mean(axis=1).values


def combine_rank(df, cols, n_folds):
    """Mean within-fold percentile rank. Ranking inside the fold matters: folds
    have slightly different prevalence, and a global rank would let one fold's
    score distribution shift another's."""
    out = np.zeros(len(df))
    for _, idx in df.groupby("fold").groups.items():
        block = df.loc[idx, cols]
        out[df.index.get_indexer(idx)] = block.rank(pct=True).mean(axis=1).values
    return out


def combine_lr(df, cols, n_folds):
    """Logistic meta-learner, fitted leave-one-fold-out."""
    out = np.full(len(df), np.nan)
    y = df["correct"].values
    for i in range(n_folds):
        te = (df["fold"] == i).values
        sc = StandardScaler()
        lr = LogisticRegression(max_iter=5000, class_weight="balanced",
                                random_state=RANDOM_SEED)
        lr.fit(sc.fit_transform(df.loc[~te, cols].values), y[~te])
        out[te] = lr.predict_proba(sc.transform(df.loc[te, cols].values))[:, 1]
    assert not np.isnan(out).any(), "some rows were never scored"
    return out


COMBINERS = {"mean": combine_mean, "rank": combine_rank, "lr": combine_lr}


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def per_fold_auc(df, scores, n_folds):
    return [float(roc_auc_score(df.loc[m, "correct"], scores[m]))
            for m in ((df["fold"] == i).values for i in range(n_folds))]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    src = cf_source_from_args(args)
    root = results_root(args.disease, src.name, args.cf_count)

    # --search-k lets each family choose its own K. Without it the run is
    # pinned to --cf-count on both sides, which is the old behaviour.
    ks = [int(k) for k in args.search_k.split(",")] if args.search_k else None

    if ks and not args.attr_model:
        attr_key, attr_k, attr_auc = pick_best_over_k(
            args.disease, src.name, "attributes", ks)
    else:
        attr_k = args.cf_count
        attr_key = args.attr_model or pick_best(root, "attributes")
        attr_auc = score_of(results_root(args.disease, src.name, attr_k),
                            "attributes", attr_key)

    if ks and not args.cnn_model:
        cnn_key, cnn_k, cnn_auc = pick_best_over_k(
            args.disease, src.name, "images", ks)
    else:
        cnn_k = args.cf_count
        cnn_key = args.cnn_model or pick_best(root, "images")
        cnn_auc = score_of(results_root(args.disease, src.name, cnn_k),
                           "images", cnn_key)

    attr_root = results_root(args.disease, src.name, attr_k)
    cnn_root = results_root(args.disease, src.name, cnn_k)

    print(f"\n{'='*72}")
    scope = f"K searched over {ks}" if ks else f"K={args.cf_count}"
    print(f"  ENSEMBLE: {args.disease} | {src.describe()} | {scope}")
    print(f"{'='*72}")
    print(f"  attribute model: {attr_key:<24} K={attr_k}"
          + (f"  (CV AUC {attr_auc:.4f})" if attr_auc is not None else ""))
    print(f"  CNN model:       {cnn_key:<24} K={cnn_k}"
          + (f"  (CV AUC {cnn_auc:.4f})" if cnn_auc is not None else ""))
    if attr_k != cnn_k:
        print(f"  the two families peak at different K; folds are shared across "
              f"K, so the join stays out-of-fold")

    df = load_predictions(attr_root, cnn_root, args.n_folds, cnn_key)
    a, b = f"{attr_key}_prob", f"{cnn_key}_prob"
    for c in (a, b):
        if c not in df.columns:
            raise SystemExit(f"column {c!r} not in the predictions")
    print(f"  {len(df):,} rows with both predictions\n")

    cols = [a, b]

    rows = []
    for c in cols:
        f_aucs = per_fold_auc(df, df[c].values, args.n_folds)
        rows.append({"kind": "component", "members": c, "method": "-",
                     "auc_mean": float(np.mean(f_aucs)),
                     "auc_std": float(np.std(f_aucs)),
                     "delta_vs_best": np.nan})

    # The ensemble is read against the better of its two inputs: beating the
    # baseline is not the question here, both components already do that.
    best_auc = max(np.mean(per_fold_auc(df, df[c].values, args.n_folds)) for c in cols)
    methods = list(COMBINERS) if args.method == "all" else [args.method]

    for method in methods:
        scores = COMBINERS[method](df, cols, args.n_folds)
        f_aucs = per_fold_auc(df, scores, args.n_folds)
        rows.append({"kind": "ensemble", "members": f"{a} + {b}", "method": method,
                     "auc_mean": float(np.mean(f_aucs)),
                     "auc_std": float(np.std(f_aucs)),
                     "delta_vs_best": float(np.mean(f_aucs) - best_auc)})
        df[f"ens_{method}"] = scores

    out = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    # A mixed-K ensemble belongs to neither component's K, so it is written under
    # the larger of the two rather than silently overwriting a same-K result that
    # was produced under the old fixed-K protocol.
    out_dir = os.path.join(results_root(args.disease, src.name,
                                        max(attr_k, cnn_k)), "ensemble")
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "ensemble_summary.csv"), index=False)

    keep = (["path", "patient_id", "fold", "correct"] + cols
            + [c for c in df.columns if c.startswith("ens_")])
    df[keep].to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)

    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump({"disease": args.disease, "cf_source": src.name,
                   "cf_count": args.cf_count, "n_folds": args.n_folds,
                   "attr_model": attr_key, "cnn_model": cnn_key,
                   # Each family's own K, so a mixed-K run is self-describing and
                   # the notebook never has to infer which K a column came from.
                   "attr_cf_count": attr_k, "cnn_cf_count": cnn_k,
                   "k_searched": ks,
                   "methods": methods,
                   "n_rows": int(len(df))}, fh, indent=2)

    print(f"{'='*72}\n  RESULTS  (delta vs the better single component)\n{'='*72}")
    for _, r in out.iterrows():
        line = (f"  {r['method']:<6} {r['members']:<44} "
                f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}")
        if r["kind"] == "ensemble":
            line += f"   Δ {r['delta_vs_best']:+.4f}"
        print(line)
    print(f"  wrote {os.path.join(out_dir, 'ensemble_summary.csv')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--attr-model", default="",
                    help="e.g. MLP_MCF5 (default: best by cv_summary)")
    ap.add_argument("--cnn-model", default="",
                    help="e.g. dual_sal (default: best by cv_summary)")
    ap.add_argument("--method", default="all", choices=["mean", "rank", "lr", "all"])
    ap.add_argument("--search-k", default="", metavar="1,5,10",
                    help="let each family pick its own K from this list, by its "
                         "own CV AUC, instead of pinning both to --cf-count. "
                         "Folds are identical across K, so the join stays "
                         "out-of-fold; results are written under the larger K. "
                         "An explicit --attr-model/--cnn-model still pins that "
                         "side to --cf-count.")
    add_cf_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
