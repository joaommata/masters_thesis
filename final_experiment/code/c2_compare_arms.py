"""
c2_compare_arms.py
==================
Prove that two counterfactual arms are actually comparable before reading any
number off them, and then put their scores side by side.

Why a separate script
---------------------
"Diffusion beats retrieval" is only a claim about counterfactuals if the two
arms differ in nothing else. They are built by the same code from the same
frozen subset, but that is an argument, not a check -- and the failure it
guards against is silent. A pairing built against a stale subset, a fold cache
left over from a different seed, or a derived K that came from the wrong parent
all produce complete, plausible results on different rows.

What it asserts
---------------
  * both arms exist at this K for this disease;
  * fold by fold, the (split, path) index of the two pairings is IDENTICAL --
    same queries, same fold, same side. This is the load-bearing one;
  * within each arm, the K=1 pairing is the prefix of the K=3 one, so the K
    axis means what it says (delegated to c2_folds.check_nesting);
  * the diffusion arm's slot 0 is the `high` level for every query;
  * neither arm's counterfactuals cross the patient boundary in a way the other
    does not: the KNN arm's CFs must all be train rows of their own fold, and
    the diffusion arm's must all be generated from their own query.

Usage
-----
    python c2_compare_arms.py --disease atelectasis --cf-count 1
    python c2_compare_arms.py --disease atelectasis --cf-count 1 --scores
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_cf_sources import build_cf_source                          # noqa: E402
from c2_folds import (                                             # noqa: E402
    RESULTS_DIR, N_FOLDS, read_pairing, pairing_path, fold_dir, check_nesting,
)


def results_path(disease, source_name, k, kind):
    return os.path.join(RESULTS_DIR, "C2_final_results", disease, source_name,
                        f"cf_{k}", kind, "cv_summary.csv")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--cf-count", type=int, default=1)
    ap.add_argument("--subset", default="20k")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--scores", action="store_true",
                    help="also print the two arms' cv_summary side by side")
    ap.add_argument("--nesting", action="store_true",
                    help="also run the within-arm K=1 vs K=3 prefix check (slow)")
    args = ap.parse_args()

    knn = build_cf_source("knn", subset=args.subset)
    dif = build_cf_source("diffusion", subset=args.subset)
    k = args.cf_count

    print(f"disease   {args.disease}   K={k}   subset={args.subset or '(full)'}")
    print(f"  arm A   {knn.name}")
    print(f"  arm B   {dif.name}")

    for src in (knn, dif):
        missing = [i for i in range(args.n_folds)
                   if not os.path.exists(pairing_path(args.disease, src.name, k, i))]
        if missing:
            raise SystemExit(
                f"{src.name}: folds {missing} have no pairing under "
                f"{fold_dir(args.disease, src.name, k)}")

    # ── the load-bearing check ───────────────────────────────────────────────
    print("\n--- same rows, same folds ---")
    for i in range(args.n_folds):
        a = read_pairing(args.disease, knn.name, k, i).set_index(["split", "path"])
        b = read_pairing(args.disease, dif.name, k, i).set_index(["split", "path"])
        only_a = a.index.difference(b.index)
        only_b = b.index.difference(a.index)
        if len(only_a) or len(only_b):
            raise AssertionError(
                f"fold {i}: the two arms do not score the same rows "
                f"({len(only_a)} only in {knn.name}, {len(only_b)} only in "
                f"{dif.name}). Any AUC difference between them is partly a "
                f"difference in which images were scored.")
        n_tr = int((a.index.get_level_values(0) == "train").sum())
        print(f"  fold {i}: identical index -- {len(a):,} rows "
              f"({n_tr:,} train / {len(a)-n_tr:,} test)")

    # ── arm-specific structure ───────────────────────────────────────────────
    print("\n--- arm B: slot 0 is the strongest level ---")
    bad = 0
    for i in range(args.n_folds):
        b = read_pairing(args.disease, dif.name, k, i)
        first = [str(c).split("|")[0] for c in b["cf_paths"]]
        bad += sum(f"/{dif.LEVEL_ORDER[0]}/" not in p for p in first)
    print(f"  slot 0 not '{dif.LEVEL_ORDER[0]}': {bad} of "
          f"{args.n_folds * len(b):,} rows")
    if bad:
        raise AssertionError(
            "slot 0 is what training uses and what K=1 is. If it is not the "
            "level the arm claims, the K=1 result is not the level's result.")

    print("\n--- arm A: every counterfactual is a train row of its own fold ---")
    for i in range(args.n_folds):
        a = read_pairing(args.disease, knn.name, k, i)
        pool = set(a[a["split"] == "train"]["path"])
        cfs = {p for cell in a["cf_paths"] for p in str(cell).split("|")}
        outside = cfs - pool
        print(f"  fold {i}: {len(cfs):,} distinct CFs, {len(outside)} outside the "
              f"train partition")
        if outside:
            raise AssertionError(
                f"fold {i}: {len(outside)} retrieved counterfactuals are not "
                f"train rows -- the retrieval pool leaked across the split")

    print("\n--- arm B: every counterfactual belongs to its own query ---")
    for i in range(args.n_folds):
        b = read_pairing(args.disease, dif.name, k, i)
        wrong = 0
        for q, cell in zip(b["path"], b["cf_paths"]):
            stem = q.replace("/", "_")
            stem = stem[:-4] if stem.endswith(".jpg") else stem
            wrong += sum(not p.endswith(stem + ".png") for p in str(cell).split("|"))
        print(f"  fold {i}: {wrong} counterfactuals belong to a different query")
        if wrong:
            raise AssertionError(
                "a generated counterfactual must be derived from its own query; "
                "one derived from another row would be a cross-row dependency "
                "the arm claims not to have")

    if args.nesting:
        for src in (knn, dif):
            ks = sorted({int(d[3:]) for d in
                         os.listdir(os.path.dirname(fold_dir(args.disease, src.name, 1)))
                         if d.startswith("cf_")})
            if len(ks) > 1:
                print(f"\n--- {src.name}: K={ks[0]} is the K={ks[-1]} prefix ---")
                check_nesting(args.disease, src.name, ks[0], ks[-1],
                              n_folds=args.n_folds,
                              subset=src.subset_paths(args.disease))

    print("\nOK: the two arms differ only in where their counterfactuals came from.")

    # ── scores ───────────────────────────────────────────────────────────────
    if args.scores:
        for kind in ("attributes", "images"):
            frames = {}
            for src in (knn, dif):
                p = results_path(args.disease, src.name, k, kind)
                if os.path.exists(p):
                    frames[src.name] = pd.read_csv(p)
            if len(frames) < 2:
                print(f"\n  {kind}: only {len(frames)} arm(s) finished -- "
                      f"{sorted(set(s.name for s in (knn, dif)) - set(frames))} missing")
                continue
            keycol = ["model", "config"] if kind == "attributes" else ["model"]
            merged = None
            for name, f in frames.items():
                f = f[keycol + ["auc_mean"]].rename(columns={"auc_mean": name})
                merged = f if merged is None else merged.merge(f, on=keycol, how="outer")
            merged["delta"] = merged[dif.name] - merged[knn.name]
            print(f"\n{'='*78}\n  {kind}  (delta = diffusion - KNN)\n{'='*78}")
            print(merged.sort_values("delta", ascending=False)
                        .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
