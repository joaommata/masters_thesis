"""
cf_levels_assemble.py
=====================
Join the three per-counterfactual tables into the two files DiffusionCFSource
reads, and refuse to write them if the schema does not match c2_data.csv.

Inputs
------
    CF_levels/<disease>/cf_index.csv          cf_levels_export_images.py
    CF_levels/<disease>/cf_c0.parquet         cf_levels_c0_predict.py
    CF_levels/<disease>/attributes/*.parquet  cf_levels_c1_attributes.py

Outputs
-------
    CF_levels/<disease>/cf_pairs.csv
        query_path, level_name, cf_path, prob, + the generation diagnostics.
        Small; this is what pair() and cf_prob_lut() read.

    CF_levels/<disease>/c2_cf_table.parquet
        One row per counterfactual with EXACTLY c2_data.csv's columns, in
        c2_data.csv's order. c2_folds.load_folds concatenates it onto the base
        frame, so the FoldContext resolves a generated CF path through the same
        row_of / attr_all / emb_all machinery as a real one, and the scaler is
        still fit on train QUERIES only.

Why the schema check is the point of this script
------------------------------------------------
Nothing downstream validates the column set. `_relevant_cols` builds the
attribute space by subtracting the known meta names from whatever it is handed,
so a leftover column (level_name, direction) becomes a 455th attribute in every
config and a missing one becomes a column of zeros -- in both cases silently,
and in both cases only in the diffusion arm, which is precisely the arm the
comparison is about. The assertions here are the only place that can catch it.

Usage
-----
    python cf_levels_assemble.py --disease atelectasis
    python cf_levels_assemble.py --disease atelectasis --levels high
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import c2_data_path                                  # noqa: E402

RESULTS = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")

PAIR_COLS = ["query_path", "level_name", "level", "cf_path", "prob",
             "orig_prob", "target_prob", "gen_cf_prob", "direction",
             "reached", "flipped", "l1", "cell", "patient_id"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--policy", default="ignore")
    ap.add_argument("--levels", nargs="+", default=None,
                    help="assemble only these levels (default: whatever is complete)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the tables even if some counterfactuals have no "
                         "attributes yet. Only for inspecting an in-flight run -- "
                         "a partial table makes pair() raise on the first query "
                         "that is missing a level.")
    args = ap.parse_args()

    root = os.path.join(RESULTS, "CF_levels", args.disease)
    index = pd.read_csv(os.path.join(root, "cf_index.csv"))
    if args.levels:
        index = index[index["level_name"].isin(args.levels)].reset_index(drop=True)

    c0 = pd.read_parquet(os.path.join(root, "cf_c0.parquet"))
    shards = sorted(glob.glob(os.path.join(root, "attributes", "attr_*.parquet")))
    if not shards:
        raise SystemExit(f"no attribute shards under {root}/attributes -- run "
                         f"cf_levels_c1_attributes.py --disease {args.disease}")
    attrs = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
    attrs = attrs.drop_duplicates("cf_path", keep="last")

    print(f"disease   {args.disease}")
    print(f"index     {len(index):,} counterfactuals")
    print(f"C0        {len(c0):,} scored")
    print(f"attrs     {len(attrs):,} extracted ({len(shards)} shards)")

    # ── coverage ─────────────────────────────────────────────────────────────
    want = set(index["cf_path"])
    no_c0 = want - set(c0["cf_path"])
    no_attr = want - set(attrs["cf_path"])
    print(f"\n  missing C0        : {len(no_c0):,}")
    print(f"  missing attributes: {len(no_attr):,}")
    if (no_c0 or no_attr) and not args.allow_partial:
        first = sorted(no_c0 or no_attr)[0]
        raise SystemExit(
            f"{len(no_c0):,} counterfactuals have no C0 pass and {len(no_attr):,} "
            f"have no attributes (e.g. {first!r}). Finish those stages, or pass "
            f"--levels to assemble only the levels that are complete.")
    if no_c0 or no_attr:
        index = index[~index["cf_path"].isin(no_c0 | no_attr)].reset_index(drop=True)
        print(f"  --allow-partial: keeping {len(index):,} complete rows")

    # ── the pairing index ────────────────────────────────────────────────────
    pairs = index.merge(c0[["cf_path", "prob"]], on="cf_path", how="left")
    pairs = pairs[PAIR_COLS]
    dest = os.path.join(root, "cf_pairs.csv")
    pairs.to_csv(dest, index=False)
    print(f"\n  cf_pairs.csv  {len(pairs):,} rows -> {dest}")

    # ── the c2_data-shaped table ─────────────────────────────────────────────
    head = pd.read_csv(c2_data_path(args.disease), nrows=0)
    cols = list(head.columns)

    with open(os.path.join(RESULTS, "C0_final", f"multilabel_{args.policy}",
                           args.disease, "threshold.txt")) as fh:
        threshold = float(fh.read().strip())

    cam_dir = os.path.join(RESULTS, "C0_final", f"multilabel_{args.policy}",
                           args.disease, "gradcam", "CF_levels")

    tab = (index[["cf_path", "query_path", "patient_id"]]
           .merge(c0, on="cf_path", how="left")
           .merge(attrs.drop(columns=[c for c in
                                      ("query_path", "level_name", "patient_id")
                                      if c in attrs.columns]),
                  on="cf_path", how="left"))

    tab = tab.rename(columns={"cf_path": "path"})
    # A counterfactual has no CheXpert label and is never scored as a query, so
    # `true` and `correct` stay NaN. They exist only because the column set has
    # to match; load_folds fills `correct` with -1 and nothing reads either.
    tab["true"] = np.nan
    tab["correct"] = np.nan
    tab["pred"] = (tab["prob"] > threshold).astype(int)
    tab["margin"] = tab["prob"] - threshold
    tab["cam_path"] = [os.path.join(cam_dir, p.replace("/", "_") + ".npz")
                       for p in tab["path"]]

    missing = [c for c in cols if c not in tab.columns]
    extra = [c for c in tab.columns if c not in cols and c != "query_path"]
    print(f"\n--- schema vs c2_data.csv ---")
    print(f"  expected columns : {len(cols)}")
    print(f"  missing          : {len(missing)}" +
          (f"  first {missing[:5]}" if missing else ""))
    print(f"  unexpected       : {len(extra)}" +
          (f"  first {extra[:5]}" if extra else ""))
    if missing or extra:
        raise SystemExit(
            "column set does not match c2_data.csv. An unexpected column becomes "
            "a spurious attribute and a missing one becomes zeros -- both only in "
            "the diffusion arm, which is the arm being measured.")

    tab = tab[cols]
    dest = os.path.join(root, "c2_cf_table.parquet")
    tmp = f"{dest}.tmp{os.getpid()}"
    tab.to_parquet(tmp, index=False)
    os.replace(tmp, dest)
    print(f"\n  c2_cf_table.parquet  {tab.shape[0]:,} x {tab.shape[1]} -> {dest}")

    # ── checks ───────────────────────────────────────────────────────────────
    attr_cols = [c for c in cols
                 if c not in ("path", "prob", "true", "pred", "correct",
                              "margin", "cam_path", "patient_id")
                 and not c.startswith("emb_")]
    ref = pd.read_csv(c2_data_path(args.disease), usecols=attr_cols, nrows=20000)

    print("\n--- checks ---")
    print(f"  duplicate paths   : {int(tab['path'].duplicated().sum())}")
    print(f"  levels per query  : "
          f"{pairs.groupby('query_path').size().value_counts().to_dict()}")
    n_cam = sum(os.path.exists(p) for p in tab["cam_path"].head(2000))
    print(f"  Grad-CAM present  : {n_cam}/2000 sampled")
    emb_cols = [c for c in cols if c.startswith("emb_")]
    print(f"  all-zero emb rows : "
          f"{int((tab[emb_cols].abs().sum(axis=1) == 0).sum())}")
    nan_cf = tab[attr_cols].isna().mean()
    nan_q = ref[attr_cols].isna().mean()
    worse = (nan_cf - nan_q).sort_values(ascending=False)
    print(f"  NaN fraction, CF vs query images -- worst 5 deltas:")
    for c, d in worse.head(5).items():
        print(f"    {c:55s} CF {nan_cf[c]:.3f}  query {nan_q[c]:.3f}  "
              f"({d:+.3f})")
    print("    (a segmenter that fails more often on counterfactuals than on "
          "real images is a finding to report, not necessarily a bug.)")


if __name__ == "__main__":
    main()
