"""
cf_levels_check_targets.py
==========================
Measure the one dependency the diffusion arm has on data outside its training
fold, so it can be reported as a number rather than as a caveat.

What the dependency is
----------------------
cf_generate_levels.compute_levels() sets the three guidance targets to equally
spaced probabilities between the 5th and 95th percentile of C0's probability
over the CORRECTLY-CLASSIFIED rows of the whole C2 set -- test rows included,
and using their `correct` label to select them. Those percentiles were computed
once, before the folds existed, and every counterfactual was generated against
them.

Nothing per-row leaks: the targets are six scalars per disease (three levels x
two travel directions), identical for every query, and they set only how hard
the guidance pushes. But they are a function of test-row labels, so "this arm is
fold-clean" is a claim that has to be supported rather than asserted.

What this measures
------------------
Recomputes the targets from rows that a model trained on fold i is allowed to
see -- the TRAIN partition of that fold, under the same StratifiedGroupKFold
split the pipeline uses -- and reports how far they move from the ones actually
used, in units of the spacing BETWEEN levels.

That ratio is the quantity that matters. A shift of 0.01 in probability is
meaningless if consecutive levels are 0.20 apart and fatal if they are 0.02
apart, so an absolute shift answers nothing on its own.

Reading the result
------------------
  shift << 1 level-step   the targets are a property of the marginal
                          distribution of C0's output, not of which rows it was
                          estimated on. Report the dependency as immaterial,
                          with this table as the evidence.
  shift ~  1 level-step   the levels a fold-clean run would have aimed at are
                          different levels. Either regenerate with per-fold
                          targets (~5x the generation campaign) or exclude the
                          arm from claims that need strict generation purity.

Usage
-----
    python cf_levels_check_targets.py
    python cf_levels_check_targets.py --diseases atelectasis --subset 20k
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_cf_sources import load_subset                              # noqa: E402
from c2_folds import load_c2_data, N_FOLDS, RANDOM_SEED            # noqa: E402

RESULTS = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")

LOW_PCT, HIGH_PCT = 5, 95
LEVEL_NAMES = ["low", "mid", "high"]


def targets_from(probs_pos, probs_neg):
    """The same construction compute_levels() uses, on an arbitrary row set.

    to_positive climbs from the 5th to the 95th percentile of the correctly
    classified POSITIVES; to_negative descends from the 95th to the 5th of the
    correctly classified negatives. Equally spaced probabilities, not quantiles.
    """
    out = {}
    if len(probs_pos):
        out["to_positive"] = list(np.linspace(np.percentile(probs_pos, LOW_PCT),
                                              np.percentile(probs_pos, HIGH_PCT), 3))
    if len(probs_neg):
        out["to_negative"] = list(np.linspace(np.percentile(probs_neg, HIGH_PCT),
                                              np.percentile(probs_neg, LOW_PCT), 3))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diseases", nargs="+",
                    default=["atelectasis", "cardiomegaly", "consolidation"])
    ap.add_argument("--subset", default="20k",
                    help="the subset the folds are built on ('' for the full table)")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    all_rows = []
    for disease in args.diseases:
        used = json.load(open(os.path.join(RESULTS, "CF_levels", disease,
                                           "levels.json")))
        threshold = used["threshold"]

        df = load_c2_data(disease, subset=load_subset(disease, args.subset))
        prob = df[f"{disease}_prob"].to_numpy()
        true = df[f"{disease}_true"].to_numpy()
        pred = (prob > threshold).astype(int)
        ok = pred == true

        skf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True,
                                   random_state=RANDOM_SEED)
        splits = list(skf.split(df, df["correct"].values, groups=df["patient_id"].values))

        print("\n" + "=" * 78)
        print(f"  {disease}")
        print("=" * 78)
        for dirn, sel in (("to_positive", (true == 1)), ("to_negative", (true == 0))):
            ref = np.array(used[dirn])
            step = abs(ref[1] - ref[0])
            print(f"\n  {dirn}   used: " +
                  "  ".join(f"{n}={v:.4f}" for n, v in zip(LEVEL_NAMES, ref)) +
                  f"   (level step {step:.4f})")

            for i, (tr_idx, _) in enumerate(splits):
                m = np.zeros(len(df), bool)
                m[tr_idx] = True
                p = prob[m & sel & ok]
                got = targets_from(p if dirn == "to_positive" else [],
                                   [] if dirn == "to_positive" else p)[dirn]
                d = np.abs(np.array(got) - ref)
                print(f"    fold {i} train ({len(p):,} rows): " +
                      "  ".join(f"{v:.4f}" for v in got) +
                      f"   |shift| max {d.max():.4f} = {d.max()/step:.3f} level-steps")
                all_rows.append({"disease": disease, "direction": dirn, "fold": i,
                                 "n_rows": int(len(p)), "step": step,
                                 "max_shift": float(d.max()),
                                 "shift_in_steps": float(d.max() / step)})

    res = pd.DataFrame(all_rows)
    dest = os.path.join(RESULTS, "CF_levels", "target_leak_check.csv")
    res.to_csv(dest, index=False)

    print("\n" + "=" * 78)
    print("  SUMMARY -- target shift when estimated from train rows only")
    print("=" * 78)
    print(res.groupby(["disease", "direction"])["shift_in_steps"]
             .agg(["mean", "max"]).round(4).to_string())
    worst = res["shift_in_steps"].max()
    print(f"\n  worst case across everything: {worst:.3f} level-steps")
    if worst < 0.25:
        print("  -> the targets barely move. The dependency on test-row labels is "
              "six scalars whose value is essentially determined by the marginal "
              "distribution of C0's output; report it as a stated limitation with "
              "this table as the evidence.")
    else:
        print("  -> the targets move by a meaningful fraction of the spacing "
              "between levels. A fold-clean run would have aimed somewhere else; "
              "either regenerate per fold or exclude this arm from claims that "
              "require generation purity.")
    print(f"\n  wrote {dest}")


if __name__ == "__main__":
    main()
