# c2_confidnet_compare.py
"""
Adapter: put the ConfidNet baseline into the Aim 2 bootstrap comparison, without rerunning it.

The Aim 2 statistics (figure 2a) come from bootstrapping.ipynb: a paired, patient-level
bootstrap of mean-fold AUROC (bootstrap_replicates.csv, N_BOOT draws, one draw shared by all
five diseases), plus mean-fold E-AURC on the SAME draws (c2_boot_extra.py, notebook section
8). This script adds two comparisons on those same draws:

    ConfidNet - adjusted MSP      does a learned no-CF rejector beat raw confidence?
    Ensemble  - ConfidNet         does the counterfactual system beat that rejector?

"Ensemble" is the final system's best model, the bootstrap's `best_all` pick per disease
(bootstrap_meta.json), read from cf_<K>/ensemble/oof_predictions.csv. Adjusted MSP is
c2_msp.msp_score(..., "threshold"), the variant the export was run with.

Nothing in the comparison logic is changed or re-implemented: the draws, the weighted AUROC and
E-AURC and the fold mean are c2_boot_extra.extra_bootstrap, and summarise / holm are
c2_boot_extra's copies of the notebook's. The draws are replayed from (seed, patient set), so
they are the export's draws exactly -- checked by rescoring MSP and the ensemble and comparing
with bootstrap_replicates.csv column by column before anything is written.

Per comparison and metric: one row per disease and an AVERAGE row (averaged inside each
replicate). Holm runs across the five diseases; the AVERAGE row is alone in its family and is
left as is -- the conventions of bootstrap_eaurc_deltas.csv. Signs: positive = the left-hand
model is better, for both metrics (E-AURC as reference minus model). Gap closed, as in
C2_evaluation_diffusion.ipynb's msp_gains, relative to the right-hand (reference) model:

    AUROC   gain / (1 - AUROC(ref))       E-AURC   gain / E-AURC(ref)

on the point estimates; the AVERAGE row's gap closed is the mean of the per-disease shares.

    python c2_confidnet_compare.py

Writes, next to the export in C2_final_results/:
    bootstrap_confidnet_vs_msp.csv
    bootstrap_ensemble_vs_confidnet.csv
    bootstrap_confidnet_replicates.csv     (AUROC and E-AURC of the three models, per draw)
and prints the disease x (adjusted MSP, ConfidNet, ensemble) summary table.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_boot_extra as BX                                     # noqa: E402
from c2_folds import RESULTS_DIR                                # noqa: E402
from c2_msp import msp_score                                    # noqa: E402

C2 = os.path.join(RESULTS_DIR, "C2_final_results")
SOURCE = "knn_correct_cf"      # bootstrapping.ipynb's SOURCE
MODEL = "MLP"
COLS = ["MSP", "ConfidNet", "Ensemble"]
COMPARISONS = {"confidnet_vs_msp": ("ConfidNet", "MSP"),
               "ensemble_vs_confidnet": ("Ensemble", "ConfidNet")}


def c0_threshold(disease):
    p = os.path.join(RESULTS_DIR, "C0_final", "multilabel_ignore", disease, "threshold.txt")
    return float(open(p).read().strip())


def load_attributes(disease, k, n_folds=5):
    """The main bootstrap's rows: the pooled attribute predictions at the final K."""
    parts = [pd.read_csv(os.path.join(C2, disease, SOURCE, f"cf_{k}", "attributes",
                                      f"fold_{i}_predictions.csv"),
                         usecols=["path", "patient_id", f"{disease}_prob", "correct"])
             .assign(fold=i) for i in range(n_folds)]
    return pd.concat(parts, ignore_index=True).set_index("path")


def build_table(disease, k, best_all, msp_variant):
    """One disease's rows with the three scores. Every join is checked, never filled."""
    ref = load_attributes(disease, k)
    tab = ref[["correct", "patient_id", "fold"]].copy()
    tab["MSP"] = msp_score(ref[f"{disease}_prob"].values, c0_threshold(disease), msp_variant)

    p = os.path.join(C2, disease, "baselines", "confidnet", "oof_predictions.csv")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found -- run c2_confidnet.py --disease {disease} first")
    cn = pd.read_csv(p).set_index("path")
    assert set(cn.index) == set(ref.index), \
        f"{disease}: ConfidNet rows != the bootstrap's rows ({len(cn):,} vs {len(ref):,})"
    cn = cn.reindex(ref.index)
    assert (cn["fold"].values == ref["fold"].values).all(), f"{disease}: ConfidNet folds differ"
    assert (cn["correct"].values == ref["correct"].values).all(), f"{disease}: labels differ"
    assert (cn["patient_id"].astype(str).values == ref["patient_id"].astype(str).values).all()
    tab["ConfidNet"] = cn["confidnet_score"].values

    assert best_all.startswith("ENS ") and best_all.endswith(f"@K{k}"), best_all
    ens = pd.read_csv(os.path.join(C2, disease, SOURCE, f"cf_{k}", "ensemble",
                                   "oof_predictions.csv")).set_index("path")
    col = "ens_" + best_all[len("ENS "):].rsplit("@", 1)[0]
    tab["Ensemble"] = ens[col].reindex(ref.index).values
    assert tab.notna().all().all(), f"{disease}: missing scores"
    return tab.reset_index(drop=True)


def point(tab):
    """Mean-fold AUROC and E-AURC of each score at unit weights -- the point estimates."""
    w = np.ones(len(tab))
    return (dict(zip(COLS, BX.FoldMean(tab, COLS, BX.WeightedAUC)(w))),
            dict(zip(COLS, BX.FoldMean(tab, COLS, BX.WeightedEAURC)(w))))


def deltas(auc_reps, eau_reps, points, diseases, hi, lo):
    rows = []
    for metric, reps in (("AUROC", auc_reps), ("E-AURC", eau_reps)):
        sign = 1 if metric == "AUROC" else -1        # E-AURC: lower is better
        per = {d: sign * (reps[d][hi] - reps[d][lo]).to_numpy() for d in diseases}
        block = []
        for d, x in per.items():
            a, e = points[d]
            if metric == "AUROC":
                ref, gain = a[lo], a[hi] - a[lo]
                gap = gain / (1 - ref)
            else:
                ref, gain = e[lo], e[lo] - e[hi]
                gap = gain / ref
            block.append({"comparison": f"{hi} - {lo}", "metric": metric, "disease": d,
                          f"{hi}": (a if metric == "AUROC" else e)[hi], f"{lo}": ref,
                          "gain": gain, "gap_closed_pct": 100 * gap, **BX.summarise(x)})
        block = pd.DataFrame(block)
        block["p_adj"] = BX.holm(block.p)
        avg = {"comparison": f"{hi} - {lo}", "metric": metric, "disease": "AVERAGE",
               f"{hi}": block[hi].mean(), f"{lo}": block[lo].mean(),
               "gain": block.gain.mean(), "gap_closed_pct": block.gap_closed_pct.mean(),
               **BX.summarise(np.mean(list(per.values()), axis=0))}
        avg["p_adj"] = avg["p"]
        rows.append(pd.concat([block, pd.DataFrame([avg])], ignore_index=True))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outs = {n: os.path.join(C2, f"bootstrap_{n}.csv") for n in COMPARISONS}
    if all(os.path.exists(p) for p in outs.values()) and not args.overwrite:
        print(f"  outputs already present in {C2}  (--overwrite to rebuild)")
        return

    pfx = os.path.join(C2, "bootstrap_")
    meta = json.load(open(pfx + "meta.json"))
    main_reps = pd.read_csv(pfx + "replicates.csv", index_col="replicate")
    diseases, picks, k = meta["diseases"], meta["picks"], meta["final"]["k"]
    assert not meta["final"].get("prob"), "the export is the probability system; adapt first"
    print(f"export {meta['created']}: {meta['n_boot']} draws, seed {meta['seed']}, "
          f"MSP variant {meta['msp_variant']}, K={k}")

    tables = {d: build_table(d, k, picks[d]["best_all"], meta["msp_variant"]) for d in diseases}
    points = {d: point(t) for d, t in tables.items()}
    # Unit-weight E-AURC must be the plain definition (the notebook's check).
    t0 = tables[diseases[0]]
    plain = np.mean([BX.e_aurc(g.correct, g.MSP) for _, g in t0.groupby("fold")])
    assert abs(points[diseases[0]][1]["MSP"] - plain) < 1e-12

    auc_reps, eau_reps = BX.extra_bootstrap(tables, COLS, meta["n_boot"], meta["seed"])
    # The pairing check: the replayed draws must rescore the export's own columns exactly.
    for d in diseases:
        for ours, theirs in (("MSP", "MSP"), ("Ensemble", picks[d]["best_all"])):
            diff = np.abs(auc_reps[d][ours].to_numpy() - main_reps[f"{d}|{theirs}"].to_numpy()).max()
            assert diff < 1e-12, f"{d}: replayed {ours} differs from the export by {diff:.1e}"
    print("replayed draws match the export (MSP and ensemble AUROC, every disease)")

    for name, (hi, lo) in COMPARISONS.items():
        out = deltas(auc_reps, eau_reps, points, diseases, hi, lo)
        out.to_csv(outs[name], index=False)
        print(f"\n{hi} - {lo}   (positive = {hi} better; Holm across the {len(diseases)} diseases)")
        show = out[["metric", "disease", "gain", "lo", "hi", "gap_closed_pct", "p", "p_adj"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"  wrote {outs[name]}")

    reps = pd.concat({"AUROC": auc_reps, "E-AURC": eau_reps}, axis=1)
    reps.columns = [f"{m}|{d}|{c}" for m, d, c in reps.columns]
    reps.to_csv(pfx + "confidnet_replicates.csv", index_label="replicate")

    print(f"\n{'='*72}\n  SUMMARY  mean-fold AUROC / E-AURC  (ensemble = best_all)\n{'='*72}")
    rows = []
    for d in diseases:
        a, e = points[d]
        rows.append({"disease": d, **{f"AUROC {c}": a[c] for c in COLS},
                     **{f"E-AURC {c}": e[c] for c in COLS},
                     "ensemble": picks[d]["best_all"]})
    summ = pd.DataFrame(rows)
    avg = {"disease": "AVERAGE", **summ.drop(columns=["disease", "ensemble"]).mean().to_dict(),
           "ensemble": ""}
    summ = pd.concat([summ, pd.DataFrame([avg])], ignore_index=True)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
