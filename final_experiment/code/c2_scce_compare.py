# c2_scce_compare.py
"""
Adapter: put the SCCE baseline into the Aim 2 bootstrap comparison, without rerunning it.

Sibling of c2_trustscore_compare.py, on the same replayed draws (c2_boot_extra.extra_bootstrap,
seed and patient set from bootstrap_meta.json, Holm across the five diseases):

    SCCE     - adjusted MSP      does distance to the counterfactuals beat raw confidence?
    Ensemble - SCCE              does learning from the counterfactual's content beat its distance?

Per comparison and metric: the paired difference on the same patient-level draws, its 95%
percentile interval, the bootstrap p and the Holm-adjusted p (the AVERAGE row, averaged inside
each replicate, is a single test). Positive = the left-hand model is better, for both metrics
(E-AURC as right minus left). No gap closed.

The rows and the MSP, MLP_I, ConfidNet, TrustScore and ensemble columns come from
c2_trustscore_compare.build_table with all its checks; SCCE is joined on with the same ones.
The replay is verified against bootstrap_replicates.csv (MSP, MLP_I, ensemble AUROC) before
anything is written.

    python c2_scce_compare.py [--overwrite]

Writes, next to the export in C2_final_results/:
    bootstrap_scce_vs_msp.csv
    bootstrap_ensemble_vs_scce.csv
    bootstrap_baselines_summary.csv   disease x (MSP, MLP_I, ConfidNet, TrustScore, SCCE, Ensemble),
                                      AUROC and E-AURC with 95% CI -- a superset of the file
                                      c2_trustscore_compare.py writes, on the same draws
    bootstrap_scce_replicates.csv     AUROC and E-AURC of the six models, per draw
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_boot_extra as BX                                     # noqa: E402
import c2_trustscore_compare as TSC                             # noqa: E402

C2 = TSC.C2
COLS = ["MSP", "MLP_I", "ConfidNet", "TrustScore", "SCCE", "Ensemble"]
COMPARISONS = {"scce_vs_msp": ("SCCE", "MSP"), "ensemble_vs_scce": ("Ensemble", "SCCE")}


def build_table(disease, k, best_all, msp_variant):
    tab = TSC.build_table(disease, k, best_all, msp_variant)
    ref = TSC.CC.load_attributes(disease, k)                 # build_table's row order
    p = os.path.join(C2, disease, "baselines", "scce", "oof_predictions.csv")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found -- run c2_scce.py --disease {disease} first")
    sc = pd.read_csv(p, usecols=["path", "patient_id", "fold", "correct", "scce_score"]).set_index("path")
    assert set(sc.index) == set(ref.index), \
        f"{disease}: SCCE rows != the bootstrap's rows ({len(sc):,} vs {len(ref):,})"
    sc = sc.reindex(ref.index)
    assert (sc["fold"].values == tab["fold"].values).all(), f"{disease}: SCCE folds differ"
    assert (sc["correct"].values == tab["correct"].values).all(), f"{disease}: labels differ"
    assert (sc["patient_id"].astype(str).values == tab["patient_id"].astype(str).values).all()
    tab["SCCE"] = sc["scce_score"].values
    tab = tab[["correct", "patient_id", "fold"] + COLS]
    assert tab.notna().all().all(), f"{disease}: missing scores"
    return tab


def method_summary(auc_reps, eau_reps, points, diseases):
    """Point estimate and 95% percentile interval of each method, per disease and averaged."""
    rows = []
    for d in diseases + ["AVERAGE"]:
        for c in COLS:
            r = {"disease": d, "method": c}
            for metric, reps, j in (("AUROC", auc_reps, 0), ("E-AURC", eau_reps, 1)):
                if d == "AVERAGE":
                    x = np.mean([reps[e][c].to_numpy() for e in diseases], axis=0)
                    pt = np.mean([points[e][j][c] for e in diseases])
                else:
                    x, pt = reps[d][c].to_numpy(), points[d][j][c]
                r.update({metric: pt, f"{metric}_lo": float(np.percentile(x, 2.5)),
                          f"{metric}_hi": float(np.percentile(x, 97.5))})
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outs = {n: os.path.join(C2, f"bootstrap_{n}.csv") for n in COMPARISONS}
    summ_path = os.path.join(C2, "bootstrap_baselines_summary.csv")
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
    points = {}
    for d, t in tables.items():
        w = np.ones(len(t))
        points[d] = (dict(zip(COLS, BX.FoldMean(t, COLS, BX.WeightedAUC)(w))),
                     dict(zip(COLS, BX.FoldMean(t, COLS, BX.WeightedEAURC)(w))))
    t0 = tables[diseases[0]]
    plain = np.mean([BX.e_aurc(g.correct, g.SCCE) for _, g in t0.groupby("fold")])
    assert abs(points[diseases[0]][1]["SCCE"] - plain) < 1e-12

    auc_reps, eau_reps = BX.extra_bootstrap(tables, COLS, meta["n_boot"], meta["seed"])
    for d in diseases:
        for ours, theirs in (("MSP", "MSP"), ("MLP_I", picks[d]["I"][0]),
                             ("Ensemble", picks[d]["best_all"])):
            diff = np.abs(auc_reps[d][ours].to_numpy()
                          - main_reps[f"{d}|{theirs}"].to_numpy()).max()
            assert diff < 1e-12, f"{d}: replayed {ours} differs from the export by {diff:.1e}"
    print("replayed draws match the export (MSP, MLP_I and ensemble AUROC, every disease)")

    for name, (hi, lo) in COMPARISONS.items():
        out = TSC.deltas(auc_reps, eau_reps, points, diseases, hi, lo)
        out.to_csv(outs[name], index=False)
        print(f"\n{hi} - {lo}   (positive = {hi} better; Holm across the {len(diseases)} diseases)")
        show = out[["metric", "disease", hi, lo, "gain", "lo", "hi", "p", "p_adj"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"  wrote {outs[name]}")

    reps = pd.concat({"AUROC": auc_reps, "E-AURC": eau_reps}, axis=1)
    reps.columns = [f"{m}|{d}|{c}" for m, d, c in reps.columns]
    reps.to_csv(pfx + "scce_replicates.csv", index_label="replicate")

    summ = method_summary(auc_reps, eau_reps, points, diseases)
    # Same draws as c2_trustscore_compare.py: the five methods it shares must agree exactly.
    if os.path.exists(summ_path):
        old = pd.read_csv(summ_path).set_index(["disease", "method"])
        new = summ.set_index(["disease", "method"])
        common = old.index.intersection(new.index)
        diff = (old.loc[common, new.columns] - new.loc[common]).abs().max().max()
        assert diff < 1e-12, f"summary disagrees with the existing one by {diff:.1e}"
    summ.to_csv(summ_path, index=False)
    print(f"\n{'='*72}\n  SUMMARY  mean-fold AUROC / E-AURC [95% CI]  (ensemble = best_all)\n"
          f"{'='*72}")
    fmt = lambda r, m: f"{r[m]:.4f} [{r[m + '_lo']:.4f}, {r[m + '_hi']:.4f}]"
    for metric in ("AUROC", "E-AURC"):
        wide = summ.assign(v=summ.apply(lambda r: fmt(r, metric), axis=1)) \
                   .pivot(index="disease", columns="method", values="v") \
                   .reindex(index=diseases + ["AVERAGE"], columns=COLS)
        print(f"\n{metric}\n{wide.to_string()}")
    print(f"\n  wrote {summ_path}")


if __name__ == "__main__":
    main()
