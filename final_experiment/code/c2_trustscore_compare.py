# c2_trustscore_compare.py
"""
Adapter: put the TrustScore baseline into the Aim 2 bootstrap comparison, without rerunning it.

Sibling of c2_confidnet_compare.py, on the same replayed draws (c2_boot_extra.extra_bootstrap,
seed and patient set from bootstrap_meta.json, Holm across the five diseases). It runs

    TrustScore - adjusted MSP      does the distance-to-other-class signal beat raw confidence?
    Ensemble   - TrustScore        does the counterfactual system beat it?

Per comparison and metric: the paired difference on the same patient-level draws, its 95%
percentile interval, the bootstrap p and the Holm-adjusted p (across the five diseases; the
AVERAGE row, averaged inside each replicate, is alone in its family and left as is). Positive =
the left-hand model is better for both metrics (E-AURC as right minus left). No gap closed.

Per method (adjusted MSP, MLP_I, ConfidNet, TrustScore, ensemble): mean-fold AUROC and E-AURC
with their 95% bootstrap intervals, disease x method.

The rows, MSP, ConfidNet and ensemble columns come from c2_confidnet_compare.build_table, with
all its checks. MLP_I is the attribute model's query-only block (MLP_I_prob of the attribute
fold predictions). The replay is verified against bootstrap_replicates.csv for MSP, MLP_I and
the ensemble before anything is written.

    python c2_trustscore_compare.py [--overwrite]

Writes, next to the export in C2_final_results/:
    bootstrap_trustscore_vs_msp.csv
    bootstrap_ensemble_vs_trustscore.csv
    bootstrap_baselines_summary.csv        disease x method, AUROC and E-AURC with 95% CI
    bootstrap_trustscore_replicates.csv    AUROC and E-AURC of the five models, per draw
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_boot_extra as BX                                     # noqa: E402
import c2_confidnet_compare as CC                               # noqa: E402

C2 = CC.C2
COLS = ["MSP", "MLP_I", "ConfidNet", "TrustScore", "Ensemble"]
COMPARISONS = {"trustscore_vs_msp": ("TrustScore", "MSP"),
               "ensemble_vs_trustscore": ("Ensemble", "TrustScore")}


def build_table(disease, k, best_all, msp_variant):
    tab = CC.build_table(disease, k, best_all, msp_variant)   # MSP, ConfidNet, Ensemble
    ref = CC.load_attributes(disease, k)                       # the same rows, same order
    assert (ref["fold"].values == tab["fold"].values).all()
    assert (ref["correct"].values == tab["correct"].values).all()

    parts = [pd.read_csv(os.path.join(C2, disease, CC.SOURCE, f"cf_{k}", "attributes",
                                      f"fold_{i}_predictions.csv"), usecols=["path", "MLP_I_prob"])
             for i in range(5)]
    mi = pd.concat(parts, ignore_index=True).set_index("path")
    assert mi.index.is_unique and set(mi.index) == set(ref.index)
    tab["MLP_I"] = mi["MLP_I_prob"].reindex(ref.index).values

    p = os.path.join(C2, disease, "baselines", "trustscore", "oof_predictions.csv")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found -- run c2_trustscore.py --disease {disease} first")
    ts = pd.read_csv(p).set_index("path")
    assert set(ts.index) == set(ref.index), \
        f"{disease}: TrustScore rows != the bootstrap's rows ({len(ts):,} vs {len(ref):,})"
    ts = ts.reindex(ref.index)
    assert (ts["fold"].values == ref["fold"].values).all(), f"{disease}: TrustScore folds differ"
    assert (ts["correct"].values == ref["correct"].values).all(), f"{disease}: labels differ"
    assert (ts["patient_id"].astype(str).values == ref["patient_id"].astype(str).values).all()
    tab["TrustScore"] = ts["trustscore"].values
    tab = tab[["correct", "patient_id", "fold"] + COLS]
    assert tab.notna().all().all(), f"{disease}: missing scores"
    return tab


def deltas(auc_reps, eau_reps, points, diseases, hi, lo):
    rows = []
    for metric, reps in (("AUROC", auc_reps), ("E-AURC", eau_reps)):
        sign = 1 if metric == "AUROC" else -1        # E-AURC: lower is better
        j = 0 if metric == "AUROC" else 1
        per = {d: sign * (reps[d][hi] - reps[d][lo]).to_numpy() for d in diseases}
        block = pd.DataFrame([{"comparison": f"{hi} - {lo}", "metric": metric, "disease": d,
                               hi: points[d][j][hi], lo: points[d][j][lo],
                               "gain": sign * (points[d][j][hi] - points[d][j][lo]),
                               **BX.summarise(x)} for d, x in per.items()])
        block["p_adj"] = BX.holm(block.p)
        avg = {"comparison": f"{hi} - {lo}", "metric": metric, "disease": "AVERAGE",
               hi: block[hi].mean(), lo: block[lo].mean(), "gain": block.gain.mean(),
               **BX.summarise(np.mean(list(per.values()), axis=0))}
        avg["p_adj"] = avg["p"]
        rows.append(pd.concat([block, pd.DataFrame([avg])], ignore_index=True))
    return pd.concat(rows, ignore_index=True)


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
    if all(os.path.exists(p) for p in [*outs.values(), summ_path]) and not args.overwrite:
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
    plain = np.mean([BX.e_aurc(g.correct, g.TrustScore) for _, g in t0.groupby("fold")])
    assert abs(points[diseases[0]][1]["TrustScore"] - plain) < 1e-12

    auc_reps, eau_reps = BX.extra_bootstrap(tables, COLS, meta["n_boot"], meta["seed"])
    for d in diseases:
        for ours, theirs in (("MSP", "MSP"), ("MLP_I", picks[d]["I"][0]),
                             ("Ensemble", picks[d]["best_all"])):
            diff = np.abs(auc_reps[d][ours].to_numpy()
                          - main_reps[f"{d}|{theirs}"].to_numpy()).max()
            assert diff < 1e-12, f"{d}: replayed {ours} differs from the export by {diff:.1e}"
    print("replayed draws match the export (MSP, MLP_I and ensemble AUROC, every disease)")

    for name, (hi, lo) in COMPARISONS.items():
        out = deltas(auc_reps, eau_reps, points, diseases, hi, lo)
        out.to_csv(outs[name], index=False)
        print(f"\n{hi} - {lo}   (positive = {hi} better; Holm across the {len(diseases)} diseases)")
        show = out[["metric", "disease", hi, lo, "gain", "lo", "hi", "p", "p_adj"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"  wrote {outs[name]}")

    reps = pd.concat({"AUROC": auc_reps, "E-AURC": eau_reps}, axis=1)
    reps.columns = [f"{m}|{d}|{c}" for m, d, c in reps.columns]
    reps.to_csv(pfx + "trustscore_replicates.csv", index_label="replicate")

    summ = method_summary(auc_reps, eau_reps, points, diseases)
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
