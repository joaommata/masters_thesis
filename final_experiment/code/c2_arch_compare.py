# c2_arch_compare.py
"""
Does the final C2 system generalise across C0 architectures?

For every C0 backbone (DenseNet121 = the main C0, ViT-B/16, ResNet-50) the final system
was rerun with its recipe frozen (jobs/submit_arch.sh): KNN arm, K=10, attribute MLP
I+C+D, CNN dual_sal, and their ensemble with the combiner DenseNet's bootstrap picked
per disease (bootstrap_meta.json picks[d]["best_all"]). Nothing was re-tuned per C0, so
this measures transfer of the method, not of a model.

Per (arch, disease) the scores are, on identical rows and folds:
    MSP          adjusted MSP (threshold variant at that C0's own Youden threshold)
    Attr         MLP I+C+D
    CNN          dual_sal
    Ensemble     the final system

One patient-level bootstrap covers every arch and disease, so each draw resamples the
same patients everywhere: within an arch the comparisons are paired as in
bootstrapping.ipynb, and across archs the gains are paired too. Metrics are mean-fold
AUROC and E-AURC (c2_boot_extra), Holm across the diseases within each (arch,
comparison, metric); the AVERAGE row is averaged inside each replicate.

Pre-registered success criterion (fixed before the non-DenseNet runs): the method
generalises to a C0 if Ensemble - MSP has a positive AUROC gain with Holm p < 0.05 on
the same diseases where it holds for DenseNet.

ConfidNet and TrustScore are deliberately not rerun per arch: the prior-art comparison
is a DenseNet result (bootstrapping.ipynb / c2_*_compare.py). This script asks only
whether the final system's gain over MSP transfers to other C0s.

A non-significant arch difference in the gain (arch_gain_deltas.csv) is NOT evidence
that the gains are equal; read its interval.

    python c2_arch_compare.py                              # all three archs
    python c2_arch_compare.py --archs densenet121 vit_b_16

Writes to /work3/s251710/thesis_results/C2_arch_compare/:
    summary.csv           arch x disease: C0 AUROC/accuracy, each method's AUROC and
                          E-AURC with 95% bootstrap intervals
    deltas.csv            per arch: Ensemble, Attr and CNN vs MSP
    arch_gain_deltas.csv  (Ensemble - MSP) of each arch minus that of DenseNet
    replicates.csv        every draw, every score
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_boot_extra as BX                                      # noqa: E402
from c0_arch import BASE_RESULTS, EMB_DIM, arch_root, c0_arch   # noqa: E402
from c2_msp import msp_score                                    # noqa: E402

SOURCE = "knn_correct_cf"
K = 10
ATTR = "MLP_I+C+D"
CNN = "dual_sal"
DISEASES = ["effusion", "atelectasis", "cardiomegaly", "consolidation", "edema"]
COLS = ["MSP", "Attr", "CNN", "Ensemble"]
COMPARISONS = [("Ensemble", "MSP"), ("Attr", "MSP"), ("CNN", "MSP")]
MAIN_META = os.path.join(BASE_RESULTS, "C2_final_results", "bootstrap_meta.json")
MAIN_REPS = os.path.join(BASE_RESULTS, "C2_final_results", "bootstrap_replicates.csv")
OUT = os.path.join(BASE_RESULTS, "C2_arch_compare")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

def read_oof(p, what):
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found -- {what} has not run for this arch/disease")
    return pd.read_csv(p).set_index("path")


def build_table(arch, disease, ens_col, msp_variant):
    """One (arch, disease): the four scores on the attribute model's rows. Every join is
    checked, never filled."""
    root = arch_root(arch)
    assert c0_arch(root) == arch, f"{root} holds a {c0_arch(root)} C0, not {arch}"
    run = os.path.join(root, "C2_final_results", disease, SOURCE, f"cf_{K}")

    parts = []
    for i in range(5):
        p = os.path.join(run, "attributes", f"fold_{i}_predictions.csv")
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found -- c2_train_attributes.py has not run")
        parts.append(pd.read_csv(p, usecols=["path", "patient_id", f"{disease}_prob",
                                             "correct", f"{ATTR}_prob"]).assign(fold=i))
    ref = pd.concat(parts, ignore_index=True).set_index("path")
    assert ref.index.is_unique

    t = float(open(os.path.join(root, "C0_final", "multilabel_ignore", disease,
                                "threshold.txt")).read().strip())
    prob = ref[f"{disease}_prob"].to_numpy()
    tab = ref[["correct", "patient_id", "fold"]].copy()
    tab["MSP"] = msp_score(prob, t, msp_variant)
    tab["Attr"] = ref[f"{ATTR}_prob"].to_numpy()

    def attach(df, col, name):
        assert set(df.index) == set(ref.index), \
            f"{arch}/{disease}: {name} rows != the attribute rows ({len(df):,} vs {len(ref):,})"
        df = df.reindex(ref.index)
        assert (df["fold"].values == ref["fold"].values).all(), f"{arch}/{disease}: {name} folds"
        assert (df["correct"].values == ref["correct"].values).all(), f"{arch}/{disease}: {name} labels"
        tab[name] = df[col].to_numpy()

    ens = read_oof(os.path.join(run, "ensemble", "oof_predictions.csv"), "c2_ensemble.py")
    attach(ens, f"{CNN}_prob", "CNN")
    attach(ens, ens_col, "Ensemble")
    assert tab[COLS].notna().all().all(), f"{arch}/{disease}: missing scores"

    # C0 itself on these rows: the true label is recoverable from its decision and
    # whether that decision was correct.
    pred = (prob > t).astype(int)
    true = np.where(ref["correct"].to_numpy() == 1, pred, 1 - pred)
    c0 = {"C0 AUROC": float(roc_auc_score(true, prob)),
          "C0 accuracy": float(ref["correct"].mean()), "n": len(ref)}
    return tab.reset_index(drop=True), c0


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def point(tab):
    w = np.ones(len(tab))
    return (dict(zip(COLS, BX.FoldMean(tab, COLS, BX.WeightedAUC)(w))),
            dict(zip(COLS, BX.FoldMean(tab, COLS, BX.WeightedEAURC)(w))))


def gain_draws(reps, key, hi, lo, metric):
    """Per-draw gain of `hi` over `lo`; positive = hi better (E-AURC: lower is better)."""
    sign = 1 if metric == "AUROC" else -1
    return sign * (reps[key][hi] - reps[key][lo]).to_numpy()


def deltas(archs, diseases, reps, points):
    rows = []
    for arch in archs:
        for hi, lo in COMPARISONS:
            for metric in ("AUROC", "E-AURC"):
                m = 0 if metric == "AUROC" else 1
                per = {d: gain_draws(reps[metric], f"{arch}|{d}", hi, lo, metric)
                       for d in diseases}
                block = []
                for d, x in per.items():
                    p = points[f"{arch}|{d}"][m]
                    gain = (p[hi] - p[lo]) if metric == "AUROC" else (p[lo] - p[hi])
                    block.append({"arch": arch, "comparison": f"{hi} - {lo}", "metric": metric,
                                  "disease": d, hi: p[hi], lo: p[lo], "gain": gain,
                                  **BX.summarise(x)})
                block = pd.DataFrame(block)
                block["p_adj"] = BX.holm(block.p)
                avg = {"arch": arch, "comparison": f"{hi} - {lo}", "metric": metric,
                       "disease": "AVERAGE", hi: block[hi].mean(), lo: block[lo].mean(),
                       "gain": block.gain.mean(),
                       **BX.summarise(np.mean(list(per.values()), axis=0))}
                avg["p_adj"] = avg["p"]
                rows.append(pd.concat([block, pd.DataFrame([avg])], ignore_index=True))
    return pd.concat(rows, ignore_index=True)


def arch_gain_deltas(archs, diseases, reps):
    """(Ensemble - MSP) of each arch minus DenseNet's, on the same draws."""
    rows = []
    for arch in [a for a in archs if a != "densenet121"]:
        for metric in ("AUROC", "E-AURC"):
            per = {d: gain_draws(reps[metric], f"{arch}|{d}", "Ensemble", "MSP", metric)
                      - gain_draws(reps[metric], f"densenet121|{d}", "Ensemble", "MSP", metric)
                   for d in diseases}
            block = pd.DataFrame([{"arch": arch, "metric": metric, "disease": d,
                                   **BX.summarise(x)} for d, x in per.items()])
            block["p_adj"] = BX.holm(block.p)
            avg = {"arch": arch, "metric": metric, "disease": "AVERAGE",
                   **BX.summarise(np.mean(list(per.values()), axis=0))}
            avg["p_adj"] = avg["p"]
            rows.append(pd.concat([block, pd.DataFrame([avg])], ignore_index=True))
    if not rows:                                                # DenseNet alone
        return pd.DataFrame(columns=["arch", "metric", "disease", "delta", "lo", "hi", "p_adj"])
    return pd.concat(rows, ignore_index=True)


def interval_summary(archs, diseases, reps, points, c0):
    rows = []
    for arch in archs:
        for d in diseases:
            key = f"{arch}|{d}"
            row = {"arch": arch, "disease": d, **c0[key]}
            for metric, m in (("AUROC", 0), ("E-AURC", 1)):
                for c in COLS:
                    x = reps[metric][key][c].to_numpy()
                    row[f"{metric} {c}"] = points[key][m][c]
                    row[f"{metric} {c} lo"] = float(np.percentile(x, 2.5))
                    row[f"{metric} {c} hi"] = float(np.percentile(x, 97.5))
            rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", nargs="+", default=list(EMB_DIM), choices=list(EMB_DIM))
    ap.add_argument("--diseases", nargs="+", default=DISEASES, choices=DISEASES)
    ap.add_argument("--n-boot", type=int, default=None, help="default: the main export's")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if "densenet121" not in args.archs:
        ap.error("densenet121 is the reference every other arch is compared against")

    if os.path.exists(os.path.join(OUT, "deltas.csv")) and not args.overwrite:
        raise SystemExit(f"outputs already in {OUT} (--overwrite to rebuild)")

    # The DenseNet export fixes everything that is not re-chosen per arch.
    meta = json.load(open(MAIN_META))
    assert meta["final"] == {"k": K, "attr": "I+C+D", "cnn": CNN, "prob": False}, meta["final"]
    n_boot, seed, msp_variant = args.n_boot or meta["n_boot"], meta["seed"], meta["msp_variant"]
    ens_cols = {}
    for d in args.diseases:
        best = meta["picks"][d]["best_all"]                     # e.g. "ENS lr@K10"
        assert best.startswith("ENS ") and best.endswith(f"@K{K}"), best
        ens_cols[d] = "ens_" + best[len("ENS "):].rsplit("@", 1)[0]
    print(f"archs {args.archs} | diseases {args.diseases}")
    print(f"MSP variant {msp_variant} | {n_boot} draws, seed {seed}")
    print("ensemble combiner per disease (frozen from DenseNet): "
          + ", ".join(f"{d}={c}" for d, c in ens_cols.items()))

    tables, c0 = {}, {}
    for arch in args.archs:
        for d in args.diseases:
            tables[f"{arch}|{d}"], c0[f"{arch}|{d}"] = build_table(arch, d, ens_cols[d],
                                                                   msp_variant)
    # C0's correctness label is undefined only on uncertain (-1) labels, which do not
    # depend on C0, so every arch should score the same rows.
    for d in args.diseases:
        pats = {a: set(tables[f"{a}|{d}"].patient_id) for a in args.archs}
        same = all(p == pats["densenet121"] for p in pats.values())
        print(f"  {d}: rows " + ", ".join(f"{a} {len(tables[f'{a}|{d}']):,}" for a in args.archs)
              + ("" if same else "  !! patient sets differ across archs"))

    points = {k: point(t) for k, t in tables.items()}
    auc, eau = BX.extra_bootstrap(tables, COLS, n_boot, seed)
    reps = {"AUROC": auc, "E-AURC": eau}

    # With the main export's diseases the patient set is the export's, so DenseNet's
    # replayed draws must reproduce it exactly -- the proof the pipeline under test is
    # the one the thesis reports. Draws are sequential from one seed, so a shorter run
    # is the export's first n_boot draws.
    if os.path.exists(MAIN_REPS) and args.diseases == meta["diseases"] and n_boot <= meta["n_boot"]:
        main_reps = pd.read_csv(MAIN_REPS, index_col="replicate").iloc[:n_boot]
        for d in args.diseases:
            for ours, theirs in (("MSP", "MSP"), ("Ensemble", meta["picks"][d]["best_all"])):
                diff = np.abs(auc[f"densenet121|{d}"][ours].to_numpy()
                              - main_reps[f"{d}|{theirs}"].to_numpy()).max()
                assert diff < 1e-12, f"{d}: DenseNet {ours} differs from the export by {diff:.1e}"
        print("DenseNet draws reproduce the main bootstrap export exactly")
    else:
        print("(not comparable to the main export -- different diseases or more draws)")

    os.makedirs(OUT, exist_ok=True)
    summ = interval_summary(args.archs, args.diseases, reps, points, c0)
    dl = deltas(args.archs, args.diseases, reps, points)
    ag = arch_gain_deltas(args.archs, args.diseases, reps)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    dl.to_csv(os.path.join(OUT, "deltas.csv"), index=False)
    ag.to_csv(os.path.join(OUT, "arch_gain_deltas.csv"), index=False)
    flat = pd.concat(reps, axis=1)
    flat.columns = [f"{m}|{k}|{c}" for m, k, c in flat.columns]
    flat.to_csv(os.path.join(OUT, "replicates.csv"), index_label="replicate")
    with open(os.path.join(OUT, "run_config.json"), "w") as f:
        json.dump({"archs": args.archs, "diseases": args.diseases, "n_boot": n_boot,
                   "seed": seed, "msp_variant": msp_variant, "source": SOURCE, "k": K,
                   "attr": ATTR, "cnn": CNN, "ensemble": ens_cols}, f, indent=2)

    fmt = lambda x: f"{x:.4f}"                                   # noqa: E731
    print(f"\n{'='*78}\n  C0 and mean-fold AUROC per method\n{'='*78}")
    print(summ[["arch", "disease", "C0 AUROC", "C0 accuracy"]
               + [f"AUROC {c}" for c in COLS]].to_string(index=False, float_format=fmt))
    print(f"\n{'='*78}\n  Ensemble vs adjusted MSP (AUROC; Holm across diseases)\n{'='*78}")
    show = dl[dl.metric == "AUROC"]
    print(show[["arch", "comparison", "disease", "gain", "lo", "hi", "p_adj"]]
          .to_string(index=False, float_format=fmt))
    print(f"\n{'='*78}\n  Gain over MSP relative to DenseNet (AUROC)\n{'='*78}")
    print(ag[ag.metric == "AUROC"][["arch", "disease", "delta", "lo", "hi", "p_adj"]]
          .to_string(index=False, float_format=fmt))

    print(f"\n{'='*78}\n  PRE-REGISTERED CRITERION: Ensemble - MSP, AUROC gain > 0 at Holm p < 0.05\n{'='*78}")
    em = dl[(dl.metric == "AUROC") & (dl.comparison == "Ensemble - MSP") & (dl.disease != "AVERAGE")]
    sig = {a: set(g[(g.gain > 0) & (g.p_adj < 0.05)].disease) for a, g in em.groupby("arch")}
    for a in args.archs:
        held = sig[a] >= sig["densenet121"] if a != "densenet121" else True
        print(f"  {a:<12} significant on {len(sig[a])}/{len(args.diseases)}: "
              f"{sorted(sig[a]) or '-'}"
              + ("" if a == "densenet121" else
                 f"   -> {'HOLDS' if held else 'does NOT hold'} on every DenseNet disease"))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
