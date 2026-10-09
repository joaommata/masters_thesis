"""Export a curated case set for the static demo viewer in final_experiment/demo/.

Everything the viewer shows is precomputed here, because a diffusion counterfactual costs
GPU-minutes per image. Per disease (default effusion), on the 20k diffusion-CF subset:

* C2 score  = the out-of-fold rank ensemble (attribute MLP I+C+D + dual_sal CNN) at K=1,
              read from C2_final_results/<disease>/diffusion_hml_sub20k/cf_1/ensemble/.
* MSP       = the threshold variant (p if p >= t else 1-p), as in C2_evaluation.
* Triage    = accuracy of the accepted calls when the least-trusted fraction is rejected,
              for both scores, over all 20k rows (not only the curated cases).
* Cases     = N_PER_CELL per C0 outcome (TP/FP/TN/FN): the cases where C2 and MSP disagree
              most in both directions, plus random ones, so the set is not cherry-picked
              in C2's favour.
* Per case  = the sampler's input x0, the low/mid/high counterfactuals, a signed
              difference overlay per level, the C0 probability trajectory of every level,
              and a dumbbell of the side's top attributes (image vs high-level CF).

The attribute selection mirrors attribute_explainability.ipynb (per side of C0's call,
|r| with correctness over image / CF / delta forms, greedy non-redundant), but on the
diffusion arm with the high level as the counterfactual, since that is slot 0 at K=1.

Output (not committed, CheXpert images are under a data-use agreement):
    final_experiment/demo/data.js        window.DEMO = {...}
    final_experiment/demo/img/<id>_*.png

Usage:
    python final_experiment/code/demo_export.py [--disease effusion] [--per-cell 10]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_msp import msp_score                                        # noqa: E402

REPO    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
sys.path.insert(0, os.path.join(REPO, "code", "c2"))
from c2_feature_spec import C0_DERIVED_COLS, RSNA_META_COLS         # noqa: E402

LEVELS    = ["low", "mid", "high"]         # cf_level_1..3 in the npz, weakest edit first
SOURCE    = "diffusion_hml_sub20k"
SCORE_COL = "ens_rank"
N_ATTR, MAX_REDUNDANCY = 6, 0.6
ORANGE, BLUE = (214, 117, 35), (40, 99, 171)   # thesis FAMILY traj / attr: brighter / darker


def short(f):
    """'Heart_original_firstorder_Mean' -> 'Heart mean' (as in the explainability notebook)."""
    s = f.replace("_original_", " ").replace("firstorder_", "").replace("shape2D_", "shape ")
    return s.replace("_", " ")


def triage_curve(score, correct, n=101):
    """Accuracy of the kept calls when only the top-`coverage` fraction by score is kept."""
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(correct[order])
    cov = np.linspace(0.02, 1.0, n)
    k = np.maximum(1, np.round(cov * len(score)).astype(int))
    return [round(float(x), 4) for x in cum[k - 1] / k], [round(float(c), 3) for c in cov]


def num(x):
    """Rounded float, or None for a missing value (JSON has no NaN)."""
    return None if pd.isna(x) else round(float(x), 3)


def per_fold_auc(df, col):
    a = [roc_auc_score(g["correct"], g[col]) for _, g in df.groupby("fold")]
    return round(float(np.mean(a)), 4), round(float(np.std(a)), 4)


def to_png(arr01, path):
    Image.fromarray((np.clip(arr01, 0, 1) * 255).round().astype(np.uint8), "L").save(path)


def diff_png(diff, vmax, path):
    """Signed change as RGBA: orange where the edit brightened, blue where it darkened."""
    a = np.clip((np.abs(diff) / vmax - 0.15) / 0.85, 0, 1) ** 0.8   # drop the faint haze
    rgba = np.zeros(diff.shape + (4,), np.uint8)
    rgba[diff > 0, :3], rgba[diff <= 0, :3] = ORANGE, BLUE
    rgba[..., 3] = (a * 235).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", default="effusion")
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(REPO, "final_experiment", "demo"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    D, rng = args.disease, np.random.default_rng(args.seed)
    cfl = os.path.join(RESULTS, "CF_levels", D)
    os.makedirs(os.path.join(args.out, "img"), exist_ok=True)

    # ── scores: OOF C2 ensemble + threshold MSP, one row per query of the 20k subset ──
    t = float(open(os.path.join(RESULTS, "C0_final", "multilabel_ignore", D, "threshold.txt")).read())
    oof = pd.read_csv(os.path.join(RESULTS, "C2_final_results", D, SOURCE, "cf_1", "ensemble",
                                   "oof_predictions.csv"))
    idx = pd.read_csv(os.path.join(cfl, "cf_index.csv"))
    q = (idx.drop_duplicates("query_path")
            .set_index("query_path")[["key", "true", "orig_prob", "orig_pred", "cell", "direction"]])
    df = oof.merge(q, left_on="path", right_index=True, how="inner", validate="1:1")
    assert len(df) == len(oof), f"{len(oof) - len(df)} OOF rows have no CF index entry"
    # `cell` and `correct` come from the pipeline; a row within float noise of t can disagree
    # with orig_pred (one on effusion), so the call is taken from the cell.
    df["orig_pred"] = df["cell"].isin(["TP", "FP"]).astype(int)
    assert (df["correct"] == (df["orig_pred"] == df["true"]).astype(int)).all()
    assert ((df["orig_prob"] >= t).astype(int) != df["orig_pred"]).sum() <= 5, "pred != (p >= t)"
    df["msp"] = msp_score(df["orig_prob"].values, t, "threshold")
    df["c2"] = df[SCORE_COL]
    for s in ("c2", "msp"):          # percentile among calls of the same kind (C2 is uncalibrated)
        df[f"{s}_pct"] = df.groupby("orig_pred")[s].rank(pct=True)

    y = df["correct"].values
    acc_c2, cov = triage_curve(df["c2"].values, y)
    acc_msp, _ = triage_curve(df["msp"].values, y)
    acc_oracle, _ = triage_curve(y + 1e-9 * rng.random(len(y)), y)

    # ── per-CF C0 probabilities, recomputed on the PNGs ──
    pairs = pd.read_csv(os.path.join(cfl, "cf_pairs.csv"))
    cf_prob = pairs.pivot(index="query_path", columns="level_name", values="prob")[LEVELS]
    flipped = pairs.pivot(index="query_path", columns="level_name", values="flipped")[LEVELS]

    # ── attributes: query from c2_data.csv, counterfactual = high level from the CF table ──
    cf_tab = pd.read_parquet(os.path.join(cfl, "c2_cf_table.parquet"))
    meta = {"prob", "pred", "true", "correct", "path", "cam_path", "patient_id",
            *C0_DERIVED_COLS, *RSNA_META_COLS}
    attrs = [c for c in cf_tab.columns if c not in meta and not c.startswith("emb_")]
    cf_tab = cf_tab[cf_tab["path"].str.contains("/high/")].copy()
    cf_tab["query_path"] = cf_tab["path"].map(
        pairs[pairs.level_name == "high"].set_index("cf_path")["query_path"])
    cf_A = cf_tab.set_index("query_path")[attrs].loc[df["path"]].astype(float)

    print("reading c2_data.csv attributes ...", flush=True)
    wanted = set(df["path"])
    chunks = pd.read_csv(os.path.join(RESULTS, "C2_final", D, "c2_data.csv"),
                         usecols=["path"] + attrs, chunksize=20000)
    img_A = pd.concat(c[c["path"].isin(wanted)] for c in chunks).set_index("path")
    img_A = img_A.loc[df["path"], attrs].astype(float)

    # Unsegmented organs leave NaNs (~12% of cells); keep attributes that are nearly complete.
    ok = img_A.std().gt(0) & img_A.isna().mean().lt(0.05) & cf_A.isna().mean().lt(0.05)
    attrs = [a for a in attrs if ok[a]]
    mu, sd = img_A[attrs].mean(), img_A[attrs].std()
    zi = ((img_A[attrs] - mu) / sd).set_axis(df.index)
    zc = ((cf_A[attrs] - mu) / sd).set_axis(df.index)
    zd = zi - zc

    selected, side_stats = {}, {}
    for p in (0, 1):
        m = (df["orig_pred"] == p).values
        yc = df.loc[m, "correct"]
        r = pd.DataFrame({f: z.loc[m].corrwith(yc) for f, z in
                          (("image", zi), ("cf", zc), ("delta", zd))}).fillna(0)
        best = r.abs().max(axis=1).sort_values(ascending=False)
        kept = []
        for a in best.index:
            if all(abs(zi.loc[m, a].corr(zi.loc[m, s])) < MAX_REDUNDANCY for s in kept):
                kept.append(a)
            if len(kept) == N_ATTR:
                break
        selected[p] = kept
        right, wrong = m & (df["correct"] == 1).values, m & (df["correct"] == 0).values
        side_stats[p] = [{
            "name": short(a), "raw": a,
            "form": r.loc[a].abs().idxmax(), "r": round(float(r.loc[a, r.loc[a].abs().idxmax()]), 3),
            "right": [round(float(zi.loc[right, a].mean()), 3), round(float(zc.loc[right, a].mean()), 3)],
            "wrong": [round(float(zi.loc[wrong, a].mean()), 3), round(float(zc.loc[wrong, a].mean()), 3)],
        } for a in kept]
        print(f"side {p}: " + ", ".join(f"{short(a)} ({s['form']} r={s['r']})"
                                        for a, s in zip(kept, side_stats[p])))

    # ── curate cases: per outcome cell, strongest disagreements both ways + random ──
    df["gap"] = df["c2_pct"] - df["msp_pct"]
    picks = []
    n_gap = max(1, args.per_cell * 3 // 10)
    for cell in ("TP", "FP", "TN", "FN"):
        g = df[df["cell"] == cell]
        hi, lo = list(g.nlargest(n_gap, "gap").index), list(g.nsmallest(n_gap, "gap").index)
        rest = g.drop(hi + lo)
        rnd = list(rng.choice(rest.index, args.per_cell - len(hi + lo), replace=False))
        picks += ([(i, cell, "c2_trusts_more") for i in hi] + [(i, cell, "msp_trusts_more") for i in lo]
                  + [(i, cell, "random") for i in rnd])

    # One shared overlay scale so the strength of an edit is comparable across cases.
    raws = {}
    for i, _, _ in picks:
        z = np.load(os.path.join(cfl, "cf", df.at[i, "key"] + ".npz"))
        raws[i] = {"x0": z["x0"], **{L: z[f"cf_level_{j + 1}"] for j, L in enumerate(LEVELS)}}
    # Two views of the change. "vs input" (level - x0) is dominated by the diffusion
    # reconstruction, which every level shares; "guided" (level - low) isolates what the stronger
    # push adds and is ~14x smaller, so each view gets its own shared scale.
    vmax = float(np.percentile(np.concatenate(
        [np.abs(r["high"] - r["x0"]).ravel() for r in raws.values()]), 99.5))
    vmax_g = float(np.percentile(np.concatenate(
        [np.abs(r["high"] - r["low"]).ravel() for r in raws.values()]), 99.5))

    cases = []
    for n, (i, cell, kind) in enumerate(picks):
        row, cid, qp = df.loc[i], f"c{n:03d}", df.at[i, "path"]
        r = raws[i]
        to_png(r["x0"], os.path.join(args.out, "img", f"{cid}_x0.png"))
        for L in LEVELS:
            to_png(r[L], os.path.join(args.out, "img", f"{cid}_{L}.png"))
            diff_png(r[L] - r["x0"], vmax, os.path.join(args.out, "img", f"{cid}_{L}_d.png"))
            if L != "low":
                diff_png(r[L] - r["low"], vmax_g, os.path.join(args.out, "img", f"{cid}_{L}_g.png"))
        tr = np.load(os.path.join(cfl, "trajectory", row["key"] + ".npz"))
        p = int(row["orig_pred"])
        cases.append({
            "id": cid, "cell": cell, "pred": p, "true": int(row["true"]),
            "correct": int(row["correct"]), "fold": int(row["fold"]),
            "study": "/".join(qp.split("/")[2:4]),
            "prob": round(float(row["orig_prob"]), 4),
            "c2_pct": round(float(row["c2_pct"]), 4), "msp_pct": round(float(row["msp_pct"]), 4),
            "msp": round(float(row["msp"]), 4),
            "levels": [{"name": L, "prob": round(float(cf_prob.at[qp, L]), 4),
                        "flipped": int(flipped.at[qp, L]),
                        "change": round(float(np.abs(r[L] - r["x0"]).mean()), 5),
                        "guided": round(float(np.abs(r[L] - r["low"]).mean()), 5),
                        "traj": [[int(a), round(float(b), 4)] for a, b in tr[f"level_{j + 1}"]]}
                       for j, L in enumerate(LEVELS)],
            "attrs": [[num(zi.at[i, a]), num(zc.at[i, a])] for a in selected[p]],
            "pick": kind,
        })

    lv = json.load(open(os.path.join(cfl, "levels.json")))
    auc_c2, auc_msp = per_fold_auc(df, "c2"), per_fold_auc(df, "msp")
    out = {
        "disease": D, "threshold": round(t, 4), "n": int(len(df)),
        "accuracy": round(float(y.mean()), 4),
        "cells": df["cell"].value_counts().to_dict(),
        "auc": {"c2": auc_c2, "msp": auc_msp},
        "triage": {"coverage": cov, "c2": acc_c2, "msp": acc_msp, "oracle": acc_oracle},
        "targets": {"to_positive": lv["to_positive"], "to_negative": lv["to_negative"]},
        "tau": lv["tau"], "overlay_vmax": round(vmax, 4), "guided_vmax": round(vmax_g, 4),
        "attributes": {str(p): side_stats[p] for p in (0, 1)},
        "cases": cases,
    }
    with open(os.path.join(args.out, "data.js"), "w") as f:
        f.write("window.DEMO = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"{len(cases)} cases -> {args.out}  |  AUROC C2 {auc_c2[0]:.4f}  MSP {auc_msp[0]:.4f}")


if __name__ == "__main__":
    main()
