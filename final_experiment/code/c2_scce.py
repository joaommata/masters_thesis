# c2_scce.py
"""
Aim 2 baseline: SCCE -- abstention by distance to the counterfactuals, on OUR counterfactuals.

SCCE / SC-CE (Bonsignori, Punzi, Pellungrini & Giannotti, "'I know that I don't know... and I
explain why': Robust abstention via counterfactual explanations", IEEE Access 2026,
doi:10.1109/ACCESS.2026.3705102) is a post-hoc, model-agnostic selective classifier. Its
confidence score is the distance from an input to its counterfactual(s): if a small change flips
the prediction, the classifier should abstain. It does not use the model's confidence at all.

    python c2_scce.py --disease effusion --cf-source knn --cf-strategy correct_cf --cf-count 10

What the reference does
-----------------------
Code: github.com/valevalerio/scce-light, commit 7387193 (2026-10-06).
  scce_light/distances.py L23-25: euclidean(cfs, x), the L2 distance from x to each CF.
  scce_light/distances.py L28-33: aggregate(distances, how='min'), how in {min, mean, max}; +inf
    if no CF was found. The default is 'min'.
  scce_light/pipeline.py L37-40: cf_distances(X, cfs, distance=euclidean, how='min'), one
    aggregated distance per instance. Also defaults to 'min'.
  scce_light/rejectors.py L83-107: CFDistRejector. The score is the distance (higher = more
    trustworthy). calibrate() only sets thresholds, as empirical (or Gamma) quantiles of the
    calibration distances (L37-51), and accept() compares against them (L53-60). AUROC and
    E-AURC are threshold-free, so only the score matters here and nothing is calibrated.
  demo.ipynb: features are standardised (StandardScaler fit on the training split) before any
    distance, 32 CFs per instance, and all three aggregations are shown. The worked example
    (rejection explanation) uses 'CF min'.
  Original pipeline, github.com/clarapunzi/SC-CE @ 7cad644: evaluates min / mean / max over many
    metrics (4_selective_classifiers.py); its explanation script fixes aggr_function = 'min'
    (6_get_explanation.py L668).
  The paper's full text was not reachable from here (IEEE Xplore and OpenReview both need a
  browser). The rule followed: use the paper's main aggregation if it names one, else 'min'. The
  reference code defaults to 'min' everywhere, so 'min' is used: the distance to the NEAREST
  counterfactual.

The adaptation, and why
-----------------------
SCCE generates its counterfactuals with Growing Spheres in the raw feature space, on tabular
data. That cannot be transplanted here:
  * In pixel space Growing Spheres finds adversarial noise, not counterfactuals, at great cost.
  * In C0's embedding space the head is linear (logit = w.z + b), so the smallest flip distance
    is exactly |logit - logit(t)| / ||w||: a monotone function of |logit - logit(t)|. That ranks
    cases identically to the logit-shifted MSP already reported in the appendix (c2_msp.py,
    "logit" variant), so it would add nothing.
So SCCE's SCORE is computed on the counterfactuals this thesis already uses: the K=10
nearest-opposite-prediction KNN counterfactuals of the Aim 2 ensemble (knn_correct_cf), with
distances in exactly the space the retrieval searched -- the fold's standardised attribute
space, scaler fit on the fold's training rows (see below) -- under the retrieval's metric, read
from the pairing manifest (L1). It answers one question: does
counterfactual DISTANCE alone match the ensemble, which also learns from the counterfactual's
CONTENT?

The retrieval space is NOT today's FoldContext.attr_scaled_all
---------------------------------------------------------------
The knn_correct_cf pairings were built on 2026-09-09. On 2026-09-24 add_clinical_ratios
(code/c2/c2_prepare_data_simulated_cf.py) was changed: repair_bbox_ratios was added, and the
seven clinical ratios went from `num / (den + 1e-6)` to NaN on a zero denominator, filled with
0 afterwards. FoldContext imports the CURRENT function, so its attr_scaled_all is a different
space from the one kneighbors searched. Measured on effusion fold 0 (3,000 test rows), the
nearest-first order of the 10 cached CFs breaks in 2,838 rows with today's ratios, and in 0
rows with the pre-09-24 formula and today's column set (C0_DERIVED_COLS excluded, i.e.
`margin` out). So the space is rebuilt here exactly as the matcher built it on 09-09:
retrieval_ratios() below is the committed pre-09-24 add_clinical_ratios (git 4e267f0)
verbatim, then c2_folds._relevant_cols, fillna(0), and a StandardScaler fit on the fold's
training rows (compute_cf_for_split_correct_cf, L632-648 of the current file). The ordering
assert on every row of every fold is what proves it.
(The attribute models build their delta features with today's repaired ratios. That is a
separate, deliberate choice and does not change which counterfactuals they were given.)

Deviations from the reference: L1 instead of L2 (the retrieval's metric -- the distances must
reproduce the search, and the nearest-first check below proves they do); retrieved real images
instead of generated points; K=10 counterfactuals per query instead of 32. Nothing is learned or
tuned.

Checks: the folds are patient-disjoint, and the test paths and folds equal the ensemble's
oof_predictions.csv (as in c2_confidnet.py / c2_trustscore.py). Every test row carries exactly K
counterfactuals, and their K distances must be non-decreasing in cf_paths order -- the order
kneighbors returned them in. Exact ties are allowed and counted. This is what shows the
distances are the retrieval's own.

Outputs, in C2_final_results/<disease>/baselines/scce/
------------------------------------------------------
    fold_<i>_predictions.csv  path, patient_id, fold, <d>_prob, correct, scce_score, cf_dists
                              (the K distances, pipe-separated, nearest first)
    oof_predictions.csv       the five folds concatenated
    cv_summary.csv            per-fold and mean / std AUROC and E-AURC, adjusted MSP beside them
    run_config.json           reference, adaptation, ties, diagnostics, versions
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_folds as F                                            # noqa: E402
from c2_folds import (                                          # noqa: E402
    RESULTS_DIR, N_FOLDS, add_cf_args, cf_source_from_args,
)
from c2_confidnet import (                                      # noqa: E402
    c0_threshold, load_fold_membership, check_against_ensemble, aupr_error,
)
from c2_boot_extra import e_aurc                                # noqa: E402
from c2_msp import msp_score                                    # noqa: E402

SCORE = "scce_score"
AGGREGATION = "min"           # see the docstring: the reference's default
AGG = {"min": np.min, "mean": np.mean, "max": np.max}         # distances.py L33
TIE_RTOL = 1e-9               # c2_folds._is_tie's tolerance


def retrieval_ratios(df):
    """add_clinical_ratios as committed in git 4e267f0 and used by the 2026-09-09 pairing,
    verbatim (the +1e-6 denominators, no bbox-ratio repair). See the docstring."""
    df = df.copy()

    # Cardiothoracic ratio - most important for cardiomegaly. Normal < 0.5
    df["cardiothoracic_ratio"] = df["Heart_bbox_width"] / (
        df["Left Lung_bbox_width"] + df["Right Lung_bbox_width"]
    )

    # Lung symmetry - asymmetry can indicate effusion or pneumothorax
    df["lung_area_ratio"] = df["Left Lung_area_pixels"] / (df["Right Lung_area_pixels"] + 1e-6)
    df["lung_height_ratio"] = df["Left Lung_bbox_height"] / (df["Right Lung_bbox_height"] + 1e-6)
    df["lung_width_ratio"] = df["Left Lung_bbox_width"] / (df["Right Lung_bbox_width"] + 1e-6)

    # Lung area relative to total - captures hyperinflation/collapse
    total_area = df["Left Lung_area_pixels"] + df["Right Lung_area_pixels"]
    df["left_lung_fraction"] = df["Left Lung_area_pixels"] / (total_area + 1e-6)
    df["right_lung_fraction"] = df["Right Lung_area_pixels"] / (total_area + 1e-6)

    # Mediastinal width relative to lung width - widens in effusion/cardiomegaly
    df["mediastinal_ratio"] = df["Mediastinum_bbox_width"] / (
        df["Left Lung_bbox_width"] + df["Right Lung_bbox_width"] + 1e-6
    )

    return df


class RetrievalSpace:
    """The attribute space the matcher searched, for one fold: compute_cf_for_split_correct_cf
    with retrieval_ratios -- ratios, the matcher's column set, fillna(0), StandardScaler fit on
    the fold's training rows."""

    def __init__(self, attr, row_of, train_paths):
        self.row_of = row_of
        self.scaled = StandardScaler().fit(attr[F._row_ids(train_paths, row_of, "query")]) \
                                      .transform(attr)

    def l1_to_cfs(self, paths, cf_paths, k):
        """(n, k) L1 distances from each query to its k counterfactuals, in cf_paths order."""
        q = self.scaled[F._row_ids(paths, self.row_of, "query")]
        cf_rows, _, k_per_row = F._cf_row_ids(cf_paths, self.row_of)
        assert (k_per_row == k).all(), (f"rows carry between {k_per_row.min()} and "
                                        f"{k_per_row.max()} counterfactuals, expected {k}")
        cf = self.scaled[cf_rows].reshape(len(q), k, -1)
        return np.abs(cf - q[:, None, :]).sum(axis=2)


def retrieval_attributes(base, disease):
    """(attribute matrix, column names, path -> row) of the whole table, unscaled."""
    x = retrieval_ratios(base)
    rel = F._relevant_cols(x, disease)
    x[rel] = x[rel].fillna(0)
    attr = x[rel].values.astype(float)
    assert np.isfinite(attr).all(), "non-finite attribute values (a zero cardiothoracic denominator?)"
    return attr, rel, {p: i for i, p in enumerate(x["path"])}


def check_order(d):
    """Non-decreasing along each row up to floating-point ties. Returns (ties, rows with ties)."""
    step = np.diff(d, axis=1)
    tol = TIE_RTOL * np.maximum(1.0, np.abs(d[:, 1:]))
    bad = step < -tol
    if bad.any():
        r, c = np.argwhere(bad)[0]
        raise AssertionError(
            f"{int(bad.sum()):,} out-of-order CF pairs in {int(bad.any(axis=1).sum()):,} rows -- "
            f"the distances do not reproduce the retrieval. First: row {r}, slot {c} -> {c + 1}: "
            f"{d[r, c]!r} > {d[r, c + 1]!r}")
    tie = np.abs(step) <= tol
    return int(tie.sum()), int(tie.any(axis=1).sum())


def fold_metric(df, col, metric):
    vals = []
    for _, g in df.groupby("fold"):
        y, s = g["correct"].values, g[col].values
        vals.append(roc_auc_score(y, s) if metric == "auroc" else e_aurc(y, s))
    return float(np.mean(vals))


def run(args):
    src = cf_source_from_args(args)
    d, k = args.disease, args.cf_count
    prob = f"{d}_prob"
    out_dir = os.path.join(RESULTS_DIR, "C2_final_results", d, "baselines", "scce")
    if os.path.exists(os.path.join(out_dir, "cv_summary.csv")) and not args.overwrite:
        print(f"  cv_summary.csv already present in {out_dir}  (--overwrite to rebuild)")
        return
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*72}\n  SCCE (CF distance, {AGGREGATION}): {d} | {src.describe()} K={k}\n{'='*72}")
    metric = F._manifest_distance(d, src.name, k)
    assert metric == "l1", f"pairing manifest says distance={metric}; this script measures L1"
    assert src.aux_table(d, k) is None, "a generative CF source -- this baseline is KNN only"
    t = c0_threshold(d)

    # The base frame c2_folds.load_folds builds its folds from.
    base = F.load_c2_data(d, subset=src.subset_paths(d))
    base = base[[c for c in base.columns if not c.startswith("emb_")]]   # attributes only
    assert base["path"].is_unique
    redo = ((base[prob].values > t) == (base[f"{d}_true"].values == 1)).astype(int)
    assert (redo == base["correct"].values).all(), "correct != ((prob > t) == true)"

    folds, test_all = load_fold_membership(d, src, k, args.n_folds, base)
    ens_path = check_against_ensemble(test_all, d, src.name, k)
    print(f"  t = {t!r}; folds patient-disjoint; test rows == ensemble oof rows, same folds "
          f"({len(test_all):,}); distance = {metric}")

    meta = base.set_index("path")[["patient_id", prob, "correct"]]
    attr, rel, row_of = retrieval_attributes(base, d)
    print(f"  retrieval space: {len(rel)} attributes, pre-09-24 ratios, C0-derived columns out")
    rows, preds, ties = [], [], {}
    for i in range(args.n_folds):
        t0 = time.time()
        f = F.read_pairing(d, src.name, k, i)
        tr, te = f[f["split"] == "train"], f[f["split"] == "test"]
        assert set(te["path"]) == set(folds[i].loc[folds[i].split == "test", "path"])
        space = RetrievalSpace(attr, row_of, tr["path"].values)
        dist = space.l1_to_cfs(te["path"].values, te["cf_paths"].values, k)
        n_tie, n_tie_rows = check_order(dist)
        ties[i] = {"tied_adjacent_pairs": n_tie, "rows_with_a_tie": n_tie_rows}
        score = AGG[AGGREGATION](dist, axis=1)
        assert np.isfinite(score).all()

        p = meta.loc[te["path"].values].reset_index()
        p.insert(2, "fold", i)
        p[SCORE] = score
        p["cf_dists"] = ["|".join(f"{x:.10g}" for x in r) for r in dist]
        p = p[["path", "patient_id", "fold", prob, "correct", SCORE, "cf_dists"]]
        p.to_csv(os.path.join(out_dir, f"fold_{i}_predictions.csv"), index=False)
        preds.append(p)

        y = p["correct"].values
        msp = msp_score(p[prob].values, t, "threshold")
        rows.append({"fold": i, "n_test": len(p),
                     "auroc": float(roc_auc_score(y, score)),
                     "auroc_negated": float(roc_auc_score(y, -score)),
                     "e_aurc": e_aurc(y, score), "aupr_error": aupr_error(y, score),
                     "msp_auroc": float(roc_auc_score(y, msp)), "msp_e_aurc": e_aurc(y, msp),
                     "spearman_vs_msp": float(spearmanr(score, msp)[0]),
                     "tied_adjacent_pairs": n_tie, "rows_with_a_tie": n_tie_rows})
        r = rows[-1]
        print(f"  fold {i}: {len(p):,} test rows; order ok ({n_tie:,} exact ties in "
              f"{n_tie_rows:,} rows)  AUROC {r['auroc']:.4f} (adj. MSP {r['msp_auroc']:.4f})  "
              f"E-AURC {r['e_aurc']:.4f} (MSP {r['msp_e_aurc']:.4f})  [{time.time() - t0:.0f}s]",
              flush=True)
        del space

    oof = pd.concat(preds, ignore_index=True)
    assert set(oof["path"]) == set(test_all["path"]) and oof["path"].is_unique
    oof.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)

    summ = pd.DataFrame(rows)
    metrics = ["auroc", "auroc_negated", "e_aurc", "aupr_error", "msp_auroc", "msp_e_aurc",
               "spearman_vs_msp"]
    mean = {"fold": "mean", "n_test": int(summ.n_test.sum()),
            "tied_adjacent_pairs": int(summ.tied_adjacent_pairs.sum()),
            "rows_with_a_tie": int(summ.rows_with_a_tie.sum()),
            **{c: float(summ[c].mean()) for c in metrics}}
    std = {"fold": "std", **{c: float(summ[c].std(ddof=0)) for c in metrics}}
    summ = pd.concat([summ, pd.DataFrame([mean, std])], ignore_index=True)
    for c in ("n_test", "tied_adjacent_pairs", "rows_with_a_tie"):
        summ[c] = summ[c].astype("Int64")
    summ.insert(0, "model", "SCCE")
    summ.to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)

    diag = diagnostics(d, src.name, k, oof, summ, t)

    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump({
            "disease": d, "model": f"SCCE score (CF distance, {AGGREGATION})",
            "reference": "Bonsignori et al., IEEE Access 2026, doi:10.1109/ACCESS.2026.3705102; "
                         "github.com/valevalerio/scce-light @ 7387193 (distances.py L23-33, "
                         "pipeline.py L37-40, rejectors.py L83-107)",
            "adaptation": "SCCE's score on the ensemble's KNN counterfactuals instead of "
                          "Growing Spheres; L1 in the retrieval's standardised attribute space",
            "aggregation": AGGREGATION, "distance": metric,
            "space": "the matcher's attribute space as of the 2026-09-09 pairing: pre-09-24 "
                     "add_clinical_ratios (git 4e267f0), c2_folds._relevant_cols, fillna(0), "
                     "StandardScaler fit on the fold's train rows. NOT today's "
                     "FoldContext.attr_scaled_all, whose ratios changed on 2026-09-24",
            "n_attributes": len(rel),
            "score_direction": "higher distance = more confident",
            "calibration": "none -- CFDistRejector.calibrate only sets thresholds; AUROC and "
                           "E-AURC are threshold-free",
            "threshold_t": t,
            "fold_source": {"cf_source": src.name, "cf_count": k, "n_folds": args.n_folds,
                            "matched_ensemble_oof": ens_path},
            "order_check": {"tolerance_rel": TIE_RTOL, "per_fold": ties},
            "n_rows": int(len(oof)),
            "diagnostics": diag,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "sklearn": sklearn.__version__},
        }, fh, indent=2)
    print(f"\n  wrote {out_dir}")


def other_methods(disease, source_name, k):
    """{name: (frame with fold/correct, score column)} for the methods SCCE is set beside,
    each read from its own out-of-fold predictions. Missing ones are skipped."""
    c2 = os.path.join(RESULTS_DIR, "C2_final_results", disease)
    out = {}
    attr = [os.path.join(c2, source_name, f"cf_{k}", "attributes", f"fold_{i}_predictions.csv")
            for i in range(N_FOLDS)]
    if all(os.path.exists(p) for p in attr):
        a = pd.concat([pd.read_csv(p, usecols=["path", "correct", "MLP_I_prob"]).assign(fold=i)
                       for i, p in enumerate(attr)], ignore_index=True)
        out["MLP_I"] = (a, "MLP_I_prob")
    for name, sub, col in (("ConfidNet", "confidnet", "confidnet_score"),
                           ("TrustScore", "trustscore", "trustscore")):
        p = os.path.join(c2, "baselines", sub, "oof_predictions.csv")
        if os.path.exists(p):
            out[name] = (pd.read_csv(p, usecols=["path", "fold", "correct", col]), col)
    meta = os.path.join(RESULTS_DIR, "C2_final_results", "bootstrap_meta.json")
    p = os.path.join(c2, source_name, f"cf_{k}", "ensemble", "oof_predictions.csv")
    if os.path.exists(meta) and os.path.exists(p):
        best = json.load(open(meta))["picks"][disease]["best_all"]
        if best.startswith("ENS "):
            col = "ens_" + best[len("ENS "):].rsplit("@", 1)[0]
            out[f"ensemble ({best})"] = (pd.read_csv(p, usecols=["path", "fold", "correct", col]),
                                         col)
    return out


def diagnostics(disease, source_name, k, oof, summ, t):
    """Direction; mean CV AUROC and E-AURC beside the other methods; Spearman vs adjusted MSP."""
    m = summ[summ.fold == "mean"].iloc[0]
    out = {}
    print(f"\n{'='*72}\n  DIAGNOSTICS\n{'='*72}")
    ok = bool(m["auroc"] > m["auroc_negated"])
    out["direction"] = {"auroc": float(m["auroc"]), "auroc_negated": float(m["auroc_negated"]),
                        "ok": ok}
    print(f"  1. direction  mean-fold AUROC distance {m['auroc']:.4f}  vs  -distance "
          f"{m['auroc_negated']:.4f}   " + ("ok (larger distance = more confident)" if ok else
          "!! the NEGATED distance ranks better -- reported as is, not flipped"))

    pooled = float(spearmanr(oof[SCORE], msp_score(oof[f"{disease}_prob"].values, t,
                                                   "threshold"))[0])
    out["spearman_vs_msp"] = {"pooled": pooled, "mean_fold": float(m["spearman_vs_msp"])}
    print(f"  2. Spearman(scce_score, adjusted MSP): pooled {pooled:.3f}, mean over folds "
          f"{m['spearman_vs_msp']:.3f}")

    table = {"SCCE": (float(m["auroc"]), float(m["e_aurc"])),
             "adjusted MSP": (float(m["msp_auroc"]), float(m["msp_e_aurc"]))}
    for name, (df, col) in other_methods(disease, source_name, k).items():
        df = df.set_index("path").reindex(oof["path"])
        assert df[col].notna().all() and (df["fold"].values == oof["fold"].values).all(), name
        df = df.reset_index()
        table[name] = (fold_metric(df, col, "auroc"), fold_metric(df, col, "e_aurc"))
    out["mean_cv"] = {n: {"auroc": a, "e_aurc": e} for n, (a, e) in table.items()}
    print("  3. mean CV        AUROC ↑     E-AURC ↓")
    for n, (a, e) in table.items():
        print(f"     {n:<26} {a:.4f}     {e:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if cv_summary.csv already exists")
    add_cf_args(ap)
    ap.set_defaults(cf_count=10)
    args = ap.parse_args()
    if args.n_folds != N_FOLDS:
        print(f"  note: --n-folds {args.n_folds} (default {N_FOLDS})")
    run(args)


if __name__ == "__main__":
    main()
