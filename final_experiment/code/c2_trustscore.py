# c2_trustscore.py
"""
Aim 2 baseline: TrustScore -- the distance-based confidence signal, with NO counterfactual.

TrustScore (Jiang, Kim, Guan & Gupta, "To Trust Or Not To Trust A Classifier", NeurIPS 2018)
scores a prediction by how much closer the test point is to the training points of the class
it was given than to those of any other class:

    trust(x) = d(x, H_alpha(other class)) / d(x, H_alpha(predicted class))

where H_alpha(c) is the class-c training set with its alpha-fraction of lowest-density points
removed. Higher = more trustworthy. It is the main prior art for "distance to the other class
as a confidence signal", the nearest relative of the nearest-opposite-neighbour counterfactuals.
It is not learned, so there is no training loop: each fold's training rows are the reference
set, and the test rows are scored against it.

    python c2_trustscore.py --disease effusion --cf-source knn --cf-strategy correct_cf --cf-count 10

As in c2_confidnet.py, the --cf-* arguments only name the cached pairing whose fold membership
is reused (knn_correct_cf, K=10, the Aim 2 ensemble's). No counterfactual column is read. Test
paths and fold numbers are asserted equal to the ensemble's oof_predictions.csv.

What the reference does
-----------------------
Code: github.com/google/TrustScore, commit 07a73be (2021-11-04), trustscore/trustscore.py.
  TrustScore.__init__, L24: defaults k=10, alpha=0.0, filtering="none", min_dist=1e-12.
  filter_by_density, L36-49: per class, KDTree(X).query(X, k=self.k)[0][:, -1] -- the k-NN
    radius with the query point itself counted as its own first neighbour (X is queried against
    itself), i.e. the distance to the (k-1)-th OTHER point; eps = np.percentile(radii,
    (1 - alpha) * 100); keep the points with radius <= eps. So alpha is the fraction of
    lowest-density (largest-radius) points removed, per class.
  fit, L68-98: one KDTree per class on that class's (filtered) training points; class labels
    are the ground-truth training labels. With filtering="none" nothing is removed.
  get_score, L100-122: distance to a class = kdtrees[label].query(X, k=2)[0][:, -1], i.e. the
    SECOND-nearest neighbour, not the nearest (the paper says nearest; the code takes k=2 so
    that scoring a training point does not match itself, and uses it for test points too).
    d_to_closest_not_pred = the smallest distance among the other classes;
    score = d_to_closest_not_pred / (d_to_pred + min_dist).
  The `k` parameter therefore only matters to the density filter; scoring always uses the 2nd
  neighbour. trustscore_evaluation.py imports PCA (L18) but never calls it.
Paper (arXiv 1805.11783 v2):
  Sec. 3, Algorithms 1-2: the same density filter (k-NN radius, drop alpha * n points with the
    largest radius) and ratio.
  Sec. 5, "Choosing Hyperparameters": "we fix k = 10 and choose alpha using cross-validation";
    for MNIST/SVHN/CIFAR they "skipped the initial filtering step ... and reduced the
    intermediate layers down to 20 dimensions using PCA", Euclidean distance throughout.
  Sec. 5.3: on CNN layers (logits, penultimate, penultimate + PCA-20) "the trust score may only
    provide little or no improvement over the model confidence".

What is used here
-----------------
Features: emb_0..emb_1023 of c2_data.csv (C0's penultimate DenseNet-121 features), PCA fit on
  the reference rows only (centred, not whitened, svd_solver="full"), Euclidean distance.
Reference set: the fold's training rows, labelled with ground truth <disease>_true. Test rows
  are never in it.
Predicted class: C0's decision at its threshold t, (prob > t), asserted equal to <disease>_pred.
Score: the reference's get_score, verbatim (2nd-nearest neighbour, closest non-predicted class,
  min_dist = 1e-12). Higher = more trustworthy.
Hyperparameters, pre-declared and chosen per fold on inner validation: the fold's training side
  is split 90/10 by patient (GroupShuffleSplit, seed 42); the 10% is scored with the 90% as the
  reference set, and the setting with the best AUROC for predicting `correct` is kept (ties ->
  the first in grid order). Grid:
      alpha        {0, 0.05, 0.1}            density filter, as in the paper's CV
      k            {10}                       {reference default, 10}; the default IS 10
      PCA dims     {32, 64, 128}              paper reduces CNN layers to 20 with PCA
  Then the PCA and the reference set are refit on the full fold training side and the test fold
  is scored once. The chosen settings and every inner AUROC go to run_config.json.

Deviations from the reference, and why
--------------------------------------
1. Exact brute-force nearest neighbours (sklearn NearestNeighbors(algorithm="brute"), float64)
   instead of KDTree: same neighbours and distances up to floating-point rounding, much faster
   than a KD-tree above ~20 dimensions. test_against_reference() checks it against a verbatim
   copy of the reference class on random data at every run.
2. alpha = 0 means filtering="none" (identical: the 100th percentile keeps every point); alpha
   > 0 is filtering="density". The "uncertainty" filter is not used (not in the paper).
3. The paper selects alpha by precision at a recall level; here AUROC for `correct` is used, the
   metric every Aim 2 model is compared on. PCA dims come from the task grid, not the paper's 20.
4. The 2-class case only, so "closest non-predicted class" is simply the other class.

Outputs, in C2_final_results/<disease>/baselines/trustscore/
-------------------------------------------------------------
    fold_<i>_predictions.csv  path, patient_id, fold, <d>_prob, <d>_pred, correct, d_class0,
                              d_class1, d_pred, d_other, trustscore
    oof_predictions.csv       the five folds concatenated (what c2_trustscore_compare.py reads)
    cv_summary.csv            per-fold and mean AUROC / AUPR-Error / E-AURC, adjusted MSP beside
    run_config.json           grid, inner results, chosen settings per fold, diagnostics, versions
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
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KDTree, NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import (                                          # noqa: E402
    RESULTS_DIR, RANDOM_SEED, N_FOLDS, load_c2_data, add_cf_args, cf_source_from_args,
)
from c2_confidnet import (                                      # noqa: E402
    c0_threshold, load_fold_membership, check_against_ensemble, aupr_error,
)
from c2_boot_extra import e_aurc                                # noqa: E402
from c2_msp import msp_score                                    # noqa: E402

from c0_arch import n_emb                                       # noqa: E402

# C0 embedding width: 1024 for DenseNet, else from the arch root's C0 (c0_arch.py).
N_EMB = n_emb(RESULTS_DIR)
SCORE = "trustscore"
MIN_DIST = 1e-12              # trustscore.py L24
ALPHAS = (0.0, 0.05, 0.1)
KS = (10,)                    # {reference default (10), 10}
PCA_DIMS = (32, 64, 128)
INNER_VAL = 0.1


# ══════════════════════════════════════════════════════════════════════════════
# TRUST SCORE  (trustscore.py, KDTree -> exact brute force)
# ══════════════════════════════════════════════════════════════════════════════

def knn_dist(ref, query, n, n_jobs=-1):
    """Distances from each query row to its n nearest reference rows, ascending. Exact."""
    nn = NearestNeighbors(n_neighbors=n, algorithm="brute", metric="euclidean", n_jobs=n_jobs)
    return nn.fit(ref).kneighbors(query, n_neighbors=n, return_distance=True)[0]


def knn_radii(X, k):
    """filter_by_density L46-47: k-NN radius of each point, the point itself counted (X is
    queried against itself, so column 0 is the self-match at distance 0)."""
    return knn_dist(X, X, k)[:, -1]


def density_keep(radii, alpha):
    """filter_by_density L48-49: keep the points whose radius is within the (1-alpha) quantile."""
    eps = np.percentile(radii, (1 - alpha) * 100)
    return radii <= eps


def class_distance(ref, query):
    """get_score L115: distance to the SECOND-nearest point of the class set."""
    return knn_dist(ref, query, 2)[:, -1]


def ratio(d, y_pred, min_dist=MIN_DIST):
    """get_score L117-122, verbatim. d is (n, n_labels)."""
    sorted_d = np.sort(d, axis=1)
    d_to_pred = d[range(d.shape[0]), y_pred]
    d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1])
    return d_to_closest_not_pred / (d_to_pred + min_dist)


class Reference:
    """The fitted class sets for one feature space. Radii are computed once per k, so the
    alpha grid costs one threshold each (filter_by_density recomputes them per call)."""

    def __init__(self, Z, y):
        y = np.asarray(y)
        assert np.isin(y, (0, 1)).all()
        self.sets = [Z[y == c] for c in (0, 1)]
        assert all(len(s) for s in self.sets), "a class is missing from the reference set"
        self._radii = {}

    def class_sets(self, k, alpha):
        if alpha == 0:                       # filtering="none" (fit L84-86)
            return self.sets
        out = []
        for c, s in enumerate(self.sets):
            if (c, k) not in self._radii:
                self._radii[(c, k)] = knn_radii(s, k)
            keep = density_keep(self._radii[(c, k)], alpha)
            assert keep.any(), "Filtered too much"            # fit L94-98
            out.append(s[keep])
        return out

    def score(self, Zq, y_pred, k, alpha):
        d = np.column_stack([class_distance(s, Zq) for s in self.class_sets(k, alpha)])
        return ratio(d, np.asarray(y_pred, int)), d


class _ReferenceTrustScore:
    """github.com/google/TrustScore trustscore/trustscore.py L19-122, verbatim minus the
    'uncertainty' branch -- used only by test_against_reference()."""

    def __init__(self, k=10, alpha=0.0, filtering="none", min_dist=1e-12):
        self.k, self.filtering, self.alpha, self.min_dist = k, filtering, alpha, min_dist

    def filter_by_density(self, X):
        kdtree = KDTree(X)
        knn_radii = kdtree.query(X, k=self.k)[0][:, -1]
        eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
        return X[np.where(knn_radii <= eps)[0], :]

    def fit(self, X, y):
        self.n_labels = np.max(y) + 1
        self.kdtrees = [None] * self.n_labels
        for label in range(self.n_labels):
            if self.filtering == "none":
                X_to_use = X[np.where(y == label)[0]]
            elif self.filtering == "density":
                X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
            self.kdtrees[label] = KDTree(X_to_use)

    def get_score(self, X, y_pred):
        d = np.tile(None, (X.shape[0], self.n_labels))
        for label_idx in range(self.n_labels):
            d[:, label_idx] = self.kdtrees[label_idx].query(X, k=2)[0][:, -1]
        sorted_d = np.sort(d, axis=1)
        d_to_pred = d[range(d.shape[0]), y_pred]
        d_to_closest_not_pred = np.where(
            sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1])
        return d_to_closest_not_pred / (d_to_pred + self.min_dist)


def test_against_reference():
    """Brute force + cached radii must reproduce the reference class. Runs every time."""
    rng = np.random.default_rng(0)
    for dim in (2, 16, 64):
        X = rng.normal(size=(1500, dim))
        y = (X[:, 0] + 0.7 * rng.normal(size=1500) > 0.3).astype(int)
        Q = rng.normal(size=(400, dim))
        yp = (Q[:, 0] > 0).astype(int)
        ref = Reference(X, y)
        for k in (5, 10):
            for alpha in (0.0, 0.05, 0.1):
                theirs = _ReferenceTrustScore(k=k, alpha=alpha,
                                              filtering="density" if alpha else "none")
                theirs.fit(X, y)
                want = theirs.get_score(Q, yp).astype(float)
                got, d = ref.score(Q, yp, k, alpha)
                assert np.allclose(got, want, rtol=1e-9, atol=0), (dim, k, alpha)
                # the predicted class's distance is the denominator, for both predictions
                assert np.allclose(got, d[np.arange(len(yp)), 1 - yp]
                                   / (d[np.arange(len(yp)), yp] + MIN_DIST), rtol=1e-12)
    # alpha = 0 with filtering="density" keeps every point, i.e. equals "none"
    r = rng.random(100)
    assert density_keep(r, 0.0).all()


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_inputs(disease, t):
    """c2_data.csv restricted to what TrustScore may see. The asserts are c2_confidnet's, plus
    the predicted class: (prob > t) must be C0's stored prediction."""
    df = load_c2_data(disease)
    prob, pred, true = f"{disease}_prob", f"{disease}_pred", f"{disease}_true"
    emb = [f"emb_{i}" for i in range(N_EMB)]
    missing = [c for c in ["path", "patient_id", prob, pred, true, "correct"] + emb if c not in df]
    assert not missing, f"c2_data.csv lacks {missing[:5]}"
    assert not [c for c in df.columns if c.startswith("emb_") and c not in set(emb)], \
        f"c2_data.csv has more than {N_EMB} emb_* columns"
    df = df[["path", "patient_id", prob, pred, true, "correct"] + emb]
    assert not any(c.startswith(("cf_", "delta_")) for c in df.columns)
    assert df["path"].is_unique, "duplicate paths in c2_data.csv"
    assert df[emb].notna().all().all(), "NaN in the embeddings"

    y = df[true].values
    assert np.isin(y, (0, 1)).all(), f"{true} is not 0/1 after dropping undefined rows"
    hard = (df[prob].values > t).astype(int)
    bad = int((hard != df[pred].values).sum())
    assert bad == 0, (f"{disease}: (prob > t) != {pred} on {bad:,} rows at t={t!r} -- "
                      f"the stored prediction is not C0's decision at t")
    bad = int((((hard == y)).astype(int) != df["correct"].values).sum())
    assert bad == 0, f"{disease}: correct != ((prob > t) == true) on {bad:,} rows"
    df[true] = y.astype(int)
    df[pred] = hard
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PER FOLD
# ══════════════════════════════════════════════════════════════════════════════

def pca_fit(X, n):
    return PCA(n_components=n, svd_solver="full", random_state=RANDOM_SEED).fit(X)


def inner_select(X_tr, y_tr, pred_tr, correct_tr, groups):
    """Score the patient-disjoint 10% with the 90% as reference, for every grid point."""
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=INNER_VAL, random_state=RANDOM_SEED)
                    .split(X_tr, correct_tr, groups))
    assert not set(groups[fit]) & set(groups[val]), "inner val shares patients with fit"
    pca = pca_fit(X_tr[fit], max(PCA_DIMS))   # full SVD: the first n comps = PCA(n)
    Zf, Zv = pca.transform(X_tr[fit]), pca.transform(X_tr[val])
    res = []
    for dim in PCA_DIMS:
        ref = Reference(Zf[:, :dim], y_tr[fit])
        for k in KS:
            for alpha in ALPHAS:
                s, _ = ref.score(Zv[:, :dim], pred_tr[val], k, alpha)
                res.append({"pca": dim, "k": k, "alpha": alpha,
                            "val_auroc": float(roc_auc_score(correct_tr[val], s))})
                print(f"    inner  pca {dim:>3}  k {k}  alpha {alpha:<4}  val AUROC "
                      f"{res[-1]['val_auroc']:.4f}", flush=True)
    best = max(res, key=lambda r: r["val_auroc"])   # first max = first in grid order
    return best, res, (len(fit), len(val))


def decile_agreement(score, msp):
    """Share of TrustScore's top (bottom) decile that is also adjusted MSP's top (bottom)
    decile. Chance is 0.10."""
    n = len(score)
    m = int(round(0.1 * n))
    a = np.argsort(score, kind="mergesort")
    b = np.argsort(msp, kind="mergesort")
    top = len(np.intersect1d(a[-m:], b[-m:])) / m
    bot = len(np.intersect1d(a[:m], b[:m])) / m
    return top, bot


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    test_against_reference()
    src = cf_source_from_args(args)
    d = args.disease
    prob, pred = f"{d}_prob", f"{d}_pred"
    out_dir = os.path.join(RESULTS_DIR, "C2_final_results", d, "baselines", "trustscore")
    if os.path.exists(os.path.join(out_dir, "cv_summary.csv")) and not args.overwrite:
        print(f"  cv_summary.csv already present in {out_dir}  (--overwrite to rebuild)")
        return
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*72}\n  TRUSTSCORE (frozen C0): {d} | folds of {src.describe()} "
          f"K={args.cf_count}\n{'='*72}")
    t = c0_threshold(d)
    print(f"  t = {t!r}   reference check vs trustscore.py: ok")

    df = load_inputs(d, t)
    print(f"  {len(df):,} rows; {pred} == (prob > t) and correct == (pred == true) on all")

    folds, test_all = load_fold_membership(d, src, args.cf_count, args.n_folds, df)
    ens_path = check_against_ensemble(test_all, d, src.name, args.cf_count)
    print(f"  folds: patient-disjoint in all {args.n_folds}; test rows == ensemble oof rows, "
          f"same folds ({len(test_all):,})")

    row = pd.Series(np.arange(len(df)), index=df["path"].values)
    emb = df[[f"emb_{i}" for i in range(N_EMB)]].values.astype(np.float64)
    y_true = df[f"{d}_true"].values.astype(int)
    y_pred = df[pred].values.astype(int)
    correct = df["correct"].values.astype(int)
    groups = df["patient_id"].values

    rows, preds, chosen, inner_all = [], [], [], {}
    for f in folds:
        i = int(f["fold"].iloc[0])
        tr = row[f.loc[f.split == "train", "path"]].values
        te = row[f.loc[f.split == "test", "path"]].values
        assert not set(tr) & set(te)
        print(f"\n{'-'*72}\n  FOLD {i + 1}/{args.n_folds}  (reference {len(tr):,} | test "
              f"{len(te):,})\n{'-'*72}", flush=True)
        t0 = time.time()
        best, res, (n_fit, n_val) = inner_select(emb[tr], y_true[tr], y_pred[tr], correct[tr],
                                                 groups[tr])
        inner_all[i] = res
        print(f"  chosen: pca {best['pca']}  k {best['k']}  alpha {best['alpha']}  "
              f"(inner val AUROC {best['val_auroc']:.4f})", flush=True)

        pca = pca_fit(emb[tr], best["pca"])
        ref = Reference(pca.transform(emb[tr]), y_true[tr])
        score, dist = ref.score(pca.transform(emb[te]), y_pred[te], best["k"], best["alpha"])

        p = df.iloc[te][["path", "patient_id", prob, pred, "correct"]].copy()
        p.insert(2, "fold", i)
        p["d_class0"], p["d_class1"] = dist[:, 0], dist[:, 1]
        yp = p[pred].values
        p["d_pred"] = dist[np.arange(len(yp)), yp]
        p["d_other"] = dist[np.arange(len(yp)), 1 - yp]
        p[SCORE] = score
        # the swap check, row by row: the denominator is the predicted class's distance
        assert np.allclose(score, p["d_other"] / (p["d_pred"] + MIN_DIST), rtol=1e-12)
        assert np.isfinite(score).all()
        p.to_csv(os.path.join(out_dir, f"fold_{i}_predictions.csv"), index=False)
        preds.append(p)

        yc = p["correct"].values
        msp = msp_score(p[prob].values, t, "threshold")
        top, bot = decile_agreement(score, msp)
        rows.append({"fold": i, "n_test": len(te), "n_ref": len(tr), "n_fit": n_fit,
                     "n_val": n_val, "pca": best["pca"], "k": best["k"],
                     "alpha": best["alpha"], "val_auroc": best["val_auroc"],
                     "auroc": float(roc_auc_score(yc, score)),
                     "auroc_negated": float(roc_auc_score(yc, -score)),
                     "aupr_error": aupr_error(yc, score),
                     "e_aurc": e_aurc(yc, score),
                     "msp_auroc": float(roc_auc_score(yc, msp)),
                     "msp_aupr_error": aupr_error(yc, msp),
                     "msp_e_aurc": e_aurc(yc, msp),
                     "top_decile_agree": top, "bottom_decile_agree": bot,
                     "spearman_vs_msp": float(spearmanr(score, msp)[0])})
        chosen.append(best)
        r = rows[-1]
        print(f"  test AUROC {r['auroc']:.4f}  (adjusted MSP {r['msp_auroc']:.4f})  "
              f"E-AURC {r['e_aurc']:.4f} (MSP {r['msp_e_aurc']:.4f})  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    oof = pd.concat(preds, ignore_index=True)
    assert set(oof["path"]) == set(test_all["path"]) and oof["path"].is_unique
    oof.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)

    summ = pd.DataFrame(rows)
    metrics = ["val_auroc", "auroc", "auroc_negated", "aupr_error", "e_aurc", "msp_auroc",
               "msp_aupr_error", "msp_e_aurc", "top_decile_agree", "bottom_decile_agree",
               "spearman_vs_msp"]
    mean = {"fold": "mean", "n_test": int(summ.n_test.sum()),
            **{c: float(summ[c].mean()) for c in metrics}}
    std = {"fold": "std", **{c: float(summ[c].std(ddof=0)) for c in metrics}}
    summ = pd.concat([summ, pd.DataFrame([mean, std])], ignore_index=True)
    for c in ("n_test", "n_ref", "n_fit", "n_val", "pca", "k"):
        summ[c] = summ[c].astype("Int64")
    summ.insert(0, "model", "TrustScore")
    summ.to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)

    diag = diagnostics(d, src.name, args.cf_count, oof, summ, t)

    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump({
            "disease": d, "model": "TrustScore",
            "reference": "Jiang et al., NeurIPS 2018; github.com/google/TrustScore @ 07a73be, "
                         "trustscore/trustscore.py L24-122",
            "inputs": f"emb_0..emb_{N_EMB - 1} of c2_data.csv (C0 penultimate features)",
            "reference_set": "the fold's training rows, ground-truth labels",
            "predicted_class": "prob > t (== <disease>_pred, asserted)",
            "score": "d(x, other class) / (d(x, predicted class) + min_dist), d = distance to "
                     "the 2nd-nearest point of the (density-filtered) class set",
            "metric": "euclidean", "neighbours": "exact, sklearn brute force (float64)",
            "pca": "centred, not whitened, svd_solver=full, fit on the reference rows only",
            "min_dist": MIN_DIST,
            "grid": {"alpha": list(ALPHAS), "k": list(KS), "pca": list(PCA_DIMS)},
            "selection": "inner-validation AUROC for `correct`; first max in grid order",
            "inner_split": f"GroupShuffleSplit(test_size={INNER_VAL}, patient_id, seed "
                           f"{RANDOM_SEED}) of each fold's training side",
            "chosen_per_fold": {str(r["fold"]): {"pca": int(c["pca"]), "k": int(c["k"]),
                                                 "alpha": c["alpha"],
                                                 "val_auroc": c["val_auroc"]}
                                for r, c in zip(rows, chosen)},
            "inner_results": {str(k): v for k, v in inner_all.items()},
            "threshold_t": t,
            "threshold_file": os.path.join(RESULTS_DIR, "C0_final", "multilabel_ignore", d,
                                           "threshold.txt"),
            "fold_source": {"cf_source": src.name, "cf_count": args.cf_count,
                            "n_folds": args.n_folds, "matched_ensemble_oof": ens_path},
            "n_rows": int(len(oof)),
            "diagnostics": diag,
            "seed": RANDOM_SEED,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "sklearn": sklearn.__version__},
        }, fh, indent=2)
    print(f"\n  wrote {out_dir}")


def diagnostics(disease, source_name, cf_count, oof, summ, t):
    """Task 5: direction, predicted/other assignment, decile overlap with adjusted MSP, and
    the mean CV AUROC next to adjusted MSP, ConfidNet and MLP_I. Logged and returned."""
    m = summ[summ.fold == "mean"].iloc[0]
    pred = oof[f"{disease}_pred"].values
    yc = oof["correct"].values
    s = oof[SCORE].values
    out = {}
    print(f"\n{'='*72}\n  DIAGNOSTICS\n{'='*72}")

    # 1. direction
    ok = bool(m["auroc"] > m["auroc_negated"])
    out["direction"] = {"auroc": float(m["auroc"]), "auroc_negated": float(m["auroc_negated"]),
                        "ok": ok}
    print(f"  1. direction   mean-fold AUROC score {m['auroc']:.4f}  vs  -score "
          f"{m['auroc_negated']:.4f}   {'ok' if ok else '!! FAIL: the negated score ranks better'}")

    # 2. predicted vs other class, within each predicted class. The swapped score (predicted
    #    and other exchanged) is 1/score up to min_dist, so it ranks exactly opposite; what
    #    can go wrong is a branch for one predicted class. Within each, the score must rank
    #    correct above wrong, and the nearest class set should mostly be the predicted one
    #    among correct rows.
    out["by_predicted_class"] = {}
    print("  2. by predicted class   (AUROC within the group; share whose nearest class "
          "set is the predicted one, correct / wrong rows)")
    for c in (0, 1):
        g = pred == c
        near = (oof["d_pred"].values < oof["d_other"].values)[g]
        a = float(roc_auc_score(yc[g], s[g])) if 0 < yc[g].sum() < g.sum() else float("nan")
        a_swap = float(roc_auc_score(yc[g], oof["d_pred"].values[g]
                                     / (oof["d_other"].values[g] + MIN_DIST))) \
            if not np.isnan(a) else float("nan")
        rec = {"n": int(g.sum()), "acc": float(yc[g].mean()), "auroc": a,
               "auroc_swapped": a_swap,
               "nearest_is_pred_correct": float(near[yc[g] == 1].mean()),
               "nearest_is_pred_wrong": float(near[yc[g] == 0].mean())
               if (yc[g] == 0).any() else float("nan"),
               "ok": bool(a > a_swap)}
        out["by_predicted_class"][str(c)] = rec
        print(f"     pred={c}: n {rec['n']:>6,}  acc {rec['acc']:.3f}  AUROC {a:.4f}  "
              f"(swapped {a_swap:.4f})  nearest=pred: correct {rec['nearest_is_pred_correct']:.3f}"
              f" / wrong {rec['nearest_is_pred_wrong']:.3f}"
              f"   {'ok' if rec['ok'] else '!! FAIL: swapped assignment ranks better'}")

    # 3. decile agreement with adjusted MSP
    out["decile_agreement"] = {"top": float(m["top_decile_agree"]),
                               "bottom": float(m["bottom_decile_agree"]),
                               "spearman": float(m["spearman_vs_msp"]), "chance": 0.1}
    print(f"  3. vs adjusted MSP (mean over folds): top-decile overlap "
          f"{m['top_decile_agree']:.3f}, bottom-decile overlap {m['bottom_decile_agree']:.3f} "
          f"(chance 0.10); Spearman {m['spearman_vs_msp']:.3f}")

    # 4. next to the other baselines
    table = {"TrustScore": float(m["auroc"]), "adjusted MSP": float(m["msp_auroc"])}
    p = os.path.join(RESULTS_DIR, "C2_final_results", disease, "baselines", "confidnet",
                     "cv_summary.csv")
    if os.path.exists(p):
        cn = pd.read_csv(p)
        table["ConfidNet"] = float(cn.loc[cn.fold == "mean", "auroc"].iloc[0])
    p = os.path.join(RESULTS_DIR, "C2_final_results", disease, source_name, f"cf_{cf_count}",
                     "attributes", "cv_summary.csv")
    if os.path.exists(p):
        a = pd.read_csv(p)
        hit = a[(a.model == "MLP") & (a.config == "I")]
        if len(hit):
            table["MLP_I"] = float(hit.auc_mean.iloc[0])
    out["mean_cv_auroc"] = table
    print("  4. mean CV AUROC")
    for k, v in table.items():
        print(f"     {k:<14} {v:.4f}" + ("" if k == "TrustScore" else
                                          f"   (TrustScore {table['TrustScore'] - v:+.4f})"))
    for k in ("ConfidNet", "MLP_I"):
        if k not in table:
            print(f"     {k:<14} not available yet")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if cv_summary.csv already exists")
    ap.add_argument("--self-test", action="store_true",
                    help="check against the reference implementation and exit")
    add_cf_args(ap)
    ap.set_defaults(cf_count=10)
    args = ap.parse_args()
    if args.self_test:
        test_against_reference()
        print("test_against_reference: ok")
        return
    if args.n_folds != N_FOLDS:
        print(f"  note: --n-folds {args.n_folds} (default {N_FOLDS})")
    run(args)


if __name__ == "__main__":
    main()
