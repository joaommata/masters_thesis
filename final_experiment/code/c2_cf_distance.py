"""CF distance as a confidence score.

For each query the pipeline already retrieved a counterfactual; here we recover
how FAR that counterfactual was. The distance is computed in exactly the space
the retrieval used -- the fold's standardised attribute space, scaler fit on the
fold's TRAIN rows -- so it is the same number the neighbour search minimised.

Read it as a confidence score: a query sitting close to an opposite-prediction
neighbour is near the decision boundary, so a SMALL distance should mean a
likely error. AUC is therefore computed with (-distance) as the error score.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_folds as F

DISEASES = ["effusion", "cardiomegaly", "consolidation", "edema", "atelectasis"]
SOURCE = "knn_correct_cf"


def _pair_dist(ctx, pair, distance="l1"):
    """Distance from each query to its NEAREST counterfactual (cf_paths[0])."""
    q = ctx.attr_scaled_all[[ctx.row_of[p] for p in pair["path"]]]
    cf = ctx.attr_scaled_all[[ctx.row_of[str(c).split("|")[0]] for c in pair["cf_paths"]]]
    if distance == "l1":
        return np.abs(q - cf).sum(axis=1)
    if distance == "l2":
        return np.linalg.norm(q - cf, axis=1)
    if distance == "cosine":
        q_norm = np.linalg.norm(q, axis=1)
        cf_norm = np.linalg.norm(cf, axis=1)
        return 1 - (q * cf).sum(axis=1) / (q_norm * cf_norm)
    raise ValueError(distance)


def cf_distances(disease, cf_count=1, n_folds=F.N_FOLDS, distance="l1"):
    """Test-side CF distance for every query, pooled over the CV folds.
    Only the test side is used: on the train side the query is itself in the CF
    pool's fold, and the model that defines `correct` was fit on those rows.
    """
    base = F.load_c2_data(disease)[["path", "correct", f"{disease}_prob"]]
    out = []
    for i in range(n_folds):
        f = F.read_pairing(disease, SOURCE, cf_count, i)
        ctx = F._AttrOnlyContext(disease, f[f["split"] == "train"])
        te = f[f["split"] == "test"]
        out.append(pd.DataFrame({
            "fold": i,
            "path": te["path"].values,
            "cf_dist": _pair_dist(ctx, te, distance),
        }))
    df = pd.concat(out, ignore_index=True).merge(base, on="path", how="left")
    df["disease"] = disease
    df["error"] = 1 - df["correct"]
    return df

def cf_distances_all_metrics(disease, cf_count=1, n_folds=F.N_FOLDS, distances=("l1", "l2", "cosine")):
    base = F.load_c2_data(disease)[["path", "correct", f"{disease}_prob"]]
    out = {m: [] for m in distances}
    for i in range(n_folds):
        f = F.read_pairing(disease, SOURCE, cf_count, i)
        ctx = F._AttrOnlyContext(disease, f[f["split"] == "train"])
        te = f[f["split"] == "test"]
        for m in distances:
            out[m].append(pd.DataFrame({
                "fold": i, "path": te["path"].values,
                "cf_dist": _pair_dist(ctx, te, m),
            }))
    result = {}
    for m in distances:
        df = pd.concat(out[m], ignore_index=True).merge(base, on="path", how="left")
        df["disease"], df["error"] = disease, 1 - df["correct"]
        result[m] = df
    return result


def error_auc(df):
    """AUC for predicting ERROR. Small distance = near boundary = likely error."""
    return roc_auc_score(df["error"], -df["cf_dist"])


def msp_auc(df, disease):
    """Baseline: maximum softmax probability, the standard confidence score."""
    p = df[f"{disease}_prob"].values
    return roc_auc_score(df["error"], -np.maximum(p, 1 - p))


def summarise(disease, cf_count=1, distance="l1"):
    df = cf_distances(disease, cf_count=cf_count, distance=distance)
    return df, {
        "disease": disease,
        "n": len(df),
        "error_rate": df["error"].mean(),
        "dist_mean": df["cf_dist"].mean(),
        "dist_median": df["cf_dist"].median(),
        "dist_std": df["cf_dist"].std(),
        "auc_cf_dist": error_auc(df),
        "auc_msp": msp_auc(df, disease),
    }
