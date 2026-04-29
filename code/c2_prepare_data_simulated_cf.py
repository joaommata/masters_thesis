"""
Fake Counterfactual C2 - Proof of Concept
=========================================
For each sample i, find the nearest training sample that C0 predicted
with the OPPOSITE label, restricted to disease-relevant attribute space.

The signal is a DIFFERENCE VECTOR (query - nearest_opposite_neighbour),
computed in standardised attribute space, rather than a single scalar distance.

Hypothesis:
  Correct   -> sample sits firmly in C0's territory -> large diff in relevant attrs
  Incorrect -> sample is near C0's decision boundary -> small diff in relevant attrs

Counterfactual matching strategy (4-pool):
  correct sample   -> nearest correct sample with opposite prediction
  incorrect sample -> nearest incorrect sample with opposite prediction

This module now exposes compute_cf_for_split() so the same CF computation can
be reused per fold in cross-validation without file I/O.
"""


import os
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ==========================================================================
# CONFIG - edit this block only
# ==========================================================================

DISEASE = "Effusion"    # "Effusion" | "Cardiomegaly" | "Pneumothorax" | "Atelectasis"
DISTANCE = "l1"         # l2 (Euclidean) or l1 (Manhattan)

BASE_DIR = "/zhome/d0/a/221493/thesis"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, f"C2_sim_cf/{DISEASE.lower()}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of nearest neighbours to retrieve
cf_count = 1


def _get_disease_cols(disease):
    disease_lower = disease.lower()
    return (
        f"{disease_lower}_prob",
        f"{disease_lower}_pred",
        f"{disease_lower}_true",
    )


disease_prob_col, disease_pred_col, disease_true_col = _get_disease_cols(DISEASE)


def add_clinical_ratios(df):
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


def compute_diff_vectors(
    df,
    query_scaled,
    disease_pred_col,
    nn_pred1_corr,
    nn_pred1_incorr,
    nn_pred0_corr,
    nn_pred0_incorr,
    train_scaled_pred1_corr,
    train_scaled_pred1_incorr,
    train_scaled_pred0_corr,
    train_scaled_pred0_incorr,
    train_prob_pred1_corr,
    train_prob_pred1_incorr,
    train_prob_pred0_corr,
    train_prob_pred0_incorr,
    train_paths,
    pool_global_idx,
):
    query_preds = df[disease_pred_col].values.astype(int)
    query_correct = df["correct"].values.astype(int)
    n, d = query_scaled.shape
    diff_vecs = np.empty((n, d), dtype=np.float64)
    cf_probs = np.empty(n, dtype=np.float64)
    cf_paths = np.empty(n, dtype=object)

    def get_cf_paths(idxs, pool_key):
        global_idxs = pool_global_idx[pool_key][idxs]
        return np.array(["|".join(train_paths[row]) for row in global_idxs])

    # pred=1, correct=1 -> search in pred=0, correct=1
    mask = (query_preds == 1) & (query_correct == 1)
    if mask.any():
        _, idxs = nn_pred0_corr.kneighbors(query_scaled[mask])
        diff_vecs[mask] = (
            query_scaled[mask][:, np.newaxis, :] - train_scaled_pred0_corr[idxs]
        ).mean(axis=1)
        cf_probs[mask] = train_prob_pred0_corr[idxs].mean(axis=1)
        cf_paths[mask] = get_cf_paths(idxs, "pred0_corr")

    # pred=1, correct=0 -> search in pred=0, correct=0
    mask = (query_preds == 1) & (query_correct == 0)
    if mask.any():
        _, idxs = nn_pred0_incorr.kneighbors(query_scaled[mask])
        diff_vecs[mask] = (
            query_scaled[mask][:, np.newaxis, :] - train_scaled_pred0_incorr[idxs]
        ).mean(axis=1)
        cf_probs[mask] = train_prob_pred0_incorr[idxs].mean(axis=1)
        cf_paths[mask] = get_cf_paths(idxs, "pred0_incorr")

    # pred=0, correct=1 -> search in pred=1, correct=1
    mask = (query_preds == 0) & (query_correct == 1)
    if mask.any():
        _, idxs = nn_pred1_corr.kneighbors(query_scaled[mask])
        diff_vecs[mask] = (
            query_scaled[mask][:, np.newaxis, :] - train_scaled_pred1_corr[idxs]
        ).mean(axis=1)
        cf_probs[mask] = train_prob_pred1_corr[idxs].mean(axis=1)
        cf_paths[mask] = get_cf_paths(idxs, "pred1_corr")

    # pred=0, correct=0 -> search in pred=1, correct=0
    mask = (query_preds == 0) & (query_correct == 0)
    if mask.any():
        _, idxs = nn_pred1_incorr.kneighbors(query_scaled[mask])
        diff_vecs[mask] = (
            query_scaled[mask][:, np.newaxis, :] - train_scaled_pred1_incorr[idxs]
        ).mean(axis=1)
        cf_probs[mask] = train_prob_pred1_incorr[idxs].mean(axis=1)
        cf_paths[mask] = get_cf_paths(idxs, "pred1_incorr")

    return diff_vecs, cf_probs, cf_paths


def compute_cf_for_split(train_df, test_df, cf_count, disease, distance="l1"):
    """
    Compute counterfactuals for a single train/test split.

    This is the core CF logic extracted from main() so it can be called
    during cross-validation without file I/O.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (already merged C0 + C1)
    test_df : pd.DataFrame
        Test data (already merged C0 + C1)
    cf_count : int
        Number of nearest neighbors to retrieve
    disease : str
        Disease name (e.g., 'effusion')
    distance : str
        'l1' or 'l2'

    Returns
    -------
    train_with_cf : pd.DataFrame
        Training data enriched with delta columns and cf_prob
    test_with_cf : pd.DataFrame
        Test data enriched with delta columns and cf_prob
    attr_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler (needed if you want to inverse-transform later)
    """

    disease_prob_col, disease_pred_col, disease_true_col = _get_disease_cols(disease)

    # Add clinical ratios
    train_df = add_clinical_ratios(train_df.copy())
    test_df = add_clinical_ratios(test_df.copy())

    # Identify columns
    meta_cols = {
        disease_prob_col,
        disease_pred_col,
        disease_true_col,
        "correct",
        "path",
        "patient_id",
    }
    emb_cols = [c for c in train_df.columns if c.startswith("emb_")]
    relevant_cols = [c for c in train_df.columns if c not in meta_cols and c not in emb_cols]

    # Fill NaNs
    train_clean = train_df.copy()
    test_clean = test_df.copy()
    train_clean[relevant_cols] = train_clean[relevant_cols].fillna(0)
    test_clean[relevant_cols] = test_clean[relevant_cols].fillna(0)

    # Scale attributes
    attr_scaler = StandardScaler()
    train_scaled = attr_scaler.fit_transform(train_clean[relevant_cols].values.astype(float))
    test_scaled = attr_scaler.transform(test_clean[relevant_cols].values.astype(float))

    # Build 4 NN pools
    metric = "manhattan" if distance == "l1" else "euclidean"

    train_preds = train_clean[disease_pred_col].values.astype(int)
    train_correct = train_clean["correct"].values.astype(int)

    idx_pred1_corr = (train_preds == 1) & (train_correct == 1)
    idx_pred1_incorr = (train_preds == 1) & (train_correct == 0)
    idx_pred0_corr = (train_preds == 0) & (train_correct == 1)
    idx_pred0_incorr = (train_preds == 0) & (train_correct == 0)

    pool_global_idx = {
        "pred1_corr": np.where(idx_pred1_corr)[0],
        "pred1_incorr": np.where(idx_pred1_incorr)[0],
        "pred0_corr": np.where(idx_pred0_corr)[0],
        "pred0_incorr": np.where(idx_pred0_incorr)[0],
    }

    train_scaled_pred1_corr = train_scaled[idx_pred1_corr]
    train_scaled_pred1_incorr = train_scaled[idx_pred1_incorr]
    train_scaled_pred0_corr = train_scaled[idx_pred0_corr]
    train_scaled_pred0_incorr = train_scaled[idx_pred0_incorr]

    train_prob_pred1_corr = train_clean[disease_prob_col].values[idx_pred1_corr]
    train_prob_pred1_incorr = train_clean[disease_prob_col].values[idx_pred1_incorr]
    train_prob_pred0_corr = train_clean[disease_prob_col].values[idx_pred0_corr]
    train_prob_pred0_incorr = train_clean[disease_prob_col].values[idx_pred0_incorr]

    nn_pred1_corr = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(
        train_scaled_pred1_corr
    )
    nn_pred1_incorr = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(
        train_scaled_pred1_incorr
    )
    nn_pred0_corr = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(
        train_scaled_pred0_corr
    )
    nn_pred0_incorr = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(
        train_scaled_pred0_incorr
    )

    train_diff, train_cf_probs, train_cf_paths = compute_diff_vectors(
        train_clean,
        train_scaled,
        disease_pred_col,
        nn_pred1_corr,
        nn_pred1_incorr,
        nn_pred0_corr,
        nn_pred0_incorr,
        train_scaled_pred1_corr,
        train_scaled_pred1_incorr,
        train_scaled_pred0_corr,
        train_scaled_pred0_incorr,
        train_prob_pred1_corr,
        train_prob_pred1_incorr,
        train_prob_pred0_corr,
        train_prob_pred0_incorr,
        train_clean["path"].values,
        pool_global_idx,
    )

    test_diff, test_cf_probs, test_cf_paths = compute_diff_vectors(
        test_clean,
        test_scaled,
        disease_pred_col,
        nn_pred1_corr,
        nn_pred1_incorr,
        nn_pred0_corr,
        nn_pred0_incorr,
        train_scaled_pred1_corr,
        train_scaled_pred1_incorr,
        train_scaled_pred0_corr,
        train_scaled_pred0_incorr,
        train_prob_pred1_corr,
        train_prob_pred1_incorr,
        train_prob_pred0_corr,
        train_prob_pred0_incorr,
        train_clean["path"].values,
        pool_global_idx,
    )

        # Build output DataFrames
    diff_col_names = [f"delta_{c}" for c in relevant_cols]

    # Start from clean copies
    train_out = train_clean.copy()
    test_out  = test_clean.copy()

    # --- Add all delta columns at once to avoid fragmentation ---
    train_diff_df = pd.DataFrame(train_diff, columns=diff_col_names, index=train_out.index)
    test_diff_df  = pd.DataFrame(test_diff, columns=diff_col_names, index=test_out.index)

    train_out = pd.concat([train_out, train_diff_df], axis=1)
    test_out  = pd.concat([test_out, test_diff_df], axis=1)

    # Add counterfactual info
    train_out["cf_prob"]  = train_cf_probs
    test_out["cf_prob"]   = test_cf_probs
    train_out["cf_paths"] = train_cf_paths
    test_out["cf_paths"]   = test_cf_paths

    return train_out, test_out, attr_scaler


# VERSION FOR UNMATCHED TEST:
def compute_cf_for_split_unmatched(train_df, test_df, cf_count, disease, distance="l1"):
    """
    Same interface as compute_cf_for_split() but uses unmatched pools.
    CF candidates are selected by opposite prediction only — correctness ignored.
    Used as an ablation to show the matched strategy is necessary.
    """
    disease_prob_col, disease_pred_col, disease_true_col = _get_disease_cols(disease)

    train_df = add_clinical_ratios(train_df.copy())
    test_df  = add_clinical_ratios(test_df.copy())

    meta_cols = {disease_prob_col, disease_pred_col, disease_true_col,
                 "correct", "path", "patient_id"}
    emb_cols  = [c for c in train_df.columns if c.startswith("emb_")]
    relevant_cols = [c for c in train_df.columns
                     if c not in meta_cols and c not in emb_cols]

    train_clean = train_df.copy()
    test_clean  = test_df.copy()
    train_clean[relevant_cols] = train_clean[relevant_cols].fillna(0)
    test_clean[relevant_cols]  = test_clean[relevant_cols].fillna(0)

    # Scaler fit on train only — same as matched version
    attr_scaler  = StandardScaler()
    train_scaled = attr_scaler.fit_transform(train_clean[relevant_cols].values.astype(float))
    test_scaled  = attr_scaler.transform(test_clean[relevant_cols].values.astype(float))

    metric = "manhattan" if distance == "l1" else "euclidean"
    train_preds = train_clean[disease_pred_col].values.astype(int)

    # 2 pools instead of 4 — correctness not enforced
    idx_pred1 = (train_preds == 1)
    idx_pred0 = (train_preds == 0)

    pool_global_idx = {
        'pred1': np.where(idx_pred1)[0],
        'pred0': np.where(idx_pred0)[0],
    }

    train_scaled_pred1 = train_scaled[idx_pred1]
    train_scaled_pred0 = train_scaled[idx_pred0]
    train_prob_pred1   = train_clean[disease_prob_col].values[idx_pred1]
    train_prob_pred0   = train_clean[disease_prob_col].values[idx_pred0]

    nn_pred1 = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred1)
    nn_pred0 = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred0)

    def _compute_diffs(df, query_scaled):
        """Compute diff vectors for a query set against the 2 unmatched pools."""
        query_preds = df[disease_pred_col].values.astype(int)
        n, d = query_scaled.shape
        diff_vecs = np.empty((n, d), dtype=np.float64)
        cf_probs  = np.empty(n, dtype=np.float64)
        cf_paths  = np.empty(n, dtype=object)

        mask = (query_preds == 1)
        if mask.any():
            _, idxs = nn_pred0.kneighbors(query_scaled[mask])
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred0[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_pred0[idxs].mean(axis=1)
            global_idxs     = pool_global_idx['pred0'][idxs]
            cf_paths[mask]  = np.array(["|".join(train_clean["path"].values[row]) for row in global_idxs])

        mask = (query_preds == 0)
        if mask.any():
            _, idxs = nn_pred1.kneighbors(query_scaled[mask])
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred1[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_pred1[idxs].mean(axis=1)
            global_idxs     = pool_global_idx['pred1'][idxs]
            cf_paths[mask]  = np.array(["|".join(train_clean["path"].values[row]) for row in global_idxs])

        return diff_vecs, cf_probs, cf_paths

    train_diff, train_cf_probs, train_cf_paths = _compute_diffs(train_clean, train_scaled)
    test_diff,  test_cf_probs,  test_cf_paths  = _compute_diffs(test_clean,  test_scaled)

    diff_col_names = [f"delta_{c}" for c in relevant_cols]

    train_out = pd.concat([train_clean,
                           pd.DataFrame(train_diff, columns=diff_col_names, index=train_clean.index)], axis=1)
    test_out  = pd.concat([test_clean,
                           pd.DataFrame(test_diff,  columns=diff_col_names, index=test_clean.index)],  axis=1)

    train_out["cf_prob"]  = train_cf_probs
    test_out["cf_prob"]   = test_cf_probs
    train_out["cf_paths"] = train_cf_paths
    test_out["cf_paths"]  = test_cf_paths

    return train_out, test_out, attr_scaler


def main():
    c0_train = pd.read_csv(
        os.path.join(RESULTS_DIR, f"C0_baseline/{DISEASE.lower()}/train_c0_{DISEASE.lower()}.csv")
    )
    c0_valid = pd.read_csv(
        os.path.join(RESULTS_DIR, f"C0_baseline/{DISEASE.lower()}/valid_c0_{DISEASE.lower()}.csv")
    )
    c1_train = pd.read_csv(os.path.join(RESULTS_DIR, "C1_attributes/train_c1_attribute_vector_rad.csv"))
    c1_valid = pd.read_csv(os.path.join(RESULTS_DIR, "C1_attributes/valid_c1_attribute_vector_rad.csv"))

    print(f"C0 - train: {len(c0_train):,}  valid: {len(c0_valid):,}")
    print(f"C1 - train: {len(c1_train):,}  valid: {len(c1_valid):,}")

    train_df = c0_train.merge(c1_train, on="path", how="inner")
    valid_df = c0_valid.merge(c1_valid, on="path", how="inner")

    for df in (train_df, valid_df):
        df.rename(
            columns={
                col: f"{DISEASE.lower()}_{col}"
                for col in ("prob", "true", "pred")
                if col in df.columns
            },
            inplace=True,
        )

    print(f"Merged - train: {len(train_df):,}  valid: {len(valid_df):,}")
    print(f"Train correctness: {train_df['correct'].value_counts().to_dict()}")
    print(f"Valid correctness: {valid_df['correct'].value_counts().to_dict()}")

    train_out, valid_out, attr_scaler = compute_cf_for_split(
        train_df=train_df,
        test_df=valid_df,
        cf_count=cf_count,
        disease=DISEASE,
        distance=DISTANCE,
    )

    train_dist = np.linalg.norm(
        train_out[[c for c in train_out.columns if c.startswith("delta_")]].values,
        axis=1,
    )
    correct_mask = train_out["correct"].values == 1

    print("\nMean dA magnitude (L2 norm):")
    print(f"  Correct   : {train_dist[correct_mask].mean():.4f}")
    print(f"  Incorrect : {train_dist[~correct_mask].mean():.4f}")
    print(
        f"  d         : {train_dist[correct_mask].mean() - train_dist[~correct_mask].mean():+.4f}"
    )
    print("\n  If Correct > Incorrect -> the signal is working as hypothesised.")

    train_out.to_csv(os.path.join(OUTPUT_DIR, f"train_with_diff_vectors_{cf_count}.csv"), index=False)
    valid_out.to_csv(os.path.join(OUTPUT_DIR, f"valid_with_diff_vectors_{cf_count}.csv"), index=False)
    joblib.dump(attr_scaler, os.path.join(OUTPUT_DIR, f"attr_scaler_{cf_count}.pkl"))

    print(f"\nAll outputs saved -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
