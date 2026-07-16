"""
Simulated Counterfactual C2
===========================
For each sample, find the nearest training sample that C0 predicted with the
OPPOSITE label in standardised attribute space. The signal is a DIFFERENCE
VECTOR (query - nearest_opposite_neighbour).

Hypothesis:
  Correct   -> sample sits firmly in C0's territory -> large diff in relevant attrs
  Incorrect -> sample is near C0's decision boundary -> small diff in relevant attrs

Available CF strategies
-----------------------
compute_cf_for_split_matched_train_unmatched_test:
  Train: 4-pool matched (pred x correct) — nearest opposite with same correctness.
  Test:  2-pool unmatched (pred only) — avoids label leakage at inference time.

compute_cf_for_split_unmatched:
  Both train and test use 2-pool (pred only). Ablation baseline.

compute_cf_for_split_further:
  Same as matched_train_unmatched_test but applies a k_offset to pred=0 queries,
  pushing their CFs deeper into the pred=1 pool.

compute_cf_for_split_correct_cf:
  CFs always come from correctly-classified training examples (TP or TN pool).
  Routing uses prediction only -> not leaky at test time.
    pred=1 (TP or FP) -> nearest TN  (train pred=0, correct=1)
    pred=0 (TN or FN) -> nearest TP  (train pred=1, correct=1)

compute_cf_for_split_gt_routing:
  Both train and test route by the query's prediction, but the CF pool is
  partitioned by ground-truth label of the training neighbours (not their pred).
    pred=1 -> nearest training sample with true=0
    pred=0 -> nearest training sample with true=1
  No leakage: CFs always come from training, where true labels are always known.

"""


import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ==========================================================================
# CONFIG - edit this block only
# ==========================================================================

DISEASE = "Effusion"    # "Effusion" | "Cardiomegaly" | "Pneumothorax" | "Atelectasis"
DISTANCE = "l1"         # l2 (Euclidean) or l1 (Manhattan)

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
RESULTS_DIR = os.path.join(RESULTS_DIR, "")
OUTPUT_DIR = os.path.join(RESULTS_DIR, f"C2_sim_cf/{DISEASE.lower()}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of nearest neighbours to retrieve
cf_count = 1


def _resolve_metric(distance):
    """Map a --distance flag to a sklearn NearestNeighbors metric name."""
    if distance == "l1":
        return "manhattan"
    elif distance == "cosine":
        return "cosine"
    else:
        return "euclidean"


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



# VERSION FOR UNMATCHED TEST - TRAINING MATCHED, TEST UNMATCHED:
def compute_cf_for_split_matched_train_unmatched_test(train_df, test_df, cf_count, disease, distance="l1"):
    """
    Hybrid CF computation:
    - Training fold: 4-pool matched (pred x correct) — legitimate, labels available
    - Test fold: 2-pool unmatched (pred only) — deployment-realistic, no label leakage
    """
    disease_prob_col, disease_pred_col, disease_true_col = _get_disease_cols(disease)

    train_df = add_clinical_ratios(train_df.copy())
    test_df  = add_clinical_ratios(test_df.copy())

    meta_cols = {disease_prob_col, disease_pred_col, disease_true_col,
                 "correct", "path", "cam_path", "patient_id"}
    emb_cols  = [c for c in train_df.columns if c.startswith("emb_")]
    relevant_cols = [c for c in train_df.columns
                     if c not in meta_cols and c not in emb_cols]

    train_clean = train_df.copy()
    test_clean  = test_df.copy()
    train_clean[relevant_cols] = train_clean[relevant_cols].fillna(0)
    test_clean[relevant_cols]  = test_clean[relevant_cols].fillna(0)

    # Scaler fit on train only
    attr_scaler  = StandardScaler()
    train_scaled = attr_scaler.fit_transform(train_clean[relevant_cols].values.astype(float))
    test_scaled  = attr_scaler.transform(test_clean[relevant_cols].values.astype(float))

    metric = _resolve_metric(distance)
    train_preds   = train_clean[disease_pred_col].values.astype(int)
    train_correct = train_clean["correct"].values.astype(int)

    # ── 4 matched pools for training ─────────────────────────────────────
    idx_pred1_corr   = (train_preds == 1) & (train_correct == 1)
    idx_pred1_incorr = (train_preds == 1) & (train_correct == 0)
    idx_pred0_corr   = (train_preds == 0) & (train_correct == 1)
    idx_pred0_incorr = (train_preds == 0) & (train_correct == 0)

    pool_global_idx = {
        "pred1_corr":   np.where(idx_pred1_corr)[0],
        "pred1_incorr": np.where(idx_pred1_incorr)[0],
        "pred0_corr":   np.where(idx_pred0_corr)[0],
        "pred0_incorr": np.where(idx_pred0_incorr)[0],
    }

    train_scaled_pred1_corr   = train_scaled[idx_pred1_corr]
    train_scaled_pred1_incorr = train_scaled[idx_pred1_incorr]
    train_scaled_pred0_corr   = train_scaled[idx_pred0_corr]
    train_scaled_pred0_incorr = train_scaled[idx_pred0_incorr]

    train_prob_pred1_corr   = train_clean[disease_prob_col].values[idx_pred1_corr]
    train_prob_pred1_incorr = train_clean[disease_prob_col].values[idx_pred1_incorr]
    train_prob_pred0_corr   = train_clean[disease_prob_col].values[idx_pred0_corr]
    train_prob_pred0_incorr = train_clean[disease_prob_col].values[idx_pred0_incorr]

    nn_pred1_corr   = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred1_corr)
    nn_pred1_incorr = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred1_incorr)
    nn_pred0_corr   = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred0_corr)
    nn_pred0_incorr = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred0_incorr)

    # ── Train diff vectors: matched routing (correct available) ───────────
    train_diff, train_cf_probs, train_cf_paths = compute_diff_vectors(
        train_clean, train_scaled, disease_pred_col,
        nn_pred1_corr, nn_pred1_incorr, nn_pred0_corr, nn_pred0_incorr,
        train_scaled_pred1_corr, train_scaled_pred1_incorr,
        train_scaled_pred0_corr, train_scaled_pred0_incorr,
        train_prob_pred1_corr, train_prob_pred1_incorr,
        train_prob_pred0_corr, train_prob_pred0_incorr,
        train_clean["path"].values, pool_global_idx,
    )

    # ── 2 unmatched pools for test routing (pred only, no correct) ────────
    idx_pred1 = (train_preds == 1)
    idx_pred0 = (train_preds == 0)

    train_scaled_pred1 = train_scaled[idx_pred1]
    train_scaled_pred0 = train_scaled[idx_pred0]
    train_prob_pred1   = train_clean[disease_prob_col].values[idx_pred1]
    train_prob_pred0   = train_clean[disease_prob_col].values[idx_pred0]
    pool_global_idx_2  = {
        'pred1': np.where(idx_pred1)[0],
        'pred0': np.where(idx_pred0)[0],
    }

    nn_pred1 = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred1)
    nn_pred0 = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred0)

    # ── Test diff vectors: unmatched routing (pred only) ──────────────────
    def _compute_diffs_unmatched(df, query_scaled):
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
            global_idxs     = pool_global_idx_2['pred0'][idxs]
            cf_paths[mask]  = np.array(["|".join(train_clean["path"].values[row]) for row in global_idxs])

        mask = (query_preds == 0)
        if mask.any():
            _, idxs = nn_pred1.kneighbors(query_scaled[mask])
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred1[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_pred1[idxs].mean(axis=1)
            global_idxs     = pool_global_idx_2['pred1'][idxs]
            cf_paths[mask]  = np.array(["|".join(train_clean["path"].values[row]) for row in global_idxs])

        return diff_vecs, cf_probs, cf_paths

    test_diff, test_cf_probs, test_cf_paths = _compute_diffs_unmatched(test_clean, test_scaled)

    # ── Assemble output DataFrames ────────────────────────────────────────
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

# VERSION FOR UNMATCHED TEST - BOTH TRAINING AND TEST UNMATCHED:
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
                 "correct", "path", "cam_path", "patient_id"}
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

    metric = _resolve_metric(distance)
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


def attach_cf_features(df_out, train_clean, relevant_cols, emb_cols, disease):
    """Dereference cf_paths -> mean attrs/emb of retrieved CF neighbours.

    train_clean must be the TRAIN fold pool; CF paths always point into train.
    Returns a copy of df_out with cf_attr_*, cf_emb_*, and delta_prob columns added.
    """
    assert train_clean["path"].is_unique, "train_clean has duplicate paths — cannot build lookup"

    lut = train_clean.set_index("path")

    split_paths = [cell.split("|") for cell in df_out["cf_paths"]]
    flat_paths = [p for paths in split_paths for p in paths]
    row_ids = np.repeat(np.arange(len(df_out)), [len(p) for p in split_paths])

    missing = set(flat_paths) - set(lut.index)
    if missing:
        raise ValueError(f"CF path not found in train_clean lookup: {next(iter(missing))!r}")

    all_cols = relevant_cols + emb_cols
    looked_up = lut.loc[flat_paths, all_cols].values.astype(float)

    accum = np.zeros((len(df_out), len(all_cols)))
    counts = np.zeros(len(df_out))
    np.add.at(accum, row_ids, looked_up)
    np.add.at(counts, row_ids, 1)
    means = accum / counts[:, None]

    new_cols = {}
    for j, c in enumerate(relevant_cols):
        new_cols[f"cf_attr_{c}"] = means[:, j]
    for j, c in enumerate(emb_cols):
        new_cols[f"cf_emb_{c}"] = means[:, len(relevant_cols) + j]
    new_cols["delta_prob"] = df_out[f"{disease}_prob"].values - df_out["cf_prob"].values

    return pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)


def compute_cf_for_split_further(train_df, test_df, cf_count, disease, distance="l1", k_offset=0):
    """
    Same as compute_cf_for_split but applies a neighbor offset to disease-absent
    (pred=0) queries, pushing their CFs deeper into the pred=1 pool.
    k_offset=0 reproduces the original behavior exactly.
    """
    disease_prob_col, disease_pred_col, disease_true_col = _get_disease_cols(disease)

    train_df = add_clinical_ratios(train_df.copy())
    test_df = add_clinical_ratios(test_df.copy())

    meta_cols = {disease_prob_col, disease_pred_col, disease_true_col, "correct", "path", "cam_path", "patient_id"}
    emb_cols = [c for c in train_df.columns if c.startswith("emb_")]
    relevant_cols = [c for c in train_df.columns if c not in meta_cols and c not in emb_cols]

    train_clean = train_df.copy()
    test_clean = test_df.copy()
    train_clean[relevant_cols] = train_clean[relevant_cols].fillna(0)
    test_clean[relevant_cols] = test_clean[relevant_cols].fillna(0)

    attr_scaler = StandardScaler()
    train_scaled = attr_scaler.fit_transform(train_clean[relevant_cols].values.astype(float))
    test_scaled = attr_scaler.transform(test_clean[relevant_cols].values.astype(float))

    metric = _resolve_metric(distance)
    train_preds = train_clean[disease_pred_col].values.astype(int)
    train_correct = train_clean["correct"].values.astype(int)

    idx_pred1_corr   = (train_preds == 1) & (train_correct == 1)
    idx_pred1_incorr = (train_preds == 1) & (train_correct == 0)
    idx_pred0_corr   = (train_preds == 0) & (train_correct == 1)
    idx_pred0_incorr = (train_preds == 0) & (train_correct == 0)

    pool_global_idx = {
        "pred1_corr":   np.where(idx_pred1_corr)[0],
        "pred1_incorr": np.where(idx_pred1_incorr)[0],
        "pred0_corr":   np.where(idx_pred0_corr)[0],
        "pred0_incorr": np.where(idx_pred0_incorr)[0],
    }

    train_scaled_pred1_corr   = train_scaled[idx_pred1_corr]
    train_scaled_pred1_incorr = train_scaled[idx_pred1_incorr]
    train_scaled_pred0_corr   = train_scaled[idx_pred0_corr]
    train_scaled_pred0_incorr = train_scaled[idx_pred0_incorr]

    train_prob_pred1_corr   = train_clean[disease_prob_col].values[idx_pred1_corr]
    train_prob_pred1_incorr = train_clean[disease_prob_col].values[idx_pred1_incorr]
    train_prob_pred0_corr   = train_clean[disease_prob_col].values[idx_pred0_corr]
    train_prob_pred0_incorr = train_clean[disease_prob_col].values[idx_pred0_incorr]

    # pred=1 pools need cf_count + k_offset neighbors (offset applied to pred=0 queries)
    nn_pred1_corr   = NearestNeighbors(n_neighbors=cf_count + k_offset, metric=metric, n_jobs=-1).fit(train_scaled_pred1_corr)
    nn_pred1_incorr = NearestNeighbors(n_neighbors=cf_count + k_offset, metric=metric, n_jobs=-1).fit(train_scaled_pred1_incorr)
    # pred=0 pools unchanged
    nn_pred0_corr   = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred0_corr)
    nn_pred0_incorr = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred0_incorr)

    def compute_diffs(df, query_scaled):
        query_preds   = df[disease_pred_col].values.astype(int)
        query_correct = df["correct"].values.astype(int)
        n, d = query_scaled.shape
        diff_vecs = np.empty((n, d), dtype=np.float64)
        cf_probs  = np.empty(n, dtype=np.float64)
        cf_paths  = np.empty(n, dtype=object)

        def get_cf_paths(idxs, pool_key):
            global_idxs = pool_global_idx[pool_key][idxs]
            return np.array(["|".join(train_clean["path"].values[row]) for row in global_idxs])

        # pred=1, correct=1 -> pred=0, correct=1 (no offset)
        mask = (query_preds == 1) & (query_correct == 1)
        if mask.any():
            _, idxs = nn_pred0_corr.kneighbors(query_scaled[mask])
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred0_corr[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_pred0_corr[idxs].mean(axis=1)
            cf_paths[mask]  = get_cf_paths(idxs, "pred0_corr")

        # pred=1, correct=0 -> pred=0, correct=0 (no offset)
        mask = (query_preds == 1) & (query_correct == 0)
        if mask.any():
            _, idxs = nn_pred0_incorr.kneighbors(query_scaled[mask])
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred0_incorr[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_pred0_incorr[idxs].mean(axis=1)
            cf_paths[mask]  = get_cf_paths(idxs, "pred0_incorr")

        # pred=0, correct=1 -> pred=1, correct=1 (OFFSET APPLIED)
        mask = (query_preds == 0) & (query_correct == 1)
        if mask.any():
            _, idxs = nn_pred1_corr.kneighbors(query_scaled[mask])
            idxs = idxs[:, k_offset:]  # skip k_offset nearest, take the rest
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred1_corr[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_pred1_corr[idxs].mean(axis=1)
            cf_paths[mask]  = get_cf_paths(idxs, "pred1_corr")

        # pred=0, correct=0 -> pred=1, correct=0 (OFFSET APPLIED)
        mask = (query_preds == 0) & (query_correct == 0)
        if mask.any():
            _, idxs = nn_pred1_incorr.kneighbors(query_scaled[mask])
            idxs = idxs[:, k_offset:]  # skip k_offset nearest, take the rest
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred1_incorr[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_pred1_incorr[idxs].mean(axis=1)
            cf_paths[mask]  = get_cf_paths(idxs, "pred1_incorr")

        return diff_vecs, cf_probs, cf_paths

    train_diff, train_cf_probs, train_cf_paths = compute_diffs(train_clean, train_scaled)
    test_diff,  test_cf_probs,  test_cf_paths  = compute_diffs(test_clean,  test_scaled)

    diff_col_names = [f"delta_{c}" for c in relevant_cols]

    train_out = pd.concat([train_clean, pd.DataFrame(train_diff, columns=diff_col_names, index=train_clean.index)], axis=1)
    test_out  = pd.concat([test_clean,  pd.DataFrame(test_diff,  columns=diff_col_names, index=test_clean.index)],  axis=1)

    train_out["cf_prob"]  = train_cf_probs
    test_out["cf_prob"]   = test_cf_probs
    train_out["cf_paths"] = train_cf_paths
    test_out["cf_paths"]  = test_cf_paths

    return train_out, test_out, attr_scaler



def compute_cf_for_split_correct_cf(train_df, test_df, cf_count, disease, distance="l1"):
    """
    CF strategy where the counterfactual is always a correctly-classified training example.

    Routing by prediction only (no query correctness needed -> not leaky at test time):
      pred=1 (TP or FP) -> nearest TN  (train pred=0, correct=1)
      pred=0 (TN or FN) -> nearest TP  (train pred=1, correct=1)
    """
    disease_prob_col, disease_pred_col, disease_true_col = _get_disease_cols(disease)

    train_df = add_clinical_ratios(train_df.copy())
    test_df  = add_clinical_ratios(test_df.copy())

    meta_cols = {disease_prob_col, disease_pred_col, disease_true_col,
                 "correct", "path", "cam_path", "patient_id"}
    emb_cols  = [c for c in train_df.columns if c.startswith("emb_")]
    relevant_cols = [c for c in train_df.columns if c not in meta_cols and c not in emb_cols]

    train_clean = train_df.copy()
    test_clean  = test_df.copy()
    train_clean[relevant_cols] = train_clean[relevant_cols].fillna(0)
    test_clean[relevant_cols]  = test_clean[relevant_cols].fillna(0)

    attr_scaler  = StandardScaler()
    train_scaled = attr_scaler.fit_transform(train_clean[relevant_cols].values.astype(float))
    test_scaled  = attr_scaler.transform(test_clean[relevant_cols].values.astype(float))

    metric = _resolve_metric(distance)
    train_preds   = train_clean[disease_pred_col].values.astype(int)
    train_correct = train_clean["correct"].values.astype(int)

    # TP pool: pred=1 correct=1;  TN pool: pred=0 correct=1
    idx_tp = (train_preds == 1) & (train_correct == 1)
    idx_tn = (train_preds == 0) & (train_correct == 1)

    pool_global_idx = {
        "tp": np.where(idx_tp)[0],
        "tn": np.where(idx_tn)[0],
    }

    train_scaled_tp = train_scaled[idx_tp]
    train_scaled_tn = train_scaled[idx_tn]
    train_prob_tp   = train_clean[disease_prob_col].values[idx_tp]
    train_prob_tn   = train_clean[disease_prob_col].values[idx_tn]

    nn_tp = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_tp)
    nn_tn = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_tn)

    def _compute_diffs(df, query_scaled):
        query_preds = df[disease_pred_col].values.astype(int)
        n, d = query_scaled.shape
        diff_vecs = np.empty((n, d), dtype=np.float64)
        cf_probs  = np.empty(n, dtype=np.float64)
        cf_paths  = np.empty(n, dtype=object)

        # pred=1 (TP or FP) -> TN pool
        mask = (query_preds == 1)
        if mask.any():
            _, idxs = nn_tn.kneighbors(query_scaled[mask])
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_tn[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_tn[idxs].mean(axis=1)
            global_idxs     = pool_global_idx["tn"][idxs]
            cf_paths[mask]  = np.array(["|".join(train_clean["path"].values[row]) for row in global_idxs])

        # pred=0 (TN or FN) -> TP pool
        mask = (query_preds == 0)
        if mask.any():
            _, idxs = nn_tp.kneighbors(query_scaled[mask])
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_tp[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_tp[idxs].mean(axis=1)
            global_idxs     = pool_global_idx["tp"][idxs]
            cf_paths[mask]  = np.array(["|".join(train_clean["path"].values[row]) for row in global_idxs])

        return diff_vecs, cf_probs, cf_paths

    train_diff, train_cf_probs, train_cf_paths = _compute_diffs(train_clean, train_scaled)
    test_diff,  test_cf_probs,  test_cf_paths  = _compute_diffs(test_clean,  test_scaled)

    diff_col_names = [f"delta_{c}" for c in relevant_cols]

    train_out = pd.concat([train_clean, pd.DataFrame(train_diff, columns=diff_col_names, index=train_clean.index)], axis=1)
    test_out  = pd.concat([test_clean,  pd.DataFrame(test_diff,  columns=diff_col_names, index=test_clean.index)],  axis=1)

    train_out["cf_prob"]  = train_cf_probs
    test_out["cf_prob"]   = test_cf_probs
    train_out["cf_paths"] = train_cf_paths
    test_out["cf_paths"]  = test_cf_paths

    return train_out, test_out, attr_scaler

def compute_cf_for_split_gt_routing(train_df, test_df, cf_count, disease, distance="l1"):
    """
    CF strategy where the CF pool is partitioned by the ground-truth label of
    the training neighbours (not by their prediction).

    Routing for both train and test queries uses the query's prediction:
      pred=1 -> nearest training sample with true=0
      pred=0 -> nearest training sample with true=1

    Self-match guard: on the TRAIN split, incorrect samples (FP/FN) sit inside
    the pool they route to, so kneighbors(X=...) would return the query itself
    at distance 0. We fetch one extra candidate and drop that self-match. The
    TEST split can't overlap the train pool, so no guard is needed there.
    """
    disease_prob_col, disease_pred_col, disease_true_col = _get_disease_cols(disease)

    train_df = add_clinical_ratios(train_df.copy())
    test_df  = add_clinical_ratios(test_df.copy())

    meta_cols = {disease_prob_col, disease_pred_col, disease_true_col,
                 "correct", "path", "cam_path", "patient_id"}
    emb_cols  = [c for c in train_df.columns if c.startswith("emb_")]
    relevant_cols = [c for c in train_df.columns if c not in meta_cols and c not in emb_cols]

    train_clean = train_df.copy()
    test_clean  = test_df.copy()
    train_clean[relevant_cols] = train_clean[relevant_cols].fillna(0)
    test_clean[relevant_cols]  = test_clean[relevant_cols].fillna(0)

    attr_scaler  = StandardScaler()
    train_scaled = attr_scaler.fit_transform(train_clean[relevant_cols].values.astype(float))
    test_scaled  = attr_scaler.transform(test_clean[relevant_cols].values.astype(float))

    metric = _resolve_metric(distance)
    train_true = train_clean[disease_true_col].values.astype(int)

    # 2 pools partitioned by ground-truth label of the training neighbours
    idx_true1 = (train_true == 1)
    idx_true0 = (train_true == 0)

    pool_global_idx = {
        "true1": np.where(idx_true1)[0],
        "true0": np.where(idx_true0)[0],
    }

    train_scaled_true1 = train_scaled[idx_true1]
    train_scaled_true0 = train_scaled[idx_true0]
    train_prob_true1   = train_clean[disease_prob_col].values[idx_true1]
    train_prob_true0   = train_clean[disease_prob_col].values[idx_true0]

    nn_true1 = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_true1)
    nn_true0 = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_true0)

    def _get_cf_paths(idxs, pool_key):
        global_idxs = pool_global_idx[pool_key][idxs]
        return np.array(["|".join(train_clean["path"].values[row]) for row in global_idxs])

    def _compute_diffs(df, query_scaled, query_global_idx=None):
        # query_global_idx: row positions of the queries within train_clean.
        # Pass None for the test split (no overlap with the train pool is possible).
        query_preds = df[disease_pred_col].values.astype(int)
        n, d = query_scaled.shape
        diff_vecs = np.empty((n, d), dtype=np.float64)
        cf_probs  = np.empty(n, dtype=np.float64)
        cf_paths  = np.empty(n, dtype=object)

        def _kneighbors_excl_self(nn_model, pool_key, q_scaled, q_global_idx):
            # fetch one extra candidate so there's a fallback after dropping self-matches
            _, idxs = nn_model.kneighbors(q_scaled, n_neighbors=cf_count + 1)
            if q_global_idx is None:
                return idxs[:, :cf_count]
            pool_glob = pool_global_idx[pool_key]
            out = np.empty((q_scaled.shape[0], cf_count), dtype=idxs.dtype)
            for row in range(idxs.shape[0]):
                cand = idxs[row]
                keep = cand[pool_glob[cand] != q_global_idx[row]]
                out[row] = keep[:cf_count]
            return out

        # pred=1 -> nearest training sample with true=0
        mask = (query_preds == 1)
        if mask.any():
            g = query_global_idx[mask] if query_global_idx is not None else None
            idxs = _kneighbors_excl_self(nn_true0, "true0", query_scaled[mask], g)
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_true0[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_true0[idxs].mean(axis=1)
            cf_paths[mask]  = _get_cf_paths(idxs, "true0")

        # pred=0 -> nearest training sample with true=1
        mask = (query_preds == 0)
        if mask.any():
            g = query_global_idx[mask] if query_global_idx is not None else None
            idxs = _kneighbors_excl_self(nn_true1, "true1", query_scaled[mask], g)
            diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_true1[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_true1[idxs].mean(axis=1)
            cf_paths[mask]  = _get_cf_paths(idxs, "true1")

        return diff_vecs, cf_probs, cf_paths

    # TRAIN: pass global indices so self-matches (FP/FN in their own pool) are dropped
    train_diff, train_cf_probs, train_cf_paths = _compute_diffs(
        train_clean, train_scaled, query_global_idx=np.arange(len(train_clean))
    )
    # TEST: no guard needed — test queries are never in the train pool
    test_diff, test_cf_probs, test_cf_paths = _compute_diffs(
        test_clean, test_scaled, query_global_idx=None
    )

    diff_col_names = [f"delta_{c}" for c in relevant_cols]

    train_out = pd.concat(
        [train_clean, pd.DataFrame(train_diff, columns=diff_col_names, index=train_clean.index)], axis=1
    )
    test_out = pd.concat(
        [test_clean, pd.DataFrame(test_diff, columns=diff_col_names, index=test_clean.index)], axis=1
    )

    train_out["cf_prob"]  = train_cf_probs
    test_out["cf_prob"]   = test_cf_probs
    train_out["cf_paths"] = train_cf_paths
    test_out["cf_paths"]  = test_cf_paths

    return train_out, test_out, attr_scaler