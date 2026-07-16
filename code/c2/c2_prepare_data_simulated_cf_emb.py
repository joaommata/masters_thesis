"""
Simulated Counterfactual C2 — Embedding-space CF selection (standalone experiment)
===================================================================================
Standalone variant of c2_prepare_data_simulated_cf.py: the nearest opposite-prediction
neighbour is found in standardized CNN/ViT EMBEDDING space (cosine by default) instead
of standardized attribute space. The output schema is unchanged — delta_<attr_col> is
still the attribute-space difference between query and the selected neighbour, so the
result stays plug-compatible with build_feature_matrices() in c2_cv_pipeline_new_split.py.
Only the *choice* of which training sample is the counterfactual changes.

Does not modify c2_prepare_data_simulated_cf.py or c2_cv_pipeline_new_split.py.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from c2_prepare_data_simulated_cf import add_clinical_ratios, _get_disease_cols, _resolve_metric


def compute_cf_for_split_correct_cf_emb(train_df, test_df, cf_count, disease, distance="cosine"):
    """
    Embedding-space version of compute_cf_for_split_correct_cf.

    Neighbour search runs on standardized emb_* columns (cosine by default).
    Routing by prediction only (no query correctness needed -> not leaky at test time):
      pred=1 (TP or FP) -> nearest TN  (train pred=0, correct=1)
      pred=0 (TN or FN) -> nearest TP  (train pred=1, correct=1)
    """
    disease_prob_col, disease_pred_col, disease_true_col = _get_disease_cols(disease)

    train_df = add_clinical_ratios(train_df.copy())
    test_df  = add_clinical_ratios(test_df.copy())

    meta_cols = {disease_prob_col, disease_pred_col, disease_true_col,
                 "correct", "path", "patient_id"}
    emb_cols      = [c for c in train_df.columns if c.startswith("emb_")]
    relevant_cols = [c for c in train_df.columns if c not in meta_cols and c not in emb_cols]

    if not emb_cols:
        raise ValueError("No emb_* columns found — embedding-space CF search requires embedding features.")

    train_clean = train_df.copy()
    test_clean  = test_df.copy()
    train_clean[relevant_cols] = train_clean[relevant_cols].fillna(0)
    test_clean[relevant_cols]  = test_clean[relevant_cols].fillna(0)
    train_clean[emb_cols] = train_clean[emb_cols].fillna(0)
    test_clean[emb_cols]  = test_clean[emb_cols].fillna(0)

    # Attribute space: only used to compute delta_* output, not for neighbour search.
    attr_scaler  = StandardScaler()
    train_attr_scaled = attr_scaler.fit_transform(train_clean[relevant_cols].values.astype(float))
    test_attr_scaled  = attr_scaler.transform(test_clean[relevant_cols].values.astype(float))

    # Embedding space: used for neighbour search.
    emb_scaler  = StandardScaler()
    train_emb_scaled = emb_scaler.fit_transform(train_clean[emb_cols].values.astype(float))
    test_emb_scaled  = emb_scaler.transform(test_clean[emb_cols].values.astype(float))

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

    train_emb_tp  = train_emb_scaled[idx_tp]
    train_emb_tn  = train_emb_scaled[idx_tn]
    train_attr_tp = train_attr_scaled[idx_tp]
    train_attr_tn = train_attr_scaled[idx_tn]
    train_prob_tp = train_clean[disease_prob_col].values[idx_tp]
    train_prob_tn = train_clean[disease_prob_col].values[idx_tn]

    nn_tp = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_emb_tp)
    nn_tn = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_emb_tn)

    def _compute_diffs(df, query_emb_scaled, query_attr_scaled):
        query_preds = df[disease_pred_col].values.astype(int)
        n, d = query_attr_scaled.shape
        diff_vecs = np.empty((n, d), dtype=np.float64)
        cf_probs  = np.empty(n, dtype=np.float64)
        cf_paths  = np.empty(n, dtype=object)

        # pred=1 (TP or FP) -> TN pool, selected by embedding similarity
        mask = (query_preds == 1)
        if mask.any():
            _, idxs = nn_tn.kneighbors(query_emb_scaled[mask])
            diff_vecs[mask] = (query_attr_scaled[mask][:, np.newaxis, :] - train_attr_tn[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_tn[idxs].mean(axis=1)
            global_idxs     = pool_global_idx["tn"][idxs]
            cf_paths[mask]  = np.array(["|".join(train_clean["path"].values[row]) for row in global_idxs])

        # pred=0 (TN or FN) -> TP pool, selected by embedding similarity
        mask = (query_preds == 0)
        if mask.any():
            _, idxs = nn_tp.kneighbors(query_emb_scaled[mask])
            diff_vecs[mask] = (query_attr_scaled[mask][:, np.newaxis, :] - train_attr_tp[idxs]).mean(axis=1)
            cf_probs[mask]  = train_prob_tp[idxs].mean(axis=1)
            global_idxs     = pool_global_idx["tp"][idxs]
            cf_paths[mask]  = np.array(["|".join(train_clean["path"].values[row]) for row in global_idxs])

        return diff_vecs, cf_probs, cf_paths

    train_diff, train_cf_probs, train_cf_paths = _compute_diffs(train_clean, train_emb_scaled, train_attr_scaled)
    test_diff,  test_cf_probs,  test_cf_paths  = _compute_diffs(test_clean,  test_emb_scaled,  test_attr_scaled)

    diff_col_names = [f"delta_{c}" for c in relevant_cols]

    train_out = pd.concat([train_clean, pd.DataFrame(train_diff, columns=diff_col_names, index=train_clean.index)], axis=1)
    test_out  = pd.concat([test_clean,  pd.DataFrame(test_diff,  columns=diff_col_names, index=test_clean.index)],  axis=1)

    train_out["cf_prob"]  = train_cf_probs
    test_out["cf_prob"]   = test_cf_probs
    train_out["cf_paths"] = train_cf_paths
    test_out["cf_paths"]  = test_cf_paths

    return train_out, test_out, attr_scaler
