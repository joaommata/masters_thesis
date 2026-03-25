"""
c2_prepare_unmatched_cf.py
==========================
Identical pipeline to the matched counterfactual preparation script,
but with the correctness-matching condition REMOVED from pool construction.

Instead of 4 pools (pred x correctness), we use 2 pools (pred only):
    pred=1 → search in all pred=0 training samples (regardless of correctness)
    pred=0 → search in all pred=1 training samples (regardless of correctness)

Purpose: ablation study to test whether correctness-matching matters.
Compare the C2 performance trained on these CSVs vs the matched versions.

Outputs saved to results/C2_sim_cf/{disease}/ with '_unmatched' suffix:
    - train_with_diff_vectors_{cf_count}_unmatched.csv
    - valid_with_diff_vectors_{cf_count}_unmatched.csv
    - attr_scaler_{cf_count}_unmatched.pkl
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit this block only
# ══════════════════════════════════════════════════════════════════════════════

DISEASE   = "Effusion"
DISTANCE  = "l1"
cf_count  = 1       # set to 3 or 5 to match your multi-CF experiments
BASE_DIR  = "/zhome/d0/a/221493/thesis"

# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR  = os.path.join(RESULTS_DIR, f"C2_sim_cf/{DISEASE.lower()}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

disease_prob_col = f"{DISEASE.lower()}_prob"
disease_pred_col = f"{DISEASE.lower()}_pred"
disease_true_col = f"{DISEASE.lower()}_true"


# ── Clinical ratios (identical to matched script) ─────────────────────────────

def add_clinical_ratios(df):
    df = df.copy()
    df['cardiothoracic_ratio'] = df['Heart_bbox_width'] / (df['Left Lung_bbox_width'] + df['Right Lung_bbox_width'])
    df['lung_area_ratio']      = df['Left Lung_area_pixels'] / (df['Right Lung_area_pixels'] + 1e-6)
    df['lung_height_ratio']    = df['Left Lung_bbox_height'] / (df['Right Lung_bbox_height'] + 1e-6)
    df['lung_width_ratio']     = df['Left Lung_bbox_width']  / (df['Right Lung_bbox_width']  + 1e-6)
    total_area = df['Left Lung_area_pixels'] + df['Right Lung_area_pixels']
    df['left_lung_fraction']   = df['Left Lung_area_pixels'] / (total_area + 1e-6)
    df['right_lung_fraction']  = df['Right Lung_area_pixels'] / (total_area + 1e-6)
    df['mediastinal_ratio']    = df['Mediastinum_bbox_width'] / (df['Left Lung_bbox_width'] + df['Right Lung_bbox_width'] + 1e-6)
    return df


# ── Load & merge ──────────────────────────────────────────────────────────────

c0_train = pd.read_csv(os.path.join(RESULTS_DIR, f"C0_baseline/{DISEASE.lower()}/train_c0_{DISEASE.lower()}.csv"))
c0_valid = pd.read_csv(os.path.join(RESULTS_DIR, f"C0_baseline/{DISEASE.lower()}/valid_c0_{DISEASE.lower()}.csv"))
c1_train = pd.read_csv(os.path.join(RESULTS_DIR, "C1_attributes/train_c1_attribute_vector_rad.csv"))
c1_valid = pd.read_csv(os.path.join(RESULTS_DIR, "C1_attributes/valid_c1_attribute_vector_rad.csv"))

train_df = c0_train.merge(c1_train, on="path", how="inner")
valid_df = c0_valid.merge(c1_valid, on="path", how="inner")

for df in (train_df, valid_df):
    df.rename(columns={col: f"{DISEASE.lower()}_{col}"
                       for col in ("prob", "true", "pred") if col in df.columns},
              inplace=True)

print(f"Merged — train: {len(train_df):,}  valid: {len(valid_df):,}")

train_df = add_clinical_ratios(train_df)
valid_df = add_clinical_ratios(valid_df)


# ── Feature selection ─────────────────────────────────────────────────────────

meta_cols = {disease_prob_col, disease_pred_col, disease_true_col,
             "correct", "path", "patient_id"}
emb_cols  = [c for c in train_df.columns if c.startswith("emb_")]
relevant_cols = [c for c in train_df.columns if c not in meta_cols and c not in emb_cols]

train_clean = train_df.copy()
valid_clean = valid_df.copy()
train_clean[relevant_cols] = train_clean[relevant_cols].fillna(0)
valid_clean[relevant_cols] = valid_clean[relevant_cols].fillna(0)

attr_scaler  = StandardScaler()
train_scaled = attr_scaler.fit_transform(train_clean[relevant_cols].values.astype(float))
valid_scaled = attr_scaler.transform(valid_clean[relevant_cols].values.astype(float))

print(f"Scaled {len(relevant_cols)} attributes.")


# ── Build 2 UNMATCHED pools (pred only, correctness ignored) ──────────────────
# KEY DIFFERENCE FROM MATCHED SCRIPT:
# Matched script builds 4 pools (pred=1/correct, pred=1/incorrect,
#                                  pred=0/correct, pred=0/incorrect)
# This script collapses to 2 pools — all pred=1 and all pred=0,
# ignoring whether those samples were correctly predicted by C0.

metric = "manhattan" if DISTANCE == "l1" else "euclidean"

train_preds = train_clean[disease_pred_col].values.astype(int)

idx_pred1 = (train_preds == 1)   # all pred=1, any correctness
idx_pred0 = (train_preds == 0)   # all pred=0, any correctness

# Global index mapping — needed to recover CF paths from pool-local kneighbors indices
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

print(f"\nUnmatched NN pools ({DISTANCE.upper()}):")
print(f"  pred=1 (all correctness) : {idx_pred1.sum():,}")
print(f"  pred=0 (all correctness) : {idx_pred0.sum():,}")
print("Matching logic: pred=1 → nearest pred=0 | pred=0 → nearest pred=1 (correctness ignored)")


# ── Compute difference vectors ────────────────────────────────────────────────

def compute_diff_vectors(df, query_scaled):
    """
    For each sample, find the k nearest training samples with the OPPOSITE
    prediction (correctness ignored) and return the mean difference vector.
    Also returns the mean CF probability and the CF paths.
    """
    query_preds = df[disease_pred_col].values.astype(int)
    n, d        = query_scaled.shape

    diff_vecs = np.empty((n, d),      dtype=np.float64)
    cf_probs  = np.empty(n,           dtype=np.float64)
    cf_paths  = np.empty(n,           dtype=object)

    def get_cf_paths(idxs, pool_key):
        # idxs is (n_query, k) — pool-local indices from kneighbors
        global_idxs = pool_global_idx[pool_key][idxs]
        return np.array([
            "|".join(train_clean["path"].values[row]) for row in global_idxs
        ])

    # pred=1 → search in all pred=0
    mask = (query_preds == 1)
    if mask.any():
        _, idxs      = nn_pred0.kneighbors(query_scaled[mask])
        diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred0[idxs]).mean(axis=1)
        cf_probs[mask]  = train_prob_pred0[idxs].mean(axis=1)
        cf_paths[mask]  = get_cf_paths(idxs, 'pred0')

    # pred=0 → search in all pred=1
    mask = (query_preds == 0)
    if mask.any():
        _, idxs      = nn_pred1.kneighbors(query_scaled[mask])
        diff_vecs[mask] = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred1[idxs]).mean(axis=1)
        cf_probs[mask]  = train_prob_pred1[idxs].mean(axis=1)
        cf_paths[mask]  = get_cf_paths(idxs, 'pred1')

    return diff_vecs, cf_probs, cf_paths


print("\nComputing unmatched ΔA difference vectors...")
train_diff, train_cf_probs, train_cf_paths = compute_diff_vectors(train_clean, train_scaled)
valid_diff, valid_cf_probs, valid_cf_paths = compute_diff_vectors(valid_clean, valid_scaled)

print(f"ΔA shape — train: {train_diff.shape}  valid: {valid_diff.shape}")


# ── Sanity check ──────────────────────────────────────────────────────────────
# Expect correct > incorrect if signal is present, same as matched script.
# If this gap is smaller than in the matched version, that supports matching.

train_dist   = np.linalg.norm(train_diff, axis=1)
correct_mask = train_clean["correct"].values == 1

print(f"\nMean ΔA magnitude (L2 norm) — UNMATCHED:")
print(f"  Correct   : {train_dist[correct_mask].mean():.4f}")
print(f"  Incorrect : {train_dist[~correct_mask].mean():.4f}")
print(f"  Δ         : {train_dist[correct_mask].mean() - train_dist[~correct_mask].mean():+.4f}")
print(f"\n  Compare this gap to the matched script output.")
print(f"  A smaller gap here supports the correctness-matching design choice.")


# ── Save outputs ──────────────────────────────────────────────────────────────

diff_col_names = [f"delta_{c}" for c in relevant_cols]

train_out = train_clean.copy()
valid_out = valid_clean.copy()

for i, col in enumerate(diff_col_names):
    train_out[col] = train_diff[:, i]
    valid_out[col] = valid_diff[:, i]

train_out['cf_prob'] = train_cf_probs
valid_out['cf_prob'] = valid_cf_probs

train_out['cf_paths'] = train_cf_paths
valid_out['cf_paths'] = valid_cf_paths

# '_unmatched' suffix prevents any confusion with matched output files
train_out.to_csv(os.path.join(OUTPUT_DIR, f"train_with_diff_vectors_{cf_count}_unmatched.csv"), index=False)
valid_out.to_csv(os.path.join(OUTPUT_DIR, f"valid_with_diff_vectors_{cf_count}_unmatched.csv"), index=False)
joblib.dump(attr_scaler, os.path.join(OUTPUT_DIR, f"attr_scaler_{cf_count}_unmatched.pkl"))

print(f"\nAll unmatched outputs saved → {OUTPUT_DIR}")