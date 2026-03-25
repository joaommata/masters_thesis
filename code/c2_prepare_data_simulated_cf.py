"""
Fake Counterfactual C2 — Proof of Concept
==========================================
For each sample i, find the nearest training sample that C0 predicted
with the OPPOSITE label, restricted to disease-relevant attribute space.

The signal is a DIFFERENCE VECTOR (query - nearest_opposite_neighbour),
computed in standardised attribute space, rather than a single scalar distance.

Hypothesis:
  Correct   → sample sits firmly in C0's territory → large diff in relevant attrs
  Incorrect → sample is near C0's decision boundary → small diff in relevant attrs

Counterfactual matching strategy (4-pool):
  correct sample   → nearest correct sample with opposite prediction
  incorrect sample → nearest incorrect sample with opposite prediction
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

DISEASE  = "Effusion"    # "Effusion" | "Cardiomegaly" | "Pneumothorax" | "Atelectasis"
DISTANCE = "l1"   # l2 (Euclidean): penalises large deviations heavily (squares differences),
                  #    sensitive to outliers but rewards overall similarity across all attrs.
                  # l1 (Manhattan): sums absolute differences, more robust to outliers,
                  #    treats all attribute deviations equally regardless of magnitude.

BASE_DIR    = "/zhome/d0/a/221493/thesis"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR  = os.path.join(RESULTS_DIR, f"C2_sim_cf/{DISEASE.lower()}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the number of nearest neighbours to retrieve (k=1 for single CF)
cf_count = 1
# ══════════════════════════════════════════════════════════════════════════════

disease_prob_col = f"{DISEASE.lower()}_prob"
disease_pred_col = f"{DISEASE.lower()}_pred"
disease_true_col = f"{DISEASE.lower()}_true"


# ── Clinical ratio computation ────────────────────────────────────────────────

def add_clinical_ratios(df):
    df = df.copy()

    # Cardiothoracic ratio — most important for cardiomegaly. Normal < 0.5
    df['cardiothoracic_ratio'] = df['Heart_bbox_width'] / (df['Left Lung_bbox_width'] + df['Right Lung_bbox_width'])

    # Lung symmetry — asymmetry can indicate effusion or pneumothorax
    df['lung_area_ratio']   = df['Left Lung_area_pixels'] / (df['Right Lung_area_pixels'] + 1e-6)
    df['lung_height_ratio'] = df['Left Lung_bbox_height'] / (df['Right Lung_bbox_height'] + 1e-6)
    df['lung_width_ratio']  = df['Left Lung_bbox_width']  / (df['Right Lung_bbox_width']  + 1e-6)

    # Lung area relative to total — captures hyperinflation/collapse
    total_area = df['Left Lung_area_pixels'] + df['Right Lung_area_pixels']
    df['left_lung_fraction']  = df['Left Lung_area_pixels'] / (total_area + 1e-6)
    df['right_lung_fraction'] = df['Right Lung_area_pixels'] / (total_area + 1e-6)

    # Mediastinal width relative to lung width — widens in effusion/cardiomegaly
    df['mediastinal_ratio'] = df['Mediastinum_bbox_width'] / (df['Left Lung_bbox_width'] + df['Right Lung_bbox_width'] + 1e-6)

    return df


# ── Load & merge ──────────────────────────────────────────────────────────────

c0_train = pd.read_csv(os.path.join(RESULTS_DIR, f"C0_baseline/{DISEASE.lower()}/train_c0_{DISEASE.lower()}.csv"))
c0_valid = pd.read_csv(os.path.join(RESULTS_DIR, f"C0_baseline/{DISEASE.lower()}/valid_c0_{DISEASE.lower()}.csv"))
c1_train = pd.read_csv(os.path.join(RESULTS_DIR, "C1_attributes/train_c1_attribute_vector_rad.csv"))
c1_valid = pd.read_csv(os.path.join(RESULTS_DIR, "C1_attributes/valid_c1_attribute_vector_rad.csv"))

print(f"C0 — train: {len(c0_train):,}  valid: {len(c0_valid):,}")
print(f"C1 — train: {len(c1_train):,}  valid: {len(c1_valid):,}")

train_df = c0_train.merge(c1_train, on="path", how="inner")
valid_df = c0_valid.merge(c1_valid, on="path", how="inner")

for df in (train_df, valid_df):
    df.rename(columns={col: f"{DISEASE.lower()}_{col}"
                       for col in ("prob", "true", "pred") if col in df.columns},
              inplace=True)

print(f"Merged — train: {len(train_df):,}  valid: {len(valid_df):,}")
print(f"Train correctness: {train_df['correct'].value_counts().to_dict()}")
print(f"Valid correctness: {valid_df['correct'].value_counts().to_dict()}")


# ── Add clinical ratios ───────────────────────────────────────────────────────

train_df = add_clinical_ratios(train_df)
valid_df = add_clinical_ratios(valid_df)

# ── Select relevant columns (all non-meta attrs + clinical ratios) ────────────

# Non-attributes - shouldn't be included in delta-A
meta_cols = {disease_prob_col, disease_pred_col, disease_true_col,
             "correct", "path", "patient_id"}

# Added the embeddings from the last layer representations from C0
emb_cols = [c for c in train_df.columns if c.startswith("emb_")]

relevant_cols = [c for c in train_df.columns if c not in meta_cols and c not in emb_cols]
print(f"\nTotal feature columns : {len(relevant_cols)}")

# ── Fill NaN (from absent segmentations) with 0 ────────────
train_clean = train_df.copy()
valid_clean = valid_df.copy()
train_clean[relevant_cols] = train_clean[relevant_cols].fillna(0)
valid_clean[relevant_cols] = valid_clean[relevant_cols].fillna(0)
print(f"Filled NaNs with '0' — train: {len(train_clean):,}  valid: {len(valid_clean):,}")

# ── Scale relevant attributes (fit on train only) ─────────────────────────────

attr_scaler = StandardScaler()
train_scaled = attr_scaler.fit_transform(train_clean[relevant_cols].values.astype(float))
valid_scaled = attr_scaler.transform(valid_clean[relevant_cols].values.astype(float))

print(f"\nScaled {len(relevant_cols)} attributes (mean≈0, std≈1 on train).")


# ── Build 4 NN pools (pred × correctness) ────────────────────────────────────
# Matching strategy: correct samples match to correct CFs | incorrect → incorrect CFs
# This ensures the ΔA signal reflects C0's decision boundary, not class difficulty.

metric = "manhattan" if DISTANCE == "l1" else "euclidean"

train_preds   = train_clean[disease_pred_col].values.astype(int)
train_correct = train_clean["correct"].values.astype(int)

idx_pred1_corr   = (train_preds == 1) & (train_correct == 1)
idx_pred1_incorr = (train_preds == 1) & (train_correct == 0)
idx_pred0_corr   = (train_preds == 0) & (train_correct == 1)
idx_pred0_incorr = (train_preds == 0) & (train_correct == 0)

# Pool id to global idx mapping for retrieving paths later
pool_global_idx = {
    'pred1_corr':   np.where(idx_pred1_corr)[0],
    'pred1_incorr': np.where(idx_pred1_incorr)[0],
    'pred0_corr':   np.where(idx_pred0_corr)[0],
    'pred0_incorr': np.where(idx_pred0_incorr)[0],
}

# Extract the scaled relevant attributes for each pool to build separate NN models
train_scaled_pred1_corr   = train_scaled[idx_pred1_corr]
train_scaled_pred1_incorr = train_scaled[idx_pred1_incorr]
train_scaled_pred0_corr   = train_scaled[idx_pred0_corr]
train_scaled_pred0_incorr = train_scaled[idx_pred0_incorr]

# We want to save the probabilities of the matched CFs to use as features in M6 (ΔA + Prob(xi) + Prob(cf) + Attrs)
train_prob_pred1_corr   = train_clean[disease_prob_col].values[idx_pred1_corr]
train_prob_pred1_incorr = train_clean[disease_prob_col].values[idx_pred1_incorr]
train_prob_pred0_corr   = train_clean[disease_prob_col].values[idx_pred0_corr]
train_prob_pred0_incorr = train_clean[disease_prob_col].values[idx_pred0_incorr]

# The neighbour is *always* selected from the Train set since we don't "know" the correctness of the samples in the test set
nn_pred1_corr   = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred1_corr)
nn_pred1_incorr = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred1_incorr)
nn_pred0_corr   = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred0_corr)
nn_pred0_incorr = NearestNeighbors(n_neighbors=cf_count, metric=metric, n_jobs=-1).fit(train_scaled_pred0_incorr)

print(f"\nNN pools built ({DISTANCE.upper()}):")
print(f"  pred=1, correct   : {idx_pred1_corr.sum():,}")
print(f"  pred=1, incorrect : {idx_pred1_incorr.sum():,}")
print(f"  pred=0, correct   : {idx_pred0_corr.sum():,}")
print(f"  pred=0, incorrect : {idx_pred0_incorr.sum():,}")
print("Matching logic: correct sample → nearest correct CF | incorrect → nearest incorrect CF")


# ── Compute difference vectors ────────────────────────────────────────────────

def compute_diff_vectors(df, query_scaled,
                         nn_pred1_corr, nn_pred1_incorr,
                         nn_pred0_corr, nn_pred0_incorr,
                         train_scaled_pred1_corr, train_scaled_pred1_incorr,
                         train_scaled_pred0_corr, train_scaled_pred0_incorr,
                         train_prob_pred1_corr, train_prob_pred1_incorr,
                         train_prob_pred0_corr, train_prob_pred0_incorr,
                         train_paths,        # <-- pass train_clean["path"].values
                         pool_global_idx):   # <-- pass the dict above

    query_preds   = df[disease_pred_col].values.astype(int)
    query_correct = df["correct"].values.astype(int)
    n, d          = query_scaled.shape
    diff_vecs     = np.empty((n, d), dtype=np.float64)
    cf_probs      = np.empty(n, dtype=np.float64)
    # Each row: up to cf_count paths, separated by | so it fits in one CSV column
    cf_paths      = np.empty(n, dtype=object)

    # Helper: given pool-local idxs (n_query x k), recover paths from train_clean
    def get_cf_paths(idxs, pool_key):
        global_idxs = pool_global_idx[pool_key][idxs]  # (n_query, k)
        return np.array([
            "|".join(train_paths[row]) for row in global_idxs
        ])

    # pred=1, correct=1 → search in pred=0, correct=1
    mask = (query_preds == 1) & (query_correct == 1)
    if mask.any():
        _, idxs = nn_pred0_corr.kneighbors(query_scaled[mask])
        diff_vecs[mask]  = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred0_corr[idxs]).mean(axis=1)
        cf_probs[mask]   = train_prob_pred0_corr[idxs].mean(axis=1)
        cf_paths[mask]   = get_cf_paths(idxs, 'pred0_corr')

    # pred=1, correct=0 → search in pred=0, correct=0
    mask = (query_preds == 1) & (query_correct == 0)
    if mask.any():
        _, idxs = nn_pred0_incorr.kneighbors(query_scaled[mask])
        diff_vecs[mask]  = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred0_incorr[idxs]).mean(axis=1)
        cf_probs[mask]   = train_prob_pred0_incorr[idxs].mean(axis=1)
        cf_paths[mask]   = get_cf_paths(idxs, 'pred0_incorr')

    # pred=0, correct=1 → search in pred=1, correct=1
    mask = (query_preds == 0) & (query_correct == 1)
    if mask.any():
        _, idxs = nn_pred1_corr.kneighbors(query_scaled[mask])
        diff_vecs[mask]  = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred1_corr[idxs]).mean(axis=1)
        cf_probs[mask]   = train_prob_pred1_corr[idxs].mean(axis=1)
        cf_paths[mask]   = get_cf_paths(idxs, 'pred1_corr')

    # pred=0, correct=0 → search in pred=1, correct=0
    mask = (query_preds == 0) & (query_correct == 0)
    if mask.any():
        _, idxs = nn_pred1_incorr.kneighbors(query_scaled[mask])
        diff_vecs[mask]  = (query_scaled[mask][:, np.newaxis, :] - train_scaled_pred1_incorr[idxs]).mean(axis=1)
        cf_probs[mask]   = train_prob_pred1_incorr[idxs].mean(axis=1)
        cf_paths[mask]   = get_cf_paths(idxs, 'pred1_incorr')

    return diff_vecs, cf_probs, cf_paths

train_diff, train_cf_probs, train_cf_paths = compute_diff_vectors(
    train_clean, train_scaled,
    nn_pred1_corr, nn_pred1_incorr,
    nn_pred0_corr, nn_pred0_incorr,
    train_scaled_pred1_corr, train_scaled_pred1_incorr,
    train_scaled_pred0_corr, train_scaled_pred0_incorr,
    train_prob_pred1_corr, train_prob_pred1_incorr,
    train_prob_pred0_corr, train_prob_pred0_incorr,
    train_clean["path"].values,  # <-- new
    pool_global_idx,             # <-- new
)

valid_diff, valid_cf_probs, valid_cf_paths = compute_diff_vectors(
    valid_clean, valid_scaled,
    nn_pred1_corr, nn_pred1_incorr,
    nn_pred0_corr, nn_pred0_incorr,
    train_scaled_pred1_corr, train_scaled_pred1_incorr,
    train_scaled_pred0_corr, train_scaled_pred0_incorr,
    train_prob_pred1_corr, train_prob_pred1_incorr,
    train_prob_pred0_corr, train_prob_pred0_incorr,
    train_clean["path"].values,  # <-- same, always train paths
    pool_global_idx,             # <-- same
)


# ── Sanity check: is the ΔA magnitude different for correct/incorrect? ─────────

train_dist   = np.linalg.norm(train_diff, axis=1)
correct_mask = train_clean["correct"].values == 1

print(f"\nMean ΔA magnitude (L2 norm):")
print(f"  Correct   : {train_dist[correct_mask].mean():.4f}")
print(f"  Incorrect : {train_dist[~correct_mask].mean():.4f}")
print(f"  Δ         : {train_dist[correct_mask].mean() - train_dist[~correct_mask].mean():+.4f}")
print(f"\n  If Correct > Incorrect → the signal is working as hypothesised.")


# ── Save enriched dataframes and scaler ──────────────────────────────────────
diff_col_names = [f"delta_{c}" for c in relevant_cols]


train_out = train_clean.copy()
valid_out = valid_clean.copy()
for i, col in enumerate(diff_col_names):
    train_out[col] = train_diff[:, i]
    valid_out[col] = valid_diff[:, i]
    
# Save the probs for the found CF
train_out['cf_prob'] = train_cf_probs
valid_out['cf_prob'] = valid_cf_probs

# Save the paths of the found CFs (pipe-separated if multiple) for manual inspection later
train_out['cf_paths'] = train_cf_paths
valid_out['cf_paths'] = valid_cf_paths

train_out.to_csv(os.path.join(OUTPUT_DIR, f"train_with_diff_vectors_{cf_count}.csv"), index=False)
valid_out.to_csv(os.path.join(OUTPUT_DIR, f"valid_with_diff_vectors_{cf_count}.csv"), index=False)
joblib.dump(attr_scaler, os.path.join(OUTPUT_DIR, f"attr_scaler_{cf_count}.pkl"))

print(f"\nAll outputs saved → {OUTPUT_DIR}")