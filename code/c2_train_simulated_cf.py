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
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report


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
c1_train = pd.read_csv(os.path.join(RESULTS_DIR, "C1_attributes/train_c1_attribute_vector.csv")) ### CHANGED
c1_valid = pd.read_csv(os.path.join(RESULTS_DIR, "C1_attributes/valid_c1_attribute_vector.csv"))

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

train_scaled_pred1_corr   = train_scaled[idx_pred1_corr]
train_scaled_pred1_incorr = train_scaled[idx_pred1_incorr]
train_scaled_pred0_corr   = train_scaled[idx_pred0_corr]
train_scaled_pred0_incorr = train_scaled[idx_pred0_incorr]

# The neighboru is *always* selected from the Train set since we don't "know" the correctness of the samples in the test set
nn_pred1_corr   = NearestNeighbors(n_neighbors=1, metric=metric, n_jobs=-1).fit(train_scaled_pred1_corr)
nn_pred1_incorr = NearestNeighbors(n_neighbors=1, metric=metric, n_jobs=-1).fit(train_scaled_pred1_incorr)
nn_pred0_corr   = NearestNeighbors(n_neighbors=1, metric=metric, n_jobs=-1).fit(train_scaled_pred0_corr)
nn_pred0_incorr = NearestNeighbors(n_neighbors=1, metric=metric, n_jobs=-1).fit(train_scaled_pred0_incorr)

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
                         train_scaled_pred0_corr, train_scaled_pred0_incorr):
    """
    For each sample, find the nearest training sample that C0 predicted with the
    OPPOSITE label (matched by correctness) and return (query - neighbour).

    KEY CHOICE: We use C0's PREDICTION (pred), NOT the ground truth (true).
    A real counterfactual flips C0's decision — so our fake CF must too.

    Shape of output: (n_samples, n_relevant_attrs)
    """
    query_preds   = df[disease_pred_col].values.astype(int)
    query_correct = df["correct"].values.astype(int)
    n, d          = query_scaled.shape
    diff_vecs     = np.empty((n, d), dtype=np.float64)

    # pred=1, correct=1 → search in pred=0, correct=1
    mask = (query_preds == 1) & (query_correct == 1)
    if mask.any():
        _, idxs = nn_pred0_corr.kneighbors(query_scaled[mask])
        diff_vecs[mask] = query_scaled[mask] - train_scaled_pred0_corr[idxs[:, 0]]

    # pred=1, correct=0 → search in pred=0, correct=0
    mask = (query_preds == 1) & (query_correct == 0)
    if mask.any():
        _, idxs = nn_pred0_incorr.kneighbors(query_scaled[mask])
        diff_vecs[mask] = query_scaled[mask] - train_scaled_pred0_incorr[idxs[:, 0]]

    # pred=0, correct=1 → search in pred=1, correct=1
    mask = (query_preds == 0) & (query_correct == 1)
    if mask.any():
        _, idxs = nn_pred1_corr.kneighbors(query_scaled[mask])
        diff_vecs[mask] = query_scaled[mask] - train_scaled_pred1_corr[idxs[:, 0]]

    # pred=0, correct=0 → search in pred=1, correct=0
    mask = (query_preds == 0) & (query_correct == 0)
    if mask.any():
        _, idxs = nn_pred1_incorr.kneighbors(query_scaled[mask])
        diff_vecs[mask] = query_scaled[mask] - train_scaled_pred1_incorr[idxs[:, 0]]

    return diff_vecs


print("\nComputing ΔA difference vectors...")
train_diff = compute_diff_vectors(
    train_clean, train_scaled,
    nn_pred1_corr, nn_pred1_incorr,
    nn_pred0_corr, nn_pred0_incorr,
    train_scaled_pred1_corr, train_scaled_pred1_incorr,
    train_scaled_pred0_corr, train_scaled_pred0_incorr,
)
valid_diff = compute_diff_vectors(
    valid_clean, valid_scaled,
    nn_pred1_corr, nn_pred1_incorr,
    nn_pred0_corr, nn_pred0_incorr,
    train_scaled_pred1_corr, train_scaled_pred1_incorr,
    train_scaled_pred0_corr, train_scaled_pred0_incorr,
)

diff_col_names = [f"delta_{c}" for c in relevant_cols]
print(f"ΔA shape — train: {train_diff.shape}  valid: {valid_diff.shape}")


# ── Sanity check: is the ΔA magnitude different for correct/incorrect? ─────────

train_dist   = np.linalg.norm(train_diff, axis=1)
correct_mask = train_clean["correct"].values == 1

print(f"\nMean ΔA magnitude (L2 norm):")
print(f"  Correct   : {train_dist[correct_mask].mean():.4f}")
print(f"  Incorrect : {train_dist[~correct_mask].mean():.4f}")
print(f"  Δ         : {train_dist[correct_mask].mean() - train_dist[~correct_mask].mean():+.4f}")
print(f"\n  If Correct > Incorrect → the signal is working as hypothesised.")


# ── Train & evaluate C2 models (LR only) ──────────────────────────────────────

y_train = train_clean["correct"].values
y_valid = valid_clean["correct"].values


def train_and_eval(X_train, X_valid, y_train, y_valid, label):
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_valid)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_tr_sc, y_train)

    y_prob      = model.predict_proba(X_va_sc)[:, 1]
    y_pred      = model.predict(X_va_sc)
    auc         = roc_auc_score(y_valid, y_prob)
    fpr, tpr, _ = roc_curve(y_valid, y_prob)

    print(f"\n{'='*55}")
    print(f"  {label:<47}  AUC = {auc:.4f}")
    print(f"{'='*55}")
    print(classification_report(y_valid, y_pred, target_names=["Incorrect", "Correct"]))

    return auc, fpr, tpr, model, scaler

# ── Baselines ─────────────────────────────────────────────────────────────────

# B1: C0 probability only
auc_base, fpr_base, tpr_base, m_base, sc_base = train_and_eval(
    train_clean[[disease_prob_col]].values,
    valid_clean[[disease_prob_col]].values,
    y_train, y_valid,
    "B1 — prob only"
)

# B2: original attributes only
auc_attr, fpr_attr, tpr_attr, m_attr, sc_attr = train_and_eval(
    train_scaled,
    valid_scaled,
    y_train, y_valid,
    "B2 — attributes only"
)

# B3: C0 embeddings only
auc_emb, fpr_emb, tpr_emb, m_emb, sc_emb = train_and_eval(
    train_clean[emb_cols].values,
    valid_clean[emb_cols].values,
    y_train, y_valid,
    "B3 — embeddings only"
)

# B4: prob + attributes (no ΔA)
auc_prob_attr, fpr_prob_attr, tpr_prob_attr, m_prob_attr, sc_prob_attr = train_and_eval(
    np.hstack([train_clean[[disease_prob_col]].values, train_scaled]),
    np.hstack([valid_clean[[disease_prob_col]].values, valid_scaled]),
    y_train, y_valid,
    "B4 — prob + attributes"
)

# B5: prob + embeddings (no ΔA)
auc_prob_emb, fpr_prob_emb, tpr_prob_emb, m_prob_emb, sc_prob_emb = train_and_eval(
    np.hstack([train_clean[[disease_prob_col]].values, train_clean[emb_cols].values]),
    np.hstack([valid_clean[[disease_prob_col]].values, valid_clean[emb_cols].values]),
    y_train, y_valid,
    "B5 — prob + embeddings"
)

# ── ΔA models ─────────────────────────────────────────────────────────────────

# M1: ΔA only
auc_diff, fpr_diff, tpr_diff, m_diff, sc_diff = train_and_eval(
    train_diff,
    valid_diff,
    y_train, y_valid,
    f"M1 — ΔA only  ({DISTANCE.upper()})"
)

# M2: prob + ΔA
auc_comb, fpr_comb, tpr_comb, m_comb, sc_comb = train_and_eval(
    np.hstack([train_clean[[disease_prob_col]].values, train_diff]),
    np.hstack([valid_clean[[disease_prob_col]].values, valid_diff]),
    y_train, y_valid,
    f"M2 — prob + ΔA  ({DISTANCE.upper()})"
)

# M3: prob + ΔA + attributes
auc_ext, fpr_ext, tpr_ext, m_ext, sc_ext = train_and_eval(
    np.hstack([train_clean[[disease_prob_col]].values, train_diff, train_scaled]),
    np.hstack([valid_clean[[disease_prob_col]].values, valid_diff, valid_scaled]),
    y_train, y_valid,
    f"M3 — prob + ΔA + attributes  ({DISTANCE.upper()})"
)

# M4: prob + ΔA + embeddings
auc_emb_comb, fpr_emb_comb, tpr_emb_comb, m_emb_comb, sc_emb_comb = train_and_eval(
    np.hstack([train_clean[[disease_prob_col]].values, train_diff, train_clean[emb_cols].values]),
    np.hstack([valid_clean[[disease_prob_col]].values, valid_diff, valid_clean[emb_cols].values]),
    y_train, y_valid,
    f"M4 — prob + ΔA + embeddings  ({DISTANCE.upper()})"
)

# M5: prob + ΔA + attributes + embeddings
auc_full, fpr_full, tpr_full, m_full, sc_full = train_and_eval(
    np.hstack([train_clean[[disease_prob_col]].values, train_diff, train_scaled, train_clean[emb_cols].values]),
    np.hstack([valid_clean[[disease_prob_col]].values, valid_diff, valid_scaled, valid_clean[emb_cols].values]),
    y_train, y_valid,
    f"M5 — prob + ΔA + attributes + embeddings  ({DISTANCE.upper()})"
)


# ── Plot ROC ──────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 7))

# Baselines (dashed)
ax.plot(fpr_base,      tpr_base,      color="gray",      lw=2, ls="--", label=f"B1 — prob only                       AUC={auc_base:.3f}")
ax.plot(fpr_attr,      tpr_attr,      color="purple",    lw=2, ls="--", label=f"B2 — attributes only                 AUC={auc_attr:.3f}")
ax.plot(fpr_emb,       tpr_emb,       color="red",       lw=2, ls="--", label=f"B3 — embeddings only                 AUC={auc_emb:.3f}")
ax.plot(fpr_prob_attr, tpr_prob_attr, color="plum",      lw=2, ls="--", label=f"B4 — prob + attributes               AUC={auc_prob_attr:.3f}")
ax.plot(fpr_prob_emb,  tpr_prob_emb,  color="salmon",    lw=2, ls="--", label=f"B5 — prob + embeddings               AUC={auc_prob_emb:.3f}")

# ΔA models (solid)
ax.plot(fpr_diff,      tpr_diff,      color="darkorange", lw=2, label=f"M1 — ΔA only                          AUC={auc_diff:.3f}")
ax.plot(fpr_comb,      tpr_comb,      color="steelblue",  lw=2, label=f"M2 — prob + ΔA                        AUC={auc_comb:.3f}")
ax.plot(fpr_ext,       tpr_ext,       color="green",      lw=2, label=f"M3 — prob + ΔA + attributes           AUC={auc_ext:.3f}")
ax.plot(fpr_emb_comb,  tpr_emb_comb,  color="teal",       lw=2, label=f"M4 — prob + ΔA + embeddings           AUC={auc_emb_comb:.3f}")
ax.plot(fpr_full,      tpr_full,      color="darkgreen",  lw=2, label=f"M5 — prob + ΔA + attrs + embeddings   AUC={auc_full:.3f}")

ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"C2 Quality Control — {DISEASE}  ({DISTANCE.upper()})")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_comparison.png"), dpi=150)
plt.show()


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\nDisease        : {DISEASE}")
print(f"Distance       : {DISTANCE.upper()}")
print(f"Train size     : {len(train_clean):,}  (correct: {y_train.sum():,} | incorrect: {(1-y_train).sum():,})")
print(f"Valid size     : {len(valid_clean):,}  (correct: {y_valid.sum():,} | incorrect: {(1-y_valid).sum():,})")
print(f"Attribute cols : {len(relevant_cols)}")
print(f"Embedding cols : {len(emb_cols)}")
print(f"ΔA shape       : train={train_diff.shape}  valid={valid_diff.shape}")
print(f"\n── Baselines ──────────────────────────────────────")
print(f"  B1  prob only                     : {auc_base:.4f}")
print(f"  B2  attributes only               : {auc_attr:.4f}")
print(f"  B3  embeddings only               : {auc_emb:.4f}")
print(f"  B4  prob + attributes             : {auc_prob_attr:.4f}")
print(f"  B5  prob + embeddings             : {auc_prob_emb:.4f}")
print(f"\n── ΔA Models ──────────────────────────────────────")
print(f"  M1  ΔA only                       : {auc_diff:.4f}")
print(f"  M2  prob + ΔA                     : {auc_comb:.4f}")
print(f"  M3  prob + ΔA + attributes        : {auc_ext:.4f}")
print(f"  M4  prob + ΔA + embeddings        : {auc_emb_comb:.4f}")
print(f"  M5  prob + ΔA + attrs + embeddings: {auc_full:.4f}")


# ── Save artefacts ────────────────────────────────────────────────────────────

for name, m, sc in [
    ("baseline",  m_base,      sc_base),
    ("attr",      m_attr,      sc_attr),
    ("emb",       m_emb,       sc_emb),
    ("prob_attr", m_prob_attr, sc_prob_attr),
    ("prob_emb",  m_prob_emb,  sc_prob_emb),
    ("diff",      m_diff,      sc_diff),
    ("comb",      m_comb,      sc_comb),
    ("ext",       m_ext,       sc_ext),
    ("emb_comb",  m_emb_comb,  sc_emb_comb),
    ("full",      m_full,      sc_full),
]:
    joblib.dump(m,  os.path.join(OUTPUT_DIR, f"c2_{name}_model.pkl"))
    joblib.dump(sc, os.path.join(OUTPUT_DIR, f"c2_{name}_scaler.pkl"))

joblib.dump(attr_scaler, os.path.join(OUTPUT_DIR, "attr_scaler.pkl"))


# ── Save results summary ─────────────────────────────────────────────────────

summary_path = os.path.join(OUTPUT_DIR, f"c2_results_summary_{DISEASE.lower()}_{DISTANCE}.txt")
with open(summary_path, "w") as f:
    f.write(f"C2 Quality Control — Results Summary\n")
    f.write(f"{'='*55}\n")
    f.write(f"Disease        : {DISEASE}\n")
    f.write(f"Distance       : {DISTANCE.upper()}\n")
    f.write(f"Train size     : {len(train_clean):,}  (correct: {y_train.sum():,} | incorrect: {(1-y_train).sum():,})\n")
    f.write(f"Valid size     : {len(valid_clean):,}  (correct: {y_valid.sum():,} | incorrect: {(1-y_valid).sum():,})\n")
    f.write(f"Attribute cols : {len(relevant_cols)}\n")
    f.write(f"Embedding cols : {len(emb_cols)}\n")
    f.write(f"ΔA shape       : train={train_diff.shape}  valid={valid_diff.shape}\n")
    f.write(f"\n── Baselines ──────────────────────────────────────\n")
    f.write(f"  B1  prob only                     : {auc_base:.4f}\n")
    f.write(f"  B2  attributes only               : {auc_attr:.4f}\n")
    f.write(f"  B3  embeddings only               : {auc_emb:.4f}\n")
    f.write(f"  B4  prob + attributes             : {auc_prob_attr:.4f}\n")
    f.write(f"  B5  prob + embeddings             : {auc_prob_emb:.4f}\n")
    f.write(f"\n── ΔA Models ──────────────────────────────────────\n")
    f.write(f"  M1  ΔA only                       : {auc_diff:.4f}\n")
    f.write(f"  M2  prob + ΔA                     : {auc_comb:.4f}\n")
    f.write(f"  M3  prob + ΔA + attributes        : {auc_ext:.4f}\n")
    f.write(f"  M4  prob + ΔA + embeddings        : {auc_emb_comb:.4f}\n")
    f.write(f"  M5  prob + ΔA + attrs + embeddings: {auc_full:.4f}\n")
    f.write(f"\n── Best model ─────────────────────────────────────\n")
    all_aucs = {
        "B1 prob only": auc_base, "B2 attributes only": auc_attr,
        "B3 embeddings only": auc_emb, "B4 prob + attributes": auc_prob_attr,
        "B5 prob + embeddings": auc_prob_emb, "M1 ΔA only": auc_diff,
        "M2 prob + ΔA": auc_comb, "M3 prob + ΔA + attributes": auc_ext,
        "M4 prob + ΔA + embeddings": auc_emb_comb,
        "M5 prob + ΔA + attrs + embeddings": auc_full,
    }
    best_name = max(all_aucs, key=all_aucs.get)
    f.write(f"  {best_name:<40}: {all_aucs[best_name]:.4f}\n")
print(f"Results summary saved → {summary_path}")

# Save enriched dataframes with ΔA vector components attached
train_out = train_clean.copy()
valid_out = valid_clean.copy()
for i, col in enumerate(diff_col_names):
    train_out[col] = train_diff[:, i]
    valid_out[col] = valid_diff[:, i]

train_out.to_csv(os.path.join(OUTPUT_DIR, "train_with_diff_vectors.csv"), index=False)
valid_out.to_csv(os.path.join(OUTPUT_DIR, "valid_with_diff_vectors.csv"), index=False)

print(f"\nAll outputs saved → {OUTPUT_DIR}")