"""
c2_train_simulated_cf.py
========================
Trains C2 quality control classifier using simulated counterfactuals.

Instead of generating real counterfactuals (requires diffusion model), we simulate
them by finding the nearest training sample that C0 predicted with the OPPOSITE label.
The difference vector ΔA = A(sample_i) - A(fake_cf) approximates how far each sample
sits from C0's decision boundary in attribute space.

Hypothesis:
    Correct   → sample sits firmly in C0's territory → large ΔA
    Incorrect → sample is near C0's decision boundary → small ΔA

Inputs:
    - results/C0_baseline/{disease}/train_c0_{disease}.csv
    - results/C0_baseline/{disease}/valid_c0_{disease}.csv
    - results/C1_attributes/train_c1_attribute_vector.csv
    - results/C1_attributes/valid_c1_attribute_vector.csv

Outputs (saved to results/C2_sim_cf/{disease}/):
    - roc_comparison.png
    - feature_importances.png
    - delta_magnitude_distribution.png
    - mean_attribute_difference.png
    - results_summary.txt
    - train_with_diff_vectors.csv
    - valid_with_diff_vectors.csv
    - attr_scaler.pkl

Usage:
    python c2_train_simulated_cf.py
    Change DISEASE at the top to switch between diseases.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit this block only
# ══════════════════════════════════════════════════════════════════════════════

DISEASE   = 'cardiomegaly'   # effusion | pneumothorax | cardiomegaly | atelectasis
BASE_DIR  = '/zhome/d0/a/221493/thesis/'
DATA_DIR  = BASE_DIR + 'data/'

# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_DIR  = os.path.join(RESULTS_DIR, f'C2_sim_cf/{DISEASE}')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Disease-relevant attribute configuration ──────────────────────────────────

DISEASE_PREFIXES = {
    'effusion':     ["Left Lung", "Right Lung", "Facies Diaphragmatica",
                     "Left Hilus Pulmonis", "Right Hilus Pulmonis", "Mediastinum"],
    'pneumothorax': ["Left Lung", "Right Lung", "Mediastinum"],
    'cardiomegaly': ["Heart", "Aorta", "Mediastinum"],
    'atelectasis':  ["Left Lung", "Right Lung", "Facies Diaphragmatica",
                     "Left Hilus Pulmonis", "Right Hilus Pulmonis", "Mediastinum", "Heart"],
}

DISEASE_RATIO_COLS = {
    'effusion':     ['lung_area_ratio', 'lung_height_ratio', 'lung_width_ratio',
                     'left_lung_fraction', 'right_lung_fraction', 'mediastinal_ratio'],
    'pneumothorax': ['lung_area_ratio', 'lung_height_ratio', 'lung_width_ratio',
                     'left_lung_fraction', 'right_lung_fraction', 'mediastinal_ratio'],
    'cardiomegaly': ['cardiothoracic_ratio', 'mediastinal_ratio'],
    'atelectasis':  ['lung_area_ratio', 'lung_height_ratio', 'lung_width_ratio',
                     'left_lung_fraction', 'right_lung_fraction', 'mediastinal_ratio'],
}


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


# ── Difference vector computation ─────────────────────────────────────────────

def compute_diff_vectors(df, nn_pred1, nn_pred0, train_pred1, train_pred0, relevant_columns):
    """
    For each sample, find the nearest training sample that C0 predicted with the
    OPPOSITE label and return (query - neighbour) in scaled attribute space.

    KEY CHOICE: We use C0's PREDICTION (pred), NOT the ground truth (true).
    Why? Because we are probing C0's behaviour, not the disease severity.
    A real counterfactual flips C0's decision — so our fake CF must too.
    """
    query       = df[relevant_columns].values
    query_preds = df['pred'].values.astype(int)
    diff_vecs   = np.empty((len(df), len(relevant_columns)), dtype=np.float64)

    mask1 = query_preds == 1
    mask0 = ~mask1

    if mask1.any():
        _, idxs    = nn_pred0.kneighbors(query[mask1])
        neighbours = train_pred0[relevant_columns].values[idxs[:, 0]]
        diff_vecs[mask1] = query[mask1] - neighbours

    if mask0.any():
        _, idxs    = nn_pred1.kneighbors(query[mask0])
        neighbours = train_pred1[relevant_columns].values[idxs[:, 0]]
        diff_vecs[mask0] = query[mask0] - neighbours

    return diff_vecs


# ── Model training and evaluation ─────────────────────────────────────────────

def train_and_eval_lr(X_train, X_valid, y_train, y_valid, label):
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_valid)
    model   = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_tr_sc, y_train)
    y_prob  = model.predict_proba(X_va_sc)[:, 1]
    y_pred  = model.predict(X_va_sc)
    auc_score = roc_auc_score(y_valid, y_prob)
    print(f"LR — {label:<45} AUC = {auc_score:.4f}")
    print(classification_report(y_valid, y_pred, target_names=['Incorrect', 'Correct']))
    # rerutn fpr, tpr for ROC plotting
    fpr, tpr, _ = roc_curve(y_valid, y_prob)
    return auc_score, fpr, tpr


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Disease    : {DISEASE}")
    print(f"Output dir : {OUTPUT_DIR}\n")

    # ── Load data ─────────────────────────────────────────────────────────────

    train_c0 = pd.read_csv(os.path.join(RESULTS_DIR, f'C0_baseline/{DISEASE}/train_c0_{DISEASE}.csv'))
    valid_c0 = pd.read_csv(os.path.join(RESULTS_DIR, f'C0_baseline/{DISEASE}/valid_c0_{DISEASE}.csv'))
    c1_train = pd.read_csv(os.path.join(RESULTS_DIR, 'C1_attributes/train_c1_attribute_vector.csv'))
    c1_valid = pd.read_csv(os.path.join(RESULTS_DIR, 'C1_attributes/valid_c1_attribute_vector.csv'))

    print(f"C0 — train: {len(train_c0):,}  valid: {len(valid_c0):,}")
    print(f"C1 — train: {len(c1_train):,}  valid: {len(c1_valid):,}")

    y_true_c0 = valid_c0['true']
    y_pred_c0 = valid_c0['pred']
    y_prob_c0 = valid_c0['prob']

    fpr_c0, tpr_c0, _ = roc_curve(y_true_c0, y_prob_c0)
    auc_c0 = roc_auc_score(y_true_c0, y_prob_c0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true_c0, y_pred_c0),
                           display_labels=['No Disease', 'Disease']).plot(cmap='Blues', ax=axes[0])
    axes[0].set_title(f'C0 Confusion Matrix — {DISEASE}')
    axes[1].plot(fpr_c0, tpr_c0, color='blue', label=f'ROC curve (AUC = {auc_c0:.2f})')
    axes[1].plot([0, 1], [0, 1], color='red', linestyle='--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'C0 ROC Curve — {DISEASE}')
    axes[1].legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'c0_performance.png'), dpi=150)
    plt.close()
    
    # ── Merge and add clinical ratios ─────────────────────────────────────────

    train_c2 = train_c0.merge(c1_train, on='path', how='left')
    valid_c2 = valid_c0.merge(c1_valid, on='path', how='left')

    train_c2 = add_clinical_ratios(train_c2)
    valid_c2 = add_clinical_ratios(valid_c2)

    # ── Select relevant columns ───────────────────────────────────────────────

    pixel_cols = [
        col for col in train_c2.columns
        if any(col.lower().startswith(p.lower()) for p in DISEASE_PREFIXES[DISEASE])
    ]
    relevant_columns = DISEASE_RATIO_COLS[DISEASE] + pixel_cols

    cols_to_keep = ["path", "prob", "true", "pred", "correct", "patient_id"] + relevant_columns

    train_filtered = train_c2[cols_to_keep].dropna(subset=relevant_columns).copy()
    valid_filtered = valid_c2[cols_to_keep].dropna(subset=relevant_columns).copy()

    # ── Scale relevant attributes (fit on train only) ─────────────────────────

    attr_scaler = StandardScaler()
    train_filtered[relevant_columns] = attr_scaler.fit_transform(
        train_filtered[relevant_columns].values.astype(float))
    valid_filtered[relevant_columns] = attr_scaler.transform(
        valid_filtered[relevant_columns].values.astype(float))

    print(f"\nRatio columns  : {DISEASE_RATIO_COLS[DISEASE]}")
    print(f"Pixel columns  : {pixel_cols}")
    print(f"Total features : {len(relevant_columns)}")
    print(f"Shape — train  : {train_filtered.shape}  valid: {valid_filtered.shape}")

    y_train = train_filtered['correct'].values
    y_valid = valid_filtered['correct'].values

    print(f"\nTrain — correct: {y_train.sum():,} ({y_train.mean():.1%})  incorrect: {(1-y_train).sum():,} ({(1-y_train).mean():.1%})")
    print(f"Valid  — correct: {y_valid.sum():,} ({y_valid.mean():.1%})  incorrect: {(1-y_valid).sum():,} ({(1-y_valid).mean():.1%})")

    # ── Plot mean attribute difference between classes ─────────────────────────

    disease_positive_mean = train_filtered[train_filtered['true'] == 1][relevant_columns].mean()
    disease_negative_mean = train_filtered[train_filtered['true'] == 0][relevant_columns].mean()
    difference = disease_positive_mean - disease_negative_mean

    plt.figure(figsize=(12, 6))
    sns.barplot(x=difference.index, y=difference.values)
    plt.xticks(rotation=90)
    plt.title(f'Difference in Mean Attribute Values Between Positive and Negative Classes - {DISEASE}')
    plt.xlabel('Attribute')
    plt.ylabel('Difference in Mean Value (Positive - Negative)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mean_attribute_difference.png'), dpi=150)
    plt.close()

    # ── Build NN indices on scaled train vectors ──────────────────────────────

    train_pred1 = train_filtered[train_filtered['pred'] == 1]
    train_pred0 = train_filtered[train_filtered['pred'] == 0]

    print(f"\nTraining pool — pred=1: {len(train_pred1):,}  pred=0: {len(train_pred0):,}")

    # Euclidean distance: penalises large deviations in a single attribute heavily,
    # more clinically meaningful than Manhattan for ratio-based features
    nn_pred1 = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
    nn_pred1.fit(train_pred1[relevant_columns].values)

    nn_pred0 = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
    nn_pred0.fit(train_pred0[relevant_columns].values)

    # ── Compute difference vectors ────────────────────────────────────────────

    print("\nComputing ΔA difference vectors...")
    train_diff = compute_diff_vectors(train_filtered, nn_pred1, nn_pred0,
                                      train_pred1, train_pred0, relevant_columns)
    valid_diff = compute_diff_vectors(valid_filtered, nn_pred1, nn_pred0,
                                      train_pred1, train_pred0, relevant_columns)

    diff_col_names = [f"delta_{c}" for c in relevant_columns]
    print(f"ΔA shape — train: {train_diff.shape}  valid: {valid_diff.shape}")

    # ── Sanity check ──────────────────────────────────────────────────────────

    train_dist   = np.linalg.norm(train_diff, axis=1)
    correct_mask = train_filtered['correct'].values == 1

    print(f"\nMean ΔA magnitude (L2 norm):")
    print(f"  Correct   : {train_dist[correct_mask].mean():.4f}")
    print(f"  Incorrect : {train_dist[~correct_mask].mean():.4f}")
    print(f"  Δ         : {train_dist[correct_mask].mean() - train_dist[~correct_mask].mean():+.4f}")
    print(f"\n  If Correct > Incorrect → the signal is working as hypothesised.")

    plt.figure(figsize=(8, 5))
    sns.kdeplot(train_dist[correct_mask],  label='Correct',   fill=True, alpha=0.4)
    sns.kdeplot(train_dist[~correct_mask], label='Incorrect', fill=True, alpha=0.4)
    plt.xlabel('ΔA magnitude (L2 norm)')
    plt.ylabel('Density')
    plt.title(f'Distance to Fake Counterfactual — Correct vs Incorrect\n{DISEASE}')
    plt.legend()
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'delta_magnitude_distribution.png'), dpi=150)
    plt.close()

    # Mean ΔA per attribute — correct vs incorrect predictions
    delta_df = pd.DataFrame(train_diff, columns=diff_col_names)
    delta_df['correct'] = train_filtered['correct'].values

    delta_correct   = delta_df[delta_df['correct'] == 1].drop(columns='correct').mean()
    delta_incorrect = delta_df[delta_df['correct'] == 0].drop(columns='correct').mean()
    delta_diff      = delta_correct - delta_incorrect

    plt.figure(figsize=(12, 6))
    sns.barplot(x=delta_diff.index, y=delta_diff.values)
    plt.xticks(rotation=90)
    plt.title(f'ΔA Difference — Correct vs Incorrect Predictions\n{DISEASE}')
    plt.xlabel('Attribute')
    plt.ylabel('Mean ΔA (Correct) - Mean ΔA (Incorrect)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'delta_correct_vs_incorrect.png'), dpi=150)
    plt.close()
    
    # ── Train and evaluate C2 models ──────────────────────────────────────────

    print("\n--- Logistic Regression ---")
    auc_base, fpr_base, tpr_base = train_and_eval_lr(
        train_filtered[['prob']].values, valid_filtered[['prob']].values,
        y_train, y_valid, "Baseline — C0 prob only")

    auc_diff, fpr_diff, tpr_diff = train_and_eval_lr(
        train_diff, valid_diff,
        y_train, y_valid, "ΔA only — fake CF difference vector")

    auc_comb, fpr_comb, tpr_comb = train_and_eval_lr(
        np.hstack([train_filtered[['prob']].values, train_diff]),
        np.hstack([valid_filtered[['prob']].values, valid_diff]),
        y_train, y_valid, "Combined — C0 prob + ΔA")


    # ── ROC comparison plot ───────────────────────────────────────────────────

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_base, tpr_base, color='gray',       lw=2, ls='--', label=f'Baseline — prob only  AUC={auc_base:.3f}')
    plt.plot(fpr_diff, tpr_diff, color='darkorange',  lw=2,          label=f'C2 — ΔA only          AUC={auc_diff:.3f}')
    plt.plot(fpr_comb, tpr_comb, color='steelblue',   lw=2,          label=f'C2 — prob + ΔA        AUC={auc_comb:.3f}')
    plt.plot([0, 1], [0, 1], 'k:', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'C2 Quality Control — {DISEASE}\nSimulated Counterfactual vs Baseline')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_comparison.png'), dpi=150)
    plt.close()
    

    # ── Summary table ─────────────────────────────────────────────────────────

    summary = (
        f"=== C2 Simulated Counterfactual Results — {DISEASE} ===\n\n"
        f"Train size : {len(train_filtered):,}  (correct: {y_train.sum():,} | incorrect: {(1-y_train).sum():,})\n"
        f"Valid size : {len(valid_filtered):,}  (correct: {y_valid.sum():,} | incorrect: {(1-y_valid).sum():,})\n"
        f"Features   : {len(relevant_columns)}  {relevant_columns}\n\n"
        f"ΔA magnitude — Correct: {train_dist[correct_mask].mean():.4f}  Incorrect: {train_dist[~correct_mask].mean():.4f}  Δ: {train_dist[correct_mask].mean() - train_dist[~correct_mask].mean():+.4f}\n\n"
        f"| Model | Baseline (prob only) | ΔA only | Combined (prob + ΔA) |\n"
        f"|-------|---------------------|---------|---------------------|\n"
        f"| LR    | {auc_base:.3f}               | {auc_diff:.3f}   | {auc_comb:.3f}               |\n"
    )
    print("\n" + summary)

    with open(os.path.join(OUTPUT_DIR, 'results_summary.txt'), 'w') as f:
        f.write(summary)

    # ── Save artefacts ────────────────────────────────────────────────────────

    joblib.dump(attr_scaler,   os.path.join(OUTPUT_DIR, 'attr_scaler.pkl'))

    train_out = train_filtered.copy()
    valid_out = valid_filtered.copy()
    for i, col in enumerate(diff_col_names):
        train_out[col] = train_diff[:, i]
        valid_out[col] = valid_diff[:, i]

    train_out.to_csv(os.path.join(OUTPUT_DIR, 'train_with_diff_vectors.csv'), index=False)
    valid_out.to_csv(os.path.join(OUTPUT_DIR, 'valid_with_diff_vectors.csv'), index=False)

    print(f"\nAll outputs saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()