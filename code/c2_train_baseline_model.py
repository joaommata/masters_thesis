"""
c2_train_baseline_model.py

Trains a logistic regression (C2 baseline) to predict whether C0's prediction
for a target disease was correct or wrong, using only C0's output probability
as input feature.

This is the simplest possible quality control baseline:
"Can knowing how confident C0 was tell you if it was right?"

Inputs  : train_c0_{disease}.csv and valid_c0_{disease}.csv
Outputs : c2_baseline.pkl        (trained model)
          c2_baseline_scaler.pkl (fitted scaler)
          c2_baseline_cm.png
          c2_baseline_results.txt

Usage:
    python c2_train_baseline_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_DIR       = "/zhome/d0/a/221493/thesis"
TARGET_DISEASE = "Effusion"
DATA_DIR       = os.path.join(BASE_DIR, "results/C0_baseline")
OUTPUT_DIR     = os.path.join(BASE_DIR, "results/C2_baseline")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────────

def load_data(disease):
    tag      = disease.lower().replace(" ", "_")
    train_df = pd.read_csv(os.path.join(DATA_DIR, f"train_c0_{tag}.csv"))
    valid_df = pd.read_csv(os.path.join(DATA_DIR, f"valid_c0_{tag}.csv"))
    return train_df, valid_df


def get_features_and_target(df, prob_col):
    X = df[[prob_col]].values   # single feature: C0's probability for target disease
    y = df["correct"].values    # binary target: 1=correct, 0=wrong
    return X, y


# ── Train ──────────────────────────────────────────────────────────────────────

def train_c2(X_train, y_train):
    """
    Logistic regression with class_weight='balanced' to handle the
    imbalance between correct and wrong predictions.
    Without this, the model collapses to always predicting 'correct'.
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    return model, scaler


# ── Evaluate ───────────────────────────────────────────────────────────────────

def evaluate(model, scaler, X_valid, y_valid, output_dir):

    X_valid_scaled = scaler.transform(X_valid)
    y_pred         = model.predict(X_valid_scaled)
    y_pred_prob    = model.predict_proba(X_valid_scaled)[:, 1]

    auc    = roc_auc_score(y_valid, y_pred_prob)
    report = classification_report(
        y_valid, y_pred,
        target_names=["Wrong", "Correct"],
        zero_division=0
    )

    # ── Print ──────────────────────────────────────────────────────
    print(f"=== C2 Baseline Results {TARGET_DISEASE} ===\n")
    print(f"Feature      : C0 output probability (single disease)")
    print(f"Model        : Logistic Regression (class_weight=balanced)\n")
    print(report)
    print(f"AUC          : {auc:.3f}\n")
    print(f"Coefficients : {dict(zip(['prob'], model.coef_[0].tolist()))}")
    print(f"Intercept    : {model.intercept_[0]:.4f}")

    # ── Save text report ───────────────────────────────────────────
    results_path = os.path.join(output_dir, f"c2_baseline_results_{TARGET_DISEASE.lower().replace(' ', '_')}.txt")
    with open(results_path, "w") as f:
        f.write(f"=== C2 Baseline Results {TARGET_DISEASE} ===\n\n")
        f.write(f"Feature      : C0 output probability (single disease)\n")
        f.write(f"Model        : Logistic Regression (class_weight=balanced)\n\n")
        f.write(report)
        f.write(f"\nAUC          : {auc:.3f}\n")
        f.write(f"Coefficients : {dict(zip(['prob'], model.coef_[0].tolist()))}\n")
        f.write(f"Intercept    : {model.intercept_[0]:.4f}\n")
    print(f"Results saved → {results_path}")

    # ── ROC curve ──────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"C2 Baseline (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("C2 Baseline ROC\nCan C0 confidence predict correctness?")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(output_dir, f"c2_baseline_roc_{TARGET_DISEASE.lower().replace(' ', '_')}.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"ROC curve saved → {roc_path}")

    # ── Confusion matrix ───────────────────────────────────────────
    cm = confusion_matrix(y_valid, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Wrong", "Correct"]).plot()
    plt.title("C2 Baseline Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"c2_baseline_cm_{TARGET_DISEASE.lower().replace(' ', '_')}.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {cm_path}")

    return auc


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Target disease : {TARGET_DISEASE}")
    print(f"Data dir       : {DATA_DIR}")
    print(f"Output dir     : {OUTPUT_DIR}\n")

    # Load
    train_df, valid_df = load_data(TARGET_DISEASE)
    print(f"Train rows : {len(train_df)}")
    print(f"Valid rows : {len(valid_df)}")

    prob_col = "prob"  # column name saved by run_c0_baseline.py

    # Features and targets
    X_train, y_train = get_features_and_target(train_df, prob_col)
    X_valid, y_valid = get_features_and_target(valid_df, prob_col)

    print(f"\nClass balance — Train: {y_train.mean():.2f} | Valid: {y_valid.mean():.2f}")

    # Train
    c2, scaler = train_c2(X_train, y_train)

    # Evaluate
    evaluate(c2, scaler, X_valid, y_valid, OUTPUT_DIR)

    # Save model and scaler
    model_path = os.path.join(OUTPUT_DIR, f"c2_baseline_{TARGET_DISEASE.lower().replace(' ', '_')}.pkl")
    joblib.dump(c2, model_path)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"c2_baseline_scaler_{TARGET_DISEASE.lower().replace(' ', '_')}.pkl"))
    print(f"\nModel and scaler saved to {model_path}.")


if __name__ == "__main__":
    main()