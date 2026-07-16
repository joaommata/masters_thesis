"""
c2_sweep_cf_count.py
====================
Trains M6 (prob + ΔA + attrs + cf_prob) across different cf_count values
and compares ROC curves and AUC scores.

Assumes the data preparation script has already been run for each cf_count,
producing:
    train_with_diff_vectors_{cf_count}.csv
    valid_with_diff_vectors_{cf_count}.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DISEASE    = "effusion"
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
INPUT_DIR  = os.path.join(RESULTS_DIR, f"C2_sim_cf/{DISEASE}")
OUTPUT_DIR = os.path.join(INPUT_DIR, "cf_count")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CF_COUNTS = [1, 3, 5]  # the cf_count values you want to compare

# ══════════════════════════════════════════════════════════════════════════════


def load_data(cf_count):
    """Load the pre-computed CSVs for a given cf_count."""
    train_df = pd.read_csv(os.path.join(INPUT_DIR, f"train_with_diff_vectors_{cf_count}.csv"))
    valid_df = pd.read_csv(os.path.join(INPUT_DIR, f"valid_with_diff_vectors_{cf_count}.csv"))
    return train_df, valid_df


def get_feature_matrix(df, disease, diff_cols, attr_cols):
    """Build the M6 feature matrix: prob + ΔA + attrs + cf_prob."""
    prob    = df[[f"{disease}_prob"]].values
    diff    = df[diff_cols].values
    attrs   = df[attr_cols].values
    cf_prob = df[['cf_prob']].values
    return np.hstack([prob, diff, attrs, cf_prob])


def train_and_eval(X_train, X_valid, y_train, y_valid):
    """Train logistic regression and return AUC, fpr, tpr."""
    scaler = StandardScaler()
    model  = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    model.fit(scaler.fit_transform(X_train), y_train)

    y_prob      = model.predict_proba(scaler.transform(X_valid))[:, 1]
    auc         = roc_auc_score(y_valid, y_prob)
    fpr, tpr, _ = roc_curve(y_valid, y_prob)

    return auc, fpr, tpr


# ── Main ──────────────────────────────────────────────────────────────────────

results = {}  # cf_count → {auc, fpr, tpr}

for cf_count in CF_COUNTS:
    print(f"\nTraining M6 with cf_count={cf_count}...")

    train_df, valid_df = load_data(cf_count)

    # Identify column groups (same logic as your main scripts)
    diff_cols = [c for c in train_df.columns if c.startswith('delta_')]
    emb_cols  = [c for c in train_df.columns if c.startswith('emb_')]
    meta_cols = {f"{DISEASE}_prob", f"{DISEASE}_pred", f"{DISEASE}_true",
                 'correct', 'path', 'patient_id', 'cf_prob'}
    attr_cols = [c for c in train_df.columns
                 if c not in meta_cols
                 and not c.startswith('delta_')
                 and not c.startswith('emb_')]

    y_train = train_df['correct'].values
    y_valid = valid_df['correct'].values

    X_train = get_feature_matrix(train_df, DISEASE, diff_cols, attr_cols)
    X_valid = get_feature_matrix(valid_df, DISEASE, diff_cols, attr_cols)

    auc, fpr, tpr = train_and_eval(X_train, X_valid, y_train, y_valid)
    results[cf_count] = {'auc': auc, 'fpr': fpr, 'tpr': tpr}

    print(f"  M6 cf_count={cf_count}  AUC={auc:.4f}")


# ── Plot ──────────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 7))
for cf_count, res in results.items():
    plt.plot(res['fpr'], res['tpr'], label=f"M6 cf_count={cf_count}  AUC={res['auc']:.3f}")
plt.plot([0, 1], [0, 1], 'k:', lw=1, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'M6 ROC — effect of cf_count ({DISEASE})')
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_cf_count_lr_comparison.png'), dpi=300)
plt.close()

# ── Summary table ─────────────────────────────────────────────────────────────

print("\ncf_count  |  AUC")
print("-" * 20)
for cf_count, res in results.items():
    print(f"  {cf_count:<8}  {res['auc']:.4f}")

with open(os.path.join(OUTPUT_DIR, 'summary_cf_count_lr_comparison.txt'), 'w') as f:
    f.write(f"M6 cf_count sweep — {DISEASE}\n\n")
    f.write("cf_count  |  AUC\n")
    f.write("-" * 20 + "\n")
    for cf_count, res in results.items():
        f.write(f"  {cf_count:<8}  {res['auc']:.4f}\n")

print(f"\nOutputs saved → {OUTPUT_DIR}")