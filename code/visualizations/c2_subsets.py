import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
INPUT_DIR  = os.path.join(RESULTS_DIR, "C2_sim_cf/effusion/cv_results")
OUTPUT_DIR = os.path.join(INPUT_DIR, "dataset_size_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL  = 'MLP'
CONFIG = 'M6'

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_dataset_size(folder_name):
    match = re.search(r"sub(\d+)", folder_name)
    return int(match.group(1)) if match else None


def compute_mean_roc(fold_results):
    """Average ROC curve across folds. Reused from cv_analyse_results.py."""
    common_fpr = np.linspace(0, 1, 100)
    tpr_folds, aucs = [], []

    for fold in fold_results:
        fpr = np.array(fold['fpr'])
        tpr = np.array(fold['tpr'])
        # Interpolate this fold's TPR onto common FPR grid
        f = interpolate.interp1d(fpr, tpr, kind='linear',
                                 bounds_error=False, fill_value=(0, 1))
        tpr_folds.append(f(common_fpr))
        aucs.append(fold['auc'])

    tpr_array = np.array(tpr_folds)
    return {
        'fpr':      common_fpr,
        'tpr_mean': np.mean(tpr_array, axis=0),
        'tpr_std':  np.std(tpr_array, axis=0),
        'auc_mean': np.mean(aucs),
        'auc_std':  np.std(aucs),
    }

# ══════════════════════════════════════════════════════════════════════════════
# COLLECT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

results = []

for folder in os.listdir(INPUT_DIR):
    folder_path = os.path.join(INPUT_DIR, folder)

    if not os.path.isdir(folder_path):
        continue
    if "sub" not in folder:
        continue

    dataset_size = extract_dataset_size(folder)
    if dataset_size is None:
        continue

    # --- AUC summary ---
    summary_path = os.path.join(folder_path, "cv_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Missing summary: {folder}")
        continue

    df = pd.read_csv(summary_path)
    row = df[(df['model'] == MODEL) & (df['config'] == CONFIG)]
    if row.empty:
        print(f"No {MODEL} {CONFIG} row in {folder}")
        continue

    # --- ROC data ---
    json_path = os.path.join(folder_path, "cv_detailed.json")
    if not os.path.exists(json_path):
        print(f"Missing cv_detailed.json: {folder}")
        continue

    with open(json_path, 'r') as f:
        cv_detailed = json.load(f)

    fold_results = cv_detailed[MODEL][CONFIG]
    roc = compute_mean_roc(fold_results)

    results.append({
        "dataset_size": dataset_size,
        "auc_mean": float(row['mean_auc'].values[0]),
        "auc_std":  float(row['std_auc'].values[0]),
        "roc":      roc,
    })

results.sort(key=lambda x: x['dataset_size'])
print(f"Found {len(results)} subsampled folders")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: AUC vs Dataset Size
# ══════════════════════════════════════════════════════════════════════════════

sizes    = [r['dataset_size'] for r in results]
auc_mean = [r['auc_mean'] for r in results]
auc_std  = [r['auc_std']  for r in results]

plt.figure(figsize=(6, 6))
plt.plot(sizes, auc_mean, marker='o', color='steelblue')
plt.fill_between(sizes,
                 np.array(auc_mean) - np.array(auc_std),
                 np.array(auc_mean) + np.array(auc_std),
                 alpha=0.2, color='steelblue')
plt.xlabel("Dataset Size")
plt.ylabel("AUC")
plt.title(f"{MODEL} {CONFIG} — AUC vs Dataset Size")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "auc_vs_dataset_size.png"), dpi=300)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: ROC curves — one per dataset size, shades of blue
# ══════════════════════════════════════════════════════════════════════════════

# Generate shades from light to dark blue based on how many sizes we have
n = len(results)
blues = plt.cm.Blues(np.linspace(0.3, 0.9, n))  # 0.3 avoids too-light shades

plt.figure(figsize=(6, 6))

for i, r in enumerate(results):
    roc   = r['roc']
    label = f"n={r['dataset_size']:,}  AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}"
    color = blues[i]

    plt.plot(roc['fpr'], roc['tpr_mean'], color=color, lw=2, label=label)
    plt.fill_between(roc['fpr'],
                     roc['tpr_mean'] - roc['tpr_std'],
                     roc['tpr_mean'] + roc['tpr_std'],
                     color=color, alpha=0.15)

plt.plot([0, 1], [0, 1], 'k:', lw=0.8, alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"{MODEL} {CONFIG} — ROC by Dataset Size")
plt.legend(loc='lower right', fontsize=10, prop={'family': 'monospace'})
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_vs_dataset_size.png"), dpi=300)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'Dataset Size':<15} {'AUC Mean':<12} {'AUC Std'}")
print("-" * 40)
for r in results:
    print(f"{r['dataset_size']:<15} {r['auc_mean']:.4f}       {r['auc_std']:.4f}")

pd.DataFrame([{k: v for k, v in r.items() if k != 'roc'} for r in results]).to_csv(
    os.path.join(OUTPUT_DIR, "summary_dataset_size.csv"), index=False
)
print(f"\nOutputs saved → {OUTPUT_DIR}")