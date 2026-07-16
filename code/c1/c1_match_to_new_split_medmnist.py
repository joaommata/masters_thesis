"""
c1_match_to_new_split_medmnist.py
==================================
MedMNIST equivalent of c1_match_to_new_split.py.
Merges C0 test predictions with C1 test attributes to produce c2_data.csv.

Differences from the CheXpert version:
  - Uses test split (not val) — val is reserved for C0 epoch selection only
  - No frontal-view filter (MedMNIST has no lateral views)
  - Drops cam_path (string column not needed by C2) and the duplicate 'true'
    column that appears in both C0 and C1 outputs
"""
import pandas as pd
import os

RESULTS_DIR  = "/work3/s251710/thesis_results/"
PATHOLOGIES  = ["effusion"]

c1 = pd.read_csv(os.path.join(RESULTS_DIR, "C1_medmnist/effusion/test_c1_attributes.csv"))
print(f"C1 test attributes: {len(c1):,}")

# Drop duplicate 'true' — already present in C0 predictions
c1 = c1.drop(columns=["true"], errors="ignore")

for pathology in PATHOLOGIES:
    print(f"\nProcessing medmnist - {pathology}...")

    c0_path = os.path.join(RESULTS_DIR, f"C0_medmnist/{pathology}/test_c0_{pathology}.csv")
    if not os.path.exists(c0_path):
        print(f"  ⚠ File not found: {c0_path}")
        continue

    c0 = pd.read_csv(c0_path)
    c0 = c0.drop(columns=["cam_path"], errors="ignore")
    print(f"  C0 test samples: {len(c0):,}")

    merged = c0.merge(c1, on="path", how="inner")
    print(f"  Merged: {len(merged):,}")

    missing = set(c0["path"]) - set(c1["path"])
    if missing:
        print(f"  ⚠ WARNING: {len(missing):,} paths have no C1 attributes")
    else:
        print(f"  ✓ All paths matched")

    print(f"  Correctness distribution: {merged['correct'].value_counts().to_dict()}")
    print(f"  Correct rate: {merged['correct'].mean():.3f}")

    out_dir  = os.path.join(RESULTS_DIR, f"C2_medmnist/{pathology}")
    out_path = os.path.join(out_dir, "c2_data.csv")
    os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"  ✓ Saved to {out_path}  ({merged.shape[1]} cols)")

print("\nDone!")
