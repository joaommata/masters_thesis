"""
extract_c1_for_holdout.py
=========================
Extracts C1 attribute vectors for the C0 holdout set by matching paths.
Filters to frontal views only, then merges C0 predictions with C1 attributes.
Output is the ready-to-use input for c2_cv_pipeline.py.
"""
import pandas as pd
import os

RESULTS_DIR = "/zhome/d0/a/221493/thesis/results/"

# Load C0 holdout predictions
c0_holdout = pd.read_csv(os.path.join(RESULTS_DIR, "C0_custom/effusion/val_c0_effusion.csv"))
print(f"C0 holdout samples: {len(c0_holdout):,}")

# Filter to frontals only
c0_frontal = c0_holdout[c0_holdout["path"].str.contains("frontal")]
print(f"C0 frontal only:    {len(c0_frontal):,}")

# Load C1 attributes
c1_train = pd.read_csv(os.path.join(RESULTS_DIR, "C1_attributes/train_c1_attribute_vector_rad.csv"))
print(f"C1 train attributes: {len(c1_train):,}")

# Merge — inner join keeps only paths present in both
merged = c0_frontal.merge(c1_train, on="path", how="inner")
print(f"Merged (C0 frontal ∩ C1): {len(merged):,}")

# Sanity checks
missing = set(c0_frontal["path"]) - set(c1_train["path"])
if missing:
    print(f"WARNING: {len(missing):,} frontal paths have no C1 attributes")
else:
    print("All frontal paths matched successfully")

print(f"Correctness distribution:\n{merged['correct'].value_counts().to_dict()}")
print(f"Correct rate: {merged['correct'].mean():.3f}")

# Save — this is the direct input to c2_cv_pipeline.py
out_path = os.path.join(RESULTS_DIR, "C2_custom/c2_data.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
merged.to_csv(out_path, index=False)
print(f"Saved to {out_path}")