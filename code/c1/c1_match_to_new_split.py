"""
extract_c1_for_holdout.py
=========================
Extracts C1 attribute vectors for the C0 holdout set by matching paths.
Filters to frontal views only, then merges C0 predictions with C1 attributes.
Output is the ready-to-use input for c2_cv_pipeline.py.

Adapted to create c2_data.csv for each backbone and pathology combination.
"""
import pandas as pd
import os

RESULTS_DIR = "/work3/s251710/thesis_results/"

# Define backbones and pathologies
BACKBONES = ["custom", "resnet50", "vit"]
BACKBONES = ["custom"]

PATHOLOGIES = ["effusion", "pneumothorax", "cardiomegaly"]
PATHOLOGIES = ["effusion"]

# Load C1 attributes (shared across all backbones)
c1_train = pd.read_csv(os.path.join(RESULTS_DIR, "C1_attributes/train_c1_attribute_vector_rad.csv"))
print(f"C1 train attributes: {len(c1_train):,}\n")

# Process each backbone and pathology combination
for backbone in BACKBONES:
    for pathology in PATHOLOGIES:
        print(f"Processing {backbone} - {pathology}...")
        
        # Load C0 predictions
        c0_path = os.path.join(RESULTS_DIR, f"C0_{backbone}/{pathology}/val_c0_{pathology}.csv")
        if not os.path.exists(c0_path):
            print(f"  ⚠ File not found: {c0_path}")
            continue
        
        c0_holdout = pd.read_csv(c0_path)
        print(f"  C0 holdout samples: {len(c0_holdout):,}")

        # Filter to frontals only
        c0_frontal = c0_holdout[c0_holdout["path"].str.contains("frontal")]
        print(f"  C0 frontal only:    {len(c0_frontal):,}")

        # Merge — inner join keeps only paths present in both
        merged = c0_frontal.merge(c1_train, on="path", how="inner")
        print(f"  Merged (C0 frontal ∩ C1): {len(merged):,}")

        # Sanity checks
        missing = set(c0_frontal["path"]) - set(c1_train["path"])
        if missing:
            print(f"  ⚠ WARNING: {len(missing):,} frontal paths have no C1 attributes")
        else:
            print(f"  ✓ All frontal paths matched successfully")

        print(f"  Correctness distribution: {merged['correct'].value_counts().to_dict()}")
        print(f"  Correct rate: {merged['correct'].mean():.3f}")

        # Save — this is the direct input to c2_cv_pipeline.py
        out_dir = os.path.join(RESULTS_DIR, f"C2_{backbone}/{pathology}")
        out_path = os.path.join(out_dir, "c2_data.csv")
        os.makedirs(out_dir, exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(f"  ✓ Saved to {out_path}\n")

print("All combinations processed!")