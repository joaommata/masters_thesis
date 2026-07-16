"""
c1_match_to_new_split_rsna.py
=============================
Builds c2_data.csv for the RSNA Pneumonia binary task by merging the C0 val
predictions with the C1 attribute vectors on `path`.

Adapted from c1_match_to_new_split.py. Differences:
  - no frontal-view filter: RSNA is one frontal image per study, and the AP/PA
    view is already balanced in the split (and kept as a `view` column) rather
    than filtered out.
  - C1 comes from the val-split extraction (C1_rsna/pneumonia/val_c1_attributes.csv),
    not from a file built over a superset. So this is a plain 1:1 merge and the
    row count must be preserved exactly.
  - single backbone / single pathology, so no nested loop.

Output is the ready-to-use input for the C2 pipeline.
"""
import os

import pandas as pd

RESULTS_DIR = "/work3/s251710/thesis_results/"

C0_PATH = os.path.join(RESULTS_DIR, "C0_custom/rsna_pneumonia/val_c0_rsna_pneumonia.csv")
C1_PATH = os.path.join(RESULTS_DIR, "C1_rsna/pneumonia/val_c1_attributes.csv")
OUT_DIR = os.path.join(RESULTS_DIR, "C2_custom/rsna_pneumonia")
OUT_PATH = os.path.join(OUT_DIR, "c2_data.csv")

# ── Load ──────────────────────────────────────────────────────────────────────
c0 = pd.read_csv(C0_PATH)
print(f"C0 val predictions: {len(c0):,}")

c1 = pd.read_csv(C1_PATH)
print(f"C1 attributes:      {len(c1):,}")

# `true` is present in both; C1's copy is redundant. Drop it before merging so we
# don't end up with true_x/true_y (the C2 pipeline reads a bare `true`).
if "true" in c1.columns:
    c1 = c1.drop(columns=["true"])

# ── Merge ─────────────────────────────────────────────────────────────────────
merged = c0.merge(c1, on="path", how="inner")
print(f"Merged (C0 ∩ C1):   {len(merged):,}")

# ── Sanity checks ─────────────────────────────────────────────────────────────
# Unlike the CheXpert version, both sides cover exactly the same val split, so a
# 1:1 merge is expected. Any shortfall means one side is stale or mid-write.
missing = set(c0["path"]) - set(c1["path"])
if missing:
    print(f"  ⚠ WARNING: {len(missing):,} C0 paths have no C1 attributes")
if len(merged) != len(c0):
    raise SystemExit(
        f"ABORT: merge lost rows ({len(c0):,} C0 -> {len(merged):,} merged). "
        "Is one of the inputs stale or still being written?"
    )
if merged["path"].duplicated().any():
    raise SystemExit("ABORT: duplicate paths after merge")

print("  ✓ All C0 val paths matched 1:1")

print(f"\nCorrectness distribution: {merged['correct'].value_counts().to_dict()}")
print(f"Correct rate: {merged['correct'].mean():.3f}")
print(f"Error set for C2: {int((1 - merged['correct']).sum()):,} misclassified")

if "view" in merged.columns:
    print("\nAccuracy by view (should be similar -- view was balanced):")
    print(merged.groupby("view")["correct"].agg(["mean", "count"]).round(3).to_string())
if "label_name" in merged.columns:
    print("\nAccuracy by true class (hard negatives should be worst):")
    print(merged.groupby("label_name")["correct"].agg(["mean", "count"]).round(3).to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
merged.to_csv(OUT_PATH, index=False)
print(f"\n✓ Saved {len(merged):,} rows × {merged.shape[1]} cols to {OUT_PATH}")
