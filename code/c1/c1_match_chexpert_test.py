"""
c1_match_chexpert_test.py
=========================
Builds the C2 input frame for the OFFICIAL CheXpert test set (valid.csv, 202
frontal images), the external counterpart of what `c1_match_to_new_split.py`
does for C0's held-out half of train.csv.

    C0 predictions on the test set  x  C1 attributes for the test set
    -> C2_<backbone>/<disease>/c2_data_chexpert_test.csv

The C1 side already exists and needs no recomputation: `c1_build_attribute_vector.py`
built `C1_attributes/valid_c1_attribute_vector_rad.csv` from valid.csv in the same
pass that built the training attributes, with the same models, the same PSPNet
segmentation and the same firstorder+shape2D radiomics extractor. It carries all
202 frontal paths, matching valid.csv 1:1.

Why the column order is forced to match c2_data.csv exactly
-----------------------------------------------------------
`c2_cv_pipeline_cf_ensemble.py` derives the attribute space by subtraction from
the TRAIN frame's columns:

    relevant_cols = [c for c in train_df.columns if c not in meta_cols
                                                 and not c.startswith('emb_')]

and then indexes `test_clean[relevant_cols]`. So the external test frame must be
a column-for-column match of c2_data.csv or the standardised attribute vectors --
and hence every delta feature and every nearest-neighbour distance -- would be
built over misaligned columns. This script asserts that match rather than
trusting it.

Usage:
    python c1_match_chexpert_test.py --disease effusion
"""

import argparse
import os

import pandas as pd

RESULTS_DIR = "/work3/s251710/thesis_results"

BACKBONE_MAP = {"densenet": "C2_custom", "resnet50": "C2_resnet50", "vit": "C2_vit"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", type=str, default="effusion")
    ap.add_argument("--backbone", type=str, default="densenet",
                    choices=list(BACKBONE_MAP.keys()))
    ap.add_argument("--c1_file", type=str,
                    default="C1_attributes/valid_c1_attribute_vector_rad.csv",
                    help="C1 attributes for the official test split, relative to RESULTS_DIR")
    args = ap.parse_args()

    disease  = args.disease
    c2_dir   = os.path.join(RESULTS_DIR, BACKBONE_MAP[args.backbone], disease)
    ref_csv  = os.path.join(c2_dir, "c2_data.csv")
    c0_csv   = os.path.join(RESULTS_DIR, f"C0_custom/{disease}/chexperttest_c0_{disease}.csv")
    c1_csv   = os.path.join(RESULTS_DIR, args.c1_file)
    out_csv  = os.path.join(c2_dir, "c2_data_chexpert_test.csv")

    print(f"Disease:   {disease}  ({args.backbone})")
    print(f"C0 test:   {c0_csv}")
    print(f"C1 test:   {c1_csv}")
    print(f"Reference: {ref_csv}\n")

    if not os.path.exists(c0_csv):
        raise FileNotFoundError(
            f"{c0_csv} not found -- run code/c0/c0_predict_chexpert_test.py "
            f"--disease {disease} first.")

    c0 = pd.read_csv(c0_csv)
    c1 = pd.read_csv(c1_csv)
    print(f"C0 test rows: {len(c0):,}")
    print(f"C1 test rows: {len(c1):,}")

    # Both sides are already frontal-only; an inner join is a no-op if they agree.
    only_c0 = set(c0["path"]) - set(c1["path"])
    only_c1 = set(c1["path"]) - set(c0["path"])
    if only_c0:
        print(f"  ! {len(only_c0):,} C0 paths have no C1 attributes (dropped)")
    if only_c1:
        print(f"  ! {len(only_c1):,} C1 paths have no C0 prediction (dropped)")

    merged = c0.merge(c1, on="path", how="inner")
    print(f"Merged:       {len(merged):,}\n")

    # ── Force the reference schema ────────────────────────────────────────────
    ref_cols = list(pd.read_csv(ref_csv, nrows=0).columns)
    missing  = [c for c in ref_cols if c not in merged.columns]
    extra    = [c for c in merged.columns if c not in ref_cols]
    if missing:
        raise ValueError(
            f"{len(missing)} column(s) required by {os.path.basename(ref_csv)} are "
            f"absent from the merged test frame, e.g. {missing[:8]}. The C2 pipeline "
            f"indexes the test frame by the train frame's columns, so this must be "
            f"fixed before running it.")
    if extra:
        print(f"Dropping {len(extra)} column(s) not in the reference schema: {extra[:8]}")

    merged = merged[ref_cols]
    assert list(merged.columns) == ref_cols, "column order still differs from c2_data.csv"
    print(f"Schema matches {os.path.basename(ref_csv)}: {len(ref_cols)} columns, same order")

    # ── What C2 will actually see ─────────────────────────────────────────────
    n_err = int((merged["correct"] == 0).sum())
    print(f"\nC2 target on this set:")
    print(f"  correct=1 (C0 right):  {int((merged['correct']==1).sum()):,}")
    print(f"  correct=0 (C0 wrong):  {n_err:,}   <-- positive class for error detection")
    print(f"  accuracy:              {merged['correct'].mean():.4f}")
    print(f"  patients:              {merged['patient_id'].nunique():,}")

    nan_frac = merged.isna().mean().mean()
    worst = merged.isna().mean().sort_values(ascending=False)
    print(f"\nNaN rate overall: {nan_frac:.4%}")
    if worst.iloc[0] > 0:
        print("  worst columns (filled with 0 by the pipeline, as on train):")
        for c, v in worst[worst > 0].head(5).items():
            print(f"    {c:<50} {v:.2%}")

    merged.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(merged):,} rows x {merged.shape[1]} cols)")


if __name__ == "__main__":
    main()
