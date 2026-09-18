# c2_build_dataset.py
"""
Stage 4 of the final pipeline: join one disease's C0 predictions with the shared
C1 attribute table to produce the `c2_data.csv` that the C2 pipelines consume.

    stage 1  train C0            c0_train_multilabel_final.py   (once)
    stage 2  C0 predictions      c0_final_predictions.py        (once per disease)
    stage 3  C1 attributes       c1_build_attribute_vector_subset.py
                                 + c1_assemble_c2_attributes.py (ONCE, total)
    stage 4  build c2_data.csv   this script                    (once per disease)

Stage 3 runs only once for everything. C1 attributes are properties of the IMAGE
-- demographics, per-anatomy geometry, radiomics -- so they do not depend on
which disease C0 was asked about. The same 95,825-row table is reused by every
disease; only the C0 half (prob / pred / correct / embeddings) changes.

Output schema (1,486 columns), matching what the C2 pipelines expect:

    path, prob, true, cam_path, pred, correct, margin, patient_id,
    <454 attribute columns>, emb_0 .. emb_1023

That is the legacy `C2_custom/effusion/c2_data.csv` schema minus the 7 SDN
`prob_<layer>` columns, which were dropped from the project. Nothing downstream
needs them: `c2_feature_spec.C0_DERIVED_COLS` only ever gets SUBTRACTED from the
frame to form `meta_cols`, so absent names are a no-op.

The 454 attribute columns are asserted to be byte-identical, in the same order,
to the legacy table's attribute space, so the existing pipelines run unchanged.

Leakage guards
--------------
Four groups of columns must never reach the attribute space, which the pipelines
build by SUBTRACTION -- anything not named in `meta_cols` silently becomes a
feature. `c2_feature_spec.py` exists because that exact mistake shipped once.

  * `label_raw`  the raw 1/0/-1 CheXpert value for this disease. It IS the
                 target. Not in `meta_cols`, so it would land in the attribute
                 space and every attribute config would score near 1.0.
  * `certain`    same origin, same problem.
  * the 14 CheXpert observation columns on the attribute side -- one of them is
    this disease's ground truth, and the other 13 are radiologist labels for the
    same image, not image-derived features.
  * `Sex`, `Frontal/Lateral`, `AP/PA` -- strings, which crash StandardScaler the
    way `cam_path` did. Their numeric forms (`age_pred`, `sex_male`,
    `sex_female`) are already inside the 454 and are kept.

All four are dropped here AND asserted absent from the output.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FE = os.path.join(REPO, "final_experiment")
RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")

# Written by c0_final_predictions.py; the raw label and its certainty flag.
DROP_FROM_C0 = ["label_raw", "certain"]

# The split CSV's own columns, carried into the attribute table by the C1
# assembly step. `patient_id` is KEPT -- the C2 pipelines group folds on it.
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]
DROP_FROM_ATTR = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"] + CHEXPERT_LABELS

# Legacy table whose attribute space the output must reproduce exactly.
LEGACY_REFERENCE = os.path.join(RESULTS_DIR, "C2_custom", "effusion", "c2_data.csv")
N_ATTRS_EXPECTED = 454
N_EMB_EXPECTED = 1024


def main():
    p = argparse.ArgumentParser(
        description="Join C0 predictions with C1 attributes into c2_data.csv.")
    p.add_argument("--disease", required=True)
    p.add_argument("--split", default="C2_dataset",
                   help="which split to build (C2_dataset | Original_Test | "
                        "C0_checkpoint_selection)")
    p.add_argument("--attrs", default=os.path.join(
        FE, "results", "C1_attributes", "C2_dataset_attributes.csv"))
    p.add_argument("--out", default=None,
                   help="default: $THESIS_RESULTS/C2_final/<disease>/c2_data.csv")
    p.add_argument("--how", default="inner", choices=["inner", "left"])
    p.add_argument("--keep-uncertain", action="store_true",
                   help="keep rows whose C0 correctness is undefined (see below). "
                        "Off by default: they have no C2 target.")
    p.add_argument("--float32", action="store_true", default=True,
                   help="store the 1024 embeddings as float32 (default: on)")
    p.add_argument("--no-check-legacy", action="store_true",
                   help="skip the attribute-space comparison against the legacy table")
    args = p.parse_args()

    c0_dir = os.path.join(RESULTS_DIR, "C0_final", "multilabel_ignore", args.disease)
    c0_csv = os.path.join(c0_dir, f"{args.split}_c0_{args.disease}.csv")
    out = args.out or os.path.join(RESULTS_DIR, "C2_final", args.disease, "c2_data.csv")

    print(f"disease   : {args.disease}")
    print(f"split     : {args.split}")
    print(f"C0 side   : {c0_csv}")
    print(f"attributes: {args.attrs}")
    print(f"out       : {out}\n")

    if not os.path.exists(c0_csv):
        sys.exit(f"ERROR: {c0_csv} does not exist.\n"
                 f"Run stage 2 first:  python {FE}/code/c0_final_predictions.py "
                 f"--disease {args.disease}")

    # ── Attribute side ────────────────────────────────────────────────────────
    # Select columns up front so the split metadata never enters memory, let alone the merge.
    attr_header = pd.read_csv(args.attrs, nrows=0).columns.tolist()

    # Remove the ones we know shouldnt be in the attribute space   
    keep_attr = [c for c in attr_header if c not in DROP_FROM_ATTR]
    dropped = [c for c in attr_header if c in DROP_FROM_ATTR]
    print(f"attributes: {len(attr_header)} cols -> keeping {len(keep_attr)} "
          f"(dropped {len(dropped)}: split metadata + the 14 CheXpert labels)")

    # Whats left is the 454 attributes that the C2 pipelines expect, plus the "path" column to join on.
    attrs = pd.read_csv(args.attrs, usecols=["Path"] + keep_attr)
    attrs = attrs.rename(columns={"Path": "path"})
    print(f"            {len(attrs):,} rows")

    # ── C0 side ───────────────────────────────────────────────────────────────
    c0_header = pd.read_csv(c0_csv, nrows=0).columns.tolist()
    emb_cols = [c for c in c0_header if c.startswith("emb_")]
    
    # Again remove the columns we can't let into the attribute space, which is built by SUBTRACTION in the C2 pipelines.
    keep_c0 = [c for c in c0_header if c not in DROP_FROM_C0]
    dtype = {c: np.float32 for c in emb_cols} if args.float32 else None
    print(f"C0 side   : {len(c0_header)} cols -> keeping {len(keep_c0)} "
          f"(dropped {DROP_FROM_C0}: raw label + certainty flag = target leakage)")

    c0 = pd.read_csv(c0_csv, usecols=keep_c0, dtype=dtype)
    print(f"            {len(c0):,} rows, {len(emb_cols)} embedding dims\n")

    # ── Join C0 info with C1 info ───────────────────────────────────────────────────────────────────
    merged = c0.merge(attrs, on="path", how=args.how, validate="one_to_one")
    lost = len(c0) - len(merged)
    print(f"join      : {len(merged):,} rows ({args.how}); "
          f"{lost} C0 row(s) had no attributes")

    # ── Drop rows with no C2 target ───────────────────────────────────────────
    # An uncertain (-1) row has no ground truth, so c0_final_predictions.py sets
    # true/pred/correct to NaN for it -- `prob` still exists, correctness does not.

    
    n_undef = int(merged["correct"].isna().sum())
    if n_undef and not args.keep_uncertain:
        # Drop the rows with undefined correctness, which are the uncertain (-1) labels. 
        # Reset the index of the merged DataFrame after dropping these rows. 
        # Print out how many rows were dropped and the new total number of rows in the merged DataFrame. Convert the "correct" and "pred" columns to integer type.
        merged = merged[merged["correct"].notna()].reset_index(drop=True)
        print(f"          : dropped {n_undef:,} rows with undefined correctness "
              f"(uncertain -1 labels) -> {len(merged):,} rows")
        merged["correct"] = merged["correct"].astype(int)
        merged["pred"] = merged["pred"].astype(int)
    elif n_undef:
        print(f"          : WARNING keeping {n_undef:,} rows with NaN correct; "
              "the C2 pipelines will cast these to a garbage int label")

    # ── Column order: metadata, attributes, then the embedding block last ─────
    meta_order = [c for c in ["path", "prob", "true", "cam_path", "pred",
                              "correct", "margin", "patient_id"]
                  if c in merged.columns]
    attr_cols = [c for c in merged.columns
                 if c not in meta_order and not c.startswith("emb_")]
    merged = merged[meta_order + attr_cols + emb_cols]

    # ── Assertions (JUST MAKING SURE THAT THE DATA IS IN THE RIGHT FORMAT) ─────────────────────────────────────────────────────────────
    leaked = [c for c in DROP_FROM_C0 + DROP_FROM_ATTR if c in merged.columns]
    assert not leaked, f"leakage columns survived into the output: {leaked}"
    assert len(emb_cols) == N_EMB_EXPECTED, \
        f"expected {N_EMB_EXPECTED} embedding dims, got {len(emb_cols)}"
    assert len(attr_cols) == N_ATTRS_EXPECTED, \
        f"expected {N_ATTRS_EXPECTED} attribute columns, got {len(attr_cols)}"

    if not args.no_check_legacy and os.path.exists(LEGACY_REFERENCE):
        legacy = pd.read_csv(LEGACY_REFERENCE, nrows=0).columns.tolist()
        legacy_meta = ({"path", "prob", "true", "cam_path", "pred", "correct",
                        "margin", "patient_id"}
                       | {c for c in legacy if c.startswith("prob_")})
        legacy_attrs = [c for c in legacy if c not in legacy_meta
                        and not c.startswith(("emb_", "delta_", "cf_"))]
        same = attr_cols == legacy_attrs
        print(f"legacy    : attribute space matches C2_custom/effusion "
              f"({len(legacy_attrs)} cols, same order: {same})")
        assert same, ("attribute space diverged from the legacy table; the C2 "
                      "pipelines assume this layout")

    # Any non-numeric column left outside the metadata block would crash
    # StandardScaler downstream, so catch it here rather than mid-CV.
    non_numeric = [c for c in attr_cols
                   if not pd.api.types.is_numeric_dtype(merged[c])]
    assert not non_numeric, f"non-numeric attribute columns: {non_numeric[:10]}"

    # ── Write ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out), exist_ok=True)
    
    # Save the final csv called `c2_data.csv` to the specified output path. 
    merged.to_csv(out, index=False)
    size_gb = os.path.getsize(out) / 1e9

    print(f"\nwrote {out}  ({size_gb:.2f} GB)")
    print(f"  {len(merged):,} rows x {merged.shape[1]:,} cols "
          f"= {len(meta_order)} meta + {len(attr_cols)} attrs + {len(emb_cols)} emb")
    assert args.keep_uncertain or merged["correct"].notna().all()
    print(f"  positives (correct==1): {int(merged['correct'].sum()):,} "
          f"({merged['correct'].mean():.1%} of rows)")
    print(f"  patients: {merged['patient_id'].nunique():,}")
    print(f"\nRun the C2 CV with:  --backbone final --disease {args.disease}")


if __name__ == "__main__":
    main()