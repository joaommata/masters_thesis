"""
c1_assemble_c2_attributes.py
============================
Join final_experiment/C2_dataset.csv with its C1 attribute vectors, pulling rows
from the March full-split extraction and from the subset top-up produced by
c1_build_attribute_vector_subset.py.

The output is deliberately written to the results dir, not into the repo's
final_experiment/ — it is ~700 MB, and final_experiment/ holds split definitions.
"""
import argparse
import os
import pandas as pd

REPO        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
C1_DIR      = os.path.join(RESULTS_DIR, "C1_attributes")

DEFAULT_SPLIT = os.path.join(REPO, "final_experiment", "C2_dataset.csv")
DEFAULT_PARTS = [os.path.join(C1_DIR, "train_c1_attribute_vector_rad.csv"),
                 os.path.join(C1_DIR, "final_experiment", "C2_dataset_c1_new.csv")]
DEFAULT_OUT   = os.path.join(C1_DIR, "final_experiment", "C2_dataset_attributes.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", default=DEFAULT_SPLIT)
    p.add_argument("--parts", nargs="+", default=DEFAULT_PARTS)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--how", default="inner", choices=["inner", "left"],
                   help="left keeps split rows with no attributes (all-NaN feature block)")
    p.add_argument("--link-into", default=os.path.join(REPO, "final_experiment", "results"),
                   help="directory to symlink the result into (\"\" to skip)")
    args = p.parse_args()

    # Loads the split, in this case the C2 dataset, and collects the unique paths that we want to keep from the C1 attribute vectors.
    split = pd.read_csv(args.split_csv)
    want = set(split["Path"])
    print(f"Split: {len(split):,} rows / {len(want):,} unique paths")

    frames, schema = [], None
    for f in args.parts:
        if not os.path.exists(f):
            print(f"  [WARN] missing, skipped: {f}")
            continue
        # Look at groups of 20,000 rows at a time, keeping only those rows whose "path" is in the set of paths we want. 
        # This is done to avoid loading the entire 1.2 GB March file into memory at once.
        kept = []
        for chunk in pd.read_csv(f, chunksize=20_000):
            kept.append(chunk[chunk["path"].isin(want)])
        part = pd.concat(kept, ignore_index=True)
        if schema is None:
            schema = list(part.columns)
        elif set(part.columns) != set(schema):
            # Determine if something is missing or extra in the schema of the current part compared to the first part's schema, and raise an error if there is a mismatch.
            missing = set(schema) - set(part.columns)
            extra = set(part.columns) - set(schema)
            raise SystemExit(
                f"Schema mismatch in {f}: {len(missing)} missing, {len(extra)} extra.\n"
                f"  missing e.g. {sorted(missing)[:5]}\n  extra e.g. {sorted(extra)[:5]}")
        part = part[schema]
        print(f"  {os.path.basename(f)}: {len(part):,} rows relevant to the split")
        frames.append(part)

    # Concatenate all the relevant parts into a single DataFrame, check for duplicate paths, and drop them if any are found. 
    # Then merge this DataFrame with the original split DataFrame on the "path" column, keeping only the rows that match based on the specified merge method (inner or left). 
    c1 = pd.concat(frames, ignore_index=True)
    dupes = c1["path"].duplicated().sum()
    if dupes:
        print(f"  [NOTE] dropping {dupes:,} duplicate path(s) across parts")
        c1 = c1.drop_duplicates(subset="path", keep="first")
    print(f"C1 rows assembled: {len(c1):,}")

    # patient_id is re-derived identically on both sides; keep the split's
    c1 = c1.drop(columns=["patient_id"], errors="ignore")

    # Then merge this DataFrame with the original split DataFrame on the "path" column, keeping only the rows that match based on the specified merge method (inner or left). 
    merged = split.merge(c1, left_on="Path", right_on="path", how=args.how).drop(columns=["path"])
    missing = want - set(c1["path"])
    print(f"\nMerged ({args.how}): {len(merged):,} rows x {merged.shape[1]} cols")
    if missing:
        print(f"  [WARN] {len(missing):,} split path(s) have no C1 attributes, e.g.:")
        for m in sorted(missing)[:5]:
            print(f"    {m}")
    else:
        print("  All split paths have attributes.")

    # Calculate the fraction of NaN values in the feature columns of the merged DataFrame, which indicates how many features were not found for certain structures by the segmenter. Print out the number of feature columns and the mean NaN fraction. 
    feat_cols = [c for c in merged.columns if c not in split.columns]
    nan_frac = merged[feat_cols].isna().mean().mean()
    print(f"  Feature block: {len(feat_cols)} cols, mean NaN fraction {nan_frac:.4f} "
          f"(NaN = segmenter found no mask for that structure)")
    
    # Finally, save the merged DataFrame to a CSV file and optionally create a symlink to it in a specified directory.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False)
    size_gb = os.path.getsize(args.out) / 1e9
    print(f"\nSaved -> {args.out} ({size_gb:.2f} GB)")

    # The bytes live on work3 with every other large artifact; final_experiment/
    # gets a symlink so the table sits next to the split CSVs that define it.
    # Same pattern the repo already uses for data/ and results/.
    if args.link_into:
        link = os.path.join(args.link_into, os.path.basename(args.out))
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(args.out, link)
        print(f"Linked  -> {link}")


if __name__ == "__main__":
    main()
