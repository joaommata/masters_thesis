"""
c1_build_attribute_vector_subset.py
===================================
Extract C1 attribute vectors for an explicit set of images, instead of walking
all of train.csv in dataset order like c1_build_attribute_vector.py does.

We already had a full script to build all 455 attributes for all 191,010 AP/PA frontals in CheXpert train+valid, but the March run was killed by LSF after 163,504 rows. 
This script builds the missing 27,506 rows, appending them to the existing C1 CSVs in a resume-safe way. 
It is used to build the final_experiment/C2_dataset.csv subset of 13,830 rows.

Preprocessing is delegated to xrv.datasets.CheX_Dataset on a filtered copy of the
split CSV, so the pixels reaching the models are identical to the March run.

Usage:
    python c1_build_attribute_vector_subset.py                  # C2_dataset, resume-safe
"""
import argparse
import logging
import os
import signal
import sys

import numpy as np
import pandas as pd
import torch
from radiomics import featureextractor
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchxrayvision as xrv

# FeatureVectorBuilder stays in ../../code/c1/: it is the single canonical
# attribute builder, shared with four other c1 scripts. Copying it in here
# would fork the feature schema, so put its directory on the path instead.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "code", "c1"))
from c1_build_attribute_vector import FeatureVectorBuilder

logging.getLogger('radiomics').setLevel(logging.ERROR)

REPO        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
C1_DIR      = os.path.join(RESULTS_DIR, "C1_attributes")

DEFAULT_SPLIT  = os.path.join(REPO, "final_experiment", "C2_dataset.csv")
DEFAULT_SCHEMA = os.path.join(C1_DIR, "train_c1_attribute_vector_rad.csv")
DEFAULT_HAVE   = [os.path.join(C1_DIR, "train_c1_attribute_vector_rad.csv"),
                  os.path.join(C1_DIR, "valid_c1_attribute_vector_rad.csv")]
DEFAULT_OUT    = os.path.join(C1_DIR, "final_experiment", "C2_dataset_c1_new.csv")


# Read a single column without the cost of reading all - this way we can read the path to avoid re-extracting images that are already done.
def read_paths(csv_path, column):
    """Read one column of a possibly huge CSV without loading 455 float columns."""
    return set(pd.read_csv(csv_path, usecols=[column])[column].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", default=DEFAULT_SPLIT,
                   help="CheXpert-format CSV whose Path column defines the target set")
    p.add_argument("--have", nargs="*", default=DEFAULT_HAVE,
                   help="existing C1 CSVs whose paths are already done")
    p.add_argument("--schema-csv", default=DEFAULT_SCHEMA,
                   help="C1 CSV whose header pins the output column order")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--flush-every", type=int, default=512,
                   help="append to --out after this many new rows")
    p.add_argument("--limit", type=int, default=None, help="stop after N images (smoke test)")
    p.add_argument("--report-remaining", action="store_true",
                   help="print how many images are still missing, then exit without extracting")
    args = p.parse_args()

    # The schema is pinned to a specific CSV so that the output column order is stable and reproducible, even if the builder code changes.
    schema = list(pd.read_csv(args.schema_csv, nrows=0).columns)
    print(f"Schema pinned to {args.schema_csv} ({len(schema)} columns)")

    split = pd.read_csv(args.split_csv)
    # CheX_Dataset(views=["PA","AP"]) drops rows that are Frontal/Lateral == Frontal
    # but AP/PA in {LL, RL} (16 LL + 1 RL across CheXpert train, 10 of them in
    # C2_dataset). Excluding them here rather than downstream keeps the remaining
    # count exact, which the job script's resubmit decision depends on
    usable = split[split["AP/PA"].isin(["AP", "PA"])]
    n_unusable = len(split) - len(usable)
    want = set(usable["Path"])
    print(f"Target split: {args.split_csv} ({len(split):,} rows)")
    if n_unusable:
        print(f"  {n_unusable} row(s) are not AP/PA and cannot be extracted by the view filter")
    print(f"  extractable: {len(want):,} unique paths")

    have = set()
    for f in args.have:
        if os.path.exists(f):
            # Read the paths from the existing CSV file and count which ones I already have, so I don't re-extract them
            got = read_paths(f, "path")
            have |= got
            print(f"  already extracted in {os.path.basename(f)}: {len(got & want):,} of target")
        else:
            print(f"  [WARN] missing, ignored: {f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        done = read_paths(args.out, "path")
        have |= done
        print(f"  [RESUME] {len(done & want):,} of target already in {os.path.basename(args.out)}")

    # Determine which images are still missing and need to be extracted
    todo = want - have
    print(f"To extract: {len(todo):,}")
    if args.report_remaining:
        print(f"REMAINING={len(todo)}")
        return
    if not todo:
        print("Nothing to do.")
        return

    # CheX_Dataset keeps only views=["PA","AP"]; a handful of CheXpert rows are
    # Frontal/Lateral == Frontal but AP/PA in {LL, RL} and get dropped here.
    # Its patientid parsing also branches on 'train'/'valid' being in the csv
    # filename, so the temp file has to be named accordingly.
    
    # Put the paths of the images we still need to extract into a temporary CSV file
    todo_csv = os.path.join(os.path.dirname(args.out), "_todo_train.csv")
    split[split["Path"].isin(todo)].to_csv(todo_csv, index=False)
    print(f"  Created temporary CSV: {os.path.basename(todo_csv)}")
    # Build a dataset that will have only the images we still need to extract.
    dataset = xrv.datasets.CheX_Dataset(
        imgpath=os.path.join(DATA_ROOT, "CheXpert-v1.0-small"),
        csvpath=todo_csv,
        views=["PA", "AP"],
        transform=xrv.datasets.XRayResizer(224),
        unique_patients=False,
    )
    # Test
    if len(dataset) != len(todo):
        print(f"  [WARN] dataset has {len(dataset):,} samples but {len(todo):,} were requested")
    print(f"Dataset samples: {len(dataset):,}")

    if args.limit:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.limit, len(dataset)))))
        print(f"  [SMOKE] limited to {len(dataset):,}")

    print("Loading models...")
    # Load the xrv pretrained models
    models = {
        'age':  xrv.baseline_models.riken.AgeModel(),
        'sex':  xrv.baseline_models.mira.SexModel(),
        'race': xrv.baseline_models.emory_hiti.RaceModel(),
    }
    
    # Initialize the radiomics feature extractor to compute first-order and 2D shape features.
    extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)
    # We disable all features then activate only the ones we want to compute
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape2D')
    
    # Create a FeatureVectorBuilder instance that will handle the extraction of features from the images using the loaded models and the radiomics extractor.
    # FeatureVectorBuilder is a custom class. It takes in the models, a segmentation model, and a radiomics extractor to build feature vectors for the images. 
    # It ouptuts a dictionary of features for each image, which will be later converted to a DataFrame and saved to a CSV file.
    builder = FeatureVectorBuilder(
        models=models,
        segmentation_model=xrv.baseline_models.chestx_det.PSPNet(),
        radiomics_extractor=extractor,
    )

    # Base is the original CSV that the dataset was built from, which is used to look up the image paths for the indices in the dataset.
    base_csv = dataset.dataset.csv if isinstance(dataset, torch.utils.data.Subset) else dataset.csv
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        prefetch_factor=2 if args.num_workers > 0 else None)

    buffer, written, unseen_cols = [], 0, set()
    stopping = {"now": False}

    def on_term(signum, _frame):
        # Only raise a flag here — flushing from inside a signal handler while
        # pandas may already be writing is asking for a truncated CSV.
        print(f"\n[SIGNAL] caught {signal.Signals(signum).name}; "
              f"finishing current batch, flushing, then exiting.", flush=True)
        stopping["now"] = True

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def flush():
        nonlocal buffer, written
        if not buffer:
            return
        df = pd.DataFrame(buffer)
        extra = set(df.columns) - set(schema)
        if extra:
            unseen_cols.update(extra)
        # reindex drops anything off-schema and fills absent structures with NaN,
        # exactly as the old file represents a mask the segmenter did not find
        df = df.reindex(columns=schema)
        header = not os.path.exists(args.out)
        df.to_csv(args.out, mode="a", header=header, index=False)
        written += len(df)
        buffer = []
        print(f"[INFO] appended {len(df)} rows ({written:,} this run) -> {args.out}", flush=True)

    for batch in tqdm(loader, desc="C1 subset"):
        try:
            img_tensors = batch['img'].float()
            idxs = batch['idx'].tolist()
            img_paths = [base_csv['Path'].iloc[i] for i in idxs]
            img_nps = [img_tensors[b, 0].numpy() for b in range(img_tensors.shape[0])]
            buffer.extend(builder.build_vectors_batch(img_tensors, img_nps, img_paths, plot=False))
        except Exception as e:
            print(f"[SKIP] batch failed: {e}", flush=True)
        if len(buffer) >= args.flush_every or stopping["now"]:
            flush()
        if stopping["now"]:
            print("[SIGNAL] stopped cleanly; rerun to resume from here.", flush=True)
            break
    flush()

    if unseen_cols:
        print(f"\n[WARN] {len(unseen_cols)} column(s) produced but absent from the pinned schema "
              f"and therefore DROPPED: {sorted(unseen_cols)[:10]}")
    print(f"\nDone. {written:,} new rows in {args.out}")


if __name__ == "__main__":
    main()
