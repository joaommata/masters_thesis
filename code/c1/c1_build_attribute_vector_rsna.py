# c1_build_attribute_vector_rsna.py
#
# Build C1 attribute vectors for the RSNA Pneumonia binary task
# (Lung Opacity vs Normal; see c0_rsna_split.py).
#
# Reuses FeatureVectorBuilder from c1_build_attribute_vector.py unchanged --
# only the dataset loading differs (RSNA DICOMs instead of CheXpert JPEGs).
#
# Radiomics: firstorder + shape2D only (no texture classes), matching
# c1_build_attribute_vector.py. Texture (glrlm/glszm) lives in
# c1_extend_radiomics.py and is deliberately not enabled here.
#
# Only the *val* split is needed: C2 consumes C0's held-out predictions, and the
# train split is used solely to fit C0. (--split train exists but nothing downstream
# reads it.) The CheXpert equivalent looks like it extracts "train" only because
# train_c1_attribute_vector_rad.csv is built over CheXpert's ORIGINAL train.csv,
# which contains both halves of the 50/50 C0 split; c1_match_to_new_split.py then
# inner-joins just the C0-val half back out of it.
#
# Usage:
#   python c1_build_attribute_vector_rsna.py --split val

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import torchxrayvision as xrv
from radiomics import featureextractor
from tqdm import tqdm

logging.getLogger('radiomics').setLevel(logging.ERROR)

sys.path.append(os.path.dirname(__file__))
from c1_build_attribute_vector import FeatureVectorBuilder

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

DATA_ROOT = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
SPLIT_DIR = os.path.join(DATA_ROOT, "rsna_pneumonia")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "C1_rsna", "pneumonia")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "Pneumonia"


# ── Dataset ───────────────────────────────────────────────────────────────────
class RSNADicomDataset(Dataset):
    """
    Reads the RSNA DICOMs listed in the C0 split CSVs and normalises them to the
    [-1024, 1024] range that torchxrayvision models expect.
    """
    def __init__(self, split_csv, data_root):
        self.df = pd.read_csv(split_csv)   # cols: StudyInstanceUID, Path, label_name, Pneumonia
        self.data_root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = pydicom.dcmread(os.path.join(self.data_root, row["Path"])).pixel_array

        # DICOM is higher bit depth than the 8-bit CheXpert JPEGs; rescale to
        # 0..255 first so both the xrv normalisation and the radiomics greylevel
        # binning see the same range they were tuned on.
        arr = arr.astype(np.float32)
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)

        # xrv models (and the PSPNet segmenter) expect 224x224
        img = xrv.datasets.XRayResizer(224)(arr[None, ...])[0]

        # xrv models expect (1, H, W) float32 in [-1024, 1024]
        img_xrv = xrv.datasets.normalize(img, maxval=255, reshape=True)
        img_tensor = torch.tensor(img_xrv, dtype=torch.float32)

        img_np = img.astype(np.float32)   # 0..255 float for radiomics

        return img_tensor, img_np, row["Path"], float(row[TARGET_COL])


# ── Build one split ───────────────────────────────────────────────────────────
def build_split(split, builder, batch_size, num_workers):
    split_csv = os.path.join(SPLIT_DIR, f"c0_{split}_split.csv")
    out_path = os.path.join(OUTPUT_DIR, f"{split}_c1_attributes.csv")

    print(f"\nProcessing {split} split  |  split CSV: {split_csv}")

    dataset = RSNADicomDataset(split_csv, DATA_ROOT)
    print(f"Dataset samples: {len(dataset):,}")

    # ── Resume support ────────────────────────────────────────────────────────
    all_vectors = []
    start_index = 0
    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path)
        if len(df_existing) > 0:
            done_paths = set(df_existing["path"].tolist())
            all_vectors = df_existing.to_dict("records")
            for i in range(len(dataset)):
                if dataset.df.iloc[i]["Path"] in done_paths:
                    start_index = i + 1
                else:
                    break
            print(f"[RESUME] {len(df_existing):,} samples already done, resuming from {start_index:,}")

    subset = torch.utils.data.Subset(dataset, list(range(start_index, len(dataset))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    samples_processed = start_index

    for img_tensors, img_nps_tensor, paths, labels in tqdm(loader, desc=split):
        try:
            img_nps = [img_nps_tensor[b].numpy() for b in range(img_tensors.shape[0])]
            vectors = builder.build_vectors_batch(img_tensors, img_nps, list(paths), plot=False)

            # attach true label from the split CSV
            for i, v in enumerate(vectors):
                v["true"] = float(labels[i])

            all_vectors.extend(vectors)

        except Exception as e:
            print(f"[SKIP] batch at {samples_processed}: {e}")
        finally:
            samples_processed += img_tensors.shape[0]

        if samples_processed % 500 < batch_size:
            pd.DataFrame(all_vectors).to_csv(out_path, index=False)
            print(f"[CHECKPOINT] {len(all_vectors):,} samples saved")

    df = pd.DataFrame(all_vectors)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} cols -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading torchxrayvision models...")
    models = {
        'age':  xrv.baseline_models.riken.AgeModel(),
        'sex':  xrv.baseline_models.mira.SexModel(),
        'race': xrv.baseline_models.emory_hiti.RaceModel(),
    }
    seg_model = xrv.baseline_models.chestx_det.PSPNet()

    extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape2D')

    builder = FeatureVectorBuilder(
        models=models,
        segmentation_model=seg_model,
        radiomics_extractor=extractor,
    )

    build_split(args.split, builder, args.batch, args.workers)
