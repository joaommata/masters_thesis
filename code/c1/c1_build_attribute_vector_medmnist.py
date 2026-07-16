# c1_build_attribute_vector_medmnist.py
#
# Build C1 attribute vectors for ChestMNIST images.
# Reuses FeatureVectorBuilder from c1_build_attribute_vector.py unchanged —
# only the dataset loading is different (PNGs saved by c0_save_medmnist_images.py
# instead of CheXpert file paths).
#
# Usage:
#   python c1_build_attribute_vector_medmnist.py --disease effusion

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchxrayvision as xrv
from radiomics import featureextractor
from tqdm import tqdm
import logging
import cv2

logging.getLogger('radiomics').setLevel(logging.ERROR)

sys.path.append(os.path.dirname(__file__))
from c1_build_attribute_vector import FeatureVectorBuilder

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--disease", type=str, default="effusion")
parser.add_argument("--batch",   type=int, default=16)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

DISEASE    = args.disease
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
IMAGE_DIR  = os.path.join(DATA_ROOT, "medmnist", "images")
C0_DIR     = os.path.join(RESULTS_DIR, "", "C0_medmnist", DISEASE)
OUTPUT_DIR = os.path.join(RESULTS_DIR, "", "C1_medmnist", DISEASE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
class MedMNISTImageDataset(Dataset):
    """
    Loads PNGs saved by c0_save_medmnist_images.py and normalises them to the
    [-1024, 1024] range that torchxrayvision models expect.
    """
    def __init__(self, c0_csv_path, image_dir):
        self.df = pd.read_csv(c0_csv_path)   # columns: path, prob, true, pred, correct ...
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        # path column is like "train/000042"
        fpath = os.path.join(self.image_dir, row["path"] + ".png")

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)   # (224, 224) uint8

        # xrv models expect (1, H, W) float32 in [-1024, 1024]
        img_xrv = xrv.datasets.normalize(img, maxval=255, reshape=True)  # (1, 224, 224)
        img_tensor = torch.tensor(img_xrv, dtype=torch.float32)

        img_np = img.astype(np.float32)   # raw uint8→float for radiomics

        return img_tensor, img_np, row["path"], float(row["true"])


# ── Build one split ───────────────────────────────────────────────────────────
def build_split(split, builder, batch_size, num_workers):
    # split is the image split ("test"), c0_csv always reads from test_c0_*.csv
    c0_csv   = os.path.join(C0_DIR, f"{split}_c0_{DISEASE}.csv")
    out_path = os.path.join(OUTPUT_DIR, f"{split}_c1_attributes.csv")

    print(f"\nProcessing {split} split  |  C0 CSV: {c0_csv}")

    dataset = MedMNISTImageDataset(c0_csv, IMAGE_DIR)

    # ── Resume support ────────────────────────────────────────────────────────
    all_vectors = []
    start_index = 0
    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path)
        if len(df_existing) > 0:
            done_paths  = set(df_existing["path"].tolist())
            all_vectors = df_existing.to_dict("records")
            for i in range(len(dataset)):
                if dataset.df.iloc[i]["path"] in done_paths:
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

            # attach true label from C0 CSV
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
    print(f"Saved {len(df):,} rows × {df.shape[1]} cols → {out_path}")


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

    # Run on test split only. Val is used solely for C0 epoch selection;
    # running C1/C2 on test keeps those predictions uncontaminated.
    build_split("test", builder, args.batch, args.workers)
