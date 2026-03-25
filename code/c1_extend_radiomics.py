# scripts/c1_extend_radiomics.py
# Extends existing attribute CSVs with additional radiomics feature classes
# WITHOUT re-running demographic/segmentation predictions
# Saves to a NEW file, leaving originals untouched.

import os
import logging
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from radiomics import featureextractor
import SimpleITK as sitk
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader

logger = logging.getLogger('radiomics')
logger.setLevel(logging.ERROR)


# ── Configure which NEW feature classes you want to add ──────────────────────
NEW_FEATURE_CLASSES = ['glcm']  # add/remove as needed
# ─────────────────────────────────────────────────────────────────────────────


def build_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)
    extractor.disableAllFeatures()
    for fc in NEW_FEATURE_CLASSES:
        extractor.enableFeatureClassByName(fc)
    return extractor


def extract_radiomics_new(extractor, img_np, mask_np, class_name):
    mask_resized = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0).astype(np.uint8)

    if mask_resized.ndim != 2 or mask_resized.sum() < 10:
        return {}
    ys, xs = np.where(mask_resized > 0)
    if (ys.max() - ys.min()) < 2 or (xs.max() - xs.min()) < 2:
        return {}

    img_sitk  = sitk.GetImageFromArray(img_np.astype(np.float32))
    mask_sitk = sitk.GetImageFromArray(mask_resized.astype(np.uint8))

    try:
        feats = extractor.execute(img_sitk, mask_sitk)
        return {f"{class_name}_{k}": v for k, v in feats.items()
                if not k.startswith("diagnostics_")}
    except Exception as e:
        print(f"[ERROR] {class_name}: {e}")
        return {}


def extend_split(csv_filename, attribute_csv, base_dir, segmentation_model,
                 batch_size=16, num_workers=12):

    data_path    = os.path.join(base_dir, "data", "CheXpert-v1.0-small")
    csv_path     = os.path.join(data_path, csv_filename)
    input_path   = os.path.join(base_dir, "results/C1_attributes", attribute_csv)
    output_path  = os.path.join(base_dir, "results/C1_attributes",
                                attribute_csv.replace(".csv", "_extended.csv"))

    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}  (original untouched)")

    # ── Load existing results ─────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} existing rows, {df.shape[1]} columns")

    # ── Resume: if output already exists, load it instead to continue from checkpoint
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        print(f"[RESUME] Found existing output with {df.shape[1]} columns, resuming...")

    extractor = build_extractor()

    # ── Dataset ───────────────────────────────────────────────────────────────
    transform = xrv.datasets.XRayResizer(224)
    dataset = xrv.datasets.CheX_Dataset(
        imgpath=data_path, csvpath=csv_path,
        views=["PA", "AP"], transform=transform, unique_patients=False
    )

    segmentation_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentation_model.to(device)

    # Index df by path for fast lookup
    df = df.set_index("path")
    new_cols_data = {path: {} for path in df.index}

    # Identify paths that still need processing
    already_extended = any(
        col for col in df.columns
        if any(fc in col.lower() for fc in NEW_FEATURE_CLASSES)
    )

    paths_done = set(
        path for path in df.index
        if already_extended and any(
            pd.notna(df.loc[path, col])
            for col in df.columns
            if any(fc in col.lower() for fc in NEW_FEATURE_CLASSES)
        )
    )

    paths_todo = [p for p in df.index if p not in paths_done]
    print(f"[INFO] {len(paths_todo)} rows need new radiomics features")

    if not paths_todo:
        print("[INFO] Nothing to do — all rows already have the new features!")
        _save(df, new_cols_data, output_path)
        return

    paths_todo_set = set(paths_todo)

    # ── Process in batches ────────────────────────────────────────────────────
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    samples_processed = 0

    for batch in tqdm(loader, desc=f"Extending {attribute_csv}"):
        img_tensors = batch['img'].float()
        B = img_tensors.shape[0]
        img_paths = dataset.csv.iloc[samples_processed:samples_processed + B]['Path'].tolist()

        # Skip entire batch if none of these paths need processing
        if not any(p in paths_todo_set for p in img_paths):
            samples_processed += B
            continue

        # Segmentation
        with torch.no_grad():
            seg_output = segmentation_model(img_tensors.to(device)).cpu().numpy()
        seg_output = 1 / (1 + np.exp(-seg_output))
        seg_output = (seg_output >= 0.5).astype(np.uint8)

        # Build tasks only for needed paths
        tasks = []
        for b, path in enumerate(img_paths):
            if path not in paths_todo_set:
                continue
            img_np = img_tensors[b, 0].numpy()
            for class_idx, class_name in enumerate(segmentation_model.targets):
                mask_np = seg_output[b, class_idx]
                if mask_np.sum() == 0:
                    continue
                tasks.append((path, class_name, img_np, mask_np))

        def run(args):
            path, class_name, img_np, mask_np = args
            feats = extract_radiomics_new(extractor, img_np, mask_np, class_name)
            return path, feats

        with ThreadPoolExecutor(max_workers=8) as executor:
            for path, feats in executor.map(run, tasks):
                new_cols_data[path].update(feats)

        samples_processed += B

        # Checkpoint every ~500 samples
        if samples_processed % 500 < batch_size:
            _save(df, new_cols_data, output_path)
            print(f"[CHECKPOINT] {samples_processed} processed → {output_path}")

    _save(df, new_cols_data, output_path)
    print(f"Done. Final shape: {pd.read_csv(output_path).shape}")


def _save(df, new_cols_data, output_path):
    """Merge new feature columns into df and save."""
    new_df = pd.DataFrame.from_dict(new_cols_data, orient='index')
    merged = df.join(new_df, how='left')
    merged.index.name = 'path'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.reset_index().to_csv(output_path, index=False)


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    segmentation_model = xrv.baseline_models.chestx_det.PSPNet()

    for csv_file, attr_file in [
        ("train.csv", "train_c1_attribute_vector_rad.csv"),
        ("valid.csv", "valid_c1_attribute_vector_rad.csv"),
    ]:
        extend_split(
            csv_filename=csv_file,
            attribute_csv=attr_file,
            base_dir=base_dir,
            segmentation_model=segmentation_model,
            batch_size=16,
            num_workers=4,
        )


if __name__ == "__main__":
    main()