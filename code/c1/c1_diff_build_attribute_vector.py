"""
cf_extract_attributes.py
========================
Extracts C1 attributes for diffusion-generated CF images.
Reads cf_manifest.csv, runs each CF image through FeatureVectorBuilder,
and saves cf_attributes.csv alongside the manifest.

Usage:
    python cf_extract_attributes.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torchxrayvision as xrv
from radiomics import featureextractor

# Import the builder class from your existing C1 script
sys.path.insert(0, "/zhome/d0/a/221493/thesis/code")
from c1_build_attribute_vector import FeatureVectorBuilder

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
MANIFEST_PATH = f"{RESULTS_DIR}/diffusion_cf/cf_manifest_10_0.25_n3.csv"
OUTPUT_PATH   = f"{RESULTS_DIR}/diffusion_cf/cf_attributes_10_0.25_n3.csv"
BATCH_SIZE    = 16
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_cf_image(cf_path):
    """
    Load a saved CF PNG (grayscale, 224x224) and convert to the tensor format
    that FeatureVectorBuilder expects: shape (1, 224, 224), range [-1024, 1024].
    """
    from PIL import Image
    img = Image.open(cf_path).convert('L')  # grayscale
    img_np = np.array(img).astype(np.float32) / 255.0  # [0, 1]
    
    # torchxrayvision models expect [-1024, 1024]
    img_np = (img_np * 2048.0) - 1024.0
    
    # Shape: (1, H, W) — single channel
    return torch.tensor(img_np).unsqueeze(0)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    
    # Load manifest
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Manifest loaded: {len(manifest):,} CF images")
    
    # Resume support
    if os.path.exists(OUTPUT_PATH):
        done = pd.read_csv(OUTPUT_PATH)
        done_paths = set(done['cf_path'].tolist())
        manifest = manifest[~manifest['cf_path'].isin(done_paths)].reset_index(drop=True)
        print(f"Resuming — {len(done_paths):,} already done, {len(manifest):,} remaining")
        all_vectors = done.to_dict('records')
    else:
        all_vectors = []
    
    # Load models — same as c1_build_attribute_vector.py
    print("Loading models...")
    models = {
        'age':  xrv.baseline_models.riken.AgeModel(),
        'sex':  xrv.baseline_models.mira.SexModel(),
        'race': xrv.baseline_models.emory_hiti.RaceModel()
    }
    seg_model = xrv.baseline_models.chestx_det.PSPNet()
    print("Loaded models:", list(models.keys()) + ['segmentation'])
    
    print("Loading radiomics extractor...")
    extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape2D')
    
    builder = FeatureVectorBuilder(
        models=models,
        segmentation_model=seg_model,
        radiomics_extractor=extractor,
        device=DEVICE
    )
    print("Models loaded\n")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(manifest), BATCH_SIZE), desc="Extracting CF attributes"):
        print(f"\nProcessing batch {batch_start // BATCH_SIZE + 1} / {(len(manifest) + BATCH_SIZE - 1) // BATCH_SIZE}")
        batch = manifest.iloc[batch_start : batch_start + BATCH_SIZE]
        
        img_tensors = []
        img_nps     = []
        img_paths   = []
        
        for _, row in batch.iterrows():
            print(f"Loading {row['cf_path']}...")
            try:
                t = load_cf_image(row['cf_path'])  # (1, H, W)
                img_tensors.append(t)
                img_nps.append(t[0].numpy())       # (H, W)
                img_paths.append(row['cf_path'])
            except Exception as e:
                print(f"[SKIP] {row['cf_path']}: {e}")
        
        if not img_tensors:
            continue
        
        # Stack into batch tensor: (B, 1, H, W)
        img_batch = torch.stack(img_tensors, dim=0)
        
        vectors = builder.build_vectors_batch(img_batch, img_nps, img_paths, plot=False)
        print(f"Extracted {len(vectors)} attribute vectors from batch")
        
        # Overwrite the 'path' key with 'cf_path' so we can join on it later
        for v, row in zip(vectors, batch.itertuples()):
            v['cf_path'] = v.pop('path')
            all_vectors.append(v)
            print(f"Added vector for {v['cf_path']}")
        
        # Checkpoint every ~500 samples
        if len(all_vectors) % 16 < BATCH_SIZE:
            pd.DataFrame(all_vectors).to_csv(OUTPUT_PATH, index=False)
    
    # Final save
    df_out = pd.DataFrame(all_vectors)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df_out):,} CF attribute vectors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()