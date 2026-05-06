"""
cf_generate_diffusion_counterfactuals.py
========================
Generates diffusion counterfactuals for the held-out set. They will be used to train and test C2 for the new version of the thesis.
They will have to have their attributes extracted.
Reuses generate_cf() from cf_generation.py with T_START=100, WEIGHT=0.01.

Output:
    results/diffusion_cf/cf_manifest.csv
        columns: path, cf_path, cf_prob

    results/diffusion_cf/images/
        one PNG per sample named by a hash of the original path
"""

import os
import re
import sys
import hashlib
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import re

# ── Stub out modules cf_generation.py imports but we don't need ──
from unittest.mock import MagicMock
for mod in ['pytorch_lightning', 'pytorch_lightning.core',
            'pytorch_lightning.core.lightning', 'pytorch_lightning.core.module',
            'torchaudio', 'models.base_classifier', 'models.resnet']:
    sys.modules[mod] = MagicMock()

sys.path.insert(0, "/zhome/d0/a/221493/thesis/FastDiME_Med")
sys.path.insert(0, "/zhome/d0/a/221493/thesis/code")

# Import everything we need from your existing cf_generation.py
from cf_generation import load_models, load_image, generate_cf

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — only changes from cf_generation.py defaults
# ══════════════════════════════════════════════════════════════════════════════


T_START         = 50
GUIDANCE_WEIGHT = 0.025
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR       = "/zhome/d0/a/221493/thesis"
C2_DATA_CSV    = f"{BASE_DIR}/results/C2_custom/c2_data.csv"
OUTPUT_DIR     = f"{BASE_DIR}/results/diffusion_cf"
CF_IMAGE_DIR   = os.path.join(OUTPUT_DIR, f"images/{T_START}_{GUIDANCE_WEIGHT}")
MANIFEST_PATH  = os.path.join(OUTPUT_DIR, f"cf_manifest_{T_START}_{GUIDANCE_WEIGHT}.csv")

SUBSIZE        = 45000

os.makedirs(CF_IMAGE_DIR, exist_ok=True)

print(f"Config:\n  T_START={T_START}\n  GUIDANCE_WEIGHT={GUIDANCE_WEIGHT}\n  DEVICE={DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def path_to_filename(path):
    # Find dataset root dynamically
    if "CheXpert-v1.0-small/" in path:
        path = path.split("CheXpert-v1.0-small/", 1)[1]

    # Sanitize
    safe = path.replace("/", "_").replace("\\", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", safe)

    # Remove macOS artifacts like "._"
    safe = safe.replace("._", "")

    return f"cf_{safe}"

def tensor_to_pil(x_t):
    """Convert [-1,1] tensor (1,1,H,W) to PIL grayscale image."""
    img_np = ((x_t[0, 0].cpu().numpy() + 1) / 2).clip(0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np, mode='L')

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():


    # Load the held-out C2 dataset
    df = pd.read_csv(C2_DATA_CSV)
    print(f"Loaded {len(df):,} samples from {C2_DATA_CSV}")

    # Subsample 10K randomly
    df = df.sample(n=SUBSIZE, random_state=42).reset_index(drop=True)
    print(f"Subsampled to {len(df):,} samples")
    
    # Subsample if SUBSIZE is set, otherwise use all samples
    if SUBSIZE is not None:
        df = df.sample(n=SUBSIZE, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(df):,} samples")
    else:
        print(f"Using all {len(df):,} samples")

    # Resume support — skip already processed paths
    if os.path.exists(MANIFEST_PATH):
        done = pd.read_csv(MANIFEST_PATH)
        done_paths = set(done['path'].tolist())
        df = df[~df['path'].isin(done_paths)].reset_index(drop=True)
        print(f"Resuming — {len(done_paths):,} already done, {len(df):,} remaining")
    else:
        done = pd.DataFrame(columns=['path', 'cf_path', 'cf_prob'])

    # Load models once
    print("Loading models...")
    unet, sd, classifier, _ = load_models()  # seg_model not needed here
    print("Models loaded\n")

    results = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating CFs"):

        try:
            # Load original image
            x0 = load_image(row['path'])
            print(f"\nProcessing {row['path']} (pred={row['pred']:.4f}, true={row['true']})")
            
            # Target class is the opposite of the current prediction
            target_class = 1 - int(row['pred'])
            print(f"Target class for CF: {target_class}")
            
            # Generate CF — no snapshots needed, set n_snapshots=0
            print("Generating counterfactual...")
            x_cf, cf_prob,_ ,_ = generate_cf(
                unet, sd, x0, classifier,
                target_class=target_class,
                n_snapshots=0,
                track_probs=False
            )

            # Save CF image
            print(f"Saving CF with predicted prob {cf_prob:.4f} for target class {target_class}")
            cf_filename = path_to_filename(row['path'])
            cf_path = os.path.join(CF_IMAGE_DIR, cf_filename)
            tensor_to_pil(x_cf).save(cf_path)

            results.append({
                'path':    row['path'],
                'cf_path': cf_path,
                'cf_prob': cf_prob,
            })

        except Exception as e:
            print(f"\n[SKIP] {row['path']}: {e}")
            continue

        # Checkpoint every 500 samples
        if len(results) % 500 == 0 and results:
            checkpoint = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
            checkpoint.to_csv(MANIFEST_PATH, index=False)

    # Final save
    final = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
    final.to_csv(MANIFEST_PATH, index=False)
    print(f"\nDone. Manifest saved to {MANIFEST_PATH}")
    print(f"Total CFs generated: {len(final):,}")


if __name__ == "__main__":
    main()