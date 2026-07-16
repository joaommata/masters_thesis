"""
cf_generate_diffusion_counterfactuals.py
========================
Generates diffusion counterfactuals for the held-out set. They will be used to train and test C2 for the new version of the thesis.
They will have to have their attributes extracted.

Parameters chosen from param sweep (sweep_summary.csv):
  T_START=10, GUIDANCE_WEIGHT=0.25 → best SSIM×delta_prob tradeoff, 100% flip rate.
  (Previous: T_START=50, GUIDANCE_WEIGHT=0.025 — guidance was far too weak and not in the sweep range.)
11
Generates N_CFS counterfactuals per sample using different random seeds for diversity.
All CFs are saved regardless of outcome; the `flipped` column in the manifest records
whether each one crossed the Youden-optimal C0 threshold (loaded from results/C0_custom/effusion/threshold.txt).

Output:
    results/diffusion_cf/cf_manifest_{T_START}_{GUIDANCE_WEIGHT}_n{N_CFS}.csv
        columns: path, cf_idx, cf_path, cf_prob, flipped

    results/diffusion_cf/images/{T_START}_{GUIDANCE_WEIGHT}/
        PNGs named {base}_cf{idx}.png
"""

import os
import re
import sys
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# ── Stub out modules cf_generation.py imports but we don't need ──
from unittest.mock import MagicMock
for mod in ['pytorch_lightning', 'pytorch_lightning.core',
            'pytorch_lightning.core.lightning', 'pytorch_lightning.core.module',
            'torchaudio', 'models.base_classifier', 'models.resnet']:
    sys.modules[mod] = MagicMock()

sys.path.insert(0, "/zhome/d0/a/221493/thesis/FastDiME_Med")
sys.path.insert(0, "/zhome/d0/a/221493/thesis/code")

from cf_generation import load_models, load_image, generate_cf

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Best tradeoff from param sweep: T_START=10, w=0.25 → SSIM=0.903, delta_prob=0.879, 100% flip.
# For more aggressive editing use w=0.5 (SSIM=0.894, delta_prob=0.892).
T_START         = 10
GUIDANCE_WEIGHT = 0.25
N_CFS           = 3    # number of diverse CFs per sample (different noise seeds)

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"
C2_DATA_CSV    = f"{RESULTS_DIR}/C2_custom/effusion/c2_data.csv"
OUTPUT_DIR     = f"{RESULTS_DIR}/diffusion_cf"
CF_IMAGE_DIR   = os.path.join(OUTPUT_DIR, f"images/{T_START}_{GUIDANCE_WEIGHT}")
MANIFEST_PATH  = os.path.join(OUTPUT_DIR, f"cf_manifest_{T_START}_{GUIDANCE_WEIGHT}_n{N_CFS}.csv")

SUBSIZE        = 45000

# Load the Youden-optimal C0 threshold from training
with open(f"{RESULTS_DIR}/C0_custom/effusion/threshold.txt") as _f:
    FLIP_THRESHOLD = float(_f.read().strip())

os.makedirs(CF_IMAGE_DIR, exist_ok=True)

print(f"Config:\n  T_START={T_START}\n  GUIDANCE_WEIGHT={GUIDANCE_WEIGHT}\n  N_CFS={N_CFS}\n  DEVICE={DEVICE}")

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

    if SUBSIZE is not None:
        df = df.sample(n=SUBSIZE, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(df):,} samples")
    else:
        print(f"Using all {len(df):,} samples")

    # Resume support — skip paths where all N_CFS CFs are already done
    if os.path.exists(MANIFEST_PATH):
        done = pd.read_csv(MANIFEST_PATH)
        done_complete = done.groupby('path')['cf_idx'].count()
        complete_paths = set(done_complete[done_complete >= N_CFS].index)
        df = df[~df['path'].isin(complete_paths)].reset_index(drop=True)
        print(f"Resuming — {len(complete_paths):,} fully done, {len(df):,} remaining")
    else:
        done = pd.DataFrame(columns=['path', 'cf_idx', 'cf_path', 'cf_prob', 'flipped'])

    # Load models once
    print("Loading models...")
    unet, sd, classifier, _ = load_models()  # seg_model not needed here
    print("Models loaded\n")

    results = []
    n_flipped_total, n_attempted_total = 0, 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating CFs"):

        try:
            x0 = load_image(row['path'])
            target_class = 1 - int(row['pred'])
            orig_prob = float(row['pred'])

            for cf_idx in range(N_CFS):
                # Different seed per CF → different noise draws → diverse CFs.
                # Randomness comes from torch.randn_like in forward_diffusion and
                # each reverse step, so setting the seed here is sufficient.
                torch.manual_seed(i * N_CFS + cf_idx)

                x_cf, cf_prob, _, _ = generate_cf(
                    unet, sd, x0, classifier,
                    target_class=target_class,
                    n_snapshots=0,
                    track_probs=False,
                    t_start=T_START,
                    guidance_weight=GUIDANCE_WEIGHT,
                )

                # Check flip: for positive patients target_class=0 → want cf_prob < 0.5
                #             for negative patients target_class=1 → want cf_prob > 0.5
                flipped = (target_class == 0 and cf_prob < FLIP_THRESHOLD) or \
                          (target_class == 1 and cf_prob >= FLIP_THRESHOLD)

                n_attempted_total += 1
                if flipped:
                    n_flipped_total += 1

                cf_base = path_to_filename(row['path']).replace('.jpg', '').replace('.png', '')
                cf_filename = f"{cf_base}_cf{cf_idx}.png"
                cf_path = os.path.join(CF_IMAGE_DIR, cf_filename)
                tensor_to_pil(x_cf).save(cf_path)

                results.append({
                    'path':    row['path'],
                    'cf_idx':  cf_idx,
                    'cf_path': cf_path,
                    'cf_prob': cf_prob,
                    'flipped': int(flipped),
                })

        except Exception as e:
            print(f"\n[SKIP] {row['path']}: {e}")
            continue

        # Checkpoint every 500 samples
        if len(results) % (500 * N_CFS) == 0 and results:
            checkpoint = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
            checkpoint.to_csv(MANIFEST_PATH, index=False)
            flip_rate = n_flipped_total / max(n_attempted_total, 1)
            print(f"  flip rate so far: {flip_rate:.1%}")

    # Final save
    final = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
    final.to_csv(MANIFEST_PATH, index=False)
    flip_rate = n_flipped_total / max(n_attempted_total, 1)
    print(f"\nDone. Manifest saved to {MANIFEST_PATH}")
    print(f"Total CF images: {len(final):,}  |  flip rate: {flip_rate:.1%}")
    print(f"Flipped CFs (usable for strong delta signal): {n_flipped_total:,}")


if __name__ == "__main__":
    main()