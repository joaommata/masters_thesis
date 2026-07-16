"""
cf_generate_diverse_counterfactuals.py
======================================
Generates a STRUCTURED GRID of diffusion counterfactuals (CFs) per chest-Xray
image, for later "spatial fingerprint" / Tier-3 explainability features that feed C2.

Grid per image = 3 noise levels (t_start) x 5 conditions
    conditions = [general(unmasked), Facies Diaphragmatica, Left Lung, Right Lung, Spine(control)]
    -> 15 CFs / image.

This is a NEW script. It REUSES primitives from cf_generation.py and never edits them:
    load_models, load_image, get_c0_prob, one_guided_step (body copied into one_masked_step),
    segment_image. It does NOT touch cf_generate_diffusion_counterfactuals.py.

Design decisions (see plan create-a-new-script-memoized-lecun.md):
  * Subsample = balanced 1000/cell across (pred x correct) -> 4000 images -> 60k CFs.
  * Noise    = SHARED per (image, t_start): all 5 conditions denoise from the identical
               forward-noised x_{t_start}, so general-vs-masked differences are attributable
               to the region constraint, not to different noise draws.
  * RePaint  = out-of-mask region pinned by re-noising the ORIGINAL to level t-1 (same level as
               the evolving in-mask region). This is a deliberate improvement over FastDiME's
               p_sample_once, which anchors at t (diffusion.py:462) and injects per-step flicker.

Modes (drive the plan's build order from one file):
    --smoke      Step A : 1-image mask test; assert edits stay in-region; PNG panel.
    --calibrate  Step B : t_start ladder on ~20 images -> tstart_calibration.csv.
    --run        Steps C+D+full : balanced subsample + 15-CF grid (resumable). Default.

Output layout (OUTPUT_DIR):
    config.json
    tstart_calibration.csv
    cf_manifest.csv                    (concat of manifest/ shards)
    manifest/<image_slug>.csv          (15 rows each)
    trajectories/<shard>/<cf_id>.npy   ((t_start, 2) = [[t_val, prob], ...], atomic)
    tensors/<shard>/<cf_id>.npy        (float16 CF tensor)
    smoke/                             (Step A/D artifacts, excluded from resume)
"""

import os
import re
import sys
import json
import glob
import hashlib
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# ── Stub out modules cf_generation.py imports transitively but we don't need ──
from unittest.mock import MagicMock
for _mod in ['pytorch_lightning', 'pytorch_lightning.core',
             'pytorch_lightning.core.lightning', 'pytorch_lightning.core.module',
             'torchaudio', 'models.base_classifier', 'models.resnet']:
    sys.modules[_mod] = MagicMock()

sys.path.insert(0, "/zhome/d0/a/221493/thesis/FastDiME_Med")
sys.path.insert(0, "/zhome/d0/a/221493/thesis/code")

from cf_generation import (
    load_models, load_image, get_c0_prob, one_guided_step, segment_image,
)
from train.utils import get  # Nina's gather-and-reshape utility

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"

# ── Dataset registry ──────────────────────────────────────────────────────────
# Both datasets are 224x224 grayscale chest X-rays with a DenseNet121 C0 + Youden
# threshold and a c2_data.csv carrying path/prob/true/pred/correct/patient_id, so only
# paths, the image loader and the calibrated params differ.
#
# GUIDANCE_WEIGHT / T_START_VALUES are properties of a specific (classifier, diffusion
# model) pair — they do NOT transfer across datasets and must be re-derived per dataset
# with --calibrate-regions. medmnist's are None until that is run, so the script fails
# loudly rather than silently reusing CheXpert's.
DATASETS = {
    "chexpert": dict(
        c2_csv        = f"{RESULTS_DIR}/C2_custom/effusion/c2_data.csv",
        threshold_txt = f"{RESULTS_DIR}/C0_custom/effusion/threshold.txt",
        output_dir    = f"{RESULTS_DIR}/diverse_cf",
        image_root    = None,       # paths in c2_data.csv are already DATA_ROOT-relative
        image_suffix  = "",
        # Chosen by --calibrate-regions (region_calibration.csv, 20 imgs x 4 w x 2 t x 5 cond).
        # The prod script's 0.25 saturates C0 (flip-rate 1.0 everywhere, general saturation
        # 0.95-1.0, CFs pinned at ~0.02/~0.99) and, worse, COLLAPSES the region signal: mean
        # delta_prob spread across regions peaks at 0.05 and falls away either side —
        #   w:      0.01   0.05   0.10   0.25
        #   SPREAD: 0.264  0.355  0.328  0.253   (t_start=20)
        # because strong guidance drags every region to the same extreme (Spine's delta climbs
        # 0.273 -> 0.499 while Diaphragm only 0.628 -> 0.753 — the control starts flipping too).
        # 0.05 maximises separation with masked flip-rate 0.50-0.95, masked saturation <=0.25.
        # Cost: the unmasked "general" condition is ~half saturated (0.50) — accepted, as the
        # grid exists for the spatial fingerprint and "general" is a reference condition.
        # Region order diaph > L lung > R lung > spine is stable at EVERY weight tested
        # (and matches effusion pathology: fluid pools at the lung base).
        guidance_weight = 0.05,
        # Validated WITH MASKS at w=0.05 (region_calibration_lowt.csv): masked flip-rate
        # grades 0.80/0.85/1.00 (diaphragm) and SPREAD 0.299/0.319/0.356.
        t_start_values  = [5, 10, 20],
    ),
    "medmnist": dict(
        # ChestMNIST (NIH ChestX-ray14), same effusion task, 224x224 PNGs.
        c2_csv        = f"{RESULTS_DIR}/C2_medmnist/effusion/c2_data.csv",
        threshold_txt = f"{RESULTS_DIR}/C0_medmnist/effusion/threshold.txt",   # 0.0908
        output_dir    = f"{RESULTS_DIR}/diverse_cf_medmnist",
        # c2_data.csv paths look like "test/000000" — need a root and a .png suffix.
        image_root    = f"{DATA_ROOT}/medmnist/images",
        image_suffix  = ".png",
        # NOT calibrated yet — run --dataset medmnist --calibrate-regions first.
        # The C0 threshold is 0.0908 (vs CheXpert's 0.5634), i.e. a very different decision
        # landscape, so CheXpert's values are not a safe default.
        guidance_weight = None,
        t_start_values  = None,
    ),
}

# Bound at startup by configure_dataset(); module-level so existing references keep working.
DATASET       = "chexpert"
C2_DATA_CSV   = DATASETS["chexpert"]["c2_csv"]
THRESHOLD_TXT = DATASETS["chexpert"]["threshold_txt"]
OUTPUT_DIR    = DATASETS["chexpert"]["output_dir"]
IMAGE_ROOT    = DATASETS["chexpert"]["image_root"]
IMAGE_SUFFIX  = DATASETS["chexpert"]["image_suffix"]

GUIDANCE_WEIGHT = 0.05
N_PER_CELL      = 1000      # balanced 1000/cell across (pred x correct) -> 4000 images
DILATE_K        = 9         # odd; ~4px-radius max_pool2d dilation of region masks
SEED            = 42        # subsample selection seed
N_SHARDS        = 256       # hash buckets for tensors/ and trajectories/

CONTROL_REGION  = "Spine"                        # effusion-irrelevant negative control (seg idx 13)
REGIONS         = ["Facies Diaphragmatica", "Left Lung", "Right Lung", CONTROL_REGION]
CONDITIONS      = ["general"] + REGIONS           # 5 conditions; "general" = unmasked

# t_start ladder used by --calibrate (Step B). --run reads the 3 chosen values below.
CALIB_LADDER    = [20, 35, 50, 65, 80]
CALIB_N         = 20        # images for calibration

# --calibrate2d (Step B'): the 1-D ladder above returned flip-rate 1.0 at EVERY level with
# cf_prob bimodal at ~0.02/~0.99 — guidance at w=0.25 saturates C0, so t_start is inert.
# Probe BELOW that regime on both axes: the transition is under t=20 and under w=0.25
# (the prod script already flips 100% at t=10, w=0.25).
CALIB2D_WEIGHTS = [0.005, 0.01, 0.05, 0.1, 0.25]   # 0.25 kept as the saturated reference row
CALIB2D_LADDER  = [5, 10, 20, 50]

# --calibrate-regions (Step B''): --calibrate2d measured GENERAL CFs only. A masked CF may
# edit just one region, so it is strictly weaker: at the w=0.010 chosen for general CFs,
# no region flipped a p=0.997 image (diaph 0.923 / LL 0.768 / RL 0.959 / spine 0.970 vs
# threshold 0.5634) although the ORDER was preserved. This sweeps the region axis to find
# whether any weight flips masked CFs while leaving general CFs unsaturated.
# Pass 1 (weights x {10,20}) picked w=0.05. Pass 2 fixes that weight and probes the LOW
# t_start end, which the region sweep never covered: t=5 was only ever tested on general
# CFs, and a masked CF is strictly weaker, so its usability at t=5 is unmeasured.
CALIBR_WEIGHTS  = [0.05]
CALIBR_LADDER   = [3, 5, 10, 20]

# The 3 chosen noise levels, validated WITH MASKS at w=0.05 (region_calibration_lowt.csv).
# Masked flip-rate grades properly across them (diaphragm 0.80/0.85/1.00, left lung
# 0.55/0.70/0.75) and region separation is near its ceiling (SPREAD 0.299/0.319/0.356).
#   t:      3      5      10     20
#   SPREAD: 0.257  0.299  0.319  0.356
# t=3 is dropped: weakest separation, and Spine (the control) rises to 0.35 flip-rate while
# Diaphragm falls to 0.70 — signal and control start converging. Not extended past 20:
# SPREAD is still climbing, but Diaphragm already hits flip-rate 1.0 there (no headroom)
# and t=35+ costs 2.5x the steps.
T_START_VALUES  = [5, 10, 20]

def configure_dataset(name):
    """
    Point the module at one of DATASETS. Must run before anything reads the globals.

    Rebinds paths, the calibrated params and FLIP_THRESHOLD. Refuses a dataset whose
    guidance_weight/t_start_values are still None (i.e. not yet calibrated) for --run,
    so an uncalibrated dataset can never silently inherit CheXpert's regime.
    """
    global DATASET, C2_DATA_CSV, THRESHOLD_TXT, OUTPUT_DIR, IMAGE_ROOT, IMAGE_SUFFIX
    global GUIDANCE_WEIGHT, T_START_VALUES, FLIP_THRESHOLD

    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    cfg = DATASETS[name]

    DATASET       = name
    C2_DATA_CSV   = cfg["c2_csv"]
    THRESHOLD_TXT = cfg["threshold_txt"]
    OUTPUT_DIR    = cfg["output_dir"]
    IMAGE_ROOT    = cfg["image_root"]
    IMAGE_SUFFIX  = cfg["image_suffix"]

    if cfg["guidance_weight"] is not None:
        GUIDANCE_WEIGHT = cfg["guidance_weight"]
    if cfg["t_start_values"] is not None:
        T_START_VALUES = cfg["t_start_values"]

    with open(THRESHOLD_TXT) as f:
        FLIP_THRESHOLD = float(f.read().strip())

    print(f"Dataset: {name}  |  threshold={FLIP_THRESHOLD:.4f}  |  "
          f"w={cfg['guidance_weight']}  t_start={cfg['t_start_values']}")
    return cfg


def load_image_ds(rel_path):
    """
    Load an image as a (1,1,224,224) tensor in [-1,1], honouring the active dataset.

    cf_generation.load_image joins against DATA_ROOT and takes a full relative path, which
    is right for CheXpert. MedMNIST's c2_data.csv stores bare keys like "test/000000", so
    it needs its own root and a .png suffix. Identical preprocessing either way
    (grayscale, LANCZOS to 224, [0,1] -> [-1,1]).
    """
    if IMAGE_ROOT is None:
        return load_image(rel_path)                       # CheXpert: reuse as-is

    full = os.path.join(IMAGE_ROOT, rel_path + IMAGE_SUFFIX)
    img = Image.open(full).convert("L").resize((224, 224), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return (t * 2.0 - 1.0).to(DEVICE)


# Flip / decision threshold — single source of truth (== 0.5634 for chexpert).
with open(THRESHOLD_TXT) as _f:
    FLIP_THRESHOLD = float(_f.read().strip())

# ══════════════════════════════════════════════════════════════════════════════
# PATH / ID HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def path_to_slug(path):
    """Filesystem-safe slug for an image path (dataset-root-relative)."""
    if "CheXpert-v1.0-small/" in path:
        path = path.split("CheXpert-v1.0-small/", 1)[1]
    safe = path.replace("/", "_").replace("\\", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", safe).replace("._", "")
    return safe.replace(".jpg", "").replace(".png", "")


def make_cf_id(path, region, t_start):
    """Deterministic, collision-free id — pure function of the (path, region, t_start) triple."""
    return f"{path_to_slug(path)}__{region.replace(' ', '-')}__t{t_start}"


def shard_of(cf_id):
    """Stable hash bucket for cf_id so no dir holds all 60k files (Lustre stat/glob cost)."""
    h = int(hashlib.md5(cf_id.encode()).hexdigest(), 16)
    return f"{h % N_SHARDS:03d}"


def atomic_save_npy(arr, path):
    """Write an .npy atomically: tmp + os.replace (safe under crash/resume)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    np.save(tmp, arr)                 # np.save appends .npy to the given name...
    os.replace(tmp + ".npy", path)    # ...so the real tmp file is tmp + ".npy"


def tensor_paths(cf_id):
    """(cf_tensor_path, trajectory_path) for a cf_id, sharded."""
    shard = shard_of(cf_id)
    tpath = os.path.join(OUTPUT_DIR, "tensors", shard, f"{cf_id}.npy")
    jpath = os.path.join(OUTPUT_DIR, "trajectories", shard, f"{cf_id}.npy")
    return tpath, jpath

# ══════════════════════════════════════════════════════════════════════════════
# MASKING + RePaint (new)
# ══════════════════════════════════════════════════════════════════════════════

def segment_once(x0, seg_model):
    """
    Run the segmentation model ONCE and return all 14 channels as (14, 224, 224) numpy.

    Mirrors cf_generation.segment_image (PSPNet expects [-1024,1024], sigmoid, resize to
    224, threshold at 0.5) but runs on whatever device seg_model is on, so the caller can
    put it on the GPU. cf_generation.segment_image forces .cpu() internally and is called
    per-region, i.e. 4x per image on identical x0 — 3 of those are pure waste, and the CPU
    PSPNet is serial and unbatched, which dominated the per-image time.
    """
    dev = next(seg_model.parameters()).device
    x_xrv = ((x0 + 1) / 2).clamp(0, 1) * 2048.0 - 1024.0
    with torch.no_grad():
        seg_out = seg_model(x_xrv.to(dev))
    seg_out = torch.sigmoid(seg_out)
    seg_out = F.interpolate(seg_out, size=(224, 224), mode="bilinear", align_corners=False)
    return (seg_out[0] >= 0.5).to(torch.uint8).cpu().numpy()


def get_region_mask(x0, seg_model, region_name, seg=None):
    """
    Anatomical region mask from segmentation.

    Takes the named channel of the (14, 224, 224) segmentation, dilates by DILATE_K (repo
    idiom: F.max_pool2d, GPU, no scipy), and returns a (1,1,224,224) float CUDA tensor
    (1 inside region, 0 outside).

    Pass a precomputed `seg` (from segment_once) to avoid re-running the model per region.

    Returns (mask_tensor, area) so the caller can guard empty masks.
    """
    if seg is None:
        seg = segment_once(x0, seg_model)
    idx = seg_model.targets.index(region_name)                 # exact strings verified
    mask = torch.from_numpy(seg[idx]).float().view(1, 1, 224, 224).to(DEVICE)
    mask = F.max_pool2d(mask, DILATE_K, stride=1, padding=(DILATE_K - 1) // 2)
    mask = (mask > 0.5).float()                                # keep binary after pooling
    return mask, float(mask.sum().item())


def one_masked_step(unet, sd, x_t, t_val, classifier, target_class, mask, x0,
                    guidance_weight=None):
    """
    One classifier-guided DDPM reverse step, RePaint-restricted to `mask`.

    Body is a COPY of cf_generation.one_guided_step (that function is left untouched), plus a
    single re-anchoring line at the end: outside the mask, pixels are pinned to the ORIGINAL
    forward-noised to level t-1 (same noise level as the evolving in-mask region — no flicker).

    Ref: RePaint (Lugmayr et al., CVPR 2022). Guidance is (intentionally) computed on the whole
    x0_hat; only in-mask pixels are allowed to move (global guidance, local edit).

    guidance_weight=None reads the module global at CALL time — a `=GUIDANCE_WEIGHT` default
    would bind at import and survive configure_dataset(), silently using CheXpert's weight.
    """
    if guidance_weight is None:
        guidance_weight = GUIDANCE_WEIGHT

    t_tensor       = torch.ones(1, dtype=torch.long, device=DEVICE) * t_val
    beta_t         = get(sd.beta.to(DEVICE), t_tensor)
    one_by_sqrt_at = get(sd.one_by_sqrt_alpha.to(DEVICE), t_tensor)
    sqrt_abar      = get(sd.sqrt_alpha_cumulative.to(DEVICE), t_tensor)
    sqrt_1mabar    = get(sd.sqrt_one_minus_alpha_cumulative.to(DEVICE), t_tensor)

    # --- classifier-guided step (identical to one_guided_step) ---
    x_t_grad = x_t.detach().requires_grad_(True)
    eps_pred = unet(x_t_grad, t_tensor)
    x0_hat   = ((x_t_grad - sqrt_1mabar * eps_pred) / sqrt_abar).clamp(-1, 1)

    x0_hat_3ch = ((x0_hat + 1) / 2).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    logit = classifier((x0_hat_3ch - mean) / std).squeeze()

    log_prob = (torch.log(torch.sigmoid(logit) + 1e-8) if target_class == 1
                else torch.log(1 - torch.sigmoid(logit) + 1e-8))
    grad = torch.autograd.grad(log_prob, x_t_grad)[0]

    z = torch.randn_like(x_t) if t_val > 1 else torch.zeros_like(x_t)
    with torch.no_grad():
        eps_no_grad = unet(x_t.detach(), t_tensor)

    x_prev = (
        one_by_sqrt_at * (x_t.detach() - (beta_t / sqrt_1mabar) * eps_no_grad)
        + guidance_weight * grad.detach()
        + torch.sqrt(beta_t) * z
    )

    # --- RePaint re-anchor: pin outside-mask to the original noised to level t-1 ---
    with torch.no_grad():
        if t_val > 1:
            t_prev = t_tensor - 1
            x_bg, _ = sd.forward_diffusion(x0, t_prev)         # q(x_{t-1} | x0), fresh noise
        else:
            # Final step: the schedule's level 0 still carries ~1% noise
            # (sqrt_abar[0]*x0 + 0.01*eps), so pin to the CLEAN original instead —
            # canonical RePaint: at t=0 the known region is the data itself.
            x_bg = x0
        x_prev = mask * x_prev + (1 - mask) * x_bg
    return x_prev.detach()


def generate_cf_v2(unet, sd, x0, classifier, target_class, t_start, mask=None,
                   guidance_weight=None, track_probs=True, x_start=None):
    """
    Generate one CF for x0.

    Parameters
    ----------
    mask     : None -> general (unmasked) CF via one_guided_step.
               tensor -> masked CF via one_masked_step (only in-mask pixels evolve).
    x_start  : optional pre-computed forward-noised start x_{t_start}. Passing it lets the caller
               SHARE the identical starting point across the 5 conditions at a fixed t_start.
    guidance_weight : None -> read the module global at call time (see one_masked_step).

    Returns (x_cf, cf_prob, intermediate_probs) where intermediate_probs is a list of (t_val, prob).
    No image snapshots (memory).
    """
    if guidance_weight is None:
        guidance_weight = GUIDANCE_WEIGHT

    if x_start is None:
        t_tensor = torch.tensor([t_start], dtype=torch.long, device=DEVICE)
        x, _ = sd.forward_diffusion(x0, t_tensor)
    else:
        x = x_start.clone()

    intermediate_probs = []
    for t_val in reversed(range(1, t_start + 1)):
        if mask is None:
            x = one_guided_step(unet, sd, x, t_val, classifier, target_class,
                                guidance_weight=guidance_weight)
        else:
            x = one_masked_step(unet, sd, x, t_val, classifier, target_class,
                                mask, x0, guidance_weight=guidance_weight)
        if track_probs:
            intermediate_probs.append((t_val, get_c0_prob(x, classifier)))

    cf_prob = get_c0_prob(x, classifier)
    return x, cf_prob, intermediate_probs


def shared_start(sd, x0, path, t_start):
    """Deterministic forward-noised start shared by all conditions at (image, t_start)."""
    torch.manual_seed(hash((path, t_start)) & 0xFFFFFFFF)
    t_tensor = torch.tensor([t_start], dtype=torch.long, device=DEVICE)
    x_start, _ = sd.forward_diffusion(x0, t_tensor)
    return x_start

# ══════════════════════════════════════════════════════════════════════════════
# SUBSAMPLE (Step C)
# ══════════════════════════════════════════════════════════════════════════════

def build_balanced_subsample(df, n_per_cell=N_PER_CELL, seed=SEED):
    """
    Stratified sample of the 4 (pred x correct) cells, n_per_cell each (fixed seed).
    Leakage note: balancing selects ON `correct` (allowed for inclusion), but the 15-cell grid
    per image is fixed by (region, t_start) BEFORE `correct` is read into any feature.
    """
    cells = []
    for pred_v in (0, 1):
        for corr_v in (0, 1):
            cell = df[(df["pred"] == pred_v) & (df["correct"] == corr_v)]
            take = min(n_per_cell, len(cell))
            if take < n_per_cell:
                print(f"  [warn] cell pred={pred_v} correct={corr_v} has only {len(cell)} "
                      f"(< {n_per_cell}); taking all.")
            cells.append(cell.sample(n=take, random_state=seed))
    sub = pd.concat(cells).reset_index(drop=True)
    print(f"  balanced subsample: {len(sub)} images, {sub['patient_id'].nunique()} patients")
    return sub

# ══════════════════════════════════════════════════════════════════════════════
# RESUME
# ══════════════════════════════════════════════════════════════════════════════

MANIFEST_COLS = ["cf_id", "path", "patient_id", "orig_prob", "pred", "true", "correct",
                 "region", "t_start", "cf_tensor_path", "cf_prob", "flipped", "mask_empty"]


def load_done_triples():
    """Set of (path, region, t_start) already in any manifest shard (source of truth for resume)."""
    done = set()
    for shard_csv in glob.glob(os.path.join(OUTPUT_DIR, "manifest", "*.csv")):
        try:
            d = pd.read_csv(shard_csv, usecols=["path", "region", "t_start"])
            for _, r in d.iterrows():
                done.add((r["path"], r["region"], int(r["t_start"])))
        except Exception as e:
            print(f"  [warn] could not read {shard_csv}: {e}")
    return done


def write_config():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = {
        "t_start_values": T_START_VALUES,
        "guidance_weight": GUIDANCE_WEIGHT,
        "conditions": CONDITIONS,
        "regions": REGIONS,
        "control_region": CONTROL_REGION,
        "n_per_cell": N_PER_CELL,
        "threshold": FLIP_THRESHOLD,
        "seed": SEED,
        "dilate_k": DILATE_K,
        "n_shards": N_SHARDS,
        "anchor_level": "t-1",
        "noise_matching": "shared_per_(image,t_start)",
        "source_csv": C2_DATA_CSV,
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# STEP A — 1-image mask smoke test
# ══════════════════════════════════════════════════════════════════════════════

def run_smoke(models, n_images=1, t_start=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unet, sd, classifier, seg_model = models
    smoke_dir = os.path.join(OUTPUT_DIR, "smoke")
    os.makedirs(smoke_dir, exist_ok=True)

    df = pd.read_csv(C2_DATA_CSV, usecols=["path", "prob", "true", "pred", "correct", "patient_id"])
    sub = df.sample(n=n_images, random_state=SEED)

    for _, row in sub.iterrows():
        x0 = load_image_ds(row["path"])
        target_class = 1 - int(row["pred"])
        x_start = shared_start(sd, x0, row["path"], t_start)

        panels = [("original", x0, None)]
        # general
        x_cf, p, _ = generate_cf_v2(unet, sd, x0, classifier, target_class, t_start,
                                    mask=None, x_start=x_start, track_probs=False)
        panels.append((f"general p={p:.2f}", x_cf, None))

        # each masked region + in-region containment assertion
        seg = segment_once(x0, seg_model)
        for region in REGIONS:
            mask, area = get_region_mask(x0, seg_model, region, seg=seg)
            if area == 0:
                print(f"  [smoke] region '{region}' empty for {row['path']} — skipped")
                panels.append((f"{region}\n(EMPTY)", x0, None))
                continue
            x_cf, p, _ = generate_cf_v2(unet, sd, x0, classifier, target_class, t_start,
                                        mask=mask, x_start=x_start, track_probs=False)
            leak = float((torch.abs((x_cf - x0) * (1 - mask))).max().item())
            print(f"  [smoke] {region:<24} p={p:.3f}  out-of-mask |Δ|max={leak:.2e}  area={int(area)}")
            assert leak < 1e-4, f"CF leaked outside mask for {region}: {leak:.2e}"
            panels.append((f"{region}\np={p:.2f}", x_cf, mask))

        # figure
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
        for ax, (title, xt, mask) in zip(axes, panels):
            img = ((xt[0, 0].detach().cpu().numpy() + 1) / 2).clip(0, 1)
            ax.imshow(img, cmap="gray")
            if mask is not None:
                ax.contour(mask[0, 0].cpu().numpy(), levels=[0.5], colors="r", linewidths=0.6)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        out = os.path.join(smoke_dir, f"smoke_{path_to_slug(row['path'])}.png")
        plt.suptitle(f"pred={int(row['pred'])} true={int(row['true'])} t_start={t_start}", fontsize=10)
        plt.tight_layout()
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  [smoke] saved {out}")
    print("Step A smoke test passed: masked CFs confined to region.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP B — t_start calibration
# ══════════════════════════════════════════════════════════════════════════════

def run_calibrate(models, ladder=CALIB_LADDER, n_images=CALIB_N,
                  weights=None, conditions=("general",),
                  out_name="tstart_calibration.csv"):
    """
    Sweep t_start x guidance_weight x condition on a mixed sample.

    The 1-D t_start ladder at w=0.25 came back with flip-rate 1.0 at every level and
    cf_prob bimodal at ~0.02/~0.99 (median distance from threshold 0.43, 0/100 marginal),
    i.e. guidance saturates the classifier and t_start does nothing. Pass several
    `weights` to find the regime where flips are gradual and t_start actually bites.

    Pass `conditions` to also sweep the REGION axis: a masked CF may only edit one region,
    so it is strictly weaker than a general CF and needs its own flip-rate curve — a weight
    calibrated on general CFs alone does not transfer.

    weights=None -> [GUIDANCE_WEIGHT] read at call time (see one_masked_step).
    """
    if weights is None:
        weights = [GUIDANCE_WEIGHT]

    unet, sd, classifier, seg_model = models
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(C2_DATA_CSV, usecols=["path", "prob", "true", "pred", "correct", "patient_id"])
    # mix of correct/incorrect
    half = n_images // 2
    corr = df[df["correct"] == 1].sample(n=half, random_state=SEED)
    inc  = df[df["correct"] == 0].sample(n=n_images - half, random_state=SEED)
    sub = pd.concat([corr, inc]).reset_index(drop=True)

    total = len(sub) * len(ladder) * len(weights) * len(conditions)
    print(f"Sweep: {len(sub)} images x {len(ladder)} t_start x {len(weights)} weights "
          f"x {len(conditions)} conditions = {total} CFs")

    records = []
    pbar = tqdm(total=total, desc="Calibrating")
    for _, row in sub.iterrows():
        try:
            x0 = load_image_ds(row["path"])
            target_class = 1 - int(row["pred"])

            # Masks depend only on the image — compute once, reuse across t_start/weights.
            seg = segment_once(x0, seg_model)
            masks = {c: get_region_mask(x0, seg_model, c, seg=seg)
                     for c in conditions if c != "general"}

            for t_start in ladder:
                # Shared start per (image, t_start): weights/conditions compared on identical noise.
                x_start = shared_start(sd, x0, row["path"], t_start)
                for w in weights:
                    for cond in conditions:
                        if cond == "general":
                            mask, area = None, -1
                        else:
                            mask, area = masks[cond]
                            if area == 0:
                                continue           # empty segmentation — nothing to edit
                        _, cf_prob, _ = generate_cf_v2(unet, sd, x0, classifier, target_class,
                                                       t_start, mask=mask, x_start=x_start,
                                                       guidance_weight=w, track_probs=False)
                        flipped = int((cf_prob >= FLIP_THRESHOLD) != bool(int(row["pred"])))
                        records.append({"path": row["path"], "t_start": t_start,
                                        "guidance_weight": w, "region": cond,
                                        "orig_prob": float(row["prob"]), "cf_prob": cf_prob,
                                        "flipped": flipped})
                        pbar.update(1)
        except Exception as e:
            print(f"\n[SKIP] {row['path']}: {e}")
        torch.cuda.empty_cache()
    pbar.close()

    res = pd.DataFrame(records)

    # Every image can fail (e.g. the GPU is occupied by another job -> OOM on every CF).
    # Bail BEFORE writing: an empty frame has no columns, so the report below would raise
    # KeyError, and the empty file would clobber a previous good calibration.
    if res.empty:
        n_skipped = len(sub)
        raise RuntimeError(
            f"Calibration produced 0 CFs — all {n_skipped} images failed (see [SKIP] lines "
            f"above). Nothing written to {out_name}. If those are CUDA OOM, the GPU is "
            f"likely busy with another job: check `bjobs` / `nvidia-smi` and run on a free node."
        )

    out = os.path.join(OUTPUT_DIR, out_name)
    res.to_csv(out, index=False)
    print(f"\nCalibration saved -> {out}  ({len(res)} CFs, {res.path.nunique()} images)")

    # Saturation diagnostic: a flip that lands 0.43 from the threshold is guidance
    # steamrolling the classifier, not a counterfactual. `sat` = fraction of CFs pinned
    # to the extremes; usable regimes have low sat AND a mid-range flip rate.
    res["dist_thr"] = (res["cf_prob"] - FLIP_THRESHOLD).abs()
    res["saturated"] = ((res["cf_prob"] < 0.05) | (res["cf_prob"] > 0.95)).astype(int)

    multi_region = res["region"].nunique() > 1

    if not multi_region:
        print("\nflip-rate by (guidance_weight, t_start):")
        print(res.pivot_table(index="guidance_weight", columns="t_start",
                              values="flipped", aggfunc="mean").round(2).to_string())
        print("\nsaturation (frac of cf_prob <0.05 or >0.95 — want LOW):")
        print(res.pivot_table(index="guidance_weight", columns="t_start",
                              values="saturated", aggfunc="mean").round(2).to_string())
        print("\nmedian |cf_prob - threshold| (want SMALL = near boundary):")
        print(res.pivot_table(index="guidance_weight", columns="t_start",
                              values="dist_thr", aggfunc="median").round(3).to_string())
        print("\nPick a guidance_weight whose row has a flip-rate spanning ~0.2-0.9 across "
              "t_start with low saturation; set GUIDANCE_WEIGHT + 3 T_START_VALUES, then --run.")
        return res

    # --- region sweep: does a weight exist where masked CFs flip AND general isn't saturated? ---
    for t in sorted(res["t_start"].unique()):
        sl = res[res["t_start"] == t]
        print(f"\n=== t_start={t} — flip-rate by (guidance_weight, region) ===")
        print(sl.pivot_table(index="guidance_weight", columns="region",
                             values="flipped", aggfunc="mean").round(2).to_string())
        print(f"--- t_start={t} — saturation (want LOW, esp. for 'general') ---")
        print(sl.pivot_table(index="guidance_weight", columns="region",
                             values="saturated", aggfunc="mean").round(2).to_string())

    # The fingerprint needs regions to SEPARATE, not merely to flip: spread of per-region
    # mean delta-prob at each weight. A weight where every region moves the prob equally
    # carries no spatial information, however high its flip rate.
    res["delta_prob"] = (res["orig_prob"] - res["cf_prob"]).abs()
    reg_only = res[res["region"] != "general"]
    print("\n=== region separation: spread (max-min) of mean delta_prob across regions ===")
    print("    (want LARGE = regions respond differently = spatial signal)")
    sep = (reg_only.pivot_table(index=["guidance_weight", "t_start"], columns="region",
                                values="delta_prob", aggfunc="mean"))
    sep["SPREAD"] = sep.max(axis=1) - sep.min(axis=1)
    print(sep.round(3).to_string())

    print("\nWant: masked flip-rate mid-range, 'general' saturation low, SPREAD large.")
    print("If no weight satisfies all three, prefer continuous delta_prob over `flipped` "
          "as the region feature and set GUIDANCE_WEIGHT for an unsaturated general CF.")
    return res

# ══════════════════════════════════════════════════════════════════════════════
# STEPS C+D+full — the 15-CF grid
# ══════════════════════════════════════════════════════════════════════════════

def run_grid(models, limit=None):
    unet, sd, classifier, seg_model = models
    write_config()
    os.makedirs(os.path.join(OUTPUT_DIR, "manifest"), exist_ok=True)

    df = pd.read_csv(C2_DATA_CSV,
                     usecols=["path", "prob", "true", "pred", "correct", "patient_id"])
    print(f"Loaded {len(df):,} rows from {C2_DATA_CSV}")

    sub = build_balanced_subsample(df)
    if limit is not None:
        sub = sub.head(limit).reset_index(drop=True)
        print(f"  [smoke/limit] restricting to first {len(sub)} images")

    done = load_done_triples()
    print(f"Resume: {len(done):,} (path,region,t_start) triples already done")

    n_flipped, n_generated = 0, 0

    for _, row in tqdm(sub.iterrows(), total=len(sub), desc="15-CF grid"):
        path = row["path"]
        slug = path_to_slug(path)
        shard_csv = os.path.join(OUTPUT_DIR, "manifest", f"{slug}.csv")

        try:
            # Skip whole image only if every triple is already present
            wanted = {(path, c, t) for t in T_START_VALUES for c in CONDITIONS}
            if wanted.issubset(done):
                continue

            x0 = load_image_ds(path)
            target_class = 1 - int(row["pred"])            # identical for all 15 cells

            # Segment once, then slice the 4 region channels out of it.
            seg = segment_once(x0, seg_model)
            masks = {r: get_region_mask(x0, seg_model, r, seg=seg) for r in REGIONS}

            new_rows = []
            for t_start in T_START_VALUES:
                x_start = shared_start(sd, x0, path, t_start)   # shared across 5 conditions

                for cond in CONDITIONS:
                    if (path, cond, t_start) in done:
                        continue

                    cf_id = make_cf_id(path, cond, t_start)
                    tpath, jpath = tensor_paths(cf_id)
                    mask_empty = 0

                    if cond == "general":
                        mask = None
                    else:
                        mask, area = masks[cond]
                        if area == 0:
                            mask_empty = 1

                    if mask_empty:
                        # Don't pollute the cell with a meaningless trajectory.
                        cf_prob = np.nan
                        flipped = 0
                    else:
                        x_cf, cf_prob, traj = generate_cf_v2(
                            unet, sd, x0, classifier, target_class, t_start,
                            mask=mask, x_start=x_start, track_probs=True)
                        flipped = int((cf_prob >= FLIP_THRESHOLD) != bool(int(row["pred"])))
                        n_generated += 1
                        n_flipped += flipped
                        # trajectory FIRST (atomic), then tensor, then manifest row
                        atomic_save_npy(np.asarray(traj, dtype=np.float32), jpath)
                        atomic_save_npy(x_cf[0, 0].detach().cpu().numpy().astype(np.float16), tpath)

                    new_rows.append({
                        "cf_id": cf_id, "path": path, "patient_id": row["patient_id"],
                        "orig_prob": float(row["prob"]), "pred": int(row["pred"]),
                        "true": int(row["true"]), "correct": int(row["correct"]),
                        "region": cond, "t_start": int(t_start),
                        "cf_tensor_path": ("" if mask_empty else tpath),
                        "cf_prob": cf_prob, "flipped": flipped, "mask_empty": mask_empty,
                    })
                    done.add((path, cond, t_start))

            # Per-image manifest shard (append if resuming a partially-done image)
            if new_rows:
                dfn = pd.DataFrame(new_rows, columns=MANIFEST_COLS)
                if os.path.exists(shard_csv):
                    dfn = pd.concat([pd.read_csv(shard_csv), dfn], ignore_index=True)
                tmp = shard_csv + ".tmp"
                dfn.to_csv(tmp, index=False)
                os.replace(tmp, shard_csv)

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[SKIP] {path}: {e}")
            continue

    # Concatenate shards into the final manifest
    concat_manifest()
    rate = n_flipped / max(n_generated, 1)
    print(f"\nDone. Generated {n_generated:,} CFs this run  |  flip rate {rate:.1%}")


def concat_manifest():
    shards = sorted(glob.glob(os.path.join(OUTPUT_DIR, "manifest", "*.csv")))
    if not shards:
        print("No manifest shards to concatenate.")
        return
    parts = [pd.read_csv(s) for s in shards]
    full = pd.concat(parts, ignore_index=True)
    out = os.path.join(OUTPUT_DIR, "cf_manifest.csv")
    full.to_csv(out, index=False)
    print(f"Manifest ({len(full):,} rows) -> {out}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Diverse (region x t_start) diffusion CF grid.")
    ap.add_argument("--dataset", choices=list(DATASETS), default="chexpert",
                    help="Which dataset to run against (default: %(default)s). Calibrated "
                         "params are per-dataset and do not transfer.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Step A: 1-image mask test.")
    mode.add_argument("--calibrate", action="store_true", help="Step B: t_start ladder.")
    mode.add_argument("--calibrate2d", action="store_true",
                      help="Step B': sweep guidance_weight x t_start (use when the "
                           "t_start ladder comes back saturated).")
    mode.add_argument("--calibrate-regions", action="store_true",
                      help="Step B'': sweep guidance_weight x t_start x REGION — masked CFs "
                           "are weaker than general ones and need their own flip-rate curve.")
    mode.add_argument("--run", action="store_true", help="Steps C+D+full: the 15-CF grid (default).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Restrict --run to the first N images (Step D smoke: --limit 10).")
    ap.add_argument("--smoke-images", type=int, default=1)
    ap.add_argument("--smoke-tstart", type=int, default=50)
    ap.add_argument("--calib-images", type=int, default=CALIB_N,
                    help="Images for calibration (default %(default)s).")
    ap.add_argument("--calib-weights", type=float, nargs="+", default=None,
                    help="guidance_weight values for --calibrate-regions "
                         f"(default {CALIBR_WEIGHTS}).")
    ap.add_argument("--calib-tstart", type=int, nargs="+", default=None,
                    help=f"t_start values for --calibrate-regions (default {CALIBR_LADDER}).")
    ap.add_argument("--out-name", default=None,
                    help="Override the calibration CSV filename (avoid clobbering a previous "
                         "sweep, e.g. --out-name region_calibration_pass1.csv).")
    args = ap.parse_args()

    cfg = configure_dataset(args.dataset)

    # A dataset with no calibrated params may be smoked/calibrated but never --run:
    # inheriting CheXpert's regime would silently produce data from an uncalibrated
    # (and, on the evidence so far, probably wrong) guidance/t_start setting.
    running = not (args.smoke or args.calibrate or args.calibrate2d or args.calibrate_regions)
    if running and (cfg["guidance_weight"] is None or cfg["t_start_values"] is None):
        ap.error(f"--dataset {args.dataset} has no calibrated guidance_weight/t_start_values. "
                 f"Run --dataset {args.dataset} --calibrate-regions first, then set them in "
                 f"DATASETS['{args.dataset}'].")

    print(f"Device: {DEVICE}  |  FLIP_THRESHOLD={FLIP_THRESHOLD}")
    print("Loading models...")
    unet, sd, classifier, seg_model = load_models()
    # load_models parks the segmentation model on CPU (it is only used for plotting in
    # cf_generation). Here it runs on every image, and the serial CPU PSPNet dominated the
    # per-image time — move it to the GPU. segment_once() follows the model's device.
    seg_model = seg_model.to(DEVICE)
    models = (unet, sd, classifier, seg_model)
    print(f"Models loaded (seg on {next(seg_model.parameters()).device}).\n")

    if args.smoke:
        run_smoke(models, n_images=args.smoke_images, t_start=args.smoke_tstart)
    elif args.calibrate:
        run_calibrate(models, n_images=args.calib_images,
                      out_name=args.out_name or "tstart_calibration.csv")
    elif args.calibrate2d:
        run_calibrate(models, ladder=CALIB2D_LADDER, weights=CALIB2D_WEIGHTS,
                      n_images=args.calib_images,
                      out_name=args.out_name or "guidance_tstart_calibration.csv")
    elif args.calibrate_regions:
        run_calibrate(models,
                      ladder=args.calib_tstart or CALIBR_LADDER,
                      weights=args.calib_weights or CALIBR_WEIGHTS,
                      conditions=CONDITIONS, n_images=args.calib_images,
                      out_name=args.out_name or "region_calibration_lowt.csv")
    else:
        run_grid(models, limit=args.limit)


if __name__ == "__main__":
    main()
