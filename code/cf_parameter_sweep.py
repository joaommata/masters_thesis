"""
cf_parameter_sweep.py
=====================
Grid search over (T_START, GUIDANCE_WEIGHT) to find the combination
that produces the most minimal counterfactual — i.e. flips C0's prediction
with the least image distortion.

Metrics per CF:
    - cf_prob       : C0 probability after generation
    - delta_prob    : |original_prob - cf_prob|
    - flipped       : 1 if prediction label changed
    - ssim          : structural similarity (higher = more similar to original)
    - lpips         : perceptual distance (lower = more similar), if available
    - efficiency    : delta_prob / (1 - ssim), proxy for flip-per-distortion
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from skimage.metrics import structural_similarity as ssim_fn
from PIL import Image
import torchxrayvision as xrv
import matplotlib.pyplot as plt
from tqdm import tqdm

# Attempt to import LPIPS — graceful fallback if not installed
try:
    import lpips
    LPIPS_AVAILABLE = True
    print("LPIPS available")
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS not found — using SSIM only. Install with: pip install lpips")

# ── Reuse your existing infrastructure ───────────────────────────────────────
# Adjust this path to wherever cf_generation.py lives
sys.path.insert(0, "/zhome/d0/a/221493/thesis/code")
from cf_generation import load_models, load_image, get_c0_prob, one_guided_step

BASE_DIR      = "/zhome/d0/a/221493/thesis"
C0_CSV        = f"{BASE_DIR}/results/C0_custom/effusion/val_c0_effusion.csv"
C0_THRESHOLD  = 0.5634
T_TOTAL       = 1000
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
OUTPUT_DIR    = f"{BASE_DIR}/results/cf_param_sweep"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Reduced parameter grid ────────────────────────────────────────────────────
# 10 T_START values (log-spaced to sample low end more densely, where CFs are subtler)
T_START_VALUES       = [10, 25, 50, 75, 100, 150, 200, 300, 500, 750]
print(f"T_START values: {T_START_VALUES}")
# 8 guidance weights
GUIDANCE_WEIGHT_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
print(f"Guidance weight values: {GUIDANCE_WEIGHT_VALUES}")
# Total: 80 combinations per sample

# ── Sample selection ──────────────────────────────────────────────────────────
N_SAMPLES_PER_BIN = 2  # how many samples per probability bin

def select_random_samples(df, n_samples=6, random_state=42):
    """
    Pick n_samples randomly from the full dataset.
    """
    chosen = df.sample(n_samples, random_state=random_state)
    selected = []
    for _, row in chosen.iterrows():
        selected.append(row.to_dict())
    return selected

# ── Image similarity metrics ──────────────────────────────────────────────────

def compute_ssim(img_orig_t, img_cf_t):
    """
    Compute SSIM between two image tensors in [-1, 1].
    Converts to [0, 1] numpy arrays first.
    """
    # Convert tensors to numpy [0,1]
    orig_np = ((img_orig_t[0, 0].cpu().numpy() + 1) / 2).clip(0, 1)
    cf_np   = ((img_cf_t[0, 0].cpu().numpy() + 1) / 2).clip(0, 1)
    return ssim_fn(orig_np, cf_np, data_range=1.0)


def compute_lpips(img_orig_t, img_cf_t, lpips_model):
    """
    Compute LPIPS perceptual distance.
    LPIPS expects (B, 3, H, W) in [-1, 1].
    """
    # Repeat grayscale channel to 3-channel (LPIPS expects RGB)
    orig_3ch = img_orig_t.repeat(1, 3, 1, 1)
    cf_3ch   = img_cf_t.repeat(1, 3, 1, 1)
    with torch.no_grad():
        dist = lpips_model(orig_3ch, cf_3ch)
    return dist.item()


# ── Single CF generation (parameterised) ─────────────────────────────────────

def generate_cf_with_params(unet, sd, x0, classifier, target_class, t_start, guidance_weight):
    """
    Generate a CF using the given T_START and GUIDANCE_WEIGHT.
    Stripped-down version of cf_generation.generate_cf() with explicit params.
    """
    from train.utils import get  # Nina's utility, same as in cf_generation.py
    
    # Forward diffusion: add noise up to t_start
    t_tensor = torch.tensor([t_start], dtype=torch.long, device=DEVICE)
    x, _     = sd.forward_diffusion(x0, t_tensor)
    
    # Reverse loop from t_start → 1
    for t_val in reversed(range(1, t_start + 1)):
        # one_guided_step uses the global GUIDANCE_WEIGHT from cf_generation.py
        # We override it here by calling the step logic directly
        x = _one_guided_step_parameterised(
            unet, sd, x, t_val, classifier, target_class, guidance_weight
        )
    
    cf_prob = get_c0_prob(x, classifier)
    return x, cf_prob


def _one_guided_step_parameterised(unet, sd, x_t, t_val, classifier, target_class, guidance_weight):
    """
    Same as cf_generation.one_guided_step() but with explicit guidance_weight arg.
    Avoids depending on the global GUIDANCE_WEIGHT constant.
    """
    from train.utils import get
    
    t_tensor       = torch.ones(1, dtype=torch.long, device=DEVICE) * t_val
    beta_t         = get(sd.beta.to(DEVICE), t_tensor)
    one_by_sqrt_at = get(sd.one_by_sqrt_alpha.to(DEVICE), t_tensor)
    sqrt_abar      = get(sd.sqrt_alpha_cumulative.to(DEVICE), t_tensor)
    sqrt_1mabar    = get(sd.sqrt_one_minus_alpha_cumulative.to(DEVICE), t_tensor)
    
    x_t_grad = x_t.detach().requires_grad_(True)
    eps_pred  = unet(x_t_grad, t_tensor)
    x0_hat    = ((x_t_grad - sqrt_1mabar * eps_pred) / sqrt_abar).clamp(-1, 1)
    
    x0_hat_3ch = ((x0_hat + 1) / 2).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)
    logit = classifier((x0_hat_3ch - mean) / std).squeeze()
    
    log_prob = (torch.log(torch.sigmoid(logit) + 1e-8) if target_class == 1
                else torch.log(1 - torch.sigmoid(logit) + 1e-8))
    
    grad = torch.autograd.grad(log_prob, x_t_grad)[0]
    
    z = torch.randn_like(x_t) if t_val > 1 else torch.zeros_like(x_t)
    with torch.no_grad():
        eps_no_grad = unet(x_t.detach(), t_tensor)
    
    x_prev = (
        one_by_sqrt_at * (x_t.detach() - (beta_t / sqrt_1mabar) * eps_no_grad)
        + guidance_weight * grad.detach()   # <-- explicit param here
        + torch.sqrt(beta_t) * z
    )
    return x_prev.detach()


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Grid: {len(T_START_VALUES)} T_START × {len(GUIDANCE_WEIGHT_VALUES)} G = "
          f"{len(T_START_VALUES)*len(GUIDANCE_WEIGHT_VALUES)} combinations\n")
    
    # Load models
    unet, sd, classifier, _ = load_models()  # seg model not needed here
    print("Models loaded.\n")
    
    # Load LPIPS model if available
    lpips_model = lpips.LPIPS(net='vgg').to(DEVICE) if LPIPS_AVAILABLE else None
    print("LPIPS model loaded.\n" if LPIPS_AVAILABLE else "LPIPS not available, skipping.\n")
    
    # Select stratified samples
    df = pd.read_csv(C0_CSV)
    print("Selecting stratified samples...")
    samples = select_random_samples(df, n_samples=6)
    print(f"Total samples selected: {len(samples)}\n")
    
    # ── Main loop ─────────────────────────────────────────────────────────
    records = []
    
    for sample in samples:
        x0         = load_image(sample['path'])
        orig_prob  = float(sample['prob'])
        target_cls = 1 - int(sample['pred'])  # flip direction
        
        print(f"\nSample: prob={orig_prob:.3f} | path={sample['path']}")
        
        for t_start in T_START_VALUES:
            for g_weight in GUIDANCE_WEIGHT_VALUES:
                
                # Generate CF
                x_cf, cf_prob = generate_cf_with_params(
                    unet, sd, x0, classifier, target_cls, t_start, g_weight
                )
                
                # Metrics
                delta_prob = abs(orig_prob - cf_prob)
                flipped    = int((orig_prob >= C0_THRESHOLD) != (cf_prob >= C0_THRESHOLD))
                sim_ssim   = compute_ssim(x0, x_cf)
                print(f"  T={t_start:4d}  G={g_weight:.2f}  "
                      f"cf_prob={cf_prob:.3f}  flipped={flipped}  ssim={sim_ssim:.3f}")
                # Efficiency: how much probability shift per unit of distortion
                distortion_ssim   = 1.0 - sim_ssim  # 0 = identical, 1 = completely different
                efficiency_ssim   = delta_prob / (distortion_ssim + 1e-6)
                
                rec = {
                    'path':             sample['path'],
                    'orig_prob':        orig_prob,
                    't_start':          t_start,
                    'guidance_weight':  g_weight,
                    'cf_prob':          cf_prob,
                    'delta_prob':       delta_prob,
                    'flipped':          flipped,
                    'ssim':             sim_ssim,
                    'efficiency_ssim':  efficiency_ssim,
                }
                
                # Add LPIPS if available
                if LPIPS_AVAILABLE:
                    sim_lpips = compute_lpips(x0, x_cf, lpips_model)
                    rec['lpips']            = sim_lpips
                    rec['efficiency_lpips'] = delta_prob / (sim_lpips + 1e-6)
                
                records.append(rec)
                print(f"  T={t_start:4d}  G={g_weight:.2f}  "
                      f"cf_prob={cf_prob:.3f}  flipped={flipped}  ssim={sim_ssim:.3f}")
    
    # ── Save raw results ──────────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'sweep_results.csv'), index=False)
    print(f"\nRaw results saved.")
    
    # ── Summary: average across samples ──────────────────────────────────
    # For each (t_start, guidance_weight) pair, average metrics across samples
    group_cols = ['t_start', 'guidance_weight']
    metric_cols = ['delta_prob', 'flipped', 'ssim', 'efficiency_ssim']
    if LPIPS_AVAILABLE:
        metric_cols += ['lpips', 'efficiency_lpips']
    
    summary = results_df.groupby(group_cols)[metric_cols].mean().reset_index()
    summary.to_csv(os.path.join(OUTPUT_DIR, 'sweep_summary.csv'), index=False)
    
    # ── Plot: heatmap of efficiency (flipped-only) ────────────────────────
    # Only consider combinations where the label actually flipped (on average)
    flipped_only = summary[summary['flipped'] >= 0.5]  # majority of samples flipped
    
    if len(flipped_only) > 0:
        pivot = flipped_only.pivot(index='t_start', columns='guidance_weight', values='efficiency_ssim')
        
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel('Guidance Weight')
        ax.set_ylabel('T_START')
        ax.set_title('CF Efficiency (Δprob / distortion)\nOnly combinations where ≥50% of samples flipped')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'efficiency_heatmap.png'), dpi=150)
        plt.close()
        print("Heatmap saved.")
    else:
        print("No parameter combination achieved flipping on majority of samples.")
    
    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()