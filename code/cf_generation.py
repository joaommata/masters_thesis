"""
cf_generation.py
================
Classifier-guided diffusion counterfactual generation for CheXpert.

Pipeline:
  1. Load 4 samples (TP, TN, FP, FN) from C0 predictions CSV
  2. For each sample, generate a CF via classifier-guided reverse diffusion
  3. Plot original → intermediate steps → CF with segmentation overlays
  4. Save results CSV and images

What this does NOT do:
  - ΔA computation or C2 integration
  - Hyperparameter sweeps
"""

import os
import sys
from unittest.mock import MagicMock

# Stub out modules Nina's code imports but we don't need
for mod in ['pytorch_lightning', 'pytorch_lightning.core',
            'pytorch_lightning.core.lightning', 'pytorch_lightning.core.module',
            'torchaudio', 'models.base_classifier', 'models.resnet']:
    sys.modules[mod] = MagicMock()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models
import torchxrayvision as xrv
import torch.nn.functional as F

sys.path.insert(0, "/zhome/d0/a/221493/thesis/FastDiME_Med")
from models.unet import UNet
from models.diffusion import SimpleDiffusion
from train.utils import get


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG -> defining the hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR       = "/zhome/d0/a/221493/thesis"
C0_CSV         = f"{BASE_DIR}/results/C0_custom/effusion/val_c0_effusion.csv"
C0_MODEL_PATH  = f"{BASE_DIR}/results/C0_custom/c0_best.pt"
C2_CSV = f"{BASE_DIR}/results/C2_custom/c2_data.csv"
FOLD_CSV = f"{BASE_DIR}/results/C2_custom/effusion/cv_results/cf_1/fold_0_predictions.csv"
# fold predictions already have prob/pred/true and cf_paths
DIFFUSION_CKPT = f"{BASE_DIR}/FastDiME_Med/pretrained_models/diffusion/OUT_CHEXPERT_CardioSplit/ckpt.tar"

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
T_START         = 50    # noise level before reversing: higher = more freedom, less = identity preservation
GUIDANCE_WEIGHT = 0.025    # classifier guidance strength, how much it influences the generation, multiplies by the classifier gradient at each step
T_TOTAL         = 1000   # must match the checkpoint
C0_THRESHOLD    = 0.5634 # taken from C0's training
N_SNAPSHOTS     = 5 # intermediate steps to segment and plot (evenly spaced between T_START and 1)
CLASSES_TO_SHOW = [
    'Left Lung', 'Right Lung', 'Heart',
    'Left Clavicle', 'Right Clavicle',
    'Spine', 'Diaphragm'
]

OUTPUT_PATH    = f"{BASE_DIR}/results/diffusion_cf/{T_START}/{GUIDANCE_WEIGHT}"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    """Load and return all models: UNet, diffusion schedule, C0 classifier, segmentation."""

    # Nina's UNet (frozen backbone)
    # Predicts noise at each timestep.
    
    unet = UNet(
        input_channels=1, output_channels=1,
        base_channels=64, base_channels_multiples=(1, 2, 4, 4),
        apply_attention=(False, False, False, False),
        dropout_rate=0.1, time_multiple=4,
    )
    ckpt = torch.load(DIFFUSION_CKPT, map_location='cpu')
    unet.load_state_dict(ckpt['model'])
    unet.eval().to(DEVICE)
    print(f"UNet loaded ({sum(p.numel() for p in unet.parameters())/1e6:.1f}M params)")

    # Diffusion noise schedule
    # Handles forward and reverse processes, stores beta, alpha, etc.
    # beta = noise variance at each timestep, alpha = 1 - beta, etc.
    
    sd = SimpleDiffusion(num_diffusion_timesteps=T_TOTAL, img_shape=(1, 224, 224), device=DEVICE)

    # C0 DenseNet classifier (frozen — used only for gradient guidance)
    # Binary (effusion yes or no)
    
    c0 = models.densenet121()
    c0.classifier = nn.Linear(c0.classifier.in_features, 1)
    c0.load_state_dict(torch.load(C0_MODEL_PATH, map_location=DEVICE))
    c0.eval().to(DEVICE)
    print("C0 classifier loaded")

    # Segmentation model (same as C1 pipeline)
    # Detects 14 structures, but we'll only plot a subset of them for clarity.
    # Does not influence the generation, just for visualisation of intermediates and CF.
    
    seg = xrv.baseline_models.chestx_det.PSPNet()
    seg.eval().cpu()
    print(f"Segmentation model loaded ({len(seg.targets)} classes)")

    return unet, sd, c0, seg


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_image(relative_path):
    """Load a CheXpert image as a (1,1,224,224) tensor in [-1, 1]."""
    
    full_path = os.path.join(BASE_DIR, "data", relative_path)
    
    # Loads a grayscale xray
    img = Image.open(full_path).convert('L')
    
    # Converts to 224x224, normalises [-1,1]
    img = img.resize((224, 224), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_t  = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1,1,224,224)
    return (img_t * 2.0 - 1.0).to(DEVICE)                        # [0,1] → [-1,1]


def get_c0_prob(x_t, classifier):
    """
    Run C0 classifier on a tensor in [-1,1].
    Converts to 3-channel ImageNet-normalised input.
    Returns scalar probability.
    """
    x_01  = ((x_t + 1) / 2).clamp(0, 1)
    
    # Converts to 3 channels because C0 was trained on ImageNet-pretrained backbone, which expects 3-channel input.
    x_3ch = x_01.repeat(1, 3, 1, 1)
    
    # Applies the same mean and std normalization as used during C0 training (ImageNet stats).
    mean  = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
    std   = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)
    
    with torch.no_grad():
        # Runs the classifier and applies sigmoid to get probability.
        logit = classifier((x_3ch - mean) / std).squeeze()
    return torch.sigmoid(logit).item()


def segment_image(x_t, seg_model):
    """
    Run segmentation on a tensor in [-1,1].
    TorchXRayVision PSPNet expects [-1024, 1024].
    Returns binary masks: numpy (num_classes, H, W).
    """
    # The segmentation model was trained on images in the range [-1024, 1024], so we need to rescale our [-1, 1] tensor accordingly.
    x_xrv = ((x_t + 1) / 2).clamp(0, 1) * 2048.0 - 1024.0
    
    with torch.no_grad():
        # Output is (1, num_classes, H_out, W_out) with raw logits. We apply sigmoid to get probabilities.
        seg_out = seg_model(x_xrv.cpu())
    seg_out = torch.sigmoid(seg_out)

    # Resize to match input image (224x224)
    seg_out = F.interpolate(seg_out, size=(224, 224), mode='bilinear', align_corners=False)

    # Convert to binary masks with threshold 0.5, and move to CPU numpy for plotting.
    seg_out = seg_out.cpu().numpy()[0]
    return (seg_out >= 0.5).astype(np.uint8)

# ══════════════════════════════════════════════════════════════════════════════
# DIFFUSION + GUIDANCE
# ══════════════════════════════════════════════════════════════════════════════

def one_guided_step(unet, sd, x_t, t_val, classifier, target_class):
    """
    One step of classifier-guided DDPM reverse diffusion.

    At each step we:
      1. Predict the noise with the UNet
      2. Estimate x0_hat (clean image) from the noisy x_t
      3. Compute gradient of log p(y | x0_hat) w.r.t. x_t
      4. Apply the standard DDPM reverse update + guidance nudge

    The gradient tells us: in which direction should x_t move
    to make the denoised estimate look more like target_class?
    """
    t_tensor       = torch.ones(1, dtype=torch.long, device=DEVICE) * t_val
    beta_t         = get(sd.beta.to(DEVICE), t_tensor)
    one_by_sqrt_at = get(sd.one_by_sqrt_alpha.to(DEVICE), t_tensor)
    sqrt_abar      = get(sd.sqrt_alpha_cumulative.to(DEVICE), t_tensor)
    sqrt_1mabar    = get(sd.sqrt_one_minus_alpha_cumulative.to(DEVICE), t_tensor)

    # Enable gradients on x_t for classifier guidance
    x_t_grad = x_t.detach().requires_grad_(True)
    
    # Predict the noise at this timestep with the UNet
    eps_pred  = unet(x_t_grad, t_tensor)
    
    # Estimate x0_hat (the clean image) from the noisy x_t and predicted noise.
    # "What image do we think this noisy sample came from?"
    x0_hat    = ((x_t_grad - sqrt_1mabar * eps_pred) / sqrt_abar).clamp(-1, 1)

    # Classifier on x0_hat (3-channel, ImageNet-normalised)
    x0_hat_3ch = ((x0_hat + 1) / 2).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)
    
    # Get the logit for the target class (0 or 1)
    logit = classifier((x0_hat_3ch - mean) / std).squeeze()

    # Log probability depending on the target class
    log_prob = (torch.log(torch.sigmoid(logit) + 1e-8) if target_class == 1
                else torch.log(1 - torch.sigmoid(logit) + 1e-8))

    # Calculate gradient
    # "How should we change the image x_t to increase the probability of the target class according to the classifier?"
    grad = torch.autograd.grad(log_prob, x_t_grad)[0]

    # DDPM reverse step (no grad needed here) + guidance nudge
    z = torch.randn_like(x_t) if t_val > 1 else torch.zeros_like(x_t)
    with torch.no_grad():
        # We use the original UNet prediction (without gradients) for the reverse step, to keep the noise prediction consistent with what the model was trained on.
        eps_no_grad = unet(x_t.detach(), t_tensor)

    # Moves x_t towards the predicted x0_hat while adding some noise. 
    # The guidance term (grad) nudges this update in the direction that increases the target class probability.
    # Guidance weight controls how strong this nudge is. Too high and it can overpower the diffusion process, too low and it might not achieve the desired class change.
    x_prev = (
        one_by_sqrt_at * (x_t.detach() - (beta_t / sqrt_1mabar) * eps_no_grad)
        + GUIDANCE_WEIGHT * grad.detach()
        + torch.sqrt(beta_t) * z
    )
    return x_prev.detach()


def generate_cf(unet, sd, x0, classifier, target_class, n_snapshots=N_SNAPSHOTS, track_probs=True):
    """
    Generate a counterfactual for x0.

    Steps:
      1. Corrupt x0 to x_{T_START} via the forward (noising) process
      2. Run guided reverse diffusion back to x0

    Returns
    -------
    x_cf         : CF image tensor in [-1,1]
    cf_prob      : C0 probability on CF
    intermediates: list of (t_val, tensor) at evenly spaced timesteps
    """
    
    # Add noise to x0 to get the starting point for reverse diffusion. 
    # T_START controls how noisy the initial image is: higher means more noise and more freedom for the model to change the image, but also less identity preservation.
    t_tensor = torch.tensor([T_START], dtype=torch.long, device=DEVICE)
    x, _     = sd.forward_diffusion(x0, t_tensor)

    # Save intermediate images at evenly spaced timesteps during the reverse process, to see how the image evolves.
    snapshot_ts  = set(np.linspace(T_START, 1, n_snapshots, dtype=int))
    intermediates = []
    intermediate_probs = []  # To store C0 probabilities at intermediate steps for later plotting

    # Reverse loop from T_START down to 1 (0 is the final clean image, but we stop at 1 to avoid extra noise addition in the last step).
    for t_val in tqdm(reversed(range(1, T_START + 1)), total=T_START, desc="Guided reverse"):
        # At each step, we perform one guided reverse diffusion step, which updates x_t to x_{t-1} while nudging it towards the target class according to the classifier gradient.
        x = one_guided_step(unet, sd, x, t_val, classifier, target_class)
        # Save C0 probability at this intermediate step for later plotting
        if track_probs:
            intermediate_prob = get_c0_prob(x, classifier)
            intermediate_probs.append((t_val, intermediate_prob))
        if t_val in snapshot_ts:
            intermediates.append((t_val, x.detach().clone()))


    # Get C0's probability on the final CF image, to see if we've successfully flipped the prediction.
    # CF is also an important input i the training of some configs of C2.
    cf_prob = get_c0_prob(x, classifier)
    return x, cf_prob, intermediates, intermediate_probs

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_cf_with_intermediates(x0, intermediates, x_cf, cf_prob, sample, seg_model, save_path):
    """
    One row of images: original | t=... snapshots | CF
    Each image has a segmentation overlay for Left Lung, Right Lung, Heart.
    """
    class_indices = [seg_model.targets.index(c)
                     for c in CLASSES_TO_SHOW if c in seg_model.targets]

    num_classes = len(class_indices)
    cmap = cm.get_cmap('tab10', num_classes)

    colours = [(*cmap(i)[:3], 0.35) for i in range(num_classes)]

    all_frames = (
        [("Original", x0)] +
        [(f"t={t}", xt) for t, xt in intermediates] +
        [(f"CF  p={cf_prob:.3f}", x_cf)]
    )

    fig, axes = plt.subplots(1, len(all_frames), figsize=(3 * len(all_frames), 4))

    for ax, (label, xt) in zip(axes, all_frames):
        img_np = ((xt[0, 0].cpu().numpy() + 1) / 2).clip(0, 1)
        
        # Show the original image in grayscale
        ax.imshow(img_np, cmap='gray')

        # Build RGBA overlay from segmentation masks
        # Each class gets a colour and a transparency
        seg     = segment_image(xt, seg_model)
        overlay = np.zeros((*img_np.shape, 4))
        for cls_idx, rgba in zip(class_indices, colours):
            mask = seg[cls_idx]
            for ch in range(3):
                overlay[..., ch] += mask * rgba[ch]
            overlay[..., 3] = np.clip(overlay[..., 3] + mask * rgba[3], 0, 1)

        ax.imshow(overlay)
        ax.set_title(label, fontsize=8)
        ax.axis('off')

    # Create a legend for the segmentation classes
    legend_patches = [
        mpatches.Patch(
            color=colours[i][:3],
            label=seg_model.targets[class_indices[i]]
        )
        for i in range(len(class_indices))
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=9)
    plt.suptitle(f"true={int(sample['effusion_true'])}  orig_pred={int(sample['effusion_pred'])}  orig_prob={sample['effusion_prob']:.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")

def plot_difference_map(x0, x_cf, ax=None, title="Difference Map", save_path=None):
    """
    Creates a high-contrast diverging difference map. 
    Red = Intensity Increase | Blue = Intensity Decrease.
    """
    # 1. Compute the raw difference
    # x_cf and x0 are in range [-1, 1], so diff is in [-2, 2]
    diff = (x_cf - x0).squeeze().detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # 2. Set symmetric bounds so 0 is exactly white
    # We use a robust max to avoid being skewed by a single outlier pixel
    vmax = np.percentile(np.abs(diff), 99.9) 
    
    # 'RdBu_r' gives the Red (positive) / Blue (negative) look from your image
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    # 3. Aesthetics
    ax.set_title(title, fontsize=24, pad=15)
    ax.axis('off')

    # 4. Professional Colorbar Alignment
    # This replaces the broken 'umbilical' import logic
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved → {save_path}")
    return ax


def plot_difference_triptych(
    x0,
    x_cf,
    save_path=None,
    overlay_diff_on_original=True
):
    """
    Plots:
        1. Original image
        2. Counterfactual image
        3. Difference map (CF - Original)

    Improvements included:
        - Uses [0,1] space for interpretability
        - Robust symmetric scaling
        - Optional anatomical overlay of difference map
        - Clean figure layout for thesis figures
    """

    # ─────────────────────────────────────────────
    # Convert to [0,1] for interpretability
    # ─────────────────────────────────────────────
    x0_np  = ((x0[0, 0].detach().cpu().numpy() + 1) / 2).clip(0, 1)
    xcf_np = ((x_cf[0, 0].detach().cpu().numpy() + 1) / 2).clip(0, 1)

    # ─────────────────────────────────────────────
    # Difference computed in image space (IMPORTANT CHANGE)
    # ─────────────────────────────────────────────
    diff = xcf_np - x0_np

    # Robust scaling (avoids single-pixel domination)
    vmax = np.percentile(np.abs(diff), 99.9) + 1e-8

    # ─────────────────────────────────────────────
    # Figure layout
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── 1. Original ───────────────────────────────
    axes[0].imshow(x0_np, cmap="gray")
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")

    # ── 2. Counterfactual ─────────────────────────
    axes[1].imshow(xcf_np, cmap="gray")
    axes[1].set_title("Counterfactual", fontsize=14)
    axes[1].axis("off")

    # ── 3. Difference map ──────────────────────────
    if overlay_diff_on_original:
        # More interpretable: show anatomical context
        axes[2].imshow(x0_np, cmap="gray", alpha=0.6)

    im = axes[2].imshow(
        diff,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.7 if overlay_diff_on_original else 1.0
    )

    axes[2].set_title("Difference Map (CF - Original)", fontsize=14)
    axes[2].axis("off")

    # ─────────────────────────────────────────────
    # Colorbar aligned to diff panel
    # ─────────────────────────────────────────────
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Intensity change", fontsize=10)

    plt.tight_layout()

    # ─────────────────────────────────────────────
    # Save safely
    # ─────────────────────────────────────────────
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {save_path}")
    else:
        plt.show()

    return fig

def plot_with_simulated_cf(x0, x_cf_diffusion, cf_prob_diffusion, sim_cf_path, sample, seg_model, classifier, save_path):
    """
    3-panel plot: Original | Diffusion CF | Simulated CF (nearest neighbour)
    """
    # Load the simulated CF image (first path if pipe-delimited)
    sim_path = sim_cf_path.split("|")[0]
    x_sim_cf = load_image(sim_path)
    sim_prob  = get_c0_prob(x_sim_cf, classifier)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        (x0,             f"Original\np={sample['effusion_prob']:.3f}"),
        (x_cf_diffusion, f"Diffusion CF\np={cf_prob_diffusion:.3f}"),
        (x_sim_cf,       f"Simulated CF (NN)\np={sim_prob:.3f}"),
    ]

    for ax, (img_t, title) in zip(axes, panels):
        img_np = ((img_t[0, 0].cpu().numpy() + 1) / 2).clip(0, 1)
        ax.imshow(img_np, cmap='gray')
        ax.set_title(title, fontsize=13)
        ax.axis('off')

    plt.suptitle(
        f"true={int(sample['effusion_true'])}  pred={int(sample['effusion_pred'])}",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    print("Parameter values:")
    print(f"  T_START: {T_START}")
    print(f"  GUIDANCE_WEIGHT: {GUIDANCE_WEIGHT}")

    # ── Load models ───────────────────────────────────────────────────────
    unet, sd, classifier, seg_model = load_models()

    df = pd.read_csv(FOLD_CSV)

    REQUIRED_SAMPLES = 20

    # ── Sample positives / negatives from the fold predictions CSV , half should be correct and half incorrect according to C0, to see how the diffusion CF behaves in different scenarios.
    pos_df = df[df['effusion_pred'] == 1]
    neg_df = df[df['effusion_pred'] == 0]
    
    # We want a mix of true positives, false positives, true negatives, and false negatives.
    correct_pos = pos_df[pos_df['correct'] == 1]
    incorrect_pos = pos_df[pos_df['correct'] == 0]
    correct_neg = neg_df[neg_df['correct'] == 1]
    incorrect_neg = neg_df[neg_df['correct'] == 0]

    correct_pos_samples = correct_pos.sample(
        n=min(REQUIRED_SAMPLES, len(pos_df)),
        random_state=42
    )
    incorrect_pos_samples = incorrect_pos.sample(
        n=min(REQUIRED_SAMPLES, len(pos_df)),
        random_state=42
    )
    correct_neg_samples = correct_neg.sample(
        n=min(REQUIRED_SAMPLES, len(neg_df)),
        random_state=42
    )
    incorrect_neg_samples = incorrect_neg.sample(
        n=min(REQUIRED_SAMPLES, len(neg_df)),
        random_state=42
    )

    # Combine + convert to list of dicts (IMPORTANT FIX)
    samples = pd.concat([correct_pos_samples, incorrect_pos_samples, correct_neg_samples, incorrect_neg_samples]) \
                .reset_index(drop=True) \
                .to_dict(orient="records")

    print(f"\nLoaded {len(samples)} samples")
    for s in samples:
        print(f"  pred={int(s['effusion_pred'])} correct = {int(s['correct'])} | true={int(s['effusion_true'])} "
              f"prob={s['effusion_prob']:.4f}  {s['path']}")

    # ── Generate CFs ──────────────────────────────────────────────────────
    results = []

    for i, sample in enumerate(samples):
        print(f"\n── Sample {i+1}: pred={int(sample['effusion_pred'])} true={int(sample['effusion_true'])} ──")

        x0 = load_image(sample['path'])
        print(f"Original C0 probability: {sample['effusion_prob']:.4f}")
        target_class = 1 - int(sample['effusion_pred'])

        print("Generating counterfactual...")
        x_cf, cf_prob, intermediates, intermediate_probs = generate_cf(
            unet, sd, x0, classifier, target_class
        )

        flipped = (sample['effusion_prob'] < C0_THRESHOLD) != (cf_prob < C0_THRESHOLD)
        print(f"Counterfactual C0 probability: {cf_prob:.4f}")
        print(f"  orig={sample['effusion_prob']:.4f} → CF={cf_prob:.4f}  flip={flipped}")

        results.append({
            'original_path':  sample['path'],
            'original_prob':  sample['effusion_prob'],
            'cf_prob':        cf_prob,
            'true_label':     int(sample['effusion_true']),
            'original_pred':  int(sample['effusion_pred']),
            'correct':        int(sample['correct']),
            'cf_pred':        int(cf_prob >= C0_THRESHOLD),
            'flip_achieved':  flipped,
            'intermediate_probs': intermediate_probs,
        })

        # ── Plot CF evolution ────────────────────────────────────────py─────
        plot_cf_with_intermediates(
            x0, intermediates, x_cf, cf_prob, sample, seg_model,
            save_path=os.path.join(
                OUTPUT_PATH,
                f"cf_{i}_pred{int(sample['effusion_pred'])}_true{int(sample['effusion_true'])}.png"
            )
        )

        # ── Difference maps ───────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(((x0[0,0].cpu().numpy() + 1) / 2), cmap='gray')
        axes[0].axis('off')

        plot_difference_map(
            x0, x_cf,
            ax=axes[1],
            save_path=os.path.join(OUTPUT_PATH, f"diff_map_{i}.png")
        )

        plot_difference_triptych(
            x0, x_cf,
            save_path=os.path.join(OUTPUT_PATH, f"diff_triptych_{i}.png")
        )

        plt.savefig(os.path.join(OUTPUT_PATH, f"diff_analysis_{i}.png"))

        # ── Simulated CF comparison ───────────────────────────────────────
        plot_with_simulated_cf(
            x0, x_cf, cf_prob, sample['cf_paths'], sample,
            seg_model, classifier,
            save_path=os.path.join(
                OUTPUT_PATH,
                f"comparison_{i}_pred{int(sample['effusion_pred'])}_true{int(sample['effusion_true'])}.png"
            )
        )

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_PATH, "cf_results.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")


if __name__ == "__main__":
    main()

