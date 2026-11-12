"""Unconditional DDPM sampling with the FastDiME CheXpert UNet, saving x_t frames.

For the 'diffusion model' slide: pure noise -> realistic X-ray, real model output.
Plain ancestral DDPM (linear betas 1e-4..0.02, T=1000), no classifier guidance.
"""
import os, sys
import numpy as np
import torch
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "FastDiME_Med"))
from models.unet import UNet  # noqa: E402

OUT = os.path.join(REPO, "final_experiment", "figures", "slides", "ddpm_steps")
CKPT = os.path.join(REPO, "FastDiME_Med", "pretrained_models", "diffusion",
                    "OUT_CHEXPERT_CardioSplit", "ckpt.tar")
N, T, SEED = int(os.environ.get("N_SAMPLES", 16)), 1000, 0
SAVE_T = [999, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 0]
dev = "cuda" if torch.cuda.is_available() else "cpu"

unet = UNet(input_channels=1, output_channels=1, base_channels=64,
            base_channels_multiples=(1, 2, 4, 4), apply_attention=(False,) * 4,
            dropout_rate=0.1, time_multiple=4)
unet.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
unet.eval().to(dev)

beta = torch.linspace(1e-4, 0.02, T, device=dev)
alpha = 1 - beta
abar = torch.cumprod(alpha, 0)

def to_u8(x):
    return ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte().cpu().numpy()

g = torch.Generator(device=dev).manual_seed(SEED)
x = torch.randn(N, 1, 224, 224, device=dev, generator=g)
xt, x0hat = {}, {}
with torch.no_grad():
    for t in range(T - 1, -1, -1):
        ts = torch.full((N,), t, device=dev, dtype=torch.long)
        if t in SAVE_T:
            xt[t] = to_u8(x)[:, 0]
        eps = unet(x, ts)
        if t in SAVE_T:
            x0hat[t] = to_u8((x - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt())[:, 0]
        mean = (x - beta[t] / (1 - abar[t]).sqrt() * eps) / alpha[t].sqrt()
        x = mean + (beta[t].sqrt() * torch.randn(x.shape, device=dev, generator=g) if t > 0 else 0)
        if t % 100 == 0:
            print("t", t, flush=True)
final = to_u8(x)[:, 0]

np.savez_compressed(os.path.join(OUT, "ddpm_steps.npz"), final=final,
                    save_t=np.array(SAVE_T),
                    xt=np.stack([xt[t] for t in SAVE_T], 1),
                    x0hat=np.stack([x0hat[t] for t in SAVE_T], 1))
for i in range(N):
    d = os.path.join(OUT, f"sample{i:02d}")
    os.makedirs(d, exist_ok=True)
    for t in SAVE_T:
        Image.fromarray(xt[t][i]).save(os.path.join(d, f"xt_t{t:03d}.png"))
        Image.fromarray(x0hat[t][i]).save(os.path.join(d, f"x0hat_t{t:03d}.png"))
    Image.fromarray(final[i]).save(os.path.join(d, "final.png"))
# contact sheet of finals to pick a sample
grid = np.concatenate([np.concatenate(list(final[r*4:(r+1)*4]), 1) for r in range(N // 4)], 0)
Image.fromarray(grid).save(os.path.join(OUT, "finals_grid.png"))
print("done ->", OUT)
