"""
cf_generation_final.py
======================
FastDiME counterfactual generation against the **final_experiment** C0.

Replaces `code/cf/cf_generation.py` for anything under final_experiment/, and
replaces the from-the-paper reimplementation that lived in
`cf_v3_fastdime_replication.ipynb`.

Faithfulness to the reference
-----------------------------
Settings below mirror `FastDiME_Med/CFgenerating/config_cf.py` (the authors'
CheXpert config). Where the notebook disagreed with it:

    | quantity            | notebook      | config_cf.py (authoritative) |
    |---------------------|---------------|------------------------------|
    | schedule            | 400 respaced  | 1000, rescale_t = False      |
    | tau (cf_from_timestep) | 160        | 400                          |
    | inpaint warm-up     | tau/2 = 80    | start_tau = 200, an absolute |
    |                     |               | timestep ("t <= 200")        |
    | gradient scale      | absent        | start_scale = 100            |
    | distance loss       | L1 on x_t_0,  | Denoised_L1 on the denoised  |
    |                     | weight 50     | estimate, weight 200         |
    | weight 50 on plain L1 | used        | that is the **DiME** setting |
    | mean update         | mu - Sigma*g  | mu - beta_t*g*scale/sqrt(alpha_t) |
    | sample std          | sqrt(beta~)   | sqrt(beta_t)                 |
    | background re-anchor| q_sample(t-1) | q_sample(t)                  |
    | returned CF         | x_{t-1}       | the last denoised x_t_0      |

The one deviation that cannot be removed: their classifier is a
timestep-conditioned UNet encoder (`ts_guided = True`), so their
`GuidedDiffusionTS` passes `t` into it. Ours is a plain DenseNet with no
timestep input, so we use the base `GuidedDiffusion`. This is not a loss of
fidelity in the loss itself -- with `grad_obt_from='denoised_img'`,
`GuidedDiffusionTS.get_guided_gradient` clones the *same* denoised tensor twice
and sums the two gradients, so their ['Lc', 'Denoised_L1', 'L1'] @
[1.0, 200.0, 0.0] is arithmetically identical to base ['Lc', 'L1'] @
[1.0, 200.0]. That equivalence is what FASTDIME_PRESETS encodes.
"""

import contextlib
import io
import json
import os
import sys
from unittest.mock import MagicMock

for _mod in ['pytorch_lightning', 'pytorch_lightning.core',
             'pytorch_lightning.core.lightning', 'pytorch_lightning.core.module',
             'torchaudio', 'models.base_classifier', 'models.resnet']:
    sys.modules[_mod] = MagicMock()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from PIL import Image
from torchvision import models

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "FastDiME_Med"))

from models.unet import UNet                                  # noqa: E402
from models.diffusion import GuidedDiffusion                  # noqa: E402
from CFgenerating.utils import get_fixed_mask                 # noqa: E402


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")

C0_MODEL_DIR   = os.path.join(RESULTS_DIR, "C0_final", "multilabel_ignore")
C2_DIR         = os.path.join(RESULTS_DIR, "C2_final")
DIFFUSION_CKPT = os.path.join(
    REPO, "FastDiME_Med", "pretrained_models", "diffusion",
    "OUT_CHEXPERT_CardioSplit", "ckpt.tar")

# Same mapping c0_final_predictions.py uses: directory name -> labels.json entry.
DISEASE_COL = {
    "pneumothorax":  "Pneumothorax",
    "effusion":      "Pleural Effusion",
    "cardiomegaly":  "Cardiomegaly",
    "atelectasis":   "Atelectasis",
    "consolidation": "Consolidation",
    "edema":         "Edema",
    "pneumonia":     "Pneumonia",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── FastDiME hyperparameters, from CFgenerating/config_cf.py ──────────────────

TOTAL_TIMESTEPS  = 1000   # must match the diffusion checkpoint
CF_FROM_TIMESTEP = 50    # tau: where the reverse process starts
START_TAU        = 25    # inpainting engages once timestep <= this
DILATION         = 21
INPAINT_TH       = 0.15
START_SCALE      = 100    # gradient scale; absent from the notebook entirely
W_LC             = 1.0
W_DENOISED_L1    = 10 # FastDiME
W_L1_DIME        = 50.0   # DiME uses this on the fully generated image instead

# Their CheXpert pacemaker-removal row, for reference in reporting.
FASTDIME_REF = dict(l1=0.0897, mad=0.7554, fid=61.40)

# Each preset is the base-GuidedDiffusion equivalent of one `model_var` in
# config_cf.py. See the module docstring for why ['Lc','Denoised_L1','L1'] @
# [1, 200, 0] collapses to ['Lc','L1'] @ [1, 200] here.
FASTDIME_PRESETS = {
    "FastDiME": dict(
        grad_obt_from="denoised_img", grad_loss_types=["Lc", "L1"],
        grad_loss_weights=[W_LC, W_DENOISED_L1],
        use_inpaint=True, inpaint_type="dynamic"),
    "FastDiME-woM": dict(
        grad_obt_from="denoised_img", grad_loss_types=["Lc", "L1"],
        grad_loss_weights=[W_LC, W_DENOISED_L1],
        use_inpaint=False, inpaint_type=None),
    # DiME takes the gradient on a fully generated image: an inner reverse loop
    # per step, so O(T^2). Kept for completeness; do not run it over a set.
    "DiME": dict(
        grad_obt_from="generated_img", grad_loss_types=["Lc", "L1"],
        grad_loss_weights=[W_LC, W_L1_DIME],
        use_inpaint=False, inpaint_type=None),
    # Not a FastDiME variant: the sampler with guidance switched off, so you can
    # tell an edit from the sampler's own drift.
    "noise floor": dict(
        grad_obt_from="denoised_img", grad_loss_types=["Lc", "L1"],
        grad_loss_weights=[0.0, 0.0],
        use_inpaint=False, inpaint_type=None),
}


# ── Task configuration ────────────────────────────────────────────────────────

class TaskConfig:
    """Everything that depends on which disease is being counterfactualled."""

    def __init__(self, disease, threshold_file="threshold.txt", device=DEVICE):
        if disease not in DISEASE_COL:
            raise ValueError(
                f"unknown disease {disease!r}; expected one of {sorted(DISEASE_COL)}")
        self.disease = disease
        self.device = device
        self.target_col = DISEASE_COL[disease]

        self.model_dir = C0_MODEL_DIR
        self.model_path = os.path.join(self.model_dir, "c0_best.pt")
        self.c0_dir = os.path.join(self.model_dir, disease)
        self.c2_dir = os.path.join(C2_DIR, disease)
        self.c2_csv = os.path.join(self.c2_dir, "c2_data.csv")

        with open(os.path.join(self.model_dir, "labels.json")) as f:
            self.spec = json.load(f)
        self.labels = self.spec["labels"]
        if self.target_col not in self.labels:
            raise ValueError(
                f"{self.target_col!r} is not one of the {len(self.labels)} trained labels")
        self.target_idx = self.labels.index(self.target_col)

        for p in (self.model_path, self.c0_dir, self.c2_csv):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} missing for disease={disease!r}. Run "
                    f"final_experiment/code/c0_final_predictions.py --disease {disease} "
                    f"and c2_build_dataset.py first.")

        # Youden's J on C0_checkpoint_selection. This is the threshold that
        # produced pred/correct in c2_data.csv, so anything counting flips must
        # use the same number. threshold_fixed.txt (0.5) is for reporting only.
        self.threshold_file = threshold_file
        with open(os.path.join(self.c0_dir, threshold_file)) as f:
            self.threshold = float(f.read().strip())

    def split_csv(self, split="C2_dataset"):
        """Per-disease C0 predictions, e.g. C2_dataset_c0_cardiomegaly.csv."""
        return os.path.join(self.c0_dir, f"{split}_c0_{self.disease}.csv")

    def __repr__(self):
        return (f"TaskConfig(disease={self.disease!r}, target_col={self.target_col!r}, "
                f"logit {self.target_idx}/{len(self.labels)}, "
                f"threshold={self.threshold:.4f} from {self.threshold_file})")


# ── The classifier, adapted to the guidance interface ─────────────────────────

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class C0Guidance(nn.Module):
    """final_experiment C0, wrapped for FastDiME's guidance loop.

    `GuidedDiffusion.get_classifier_loss` calls `classifier.forward(x)` on a
    1-channel image in [-1, 1] and expects logits shaped [BS, 1] to go straight
    into `binary_cross_entropy(sigmoid(logits), y_target)`. Our C0 is a 14-logit
    ImageNet-normalised 3-channel DenseNet, so this does the conversion and
    selects the target disease's logit. Differentiable throughout -- the
    gradient wrt the input image is the whole point.
    """

    def __init__(self, backbone, target_idx):
        super().__init__()
        self.backbone = backbone
        self.target_idx = target_idx

    def _prep(self, x):
        x3 = ((x + 1) / 2).repeat(1, 3, 1, 1)
        return (x3 - _MEAN.to(x.device)) / _STD.to(x.device)

    def forward(self, x):
        return self.backbone(self._prep(x))[:, self.target_idx:self.target_idx + 1]

    def all_logits(self, x):
        """All 14 logits -- for checking a CF did not drag other labels along."""
        return self.backbone(self._prep(x))


def load_c0(cfg):
    backbone = models.densenet121()
    backbone.classifier = nn.Linear(backbone.classifier.in_features, len(cfg.labels))
    backbone.load_state_dict(torch.load(cfg.model_path, map_location="cpu"))
    model = C0Guidance(backbone, cfg.target_idx).eval().to(cfg.device)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"C0 loaded: {len(cfg.labels)}-logit multilabel, guiding on "
          f"{cfg.target_col!r} (index {cfg.target_idx}) | threshold {cfg.threshold:.4f}")
    return model


def load_models(disease=None, threshold_file="threshold.txt", device=DEVICE):
    """Returns (unet, classifier, seg_model, cfg).

    `disease` defaults to $THESIS_TASK. Nothing here is hardcoded to effusion:
    the disease selects both the guided logit and the operating threshold.

    Note the signature differs from `code/cf/cf_generation.load_models()`, which
    also returned a `SimpleDiffusion`. The schedule now lives inside the
    `GuidedDiffusion` that `make_sampler()` builds, because the guidance
    parameters and the schedule have to agree.
    """
    disease = disease or os.environ.get("THESIS_TASK", "cardiomegaly")
    cfg = TaskConfig(disease, threshold_file=threshold_file, device=device)
    print(cfg)

    unet = UNet(
        input_channels=1, output_channels=1,
        base_channels=64, base_channels_multiples=(1, 2, 4, 4),
        apply_attention=(False, False, False, False),
        dropout_rate=0.1, time_multiple=4,
    )
    unet.load_state_dict(torch.load(DIFFUSION_CKPT, map_location="cpu")["model"])
    unet.eval().to(device)
    for p in unet.parameters():
        p.requires_grad_(False)
    print(f"UNet loaded ({sum(p.numel() for p in unet.parameters())/1e6:.1f}M params)")

    seg = xrv.baseline_models.chestx_det.PSPNet()
    seg.eval().cpu()
    print(f"Segmentation model loaded ({len(seg.targets)} classes)")

    return unet, load_c0(cfg), seg, cfg


# ── Sampler ───────────────────────────────────────────────────────────────────

def make_sampler(mode="FastDiME", device=DEVICE, start_tau=START_TAU,
                 dilation=DILATION, inpaint_th=INPAINT_TH, fixed_mask=None):
    """A `GuidedDiffusion` configured as one of FASTDIME_PRESETS.

    This is the authors' class, unmodified -- every diffusion step, the mask and
    the inpainting all execute their code.
    """
    if mode not in FASTDIME_PRESETS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(FASTDIME_PRESETS)}")
    preset = dict(FASTDIME_PRESETS[mode])
    if fixed_mask is not None:
        preset.update(use_inpaint=True, inpaint_type="fix")
    return GuidedDiffusion(
        num_diffusion_timesteps=TOTAL_TIMESTEPS,
        img_shape=(1, 224, 224),
        device=device,
        start_tau=start_tau,
        dilation=dilation,
        inpaint_th=inpaint_th,
        fixed_mask=fixed_mask,
        **preset,
    )


def generate_cf(x0, target_class, unet, classifier, sd=None, mode="FastDiME",
                cf_from_timestep=None, scale=None,
                track_every=0, seed=0, device=DEVICE, progress=True, quiet=True):
    """One counterfactual, following `generate_cf.one_CF_procedure`.

    Differs from their driver only in not writing PNGs and not retaining all 400
    intermediates -- `p_sample_loop` appends every x_t_0 and x_{t-1} to a list,
    which is fine for a handful of images and not for a sweep. The per-step call
    is their `sd.p_sample_once`, unchanged.

    Returns (x_cf, cf_prob, boolmask, traj) where:
      - x_cf is the final **denoised estimate** x_t_0 at t=1, which is what
        `one_CF_procedure` takes as the counterfactual (`records[0][-1]`), not
        the noisy x_{t-1}.
      - boolmask is their convention: 1 = background restored from the original,
        so the *edited* region is `1 - boolmask`. None when the mode is unmasked.
      - traj is [(t, prob)] sampled every `track_every` steps (0 disables).

    `quiet` swallows stdout from inside the loop: their `p_sample_once` prints
    a line on every inpainted step, which is 200 lines per image at the default
    settings. tqdm writes to stderr, so the progress bar survives.
    """
    # Resolved at call time, not bound as a default, so that overriding the
    # module constant (e.g. a short smoke run) actually takes effect.
    cf_from_timestep = CF_FROM_TIMESTEP if cf_from_timestep is None else cf_from_timestep
    scale = START_SCALE if scale is None else scale

    sd = sd or make_sampler(mode, device=device)
    torch.manual_seed(seed)

    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)
    target = torch.full((x0.shape[0], 1), float(target_class), device=device)

    # x_tau, exactly as one_CF_procedure does it
    t_start = torch.as_tensor(cf_from_timestep, dtype=torch.long, device=device)
    x_t, _ = sd.forward_diffusion(x0, t_start)

    # p_sample_loop scales by batch size; keep that even though BS is 1 here
    this_scale = scale * x0.shape[0]

    steps = list(reversed(range(1, cf_from_timestep)))   # their loop bounds
    if progress:
        from tqdm.auto import tqdm
        steps = tqdm(steps, desc=f"{mode} t={cf_from_timestep}", leave=False)

    x_t_0, traj = None, []
    sink = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with sink:
        for t in steps:
            x_t_0, x_t, _max_grad, prob, _losses = sd.p_sample_once(
                unet, classifier, timestep=t, x_t=x_t, x_0=x0,
                target=target, scale=this_scale)
            if track_every and (t % track_every == 0 or t == 1):
                traj.append((t, float(prob.squeeze())))

    x_cf = x_t_0.detach()
    boolmask = None
    if sd.use_inpaint:
        boolmask = (sd.fixed_mask if sd.inpaint_type == "fix"
                    else get_fixed_mask(x0, x_cf, dilation=sd.dilation,
                                        inpaint_th=sd.inpaint_th))
    return x_cf, get_c0_prob(x_cf, classifier), boolmask, traj


def edited_mask(x0, x_cf, dilation=DILATION, inpaint_th=INPAINT_TH):
    """The region FastDiME treats as edited, as a float mask in {0,1}.

    Their `boolmask` (from `generate_mask` + `< inpaint_th`) marks the
    *background* to restore; this returns its complement, which is what you want
    when asking "how much of the image changed" or overlaying on a segmentation.
    """
    return 1.0 - get_fixed_mask(x0, x_cf, dilation=dilation, inpaint_th=inpaint_th)


# ── Image / probability helpers ───────────────────────────────────────────────

def load_image(rel_path, device=DEVICE, data_root=DATA_ROOT):
    """(1,1,224,224) in [-1,1]. BILINEAR, matching C0's transforms.Resize."""
    img = Image.open(os.path.join(data_root, rel_path)).convert("L")
    img = img.resize((224, 224), Image.BILINEAR)
    a = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
    return (t * 2.0 - 1.0).to(device)


def to_np(x):
    """[-1,1] tensor -> (H,W) float array in [0,1], for plotting and L1."""
    return ((x[0, 0].detach().cpu().numpy() + 1) / 2).clip(0, 1)


def c0_logit(x, classifier):
    """Differentiable target-disease logit for a [-1,1] image tensor."""
    return classifier(x).squeeze()


def get_c0_prob(x_t, classifier):
    """Target-disease probability for a [-1,1] tensor. Scalar float."""
    with torch.no_grad():
        return float(torch.sigmoid(classifier(x_t.clamp(-1, 1))).squeeze())


def get_all_probs(x_t, classifier, cfg):
    """All 14 label probabilities as a dict -- side-effect check on a CF."""
    with torch.no_grad():
        p = torch.sigmoid(classifier.all_logits(x_t.clamp(-1, 1)))[0].cpu().numpy()
    return dict(zip(cfg.labels, p.tolist()))


def predict(prob, cfg):
    """C0's decision at the operating threshold. Matches the strict `prob > t`
    in apply_threshold() in c0_final_predictions.py."""
    return int(prob > cfg.threshold)


def segment_image(x_t, seg_model):
    """Binary masks (num_classes, 224, 224). PSPNet expects [-1024, 1024]."""
    x_xrv = ((x_t + 1) / 2).clamp(0, 1) * 2048.0 - 1024.0
    with torch.no_grad():
        seg_out = torch.sigmoid(seg_model(x_xrv.cpu()))
    seg_out = F.interpolate(seg_out, size=(224, 224), mode="bilinear", align_corners=False)
    return (seg_out.cpu().numpy()[0] >= 0.5).astype(np.uint8)


# ── Metrics ───────────────────────────────────────────────────────────────────

def l1(x0, x_cf):
    """Mean absolute difference in [0,1] pixel space. State the space when
    reporting: the paper's 0.0897 does not specify one."""
    return float(np.abs(to_np(x_cf) - to_np(x0)).mean())


def mad(orig_probs, cf_probs):
    return float(np.mean(np.abs(np.asarray(orig_probs) - np.asarray(cf_probs))))


def compute_fid(orig_list, cf_list, device=DEVICE):
    """Needs >= ~200 per side to mean anything. torchmetrics[image]."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for xs, real in ((orig_list, True), (cf_list, False)):
        for x in xs:
            fid.update(((x.to(device) + 1) / 2).clamp(0, 1).repeat(1, 3, 1, 1), real=real)
    return float(fid.compute())
