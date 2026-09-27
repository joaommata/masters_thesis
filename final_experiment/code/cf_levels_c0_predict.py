"""
cf_levels_c0_predict.py
=======================
Run C0 over the exported counterfactual PNGs and write, for each one, the three
things the C2 pipeline dereferences from a counterfactual path:

    prob          C0's probability for the target disease
    emb_0..1023   the 1024-d penultimate embedding   -> cf_emb_* / the emb configs
    cam_pos/neg   the two-directional Grad-CAM       -> the *_sal CNN configs

Recompute, do not repackage
---------------------------
Both numbers already exist in some form beside the generation run --
manifest.csv has a cf_prob and saliency/<key>.npz has the maps. Neither is used
here, on purpose:

  * those describe the float32 tensor the sampler produced. Every model
    downstream sees a uint8 PNG opened by PIL. The gap is small, but it is a gap
    between what `cf_prob` says and what `cf_emb_*` and the CNN actually see,
    and it would be invisible;
  * `build_folds` asserts that the mean of the per-CF probabilities equals the
    cf_prob the source reported, to 1e-6. That assert is only worth anything if
    both sides come from this one table;
  * atelectasis's saliency/ was written in the older positive-only CAM format.
    Recomputing here makes that irrelevant to the pipeline rather than a
    blocking dependency -- cf_regen_saliency.py only matters for figures.

Everything that defines "what C0 sees" is kept verbatim from
c0_final_predictions.py: the DenseNet121 head, the hook on model.features, the
Resize(224)+ToTensor+ImageNet-Normalize transform, the signed-CAM split into
relu(+)/relu(-) with per-map max normalisation, and the .npz key names. That
script cannot be imported -- it parses argv at module scope -- so the pieces are
copied; --verify-originals is the guard that they were copied correctly.

Outputs
-------
    CF_levels/<disease>/cf_c0.parquet
        cf_path, prob, emb_0..emb_1023
    C0_final/multilabel_<policy>/<disease>/gradcam/CF_levels/<flat cf_path>.npz
        cam_pos, cam_neg, scale_pos, scale_neg, cam, orient_t, sign

The gradcam subdirectory name is what c2_train_images.CAM_SPLITS looks for, and
the filename is cf_path.replace("/", "_") + ".npz", which is what _load_cam
reconstructs.

Usage
-----
    python cf_levels_c0_predict.py --disease atelectasis
    python cf_levels_c0_predict.py --disease atelectasis --verify-originals 200
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import c2_data_path                                  # noqa: E402

RESULTS = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
DATA_ROOT = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")

# Verbatim from c0_final_predictions.py -- the label column each disease's logit
# corresponds to in the 14-label head.
DISEASE_COL = {
    "pneumothorax": "Pneumothorax",
    "effusion": "Pleural Effusion",
    "cardiomegaly": "Cardiomegaly",
    "atelectasis": "Atelectasis",
    "consolidation": "Consolidation",
    "edema": "Edema",
    "pneumonia": "Pneumonia",
}

# Verbatim from c0_final_predictions.py. If this diverges, every probability and
# every embedding in the CF table describes a differently-preprocessed image
# from the query rows they are subtracted from.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class PathDataset(Dataset):
    """Yields (image, path) for a list of DATA_ROOT-relative paths."""

    def __init__(self, paths, data_dir, transform):
        self.paths = list(paths)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(os.path.join(self.data_dir, p)).convert("RGB")
        return self.transform(img), p


cache = {}


def hook_fn(module, inp, out):
    """Verbatim from c0_final_predictions.py. `out` must NOT be detached here."""
    acts = out
    cache["acts"] = torch.relu(acts).detach()
    cache["emb"] = cache["acts"].mean(dim=(2, 3)).detach()

    def save_grad(grad):
        cache["grads"] = grad.detach()

    if acts.requires_grad:
        acts.register_hook(save_grad)


def load_c0(model_dir, labels, device):
    model = models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, len(labels))
    model.load_state_dict(torch.load(os.path.join(model_dir, "c0_best.pt"),
                                     map_location=device))
    model = model.to(device).eval()
    model.features.register_forward_hook(hook_fn)
    return model


def run(model, loader, target_idx, device, cam_dir, cam_threshold, write_cam):
    """Probability + embedding (+ Grad-CAM) per image. Mirrors run_inference()."""
    if write_cam:
        os.makedirs(cam_dir, exist_ok=True)

    recs = []
    for imgs, paths in tqdm(loader, desc="C0"):
        imgs = imgs.to(device)
        cache.clear()
        model.zero_grad(set_to_none=True)

        if not write_cam:
            with torch.no_grad():
                logits = model(imgs)
            cams_pos = cams_neg = None
        else:
            logits = model(imgs)
            # Backprop the target logit only, so each sample's gradient is its
            # own disease's and never mixed across the 14 heads.
            tl = logits[:, target_idx]
            tl.backward(torch.ones_like(tl))
            acts, grads = cache["acts"], cache["grads"]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam_signed = (weights * acts).sum(dim=1)                    # (B,7,7)
            cams_pos = torch.relu(cam_signed).cpu().numpy()
            cams_neg = torch.relu(-cam_signed).cpu().numpy()

        probs = torch.sigmoid(logits)[:, target_idx].detach().cpu().numpy()
        emb = cache["emb"].cpu().numpy()

        for i, p in enumerate(paths):
            if write_cam:
                # Each direction scaled by ITS OWN max; scaling them jointly lets
                # the dominant one flatten the other into invisibility, and on
                # most images one direction does dominate.
                cp, cn = cams_pos[i], cams_neg[i]
                s_pos, s_neg = float(cp.max()), float(cn.max())
                cp = cp / (s_pos + 1e-8)
                cn = cn / (s_neg + 1e-8)
                sign = 1.0 if float(probs[i]) > cam_threshold else -1.0
                net = cp - cn if sign > 0 else cn - cp
                net = (net - net.min()) / (net.max() - net.min() + 1e-8)
                dest = os.path.join(cam_dir, p.replace("/", "_") + ".npz")
                # The tmp name has to end in .npz: savez_compressed silently
                # appends the extension otherwise, and os.replace would then
                # move a file that does not exist.
                tmp = f"{dest}.tmp{os.getpid()}.npz"
                np.savez_compressed(tmp,
                                    cam_pos=cp.astype(np.float32),
                                    cam_neg=cn.astype(np.float32),
                                    scale_pos=np.float32(s_pos),
                                    scale_neg=np.float32(s_neg),
                                    cam=net.astype(np.float32),
                                    orient_t=np.float32(cam_threshold),
                                    sign=np.float32(sign))
                os.replace(tmp, dest)

            rec = {"cf_path": p, "prob": float(probs[i])}
            rec.update({f"emb_{j}": float(v) for j, v in enumerate(emb[i])})
            recs.append(rec)

    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True, choices=list(DISEASE_COL))
    ap.add_argument("--policy", default="ignore")
    ap.add_argument("--levels", nargs="+", default=None,
                    help="restrict to these level names (default: all in cf_index.csv)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-cam", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify-originals", type=int, default=0, metavar="N",
                    help="also score N real C2 images and compare against "
                         "c2_data.csv. This is the proof that the model, the "
                         "transform and the head index copied here are the ones "
                         "that produced the baseline. Costs seconds; run it once "
                         "per disease.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(RESULTS, "C0_final", f"multilabel_{args.policy}")
    out_dir = os.path.join(model_dir, args.disease)
    with open(os.path.join(model_dir, "labels.json")) as fh:
        labels = json.load(fh)["labels"]
    target_idx = labels.index(DISEASE_COL[args.disease])
    with open(os.path.join(out_dir, "threshold.txt")) as fh:
        threshold = float(fh.read().strip())

    root = os.path.join(RESULTS, "CF_levels", args.disease)
    index = pd.read_csv(os.path.join(root, "cf_index.csv"))
    if args.levels:
        index = index[index["level_name"].isin(args.levels)]
    if args.limit:
        index = index.iloc[:args.limit]

    cam_dir = os.path.join(out_dir, "gradcam", "CF_levels")

    print(f"disease    {args.disease}  (logit {target_idx} = {DISEASE_COL[args.disease]!r})")
    print(f"threshold  {threshold:.6f}")
    print(f"device     {device}")
    print(f"images     {len(index):,}")
    print(f"grad-cam   {'DISABLED' if args.no_cam else cam_dir}")

    model = load_c0(model_dir, labels, device)

    # ── the guard: does this script reproduce the baseline on real images? ────
    if args.verify_originals:
        c2 = pd.read_csv(c2_data_path(args.disease), usecols=["path", "prob"])
        sample = c2.sample(args.verify_originals, random_state=0)
        ld = DataLoader(PathDataset(sample["path"], DATA_ROOT, TRANSFORM),
                        batch_size=args.batch_size, num_workers=args.num_workers)
        got = run(model, ld, target_idx, device, cam_dir, threshold, False)
        merged = got.merge(sample.rename(columns={"path": "cf_path",
                                                  "prob": "prob_baseline"}),
                           on="cf_path")
        sd = merged["prob"] - merged["prob_baseline"]
        d = sd.abs()
        # Tolerance 2e-3, and it is NOT device-dependent. Measured: re-scoring
        # the same images with the same weights reproduces the baseline to
        # ~5e-4 on CPU and ~5e-4 on a V100. float32 convolution is not
        # associative, and cuDNN picks its algorithm from the batch shape and
        # the card, so the baseline run and this one accumulate in a different
        # order. A device-conditional tolerance was tried first and was simply
        # wrong: it assumed same-device meant bit-identical, and killed two
        # jobs proving otherwise.
        #
        # The real guard is the line below it. A different transform, model or
        # logit index moves probabilities by 1e-2 or more and moves them
        # SYSTEMATICALLY, so |signed mean| >> sd. Reassociation noise has
        # |signed mean| < sd. That test is what would actually catch a divergence;
        # the magnitude threshold only catches a catastrophe.
        tol = 2e-3
        print(f"\n--- verify-originals ({len(merged)} real images, {device.type}) ---")
        print(f"  (the load-bearing test is |signed mean| vs sd, not the magnitude)")
        print(f"  max |delta| vs c2_data.csv : {d.max():.3e}   (tolerance {tol:.0e})")
        print(f"  mean|delta|                : {d.mean():.3e}")
        print(f"  signed mean / sd           : {sd.mean():+.3e} / {sd.std():.3e}")
        if d.max() > tol:
            raise SystemExit(
                f"this script does not reproduce C0's baseline probabilities "
                f"(max delta {d.max():.3e} > {tol:.0e}). The model, the transform "
                f"or the logit index has diverged from c0_final_predictions.py -- "
                f"fix that before trusting a single counterfactual probability.")
        if abs(sd.mean()) > 2 * sd.std():
            raise SystemExit(
                f"the delta vs c2_data.csv is systematically signed "
                f"(mean {sd.mean():+.3e} against sd {sd.std():.3e}). Float "
                f"reassociation is unbiased; a biased error means this script "
                f"preprocesses, weights or indexes the image differently from "
                f"c0_final_predictions.py. Fix that before trusting a single "
                f"counterfactual probability.")
        print("  OK: same model, same transform, same logit\n")

    loader = DataLoader(PathDataset(index["cf_path"], DATA_ROOT, TRANSFORM),
                        batch_size=args.batch_size, num_workers=args.num_workers)
    df = run(model, loader, target_idx, device, cam_dir, threshold,
             write_cam=not args.no_cam)

    dest = os.path.join(root, "cf_c0.parquet")
    if args.levels or args.limit:
        # A partial run must not clobber the levels already scored.
        if os.path.exists(dest):
            old = pd.read_parquet(dest)
            old = old[~old["cf_path"].isin(set(df["cf_path"]))]
            df = pd.concat([old, df], ignore_index=True)
    tmp = f"{dest}.tmp{os.getpid()}"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)
    print(f"\n  cf_c0.parquet  {df.shape[0]:,} x {df.shape[1]} -> {dest}")

    # ── checks ───────────────────────────────────────────────────────────────
    chk = index.merge(df[["cf_path", "prob"]], on="cf_path", how="left")
    print("\n--- checks ---")
    print(f"  rows without a prob : {int(chk['prob'].isna().sum())}")
    d = (chk["prob"] - chk["gen_cf_prob"]).abs()
    print(f"  |prob - gen_cf_prob|: mean {d.mean():.2e}  p99 {d.quantile(0.99):.2e}  "
          f"max {d.max():.2e}")
    print("    (this is the uint8 quantisation of the exported PNG. A large or "
          "systematically signed gap means the export and the generator disagree "
          "about the pixels, not that the quantisation is coarse.)")
    print(f"  prob by level:")
    for lv, g in chk.groupby("level_name"):
        for dr, gd in g.groupby("direction"):
            print(f"    {lv:>4s} {dr:12s} mean {gd['prob'].mean():.4f} "
                  f"(target {gd['target_prob'].iloc[0]:.4f})")


if __name__ == "__main__":
    main()
