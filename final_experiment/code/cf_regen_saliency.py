"""
cf_regen_saliency.py
====================
Rewrite the Grad-CAM maps beside an existing CF_levels run, in place.

Why this exists
---------------
The maps written during generation were oriented toward the POSITIVE class and
ReLU'd unconditionally, so for any image (or level) C0 calls negative the signed
map -- negative almost everywhere -- was clipped to exactly zero. That hit the
CF levels especially hard: a trajectory crosses the decision boundary by
construction, so the levels the run exists to produce are precisely the ones
whose maps came out blank.

The counterfactual IMAGES are unaffected. Guidance in cf_generation_final.py is
the classifier logit gradient plus an L1 term and never touches a CAM, and
cf_generate_levels.py computes saliency only after each trajectory is finished.
So the fix needs no diffusion: cf/<key>.npz already stores x0, cf_level_1..n and
their probabilities, which is everything the CAM recipe needs. Four forward+
backward passes per image, ~10 min for a disease, against ~32 h/shard to
regenerate the counterfactuals themselves.

Reads  cf/<key>.npz        (x0, cf_level_*, orig_prob, level_probs)
Writes saliency/<key>.npz  (x0, level_1..n, cam_signs)  -- overwritten in place

Usage
-----
    python cf_regen_saliency.py --disease atelectasis
    python cf_regen_saliency.py --disease atelectasis --limit 20 --dry-run
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf_generation_final as cf
from cf_generate_levels import GradCAM

RESULTS = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", default=os.environ.get("THESIS_TASK", "atelectasis"))
    ap.add_argument("--root", default=None,
                    help="CF_levels/<disease> dir (default: $THESIS_RESULTS/CF_levels/<disease>)")
    ap.add_argument("--limit", type=int, default=None, help="first N keys only")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report, write nothing")
    args = ap.parse_args()

    out_dir = args.root or os.path.join(RESULTS, "CF_levels", args.disease)
    cf_dir = os.path.join(out_dir, "cf")
    sal_dir = os.path.join(out_dir, "saliency")
    if not os.path.isdir(cf_dir):
        raise SystemExit(f"no cf/ directory under {out_dir}")
    os.makedirs(sal_dir, exist_ok=True)

    keys = sorted(glob.glob(os.path.join(cf_dir, "*.npz")))
    if args.limit:
        keys = keys[:args.limit]
    if not keys:
        raise SystemExit(f"no cf/*.npz under {cf_dir}")

    # Only C0 is needed -- no UNet, no sampler. load_models() would pull the
    # diffusion model in for nothing.
    cfg = cf.TaskConfig(args.disease)
    classifier = cf.load_c0(cfg)
    gradcam = GradCAM(classifier, cfg.target_idx, cfg.threshold)

    print(f"disease   {args.disease}")
    print(f"dir       {out_dir}")
    print(f"threshold {cfg.threshold:.6f}")
    print(f"keys      {len(keys):,}{'  (DRY RUN)' if args.dry_run else ''}")

    n_flip = 0          # maps whose orientation is negative (were blank before)
    n_empty_new = 0
    n_maps = 0
    for path in tqdm(keys, desc="regen saliency"):
        key = os.path.basename(path)[:-4]
        d = np.load(path)

        level_keys = sorted([k for k in d.files if k.startswith("cf_level_")],
                            key=lambda k: int(k.rsplit("_", 1)[1]))
        probs = [float(d["orig_prob"])] + [float(p) for p in d["level_probs"]]
        imgs = [d["x0"]] + [d[k] for k in level_keys]
        names = ["x0"] + [f"level_{i + 1}" for i in range(len(level_keys))]

        sal = {}
        for name, arr, p in zip(names, imgs, probs):
            x = torch.from_numpy(arr).float()[None, None].to(cfg.device) * 2 - 1
            pos, neg = gradcam(x)
            sal[f"{name}_pos"], sal[f"{name}_neg"] = pos, neg
            # A map is only empty now if the evidence is one-directional, which
            # is rare and genuine rather than an artifact of clipping.
            n_flip += int(neg.max() > pos.max())
            n_empty_new += int(pos.max() <= 0) + int(neg.max() <= 0)
            n_maps += 2

        if not args.dry_run:
            np.savez_compressed(os.path.join(sal_dir, key + ".npz"), **sal)

    print(f"\nmaps written     {0 if args.dry_run else n_maps:,} "
          f"({n_maps // 2:,} images x 2 directions)")
    print(f"neg-dominant     {n_flip:,} ({2 * n_flip / n_maps:.1%} of images) "
          f"-- these were the all-zero maps before")
    print(f"one-directional  {n_empty_new:,} ({n_empty_new / n_maps:.2%} of maps)")


if __name__ == "__main__":
    main()
