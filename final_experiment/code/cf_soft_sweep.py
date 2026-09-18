"""
cf_soft_sweep.py
================
Sweep (tau, scale) for the soft-target approach, and pick the setting that lands
the levels on their targets.

Why only the soft-target approach
---------------------------------
The three-way comparison (cf_compare_approaches.py) is settled: approach C, one
full trajectory per level guided to a SOFT BCE target, is the one being used.
It is the only approach whose counterfactuals are all fully denoised at t=1, so
the blur confound cannot exist -- measured sharp_range 0.6% against 1.6% (A) and
2.6% (B) -- and it needs no overshoot fudge to hit its targets.

What is left is a tuning problem. At tau=150/scale=150 on effusion, every level
undershot by a consistent -0.07 to -0.09:

    L1 low   target 0.6775  ->  0.6000   (-0.0775)
    L2 mid   target 0.8270  ->  0.7582   (-0.0687)
    L3 high  target 0.9209  ->  0.8299   (-0.0910)

Same sign on every level, similar magnitude: that is an optimiser that has not
converged, not noise. Soft-target BCE is flatter near its minimum than a hard
target, so the gradient is weaker, and a setting tuned for hard-target guidance
underpushes here. Two knobs can fix it:

  - **scale**: a stronger gradient per step. Free, but a step that is too large
    overshoots the soft minimum WITHIN a step and can oscillate around it, which
    shows up as a larger edit (l1_total) without a better mean_err.
  - **tau**: more steps to converge in. Linear cost.

The sweep measures both rather than guessing which one binds.

What is measured
----------------
Per (tau, scale) cell, over a balanced sample of images:

  mean_err     mean |cf_prob - target| over the three levels. THE objective.
  bias_L1..L3  mean SIGNED error per level. All negative = still undershooting,
               so push harder. Mixed or positive = converged or overshooting.
  reach_rate   fraction of levels on the correct side of their target.
  spread       max - min of the three probabilities. The levels must stay
               distinct: a setting that hits every target but collapses them is
               useless.
  sharp_range  max - min of per-level sharpness. Should stay ~0 for every cell
               (it is structural to the approach); if it grows, something is
               wrong.
  l1_total     size of the edit. Watch this against mean_err -- a cell that
               improves the error only by editing far more of the image is
               buying accuracy with faithfulness.
  secs         wall time per image (3 trajectories).

Usage
-----
    # the default grid, both tasks
    python cf_soft_sweep.py

    # scale-only sweep, faster
    python cf_soft_sweep.py --taus 150 --scales 150 250 350 --n 8
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf_generation_final as cf
import cf_generate_levels as cgl


def sharpness(a):
    """Mean gradient energy. Structural check: should not vary across levels."""
    gx = np.diff(a, axis=1)
    gy = np.diff(a, axis=0)
    return float(np.sqrt((gx ** 2).mean() + (gy ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diseases", nargs="+", default=["effusion", "cardiomegaly"],
                    help="two tasks with very different C0 ranges")
    ap.add_argument("--taus", type=int, nargs="+", default=[150, 250])
    ap.add_argument("--scales", type=float, nargs="+", default=[150, 250, 350])
    ap.add_argument("--n", type=int, default=8, help="images per disease")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run without a GPU anyway (very slow; for debugging only)")
    ap.add_argument("--out",
                    default="/work3/s251710/thesis_results/CF_levels/soft_sweep.csv")
    args = ap.parse_args()

    n_cells = len(args.taus) * len(args.scales)
    print(f"soft-target sweep: {args.taus} x {args.scales} = {n_cells} cells")
    print(f"{args.n} images x {len(args.diseases)} diseases")

    # A silent CPU run is the failure mode this guards: every cell is thousands
    # of UNet calls, so on a login node it grinds for hours with no output and
    # looks hung. Force the mistake to surface immediately.
    if not torch.cuda.is_available():
        print("\n*** NO GPU VISIBLE -- this needs a GPU node. ***")
        print("    You are probably on a login node. Use the interactive node")
        print("    or submit final_experiment/jobs/cf_soft_sweep.sh")
        if not args.allow_cpu:
            sys.exit(1)
    else:
        print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    rows = []
    for disease in args.diseases:
        print("=" * 78)
        print(f"  {disease}")
        print("=" * 78)
        unet, clf, _seg, cfg = cf.load_models(disease)
        gradcam = cgl.GradCAM(clf, cfg.target_idx, cfg.threshold)
        sampler = cf.make_sampler("FastDiME-woM", device=cfg.device)

        df = pd.read_csv(cfg.c2_csv,
                         usecols=["path", "prob", "true", "pred", "correct"])
        levels = cgl.compute_levels(df, cfg.threshold, disease=disease)
        for d in ("to_positive", "to_negative"):
            print(f"  {d:12s} " + "  ".join(
                f"{n}={v:.4f}" for n, v in zip(cgl.LEVEL_NAMES, levels[d])))

        # Balanced across the confusion cells so both directions of travel and
        # both correctness regimes appear.
        cells = {"TN": (0, 1), "FN": (0, 0), "TP": (1, 1), "FP": (1, 0)}
        per = max(1, args.n // 4)
        parts = []
        for cell, (p, c) in cells.items():
            pool = df[(df["pred"] == p) & (df["correct"] == c)]
            take = min(per, len(pool))
            if take < per:
                print(f"  [warn] {cell} has only {len(pool)} rows")
            parts.append(pool.sample(take, random_state=args.sample_seed)
                         .assign(cell=cell))
        sub = pd.concat(parts).reset_index(drop=True)
        print(f"  {len(sub)} images\n")

        for tau in args.taus:
            for scale in args.scales:
                t0 = time.time()
                per_img = []

                for _, r in tqdm(list(sub.iterrows()), total=len(sub),
                                 desc=f"  tau={tau} scale={scale:.0f}", leave=False):
                    x0 = cf.load_image(r["path"])
                    p_i = cf.get_c0_prob(x0, clf)
                    pred_i = cf.predict(p_i, cfg)
                    dirn = "to_positive" if pred_i == 0 else "to_negative"
                    tg = levels[dirn]
                    s_x0 = sharpness(cf.to_np(x0))

                    out = cgl.run_one(
                        x0, dirn, tg, unet, clf, sampler, gradcam, cfg,
                        tau=tau, scale=scale, seed=args.seed, track_every=0)

                    probs = [out["level_probs"][i] for i in range(3)]
                    sharps = [sharpness(out["levels"][i]) / s_x0 for i in range(3)]
                    per_img.append({
                        "signed": [probs[i] - tg[i] for i in range(3)],
                        "abs": [abs(probs[i] - tg[i]) for i in range(3)],
                        "reach": [cgl.reached(probs[i], tg[i], dirn) for i in range(3)],
                        "spread": max(probs) - min(probs),
                        "sharp_range": max(sharps) - min(sharps),
                        "l1": float(np.abs(out["levels"][2] - out["x0"]).mean()),
                    })

                elapsed = time.time() - t0
                signed = np.array([p["signed"] for p in per_img], float)
                row = {
                    "disease": disease, "tau": tau, "scale": scale,
                    "n": len(per_img),
                    "mean_err": float(np.mean([p["abs"] for p in per_img])),
                    "bias_L1": float(signed[:, 0].mean()),
                    "bias_L2": float(signed[:, 1].mean()),
                    "bias_L3": float(signed[:, 2].mean()),
                    "reach_rate": float(np.mean([p["reach"] for p in per_img])),
                    "spread": float(np.mean([p["spread"] for p in per_img])),
                    "sharp_range": float(np.mean([p["sharp_range"] for p in per_img])),
                    "l1_total": float(np.mean([p["l1"] for p in per_img])),
                    "secs": elapsed / len(per_img),
                }
                rows.append(row)
                print(f"  tau={tau:4d} scale={scale:6.0f}  "
                      f"err={row['mean_err']:.4f}  reach={row['reach_rate']:.2f}  "
                      f"bias=[{row['bias_L1']:+.3f} {row['bias_L2']:+.3f} "
                      f"{row['bias_L3']:+.3f}]  spread={row['spread']:.3f}  "
                      f"L1={row['l1_total']:.4f}  {row['secs']:5.1f}s/img")

        del unet, clf, gradcam, sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    show = ["tau", "scale", "mean_err", "bias_L1", "bias_L2", "bias_L3",
            "reach_rate", "spread", "sharp_range", "l1_total", "secs"]
    for disease, g in res.groupby("disease"):
        print("\n" + "=" * 78)
        print(f"  {disease}")
        print("=" * 78)
        print(g[show].round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("  POOLED (a setting has to work on BOTH tasks)")
    print("=" * 78)
    pooled = res.groupby(["tau", "scale"]).agg(
        mean_err=("mean_err", "mean"), reach=("reach_rate", "mean"),
        spread=("spread", "mean"), sharp_range=("sharp_range", "mean"),
        l1=("l1_total", "mean"), secs=("secs", "mean")).round(4)
    print(pooled.to_string())

    best = pooled["mean_err"].idxmin()
    print(f"\n  lowest pooled mean_err: tau={best[0]}, scale={best[1]}"
          f"  (err {pooled.loc[best, 'mean_err']:.4f}, "
          f"reach {pooled.loc[best, 'reach']:.2f}, "
          f"spread {pooled.loc[best, 'spread']:.3f}, "
          f"{pooled.loc[best, 'secs']:.1f}s/img)")
    print("\n  Before taking it: check `spread` has not collapsed (levels must stay")
    print("  distinct) and `l1` has not jumped (accuracy bought with faithfulness).")
    print("  If every bias_* is still negative at the largest cell, the grid is too")
    print("  small -- extend --scales upward and rerun.")


if __name__ == "__main__":
    main()
