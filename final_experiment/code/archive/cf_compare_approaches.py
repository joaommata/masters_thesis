"""
cf_compare_approaches.py
========================
Decide between the three graded-CF approaches, over a proper sample and more
than one task.

Why this exists
---------------
Every comparison so far has been a single image. On one effusion image at
tau=150/scale=150 the three approaches came out:

    approach                      mean_err  sharp_range  spread   secs
    A. soft x0-bar (stop early)     0.0642       0.0156  0.1052    8.5
    B. finished tail (overshoot)    0.0329       0.0263  0.1633   32.6
    C. soft target (full runs)      0.0791       0.0056  0.2299   25.2

which cannot distinguish "C undershoots systematically" from "that image was
hard", and says nothing about whether the ordering holds on another disease
whose classifier operates in a completely different range.

The three approaches
--------------------
  A  run_one(finish_denoise=False)
     One trajectory, snapshot x_t_0 at each crossing. Cheapest. Levels land on
     target because it stops exactly when it crosses -- but they stop at
     different t, so sharpness differs between them (the blur confound).

  B  run_one(finish_denoise=True)
     Same snapshots, then each is finished to t=1 with a guided tail that aims
     TAIL_OVERSHOOT past its target. Equalises sharpness; the tail drifts.

  C  run_one_soft_target()
     Three full trajectories, each guided to the level's probability as a SOFT
     BCE target. No level stops early, so sharpness is equal by construction.

What is measured
----------------
Per image and approach:

  mean_err     mean |cf_prob - target| over the three levels. Are the levels
               where they claim to be?
  reach_rate   fraction of levels on the correct side of their target.
  sharp_range  max - min of per-level sharpness (gradient energy / original's).
               THIS IS THE CONFOUND. Near zero means a downstream model cannot
               separate the levels on blur.
  spread       max - min of the three cf_probs. Are the levels distinct at all?
  dl1_min      smallest pixel distance between consecutive levels, against
               l1_total. Distinctness in image space rather than probability.
  secs         wall time per image.

Usage
-----
    python cf_compare_approaches.py --diseases effusion cardiomegaly --n 12
    python cf_compare_approaches.py --approaches A C --n 20 --tau 150
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf_generation_final as cf
import cf_generate_levels as cgl


def sharpness(a):
    """Mean gradient energy. Blur lowers it; it is the confound's proxy."""
    gx = np.diff(a, axis=1)
    gy = np.diff(a, axis=0)
    return float(np.sqrt((gx ** 2).mean() + (gy ** 2).mean()))


def run_approach(name, x0, direction, targets, unet, clf, sampler, gradcam, cfg,
                 tau, scale, seed, soft_scale, overshoot):
    """Dispatch to one approach. Returns (out_dict, seconds)."""
    t0 = time.time()
    if name == "A":
        out = cgl.run_one(x0, direction, targets, unet, clf, sampler, gradcam, cfg,
                          tau=tau, scale=scale, seed=seed, track_every=0,
                          want_unguided=False, unguided_sampler=None,
                          finish_denoise=False)
    elif name == "B":
        out = cgl.run_one(x0, direction, targets, unet, clf, sampler, gradcam, cfg,
                          tau=tau, scale=scale, seed=seed, track_every=0,
                          want_unguided=False, unguided_sampler=None,
                          finish_denoise=True, tail_overshoot=overshoot)
    elif name == "C":
        out = cgl.run_one_soft_target(x0, direction, targets, unet, clf, sampler,
                                      gradcam, cfg, tau=tau, scale=soft_scale,
                                      seed=seed, track_every=0)
    else:
        raise ValueError(f"unknown approach {name!r}")
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diseases", nargs="+", default=["effusion", "cardiomegaly"],
                    help="tasks to test; they have very different C0 ranges")
    ap.add_argument("--approaches", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--n", type=int, default=12, help="images per disease")
    ap.add_argument("--tau", type=int, default=150)
    ap.add_argument("--scale", type=float, default=150)
    ap.add_argument("--soft-scale", type=float, default=None,
                    help="scale for approach C; defaults to --scale")
    ap.add_argument("--overshoot", type=float, default=cgl.TAIL_OVERSHOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--out", default="/work3/s251710/thesis_results/CF_levels/approach_comparison.csv")
    args = ap.parse_args()
    soft_scale = args.scale if args.soft_scale is None else args.soft_scale

    print(f"approaches {args.approaches}   diseases {args.diseases}")
    print(f"n={args.n} per disease   tau={args.tau}  scale={args.scale}  "
          f"soft_scale={soft_scale}  overshoot={args.overshoot}\n")

    rows = []
    for disease in args.diseases:
        print("=" * 78)
        print(f"  {disease}")
        print("=" * 78)
        unet, clf, _seg, cfg = cf.load_models(disease)
        gradcam = cgl.GradCAM(clf, cfg.target_idx, cfg.threshold)
        sampler = cf.make_sampler("FastDiME-woM", device=cfg.device)

        df = pd.read_csv(cfg.c2_csv, usecols=["path", "prob", "true", "pred", "correct"])
        levels = cgl.compute_levels(df, cfg.threshold, disease=disease)
        for d in ("to_positive", "to_negative"):
            print(f"  {d:12s} " +
                  "  ".join(f"{n}={v:.4f}"
                            for n, v in zip(cgl.LEVEL_NAMES, levels[d])))

        # Balanced over the four confusion cells, so both directions of travel
        # and both correctness regimes are represented.
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
        print(f"  {len(sub)} images: " + sub["cell"].value_counts().to_dict().__str__())

        for _, r in tqdm(list(sub.iterrows()), desc=disease, total=len(sub)):
            x0 = cf.load_image(r["path"])
            p_i = cf.get_c0_prob(x0, clf)
            pred_i = cf.predict(p_i, cfg)
            dirn = "to_positive" if pred_i == 0 else "to_negative"
            tg = levels[dirn]
            s_x0 = sharpness(cf.to_np(x0))

            for name in args.approaches:
                out, secs = run_approach(name, x0, dirn, tg, unet, clf, sampler,
                                         gradcam, cfg, args.tau, args.scale,
                                         args.seed, soft_scale, args.overshoot)
                probs = [out["level_probs"][i] for i in range(3)]
                sharps = [sharpness(out["levels"][i]) / s_x0 for i in range(3)]
                errs = [abs(probs[i] - tg[i]) for i in range(3)]
                reach = [cgl.reached(probs[i], tg[i], dirn) for i in range(3)]
                dl1 = [float(np.abs(out["levels"][i + 1] - out["levels"][i]).mean())
                       for i in range(2)]
                l1_tot = float(np.abs(out["levels"][2] - out["x0"]).mean())

                rows.append({
                    "disease": disease, "approach": name, "path": r["path"],
                    "cell": r["cell"], "direction": dirn, "orig_prob": p_i,
                    "mean_err": float(np.mean(errs)),
                    "reach_rate": float(np.mean(reach)),
                    "sharp_min": min(sharps), "sharp_max": max(sharps),
                    "sharp_range": max(sharps) - min(sharps),
                    "spread": max(probs) - min(probs),
                    "dl1_min": min(dl1), "l1_total": l1_tot,
                    "dl1_rel": min(dl1) / max(l1_tot, 1e-8),
                    "secs": secs,
                    **{f"p_L{i+1}": probs[i] for i in range(3)},
                    **{f"err_L{i+1}": probs[i] - tg[i] for i in range(3)},
                })

        # Free the models before the next disease loads its own.
        del unet, clf, gradcam, sampler
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(res)} rows)")

    # ── summary ───────────────────────────────────────────────────────────────
    agg = dict(n=("path", "size"), mean_err=("mean_err", "mean"),
               reach=("reach_rate", "mean"), sharp_range=("sharp_range", "mean"),
               spread=("spread", "mean"), dl1_rel=("dl1_rel", "mean"),
               secs=("secs", "mean"))

    print("\n" + "=" * 78)
    print("  BY DISEASE AND APPROACH")
    print("=" * 78)
    print(res.groupby(["disease", "approach"]).agg(**agg).round(4).to_string())

    print("\n" + "=" * 78)
    print("  POOLED")
    print("=" * 78)
    print(res.groupby("approach").agg(**agg).round(4).to_string())

    print("\n" + "=" * 78)
    print("  PER-LEVEL BIAS (mean signed error; systematic if all same sign)")
    print("=" * 78)
    print(res.groupby(["disease", "approach"])[
        ["err_L1", "err_L2", "err_L3"]].mean().round(4).to_string())

    # The decision, stated rather than left to the eye. Rank on the confound
    # first: it is the thing no amount of tuning fixes after the fact.
    pooled = res.groupby("approach").agg(**agg)
    print("\n" + "=" * 78)
    print("  READING IT")
    print("=" * 78)
    print(f"  lowest sharp_range (blur confound) : "
          f"{pooled['sharp_range'].idxmin()}  ({pooled['sharp_range'].min():.4f})")
    print(f"  lowest mean_err   (on target)      : "
          f"{pooled['mean_err'].idxmin()}  ({pooled['mean_err'].min():.4f})")
    print(f"  widest spread     (levels distinct): "
          f"{pooled['spread'].idxmax()}  ({pooled['spread'].max():.4f})")
    print(f"  fastest                            : "
          f"{pooled['secs'].idxmin()}  ({pooled['secs'].min():.1f}s/img)")
    print("\n  mean_err is correctable by tuning tau/scale; sharp_range is not.")


if __name__ == "__main__":
    main()
