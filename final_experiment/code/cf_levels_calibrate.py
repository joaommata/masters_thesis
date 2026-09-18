"""
cf_levels_calibrate.py
======================
Pick (tau, scale) for the graded-CF run, and measure wall time per image.

Why this exists
---------------
The first smoke test at tau=50, scale=100 produced three levels that were not
three intensities:

    L1 boundary  stop_t=46.75   mean_L1=0.0178   reach 100%
    L2 typical   stop_t=45.75   mean_L1=0.0180   reach 100%
    L3 extreme   stop_t= 5.50   mean_L1=0.0217   reach  25%

L1 and L2 fire one step apart and differ by 0.0001 in pixel distance -- the same
counterfactual with two labels. The first guided step overshoots the boundary by
a wide margin (one image went 0.033 -> 0.415 in a single step), so both early
levels are satisfied at once. Meanwhile the extreme level sits past what the
guidance reaches in-schedule, so it misses three times out of four.

Two knobs can fix that, and this script measures both rather than guessing:

  - **tau**: more steps to traverse the same probability range, so the levels
    separate. Cost is linear in tau.
  - **scale**: a smaller gradient step overshoots less. Free.

What it reports
---------------
For each (tau, scale) cell, over a few images:

  - `sep_12`, `sep_23`: how many timesteps apart consecutive levels stopped.
    This is the number that has to be large. Anything < ~5 means those two
    levels are the same image.
  - `dL1_12`, `dL1_23`: pixel distance between consecutive levels. The other
    way of asking the same question, in image space rather than schedule space.
  - `reach_3`: fraction of images reaching the extreme level.
  - `p_max`: the highest probability the trajectory achieved, averaged. If this
    plateaus well below the p95 target, the target is the problem, not tau.
  - `sec_per_img`: guided-only wall time, for the run estimate.

Run it, read the table, then set TAU/SCALE in jobs/cf_levels.sh.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf_generation_final as cf
import cf_generate_levels as cgl


# The sweep. tau is the expensive axis, so keep the grid small.
TAU_GRID   = [50, 100, 200]
SCALE_GRID = [25, 50, 100]
N_IMAGES   = 8
DISEASES   = ["cardiomegaly", "atelectasis"]
SEED       = 0


def _dmean(per_img, dirn):
    """Mean p_max over one direction only. NaN if that direction is absent."""
    v = [p["p_max"] for p in per_img if p["dirn"] == dirn]
    return float(np.mean(v)) if v else float("nan")


def probe(disease):
    print("\n" + "=" * 78)
    print(f"  {disease}")
    print("=" * 78)

    unet, classifier, _seg, cfg = cf.load_models(disease)
    gradcam = cgl.GradCAM(classifier, cfg.target_idx, cfg.threshold)

    df = pd.read_csv(cfg.c2_csv, usecols=["path", "prob", "true", "pred", "correct"])
    levels = cgl.compute_levels(df, cfg.threshold, disease=disease)

    print(f"\nlevels to_positive: " +
          "  ".join(f"{n}={v:.4f}" for n, v in zip(cgl.LEVEL_NAMES, levels["to_positive"])))
    print(f"levels to_negative: " +
          "  ".join(f"{n}={v:.4f}" for n, v in zip(cgl.LEVEL_NAMES, levels["to_negative"])))

    # A few images from each side, so both directions get exercised.
    sub = pd.concat([
        df[df["pred"] == 0].sample(N_IMAGES // 2, random_state=1),
        df[df["pred"] == 1].sample(N_IMAGES - N_IMAGES // 2, random_state=1),
    ]).reset_index(drop=True)

    rows = []
    for tau in TAU_GRID:
        for scale in SCALE_GRID:
            sampler = cf.make_sampler("FastDiME-woM", device=cfg.device)
            per_img = []
            t0 = time.time()

            for _, r in sub.iterrows():
                x0 = cf.load_image(r["path"])
                p_i = cf.get_c0_prob(x0, classifier)
                pred_i = cf.predict(p_i, cfg)
                dirn = "to_positive" if pred_i == 0 else "to_negative"
                tg = levels[dirn]

                out = cgl.run_one(
                    x0, dirn, tg, unet, classifier, sampler, gradcam, cfg,
                    tau=tau, scale=scale, seed=SEED, track_every=1,
                    want_unguided=False, unguided_sampler=None)

                traj = out["traj"]
                # Best probability the trajectory reached, in the direction of travel.
                p_max = (float(traj[:, 1].max()) if dirn == "to_positive"
                         else float(traj[:, 1].min()))

                per_img.append({
                    "dirn": dirn,
                    "stop_t": out["level_t"],
                    "L1": [float(np.abs(out["levels"][i] - out["x0"]).mean())
                           for i in range(3)],
                    "reached": out["level_reached"],
                    "p_max": p_max,
                })

            elapsed = time.time() - t0
            st = np.array([p["stop_t"] for p in per_img], float)
            l1 = np.array([p["L1"] for p in per_img], float)
            rc = np.array([p["reached"] for p in per_img], float)

            rows.append({
                "tau": tau, "scale": scale,
                "sep_12": float(np.mean(st[:, 0] - st[:, 1])),
                "sep_23": float(np.mean(st[:, 1] - st[:, 2])),
                "dL1_12": float(np.mean(np.abs(l1[:, 1] - l1[:, 0]))),
                "dL1_23": float(np.mean(np.abs(l1[:, 2] - l1[:, 1]))),
                "reach_2": float(rc[:, 1].mean()),
                "reach_3": float(rc[:, 2].mean()),
                # Reported per direction. Averaging the two is what produced the
                # bogus "saturates at 0.49" reading on the first pass: ascending
                # p_max heads to 1, descending heads to 0, so the mean of a
                # balanced sample sits near 0.5 regardless of the guidance.
                "p_max_pos": _dmean(per_img, "to_positive"),
                "p_max_neg": _dmean(per_img, "to_negative"),
                "n_pos": sum(p["dirn"] == "to_positive" for p in per_img),
                "n_neg": sum(p["dirn"] == "to_negative" for p in per_img),
                "L1_final": float(l1[:, 2].mean()),
                "sec_per_img": elapsed / len(sub),
            })
            print(f"  tau={tau:4d} scale={scale:5.0f}  "
                  f"sep(1-2)={rows[-1]['sep_12']:5.1f}  sep(2-3)={rows[-1]['sep_23']:5.1f}  "
                  f"reach3={rows[-1]['reach_3']:.2f}  "
                  f"p_max+={rows[-1]['p_max_pos']:.3f} p_max-={rows[-1]['p_max_neg']:.3f}  "
                  f"{rows[-1]['sec_per_img']:5.2f}s/img")

    res = pd.DataFrame(rows)
    print(f"\n--- {disease} full table ---")
    print(res.round(4).to_string(index=False))

    # What would the extreme level have to be to be reachable?
    print(f"\n  extreme target to_positive: {levels['to_positive'][2]:.4f}"
          f"   best reached: {res['p_max_pos'].max():.4f}")
    print(f"  extreme target to_negative: {levels['to_negative'][2]:.4f}"
          f"   best reached: {res['p_max_neg'].min():.4f}")
    return res


def main():
    out = {}
    for d in DISEASES:
        out[d] = probe(d).assign(disease=d)

    allres = pd.concat(out.values(), ignore_index=True)
    dest = "/work3/s251710/thesis_results/CF_levels/calibration.csv"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    allres.to_csv(dest, index=False)
    print(f"\n\nWrote {dest}")

    print("\n" + "=" * 78)
    print("  RUN-TIME ESTIMATE (guided only, no unguided twin)")
    print("=" * 78)
    for tau in TAU_GRID:
        s = allres[allres["tau"] == tau]["sec_per_img"].mean()
        for n in (5000, 10000):
            print(f"  tau={tau:4d}  n={n:6d}  {s:5.2f}s/img  "
                  f"-> {s * n / 3600:6.1f} GPU-hours per disease")


if __name__ == "__main__":
    main()
