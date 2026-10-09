# cf_noise_floor_pilot.py
"""
Pilot: UNGUIDED diffusion ("noise floor") on a small balanced set of images.

The guided counterfactuals (cf_generate_levels.py) are pushed to a target probability,
so their trajectories mostly record the guidance, not the image (see
trajectory_fingerprint.ipynb). This pilot removes the guidance: each image is noised
to depth tau and denoised back with NO classifier push, and C0's probability is
recorded at EVERY step. Two questions:

1. Robustness. Does C0's opinion survive a faithful re-rendering of the image?
   Hypothesis: when C0 is right its evidence is stable, so the probability stays on
   the same side of the threshold; when it is wrong it drifts or flips. Measured over
   several seeds (do repeated runs agree?) and several depths (how much noise before
   C0 changes its mind?).

2. Edit vs reconstruction. The guided CFs differ from the original by ~0.033 mean |px|
   at ALL three levels, i.e. mostly by the noise-and-denoise itself. The "matched"
   run below uses the same seed and depth as the stored CFs; the sampler's only
   randomness is the forward noise and the per-step noise (the guidance draws none),
   so it follows the exact same noise path and is the CF minus its guidance.
   CF - matched = the edit itself.

Runs per image
--------------
    matched   tau and seed of the stored CFs (levels.json: tau=150, seed=0), 1 run
    depths    each of --depths (default 50 100 150 250), --seeds independent runs
              (default 4), batched on the GPU

Images: --n-per-cell from each of TN / FP / TP / FN, drawn from the images that
have guided CFs (CF_levels/<disease>/manifest.csv), so both questions use the same
images.

Outputs ($THESIS_RESULTS/CF_noise_floor/<disease>/)
--------------------------------------------------
    runs/<key>.npz     per image:
        matched_traj    (tau-1,)     C0 prob per step, denoising order
        matched_final   ()           C0 prob of the finished image
        matched_recon   (224, 224)   the finished image, [0, 1], float16
        traj_<d>        (seeds, d-1) C0 prob per step at depth d
        final_<d>       (seeds,)     C0 prob of each finished image
        l1_<d>          (seeds,)     mean |finished - original|
        recon_<d>       (seeds, 224, 224) finished images, float16 (only for d = matched tau)
        orig_prob, orig_pred, threshold
    images.csv         the selected images (written first; the run list)
    summary.csv        one row per image, built from runs/ at the end (--summarize
                       rebuilds it at any time)

Written per image, atomically (temp file + rename), and skipped if present, so a
killed job resumes where it stopped.

Usage
-----
    python cf_noise_floor_pilot.py --disease cardiomegaly               # 250 per cell
    python cf_noise_floor_pilot.py --smoke                              # 1 image, tiny depths
    python cf_noise_floor_pilot.py --disease cardiomegaly --summarize   # rebuild summary.csv
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf_generation_final as cf

CELLS = ["TN", "FP", "TP", "FN"]


# ══════════════════════════════════════════════════════════════════════════════
# ONE RUN
# ══════════════════════════════════════════════════════════════════════════════

def unguided_run(x0, tau, seed, n, unet, classifier, sampler):
    """n independent unguided runs from depth tau, batched.

    Returns (traj (n, tau-1), final_probs (n,), final_images (n, 1, 224, 224) in [-1, 1]).
    The per-step prob is C0 on the model's denoised estimate at that step -- the same
    quantity cf_generate_levels.py records for the guided trajectories.
    """
    xb = x0.repeat(n, 1, 1, 1)
    target = torch.zeros((n, 1), device=x0.device)        # unused: guidance weights are 0
    torch.manual_seed(seed)
    x_t, _ = sampler.forward_diffusion(xb, torch.as_tensor(tau, dtype=torch.long, device=x0.device))

    traj, x_t_0 = [], None
    with contextlib.redirect_stdout(io.StringIO()):       # p_sample_once prints per step
        for t in reversed(range(1, tau)):
            x_t_0, x_t, _mg, prob, _l = sampler.p_sample_once(
                unet, classifier, timestep=t, x_t=x_t, x_0=xb, target=target, scale=0.0)
            traj.append(prob.detach().view(-1).cpu().numpy())
    final = x_t_0.detach().clamp(-1, 1)
    with torch.no_grad():
        p_final = torch.sigmoid(classifier(final)).view(-1).cpu().numpy()
    return np.stack(traj, axis=1).astype(np.float32), p_final.astype(np.float32), final


def to_unit(x):
    """[-1, 1] tensor (n, 1, H, W) -> (n, H, W) float array in [0, 1]."""
    return ((x[:, 0].cpu().numpy() + 1) / 2).clip(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_images(disease, n_per_cell, seed):
    """n_per_cell images from each of TN / FP / TP / FN, among those with guided CFs."""
    man = pd.read_csv(os.path.join(cf.RESULTS_DIR, "CF_levels", disease, "manifest.csv"))
    img = man.drop_duplicates("path")[["path", "key", "cell", "orig_prob", "orig_pred"]]
    parts = []
    for c in CELLS:
        pool = img[img["cell"] == c]
        if len(pool) < n_per_cell:
            print(f"[warn] {c}: only {len(pool)} images")
        parts.append(pool.sample(min(n_per_cell, len(pool)), random_state=seed))
    # interleave the cells, so a partial run is still balanced
    sel = pd.concat(parts)
    sel["order"] = sel.groupby("cell").cumcount()
    return sel.sort_values(["order", "cell"]).drop(columns="order").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def summarize(out_dir, disease, depths, matched_tau):
    """One row per finished image: stability of C0's prob under unguided re-rendering,
    and the size of the guided edit once the reconstruction is taken out."""
    images = pd.read_csv(os.path.join(out_dir, "images.csv"))
    cf_dir = os.path.join(cf.RESULTS_DIR, "CF_levels", disease, "cf")
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

    rows = []
    for _, r in images.iterrows():
        f = os.path.join(out_dir, "runs", f"{r['key']}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        thr, pred, p0 = float(z["threshold"]), int(z["orig_pred"]), float(z["orig_prob"])
        row = {"path": r["path"], "cell": r["cell"], "correct": int(r["cell"] in ("TN", "TP")),
               "orig_prob": p0, "orig_pred": pred,
               "matched_final": float(z["matched_final"])}
        for d in depths:
            fin = z[f"final_{d}"]
            # drift in logit, signed so that + = towards the OTHER class
            drift = (lg(fin) - lg(p0)) * (-1 if pred else 1)
            row.update({
                f"drift_mean_{d}": drift.mean(),                        # where it goes on average
                f"drift_absmean_{d}": np.abs(drift).mean(),             # how far, either way
                f"seed_std_{d}": lg(fin).std(),                         # do repeated runs agree
                f"flip_rate_{d}": ((fin > thr) != bool(pred)).mean(),   # share of runs that flip
                f"traj_std_{d}": lg(z[f"traj_{d}"]).std(axis=1).mean(), # wobble along a run
                f"l1_{d}": z[f"l1_{d}"].mean(),
            })
        # edit vs reconstruction, against the stored guided CFs
        cf_file = os.path.join(cf_dir, f"{r['key']}.npz")
        if os.path.exists(cf_file):
            c = np.load(cf_file)
            x0 = c["x0"].astype(np.float32)
            recon = z["matched_recon"].astype(np.float32)
            row["l1_recon_vs_orig"] = np.abs(recon - x0).mean()
            for lvl in (1, 2, 3):
                cfl = c[f"cf_level_{lvl}"].astype(np.float32)
                row[f"l1_cf{lvl}_vs_orig"] = np.abs(cfl - x0).mean()
                row[f"l1_cf{lvl}_vs_recon"] = np.abs(cfl - recon).mean()   # the edit itself
        rows.append(row)

    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"summary.csv: {len(s)} images")
    if len(s) and "l1_cf3_vs_recon" in s:
        print(f"  mean |px| CF(high) vs original {s['l1_cf3_vs_orig'].mean():.4f} | "
              f"vs matched reconstruction {s['l1_cf3_vs_recon'].mean():.4f} | "
              f"reconstruction vs original {s['l1_recon_vs_orig'].mean():.4f}")
    return s


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", default=os.environ.get("THESIS_TASK", "cardiomegaly"))
    ap.add_argument("--n-per-cell", type=int, default=250)
    ap.add_argument("--depths", type=int, nargs="+", default=[50, 100, 150, 250])
    ap.add_argument("--seeds", type=int, default=4, help="independent runs per depth")
    ap.add_argument("--sample-seed", type=int, default=0, help="which images are drawn")
    ap.add_argument("--out", default=None, help="default $THESIS_RESULTS/CF_noise_floor/<disease>")
    ap.add_argument("--smoke", action="store_true", help="1 image, depths 5 10, 2 seeds, matched tau 10")
    ap.add_argument("--summarize", action="store_true", help="only rebuild summary.csv")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(cf.RESULTS_DIR, "CF_noise_floor", args.disease)
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)

    # the stored CFs' depth and seed: the matched run must use exactly these
    lv = json.load(open(os.path.join(cf.RESULTS_DIR, "CF_levels", args.disease, "levels.json")))
    matched_tau, matched_seed = int(lv["tau"]), int(lv["seed"])
    if args.smoke:
        args.n_per_cell, args.depths, args.seeds, matched_tau = 1, [5, 10], 2, 10
    # the seed images are kept for the depth the CFs were made at
    keep_images_at = int(lv["tau"])

    if args.summarize:
        summarize(out_dir, args.disease, args.depths, matched_tau)
        return

    images_csv = os.path.join(out_dir, "images.csv")
    if os.path.exists(images_csv) and not args.smoke:
        images = pd.read_csv(images_csv)          # keep the same images on resume
    else:
        images = select_images(args.disease, args.n_per_cell, args.sample_seed)
        if args.smoke:
            images = images.head(1)
        images.to_csv(images_csv, index=False)
    json.dump({"disease": args.disease, "depths": args.depths, "seeds": args.seeds,
               "matched_tau": matched_tau, "matched_seed": matched_seed,
               "n_images": len(images), "mode": "noise floor (guidance weights 0, no inpainting)"},
              open(os.path.join(out_dir, "run_config.json"), "w"), indent=2)

    unet, classifier, _seg, cfg = cf.load_models(args.disease)
    sampler = cf.make_sampler("noise floor", device=cfg.device)

    todo = images[~images["key"].apply(lambda k: os.path.exists(os.path.join(out_dir, "runs", f"{k}.npz")))]
    steps = (matched_tau - 1) + sum(d - 1 for d in args.depths)
    print(f"{len(todo)} of {len(images)} images to run | depths {args.depths} x {args.seeds} seeds "
          f"+ matched (tau {matched_tau}, seed {matched_seed}) | {steps} batched steps per image")

    t0 = time.time()
    for _, r in tqdm(list(todo.iterrows()), desc=args.disease):
        x0 = cf.load_image(r["path"])
        p0 = cf.get_c0_prob(x0, classifier)
        out = {"orig_prob": np.float32(p0), "orig_pred": np.int8(cf.predict(p0, cfg)),
               "threshold": np.float32(cfg.threshold)}

        # 1. matched: the stored CFs' exact noise path, without the guidance
        traj, fin, img = unguided_run(x0, matched_tau, matched_seed, 1, unet, classifier, sampler)
        out.update(matched_traj=traj[0], matched_final=fin[0],
                   matched_recon=to_unit(img)[0].astype(np.float16))

        # 2. depths x seeds. Seed per depth so a depth's runs don't depend on the others.
        for d in args.depths:
            traj, fin, img = unguided_run(x0, d, 1000 + d, args.seeds, unet, classifier, sampler)
            x0_unit = to_unit(x0)[0]
            out[f"traj_{d}"], out[f"final_{d}"] = traj, fin
            out[f"l1_{d}"] = np.abs(to_unit(img) - x0_unit).mean(axis=(1, 2)).astype(np.float32)
            if d == keep_images_at:
                out[f"recon_{d}"] = to_unit(img).astype(np.float16)

        # atomic write: a kill mid-save never leaves a truncated file that looks finished
        final_path = os.path.join(out_dir, "runs", f"{r['key']}.npz")
        tmp = final_path + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, final_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = len(todo)
    if n:
        print(f"{n} images in {(time.time() - t0) / 60:.1f} min ({(time.time() - t0) / n:.1f} s/image)")
    summarize(out_dir, args.disease, args.depths, matched_tau)


if __name__ == "__main__":
    main()
