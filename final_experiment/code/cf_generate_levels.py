"""
cf_generate_levels.py
=====================
Graded diffusion counterfactuals: three intensities per image, for one disease.

For each image, three FULL guided trajectories are run towards the opposite
class, each aimed at a different probability. The result is three
counterfactuals of increasing intensity -- a weak one, a middling one and a
strong one -- all fully denoised.

The two ideas
-------------
**1. The levels are three equally spaced probabilities.** They span the range C0
actually produces for the target class, so `mid` is literally halfway between a
weak and a strong member. The endpoints are per-disease (p5 and p95 of that
class), because C0's confidence differs enormously between tasks: its positive
probabilities reach 0.978 on effusion but only 0.309 on consolidation, so a
fixed high end would be unreachable for half the diseases. See `compute_levels`.

**2. Each level is a SOFT classifier target, not a stopping point.**
`get_classifier_loss` computes `binary_cross_entropy(sigmoid(logits), y_target)`
with `y_target` a float, never a hard 0/1. BCE against a soft target has its
minimum exactly at that probability and its gradient reverses sign past it, so
passing `y_target = 0.72` makes the guidance converge on 0.72 and hold there for
the whole schedule.

Why not snapshot one trajectory
-------------------------------
The first design ran a single trajectory and saved x_t_0 whenever the
probability crossed a level. Those snapshots are `out['denoised']`, the model's
one-shot estimate of a clean x_0 from the current noisy state: denoised
algebraically, but the authors' own name for it is `x_0_blur`. Estimated from a
heavily corrupted x_t at high t it is a soft, low-confidence reconstruction, and
because the levels stopped at different t they were blurred by different amounts:

    level   stop t   sharpness vs the original
    low        94            84.7%
    mid        89            85.5%
    high       42            92.0%

Sharpness tracked the stop timestep almost perfectly, so "intensity" was partly
a sharpness axis and a downstream model could separate the levels on blur alone.
Finishing the snapshots to t=1 did not rescue it: an unguided tail walked the
image back to the original class (all three collapsed to p~0.07, below the
threshold), and a guided tail drifted short by up to 0.18. Running each level as
its own complete trajectory removes the problem by construction -- measured
sharpness range 0.6%, against 1.6% and 2.6% for the two snapshot variants.

Settings
--------
tau=150, scale=150, from `cf_soft_sweep.py`. Cardiomegaly converges there (bias
-0.000/+0.001/-0.015, mixed signs); effusion lands within 0.05, about 20% of its
0.25 inter-level step. Raising scale to 600 cuts that residual bias but barely
moves the spread (0.366 -> 0.403), and spread is the thing that matters: the
levels have to be distinct, not land on exact probabilities.

Unmasked (`--mode FastDiME-woM`). The inpainting mask is FastDiME's
shortcut-detection device, which keeps the edit local; our changes are not
necessarily localized and we are not looking for shortcuts.

Outputs (under $THESIS_RESULTS/CF_levels/<disease>/)
---------------------------------------------------
    manifest.csv            one row per (image, level)
    levels.json             the targets, the settings, how they were derived
    cf/<key>.npz            x0, cf_level_1..3, level_probs, targets
    saliency/<key>.npz      Grad-CAM (7,7) at x0 and every level
    trajectory/<key>.npz    per-step (t, prob), one array per level

`<key>` is the image path with "/" -> "_", matching the cam_path convention in
c0_final_predictions.py.

Usage
-----
    python cf_generate_levels.py --disease effusion --n 4 --smoke
    python cf_generate_levels.py --disease effusion --n 5000
    # resume: images already in manifest.csv are skipped
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


# ── Levels ────────────────────────────────────────────────────────────────────

LEVEL_NAMES = ["low", "mid", "high"]

# Endpoints, as percentiles of the target class's own probability distribution.
# The true max is a single outlier image and a poor target; p95 is the edge of
# the bulk.
LOW_PCT = 5
HIGH_PCT = 95


def compute_levels(df, threshold, disease=None, low_pct=LOW_PCT, high_pct=HIGH_PCT):
    """Three equally spaced probability targets, per direction of travel.

    Returns {"to_positive": [p1,p2,p3], "to_negative": [p1,p2,p3], ...meta}.
    `to_positive` ascends, `to_negative` descends.

    The endpoints come from the target class's own distribution and the levels
    are spaced evenly between them, so the step from low to mid equals the step
    from mid to high.

    Restricted to class members C0 ALSO classifies correctly. Without that an
    endpoint can land on the wrong side of the decision boundary -- atelectasis
    p5 over all true positives is 0.188 against a threshold of 0.189, which is
    not a positive counterfactual at all.
    """
    pos = df.loc[(df["true"] == 1) & (df["prob"] > threshold), "prob"].to_numpy()
    neg = df.loc[(df["true"] == 0) & (df["prob"] < threshold), "prob"].to_numpy()

    if len(pos) < 10 or len(neg) < 10:
        raise ValueError(
            f"too few correctly-classified rows to place endpoints from "
            f"(pos={len(pos)}, neg={len(neg)}) for disease={disease!r}")

    lo_pos = float(np.percentile(pos, low_pct))
    hi_pos = float(np.percentile(pos, high_pct))
    to_pos = list(np.linspace(lo_pos, hi_pos, 3))

    # Descending: low probability is DEEP in the negative class, so the
    # percentiles mirror.
    lo_neg = float(np.percentile(neg, 100 - low_pct))
    hi_neg = float(np.percentile(neg, 100 - high_pct))
    to_neg = list(np.linspace(lo_neg, hi_neg, 3))

    return {
        "low_pct": low_pct, "high_pct": high_pct,
        "threshold": float(threshold),
        "to_positive": [float(v) for v in to_pos],
        "to_negative": [float(v) for v in to_neg],
        "level_names": LEVEL_NAMES,
        "step_pos": float(to_pos[1] - to_pos[0]),
        "step_neg": float(to_neg[1] - to_neg[0]),
        "n_pos_correct": int(len(pos)), "n_neg_correct": int(len(neg)),
        "max_pos": float(pos.max()), "min_neg": float(neg.min()),
    }


def reached(prob, target_prob, direction):
    """Is the probability at or past the target, in the direction of travel?

    Recorded per level, but do not judge a run by it: it is one-sided, so a
    level that has converged lands on either side about equally and 'reaches'
    only half the time. The signed error is the honest measure.
    """
    return prob >= target_prob if direction == "to_positive" else prob <= target_prob


# ── Saliency ──────────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM on C0's final feature map, same recipe as c0_final_predictions.py.

    Hooks `backbone.features`, backprops the target logit only, weights the
    ReLU'd activations by the spatially-averaged gradients. Returns a (7,7) map
    normalised to [0,1] -- identical in shape and scaling to the CAMs already in
    the C2 tables, so they are directly comparable.

    The map is oriented toward the class C0 PREDICTS for the image it is given,
    not toward the positive class. Backprop is on the positive logit either way,
    but on an image C0 calls negative that signed map is negative almost
    everywhere, so ReLU'ing it unconditionally (the old behaviour) returned an
    all-zero map. Flipping the sign first makes "support for the decision C0
    made" the positive direction in every case.

    This matters more here than for static images: a trajectory crosses the
    decision boundary by construction, so a map pinned to x0's prediction would
    be annihilated at exactly the flipped levels the CF is generated to produce.
    Each level is therefore oriented by its own probability, and the signs are
    returned so a downstream comparison across levels can put them back in a
    common frame.
    """

    def __init__(self, guidance_model, target_idx, threshold):
        self.model = guidance_model          # cf.C0Guidance
        self.target_idx = target_idx
        self.threshold = float(threshold)
        self.cache = {}
        self.model.backbone.features.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.cache["acts"] = torch.relu(out).detach()
        if out.requires_grad:
            out.register_hook(lambda g: self.cache.__setitem__("grads", g.detach()))

    def __call__(self, x, prob=None):
        """x: (1,1,224,224) in [-1,1]. Returns (cam_pos, cam_neg), both (7,7)
        float32 in [0,1] -- the evidence for the disease and the evidence for
        health, each scaled by its own max.

        `prob` is accepted for signature compatibility and is no longer used:
        nothing about the map depends on a threshold any more.
        """
        self.cache.clear()
        self.model.zero_grad(set_to_none=True)
        # Everything else runs under no_grad; the CAM needs a graph.
        with torch.enable_grad():
            xin = x.detach().clone().requires_grad_(True)
            logits = self.model.all_logits(xin)
            logits[:, self.target_idx].backward(
                torch.ones(logits.shape[0], device=x.device))

        acts, grads = self.cache["acts"], self.cache["grads"]
        w = grads.mean(dim=(2, 3), keepdim=True)
        signed = (w * acts).sum(dim=1)[0]
        pos = torch.relu(signed).cpu().numpy()
        neg = torch.relu(-signed).cpu().numpy()
        pos = pos / (pos.max() + 1e-8)
        neg = neg / (neg.max() + 1e-8)
        return pos.astype(np.float32), neg.astype(np.float32)


# ── One image ─────────────────────────────────────────────────────────────────

def run_one(x0, direction, targets, unet, classifier, sampler, gradcam, cfg,
            tau, scale, seed, track_every=0):
    """Three full trajectories, one per level, each guided to a soft target.

    Every level runs the complete tau -> 1 schedule, so all three come back as
    fully resolved images at t=1 and no level is blurrier than another.

    The one line that encodes the level is `y`: a float tensor at the level's
    probability, not a hard 1.0. All three trajectories share a seed, so they
    start from the same noise and differ only in what they were asked for.
    """
    n_levels = len(targets)
    levels, level_probs, level_reached, trajs = [], [], [], []

    for li in range(n_levels):
        y = torch.full((x0.shape[0], 1), float(targets[li]), device=cfg.device)

        torch.manual_seed(seed)
        t_start = torch.as_tensor(tau, dtype=torch.long, device=cfg.device)
        x_t, _ = sampler.forward_diffusion(x0, t_start)
        this_scale = scale * x0.shape[0]

        x_t_0, traj = None, []
        # p_sample_once prints a line per inpainted step; silence it.
        with contextlib.redirect_stdout(io.StringIO()):
            for t in reversed(range(1, tau)):
                x_t_0, x_t, _mg, prob, _l = sampler.p_sample_once(
                    unet, classifier, timestep=t, x_t=x_t, x_0=x0,
                    target=y, scale=this_scale)
                if track_every and (t % track_every == 0 or t == 1):
                    traj.append((t, float(prob.squeeze())))

        x_fin = x_t_0.detach()
        p_fin = cf.get_c0_prob(x_fin, classifier)
        levels.append(cf.to_np(x_fin))
        level_probs.append(p_fin)
        level_reached.append(bool(reached(p_fin, targets[li], direction)))
        trajs.append(np.array(traj, dtype=np.float32) if traj
                     else np.zeros((0, 2), np.float32))
        del x_t, x_t_0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = {
        "x0": cf.to_np(x0),
        "levels": levels,
        "level_probs": level_probs,
        "level_reached": level_reached,
        "targets": list(targets),
        "trajs": trajs,
    }

    # Two maps per image: the evidence for the disease and the evidence for
    # health, stored separately. A trajectory crosses the decision boundary by
    # construction, which is exactly why a single oriented map fails here --
    # whichever direction it is not pointing gets ReLU'd to zero, and that is
    # guaranteed to happen somewhere along every run.
    sal = {}
    sal["x0_pos"], sal["x0_neg"] = gradcam(x0)
    for li in range(n_levels):
        t_img = torch.from_numpy(levels[li]).float()[None, None].to(cfg.device) * 2 - 1
        sal[f"level_{li + 1}_pos"], sal[f"level_{li + 1}_neg"] = gradcam(t_img)
    out["saliency"] = sal
    return out


# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", default=os.environ.get("THESIS_TASK", "effusion"))
    ap.add_argument("--n", type=int, default=5000, help="images to process")
    ap.add_argument("--mode", default="FastDiME-woM", choices=sorted(cf.FASTDIME_PRESETS),
                    help="unmasked by default")
    # Both settled by cf_soft_sweep.py; see the module docstring.
    ap.add_argument("--tau", type=int, default=150)
    ap.add_argument("--scale", type=float, default=150)
    ap.add_argument("--low-pct", type=float, default=LOW_PCT)
    ap.add_argument("--high-pct", type=float, default=HIGH_PCT)
    ap.add_argument("--track-every", type=int, default=5,
                    help="record (t, prob) every N steps; 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=1, help="which images get picked")
    ap.add_argument("--balanced", action="store_true",
                    help="equal numbers from TP/TN/FP/FN instead of a random draw")
    # Sharding: run N copies of this script on N GPUs, each taking a disjoint
    # slice of the SAME selected image set. Throughput scales linearly because
    # the work is embarrassingly parallel -- there is no batching win to be had
    # (the UNet saturates by batch 4, and two thirds of each step is the
    # classifier's forward+backward, which batching does not help).
    # Each shard writes its own manifest and npz files, so nothing collides.
    ap.add_argument("--shard", type=int, default=0,
                    help="this shard's index, 0-based")
    ap.add_argument("--n-shards", type=int, default=1,
                    help="total number of shards")
    ap.add_argument("--out", default=None,
                    help="defaults to $THESIS_RESULTS/CF_levels/<disease>")
    ap.add_argument("--smoke", action="store_true", help="tiny run, verbose per image")
    ap.add_argument("--flush-every", type=int, default=25,
                    help="rewrite manifest.csv every N images")
    args = ap.parse_args()

    if args.smoke:
        args.n = min(args.n, 4)

    if not torch.cuda.is_available():
        print("*** NO GPU VISIBLE. This needs a GPU node; on a login node it "
              "will grind for hours. ***")
        sys.exit(1)

    t_wall = time.time()

    unet, classifier, _seg, cfg = cf.load_models(args.disease)
    gradcam = GradCAM(classifier, cfg.target_idx, cfg.threshold)
    sampler = cf.make_sampler(args.mode, device=cfg.device)

    out_dir = args.out or os.path.join(cf.RESULTS_DIR, "CF_levels", args.disease)
    for sub_dir in ("cf", "saliency", "trajectory"):
        os.makedirs(os.path.join(out_dir, sub_dir), exist_ok=True)

    # ── levels ────────────────────────────────────────────────────────────────
    df = pd.read_csv(cfg.c2_csv, usecols=["path", "prob", "true", "pred", "correct"])
    lv = compute_levels(df, cfg.threshold, disease=args.disease,
                        low_pct=args.low_pct, high_pct=args.high_pct)
    lv.update(tau=args.tau, scale=args.scale, mode=args.mode, seed=args.seed)
    with open(os.path.join(out_dir, "levels.json"), "w") as f:
        json.dump(lv, f, indent=2)

    print(f"\nLevels for {args.disease}  "
          f"(equally spaced, p{args.low_pct:.0f}-p{args.high_pct:.0f} of the class)")
    for d, step in (("to_positive", "step_pos"), ("to_negative", "step_neg")):
        print(f"  {d:12s} " +
              "   ".join(f"{n}={v:.4f}" for n, v in zip(LEVEL_NAMES, lv[d])) +
              f"      step {lv[step]:+.4f}")
    print(f"  threshold    {cfg.threshold:.4f}")

    # ── image selection ───────────────────────────────────────────────────────
    if args.balanced:
        cells = {"TN": (0, 1), "FN": (0, 0), "TP": (1, 1), "FP": (1, 0)}
        per = max(1, args.n // 4)
        parts = []
        for cell, (p, c) in cells.items():
            pool = df[(df["pred"] == p) & (df["correct"] == c)]
            take = min(per, len(pool))
            if take < per:
                print(f"[warn] {cell} has only {len(pool)} rows")
            parts.append(pool.sample(take, random_state=args.sample_seed)
                         .assign(cell=cell))
        sub = pd.concat(parts).reset_index(drop=True)
    else:
        sub = df.sample(min(args.n, len(df)),
                        random_state=args.sample_seed).reset_index(drop=True)
        sub["cell"] = np.where(
            sub["pred"] == 1,
            np.where(sub["correct"] == 1, "TP", "FP"),
            np.where(sub["correct"] == 1, "TN", "FN"))

    # ── shard ─────────────────────────────────────────────────────────────────
    # Strided, not blocked: shard k takes rows k, k+N, k+2N... Every shard sees
    # the same balanced mix of TP/TN/FP/FN, whereas contiguous blocks would give
    # shard 0 all the TNs.
    if args.n_shards > 1:
        if not 0 <= args.shard < args.n_shards:
            raise ValueError(f"--shard {args.shard} out of range for "
                             f"--n-shards {args.n_shards}")
        n_all = len(sub)
        sub = sub.iloc[args.shard::args.n_shards].reset_index(drop=True)
        print(f"\nshard {args.shard + 1}/{args.n_shards}: "
              f"{len(sub):,} of {n_all:,} images")

    # ── resume ────────────────────────────────────────────────────────────────
    # One manifest per shard, so concurrent shards never write the same file.
    # Merge them afterwards with cf_merge_shards.py.
    man_name = ("manifest.csv" if args.n_shards == 1
                else f"manifest_shard{args.shard}of{args.n_shards}.csv")
    man_path = os.path.join(out_dir, man_name)
    rows, done = [], set()
    if os.path.exists(man_path):
        prev = pd.read_csv(man_path)
        rows = prev.to_dict("records")
        done = set(prev["path"].unique())
        print(f"\nResuming: {len(done):,} images in {os.path.basename(man_path)}")

    todo = sub[~sub["path"].isin(done)].reset_index(drop=True)
    steps_per = (args.tau - 1) * 3
    print(f"\n{len(todo):,} images to process "
          f"({len(sub):,} selected, {len(sub) - len(todo):,} done)")
    print(f"tau={args.tau}  scale={args.scale}  mode={args.mode}")
    print(f"{steps_per} UNet calls per image -> {steps_per * len(todo):,} total\n")

    # ── loop ──────────────────────────────────────────────────────────────────
    n_new = 0
    for _, r in tqdm(list(todo.iterrows()), desc=args.disease, total=len(todo)):
        key = r["path"].replace("/", "_")
        x0 = cf.load_image(r["path"])
        p_orig = cf.get_c0_prob(x0, classifier)
        pred_orig = cf.predict(p_orig, cfg)
        direction = "to_positive" if pred_orig == 0 else "to_negative"
        targets = lv[direction]

        res = run_one(x0, direction, targets, unet, classifier, sampler, gradcam,
                      cfg, tau=args.tau, scale=args.scale, seed=args.seed,
                      track_every=args.track_every)

        # ── save ──────────────────────────────────────────────────────────────
        payload = {
            "x0": res["x0"].astype(np.float32),
            "level_probs": np.array(res["level_probs"], np.float32),
            "level_reached": np.array(res["level_reached"], bool),
            "targets": np.array(targets, np.float32),
            "orig_prob": np.float32(p_orig),
            "direction": direction,
        }
        for li, img in enumerate(res["levels"]):
            payload[f"cf_level_{li + 1}"] = img.astype(np.float32)
        np.savez_compressed(os.path.join(out_dir, "cf", key + ".npz"), **payload)

        np.savez_compressed(os.path.join(out_dir, "saliency", key + ".npz"),
                            **res["saliency"])
        if args.track_every:
            np.savez_compressed(
                os.path.join(out_dir, "trajectory", key + ".npz"),
                **{f"level_{i + 1}": t for i, t in enumerate(res["trajs"])})

        # ── manifest ──────────────────────────────────────────────────────────
        for li in range(len(targets)):
            p_cf = res["level_probs"][li]
            rows.append({
                "path": r["path"], "key": key, "cell": r["cell"],
                "true": int(r["true"]), "orig_prob": p_orig, "orig_pred": pred_orig,
                "direction": direction,
                "level": li + 1, "level_name": LEVEL_NAMES[li],
                "target_prob": targets[li], "cf_prob": p_cf,
                "err": p_cf - targets[li],
                "reached": int(res["level_reached"][li]),
                "flipped": int(cf.predict(p_cf, cfg) != pred_orig),
                "l1": float(np.abs(res["levels"][li] - res["x0"]).mean()),
            })

        n_new += 1
        if args.smoke:
            print(f"\n  {key}\n    orig p={p_orig:.4f} pred={pred_orig} -> {direction}")
            for li, name in enumerate(LEVEL_NAMES):
                print(f"    {name:5s} target={targets[li]:.4f}  "
                      f"got={res['level_probs'][li]:.4f}  "
                      f"err={res['level_probs'][li] - targets[li]:+.4f}  "
                      f"L1={np.abs(res['levels'][li] - res['x0']).mean():.4f}")

        if n_new % args.flush_every == 0:
            pd.DataFrame(rows).to_csv(man_path, index=False)

    # ── finish ────────────────────────────────────────────────────────────────
    man = pd.DataFrame(rows)
    man.to_csv(man_path, index=False)
    print(f"\nWrote {man_path}  ({len(man):,} rows, {n_new:,} new images)")
    print(f"Elapsed: {(time.time() - t_wall) / 60:.1f} min")

    if len(man):
        print("\nPer level:")
        print(man.groupby(["level", "level_name"]).agg(
            n=("path", "size"),
            mean_target=("target_prob", "mean"),
            mean_cf_prob=("cf_prob", "mean"),
            bias=("err", "mean"),
            mean_abs_err=("err", lambda s: s.abs().mean()),
            flip_rate=("flipped", "mean"),
            mean_L1=("l1", "mean"),
        ).round(4).to_string())


if __name__ == "__main__":
    main()
