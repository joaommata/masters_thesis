"""Soft targets and the counterfactuals that reach them, for one image, from saved CF_levels output.

    python viz/plot_soft_targets.py --disease cardiomegaly --key CheXpert-v1.0-small_train_patient03956_study1_view1_frontal.jpg

a  where the low / mid / high targets sit in the C0 probability distribution, with the
   original's probability and the jump to each target;
b  the guided trajectories (C0 probability over the reverse diffusion) converging on them;
c  the original and the three counterfactuals;
d  (--seg) the same images with the C1 PSPNet organ outlines, the original's outline dashed on
   each counterfactual, and the change in the disease's organ area.

No GPU: everything comes from $THESIS_RESULTS/CF_levels/<disease>/ (cf_generate_levels.py),
so the trajectories have that run's --track-every resolution (5 steps). For per-step
trajectories regenerate the image in soft_targets_trajectory.ipynb.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

FE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(FE, "code"))
from thesis_style import *
import cf_generation_final as cf

apply()
LEVELS = ["low", "mid", "high"]

# Organ colours and grouping as in viz_segmentations.ipynb; the organ whose area each disease moves.
ORGAN = {"Lungs": "#4F81BD", "Heart": "#C0504D", "Diaphragm": "#4BACC6"}
ORGAN_OF = {"Left Lung": "Lungs", "Right Lung": "Lungs", "Heart": "Heart",
            "Facies Diaphragmatica": "Diaphragm"}
AREA_OF = {"cardiomegaly": "Heart", "effusion": "Lungs"}
# Row d is zoomed to where the disease acts: (x0, x1, y0, y1) in the 224 px image.
ZOOM = {"cardiomegaly": (60, 190, 65, 195),        # heart
        "effusion": (104, 220, 102, 218)}          # patient's left base and hemidiaphragm


def segment(imgs):
    """PSPNet (xrv chestx_det) organ masks for [0, 1] images, as cf.segment_image: 0.5 threshold."""
    import torch, torch.nn.functional as F, torchxrayvision as xrv
    model = xrv.baseline_models.chestx_det.PSPNet().eval()
    x = torch.from_numpy(np.stack(imgs)[:, None].astype(np.float32)) * 2048.0 - 1024.0
    with torch.no_grad():
        out = torch.sigmoid(model(x))
    out = (F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False) >= 0.5).numpy()
    T = list(model.targets)
    return [{o: np.any([m[T.index(s)] for s, g in ORGAN_OF.items() if g == o], axis=0) for o in ORGAN}
            for m in out]


def load(disease, key):
    cfg = cf.TaskConfig(disease, device="cpu")
    root = os.path.join(cf.RESULTS_DIR, "CF_levels", disease)
    z = np.load(os.path.join(root, "cf", key + ".npz"))
    tj = np.load(os.path.join(root, "trajectory", key + ".npz"))
    with open(os.path.join(root, "levels.json")) as f:
        tau = json.load(f)["tau"]
    df = pd.read_csv(cfg.c2_csv, usecols=["prob", "true"])
    return cfg, z, [tj[f"level_{i + 1}"] for i in range(3)], tau, df


def plot(disease, key, seg=False):
    cfg, z, trajs, tau, df = load(disease, key)
    thr, p0 = cfg.threshold, float(z["orig_prob"])
    targets, reached = z["targets"], z["level_probs"]
    imgs = [z["x0"]] + [z[f"cf_level_{i + 1}"] for i in range(3)]
    to_pos = str(z["direction"]) == "to_positive"
    # positives blue, negatives green; the levels take the ramp of the class they travel to
    LV = RAMP["attr"] if to_pos else RAMP["traj"]
    name = disease.replace("_", " ")

    fig = plt.figure(figsize=(FULL, (170 if seg else 116) * MM))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.15, 0.3, 2.2 if seg else 1], hspace=0.06, wspace=0.1)
    rows = gs[2, :].subgridspec(2 if seg else 1, 4, hspace=0.16, wspace=0.1)
    top = gs[0, :].subgridspec(1, 2, wspace=0.22)          # room for b's y-label
    ax_a = fig.add_subplot(top[0])
    ax_b = fig.add_subplot(top[1])
    fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.05)

    # ── a: the targets in the C0 probability distribution ────────────────────
    grid = np.linspace(0, 1, 800)
    for side, col in ((0, FAMILY["traj"]), (1, FAMILY["attr"])):
        p = df.loc[(df.true == side) & ((df.prob > thr) if side else (df.prob <= thr)), "prob"].values
        d = gaussian_kde(p, bw_method=0.15)(grid)
        d = np.where((grid > thr) if side else (grid <= thr), d, np.nan)
        d = 0.72 * d / np.nanmax(d)
        moving = (side == 1) == to_pos
        ax_a.fill_between(grid, 0, d, color=col, alpha=0.18 if moving else 0.08, lw=0)
        ax_a.plot(grid, d, color=col, lw=LW["line"] if moving else LW["ref"] * 1.3,
                  alpha=1 if moving else 0.5)
    ax_a.axvline(thr, color=INK, lw=LW["ref"] * 1.3, ls=(0, (3, 2)), zorder=3)
    ax_a.axvline(p0, color=MUTED, lw=LW["ref"] * 1.3, ls=":", zorder=1)
    for i, (t, c) in enumerate(zip(targets, LV)):
        ax_a.axvline(t, color=c, lw=LW["line"] * 1.6, zorder=2)
        y = 0.80 + 0.07 * (2 - i)
        ax_a.annotate("", xy=(t, y), xytext=(p0, y),
                      arrowprops=dict(arrowstyle="-|>", color=c, lw=LW["line"], shrinkA=3, shrinkB=0,
                                      mutation_scale=7))
        ax_a.plot(p0, y, "o", ms=4, mfc="white", mec=INK, mew=1, zorder=5)
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
    ax_a.set_yticks([]); ax_a.set_ylabel("density (scaled)")
    ax_a.set_xlabel(f"$C_0$ probability of {name}")
    ax_a.set_title("a   Soft targets", fontsize=FS["head"])

    # ── b: the guided trajectories ────────────────────────────────────────────
    ax_b.yaxis.grid(True, color=HAIR, lw=LW["hair"] * 1.5)
    ax_b.axhline(thr, color=INK, lw=LW["ref"] * 1.3, ls=(0, (3, 2)), zorder=2)
    ax_b.axhline(p0, color=MUTED, lw=LW["ref"] * 1.3, ls=":", zorder=1)
    for t, pr, tr, c in zip(targets, reached, trajs, LV):
        ax_b.axhline(t, color=c, lw=LW["ref"], zorder=1)
        x = np.r_[tau, tr[:, 0], 0]
        y = np.r_[p0, tr[:, 1], pr]
        ax_b.plot(x, y, color=c, lw=LW["line"] * 1.4, zorder=3)
        ax_b.plot(0, pr, "o", ms=5, color=c, mec=INK, mew=0.8, zorder=6, clip_on=False)
    ax_b.plot(tau, p0, "o", ms=5, mfc="white", mec=INK, mew=1, zorder=6, clip_on=False)
    ax_b.set_xlim(tau, 0); ax_b.set_ylim(0, 1)
    ax_b.set_xlabel("diffusion timestep $t$")
    ax_b.set_ylabel(f"$C_0$ probability of {name}")
    ax_b.set_title("b   Guided counterfactuals", fontsize=FS["head"])

    # ── legend row ────────────────────────────────────────────────────────────
    lax = fig.add_subplot(gs[1, :]); lax.axis("off")
    hs = [Line2D([], [], color=c, lw=LW["line"] * 1.6, label=n) for n, c in zip(LEVELS, LV)]
    hs += [Line2D([], [], color=INK, lw=LW["ref"] * 1.3, ls=(0, (3, 2)), label="$C_0$ threshold"),
           Line2D([], [], color=MUTED, lw=LW["ref"] * 1.3, ls=":", marker="o", ms=4, mfc="white",
                  mec=INK, label="original")]
    leg = lax.legend(handles=hs, loc="center", bbox_to_anchor=(0.5, 0.27), ncol=5, fontsize=FS["base"], handlelength=2.2,
                     columnspacing=1.6)
    for txt, c in zip(leg.get_texts()[:3], LV):
        txt.set_color(c); txt.set_fontweight("bold")

    # ── c: original and counterfactuals ───────────────────────────────────────
    probs = [p0, *reached]
    zoom = ZOOM.get(disease) if seg else None
    for j, img in enumerate(imgs):
        a = fig.add_subplot(rows[0, j])
        if zoom:
            x0, x1, y0, y1 = zoom
            a.add_patch(plt.Rectangle((x0, y0), x1 - x0 - 1, y1 - y0 - 1, fill=False, ec="white",
                                      lw=0.8, ls=(0, (2, 1.5))))
        a.imshow(img, cmap="gray", vmin=0, vmax=1)
        a.set_xticks([]); a.set_yticks([])
        col = INK if j == 0 else LV[j - 1]
        for s in a.spines.values():
            s.set_visible(True); s.set_edgecolor(col); s.set_linewidth(1.0 if j == 0 else 2.2)
        a.set_title("original" if j == 0 else LEVELS[j - 1], loc="center", color=col,
                    fontsize=FS["head"])
        pos = probs[j] > thr
        a.text(0.03, 0.97, name if pos else f"no {name}", transform=a.transAxes, va="top", ha="left",
               fontsize=FS["base"], fontweight="bold", color="white",
               bbox=dict(boxstyle="round,pad=0.25", fc=FAMILY["attr"] if pos else FAMILY["traj"], lw=0))
        a.set_xlabel(f"$p = {probs[j]:.2f}$" + ("" if j == 0 else f"  (target {targets[j - 1]:.2f})"),
                     fontsize=FS["base"], labelpad=3)
        if j == 0:
            a.text(-0.08, 1.12, "c", transform=a.transAxes, fontsize=FS["head"] + 1,
                   fontweight="bold", va="bottom", ha="left")
    if seg:
        draw_seg(fig, rows, imgs, LV, disease, zoom)
    return fig


def draw_seg(fig, rows, imgs, LV, disease, zoom=None):
    masks = segment(imgs)
    organ = AREA_OF.get(disease)
    a0 = masks[0][organ].sum() if organ else None
    for j, (img, m) in enumerate(zip(imgs, masks)):
        a = fig.add_subplot(rows[1, j])
        a.imshow(img, cmap="gray", vmin=0, vmax=1)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(True); s.set_edgecolor(INK if j == 0 else LV[j - 1])
            s.set_linewidth(1.0 if j == 0 else 2.2)
        for o, c in ORGAN.items():
            if j and masks[0][o].any():
                a.contour(masks[0][o], levels=[0.5], colors="white", linewidths=0.9 if zoom else 0.6,
                          linestyles=[(0, (2, 1.5))])
            if m[o].any():
                a.contour(m[o], levels=[0.5], colors=[c], linewidths=1.4 if zoom else 1.0)
        if zoom:
            a.set_xlim(zoom[0] - 0.5, zoom[1] - 0.5); a.set_ylim(zoom[3] - 0.5, zoom[2] - 0.5)
        if organ:
            a.set_xlabel(f"{organ.lower()} area" + ("" if j == 0 else
                         f"  {100 * (m[organ].sum() / a0 - 1):+.1f}%"), fontsize=FS["base"], labelpad=3)
        if j == 0:
            a.text(-0.08, 1.04, "d", transform=a.transAxes, fontsize=FS["head"] + 1,
                   fontweight="bold", va="bottom", ha="left")
    hs = [Line2D([], [], color=c, lw=1.2, label=o.lower()) for o, c in ORGAN.items()]
    hs += [Line2D([], [], color=MUTED, lw=0.8, ls=(0, (2, 1.5)), label="original's outline")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0.012), ncol=4,
               fontsize=FS["base"], handlelength=2.0, columnspacing=1.6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", default="cardiomegaly")
    ap.add_argument("--key", required=True, help="CF_levels key: image path with '/' -> '_'")
    ap.add_argument("--out", default=None, help="path without extension")
    ap.add_argument("--seg", action="store_true", help="add row d: organ segmentations")
    args = ap.parse_args()
    fig = plot(args.disease, args.key, seg=args.seg)
    out = args.out or os.path.join(FE, "figures", "diffusion", f"soft_targets_{args.disease}")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300)
    print("saved", out)
