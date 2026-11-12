"""
xai_focus_descriptor.py
=======================
Methods figure: how one X-ray's Grad-CAM becomes the per-region focus F_r of
saliency_explainability.ipynb, with every number the real one.

    a  the regions r (C1 PSPNet masks, c1_region_masks.py) and their area a_r
    b  the map supporting C0's call, its share s_r per region, F_r = s_r / a_r
    c  the map against the call, same table
    d  F_r of both maps per region, log scale, 1 = an evenly spread map

Everything is recomputed here from the image and C0's stored Grad-CAM with the
pipeline's exact steps (xrv preprocessing, PSPNet >= 0 at 512, nearest to 224,
lung halves at the bbox mid-row, CAM bilinear to 224, align_corners=False), then
checked against region_masks.npz and saliency_attributes.csv: the script stops if
they disagree.

Default case: the cardiomegaly thesis pick TP85478 (true positive, fold 4), whose
key regions are the notebook's SEL for positive calls.

    python final_experiment/code/xai_focus_descriptor.py
    python final_experiment/code/xai_focus_descriptor.py --path CheXpert-v1.0-small/... --disease effusion \
        --regions Hila "Right Lung lower" Outside
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from PIL import Image
from skimage.io import imread

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
from c1_region_masks import REGIONS, SIZE, GRID, lung_halves         # noqa: E402
from thesis_style import *                                             # noqa: E402,F401,F403

REPO      = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS   = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
DATA_ROOT = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
MASKS     = os.path.join(RESULTS, "C1_attributes", "final_experiment", "region_masks",
                         "region_masks.npz")
CAM_SPLITS = ("C2_dataset", "Original_Test", "CF_levels")
OUT_DIR   = os.path.join(REPO, "final_experiment", "figures", "xai", "methods")
THESIS_DIR = os.path.join(REPO, "MSc_Dissertation-13", "Images", "Methods", "xai")

DEFAULT_PATH = "CheXpert-v1.0-small/train/patient35693/study3/view1_frontal.jpg"
DEFAULT_REGIONS = ["Hila", "Heart", "Left Lung lower", "Outside"]

# The saliency notebook's colours: organ groups, PRGn green = support, purple = against
ORGAN = {"Heart": "#C0504D", "Lung": "#4F81BD", "Hila": "#8064A2", "Hilus": "#8064A2",
         "Mediastinum": "#F79646", "Bones": "#9BBB59", "Spine": "#9BBB59",
         "Facies Diaphragmatica": "#4BACC6", "Outside": "#595959"}
SHORT = {"Left Lung lower": "L lung lower", "Left Lung upper": "L lung upper",
         "Right Lung lower": "R lung lower", "Right Lung upper": "R lung upper",
         "Facies Diaphragmatica": "Diaphragm", "Heart-Left Lung overlap": "Heart–L lung"}
TICK = {"Left Lung lower": "L lung\nlower", "Left Lung upper": "L lung\nupper",
        "Right Lung lower": "R lung\nlower", "Right Lung upper": "R lung\nupper",
        "Outside": "Out-\nside", "Facies Diaphragmatica": "Diaphr.",
        "Heart-Left Lung overlap": "Heart–\nL lung"}
SUP_C, OPP_C = plt.get_cmap("PRGn")(0.85), plt.get_cmap("PRGn")(0.15)
CAM_ALPHA, DISPLAY = 0.45, 320
FADE = {k: LinearSegmentedColormap.from_list(k, [to_rgba(c, 0), to_rgba(c, CAM_ALPHA + 0.3)])
        for k, c in [("sup", SUP_C), ("opp", OPP_C)]}
F_FLOOR = 0.01                    # the notebook's EPS_FOCUS: focus below 1% of even counts as 0.01


def organ_color(r):
    return next((c for k, c in ORGAN.items() if k in r), "#A6A6A6")


# ── Regions, exactly as c1_region_masks.segment builds them ──
@torch.no_grad()
def region_masks(path):
    img = xrv.datasets.normalize(imread(os.path.join(DATA_ROOT, path)), maxval=255, reshape=True)
    x = torch.from_numpy(xrv.datasets.XRayResizer(SIZE)(img)).float()[None]
    model = xrv.baseline_models.chestx_det.PSPNet().eval()
    logits = model(x)
    raw = F.interpolate((logits >= 0).float(), (SIZE, SIZE), mode="nearest") > 0.5
    t = {n: raw[:, j] for j, n in enumerate(model.targets)}
    reg = {}
    for name, src in REGIONS.items():
        if src:
            m = t[src[0]].clone()
            for s in src[1:]:
                m |= t[s]
            reg[name] = m
    reg["Left Lung upper"], reg["Left Lung lower"] = lung_halves(t["Left Lung"])
    reg["Right Lung upper"], reg["Right Lung lower"] = lung_halves(t["Right Lung"])
    reg["Heart-Left Lung overlap"] = t["Heart"] & t["Left Lung"]
    reg["Outside"] = ~raw.any(dim=1)
    return {n: reg[n][0].numpy() for n in REGIONS}


def load_cam(cam_base, path):
    fname = path.replace("/", "_") + ".npz"
    for split in CAM_SPLITS:
        fp = os.path.join(cam_base, split, fname)
        if os.path.exists(fp):
            z = np.load(fp)
            return z["cam_pos"].astype(np.float64), z["cam_neg"].astype(np.float64)
    raise FileNotFoundError(f"no Grad-CAM for {path} under {cam_base}")


def upsample(cam):
    """(7,7) -> (224,224), the call c2_train_images._load_cam makes."""
    t = torch.from_numpy(cam)[None, None]
    return F.interpolate(t, (SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()


def focus_table(masks, cam, regions):
    up = upsample(cam)
    tot = up.sum()
    rows = []
    for r in regions:
        m = masks[r].astype(np.float64)
        a = m.mean()
        s = (m * up).sum() / tot if tot > 0 else np.nan
        rows.append((r, a, s, s / a if a > 0 else np.nan))
    return pd.DataFrame(rows, columns=["region", "area", "share", "focus"]).set_index("region")


def check(path, masks, tab_pos, tab_neg, disease, regions):
    """Stop unless the recomputed numbers are the pipeline's."""
    z = np.load(MASKS)
    i = int(np.where(z["paths"].astype(str) == path)[0][0])
    names = [str(r) for r in z["regions"]]
    for r in regions:
        a_npz = float(z["area"][i, names.index(r)])
        assert abs(a_npz - masks[r].mean()) < 1e-4, f"{r}: area {masks[r].mean():.5f} vs npz {a_npz:.5f}"
    sal = pd.read_csv(os.path.join(RESULTS, "C2_final_results", disease, "knn_correct_cf",
                                  "saliency_attributes.csv"),
                      usecols=lambda c: c in ["path", "fold"] or (not c.startswith(("cf_attr_", "delta_"))
                                                                  and c.endswith("_enrichment")))
    row = sal[sal["path"] == path]
    assert len(row) == 1, f"{path} is not a query in saliency_attributes.csv"
    row = row.iloc[0]
    for m, tab in [("pos", tab_pos), ("neg", tab_neg)]:
        for r in regions:
            ref, got = row[f"{r}_{m}_enrichment"], tab.loc[r, "focus"]
            assert np.isclose(ref, got, rtol=1e-3, atol=1e-5, equal_nan=True), \
                f"{r} {m}: focus {got:.5f} vs csv {ref:.5f}"
    print(f"  checked against region_masks.npz and saliency_attributes.csv (fold {int(row['fold'])})")


# ── Figure ──
def xray(path):
    return (Image.open(os.path.join(DATA_ROOT, path)).convert("L")
            .resize((DISPLAY, DISPLAY), Image.BILINEAR))


def draw_xray(ax, img, masks, regions, cam=None, kind=None, fill=0.18, lw=0.8):
    ext = (-0.5, SIZE - 0.5, SIZE - 0.5, -0.5)                  # everything on the 224 grid
    ax.imshow(img, cmap="gray", extent=ext)
    if cam is not None:
        ax.imshow(cam, cmap=FADE[kind], vmin=0, vmax=1, interpolation="bilinear", extent=ext)
    for r in regions:
        m = masks[r].astype(float)
        if r == "Outside":                                      # outside: hatched, no outline
            if cam is None:
                ax.contourf(m, levels=[0.5, 1.5], colors="none", hatches=["///"])
            continue
        c = organ_color(r)
        if cam is None:
            ax.contourf(m, levels=[0.5, 1.5], colors=[to_rgba(c, fill)])
        ax.contour(m, levels=[0.5], colors=[c], linewidths=lw)
    ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); ax.axis("off")


def title(ax, letter, head, sub):
    ax.text(0, 1.20, letter, transform=ax.transAxes, fontsize=FS["title"], fontweight="bold",
            va="bottom")
    ax.text(0.09, 1.20, head, transform=ax.transAxes, fontsize=FS["title"], fontweight="bold",
            va="bottom")
    ax.text(0, 1.17, sub, transform=ax.transAxes, fontsize=FS["tiny"], color=MUTED, va="top",
            linespacing=1.2)


def table(ax, rows, head):
    """Small table under an X-ray: region name in its colour, right-aligned value."""
    y0, dy = -0.07, 0.085
    ax.text(0, y0, "region", transform=ax.transAxes, fontsize=FS["tiny"], color=MUTED, va="top")
    ax.text(1, y0, head, transform=ax.transAxes, fontsize=FS["tiny"], color=MUTED, va="top",
            ha="right")
    for k, (r, val) in enumerate(rows):
        y = y0 - (k + 1) * dy
        c = organ_color(r) if r != "Outside" else INK
        ax.text(0, y, SHORT.get(r, r), transform=ax.transAxes, fontsize=FS["tiny"], color=c,
                fontweight="bold", va="top")
        ax.text(1, y, val, transform=ax.transAxes, fontsize=FS["tiny"], color=INK, va="top",
                ha="right")


def pct(v):
    return f"{100 * v:.1f}%"


def fval(v):
    if not np.isfinite(v):
        return "–"
    return f"{v:.2f}" if v >= 0.1 else f"{v:.3f}" if v >= 0.001 else f"{v:.0e}".replace("e-0", "e-")


def share_rows(tab, regions):
    return [(r, f"{pct(tab.loc[r, 'share'])} / {pct(tab.loc[r, 'area'])} = {fval(tab.loc[r, 'focus'])}")
            for r in regions]


def draw_focus(ax, sup, opp, regions):
    lo = F_FLOOR
    ax.set_yscale("log")
    ax.axhline(1, color=INK, lw=LW["axis"], zorder=1)
    if "Outside" in regions:
        xi = regions.index("Outside")
        ax.axvspan(xi - 0.5, xi + 0.5, color="#000000", alpha=0.06, lw=0, zorder=0)
    floored = False
    for xi, r in enumerate(regions):
        c = organ_color(r) if r != "Outside" else INK
        s, o = max(sup[r], lo), max(opp[r], lo)
        floored |= opp[r] < lo or sup[r] < lo
        ax.plot([xi, xi], [s, o], color=c, lw=2.4, alpha=0.45, solid_capstyle="butt", zorder=2)
        ax.scatter(xi, s, s=26, color=c, edgecolor="white", lw=0.6, zorder=3)
        ax.scatter(xi, o, s=30, marker="X", color=c, edgecolor="white", lw=0.6, zorder=3)
        for v, raw in [(s, sup[r]), (o, opp[r])]:
            if raw < lo:
                ax.annotate(f"{fval(raw)}", (xi, v), xytext=(4, 0), textcoords="offset points",
                            fontsize=FS["tiny"] - 0.5, color=MUTED, va="center")
    vals = np.r_[[max(sup[r], lo) for r in regions], [max(opp[r], lo) for r in regions]]
    ax.set_ylim(min(vals.min(), 0.1) / 1.6, max(vals.max(), 10) * 1.3)
    ax.set_xlim(-0.6, len(regions) - 0.4)
    ax.set_xticks(range(len(regions)), [TICK.get(r, r) for r in regions], fontsize=FS["tiny"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="y", labelsize=FS["tiny"])
    ax.set_ylabel(r"focus $F_r$", fontsize=FS["tiny"], labelpad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(False)
    return floored


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", default="cardiomegaly")
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    ap.add_argument("--policy", default="ignore")
    ap.add_argument("--name", default="saliency_focus_descriptor")
    ap.add_argument("--no-thesis-copy", action="store_true")
    args = ap.parse_args()
    apply()
    plt.rcParams.update({"hatch.color": "#D9D9D9", "hatch.linewidth": 0.35})

    c0_dir = os.path.join(RESULTS, "C0_final", f"multilabel_{args.policy}", args.disease)
    c0 = pd.read_csv(os.path.join(c0_dir, f"C2_dataset_c0_{args.disease}.csv"),
                     usecols=["path", "prob", "true", "pred"]).set_index("path").loc[args.path]
    call = int(c0["pred"])
    outcome = {(1, 1): "true positive", (1, 0): "false positive",
               (0, 0): "true negative", (0, 1): "false negative"}[(call, int(c0["true"]))]
    print(f"  {args.path}\n  C0 p = {c0['prob']:.4f}, call {call}, {outcome}")

    masks = region_masks(args.path)
    cam_pos, cam_neg = load_cam(os.path.join(c0_dir, "gradcam"), args.path)
    tab_pos = focus_table(masks, cam_pos, args.regions)
    tab_neg = focus_table(masks, cam_neg, args.regions)
    check(args.path, masks, tab_pos, tab_neg, args.disease, args.regions)
    sup, opp = (tab_pos, tab_neg) if call == 1 else (tab_neg, tab_pos)
    sup_cam, opp_cam = (cam_pos, cam_neg) if call == 1 else (cam_neg, cam_pos)
    print("  support:\n" + sup.to_string(float_format="%.4f"))
    print("  against:\n" + opp.to_string(float_format="%.4f"))

    img = xray(args.path)
    fig = plt.figure(figsize=(FULL, 58 * MM))
    W, H = FULL / MM, 58
    side, top, left, gap = 37, 10, 4, 8.5
    box = lambda x, w: [x / W, 1 - (top + side) / H, w / W, side / H]
    axes = [fig.add_axes(box(left + k * (side + gap), side)) for k in range(3)]
    dax = fig.add_axes([(left + 3 * (side + gap) + 7) / W, 1 - (top + side) / H,
                        (W - left - 3 * (side + gap) - 9) / W, side / H])

    dis = args.disease.capitalize()
    draw_xray(axes[0], img, masks, args.regions)
    title(axes[0], "a", "Regions $r$", "C1 masks, lungs halved;\nhatched = outside every mask")
    table(axes[0], [(r, pct(sup.loc[r, "area"])) for r in args.regions], "share of image $a_r$")

    draw_xray(axes[1], img, masks, args.regions, upsample(sup_cam), "sup")
    title(axes[1], "b", "Support map", f"evidence for C0's call\n({dis} {'present' if call else 'absent'})")
    table(axes[1], share_rows(sup, args.regions), "$s_r$ / $a_r$ = $F_r$")

    draw_xray(axes[2], img, masks, args.regions, upsample(opp_cam), "opp")
    title(axes[2], "c", "Against map", "evidence against the call")
    table(axes[2], share_rows(opp, args.regions), "$s_r$ / $a_r$ = $F_r$")

    floored = draw_focus(dax, sup["focus"], opp["focus"], args.regions)
    title(dax, "d", "Focus per region",
          "$F_r$ from b and c, log scale;\n1 = evenly spread map" + ("; floor 0.01" if floored else ""))
    dax.legend(handles=[Line2D([], [], marker="o", lw=0, color=INK, ms=4, label="support for the call"),
                        Line2D([], [], marker="X", lw=0, color=INK, ms=4, label="against the call")],
               loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=False,
               fontsize=FS["tiny"], handletextpad=0.3, borderaxespad=0)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, args.name)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)
    caption = (f"{args.path} | {args.disease}, C0 p = {c0['prob']:.4f}, call {call} ({outcome})\n"
               f"Masks recomputed with the C1 PSPNet, Grad-CAM from {c0_dir}/gradcam; "
               f"focus checked against saliency_attributes.csv (knn_correct_cf).\n\n"
               f"SUPPORT MAP\n{sup.to_string(float_format='%.5f')}\n\n"
               f"AGAINST MAP\n{opp.to_string(float_format='%.5f')}\n")
    with open(f"{stem}.txt", "w", encoding="utf-8") as f:
        f.write(caption)
    print(f"  wrote {stem}.{{pdf,png,txt}}")
    if not args.no_thesis_copy and os.path.isdir(THESIS_DIR):
        for ext in ("pdf", "png", "txt"):
            shutil.copy(f"{stem}.{ext}", THESIS_DIR)
        print(f"  copied to {THESIS_DIR}")


if __name__ == "__main__":
    main()
