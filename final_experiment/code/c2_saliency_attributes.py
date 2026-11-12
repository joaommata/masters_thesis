"""
c2_saliency_attributes.py
=========================
Saliency attributes: C0's Grad-CAMs turned into named, per-organ numbers, so that
saliency_explainability.ipynb can rank, profile and draw them exactly like the C1
attributes in attribute_explainability.ipynb.

Each query is paired with its NEAREST counterfactual (slot 0 of the pairing, the
same at every K), over all five test folds. Every attribute is written in the
attribute notebook's three forms:

    <attr>           the image's value, from its own CAM and its own organ masks
    cf_attr_<attr>   the counterfactual's value, from ITS CAM and ITS masks
    delta_<attr>     image - counterfactual

Each image is measured against its own anatomy, so for a KNN counterfactual (a
different patient) the delta says how the evidence moved between organs, which a
pixel-wise map difference of two unaligned X-rays cannot.

Per map m in {pos, neg} (cam_pos = evidence FOR the disease, cam_neg = AGAINST),
the map bilinearly upsampled to 224 as c2_train_images._load_cam does:

    <Region>_<m>_share        share of the map's mass inside the region's mask
    <Region>_<m>_enrichment   share / the region's area fraction: > 1 = more focused
                              there than an evenly spread map; NaN if the region is absent
    CAM_<m>_scale             the map's raw maximum before normalisation (tracks the logit)
    CAM_<m>_peak              share of the mass in the brightest of the 49 cells
    CAM_<m>_entropy           entropy of the 49 cells, / log 49: 1 = evenly spread
    CAM_<m>_centroid_x / _y   mass centroid, 0..1 (x = 0 is the image's left = patient's right)
    Lung asymmetry_<m>        Left Lung share - Right Lung share
    CAM_balance               (scale_pos - scale_neg) / (scale_pos + scale_neg), in [-1, 1]

An empty map (scale 0: cardiomegaly has no FOR evidence on 56% of its images) has
no shares, peak, entropy or centroid: those are NaN, not 0. CAM_<m>_scale = 0 still
records that it was empty.

Regions overlap (option 1): see c1_region_masks.py, which this script reads. The
pixel algebra uses its 49-vectors: mass in region r = w_r @ cam.

    python c2_saliency_attributes.py --disease cardiomegaly
Output: C2_final_results/<disease>/<source>/saliency_attributes.csv
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import RESULTS_DIR, load_folds, add_cf_args, cf_source_from_args  # noqa: E402

MASKS = os.path.join(RESULTS_DIR, "C1_attributes", "final_experiment", "region_masks",
                     "region_masks.npz")
CAM_SPLITS = ("C2_dataset", "Original_Test", "CF_levels")       # as c2_train_images
MAPS = ("pos", "neg")
EPS = 1e-12


def load_cam(cam_base, path):
    """(2,49) normalised maps [pos, neg] and (2,) raw maxima, or None if missing."""
    fname = path.replace("/", "_") + ".npz"
    for split in CAM_SPLITS:
        fp = os.path.join(cam_base, split, fname)
        if os.path.exists(fp):
            z = np.load(fp)
            return (np.stack([z["cam_pos"].ravel(), z["cam_neg"].ravel()]),
                    np.array([z["scale_pos"], z["scale_neg"]], dtype=np.float64))
    return None


def load_cams(cam_base, paths, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        got = list(ex.map(lambda p: load_cam(cam_base, p), paths))
    missing = [p for p, g in zip(paths, got) if g is None]
    if missing:
        raise FileNotFoundError(f"{len(missing)} paths have no Grad-CAM under {cam_base}, "
                                f"e.g. {missing[:3]}")
    cams = np.stack([g[0] for g in got]).astype(np.float64)    # (n,2,49)
    scales = np.stack([g[1] for g in got])                      # (n,2)
    return cams, scales


def pixel_vectors(size, grid):
    """Per CAM cell: total upsampled mass, and its mass-weighted x and y (0..1)."""
    eye = torch.eye(grid * grid, dtype=torch.float64).view(-1, 1, grid, grid)
    up = F.interpolate(eye, (size, size), mode="bilinear", align_corners=False)[:, 0]
    coord = (torch.arange(size, dtype=torch.float64) + 0.5) / size
    return (up.sum(dim=(1, 2)).numpy(),
            (up * coord[None, None, :]).sum(dim=(1, 2)).numpy(),
            (up * coord[None, :, None]).sum(dim=(1, 2)).numpy())


def features(cams, scales, w, area, regions, u_tot, u_x, u_y):
    """cams (n,2,49), scales (n,2), w (n,R,49), area (n,R) -> DataFrame of attributes."""
    out = {}
    for j, m in enumerate(MAPS):
        c = cams[:, j]                                          # (n,49)
        total = c @ u_tot
        ok = total > EPS                                        # an all-zero map has no shares
        share = np.einsum("nrk,nk->nr", w, c) / np.where(ok, total, np.nan)[:, None]
        enrich = share / np.where(area > 0, area, np.nan)
        for r, name in enumerate(regions):
            out[f"{name}_{m}_share"] = share[:, r]
            out[f"{name}_{m}_enrichment"] = enrich[:, r]
        q = c / np.where(c.sum(1) > EPS, c.sum(1), np.nan)[:, None]
        out[f"CAM_{m}_scale"] = scales[:, j]
        out[f"CAM_{m}_peak"] = q.max(axis=1)
        out[f"CAM_{m}_entropy"] = -(q * np.log(np.clip(q, EPS, None))).sum(axis=1) / np.log(q.shape[1])
        out[f"CAM_{m}_centroid_x"] = (c @ u_x) / np.where(ok, total, np.nan)
        out[f"CAM_{m}_centroid_y"] = (c @ u_y) / np.where(ok, total, np.nan)
        out[f"Lung asymmetry_{m}"] = (share[:, regions.index("Left Lung")]
                                      - share[:, regions.index("Right Lung")])
    # Bounded: one direction is often empty (cardiomegaly: 56% of images have scale_pos = 0),
    # and a log ratio would put those at -16.
    s_sum = scales.sum(axis=1)
    out["CAM_balance"] = (scales[:, 0] - scales[:, 1]) / np.where(s_sum > 0, s_sum, np.nan)
    return pd.DataFrame(out)


def run(args):
    src = cf_source_from_args(args)
    cam_base = os.path.join(RESULTS_DIR, "C0_final", f"multilabel_{args.policy}",
                            args.disease, "gradcam")
    print(f"  {args.disease} | {src.describe()} | nearest CF of the K={args.cf_count} pairing")

    pairs = []
    for i, _, te in load_folds(args.disease, src, args.cf_count, materialize=False,
                               columns=["path", "patient_id", "correct", "cf_paths"]):
        pairs.append(pd.DataFrame({"path": te["path"].values,
                                   "cf_path": te["cf_paths"].astype(str).str.split("|").str[0]
                                                              .str.strip().values,
                                   "fold": i}))
    pairs = pd.concat(pairs, ignore_index=True)
    print(f"  {len(pairs):,} queries over {pairs['fold'].nunique()} folds")

    z = np.load(args.masks)
    regions = [str(r) for r in z["regions"]]
    row_of = pd.Series(np.arange(len(z["paths"])), index=z["paths"].astype(str))
    ok_mask = z["ok"].astype(bool)
    u_tot, u_x, u_y = pixel_vectors(int(z["size"]), int(z["grid"]))

    frames = {}
    for form, col in [("image", "path"), ("cf", "cf_path")]:
        paths = pairs[col].values
        idx = row_of.reindex(paths).values
        have = ~np.isnan(idx)
        if (~have).any():
            raise KeyError(f"{(~have).sum()} {form} paths have no region masks "
                           f"(e.g. {paths[~have][:3]}): segment them with c1_region_masks.py")
        idx = idx.astype(int)
        cams, scales = load_cams(cam_base, list(paths))
        f = features(cams, scales, z["w"][idx].astype(np.float64), z["area"][idx].astype(np.float64),
                     regions, u_tot, u_x, u_y)
        f.loc[~ok_mask[idx]] = np.nan                           # unreadable image: no attributes
        frames[form] = f
        print(f"  {form}: {f.shape[1]} attributes, {(~ok_mask[idx]).sum()} unreadable images")

    attrs = list(frames["image"].columns)
    out = pd.concat([pairs,
                     frames["image"],
                     frames["cf"].add_prefix("cf_attr_"),
                     (frames["image"] - frames["cf"]).add_prefix("delta_")], axis=1)
    path = args.out or os.path.join(RESULTS_DIR, "C2_final_results", args.disease, src.name,
                                    "saliency_attributes.csv")
    out.to_csv(path, index=False)
    print(f"  wrote {path}: {len(out):,} rows x {len(attrs)} attributes x 3 forms")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--policy", default="ignore")
    ap.add_argument("--masks", default=MASKS, help="c1_region_masks.py output")
    ap.add_argument("--out", default=None, help="default: <source dir>/saliency_attributes.csv")
    add_cf_args(ap)
    ap.set_defaults(cf_count=10)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
