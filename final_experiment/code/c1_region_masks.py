"""
c1_region_masks.py
==================
Organ regions of every C2 image, reduced to what a 7x7 Grad-CAM can read, for the
saliency attributes of c2_saliency_attributes.py.

The Grad-CAMs of c0_final_predictions.py are 7x7 maps that every consumer
upsamples bilinearly to 224 (c2_train_images._load_cam). Bilinear upsampling is
linear, so the saliency mass a 224-px mask m receives is

    sum_px  m * (U @ cam)  =  (U^T m) @ cam ,      U: (224*224, 49)

and the 49-vector w = U^T m is all a region needs: the mask itself never has to be
stored. Per image and region this script writes w and the region's area.

Segmentation is the C1 PSPNet (xrv chestx_det), with C1's preprocessing: xrv
normalize to [-1024, 1024], XRayResizer(224) (a squash, like C0's Resize((224,224)),
so mask and CAM share one grid), mask = sigmoid(logit) >= 0.5 at 512 px,
nearest-resized to 224 as for the C1 radiomics.

Regions are NOT exclusive (option 1): a pixel over the heart shadow counts for the
heart, the left lung and the mediastinum alike, the way the C1 attributes measure
each structure on its full mask. Shares therefore do not sum to 1. `Outside` is the
part of the image in no structure's mask. Merged regions are unions of masks:

    Mediastinum   Mediastinum | Aorta | Weasand
    Hila          Left | Right Hilus Pulmonis
    Bones         Left | Right Clavicle, Left | Right Scapula
    <lung> upper / lower   the lung split at the middle row of its own bounding box
    Heart-Left Lung overlap   Heart & Left Lung

Pairwise intersection areas of the 14 raw structures are stored too, to check how
much the regions overlap.

Masks do not depend on the disease, so this runs once for all diseases. Default
image set: every row of final_experiment/C2_dataset.csv (the KNN counterfactuals are
real images from that pool). Chunked and resume-safe: each chunk is an atomic
part file, and a killed job restarts at the first missing part.

    python c1_region_masks.py                 # all of C2_dataset.csv
    python c1_region_masks.py --limit 64      # smoke test
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset

REPO        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
OUT_DIR     = os.path.join(RESULTS_DIR, "C1_attributes", "final_experiment", "region_masks")
DEFAULT_SPLIT = os.path.join(REPO, "final_experiment", "C2_dataset.csv")

SIZE, GRID = 224, 7

# Region name -> raw PSPNet structures whose union it is. Names keep the C1 structure
# names so the attribute notebook's organ colours (substring match) apply unchanged.
REGIONS = {
    "Left Lung":             ["Left Lung"],
    "Right Lung":            ["Right Lung"],
    "Left Lung upper":       None,          # lung halves: derived below
    "Left Lung lower":       None,
    "Right Lung upper":      None,
    "Right Lung lower":      None,
    "Heart":                 ["Heart"],
    "Heart-Left Lung overlap": None,        # Heart & Left Lung
    "Mediastinum":           ["Mediastinum", "Aorta", "Weasand"],
    "Hila":                  ["Left Hilus Pulmonis", "Right Hilus Pulmonis"],
    "Facies Diaphragmatica": ["Facies Diaphragmatica"],
    "Spine":                 ["Spine"],
    "Bones":                 ["Left Clavicle", "Right Clavicle", "Left Scapula", "Right Scapula"],
    "Outside":               None,          # in no structure's mask
}
REGION_NAMES = list(REGIONS)


def upsample_matrix(device):
    """U^T, (49, 224*224): row j is the bilinear upsampling of the j-th 7x7 basis map,
    with the exact call c2_train_images._load_cam uses."""
    eye = torch.eye(GRID * GRID, device=device).view(-1, 1, GRID, GRID)
    up = F.interpolate(eye, (SIZE, SIZE), mode="bilinear", align_corners=False)
    return up.view(GRID * GRID, SIZE * SIZE)


class Images(Dataset):
    """C1 preprocessing (= xrv.datasets.CheX_Dataset.__getitem__ + XRayResizer(224))."""

    def __init__(self, paths):
        self.paths = paths
        self.resize = xrv.datasets.XRayResizer(SIZE)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = xrv.datasets.normalize(imread(os.path.join(DATA_ROOT, self.paths[i])),
                                         maxval=255, reshape=True)
            return torch.from_numpy(self.resize(img)).float(), True
        except Exception as e:                                  # unreadable: flagged, zeros
            print(f"  [skip] {self.paths[i]}: {e}", flush=True)
            return torch.zeros(1, SIZE, SIZE), False


def lung_halves(lung):
    """(B,H,W) bool -> upper, lower: split at the middle row of each mask's own bbox."""
    rows = lung.any(dim=2)                                      # (B,H)
    idx = torch.arange(lung.shape[1], device=lung.device)
    big = lung.shape[1]
    top = torch.where(rows, idx, big).min(dim=1).values
    bot = torch.where(rows, idx, -1).max(dim=1).values
    mid = ((top + bot) / 2.0)[:, None, None]                    # empty mask: top > bot, both halves empty
    is_upper = idx[None, :, None] <= mid
    return lung & is_upper, lung & ~is_upper


@torch.no_grad()
def segment(model, imgs, UT):
    """imgs (B,1,224,224) -> w (B,R,49), area (B,R) fraction of the image, inter (B,14,14) px."""
    logits = model(imgs)                                        # (B,14,512,512)
    raw = F.interpolate((logits >= 0).float(), (SIZE, SIZE), mode="nearest") > 0.5
    t = {name: raw[:, j] for j, name in enumerate(model.targets)}

    def union(names):
        m = t[names[0]].clone()
        for n in names[1:]:
            m |= t[n]
        return m

    reg = {name: union(src) for name, src in REGIONS.items() if src}
    reg["Left Lung upper"], reg["Left Lung lower"] = lung_halves(t["Left Lung"])
    reg["Right Lung upper"], reg["Right Lung lower"] = lung_halves(t["Right Lung"])
    reg["Heart-Left Lung overlap"] = t["Heart"] & t["Left Lung"]
    reg["Outside"] = ~raw.any(dim=1)
    M = torch.stack([reg[n] for n in REGION_NAMES], dim=1).float()    # (B,R,H,W)

    B, R = M.shape[:2]
    w = M.view(B, R, -1) @ UT.T                                 # (B,R,49)
    area = M.mean(dim=(2, 3))
    flat = raw.view(B, raw.shape[1], -1).float()
    inter = flat @ flat.transpose(1, 2)                         # (B,14,14), diagonal = area
    return w, area, inter


def part_path(start):
    return os.path.join(OUT_DIR, "parts", f"part_{start:07d}.npz")


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(set(pd.read_csv(args.split_csv, usecols=["Path"])["Path"].astype(str)))
    if args.limit:
        paths = paths[:args.limit]
    os.makedirs(os.path.join(OUT_DIR, "parts"), exist_ok=True)
    print(f"  {len(paths):,} images | device {device} | out {OUT_DIR}", flush=True)

    model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    UT = upsample_matrix(device)
    starts = range(0, len(paths), args.chunk)
    for start in starts:
        if os.path.exists(part_path(start)):
            continue
        chunk = paths[start:start + args.chunk]
        loader = DataLoader(Images(chunk), batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)
        W, A, I, OK = [], [], [], []
        for imgs, ok in loader:
            w, area, inter = segment(model, imgs.to(device, non_blocking=True), UT)
            W.append(w.cpu().numpy()); A.append(area.cpu().numpy())
            I.append(inter.cpu().numpy().astype(np.int32)); OK.append(ok.numpy())
        tmp = part_path(start)[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, paths=np.array(chunk), w=np.concatenate(W).astype(np.float32),
                            area=np.concatenate(A).astype(np.float32),
                            inter=np.concatenate(I), ok=np.concatenate(OK))
        os.replace(tmp, part_path(start))
        print(f"  {min(start + args.chunk, len(paths)):,}/{len(paths):,}", flush=True)

    merge(len(paths), args.limit, model.targets)


def merge(n_expected, limit, targets):
    parts = [np.load(p) for p in sorted(glob.glob(os.path.join(OUT_DIR, "parts", "part_*.npz")))]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("paths", "w", "area", "inter", "ok")}
    assert len(out["paths"]) == n_expected, \
        f"{len(out['paths'])} rows in parts, expected {n_expected} (stale parts from another run?)"
    name = "region_masks_smoke.npz" if limit else "region_masks.npz"
    np.savez_compressed(os.path.join(OUT_DIR, name), regions=np.array(REGION_NAMES),
                        structures=np.array(targets), grid=GRID, size=SIZE, **out)
    print(f"  wrote {OUT_DIR}/{name}: {len(out['paths']):,} images, "
          f"{(~out['ok'].astype(bool)).sum()} unreadable")
    overlap_report(out["inter"][out["ok"].astype(bool)], targets)


def overlap_report(inter, targets):
    """How much the raw structures overlap: median |A & B| / min(|A|, |B|) over images
    where both are present, the 12 largest pairs."""
    d = np.diagonal(inter, axis1=1, axis2=2)                    # (n,14) areas
    rows = []
    for a in range(len(targets)):
        for b in range(a + 1, len(targets)):
            both = (d[:, a] > 0) & (d[:, b] > 0)
            if both.any():
                frac = inter[both, a, b] / np.minimum(d[both, a], d[both, b])
                rows.append((targets[a], targets[b], np.median(frac), np.mean(frac > 0.1)))
    rep = pd.DataFrame(rows, columns=["A", "B", "median |A&B|/min", "share of images > 10%"])
    print("\n  Overlap of the raw structures (largest 12 pairs):")
    print(rep.sort_values("median |A&B|/min", ascending=False).head(12)
          .to_string(index=False, float_format="%.3f"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-csv", default=DEFAULT_SPLIT,
                    help="CSV whose Path column is the image set (paths relative to THESIS_DATA)")
    ap.add_argument("--chunk", type=int, default=4096, help="images per part file")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N images only (smoke test)")
    args = ap.parse_args()
    if args.limit:                                  # smoke run: its own parts, never mixed in
        global OUT_DIR
        OUT_DIR = os.path.join(OUT_DIR, "smoke")
    run(args)


if __name__ == "__main__":
    main()
