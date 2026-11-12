"""
cf_extract_diverse_features.py
==============================
Extracts the full scalar feature set for every (Xi, CF) pair in the diverse-CF grid
produced by cf_generate_diverse_counterfactuals.py.

One output row per manifest row, i.e. per (path, region, t_start). Feature groups:

  trajectory   : shape (cubic poly + residual), monotonicity, flip bookkeeping,
                 velocity / acceleration summaries
  diff map     : Xi - CF magnitude (l1/l2/peak/mean), spatial entropy, edited area
  alignment    : for the UNMASKED 'general' CF only — how its edit mass distributes
                 over the four anatomical regions (share / density). Region masks are
                 recovered from the masked CFs' own diff maps, so no segmentation
                 model is needed.

Conventions that matter (both were bugs before — see notebook history):
  * Trajectory .npy rows are ALREADY in denoising order (t_start -> 1) and
    traj[-1, 1] == cf_prob. Never reverse them.
  * Derivatives are taken w.r.t. STEP INDEX, not t_val. t_val decreases by 1 per
    step, so dividing by dt = -1 would flip the sign of every velocity.

Resumable: writes per-shard parquet files keyed on the image slug's hash bucket, and
skips shards that already exist. Concatenates to cf_features.parquet at the end.

Usage:
    python cf_extract_diverse_features.py                  # all images
    python cf_extract_diverse_features.py --limit 20       # smoke test
    python cf_extract_diverse_features.py --no-tensors     # trajectory features only (fast)
"""

import os
import glob
import json
import hashlib
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"

# Mirrors the DATASETS registry in cf_generate_diverse_counterfactuals.py — the two must
# agree on output_dir and on how an image is read, since this script re-loads the ORIGINAL
# to difference it against the CF tensor. A mismatch here does not crash; it silently
# produces diff maps against a differently-preprocessed original, so keep them in sync.
DATASETS = {
    "chexpert":     dict(output_dir=f"{RESULTS_DIR}/diverse_cf",
                         image_root=None, image_suffix="", loader="pil"),
    "medmnist":     dict(output_dir=f"{RESULTS_DIR}/diverse_cf_medmnist",
                         image_root=f"{DATA_ROOT}/medmnist/images",
                         image_suffix=".png", loader="pil"),
    "pneumothorax": dict(output_dir=f"{RESULTS_DIR}/diverse_cf_pneumothorax",
                         image_root=None, image_suffix="", loader="pil"),
    "rsna":         dict(output_dir=f"{RESULTS_DIR}/diverse_cf_rsna",
                         image_root="/work3/s251710/thesis_data",
                         image_suffix="", loader="dicom"),
}

# Bound by main() via configure_dataset(); defaults keep the original chexpert behaviour.
OUTPUT_DIR   = DATASETS["chexpert"]["output_dir"]
FEAT_DIR     = f"{OUTPUT_DIR}/features"
IMAGE_ROOT   = None
IMAGE_SUFFIX = ""
IMAGE_LOADER = "pil"


def configure_dataset(name):
    """Point the module at one of DATASETS. Must run before anything reads the globals."""
    global OUTPUT_DIR, FEAT_DIR, IMAGE_ROOT, IMAGE_SUFFIX, IMAGE_LOADER
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    cfg = DATASETS[name]
    OUTPUT_DIR   = cfg["output_dir"]
    FEAT_DIR     = f"{OUTPUT_DIR}/features"
    IMAGE_ROOT   = cfg["image_root"]
    IMAGE_SUFFIX = cfg["image_suffix"]
    IMAGE_LOADER = cfg["loader"]
    print(f"Dataset: {name}  |  {OUTPUT_DIR}")
    return cfg


N_SHARDS  = 256
POLY_DEG  = 3
EDIT_EPS  = 1e-4      # |diff| above which a pixel counts as edited; validated as the
                      # level at which recovered area is stable across t_start (~0.1%)

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

def traj_path(cf_id):
    """Trajectory .npy for a cf_id — same shard as its tensor."""
    h = int(hashlib.md5(cf_id.encode()).hexdigest(), 16)
    return os.path.join(OUTPUT_DIR, "trajectories", f"{h % N_SHARDS:03d}", f"{cf_id}.npy")


def load_original(rel_path):
    """
    Original at CF preprocessing: grayscale, BILINEAR to 224, [0,1].

    Must match load_image_ds() in cf_generate_diverse_counterfactuals.py exactly — this
    array is differenced against the stored CF tensor, so any preprocessing drift shows up
    as a spurious edit spread over the whole frame.

    BILINEAR (not LANCZOS) as of 2026-07-23, matching the C0 transform; see load_image_ds
    for the measurement. CONSEQUENCE: the effusion grid under thesis_results/diverse_cf was
    generated with LANCZOS, so re-running extraction over it with this loader would
    difference a bilinear original against a LANCZOS CF. Regenerate that grid before
    re-extracting it.
    """
    root = DATA_ROOT if IMAGE_ROOT is None else IMAGE_ROOT
    full = os.path.join(root, rel_path + IMAGE_SUFFIX)

    if IMAGE_LOADER == "dicom":
        # Same per-image min-max -> uint8 stretch the RSNA C0 uses; see the generator.
        import pydicom
        a = pydicom.dcmread(full).pixel_array.astype(np.float32)
        lo, hi = a.min(), a.max()
        a = (a - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(a)
        img = Image.fromarray(a.astype(np.uint8)).convert("L")
    else:
        img = Image.open(full).convert("L")

    arr = np.array(img.resize((224, 224), Image.BILINEAR))
    return arr.astype(np.float32) / 255.0

# ══════════════════════════════════════════════════════════════════════════════
# FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def vec_summary(v, prefix):
    """Scalarise a variable-length derivative vector (t5/t10/t20 give 4/9/19 points)."""
    if len(v) == 0:
        return {f"{prefix}_{k}": np.nan
                for k in ("mean", "max", "min", "std", "absmax", "argmax_pos")}
    return {
        f"{prefix}_mean":       float(v.mean()),
        f"{prefix}_max":        float(v.max()),
        f"{prefix}_min":        float(v.min()),
        f"{prefix}_std":        float(v.std()),
        f"{prefix}_absmax":     float(np.abs(v).max()),
        # where the biggest move happened, as a fraction of the descent
        f"{prefix}_argmax_pos": float(np.argmax(np.abs(v)) / max(len(v) - 1, 1)),
    }


def traj_features(probs, pred, thr):
    """
    Trajectory scalars. `probs` in DENOISING ORDER (not reversed).

    Flip bookkeeping. For a trajectory that crosses the threshold once and stays,
    first_cross_step == steps_to_flip. They differ only on re-crossings (~9% of
    flipped CFs), which is what n_crossings records:
      first_cross_step : first step ever on the far side (NaN if never)
      steps_to_flip    : step from which the crossing HELD to the end
                         (NaN unless the CF ends flipped)
      ends_flipped     : reproduces the manifest's `flipped` column exactly
    """
    sign = -1.0 if pred else 1.0       # intended direction of travel
    d    = np.diff(probs)              # velocity, per step
    a    = np.diff(d)                  # acceleration, per step
    ds   = d * sign                    # positive = productive
    tot  = np.abs(d).sum()
    net  = probs[-1] - probs[0]

    crossed      = (probs >= thr) != bool(pred)
    first_cross  = int(np.argmax(crossed)) + 1 if crossed.any() else np.nan
    ends_flipped = bool(crossed[-1])
    if ends_flipped:
        # length of the trailing run of True -> the 1-based step it starts at
        run = len(crossed) if crossed.all() else int(np.argmin(crossed[::-1]))
        steps_to_flip = len(crossed) - run + 1
    else:
        steps_to_flip = np.nan

    x = np.linspace(0, 1, len(probs))          # normalised so t5/t10/t20 compare
    c = np.polyfit(x, probs, POLY_DEG)         # highest-degree first
    resid = float(np.sqrt(np.mean((probs - np.polyval(c, x)) ** 2)))

    out = {
        "traj_start": float(probs[0]), "traj_end": float(probs[-1]),
        "n_steps": len(probs),
        "first_cross_step": first_cross,
        "ever_crossed": int(crossed.any()),
        "steps_to_flip": steps_to_flip,
        "ends_flipped": int(ends_flipped),
        "n_crossings": int((np.diff(crossed.astype(int)) != 0).sum()),
        # monotonicity
        "mono_ratio":  float(ds[ds > 0].sum() / tot) if tot > 0 else np.nan,
        "backtrack":   float(np.abs(ds[ds < 0]).sum()),
        "n_reversals": int((np.diff(np.sign(d)) != 0).sum()),
        # n_reversals scales with trajectory length -> also give a normalised rate
        "reversal_rate": float((np.diff(np.sign(d)) != 0).sum() / max(len(d) - 1, 1)),
        # tv_ratio divides by net displacement, so gate it well above the noise floor
        "tv_ratio":    float(tot / abs(net)) if abs(net) > 0.02 else np.nan,
        "spearman":    float(spearmanr(np.arange(len(probs)), probs).correlation * sign),
        "net_delta":   float(net * sign),
        "total_variation": float(tot),
        # polynomial shape (ascending: const, linear, quad, cubic)
        "p0_level": float(c[3]), "p1_slope": float(c[2]),
        "p2_curvature": float(c[1]), "p3_sshape": float(c[0]),
        "poly_resid": resid,
    }
    out.update(vec_summary(d, "vel"))
    out.update(vec_summary(a, "acc"))
    return out


def diff_map_features(dm, eps=EDIT_EPS):
    """Magnitude + spatial entropy over the pixels the CF actually changed."""
    av = np.abs(dm)
    m  = av > eps
    n  = int(m.sum())
    if n == 0:
        return dict(l1=0.0, l2=0.0, peak_pixel=0.0, mean_pixel_abs=0.0,
                    entropy=np.nan, ent_normalized_by_size=np.nan,
                    area=0, frac_frame=0.0)
    vals, a = dm[m], av[m]
    s = a.sum()
    p = a / s
    ent = float(-(p * np.log2(p)).sum())
    return dict(
        l1=float(s),
        l2=float(np.sqrt((vals ** 2).sum())),
        peak_pixel=float(a.max()),
        mean_pixel_abs=float(a.mean()),
        entropy=ent,
        # ceiling is log2(area), so normalise by it to compare regions of unequal size
        ent_normalized_by_size=ent / np.log2(n) if n > 1 else np.nan,
        area=n,
        frac_frame=n / dm.size,
    )


def build_image_features(img_rows, regions, thr, load_tensors=True):
    """All features for ONE image's 15 CFs -> list of dicts."""
    orig = load_original(img_rows.iloc[0]["path"]) if load_tensors else None
    dmaps, recs = {}, []

    for _, row in img_rows.iterrows():
        rec = {
            "path": row["path"], "cf_id": row["cf_id"],
            "patient_id": row["patient_id"],
            "region": row["region"], "t_start": int(row["t_start"]),
            "orig_prob": row["orig_prob"], "cf_prob": row["cf_prob"],
            "pred": int(row["pred"]), "true": int(row["true"]),
            "correct": int(row["correct"]), "quadrant": row["quadrant"],
            "flipped": int(row["flipped"]), "mask_empty": int(row["mask_empty"]),
            # distance from the decision boundary — the confounder to control for
            "boundary_dist": abs(row["orig_prob"] - thr),
        }

        # mask_empty rows have no trajectory and no tensor on disk (144 rows in the
        # full manifest, concentrated in Spine and Diaphragm). Emit id columns only.
        if row["mask_empty"] == 1:
            recs.append(rec)
            continue

        probs = np.load(traj_path(row["cf_id"]))[:, 1]
        rec.update(traj_features(probs, int(row["pred"]), thr))

        if load_tensors:
            cf = (np.load(row["cf_tensor_path"]).astype(np.float32) + 1) / 2
            dm = orig - cf
            dmaps[(row["region"], rec["t_start"])] = dm
            rec.update(diff_map_features(dm))

        recs.append(rec)

    # ── alignment: where did the UNMASKED CF spend its edit? ──
    # Region masks come from the masked CFs' own supports (RePaint confines them to
    # the dilated anatomical mask), so no segmentation model is needed here.
    if load_tensors and dmaps:
        masks = {}
        for r in regions:
            ms = [np.abs(dm) > EDIT_EPS for (rr, _), dm in dmaps.items() if rr == r]
            if ms:
                masks[r] = np.logical_or.reduce(ms)

        for rec in recs:
            if rec["region"] != "general" or ("general", rec["t_start"]) not in dmaps:
                continue
            a = np.abs(dmaps[("general", rec["t_start"])])
            tot, amean = a.sum(), a.mean()
            if tot <= 0:
                continue
            for r, m in masks.items():
                tag = r.replace(" ", "_")
                rec[f"align_share_{tag}"]   = float(a[m].sum() / tot)
                rec[f"align_density_{tag}"] = float(a[m].mean() / amean)

    return recs

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def shard_of_path(path):
    return int(hashlib.md5(path.encode()).hexdigest(), 16) % N_SHARDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="chexpert",
                    help="Which diverse-CF grid to extract features from.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N images (smoke test).")
    ap.add_argument("--no-tensors", action="store_true",
                    help="Trajectory features only — skips diff maps and alignment.")
    ap.add_argument("--shards", type=int, default=64,
                    help="Number of output parquet shards (resume granularity).")
    args = ap.parse_args()

    configure_dataset(args.dataset)

    if not os.path.exists(f"{OUTPUT_DIR}/config.json"):
        raise SystemExit(
            f"No generated grid at {OUTPUT_DIR}.\n"
            f"Run cf_generate_diverse_counterfactuals.py --dataset {args.dataset} "
            f"(--calibrate-regions, then --run) first.")

    with open(f"{OUTPUT_DIR}/config.json") as f:
        config = json.load(f)
    thr     = config["threshold"]
    regions = config["regions"]

    manifest = pd.read_csv(f"{OUTPUT_DIR}/cf_manifest.csv")
    manifest["quadrant"] = np.where(
        manifest["pred"] == 1,
        np.where(manifest["true"] == 1, "tp", "fp"),
        np.where(manifest["true"] == 0, "tn", "fn"))

    paths = manifest["path"].unique()
    if args.limit:
        paths = paths[:args.limit]

    os.makedirs(FEAT_DIR, exist_ok=True)
    print(f"Images: {len(paths):,}  |  manifest rows: {len(manifest):,}")
    print(f"threshold={thr}  load_tensors={not args.no_tensors}")

    # Group by output shard so a crashed run resumes at shard granularity.
    by_shard = {}
    for p in paths:
        by_shard.setdefault(shard_of_path(p) % args.shards, []).append(p)

    grouped = dict(tuple(manifest.groupby("path")))
    n_done = n_skipped = 0

    for shard in sorted(by_shard):
        out = os.path.join(FEAT_DIR, f"feat_{shard:03d}.parquet")
        if os.path.exists(out):
            n_skipped += len(by_shard[shard])
            continue

        recs = []
        for p in tqdm(by_shard[shard], desc=f"shard {shard:03d}", leave=False):
            try:
                recs += build_image_features(grouped[p], regions, thr,
                                             load_tensors=not args.no_tensors)
                n_done += 1
            except Exception as e:
                print(f"[SKIP] {p}: {e}")

        if recs:
            tmp = out + ".tmp"
            pd.DataFrame(recs).to_parquet(tmp, index=False)
            os.replace(tmp, out)
            print(f"shard {shard:03d}: {len(recs):,} rows -> {out}")

    print(f"\nImages processed {n_done:,}  |  already-done skipped {n_skipped:,}")

    # ── concatenate ──
    shards = sorted(glob.glob(os.path.join(FEAT_DIR, "feat_*.parquet")))
    if not shards:
        print("No shards to concatenate.")
        return
    full = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    final = os.path.join(OUTPUT_DIR, "cf_features.parquet")
    full.to_parquet(final, index=False)
    print(f"\ncf_features.parquet: {full.shape[0]:,} rows x {full.shape[1]} cols -> {final}")

    # ── sanity checks, printed to the job log ──
    ok = full[full["mask_empty"] == 0]
    print("\n--- checks ---")
    print(f"rows                       : {len(full):,} (manifest {len(manifest):,})")
    print(f"mask_empty rows            : {int(full['mask_empty'].sum())}")
    print(f"traj_end == cf_prob        : {np.allclose(ok['traj_end'], ok['cf_prob'], atol=1e-5)}")
    print(f"ends_flipped == flipped    : {(ok['ends_flipped'] == ok['flipped']).all()}")
    sf = ok[ok["steps_to_flip"].notna()]
    print(f"steps_to_flip <= t_start   : {(sf['steps_to_flip'] <= sf['t_start']).all()}")
    print(f"steps_to_flip >= 1st cross : {(sf['steps_to_flip'] >= sf['first_cross_step']).all()}")
    print(f"steps set only when flipped: {sf['flipped'].eq(1).all()}")
    print(f"\nflip rate by region x t_start:")
    print(ok.pivot_table(index="region", columns="t_start",
                         values="flipped", aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
