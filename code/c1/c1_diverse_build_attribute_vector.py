"""
c1_diverse_build_attribute_vector.py
====================================
Extracts the CLASSIC C1 attribute vector (demographics + per-structure geometry +
firstorder/shape2D radiomics) for every counterfactual in a diverse-CF grid produced
by cf_generate_diverse_counterfactuals.py.

This is the attribute-space counterpart of cf_extract_diverse_features.py: that script
describes the *diffusion dynamics* (trajectory shape, diff maps, region alignment), this
one describes the CF *image* with the same features the query images carry in c2_data.csv,
so delta-A = attr(query) - attr(CF) is well defined.

Output columns match c1_build_attribute_vector.py exactly (same models, same radiomics
config: firstorder + shape2D, no texture), which is what makes the delta meaningful.

Resume model — do NOT revert this to a single rolling CSV:
    c1_diff_build_attribute_vector.py rewrites one growing CSV every batch, and LSF kills
    it mid-write at the walltime warning. The next job then resumes from the truncated
    file. That run lost progress twice (78,880 -> 38,596 rows, then 63,355 -> 15,524) and
    never finished 135k images. Here each shard is written once, atomically (tmp +
    os.replace), and existing shards are skipped, so a kill costs at most one shard.

Usage:
    python c1_diverse_build_attribute_vector.py --dataset chexpert
    python c1_diverse_build_attribute_vector.py --dataset chexpert --limit 32   # smoke
"""

import os
import sys
import glob
import json
import hashlib
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torchxrayvision as xrv
from radiomics import featureextractor

sys.path.insert(0, "/zhome/d0/a/221493/thesis/code")
sys.path.insert(0, "/zhome/d0/a/221493/thesis/code/c1")
from c1_build_attribute_vector import FeatureVectorBuilder

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = "/work3/s251710/thesis_results"

# Mirrors the DATASETS registry in cf_generate_diverse_counterfactuals.py and
# cf_extract_diverse_features.py — output_dir must agree with both.
DATASETS = {
    "chexpert":     f"{RESULTS_DIR}/diverse_cf",
    "medmnist":     f"{RESULTS_DIR}/diverse_cf_medmnist",
    "pneumothorax": f"{RESULTS_DIR}/diverse_cf_pneumothorax",
    "rsna":         f"{RESULTS_DIR}/diverse_cf_rsna",
}

BATCH_SIZE = 16
N_SHARDS   = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Manifest columns carried through to the output so the attributes can be joined
# back to the query without re-reading the manifest.
KEY_COLS = ["cf_id", "path", "patient_id", "region", "t_start",
            "orig_prob", "cf_prob", "pred", "true", "correct", "flipped"]


# ══════════════════════════════════════════════════════════════════════════════
# CF TENSOR LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_cf_tensor(cf_tensor_path):
    """
    Load a saved CF tensor and convert it to what FeatureVectorBuilder expects:
    shape (1, 224, 224), range [-1024, 1024].

    The generator stores float16 in [-1, 1] (the diffusion model's own range, with a
    little overshoot from the final denoising step), so we clip before rescaling —
    the same [-1,1] -> [0,1] mapping the notebook uses to display these tensors.
    torchxrayvision's own loaders produce [-1024, 1024] from [0, 1].
    """
    arr = np.load(cf_tensor_path).astype(np.float32)
    arr = (np.clip(arr, -1.0, 1.0) + 1.0) / 2.0          # [-1,1] -> [0,1]
    arr = arr * 2048.0 - 1024.0                          # [0,1]  -> [-1024,1024]
    return torch.from_numpy(arr).unsqueeze(0)            # (1, H, W)


def shard_of(cf_id, n_shards):
    return int(hashlib.md5(cf_id.encode()).hexdigest(), 16) % n_shards


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _scalarize(v):
    """
    pyradiomics returns several features as 0-d / 1-element numpy arrays. CSV output hid
    that (everything is str on the way out), but parquet rejects an object column of
    arrays: "Can only convert 1-dimensional array values". Flatten to plain floats here.
    """
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return float(v.reshape(-1)[0]) if v.size == 1 else np.nan
    return v


def build_shard(builder, rows):
    """Extract attribute vectors for one shard's manifest rows (list of dicts)."""
    recs = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]

        tensors, nps, ids, kept = [], [], [], []
        for row in batch:
            try:
                t = load_cf_tensor(row["cf_tensor_path"])
            except Exception as e:
                print(f"[SKIP] {row['cf_id']}: {e}", flush=True)
                continue
            tensors.append(t)
            nps.append(t[0].numpy())
            # The builder splits this on "/" to derive patient_id, so it must be a real
            # path — a bare cf_id has no separators and raises IndexError. Both fields
            # are overwritten from the manifest below anyway.
            ids.append(row["cf_tensor_path"])
            kept.append(row)

        if not tensors:
            continue

        vectors = builder.build_vectors_batch(torch.stack(tensors, dim=0), nps, ids,
                                              plot=False)

        # patient_id/path as derived by the builder describe the tensor's location on
        # disk, not the query image. Replace both with the manifest's.
        for vec, row in zip(vectors, kept):
            vec.pop("patient_id", None)
            vec.pop("path", None)
            rec = {k: _scalarize(v) for k, v in vec.items()}
            rec.update({k: row[k] for k in KEY_COLS})
            recs.append(rec)

    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="chexpert",
                    help="Which diverse-CF grid to extract attributes for.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N counterfactuals (smoke test).")
    ap.add_argument("--shards", type=int, default=N_SHARDS,
                    help="Number of output parquet shards (resume granularity).")
    args = ap.parse_args()

    output_dir = DATASETS[args.dataset]
    # A --limit run only fills part of each shard it touches. Those shards would then be
    # skipped by the resume check and their remaining CFs would never be extracted, so a
    # smoke test writes to its own directory and never pollutes the real one.
    attr_dir = os.path.join(output_dir, "attributes_smoke" if args.limit else "attributes")

    manifest_path = os.path.join(output_dir, "cf_manifest.csv")
    if not os.path.exists(manifest_path):
        raise SystemExit(
            f"No grid manifest at {manifest_path}.\n"
            f"Run cf_generate_diverse_counterfactuals.py --dataset {args.dataset} first.")

    manifest = pd.read_csv(manifest_path)
    n_all = len(manifest)

    # mask_empty rows have no tensor on disk — the generator skips writing them.
    manifest = manifest[manifest["mask_empty"] == 0].reset_index(drop=True)
    if args.limit:
        manifest = manifest.iloc[:args.limit].reset_index(drop=True)

    print(f"Dataset: {args.dataset}  ->  {output_dir}")
    print(f"Manifest rows: {n_all:,}  |  with a tensor: {len(manifest):,}")
    print(f"Device: {DEVICE}  |  batch {BATCH_SIZE}  |  {args.shards} shards", flush=True)

    os.makedirs(attr_dir, exist_ok=True)

    # Group by output shard first, so we can skip finished shards WITHOUT loading
    # the models (a resubmitted job that has little left to do starts almost instantly).
    by_shard = {}
    for row in manifest.to_dict("records"):
        by_shard.setdefault(shard_of(row["cf_id"], args.shards), []).append(row)

    todo = [s for s in sorted(by_shard)
            if not os.path.exists(os.path.join(attr_dir, f"attr_{s:03d}.parquet"))]
    n_skipped = sum(len(by_shard[s]) for s in by_shard if s not in todo)
    print(f"Shards: {len(by_shard)} total, {len(todo)} to do "
          f"({n_skipped:,} CFs already extracted)", flush=True)

    if todo:
        print("Loading models...", flush=True)
        models = {
            'age':  xrv.baseline_models.riken.AgeModel(),
            'sex':  xrv.baseline_models.mira.SexModel(),
            'race': xrv.baseline_models.emory_hiti.RaceModel(),
        }
        seg_model = xrv.baseline_models.chestx_det.PSPNet()

        # firstorder + shape2D only, matching c1_build_attribute_vector.py. Texture
        # classes belong in c1_extend_radiomics.py, not here — enabling them would
        # make these columns un-joinable with the query attributes.
        extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('shape2D')

        builder = FeatureVectorBuilder(models=models, segmentation_model=seg_model,
                                       radiomics_extractor=extractor, device=DEVICE)
        print("Models loaded\n", flush=True)

        n_done = 0
        for shard in tqdm(todo, desc="shards"):
            out = os.path.join(attr_dir, f"attr_{shard:03d}.parquet")
            recs = build_shard(builder, by_shard[shard])
            if not recs:
                continue
            tmp = out + ".tmp"
            pd.DataFrame(recs).to_parquet(tmp, index=False)
            os.replace(tmp, out)                       # atomic: never a partial shard
            n_done += len(recs)
            print(f"shard {shard:03d}: {len(recs):,} rows -> {out}", flush=True)

        print(f"\nExtracted {n_done:,} CF attribute vectors this run", flush=True)

    if args.limit:
        print(f"\n[SMOKE] --limit run: shards written to {attr_dir}, "
              f"cf_attributes.parquet NOT written. Delete that dir before the real run.")
        return

    # ── concatenate ───────────────────────────────────────────────────────────
    shards = sorted(glob.glob(os.path.join(attr_dir, "attr_*.parquet")))
    if not shards:
        print("No shards to concatenate.")
        return
    full = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    final = os.path.join(output_dir, "cf_attributes.parquet")
    full.to_parquet(final, index=False)
    print(f"\ncf_attributes.parquet: {full.shape[0]:,} rows x {full.shape[1]} cols -> {final}")

    # ── checks, printed to the job log ────────────────────────────────────────
    print("\n--- checks ---")
    print(f"rows                 : {len(full):,} (expected {len(manifest):,})")
    print(f"unique cf_id         : {full['cf_id'].nunique():,}")
    print(f"unique query images  : {full['path'].nunique():,}")
    print(f"cells per image      : {full.groupby('path').size().value_counts().to_dict()}")
    n_attr = len([c for c in full.columns if c not in KEY_COLS])
    print(f"attribute columns    : {n_attr}")
    nan_frac = full[[c for c in full.columns if c not in KEY_COLS]].isna().mean()
    worst = nan_frac.sort_values(ascending=False).head(5)
    print(f"worst NaN fractions  :\n{worst.round(4).to_string()}")


if __name__ == "__main__":
    main()
