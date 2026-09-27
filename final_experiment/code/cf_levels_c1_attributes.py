"""
cf_levels_c1_attributes.py
==========================
Extract the C1 attribute vector -- demographics, per-structure geometry,
firstorder + shape2D radiomics -- for every exported counterfactual PNG, using
exactly the models and the radiomics configuration the query images went through.

Why "exactly" is the whole point
--------------------------------
The attribute ladder's only counterfactual feature is
delta_<a> = scaled(attr(query)) - scaled(attr(CF)). That subtraction is defined
only if both sides name the same 454 attributes in the same order and were
produced by the same extractors. A single extra or renamed column does not raise
anywhere: the feature blocks downstream are built by subtracting the known meta
names, so an unexpected column silently becomes a 455th attribute and a missing
one silently becomes zero.

So the output is reindexed onto the attribute column list read from
c2_data.csv's own header, and the run fails loudly if a column is absent.

Resume model
------------
Sharded parquet, each shard written once and atomically, finished shards skipped
before the models are loaded. Copied deliberately from
c1_diverse_build_attribute_vector.py and NOT from c1_diff_build_attribute_vector.py,
whose single rolling CSV was truncated by LSF at the walltime warning and then
resumed from the cut -- that run lost progress twice and never finished.

One known asymmetry, measured rather than assumed
-------------------------------------------------
The query attributes in c2_data.csv were extracted from images resized by
torchxrayvision's XRayResizer (skimage, anti-aliased). A counterfactual is a
224x224 PNG already, so no resize happens here at all. The 224x224 the generator
worked on came from PIL bilinear. delta_* therefore carries a resampling
component that the KNN arm -- where both sides went through the identical path --
does not have. `--calibrate N` quantifies it by extracting attributes for the
stored x0 arrays (the generator's own view of the ORIGINAL image) and comparing
them, column by column, against c2_data.csv for the same paths.

Usage
-----
    python cf_levels_c1_attributes.py --disease atelectasis --levels high
    python cf_levels_c1_attributes.py --disease atelectasis --calibrate 500
    python cf_levels_c1_attributes.py --disease atelectasis --limit 32   # smoke
"""

import argparse
import glob
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import c2_data_path                                  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, os.path.join(REPO, "code", "c1"))

RESULTS = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
DATA_ROOT = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")

BATCH_SIZE = 16
N_SHARDS = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Everything in c2_data.csv that is NOT an attribute. Whatever is left over in
# its header is the attribute space this script has to reproduce.
NON_ATTR = {"path", "prob", "true", "pred", "correct", "margin",
            "cam_path", "patient_id"}


def attribute_columns(disease):
    """The attribute names, in c2_data.csv's own order."""
    head = pd.read_csv(c2_data_path(disease), nrows=0)
    return [c for c in head.columns
            if c not in NON_ATTR and not c.startswith("emb_")]


def shard_of(key, n_shards):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % n_shards


def _scalarize(v):
    """pyradiomics returns some features as 0-d or 1-element arrays.

    Parquet rejects an object column of arrays ("Can only convert 1-dimensional
    array values"); CSV used to hide it because everything is str on the way out.
    """
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return float(v.reshape(-1)[0]) if v.size == 1 else np.nan
    return v


def png_to_xrv(rel_path):
    """PNG -> (1,224,224) float32 in [-1024, 1024], the range the builder wants.

    torchxrayvision's own loaders produce [-1024,1024] from [0,1], and the
    builder's segmentation and demographic models are calibrated on that range.
    """
    a = np.asarray(Image.open(os.path.join(DATA_ROOT, rel_path)).convert("L"),
                   dtype=np.float32) / 255.0
    return torch.from_numpy(a * 2048.0 - 1024.0).unsqueeze(0)


def make_builder():
    import torchxrayvision as xrv
    from radiomics import featureextractor
    from c1_build_attribute_vector import FeatureVectorBuilder

    models = {
        "age":  xrv.baseline_models.riken.AgeModel(),
        "sex":  xrv.baseline_models.mira.SexModel(),
        "race": xrv.baseline_models.emory_hiti.RaceModel(),
    }
    seg_model = xrv.baseline_models.chestx_det.PSPNet()

    # firstorder + shape2D only, matching c1_build_attribute_vector.py. Texture
    # classes belong in c1_extend_radiomics.py; enabling them here would make
    # these columns un-joinable with the query attributes.
    extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("shape2D")

    return FeatureVectorBuilder(models=models, segmentation_model=seg_model,
                                radiomics_extractor=extractor, device=DEVICE)


def build_batch(builder, rows, load_fn, id_key, keep_cols):
    """Attribute vectors for a list of row dicts."""
    recs = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        tensors, nps, ids, kept = [], [], [], []
        for row in batch:
            try:
                t = load_fn(row)
            except Exception as e:                       # noqa: BLE001
                print(f"[SKIP] {row[id_key]}: {e}", flush=True)
                continue
            tensors.append(t)
            nps.append(t[0].numpy())
            # The builder splits this on "/" to derive patient_id, so it must
            # look like a path. Both derived fields are dropped below.
            ids.append(row[id_key])
            kept.append(row)
        if not tensors:
            continue

        vectors = builder.build_vectors_batch(torch.stack(tensors, dim=0), nps,
                                              ids, plot=False)
        for vec, row in zip(vectors, kept):
            # patient_id/path as the builder derives them describe the file on
            # disk, not the query image. Both come from cf_index.csv instead.
            vec.pop("patient_id", None)
            vec.pop("path", None)
            rec = {k: _scalarize(v) for k, v in vec.items()}
            rec.update({k: row[k] for k in keep_cols})
            recs.append(rec)
    return recs


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION: how much does the resampling path alone move an attribute?
# ══════════════════════════════════════════════════════════════════════════════

def calibrate(disease, n, attr_cols):
    """Extract attributes for stored x0 arrays and diff against c2_data.csv.

    x0 is the generator's own 224x224 view of the ORIGINAL image, so any
    difference here is pure preprocessing -- no counterfactual edit is involved.
    It is the magnitude that delta_* inherits for free on the diffusion arm and
    not on the KNN arm.
    """
    root = os.path.join(RESULTS, "CF_levels", disease)
    index = pd.read_csv(os.path.join(root, "cf_index.csv"))
    queries = index.drop_duplicates("query_path").sample(n, random_state=0)

    c2 = pd.read_csv(c2_data_path(disease),
                     usecols=["path"] + attr_cols).set_index("path")

    builder = make_builder()
    rows = [{"key": r.key, "query_path": r.query_path,
             "npz": os.path.join(root, "cf", r.key + ".npz")}
            for r in queries.itertuples(index=False)]

    def load_x0(row):
        a = np.load(row["npz"])["x0"].astype(np.float32)
        return torch.from_numpy(a * 2048.0 - 1024.0).unsqueeze(0)

    recs = build_batch(builder, rows, load_x0, "query_path", ["query_path"])
    got = pd.DataFrame(recs).set_index("query_path").reindex(columns=attr_cols)
    ref = c2.loc[got.index, attr_cols]

    diff = got - ref
    sd = ref.std().replace(0, np.nan)
    bias = (diff.mean() / sd).abs()

    print(f"\n--- preprocessing calibration ({len(got)} images, x0 vs c2_data.csv) ---")
    print(f"  columns compared          : {len(attr_cols)}")
    print(f"  |standardised bias| > 0.10: {int((bias > 0.10).sum())}")
    print(f"  |standardised bias| > 0.25: {int((bias > 0.25).sum())}")
    print(f"  median |standardised bias|: {bias.median():.4f}")
    print(f"  worst 10:\n{bias.sort_values(ascending=False).head(10).round(3).to_string()}")
    print("\n  Read this as the floor on delta_* for the diffusion arm. The KNN "
          "arm has no such floor, because both of its sides came through the "
          "same extraction path.")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--levels", nargs="+", default=None,
                    help="restrict to these level names (default: all)")
    ap.add_argument("--shards", type=int, default=N_SHARDS)
    ap.add_argument("--shard-range", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"),
                    help="only shards [LO, HI) -- lets several jobs split the work")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: first N counterfactuals, own output dir")
    ap.add_argument("--calibrate", type=int, default=0, metavar="N",
                    help="measure the preprocessing-only attribute shift on N "
                         "images and exit")
    args = ap.parse_args()

    attr_cols = attribute_columns(args.disease)
    print(f"attribute columns from c2_data.csv: {len(attr_cols)}")

    if args.calibrate:
        calibrate(args.disease, args.calibrate, attr_cols)
        return

    root = os.path.join(RESULTS, "CF_levels", args.disease)
    index = pd.read_csv(os.path.join(root, "cf_index.csv"))
    if args.levels:
        index = index[index["level_name"].isin(args.levels)].reset_index(drop=True)
    if args.limit:
        index = index.iloc[:args.limit].reset_index(drop=True)

    # A --limit run only partly fills the shards it touches. Those shards would
    # then be skipped by the resume check and their remaining rows would never
    # be extracted, so a smoke test gets its own directory.
    attr_dir = os.path.join(root, "attributes_smoke" if args.limit else "attributes")
    os.makedirs(attr_dir, exist_ok=True)

    print(f"disease    {args.disease}")
    print(f"levels     {args.levels or 'all'}")
    print(f"rows       {len(index):,}")
    print(f"device     {DEVICE}  |  batch {BATCH_SIZE}  |  {args.shards} shards")
    print(f"out        {attr_dir}", flush=True)

    by_shard = {}
    for row in index.to_dict("records"):
        by_shard.setdefault(shard_of(row["cf_path"], args.shards), []).append(row)

    shards = sorted(by_shard)
    if args.shard_range:
        lo, hi = args.shard_range
        shards = [s for s in shards if lo <= s < hi]

    # A shard is finished only when it HOLDS every row it is supposed to hold --
    # not merely when its file exists. An earlier `--levels high` run wrote all 256
    # shards with only the `high` rows in them; a file-existence check then reports
    # "0 to do" and the mid/low rows are never extracted (this happened on
    # 2026-09-25, job 29488718). Re-extracting a shard is idempotent, so the
    # conservative direction here is to redo one, never to skip one.
    todo = []
    for sh in shards:
        f = os.path.join(attr_dir, f"attr_{sh:03d}.parquet")
        if not os.path.exists(f):
            todo.append(sh)
            continue
        try:
            have = set(pd.read_parquet(f, columns=["cf_path"])["cf_path"])
        except Exception:                      # truncated or unreadable -> redo it
            todo.append(sh)
            continue
        if not {r["cf_path"] for r in by_shard[sh]} <= have:
            todo.append(sh)
    print(f"shards     {len(shards)} in range, {len(todo)} to do "
          f"({sum(len(by_shard[s]) for s in shards if s not in todo):,} rows already "
          f"extracted)", flush=True)

    if todo:
        print("Loading models...", flush=True)
        builder = make_builder()
        print("Models loaded\n", flush=True)

        keep = ["cf_path", "query_path", "level_name", "patient_id"]
        n_done = 0
        for shard in tqdm(todo, desc="shards"):
            def load_png(row):
                return png_to_xrv(row["cf_path"])

            recs = build_batch(builder, by_shard[shard], load_png, "cf_path", keep)
            if not recs:
                continue
            out = os.path.join(attr_dir, f"attr_{shard:03d}.parquet")
            df_new = pd.DataFrame(recs)
            # MERGE, never clobber. `recs` only covers the levels of THIS run, so a
            # plain overwrite would drop rows an earlier `--levels` run extracted.
            if os.path.exists(out):
                try:
                    prev = pd.read_parquet(out)
                    df_new = (pd.concat([prev, df_new], ignore_index=True)
                                .drop_duplicates("cf_path", keep="last"))
                except Exception:
                    pass                             # unreadable -> this run replaces it
            tmp = out + f".tmp{os.getpid()}"
            df_new.to_parquet(tmp, index=False)
            os.replace(tmp, out)                     # atomic: never a partial shard
            n_done += len(recs)
            print(f"shard {shard:03d}: {len(recs):,} rows -> {out}", flush=True)
        print(f"\nExtracted {n_done:,} CF attribute vectors this run", flush=True)

    if args.limit:
        print(f"\n[SMOKE] shards under {attr_dir}; nothing concatenated. "
              f"Delete that directory before the real run.")
        return

    # ── checks over whatever is complete so far ──────────────────────────────
    files = sorted(glob.glob(os.path.join(attr_dir, "attr_*.parquet")))
    if not files:
        print("no shards on disk yet")
        return
    full = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print("\n--- checks ---")
    print(f"  rows extracted   : {len(full):,} of {len(index):,}")
    print(f"  unique cf_path   : {full['cf_path'].nunique():,}")
    missing = [c for c in attr_cols if c not in full.columns]
    extra = [c for c in full.columns
             if c not in attr_cols and c not in
             ("cf_path", "query_path", "level_name", "patient_id")]
    print(f"  attribute columns: {len(attr_cols) - len(missing)} of {len(attr_cols)}")
    if missing:
        print(f"  MISSING          : {len(missing)} (first {missing[:5]})")
    if extra:
        print(f"  UNEXPECTED       : {len(extra)} (first {extra[:5]})")
    if missing or extra:
        raise SystemExit(
            "the extracted attribute space does not match c2_data.csv's. "
            "delta_* would be computed against mismatched columns -- fix the "
            "extractor configuration before assembling.")
    nan = full[attr_cols].isna().mean().sort_values(ascending=False)
    print(f"  worst NaN fractions:\n{nan.head(5).round(4).to_string()}")


if __name__ == "__main__":
    main()
