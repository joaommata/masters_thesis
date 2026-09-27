"""
cf_levels_export_images.py
==========================
Turn each generated counterfactual into a first-class image on disk, so the rest
of C2 can address it exactly the way it addresses a real one.

Why this step exists
--------------------
cf_generate_levels.py stores its output as one .npz per QUERY, holding three
float32 arrays keyed cf_level_1..3. Nothing downstream can consume that:

  * the CNN ladder opens `os.path.join(THESIS_DATA, path)` with PIL, so a
    counterfactual has to be a file whose name IS its identity;
  * `key` is the query path with the slashes flattened, so all three levels of a
    query share it. The level therefore has to move into the path, or the three
    intensities are indistinguishable to every consumer;
  * the C0 pass and the radiomics pass both want an image, not a tensor, and
    they must see the SAME pixels the CNN will see -- including the uint8
    quantisation. Feeding one of them the float array and the other the PNG is
    how `cf_prob` ends up describing different pixels from `cf_emb_*`.

Identity scheme
---------------
    cf_levels/<disease>/<level_name>/<stem>.png       relative to $THESIS_DATA

where <stem> is the key with its trailing .jpg dropped. That is the string that
goes in cf_paths, that `_img` joins onto DATA_ROOT, and that `_load_cam` flattens
into the Grad-CAM filename.

Also freezes the subset
-----------------------
The 20k queries that have counterfactuals are written to

    C2_final/<disease>/subsets/<name>.txt

as a FROZEN list. Both C2 arms -- diffusion and the KNN arm it is compared
against -- are restricted to that file, so the counterfactual source is the only
difference between them. It is a file rather than a query against the manifest
because a later top-up of the generation campaign would otherwise move the
subset under results that are already finished.

Usage
-----
    python cf_levels_export_images.py --disease atelectasis
    python cf_levels_export_images.py --disease atelectasis --limit 64 --dry-run
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_cf_sources import subset_path                              # noqa: E402
from c2_folds import c2_data_path                                  # noqa: E402

RESULTS = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
DATA_ROOT = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")

#: level index in the npz (1-based) -> the name used in the path.
LEVEL_NAMES = {1: "low", 2: "mid", 3: "high"}

INDEX_COLS = ["query_path", "key", "stem", "level", "level_name", "cf_path",
              "cell", "true", "patient_id", "direction", "orig_prob", "orig_pred",
              "target_prob", "gen_cf_prob", "reached", "flipped", "l1"]


def cf_rel_path(disease, level_name, stem):
    return f"cf_levels/{disease}/{level_name}/{stem}.png"


def _write_png(arr, dest):
    """float32 in [0,1] -> 8-bit greyscale PNG, written atomically.

    Atomic because the C0 and radiomics passes are both driven off cf_index.csv
    and may start while an interrupted export is being resumed; a torn PNG reads
    as a corrupt image hours later rather than failing here.
    """
    a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
    img = Image.fromarray(np.round(a * 255.0).astype(np.uint8), mode="L")
    tmp = f"{dest}.tmp{os.getpid()}"
    img.save(tmp, format="PNG", optimize=False)
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--levels", nargs="+", default=list(LEVEL_NAMES.values()),
                    choices=list(LEVEL_NAMES.values()),
                    help="export only these intensity levels (default: all three)")
    ap.add_argument("--subset-name", default="20k",
                    help="name of the frozen subset file to write")
    ap.add_argument("--limit", type=int, default=None, help="first N queries only")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite PNGs that already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written, write nothing")
    args = ap.parse_args()

    root = os.path.join(RESULTS, "CF_levels", args.disease)
    cf_dir = os.path.join(root, "cf")
    if not os.path.isdir(cf_dir):
        raise SystemExit(f"no cf/ directory under {root}")

    manifest = pd.read_csv(os.path.join(root, "manifest.csv"))
    manifest = manifest[manifest["level_name"].isin(args.levels)]

    keys = sorted(manifest["key"].unique())
    if args.limit:
        keys = keys[:args.limit]
        manifest = manifest[manifest["key"].isin(set(keys))]

    # patient_id comes from c2_data.csv rather than from the path string: it is
    # what the fold grouping uses, and deriving it a second way here is exactly
    # how two "patient-disjoint" splits stop agreeing.
    c2 = pd.read_csv(c2_data_path(args.disease), usecols=["path", "patient_id"])
    pid = dict(zip(c2["path"], c2["patient_id"]))
    unknown = sorted(set(manifest["path"]) - set(pid))
    if unknown:
        raise SystemExit(
            f"{len(unknown)} generated queries are not rows of c2_data.csv "
            f"(first: {unknown[0]!r}). The CF run and the C2 table disagree.")

    out_dirs = {lv: os.path.join(DATA_ROOT, "cf_levels", args.disease, lv)
                for lv in args.levels}
    print(f"disease    {args.disease}")
    print(f"levels     {', '.join(args.levels)}")
    print(f"queries    {len(keys):,}   counterfactuals {len(manifest):,}")
    print(f"images ->  {os.path.join(DATA_ROOT, 'cf_levels', args.disease)}"
          f"{'  (DRY RUN)' if args.dry_run else ''}")
    if not args.dry_run:
        for d in out_dirs.values():
            os.makedirs(d, exist_ok=True)

    by_key = {k: g for k, g in manifest.groupby("key")}
    rows, n_written, n_skipped = [], 0, 0

    for key in tqdm(keys, desc="export"):
        g = by_key[key]
        stem = key[:-4] if key.endswith(".jpg") else key
        npz = None                       # opened lazily: a fully-resumed key never loads

        for r in g.itertuples(index=False):
            lv_name = r.level_name
            rel = cf_rel_path(args.disease, lv_name, stem)
            dest = os.path.join(DATA_ROOT, rel)

            if args.dry_run:
                pass
            elif os.path.exists(dest) and not args.overwrite:
                n_skipped += 1
            else:
                if npz is None:
                    npz = np.load(os.path.join(cf_dir, key + ".npz"))
                _write_png(npz[f"cf_level_{int(r.level)}"], dest)
                n_written += 1

            rows.append({
                "query_path": r.path, "key": key, "stem": stem,
                "level": int(r.level), "level_name": lv_name, "cf_path": rel,
                "cell": r.cell, "true": r.true, "patient_id": pid[r.path],
                "direction": r.direction, "orig_prob": r.orig_prob,
                "orig_pred": r.orig_pred, "target_prob": r.target_prob,
                # Named gen_cf_prob, never cf_prob: this is the probability the
                # GENERATOR measured on the float tensor. The probability the
                # pipeline uses is recomputed from the PNG by
                # cf_levels_c0_predict.py, and conflating the two would let
                # cf_prob describe different pixels from cf_emb_*.
                "gen_cf_prob": r.cf_prob, "reached": int(r.reached),
                "flipped": int(r.flipped), "l1": r.l1,
            })

    index = pd.DataFrame(rows)[INDEX_COLS].sort_values(
        ["query_path", "level"]).reset_index(drop=True)

    print(f"\n  PNGs written {n_written:,}   already present {n_skipped:,}")
    if args.dry_run:
        print("  DRY RUN: no files written, no index, no subset frozen")
        print(index.head(6).to_string())
        return

    index_path = os.path.join(root, "cf_index.csv")
    index.to_csv(index_path, index=False)
    print(f"  cf_index.csv  {len(index):,} rows -> {index_path}")

    # ── freeze the subset ────────────────────────────────────────────────────
    if args.limit:
        print("  --limit run: subset NOT frozen (it would not be the real 20k)")
    else:
        sub = sorted(index["query_path"].unique())
        dest = subset_path(args.disease, args.subset_name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            with open(dest) as fh:
                old = [ln.strip() for ln in fh if ln.strip()]
            if old != sub:
                raise SystemExit(
                    f"{dest} already exists and holds {len(old):,} DIFFERENT "
                    f"paths. It is frozen on purpose -- results built against it "
                    f"would silently change. Delete it deliberately, or pass a "
                    f"new --subset-name.")
            print(f"  subset unchanged: {len(sub):,} queries -> {dest}")
        else:
            tmp = f"{dest}.tmp{os.getpid()}"
            with open(tmp, "w") as fh:
                fh.write("\n".join(sub) + "\n")
            os.replace(tmp, dest)
            print(f"  subset frozen: {len(sub):,} queries -> {dest}")

    # ── checks ───────────────────────────────────────────────────────────────
    print("\n--- checks ---")
    per_query = index.groupby("query_path").size().value_counts().to_dict()
    print(f"  levels per query   : {per_query}")
    print(f"  unique cf_path     : {index['cf_path'].nunique():,} "
          f"of {len(index):,} rows")
    missing = [p for p in index["cf_path"]
               if not os.path.exists(os.path.join(DATA_ROOT, p))]
    print(f"  PNGs missing       : {len(missing)}")
    empty = [p for p in index["cf_path"][:2000]
             if os.path.getsize(os.path.join(DATA_ROOT, p)) == 0]
    print(f"  zero-byte (of 2000): {len(empty)}")
    print(f"  flip rate by level :")
    for lv, g in index.groupby("level_name"):
        for d, gd in g.groupby("direction"):
            print(f"    {lv:>4s} {d:12s} {gd['flipped'].mean():.1%} "
                  f"({len(gd):,} images)")


if __name__ == "__main__":
    main()
