"""
cf_merge_shards.py
==================
Merge per-shard manifests into one manifest.csv.

A sharded run writes manifest_shard<k>of<N>.csv per shard, so concurrent shards
never touch the same file. The npz files are already keyed by image path and the
shards take disjoint images, so only the manifests need combining.

Safe to run while shards are still going: it reads whatever has been flushed and
writes a snapshot. Re-run when they finish.

    python cf_merge_shards.py --disease effusion
    python cf_merge_shards.py --disease effusion --keep-shards
"""

import argparse
import glob
import os

import pandas as pd

RESULTS = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--dir", default=None,
                    help="defaults to $THESIS_RESULTS/CF_levels/<disease>")
    ap.add_argument("--keep-shards", action="store_true",
                    help="do not delete the per-shard files after merging")
    args = ap.parse_args()

    d = args.dir or os.path.join(RESULTS, "CF_levels", args.disease)
    shards = sorted(glob.glob(os.path.join(d, "manifest_shard*of*.csv")))
    if not shards:
        print(f"No shard manifests in {d}")
        return

    parts = []
    for f in shards:
        p = pd.read_csv(f)
        print(f"  {os.path.basename(f):32s} {len(p):6,} rows  "
              f"{p['path'].nunique():5,} images")
        parts.append(p)

    man = pd.concat(parts, ignore_index=True)
    before = len(man)
    # Shards take disjoint images, so duplicates here mean a shard was re-run
    # with different bounds. Keep the last write for each (path, level).
    man = man.drop_duplicates(subset=["path", "level"], keep="last")
    if len(man) != before:
        print(f"  [note] dropped {before - len(man):,} duplicate rows")

    man = man.sort_values(["path", "level"]).reset_index(drop=True)
    out = os.path.join(d, "manifest.csv")
    man.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(man):,} rows, {man['path'].nunique():,} images)")

    if not args.keep_shards:
        for f in shards:
            os.remove(f)
        print(f"Removed {len(shards)} shard files")

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
