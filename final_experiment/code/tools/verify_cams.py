"""Check the Grad-CAM npz files on disk against the CSV they belong to.

Verifies the contract c2_train_images._load_cam() relies on -- (7,7) float32 in
[0,1] -- and the orientation the maps were written with: `sign` must agree with
`prob > threshold` for every row, and `orient_t` must match threshold.txt.

The headline number is the exactly-empty rate per confusion cell. Before the
orientation fix it was 77.5% for TN and 50.0% for FN on cardiomegaly (the
positive-logit map is negative almost everywhere on a predicted-negative image,
so the unconditional ReLU clipped it to zero); after, it should be ~0 in every
cell. The second number to read is mean std per cell: "not blank" and "carries
structure" are different bars, and a recovered map that is merely non-zero would
show a much lower std than the positives.

    python verify_cams.py --disease cardiomegaly
    python verify_cams.py --disease cardiomegaly --limit 256
"""

import argparse
import os

import numpy as np
import pandas as pd

RESULTS = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")

ap = argparse.ArgumentParser()
ap.add_argument("--disease", required=True)
ap.add_argument("--policy", default="ignore")
ap.add_argument("--split", default="C2_dataset")
ap.add_argument("--limit", type=int, default=None,
                help="first N rows (match the --limit used when writing)")
ap.add_argument("--flat-std", type=float, default=0.05,
                help="std below which a map counts as visually flat")
args = ap.parse_args()

d = os.path.join(RESULTS, "C0_final", f"multilabel_{args.policy}", args.disease)
cam_dir = os.path.join(d, "gradcam", args.split)
t = float(open(os.path.join(d, "threshold.txt")).read().split()[0])

meta = pd.read_csv(os.path.join(d, f"{args.split}_c0_{args.disease}.csv"),
                   usecols=["path", "prob", "true"])
if args.limit:
    meta = meta.head(args.limit)
meta = meta.dropna(subset=["true"])
meta["cell"] = np.select(
    [(meta.prob > t) & (meta.true == 1), (meta.prob <= t) & (meta.true == 0),
     (meta.prob > t) & (meta.true == 0), (meta.prob <= t) & (meta.true == 1)],
    ["TP", "TN", "FP", "FN"], default="?")

rows, missing, bad_shape = [], 0, 0
for _, r in meta.iterrows():
    f = os.path.join(cam_dir, str(r["path"]).replace("/", "_") + ".npz")
    if not os.path.exists(f):
        missing += 1
        continue
    z = np.load(f)
    two = "cam_pos" in z.files
    c = z["cam"]
    if c.shape != (7, 7) or c.dtype != np.float32:
        bad_shape += 1
    has_meta = "sign" in z.files
    # With two channels the question is no longer "is the map blank" -- neither
    # direction can be clipped away -- but whether BOTH directions carry signal.
    pos = z["cam_pos"] if two else None
    neg = z["cam_neg"] if two else None
    rows.append(dict(
        cell=r["cell"], prob=float(r["prob"]), two=two,
        nnz=int(np.count_nonzero(c)), std=float(c.std()),
        mx=float(c.max()), mn=float(c.min()),
        pos_empty=int(pos.max() <= 0) if two else np.nan,
        neg_empty=int(neg.max() <= 0) if two else np.nan,
        pos_std=float(pos.std()) if two else np.nan,
        neg_std=float(neg.std()) if two else np.nan,
        # how lopsided the evidence is, before per-channel normalisation
        bal=(float(z["scale_pos"]) / (float(z["scale_pos"]) + float(z["scale_neg"]) + 1e-8))
            if two and "scale_pos" in z.files else np.nan,
        sign=float(z["sign"]) if has_meta else np.nan,
        orient_t=float(z["orient_t"]) if "orient_t" in z.files else np.nan,
        sign_ok=(bool(float(z["sign"]) > 0) == bool(r["prob"] > t)) if has_meta else False,
    ))

df = pd.DataFrame(rows)
print(f"disease {args.disease} | split {args.split} | threshold {t:.6f}")
print(f"rows {len(meta):,} | maps found {len(df):,} | missing {missing:,} | "
      f"bad shape/dtype {bad_shape}")

# NB: not "empty" -- DataFrame.empty is a pandas property (is the frame empty),
# so df.empty would silently return a bool instead of the column.
df["is_empty"] = df.nnz == 0
df["is_flat"] = df["std"] < args.flat_std
two_ch = bool(df["two"].all())

if two_ch:
    print("\ntwo-channel maps (cam_pos = evidence for disease, cam_neg = for health)")
    print(df.groupby("cell").agg(
        n=("cell", "size"),
        pos_empty=("pos_empty", "mean"),
        neg_empty=("neg_empty", "mean"),
        pos_std=("pos_std", "mean"),
        neg_std=("neg_std", "mean"),
        balance=("bal", "mean"),
        mean_prob=("prob", "mean"),
    ).reindex(["TP", "TN", "FP", "FN"]).dropna(how="all").round(4).to_string())
    print("  balance = scale_pos/(scale_pos+scale_neg): >0.5 the raw evidence "
          "leans positive, <0.5 negative")
else:
    print("\nsingle-channel maps (pre-two-channel format)")

print("\nnet `cam` map (kept for older readers):")
print(df.groupby("cell").agg(
    n=("cell", "size"),
    empty_rate=("is_empty", "mean"),
    flat_rate=("is_flat", "mean"),
    mean_std=("std", "mean"),
    mean_prob=("prob", "mean"),
).reindex(["TP", "TN", "FP", "FN"]).dropna(how="all").round(4).to_string())

print(f"\noverall empty {df.is_empty.mean():.2%} | flat {df.is_flat.mean():.2%}")

print("\n── contract ──")
print(f"  min >= 0 everywhere      {bool(df.mn.min() >= 0)}  (min {df.mn.min():.6f})")
print(f"  max == 1 everywhere      {bool(np.allclose(df.mx, 1.0))}")
has_meta = df.sign.notna().any()
if has_meta:
    print(f"  orient_t == threshold    {bool(np.allclose(df.orient_t.dropna(), t))}")
    print(f"  sign agrees with pred    {bool(df.sign_ok.all())}"
          f"  ({int((~df.sign_ok).sum())} mismatches)")
    print(f"  negative-oriented        {int((df.sign < 0).sum()):,} "
          f"({(df.sign < 0).mean():.1%}) -- these were the blank ones")
else:
    print("  no sign/orient_t keys -- these maps predate the orientation fix")

print("\n── verdict ──")
if two_ch:
    # What has to hold is that no IMAGE loses its explanation: at least one
    # direction carries signal for every row. A single empty channel is not a
    # failure -- on a confidently-classified image the evidence genuinely is
    # one-directional, and an empty cam_pos on a clean chest is the honest
    # answer ("nothing here argues for disease"), not a missing map.
    both_empty = ((df["pos_empty"] == 1) & (df["neg_empty"] == 1)).mean()
    print(f"  images with NO explanation at all (both channels empty): {both_empty:.3%}")
    if both_empty < 0.001:
        print("  PASS: every image has evidence in at least one direction.")
    else:
        print("  FAIL: some images have no map in either direction.")

    # The single-map versions each annihilated one side. Report what they would
    # have produced on this same data, so the comparison is explicit.
    pe = df.groupby("cell")["pos_empty"].mean()
    print(f"\n  cam_pos empty by quadrant (v1 `relu(signed)` would have lost exactly these):")
    print("   " + pe.reindex(["TP", "TN", "FP", "FN"]).round(3).to_string().replace("\n", "\n   "))
    print(f"\n  cam_neg empty by quadrant: all {df['neg_empty'].mean():.1%} -- "
          f"the negative direction always carries signal here.")

    # Where the evidence is lopsided AGAINST the reported decision: the model
    # says positive, but almost all the gradient magnitude pushes the other way.
    pos_pred = df[df["cell"].isin(["TP", "FP"])]
    if len(pos_pred) and pos_pred["bal"].notna().any():
        contra = (pos_pred["bal"] < 0.5).mean()
        print(f"\n  positive predictions whose raw evidence leans NEGATIVE: {contra:.1%}")
        print(f"  (mean balance on predicted-positive rows: {pos_pred['bal'].mean():.3f}; "
              f"0.5 = evenly matched)")
        if contra > 0.5:
            print("  -> these are calls the tuned threshold makes against the logit's "
                  "own direction, not a saliency defect. Worth knowing before "
                  "reading any explanation for this disease.")
else:
    neg = df[df.cell.isin(["TN", "FN"])]
    pos = df[df.cell.isin(["TP", "FP"])]
    if not len(neg) or not len(pos):
        print("  not enough of one class in this sample to judge")
    else:
        verdict = ("PASS" if neg.is_empty.mean() < 0.02
                   and neg["std"].mean() > 0.5 * pos["std"].mean() else "CHECK")
        print(f"  {verdict}: negatives {neg.is_empty.mean():.2%} empty, "
              f"std {neg['std'].mean():.3f} vs positives {pos['std'].mean():.3f}")
