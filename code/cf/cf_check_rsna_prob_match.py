"""
cf_check_rsna_prob_match.py
===========================
Verifies that the CF pipeline's image loading reproduces the C0 probability stored in
c2_data.csv, for any dataset in the DATASETS registry.

Why this exists: the smoke test (--smoke) only proves masked edits stay inside their
region. It never re-derives the ORIGINAL probability, so it cannot catch a preprocessing
mismatch. For RSNA that gap matters — its images are 16-bit DICOM and the C0 applies a
per-image min-max stretch to 0..255 before the uint8 round-trip. If load_image_ds drifts
from that, every CF trajectory starts from the wrong origin and every flip statistic is
measured against a probability the classifier never actually produced.

Known, accepted difference: the C0 resizes with transforms.Resize (bilinear) while the CF
pipeline uses LANCZOS. This predates RSNA (every CheXpert CF was generated that way), so a
small non-zero delta is expected. What would be alarming is a LARGE or systematically
signed delta.

Usage:
    python cf_check_rsna_prob_match.py --dataset rsna --n 25
"""

import argparse

import numpy as np
import pandas as pd

import cf_generate_diverse_counterfactuals as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(G.DATASETS), default="rsna")
    ap.add_argument("--n", type=int, default=25, help="Images to check.")
    args = ap.parse_args()

    G.configure_dataset(args.dataset)
    print(f"Device: {G.DEVICE}")

    print("Loading models...")
    unet, sd, classifier, seg_model = G.load_models()

    df = pd.read_csv(G.C2_DATA_CSV, usecols=["path", "prob", "true", "pred"])
    df = df.head(args.n)

    rows = []
    for _, r in df.iterrows():
        x = G.load_image_ds(r["path"])
        p = float(G.get_c0_prob(x, classifier))
        rows.append({"csv_prob": float(r["prob"]), "recomputed": p,
                     "delta": p - float(r["prob"])})

    out = pd.DataFrame(rows)
    d = out["delta"].values

    print(f"\nChecked {len(out)} images from {G.C2_DATA_CSV}")
    print(f"  mean |delta| : {np.abs(d).mean():.6f}")
    print(f"  max  |delta| : {np.abs(d).max():.6f}")
    print(f"  mean  delta  : {d.mean():+.6f}   (systematic sign = preprocessing drift)")
    print(f"  corr         : {np.corrcoef(out['csv_prob'], out['recomputed'])[0,1]:.6f}")

    # A flip-side check that matters more than the raw delta: does the recomputed prob land
    # on the same side of the decision threshold? That is what the CF grid actually keys on.
    thr = G.FLIP_THRESHOLD
    same_side = ((out["csv_prob"] >= thr) == (out["recomputed"] >= thr)).mean()
    print(f"  same side of threshold ({thr:.4f}): {same_side:.1%}")

    print(f"\n{out.head(10).to_string(index=False)}")

    if np.abs(d).max() < 0.05 and same_side == 1.0:
        print("\nPASS — preprocessing reproduces the stored C0 probability.")
    else:
        print("\nCHECK MANUALLY — deltas larger than the LANCZOS-vs-bilinear difference "
              "alone would explain, or a threshold-side disagreement.")


if __name__ == "__main__":
    main()
