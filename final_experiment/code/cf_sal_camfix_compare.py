"""
cf_sal_camfix_compare.py
========================
Old vs new saliency CNNs: what did fixing the Grad-CAMs buy?

The *_sal models were retrained on the regenerated CAMs (cam_pos/cam_neg instead
of a single ReLU'd map). submit_sal_retrain.sh moved each old result dir to
images_precamfix/ and the retrained one took its place in images/. This scores
both from their own fold predictions and plots the difference.

    python cf_sal_camfix_compare.py
    python cf_sal_camfix_compare.py --k 10 --out /tmp/camfix.png
    python cf_sal_camfix_compare.py --label-mode delta
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RESULTS = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
MODELS = ["xi_sal", "cf_sal", "dual_sal", "dual_sal_scalars"]


def fold_aucs(model_dir, model, n_folds=5):
    """Mean AUROC over folds, or None if this model has no results here."""
    aucs = []
    for i in range(n_folds):
        p = os.path.join(model_dir, f"fold_{i}_predictions.csv")
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p)
        aucs.append(roc_auc_score(d["correct"], d[f"{model}_prob"]))
    return float(np.mean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diseases", nargs="+",
                    default=["effusion", "atelectasis", "cardiomegaly",
                             "consolidation", "edema"])
    ap.add_argument("--k", type=int, default=1, help="cf_<k> directory")
    ap.add_argument("--source", default="knn_correct_cf")
    ap.add_argument("--out", default="/zhome/d0/a/221493/thesis/final_experiment/"
                                     "figures/sal_camfix.png")
    ap.add_argument("--label-mode", choices=["both", "delta", "none"],
                    default="delta",
                    help="both: AUROC on every bar; delta: change above the new "
                         "bar only; none: no bar labels")
    args = ap.parse_args()

    rows = []
    for d in args.diseases:
        root = os.path.join(RESULTS, "C2_final_results", d, args.source, f"cf_{args.k}")
        for m in MODELS:
            old = fold_aucs(os.path.join(root, "images_precamfix", m), m)
            new = fold_aucs(os.path.join(root, "images", m), m)
            rows.append({"disease": d, "model": m, "old": old, "new": new,
                         "delta": None if (old is None or new is None) else new - old})
    df = pd.DataFrame(rows)

    print(df.to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.4f}"))
    ok = df.dropna(subset=["delta"])
    if len(ok):
        print(f"\nmean change {ok.delta.mean():+.4f}   "
              f"better in {(ok.delta > 0).sum()}/{len(ok)} cases")

    # One grouped bar per disease: old vs new, per model.
    dd = df.dropna(subset=["old", "new"])
    if not len(dd):
        print("\nnothing to plot yet")
        return
    diseases = list(dict.fromkeys(dd.disease))
    fig, axes = plt.subplots(1, len(diseases), figsize=(3.5 * len(diseases), 3.6),
                             sharey=True, squeeze=False)
    for ax, d in zip(axes[0], diseases):
        g = dd[dd.disease == d]
        x = np.arange(len(g))
        b_old = ax.bar(x - 0.2, g.old, 0.4, label="old CAMs", color="#BFBFBF")
        b_new = ax.bar(x + 0.2, g.new, 0.4, label="new CAMs", color="#4C72B0")

        if args.label_mode == "both":
            ax.bar_label(b_old, fmt="%.3f", padding=2, fontsize=6, rotation=90)
            ax.bar_label(b_new, fmt="%.3f", padding=2, fontsize=6, rotation=90)
        elif args.label_mode == "delta":
            ax.bar_label(b_new, labels=[f"{v:+.3f}" for v in g.delta],
                         padding=2, fontsize=6, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(g.model, rotation=45, ha="right", fontsize=8)
        ax.set_title(d, fontsize=10)
        ax.grid(axis="y", alpha=.25)
    axes[0][0].set_ylabel("CV AUROC")
    axes[0][0].legend(fontsize=8)

    # Headroom above the tallest bar so the labels are not clipped.
    lo = min(dd.old.min(), dd.new.min())
    hi = max(dd.old.max(), dd.new.max())
    pad = 0.10 if args.label_mode != "none" else 0.02
    axes[0][0].set_ylim(max(0.5, lo - 0.02), min(1.0, hi + pad))

    fig.suptitle(f"Saliency CNNs before and after the Grad-CAM fix (K={args.k})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()