import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DISEASE   = os.environ.get("DISEASE", "effusion")
POLICY    = "ignore"
N_SHOW    = 8
CONF_CUT  = 0.9

DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
OUT_DIR     = os.path.join(RESULTS_DIR, "C0_final", f"multilabel_{POLICY}", DISEASE)

df = pd.read_csv(os.path.join(OUT_DIR, f"C2_dataset_c0_{DISEASE}.csv"),
                 usecols=["path", "prob", "true", "pred", "correct", "certain"])
t = float(open(os.path.join(OUT_DIR, "threshold.txt")).read())
df = df[(df["certain"] == 1) & df["correct"].notna()].copy()
mism = (df["pred"] != (df["prob"] >= t).astype(int)).sum()
print(f"sanity: pred != (prob>=t) on {mism} rows")

df["ad_msp"] = np.where(df["pred"] == 1, df["prob"], 1 - df["prob"])
err = df[df["correct"] == 0]
conf_err = err[err["ad_msp"] >= CONF_CUT]

print(f"{DISEASE}: t = {t:.3f} | {len(df):,} predictions | {len(err):,} errors")
print(f"errors with ad-MSP >= {CONF_CUT}: {len(conf_err):,} "
      f"({len(conf_err) / len(err):.1%} of errors)")
print(f"  of which false positives: {(conf_err['pred'] == 1).sum():,} | "
      f"false negatives: {(conf_err['pred'] == 0).sum():,}")

cand = err.sort_values("ad_msp", ascending=False).head(N_SHOW)
print(cand[["path", "prob", "true", "pred", "ad_msp"]].to_string(index=False))

fig, axes = plt.subplots(2, N_SHOW // 2, figsize=(3 * N_SHOW // 2, 6.6))
for ax, (_, r) in zip(axes.ravel(), cand.iterrows()):
    img = Image.open(os.path.join(DATA_ROOT, r["path"])).convert("L")
    ax.imshow(img, cmap="gray")
    ax.set_title(f"pred {'POS' if r['pred'] == 1 else 'NEG'} | label "
                 f"{'POS' if r['true'] == 1 else 'NEG'}\n"
                 f"p = {r['prob']:.3f} | ad-MSP = {r['ad_msp']:.3f}",
                 fontsize=10)
    ax.set_xlabel(r["path"].split("/")[2], fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
png = os.path.join(OUT_DIR, f"slide4_confident_errors_{DISEASE}.png")
plt.savefig(png, dpi=300)
cand.to_csv(png.replace(".png", ".csv"), index=False)
print(f"wrote {png}")
