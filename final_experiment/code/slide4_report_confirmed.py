"""
Slide 4 helper: confident C0 errors whose CheXpert Plus report CONFIRMS the label,
so the error is the model's, not label noise.
  FN: report impression/findings mention a large/moderate effusion
  FP: report explicitly negates effusion
Writes a grid PNG + CSV (with report excerpts) per side.
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DISEASE  = os.environ.get("DISEASE", "effusion")
CONF_CUT = 0.9
N_SHOW   = 12
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
OUT_DIR = os.path.join(RESULTS_DIR, "C0_final", "multilabel_ignore", DISEASE)

TERMS = {"effusion": r"effusion", "cardiomegaly": r"cardiomegaly|heart (size )?(is )?enlarged|enlarged (cardiac|heart)"}
POS_RE = {"effusion": r"\b(large|moderate|moderate[- ]to[- ]large|sizable|massive)\b[^.]{0,40}effusion",
          "cardiomegaly": r"\b(marked|moderate|severe|massive)\b[^.]{0,30}(cardiomegaly|enlarge)"}
NEG_RE = {"effusion": r"\bno\b[^.]{0,60}effusion|without[^.]{0,40}effusion|effusions?[^.]{0,20}(absent|resolved)",
          "cardiomegaly": r"(heart|cardiac)[^.]{0,30}(normal|within normal)|no[^.]{0,30}cardiomegaly"}

df = pd.read_csv(os.path.join(OUT_DIR, f"C2_dataset_c0_{DISEASE}.csv"),
                 usecols=["path", "prob", "true", "pred", "correct", "certain"])
df = df[(df["certain"] == 1) & (df["correct"] == 0)].copy()
df["ad_msp"] = np.where(df["pred"] == 1, df["prob"], 1 - df["prob"])
df = df[df["ad_msp"] >= CONF_CUT]
df["key"] = df["path"].str.replace(r"^CheXpert-v1\.0-small/", "", regex=True)

rep = pd.read_csv(os.path.join(DATA_ROOT, "CheXpertPlus", "df_chexpert_plus_240401.csv.xls"),
                  usecols=["path_to_image", "section_findings", "section_impression"])
rep["text"] = (rep["section_findings"].fillna("") + " " + rep["section_impression"].fillna("")) \
    .str.lower().str.replace(r"\s+", " ", regex=True)
df = df.merge(rep[["path_to_image", "text"]], left_on="key", right_on="path_to_image", how="left")
print(f"{len(df):,} confident errors, {df['text'].notna().sum():,} with a report")

def excerpt(text, pat):
    m = re.search(pat, text or "")
    if not m: return ""
    s = text.rfind(".", 0, m.start()) + 1
    e = text.find(".", m.end())
    return text[s:e if e > 0 else None].strip()

has_pos = df["text"].fillna("").str.contains(POS_RE[DISEASE], regex=True)
has_neg = df["text"].fillna("").str.contains(NEG_RE[DISEASE], regex=True)
any_mention = df["text"].fillna("").str.contains(TERMS[DISEASE], regex=True)

sides = {
    "FN": df[(df["pred"] == 0) & has_pos & ~has_neg],          # model says no, report says large/moderate
    "FP": df[(df["pred"] == 1) & has_neg & ~df["text"].fillna("").str.contains(POS_RE[DISEASE], regex=True)],
}
for side, d in sides.items():
    pat = POS_RE[DISEASE] if side == "FN" else NEG_RE[DISEASE]
    d = d.sort_values("ad_msp", ascending=False).copy()
    d["excerpt"] = [excerpt(t, pat) for t in d["text"]]
    print(f"\n{side}: {len(d)} report-confirmed confident errors")
    for _, r in d.head(N_SHOW).iterrows():
        print(f"  {r['key']:<45} p={r['prob']:.3f}  | {r['excerpt'][:110]}")
    cand = d.head(N_SHOW)
    if cand.empty: continue
    ncol = 4; nrow = int(np.ceil(len(cand) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.9 * nrow), squeeze=False)
    for ax in axes.ravel(): ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), cand.iterrows()):
        ax.imshow(Image.open(os.path.join(DATA_ROOT, r["path"])).convert("L"), cmap="gray")
        ax.set_title(f"{side}: pred {'POS' if r['pred'] == 1 else 'NEG'} | label {'POS' if r['true'] == 1 else 'NEG'}\n"
                     f"p = {r['prob']:.3f} | ad-MSP = {r['ad_msp']:.3f}", fontsize=9)
        ax.text(0.5, -0.03, r["key"].split("/")[1] + "/" + r["key"].split("/")[2] + "\n\"" + r["excerpt"][:60] + "\"",
                transform=ax.transAxes, ha="center", va="top", fontsize=6)
    plt.tight_layout()
    png = os.path.join(OUT_DIR, f"slide4_report_confirmed_{side}_{DISEASE}.png")
    plt.savefig(png, dpi=300); plt.close(fig)
    d.drop(columns=["text", "path_to_image"]).to_csv(png.replace(".png", ".csv"), index=False)
    print(f"  wrote {png}")
