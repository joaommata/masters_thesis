"""
Slide 4: clean confident-error examples (label confirmed two ways). DISEASE env: effusion | cardiomegaly.
FN: label POS, CheXbert(impression) POS, and the report describes a large/moderate effusion
    (excluding resolved / joint / pericardial mentions).
FP: label NEG, CheXbert(impression) NEG, and EVERY effusion/pleural-fluid sentence in
    findings+impression is a plain negation (no 'stable', 'unchanged', 'small', ...).
Model confident: ad-MSP >= CONF_CUT. Writes one grid per side + CSV with the report sentences.
"""
import os, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

CONF_CUT, N_SHOW = 0.9, 24
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")

DISEASE = os.environ.get("DISEASE", "effusion")
CB_COL  = {"effusion": "Pleural Effusion", "cardiomegaly": "Cardiomegaly"}[DISEASE]
OUT_DIR = os.path.join(RESULTS_DIR, "C0_final", "multilabel_ignore", DISEASE)
CFG = {
  "effusion": dict(
    MENTION=r"effusion|pleural fluid",
    BIG=r"\b(large|moderate|moderate[- ]sized|sizable|massive)\b[^,;]{0,30}(pleural effusion|effusion|pleural fluid)",
    BAD_FN=r"resol|no longer|knee|joint|pericardial|removed|decreas|improv|trace|small|decubitus|cannot exclude|possib|likely",
    NEG=r"\bno\b|without|negative for|absence of|free of",
    HEDGE=r"chang|stable|unchanged|again|persist|redemonstrat|decreas|increas|small|trace|minimal|tiny|residual|possible|likely|cannot|may|vs|versus|previous|prior|loculat|left|right|bilateral"),
  "cardiomegaly": dict(
    MENTION=r"cardiomegaly|heart size|heart is|cardiac silhouette|cardiac size|cardiomediastinal silhouette|enlarged heart",
    BIG=r"\b(moderate|marked|severe|massive|significant)\b[^,;]{0,15}cardiomegaly|\b(moderately|markedly|severely|massively|significantly)\b (enlarged|increased)|(heart|cardiac silhouette)[^,;]{0,25}\b(markedly|moderately|severely|massively)\b",
    BAD_FN=r"reduc|smaller|stable|unchanged|again|redemonstrat|persist|similar|no change|mild|borderline|may|likely|possib|cannot|pericardial|technique|portable|magnif|accentuat|exaggerat|decreas|improv",
    NEG=r"normal|no cardiomegaly|not enlarged|within normal limits|unremarkable",
    HEDGE=r"enlarg|cardiomegaly(?! is not)|stable|unchanged|borderline|top|upper|mild|prominent|may|likely|cannot|limited|accentuat|exaggerat|technique|portable|magnif|again|similar|chang|prior|previous"),
}[DISEASE]
CFG["HEDGE"] = CFG["HEDGE"].replace("cardiomegaly(?! is not)|", "")   # 'no cardiomegaly' is a plain negation
MENTION, BIG, BAD_FN, NEG, HEDGE = (re.compile(CFG[k]) for k in ["MENTION", "BIG", "BAD_FN", "NEG", "HEDGE"])
FIG_DIR = "final_experiment/figures/slides"

df = pd.read_csv(os.path.join(OUT_DIR, f"C2_dataset_c0_{DISEASE}.csv"),
                 usecols=["path", "prob", "true", "pred", "correct", "certain"])
df = df[(df.certain == 1) & (df.correct == 0)].copy()
df["ad_msp"] = np.where(df.pred == 1, df.prob, 1 - df.prob)
df = df[df.ad_msp >= CONF_CUT]
df["key"] = df.path.str.replace(r"^CheXpert-v1\.0-small/", "", regex=True)

rep = pd.read_csv(os.path.join(DATA_ROOT, "CheXpertPlus/df_chexpert_plus_240401.csv.xls"),
                  usecols=["path_to_image", "section_findings", "section_impression"])
cb = pd.read_json(os.path.join(DATA_ROOT, "CheXpertPlus/chexbert_labels/impression_fixed.json"), lines=True)
df = df.merge(rep, left_on="key", right_on="path_to_image", how="left") \
       .merge(cb[["path_to_image", CB_COL]].rename(columns={CB_COL: "cb"}), on="path_to_image", how="left")
df["cb"] = df.cb.fillna(0)

def sentences(r):
    txt = " ".join([str(r.section_findings) if pd.notna(r.section_findings) else "",
                    str(r.section_impression) if pd.notna(r.section_impression) else ""]).lower()
    txt = re.sub(r"\s+", " ", txt)
    return [s.strip() for s in re.split(r"(?<=[.;])\s|\d\.\s", txt) if MENTION.search(s)]

df["eff_sents"] = df.apply(sentences, axis=1)
def fn_ok(ss): return any(BIG.search(s) for s in ss) and not any(BAD_FN.search(s) for s in ss)
def fp_ok(ss): return len(ss) > 0 and all(NEG.search(s) and not HEDGE.search(s) and not BIG.search(s) for s in ss)

sides = {"FN": df[(df.pred == 0) & (df.true == 1) & (df.cb == 1) & df.eff_sents.apply(fn_ok)],
         "FP": df[(df.pred == 1) & (df.true == 0) & (df.cb == 0) & df.eff_sents.apply(fp_ok)]}

for side, d in sides.items():
    d = d.sort_values("ad_msp", ascending=False).copy()
    d["report"] = d.eff_sents.apply(lambda ss: " | ".join(ss))
    print(f"\n{side}: {len(d)} clean confident errors")
    for _, r in d.head(N_SHOW).iterrows():
        print(f"  {r.key:<46} p={r.prob:.3f}  {r.report[:120]}")
    cand = d.head(N_SHOW)
    ncol = 6; nrow = int(np.ceil(len(cand) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.8 * ncol, 3.4 * nrow), squeeze=False)
    for ax in axes.ravel(): ax.axis("off")
    for i, (ax, (_, r)) in enumerate(zip(axes.ravel(), cand.iterrows()), 1):
        ax.imshow(Image.open(os.path.join(DATA_ROOT, r.path)).convert("L"), cmap="gray")
        ax.set_title(f"#{i}  model {'POS' if r.pred == 1 else 'NEG'} ({r.ad_msp:.0%})", fontsize=9)
        pid = "/".join(r.key.split("/")[1:3])
        ax.text(0.5, -0.02, f"{pid}\n\"{r.eff_sents[0][:48]}\"", transform=ax.transAxes,
                ha="center", va="top", fontsize=5.5)
    fig.suptitle(f"{DISEASE.capitalize()} {side}s: model confident, label confirmed by report + 2 labelers", fontsize=11)
    plt.tight_layout()
    png = os.path.join(FIG_DIR, f"slide4_clean_{side}_{DISEASE}.png")
    plt.savefig(png, dpi=250); plt.close(fig)
    d.drop(columns=["section_findings", "section_impression", "path_to_image", "eff_sents"]).to_csv(png.replace(".png", ".csv"), index=False)
    print("  wrote", png)
