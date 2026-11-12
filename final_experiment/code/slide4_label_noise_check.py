"""
How much of C0's confident-error set is label noise?
Compares the CheXpert labeler label (`true`, what C0 is scored against) with an
independent CheXbert label on the same IMPRESSION section (impression_fixed.json; the original labels come from the impression -- full-report labels agree only ~60%).
CheXbert coding: 1 pos, 0 neg, -1 uncertain, null = not mentioned (-> treated as neg).
"""
import os, numpy as np, pandas as pd
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
COL = {"effusion": "Pleural Effusion", "cardiomegaly": "Cardiomegaly"}

cb = pd.read_json(os.path.join(DATA_ROOT, "CheXpertPlus/chexbert_labels/impression_fixed.json"), lines=True)
for dis, col in COL.items():
    d = pd.read_csv(os.path.join(RESULTS_DIR, "C0_final/multilabel_ignore", dis, f"C2_dataset_c0_{dis}.csv"),
                    usecols=["path", "prob", "true", "pred", "correct", "certain"])
    d = d[(d["certain"] == 1) & d["correct"].notna()].copy()
    d["key"] = d["path"].str.replace(r"^CheXpert-v1\.0-small/", "", regex=True)
    d = d.merge(cb[["path_to_image", col]].rename(columns={col: "cb"}), left_on="key", right_on="path_to_image", how="left")
    d["ad_msp"] = np.where(d["pred"] == 1, d["prob"], 1 - d["prob"])
    d["cb"] = d["cb"].fillna(0)                      # not mentioned -> negative
    d = d[d["cb"] != -1]                             # drop CheXbert-uncertain
    d["disagree"] = (d["cb"] != d["true"]).astype(int)
    d["cb_agrees_model"] = (d["cb"] == d["pred"]).astype(int)

    d["group"] = np.select(
        [(d.correct == 1) & (d.ad_msp >= .9), d.correct == 1, (d.correct == 0) & (d.ad_msp < .9), d.ad_msp >= .9],
        ["correct, ad-MSP>=0.9", "correct, ad-MSP<0.9", "error, ad-MSP<0.9", "error, ad-MSP>=0.9"])
    print(f"\n=== {dis}  (n = {len(d):,} with a definite CheXbert label)")
    t = d.groupby("group").agg(n=("disagree", "size"), labelers_disagree=("disagree", "mean"))
    print(t.to_string(float_format=lambda x: f"{x:.1%}"))
    ce = d[(d.correct == 0) & (d.ad_msp >= .9)]
    for side, s in [("FP", ce[ce.pred == 1]), ("FN", ce[ce.pred == 0])]:
        print(f"  confident {side}: n={len(s):,}  CheXbert sides with the MODEL in {s.cb_agrees_model.mean():.1%}")
