# c0_rsna_split.py
"""
Build the C0 train/val splits for the RSNA Pneumonia binary task.

Binary task: pneumonia vs everything else.
    label 1 = 'Lung Opacity'
    label 0 = 'Normal'  OR  'No Lung Opacity / Not Normal'

The middle class is folded into the NEGATIVES rather than dropped: it is the
abnormal-but-not-pneumonia population, i.e. exactly the hard negatives. With it
excluded the task reduces to the two extremes of a severity spectrum (clear
pneumonia vs clear healthy), which a pretrained DenseNet solves too easily
(~0.96 val AUC even after the view fix below). Folding it back in keeps the task
binary while restoring the full 29,684 studies and a realistic error set.

Labels come from the 'Calculated' label group of the MD.ai annotation export --
that group holds the final adjudicated label, one per study, and its counts match
the published Kaggle challenge numbers.

VIEW BALANCING (important): ViewPosition is confounded with the label -- AP is
portable/supine, i.e. patients too sick to stand for a PA, so AP skews pneumonia.
Unbalanced, ViewPosition ALONE scored AUC 0.767 on the Normal-only-negatives pool
and C0 hit a misleading 0.987 val AUC. We subsample to 50/50 pos/neg WITHIN each
view, which drives ViewPosition's AUC to ~0.500 and forces the model onto actual
pathology. Balancing is re-derived here for the wider negative pool.

Writes c0_train_split.csv / c0_val_split.csv to $THESIS_DATA/rsna_pneumonia/,
matching the layout of the CheXpert disease folders.
"""
import json
import os

import pandas as pd
import pydicom
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
IMG_SUBDIR = "mdai_rsna_project_x9N20BZa_images_2018-07-20-153330"
ANN_JSON = os.path.join(
    DATA_ROOT, "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
)
OUT_DIR = os.path.join(DATA_ROOT, "rsna_pneumonia")

POS_LABEL = "Lung Opacity"
# Both non-pneumonia classes are negatives. 'No Lung Opacity / Not Normal' is the
# hard-negative population (abnormal chest, but not pneumonia).
NEG_LABELS = ["Normal", "No Lung Opacity / Not Normal"]
ALL_LABELS = [POS_LABEL] + NEG_LABELS

# 50/50 like the CheXpert C0 splits. C2 only ever consumes the val half, so this
# keeps its dataset as large as possible.
VAL_FRAC = 0.5
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ── Read adjudicated labels ───────────────────────────────────────────────────
with open(ANN_JSON) as f:
    proj = json.load(f)

label_info = {
    lab["id"]: (grp["name"], lab["name"])
    for grp in proj["labelGroups"]
    for lab in grp["labels"]
}

rows = {}
for a in proj["datasets"][0]["annotations"]:
    group, name = label_info[a["labelId"]]
    if group != "Calculated" or name not in ALL_LABELS:
        continue
    # One Calculated label per study; a positive study also carries box
    # annotations under the same label, so dedupe on StudyInstanceUID.
    rows[a["StudyInstanceUID"]] = {
        "StudyInstanceUID": a["StudyInstanceUID"],
        # Path is relative to DATA_ROOT, mirroring the CheXpert "Path" column
        "Path": os.path.join(
            IMG_SUBDIR,
            a["StudyInstanceUID"],
            a["SeriesInstanceUID"],
            a["SOPInstanceUID"] + ".dcm",
        ),
        "label_name": name,
    }

df = pd.DataFrame(rows.values())
print("all adjudicated studies:", len(df))
print(df["label_name"].value_counts().to_string(), "\n")

# ── Binary task: pneumonia vs everything else ─────────────────────────────────
df = df[df["label_name"].isin(ALL_LABELS)].reset_index(drop=True)
df["Pneumonia"] = (df["label_name"] == POS_LABEL).astype(float)

print(f"binary set (pneumonia vs rest): {len(df):,} studies")
print(f"  positives (Pneumonia=1): {int(df.Pneumonia.sum()):,}")
print(f"  negatives (Pneumonia=0): {int((1 - df.Pneumonia).sum()):,}")
print("    negative breakdown:")
print(df[df.Pneumonia == 0]["label_name"].value_counts().to_string().replace("\n", "\n    "))
print(f"  prevalence: {df.Pneumonia.mean():.1%}\n")

# Sanity: every image must exist on disk before we commit a split
missing = (~df["Path"].map(lambda p: os.path.exists(os.path.join(DATA_ROOT, p)))).sum()
if missing:
    raise FileNotFoundError(f"{missing} DICOM paths do not exist under {DATA_ROOT}")
print("all DICOM paths verified on disk\n")

# ── Read ViewPosition from the DICOM headers ──────────────────────────────────
def read_view(path):
    dcm = pydicom.dcmread(os.path.join(DATA_ROOT, path), stop_before_pixels=True)
    return str(getattr(dcm, "ViewPosition", "UNKNOWN"))

print("reading ViewPosition from DICOM headers...")
with ThreadPoolExecutor(max_workers=16) as ex:
    df["view"] = list(ex.map(read_view, df["Path"]))

print("\nBEFORE balancing -- view is confounded with the label:")
print(df.groupby("view")["Pneumonia"].agg(["mean", "count"]).round(3).to_string())

# ── View balancing ────────────────────────────────────────────────────────────
# Within each view, keep an equal number of positives and negatives. Prevalence
# is then identical (50%) across views, so ViewPosition carries zero label
# information and cannot be used as a shortcut.
#
# The negatives are drawn with a FIXED Normal : NotNormal ratio that is the same
# in every view. Drawing them at random instead would leave the negative subtype
# mix view-dependent (AP negatives are 78% NotNormal, PA negatives only 45%),
# which re-introduces a weaker version of the same shortcut: the model could
# learn "AP + healthy-looking => pneumonia".
NEG_MIX = {"Normal": 0.5, "No Lung Opacity / Not Normal": 0.5}

balanced = []
for view, grp in df.groupby("view"):
    pos = grp[grp.Pneumonia == 1.0]
    neg = grp[grp.Pneumonia == 0.0]

    # How many negatives of each subtype can we afford at the target mix?
    n = min(
        len(pos),
        *(int(len(neg[neg.label_name == k]) / frac) for k, frac in NEG_MIX.items()),
    )
    if n == 0:
        print(f"  [SKIP] view {view}: cannot fill target mix")
        continue

    balanced.append(pos.sample(n, random_state=SEED))
    for k, frac in NEG_MIX.items():
        balanced.append(
            neg[neg.label_name == k].sample(int(round(n * frac)), random_state=SEED)
        )
    print(f"  {view}: kept {n} pos + ~{n} neg = ~{2 * n} (from {len(grp)})")

df = pd.concat(balanced).reset_index(drop=True)

print(f"\nAFTER balancing: {len(df):,} studies")
print(df.groupby("view")["Pneumonia"].agg(["mean", "count"]).round(3).to_string())
print("\nnegative subtype mix per view (must match across views):")
print(
    pd.crosstab(df[df.Pneumonia == 0]["view"], df[df.Pneumonia == 0]["label_name"],
                normalize="index").round(3).to_string()
)
print(f"\noverall prevalence: {df.Pneumonia.mean():.1%}")

# ── Split ─────────────────────────────────────────────────────────────────────
# RSNA gives one image per study and no patient IDs, so unlike the CheXpert
# splits there is no patient-level leakage to guard against -- a stratified
# study-level split is sufficient. Stratify on view x label_name (not just the
# binary label) so BOTH the view balance and the negative subtype mix survive
# inside each half.
train_df, val_df = train_test_split(
    df,
    test_size=VAL_FRAC,
    random_state=SEED,
    stratify=df["view"] + "_" + df["label_name"],
)

for name, part in [("train", train_df), ("val", val_df)]:
    print(f"\n{name}: {len(part):,} studies | prevalence {part.Pneumonia.mean():.1%}")
    print(part.groupby("view")["Pneumonia"].agg(["mean", "count"]).round(3).to_string())

train_df.to_csv(os.path.join(OUT_DIR, "c0_train_split.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "c0_val_split.csv"), index=False)
print(f"\nwrote splits to {OUT_DIR}")
