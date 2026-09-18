"""Step 0: falsify the signed-Grad-CAM premise before spending GPU time.

The fix assumes that for images C0 calls NEGATIVE the signed Grad-CAM
    cam_signed = (grads.mean(2,3) * relu(acts)).sum(1)
is negative nearly everywhere -- which is why the unconditional relu() in
c0_final_predictions.py:240 annihilates it. If instead cam_signed is only
*slightly* negative on average (frac_pos ~ 0.5), flipping the sign yields a
non-empty but meaningless map, and the whole plan is void.

Reports, per confusion-matrix cell, the fraction of the 7x7 signed map that is
positive, plus what the map looks like before/after the flip.

Read-only. Writes nothing.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DISEASE = sys.argv[1] if len(sys.argv) > 1 else "cardiomegaly"
N_PER_CELL = int(sys.argv[2]) if len(sys.argv) > 2 else 50

DATA_DIR = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RESULTS = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
MODEL_DIR = os.path.join(RESULTS, "C0_final", "multilabel_ignore")
OUT_DIR = os.path.join(MODEL_DIR, DISEASE)

disease_col = {
    "effusion": "Pleural Effusion", "cardiomegaly": "Cardiomegaly",
    "atelectasis": "Atelectasis", "consolidation": "Consolidation",
    "edema": "Edema", "pneumonia": "Pneumonia",
}
TARGET_COL = disease_col[DISEASE]

with open(os.path.join(MODEL_DIR, "labels.json")) as f:
    LABELS = json.load(f)["labels"]
TARGET_IDX = LABELS.index(TARGET_COL)
THRESH = float(open(os.path.join(OUT_DIR, "threshold.txt")).read().split()[0])

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"disease={DISEASE} target_idx={TARGET_IDX} threshold={THRESH:.6f} dev={DEV}")

# ── sample balanced across TP/TN/FP/FN ───────────────────────────────────────
csv = os.path.join(OUT_DIR, f"C2_dataset_c0_{DISEASE}.csv")
meta = pd.read_csv(csv, usecols=["path", "prob", "true"])
meta = meta.dropna(subset=["true"])
meta["pred"] = (meta["prob"] > THRESH).astype(int)
meta["cell"] = np.select(
    [(meta.pred == 1) & (meta.true == 1), (meta.pred == 0) & (meta.true == 0),
     (meta.pred == 1) & (meta.true == 0), (meta.pred == 0) & (meta.true == 1)],
    ["TP", "TN", "FP", "FN"], default="?")
rng = np.random.RandomState(0)
samp = pd.concat([g.sample(min(N_PER_CELL, len(g)), random_state=rng)
                  for _, g in meta.groupby("cell")])
print(f"sampled {len(samp)} rows: {samp.cell.value_counts().to_dict()}")

# ── model + hook (identical recipe to c0_final_predictions.py) ───────────────
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
sd = torch.load(os.path.join(MODEL_DIR, "c0_best.pt"), map_location="cpu")
sd = sd.get("model_state_dict", sd.get("state_dict", sd))
model.load_state_dict(sd)
model = model.to(DEV).eval()

cache = {}
def hook_fn(module, inp, out):
    acts = out
    cache["acts"] = torch.relu(acts).detach()
    if acts.requires_grad:
        acts.register_hook(lambda g: cache.__setitem__("grads", g.detach()))
model.features.register_forward_hook(hook_fn)

tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

rows = []
for _, r in samp.iterrows():
    img = Image.open(os.path.join(DATA_DIR, r["path"])).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEV)
    cache.clear(); model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[:, TARGET_IDX].backward(torch.ones(1, device=DEV))
    w = cache["grads"].mean(dim=(2, 3), keepdim=True)
    signed = (w * cache["acts"]).sum(dim=1)[0].cpu().numpy()   # (7,7) NO relu

    old = np.maximum(signed, 0)                                 # current behaviour
    sign = 1.0 if r["prob"] > THRESH else -1.0
    new = np.maximum(signed * sign, 0)                          # proposed fix
    rows.append(dict(
        cell=r["cell"], prob=r["prob"],
        frac_pos=float((signed > 0).mean()),
        absmean=float(np.abs(signed).mean()),
        old_empty=int(old.max() <= 0),
        new_empty=int(new.max() <= 0),
        new_std_norm=float((new / (new.max() + 1e-8)).std()),
    ))

df = pd.DataFrame(rows)
print("\n=== signed-map sign distribution, by confusion cell ===")
print(df.groupby("cell").agg(
    n=("cell", "size"),
    frac_pos_mean=("frac_pos", "mean"),
    frac_pos_min=("frac_pos", "min"),
    frac_pos_max=("frac_pos", "max"),
    absmean=("absmean", "mean"),
    old_empty_rate=("old_empty", "mean"),
    new_empty_rate=("new_empty", "mean"),
    new_std=("new_std_norm", "mean"),
).reindex(["TP", "TN", "FP", "FN"]).round(4).to_string())

print("\nVERDICT")
neg = df[df.cell.isin(["TN", "FN"])]
pos = df[df.cell.isin(["TP", "FP"])]
print(f"  negatives frac_pos = {neg.frac_pos.mean():.3f}  (want << 0.5: signed map "
      f"is negative nearly everywhere, which is what the old relu annihilated)")
print(f"  positives frac_pos = {pos.frac_pos.mean():.3f}  (no target: a good CAM is "
      f"SPARSE, so a minority of cells is expected and healthy)")
print(f"  old empty rate     = {df.old_empty.mean():.1%}")
print(f"  new empty rate     = {df.new_empty.mean():.1%}  (want ~0)")
print(f"  neg empty  {neg.old_empty.mean():.0%} -> {neg.new_empty.mean():.0%}"
      f"   |  pos empty  {pos.old_empty.mean():.0%} -> {pos.new_empty.mean():.0%}")

# What actually has to hold:
#  1. the negatives were broken and are now recovered, and
#  2. the recovered maps carry the same spatial contrast as the positives --
#     "not blank" and "meaningful" are different bars, and new_std is the one
#     that separates them. A flat map normalises to std ~0; a structured one
#     lands near the positives' std.
ok_recovered = neg.new_empty.mean() < 0.02 and neg.old_empty.mean() > 0.3
std_ratio = neg.new_std_norm.mean() / max(pos.new_std_norm.mean(), 1e-8)
ok_structured = 0.7 < std_ratio < 1.4
ok_kept = pos.new_empty.mean() <= pos.old_empty.mean() + 1e-9
print(f"  new_std  neg {neg.new_std_norm.mean():.3f} vs pos "
      f"{pos.new_std_norm.mean():.3f}  (ratio {std_ratio:.2f}, want ~1)")
if ok_recovered and ok_structured and ok_kept:
    print("  => PREMISE HOLDS. Negatives recovered, contrast matches positives, "
          "positives unharmed. Proceed.")
else:
    print("  => PREMISE IN DOUBT:"
          f"{'' if ok_recovered else ' negatives not recovered.'}"
          f"{'' if ok_structured else ' recovered maps flatter/noisier than positives.'}"
          f"{'' if ok_kept else ' positives got worse.'}")

# ── residual all-zero cases (positives whose signed map is negative everywhere) ──
res = df[(df.new_empty == 1)]
if len(res):
    print(f"\n=== {len(res)} maps still empty after the flip ===")
    print(res[["cell", "prob", "frac_pos", "absmean"]].sort_values("prob").round(5).to_string(index=False))
    print(f"\nthreshold = {THRESH:.6f}")
    print(f"their prob range: {res.prob.min():.5f} .. {res.prob.max():.5f}")
    print(f"margin from threshold: {(res.prob - THRESH).abs().min():.5f} .. {(res.prob - THRESH).abs().max():.5f}")
    print(f"absmean of these: {res.absmean.mean():.6f}  vs all: {df.absmean.mean():.6f}")
else:
    print("\nno residual empty maps")
