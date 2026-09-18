"""Compare two orientation rules for the signed Grad-CAM.

  A  sign = +1 if prob > threshold   (what is on disk now)
  B  sign = +1 if logit > 0          (orient by where the evidence points)

A is the DECISION C0 reports; B is the DIRECTION the gradient points. They differ
on every row in the t < p <= 0.5 band -- 99.9% of consolidation's positive
predictions -- and on those rows A fails to flip, so the ReLU clips the map to
zero exactly as the original positive-only bug did.
"""
import json, os, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DISEASE = sys.argv[1] if len(sys.argv) > 1 else "consolidation"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
DATA = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
RES = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
MD = os.path.join(RES, "C0_final", "multilabel_ignore"); OD = os.path.join(MD, DISEASE)
COL = {"effusion":"Pleural Effusion","cardiomegaly":"Cardiomegaly","atelectasis":"Atelectasis",
       "consolidation":"Consolidation","edema":"Edema"}[DISEASE]
LAB = json.load(open(os.path.join(MD,"labels.json")))["labels"]; TI = LAB.index(COL)
T = float(open(os.path.join(OD,"threshold.txt")).read().split()[0])
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = pd.read_csv(os.path.join(OD,f"C2_dataset_c0_{DISEASE}.csv"),
                usecols=["path","prob","true","certain"])
m = m[(m.certain==1)&m.true.notna()]
m["cell"] = np.select([(m.prob>T)&(m.true==1),(m.prob<=T)&(m.true==0),
                       (m.prob>T)&(m.true==0),(m.prob<=T)&(m.true==1)],
                      ["TP","TN","FP","FN"],default="?")
m["band"] = (m.prob>T)&(m.prob<=0.5)
rng = np.random.RandomState(0)
s = m.groupby("cell",group_keys=False)[["path","prob","cell","band"]].apply(
        lambda g: g.sample(min(N,len(g)),random_state=rng))

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(LAB))
sd = torch.load(os.path.join(MD,"c0_best.pt"),map_location="cpu")
model.load_state_dict(sd.get("model_state_dict",sd.get("state_dict",sd)))
model = model.to(DEV).eval()
cache = {}
def hook(mod,i,o):
    cache["acts"]=torch.relu(o).detach()
    if o.requires_grad: o.register_hook(lambda g: cache.__setitem__("grads",g.detach()))
model.features.register_forward_hook(hook)
tf = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

rows=[]
for _,r in s.iterrows():
    x = tf(Image.open(os.path.join(DATA,r["path"])).convert("RGB")).unsqueeze(0).to(DEV)
    cache.clear(); model.zero_grad(set_to_none=True)
    lg = model(x); lg[:,TI].backward(torch.ones(1,device=DEV))
    w = cache["grads"].mean(dim=(2,3),keepdim=True)
    signed = (w*cache["acts"]).sum(dim=1)[0].cpu().numpy()
    logit = float(lg[0,TI].detach().cpu())
    a = np.maximum(signed*(1.0 if r["prob"]>T else -1.0),0)   # rule A (on disk)
    b = np.maximum(signed*(1.0 if logit>0    else -1.0),0)    # rule B (proposed)
    rows.append(dict(cell=r["cell"],band=bool(r["band"]),prob=r["prob"],logit=logit,
                     A_empty=int(a.max()<=0),B_empty=int(b.max()<=0),
                     A_std=float((a/(a.max()+1e-8)).std()),
                     B_std=float((b/(b.max()+1e-8)).std()),
                     agree=int((r["prob"]>T)==(logit>0))))
df=pd.DataFrame(rows)
print(f"\ndisease={DISEASE} threshold={T:.4f} n={len(df)}")
print(f"rules agree on {df.agree.mean():.1%} of rows; band rows = {df.band.mean():.1%}\n")
print(df.groupby("cell").agg(n=("cell","size"),band=("band","mean"),
    A_empty=("A_empty","mean"),B_empty=("B_empty","mean"),
    A_std=("A_std","mean"),B_std=("B_std","mean")
    ).reindex(["TP","TN","FP","FN"]).round(4).to_string())
print(f"\noverall empty:  rule A (on disk) {df.A_empty.mean():.1%}   rule B {df.B_empty.mean():.1%}")
bd=df[df.band]
if len(bd):
    print(f"band rows only: rule A {bd.A_empty.mean():.1%}   rule B {bd.B_empty.mean():.1%}  (n={len(bd)})")
