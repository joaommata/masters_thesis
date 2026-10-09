# c2_confidnet.py
"""
Aim 2 baseline: ConfidNet, frozen-classifier form -- a learned rejector with NO counterfactual.

ConfidNet (Corbiere et al., "Addressing Failure Prediction by Learning Model Confidence",
NeurIPS 2019) is a small MLP on the classifier's frozen penultimate features, trained with MSE
to regress the True Class Probability (TCP). It is re-implemented here on the C0 embeddings
already stored in c2_data.csv (emb_0..emb_1023, DenseNet-121 penultimate layer). C0 is never
reloaded and never fine-tuned: the paper's second stage (fine-tuning a copy of the encoder) is
skipped, so this is the stage-1 ConfidNet the paper starts from.

    python c2_confidnet.py --disease effusion --cf-source knn --cf-count 10

The --cf-* arguments do not change the model -- it reads no counterfactual column. They only
name the cached pairing whose fold membership is reused, which must be the one the Aim 2
ensemble was scored on (knn_correct_cf, K=10; see C2_final_results/<d>/<src>/cf_10/ensemble/
run_config.json). Folds are identical across K, and that is asserted against the ensemble's
oof_predictions.csv rather than assumed.

What the official implementation does (github.com/valeoai/ConfidNet, master)
-----------------------------------------------------------------------------
Head, confidnet/models/vgg16_selfconfid_classic.py L61-65 and L147-151 (identical in
mlp_selfconfid.py L20-24 / L35-39):
    uncertainty1 = Linear(in, 400)   -> ReLU
    uncertainty2 = Linear(400, 400)  -> ReLU
    uncertainty3 = Linear(400, 400)  -> ReLU
    uncertainty4 = Linear(400, 400)  -> ReLU
    uncertainty5 = Linear(400, 1)
  5 dense layers, 400 hidden units, ReLU, no dropout and no batch-norm inside the head.
  `in` is the classifier's penultimate width (512 for VGG-16); here 1024.
Output activation, confidnet/utils/losses.py L19: confidence = sigmoid(uncertainty).
Loss, confidnet/utils/losses.py L17-28 (SelfConfidMSELoss): mean of
  weights * (sigmoid(out) - TCP)^2, TCP = softmax prob of the true class; `weighting` = 1 in
  confs/selfconfid-classif.yaml L19, i.e. plain MSE. Paper eq. (4).
Optimiser, confs/selfconfid-classif.yaml L15-24: Adam, lr 1e-4, no weight decay (commented
  out), batch 128, up to 500 epochs, no LR schedule.
Model selection: the learner checkpoints every epoch (learners/selfconfid_learner.py L119) and
  does not choose one itself. The paper "mainly focus[es] on AUPR-Error" (sec. 3.1, AUPR with
  errors as the positive class); the authors' protocol picks the ConfidNet epoch with the best
  validation fail-average-precision, i.e. AUPR-Error, as documented by Jaeger et al., "A Call to
  Reflect on Evaluation Practices for Failure Detection in Image Classification", ICLR 2023,
  App. J.3. The metric is utils/metrics.py L64:
  average_precision_score(errors, -confidence).
  The paper's validation set is 10% of the training data (selfconfid-classif.yaml L8,
  valid_size: 0.1).

What is used here
-----------------
    1024 -> 400 -> 400 -> 400 -> 400 -> 1, ReLU, sigmoid output, MSE against TCP_t,
    Adam lr 1e-4, batch 128, <= 500 epochs, epoch selected by validation AUPR-Error.
Inputs are the raw emb_* features, as in the paper (--standardize fits a StandardScaler on
the fold's training rows only; off by default). The validation split is 10% of each fold's
training side, GroupShuffleSplit on patient_id (seed 42), so no patient is on both sides.

Deviations from the paper, and why
----------------------------------
1. THE TARGET -- threshold-adapted TCP (the one deliberate change to the method).
   Plain TCP assumes the decision boundary is at 0.5: a sample is correct iff TCP > 0.5. C0
   decides at its Youden threshold t, not 0.5, so plain TCP would call some correct
   predictions "unconfident" and vice versa. p is first remapped piecewise-linearly so that t
   maps to 0.5,
       g(p) = 0.5 * p / t                     if p <= t
       g(p) = 0.5 + 0.5 * (p - t) / (1 - t)   if p >  t
   and TCP is computed on the remapped value: TCP_t = g(p) if true == 1, else 1 - g(p).
   This restores ConfidNet's defining property, TCP_t > 0.5 <=> C0 is correct (asserted on
   every row, exact ties p == t excepted and counted), and keeps TCP_t in [0, 1] so the
   sigmoid output still applies.
2. Patience-based early stopping (--patience, default 20) on the same validation AUPR-Error
   rather than always running 500 epochs and picking the best afterwards. The epoch kept is
   still the best-AUPR-Error one; only epochs that could no longer win are skipped.
3. Folds are not the paper's single train/val/test split but the five patient-grouped CV
   folds every other C2 model uses, so every test row is out-of-fold and paired with the
   ensemble's prediction for the same row.

Outputs, in C2_final_results/<disease>/baselines/confidnet/
-----------------------------------------------------------
    fold_<i>_predictions.csv  path, patient_id, fold, <disease>_prob, correct, confidnet_score
    fold_<i>_history.csv      per-epoch train MSE, val MSE, val AUPR-Error, val AUROC
    oof_predictions.csv       the five folds concatenated (what c2_confidnet_compare.py reads)
    cv_summary.csv            per-fold and mean AUROC / AUPR-Error, with adjusted MSP beside it
    run_config.json           architecture, hyperparameters, t, fold source, ties, versions
"""
import argparse
import json
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import (                                          # noqa: E402
    RESULTS_DIR, RANDOM_SEED, N_FOLDS, load_c2_data, read_pairing, add_cf_args,
    cf_source_from_args,
)
from c2_msp import msp_score                                    # noqa: E402

from c0_arch import n_emb                                       # noqa: E402

# C0 embedding width: 1024 for DenseNet, else from the arch root's C0 (c0_arch.py).
N_EMB = n_emb(RESULTS_DIR)
HIDDEN = 400
N_HIDDEN_LAYERS = 4          # 4 x (Linear + ReLU), then Linear(400, 1): 5 dense layers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCORE = "confidnet_score"


# ══════════════════════════════════════════════════════════════════════════════
# TARGET
# ══════════════════════════════════════════════════════════════════════════════

def remap_threshold(p, t):
    """Piecewise-linear map of [0, 1] onto itself sending t to 0.5 (monotone, g(0)=0, g(1)=1)."""
    p = np.asarray(p, dtype=np.float64)
    if not 0.0 < t < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {t}")
    return np.where(p <= t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / (1.0 - t))


def tcp_threshold(p, y, t):
    """Threshold-adapted True Class Probability: TCP computed on g(p) instead of p."""
    g = remap_threshold(p, t)
    y = np.asarray(y)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return np.where(y == 1, g, 1.0 - g)


def test_tcp_threshold():
    """Unit test of the target. Runs at the start of every training run; also pytest-able."""
    t = 0.2
    p = np.array([0.0, 0.1, 0.2, 0.6, 1.0])
    g = remap_threshold(p, t)
    assert np.allclose(g, [0.0, 0.25, 0.5, 0.75, 1.0]), g
    assert np.allclose(tcp_threshold(p, np.ones(5, int), t), g)
    assert np.allclose(tcp_threshold(p, np.zeros(5, int), t), 1 - g)
    # At t = 0.5 it is plain TCP.
    q = np.linspace(0, 1, 11)
    assert np.allclose(tcp_threshold(q, (q > 0.3).astype(int), 0.5),
                       np.where(q > 0.3, q, 1 - q))
    # The defining property: TCP_t > 0.5 <=> correct under the rule pred = (p > t).
    rng = np.random.default_rng(0)
    for t in (0.07, 0.19, 0.39, 0.8):
        q = rng.random(10_000)
        y = rng.integers(0, 2, 10_000)
        correct = ((q > t) == y)
        assert ((tcp_threshold(q, y, t) > 0.5) == correct).all()
        # monotone in p for positives, antitone for negatives
        s = np.sort(q)
        assert (np.diff(remap_threshold(s, t)) >= 0).all()
    for bad in (0.0, 1.0):
        try:
            remap_threshold(q, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("t outside (0, 1) must raise")


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def c0_threshold(disease):
    p = os.path.join(RESULTS_DIR, "C0_final", "multilabel_ignore", disease, "threshold.txt")
    return float(open(p).read().strip())


def ensemble_oof_path(disease, source_name, cf_count):
    return os.path.join(RESULTS_DIR, "C2_final_results", disease, source_name,
                        f"cf_{cf_count}", "ensemble", "oof_predictions.csv")


def load_inputs(disease, t):
    """c2_data.csv restricted to the columns ConfidNet may see -- no counterfactual of any kind."""
    df = load_c2_data(disease)
    prob, true = f"{disease}_prob", f"{disease}_true"
    emb = [f"emb_{i}" for i in range(N_EMB)]
    missing = [c for c in ["path", "patient_id", prob, true, "correct"] + emb if c not in df]
    assert not missing, f"c2_data.csv lacks {missing[:5]}"
    assert not [c for c in df.columns if c.startswith("emb_") and c not in set(emb)], \
        f"c2_data.csv has more than {N_EMB} emb_* columns"
    df = df[["path", "patient_id", prob, true, "correct"] + emb]
    assert not any(c.startswith(("cf_", "delta_")) for c in df.columns)
    assert df["path"].is_unique, "duplicate paths in c2_data.csv"
    assert df[emb].notna().all().all(), "NaN in the embeddings"

    # correct must be exactly C0's decision at t. Stop here if it is not -- a mismatch means
    # the threshold or the labels are not the ones C0 was scored with.
    y = df[true].values
    assert np.isin(y, (0, 1)).all(), f"{true} is not 0/1 after dropping undefined rows"
    redo = ((df[prob].values > t) == (y == 1)).astype(int)
    bad = int((redo != df["correct"].values).sum())
    assert bad == 0, (f"{disease}: correct != ((prob > t) == true) on {bad:,} rows at "
                      f"t={t!r} -- wrong threshold or labels; not working around it")
    return df


def load_fold_membership(disease, src, cf_count, n_folds, df):
    """(fold, split, path) of every cached pairing, checked for patient separation."""
    lut = dict(zip(df["path"], df[f"{disease}_prob"]))
    pid = dict(zip(df["path"], df["patient_id"]))
    folds = []
    for i in range(n_folds):
        f = read_pairing(disease, src.name, cf_count, i, prob_lut=lut)[["fold", "split", "path"]]
        assert (f["fold"] == i).all(), f"pairing file {i} holds fold {sorted(f.fold.unique())}"
        assert set(f["split"]) == {"train", "test"}, f"fold {i}: splits {set(f['split'])}"
        assert f["path"].is_unique, f"fold {i}: a path is listed twice"
        unknown = set(f["path"]) - set(lut)
        assert not unknown, (f"fold {i}: {len(unknown):,} pairing paths are not rows of "
                             f"c2_data.csv (first {next(iter(unknown))!r})")
        f = f.assign(patient_id=f["path"].map(pid))
        tr = set(f.loc[f.split == "train", "patient_id"])
        te = set(f.loc[f.split == "test", "patient_id"])
        both = tr & te
        assert not both, f"fold {i}: {len(both):,} patients on both train and test"
        folds.append(f)

    test = pd.concat([f[f.split == "test"] for f in folds], ignore_index=True)
    assert test["path"].is_unique, "a row is a test row in more than one fold"
    return folds, test


def check_against_ensemble(test, disease, source_name, cf_count):
    """The union of test rows must be exactly the ensemble's rows, each in the same fold."""
    p = ensemble_oof_path(disease, source_name, cf_count)
    assert os.path.exists(p), f"{p} not found -- the ensemble this baseline is paired with"
    ens = pd.read_csv(p, usecols=["path", "fold"])
    a, b = set(test["path"]), set(ens["path"])
    assert a == b, (f"test paths != ensemble oof paths: {len(a - b):,} only here, "
                    f"{len(b - a):,} only in the ensemble ({p})")
    m = test[["path", "fold"]].merge(ens, on="path", suffixes=("", "_ens"))
    bad = int((m["fold"] != m["fold_ens"]).sum())
    assert bad == 0, f"{bad:,} rows sit in a different fold than in the ensemble"
    return p


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confidnet_head(n_in):
    """uncertainty1..5 of the official head: 4 x (Linear, ReLU), Linear(400, 1). Logit out."""
    layers, d = [], n_in
    for _ in range(N_HIDDEN_LAYERS):
        layers += [nn.Linear(d, HIDDEN), nn.ReLU()]
        d = HIDDEN
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def aupr_error(correct, conf):
    """AUPR with errors as the positive class, ranked by -confidence (utils/metrics.py L64)."""
    return float(average_precision_score(1 - np.asarray(correct), -np.asarray(conf)))


def predict(net, X, bs=8192):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(net(X[i:i + bs]).squeeze(1)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train_fold(X_tr, tcp_tr, correct_tr, groups, args):
    """Fit on 90% of the training side, select the epoch on the patient-disjoint other 10%."""
    seed_everything(RANDOM_SEED)
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED)
                    .split(X_tr, correct_tr, groups))
    assert not set(groups[fit]) & set(groups[val]), "val split shares patients with fit"
    scaler = StandardScaler().fit(X_tr) if args.standardize else None
    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    X = to_t(scaler.transform(X_tr) if scaler else X_tr)
    T = to_t(tcp_tr)
    fit_t = torch.tensor(fit, device=DEVICE)
    X_val, c_val, T_val = X[val], correct_tr[val], T[val]

    net = confidnet_head(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    gen = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)

    best, best_state, best_epoch, wait, hist = -np.inf, None, -1, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        net.train()
        tot, n = 0.0, 0
        for idx in fit_t[torch.randperm(len(fit), generator=gen).to(DEVICE)].split(args.batch_size):
            opt.zero_grad()
            loss = mse(torch.sigmoid(net(X[idx]).squeeze(1)), T[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        conf = predict(net, X_val)
        score = aupr_error(c_val, conf)
        hist.append({"epoch": epoch, "train_mse": tot / n,
                     "val_mse": float(np.mean((conf - T_val.cpu().numpy()) ** 2)),
                     "val_aupr_error": score, "val_auroc": float(roc_auc_score(c_val, conf)),
                     "seconds": round(time.time() - t0, 1)})
        if score > best:
            best, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0 or wait == 0:
            h = hist[-1]
            print(f"    epoch {epoch:>3}  train MSE {h['train_mse']:.5f}  val MSE "
                  f"{h['val_mse']:.5f}  val AUPR-E {score:.4f}  val AUROC "
                  f"{h['val_auroc']:.4f}  [{h['seconds']:.0f}s]{'  *' if wait == 0 else ''}",
                  flush=True)
        if wait >= args.patience:
            break
    net.load_state_dict(best_state)
    net.eval()
    return net, scaler, best_epoch, best, pd.DataFrame(hist), (len(fit), len(val))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    test_tcp_threshold()
    seed_everything(RANDOM_SEED)
    torch.set_num_threads(args.threads or torch.get_num_threads())
    src = cf_source_from_args(args)
    d = args.disease
    prob = f"{d}_prob"
    out_dir = os.path.join(RESULTS_DIR, "C2_final_results", d, "baselines", "confidnet")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "cv_summary.csv")) and not args.overwrite:
        print(f"  cv_summary.csv already present in {out_dir}  (--overwrite to rebuild)")
        return

    print(f"\n{'='*72}\n  CONFIDNET (frozen C0): {d} | folds of {src.describe()} "
          f"K={args.cf_count}\n{'='*72}")
    t = c0_threshold(d)
    print(f"  t = {t!r}   device: {DEVICE}   threads: {torch.get_num_threads()}")

    df = load_inputs(d, t)
    tcp = tcp_threshold(df[prob].values, df[f"{d}_true"].values, t)
    ties = df[prob].values == t
    mism = int(((tcp > 0.5) != (df["correct"].values == 1))[~ties].sum())
    assert mism == 0, f"TCP_t > 0.5 disagrees with correct on {mism:,} non-tie rows"
    n_ties = int(ties.sum())
    print(f"  {len(df):,} rows; correct == ((p > t) == true) on all of them; "
          f"(TCP_t > 0.5) == correct off ties; {n_ties} tie rows (p == t)")

    folds, test_all = load_fold_membership(d, src, args.cf_count, args.n_folds, df)
    ens_path = check_against_ensemble(test_all, d, src.name, args.cf_count)
    print(f"  folds: patient-disjoint in all {args.n_folds}; test rows == ensemble oof rows, "
          f"same folds ({len(test_all):,})")
    unused = len(df) - len(test_all)
    if unused:
        print(f"  note: {unused:,} rows of c2_data.csv are in no test fold")

    row = pd.Series(np.arange(len(df)), index=df["path"].values)
    emb = df[[f"emb_{i}" for i in range(N_EMB)]].values.astype(np.float32)
    correct = df["correct"].values.astype(int)
    groups = df["patient_id"].values

    rows, preds = [], []
    for f in folds:
        i = int(f["fold"].iloc[0])
        tr = row[f.loc[f.split == "train", "path"]].values
        te = row[f.loc[f.split == "test", "path"]].values
        print(f"\n{'-'*72}\n  FOLD {i + 1}/{args.n_folds}  (train {len(tr):,} | test "
              f"{len(te):,})\n{'-'*72}", flush=True)
        t0 = time.time()
        net, scaler, best_epoch, val_aupr, hist, (n_fit, n_val) = train_fold(
            emb[tr], tcp[tr], correct[tr], groups[tr], args)
        X_te = scaler.transform(emb[te]) if scaler else emb[te]
        score = predict(net, torch.tensor(X_te, dtype=torch.float32, device=DEVICE))

        pred = df.iloc[te][["path", "patient_id", prob, "correct"]].copy()
        pred.insert(2, "fold", i)
        pred[SCORE] = score
        pred.to_csv(os.path.join(out_dir, f"fold_{i}_predictions.csv"), index=False)
        hist.to_csv(os.path.join(out_dir, f"fold_{i}_history.csv"), index=False)
        preds.append(pred)

        y = pred["correct"].values
        msp = msp_score(pred[prob].values, t, "threshold")
        rows.append({"fold": i, "n_test": len(te), "n_fit": n_fit, "n_val": n_val,
                     "best_epoch": best_epoch, "epochs_run": len(hist),
                     "val_aupr_error": val_aupr,
                     "auroc": float(roc_auc_score(y, score)),
                     "aupr_error": aupr_error(y, score),
                     "msp_auroc": float(roc_auc_score(y, msp)),
                     "msp_aupr_error": aupr_error(y, msp)})
        r = rows[-1]
        print(f"  best epoch {best_epoch} (val AUPR-E {val_aupr:.4f})  test AUROC "
              f"{r['auroc']:.4f}  (adjusted MSP {r['msp_auroc']:.4f})  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    oof = pd.concat(preds, ignore_index=True)
    assert set(oof["path"]) == set(test_all["path"])
    oof.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)

    summ = pd.DataFrame(rows)
    metrics = ["val_aupr_error", "auroc", "aupr_error", "msp_auroc", "msp_aupr_error"]
    mean = {"fold": "mean", "n_test": int(summ.n_test.sum()),
            **{c: float(summ[c].mean()) for c in metrics}}
    std = {"fold": "std", **{c: float(summ[c].std(ddof=0)) for c in metrics}}
    summ = pd.concat([summ, pd.DataFrame([mean, std])], ignore_index=True)
    for c in ("n_test", "n_fit", "n_val", "best_epoch", "epochs_run"):
        summ[c] = summ[c].astype("Int64")
    summ.insert(0, "model", "ConfidNet")
    summ.to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)

    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump({
            "disease": d, "model": "ConfidNet (frozen classifier, stage 1 only)",
            "reference": "Corbiere et al., NeurIPS 2019; github.com/valeoai/ConfidNet",
            "architecture": {"input": N_EMB, "hidden": [HIDDEN] * N_HIDDEN_LAYERS, "output": 1,
                             "dense_layers": N_HIDDEN_LAYERS + 1, "activation": "ReLU",
                             "output_activation": "sigmoid", "dropout": None},
            "inputs": f"emb_0..emb_{N_EMB - 1} of c2_data.csv (C0 penultimate features)",
            "standardize": bool(args.standardize),
            "target": "threshold-adapted TCP: TCP on g(p), g piecewise linear with g(t) = 0.5",
            "loss": "MSE(sigmoid(out), TCP_t)",
            "optimizer": {"name": "Adam", "lr": args.lr, "weight_decay": 0.0},
            "batch_size": args.batch_size, "max_epochs": args.epochs,
            "patience": args.patience, "selection": "validation AUPR-Error (errors positive)",
            "val_split": "GroupShuffleSplit(test_size=0.1, patient_id, seed 42) of each "
                         "fold's training side",
            "seed": RANDOM_SEED, "device": DEVICE,
            "threshold_t": t,
            "threshold_file": os.path.join(RESULTS_DIR, "C0_final", "multilabel_ignore", d,
                                           "threshold.txt"),
            "n_tie_rows": n_ties,
            "fold_source": {"cf_source": src.name, "cf_count": args.cf_count,
                            "n_folds": args.n_folds, "matched_ensemble_oof": ens_path},
            "n_rows": int(len(oof)),
            "best_epochs": [int(r["best_epoch"]) for r in rows],
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__,
                         "sklearn": sklearn.__version__},
        }, fh, indent=2)

    sanity(d, src.name, args.cf_count, summ)
    print(f"\n  wrote {out_dir}")


def sanity(disease, source_name, cf_count, summ):
    """ConfidNet next to the prob-free query-only attribute MLP and the adjusted MSP.

    The B3/B5 rows of the old ladder no longer exist (the block ablation replaced it, see
    c2_train_attributes.py); MLP_I -- attributes + C0 embedding of the query, no CF -- is the
    nearest learned no-counterfactual comparator."""
    m = summ[summ.fold == "mean"].iloc[0]
    print(f"\n{'='*72}\n  SANITY (mean fold AUROC)\n{'='*72}")
    print(f"  ConfidNet            {m['auroc']:.4f}")
    print(f"  adjusted MSP         {m['msp_auroc']:.4f}   (ConfidNet {m['auroc'] - m['msp_auroc']:+.4f})")
    p = os.path.join(RESULTS_DIR, "C2_final_results", disease, source_name, f"cf_{cf_count}",
                     "attributes", "cv_summary.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        for cfg in ("I", "I+C+D"):
            hit = s[(s.model == "MLP") & (s.config == cfg)]
            if len(hit):
                a = float(hit.auc_mean.iloc[0])
                print(f"  MLP_{cfg:<16} {a:.4f}   (ConfidNet {m['auroc'] - a:+.4f})")
    if m["auroc"] < 0.5:
        print("  !! ConfidNet AUROC < 0.5 -- scores look reversed")
    if m["auroc"] < m["msp_auroc"] - 0.02:
        print("  !! ConfidNet is >0.02 below adjusted MSP -- check target, threshold, folds")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if cv_summary.csv already exists")
    ap.add_argument("--standardize", action="store_true",
                    help="StandardScaler on the embeddings, fit on the fold's train rows")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--self-test", action="store_true", help="run the TCP_t unit test and exit")
    add_cf_args(ap)
    ap.set_defaults(cf_count=10)
    args = ap.parse_args()
    if args.self_test:
        test_tcp_threshold()
        print("test_tcp_threshold: ok")
        return
    if args.n_folds != N_FOLDS:
        print(f"  note: --n-folds {args.n_folds} (default {N_FOLDS})")
    run(args)


if __name__ == "__main__":
    main()
