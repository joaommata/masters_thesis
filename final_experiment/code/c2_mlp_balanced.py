# c2_mlp_balanced.py
"""
The two C2 attribute models chosen after the ablation (c2_mlp_ablation.py, results in
attributes_mlp_balanced/). Same folds for both; model selection on the best AUC of a
patient-grouped 20% val split of train.

    MLP   torch 64-32-16 ReLU, Adam 1e-3, batch 200, dropout 0.3, L2 1e-3,
          class-balanced BCE (pos_weight = n_neg / n_pos), mean of 5 seeds,
          patience 10, max 200 epochs. Configs that contain `prob` additionally get:
            - threshold-MSP s = p if p>=t else 1-p and the flag 1[p>=t], appended to
              the input AND fed linearly straight to the output logit (skip)
            - the body's output layer zero-initialised, so the net starts as a
              linear fit on the skip and only moves if val AUC improves
            - PCA(32) on every block wider than 32 cols (fit on train rows)
          Configs without `prob` (B2, B3, M1) get neither skip nor PCA -- without the
          skip, PCA only throws away signal (ablation: B2 0.688 vs 0.709 without).
    HGB   sklearn HistGradientBoosting (lr 0.05, 15 leaves, min leaf 50, L2 1.0,
          class_weight balanced) on the raw blocks, + the MSP features where the
          config has prob; number of trees (<=500) picked on the same val split.

t = C0's Youden threshold (C0_final/multilabel_<policy>/<disease>/threshold.txt).
Standalone: reads cached folds and the existing sklearn results, writes ONLY to
.../<source>/cf_<K>/attributes_mlp_best/.

    python c2_mlp_balanced.py --subset 20k            # knn_correct_cf_sub20k, K=1, 3 diseases
    python c2_mlp_balanced.py --subset 20k --plot
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import RESULTS_DIR, RANDOM_SEED, load_folds, add_cf_args, cf_source_from_args  # noqa: E402
from c2_train_attributes import (CONFIG_BLOCKS, CF_DEPENDENT, feature_columns,            # noqa: E402
                                 block_matrix, config_available)
from c2_msp import msp_threshold                                                           # noqa: E402

CONFIGS = ["B1", "B2", "B3", "B4", "B5", "M1", "M2", "M3", "M4", "M5", "M6",
           "MCF1", "MCF2", "MCF3", "MCF4", "MCF5"]
# name -> (class-weighted, msp skip, pca dim, dropout, weight decay, zero-init, seeds);
# skip, PCA and zero-init only apply to configs that contain `prob` (see fit_predict)
VARIANTS = {"MLP": (True, True, 32, 0.3, 1e-3, True, 5)}
MODELS = [*VARIANTS, "HGB"]
LABELS = {"MLP": "MLP (MSP skip, PCA, zero-init, 5 seeds)", "HGB": "HistGradientBoosting"}
VAL_FRAC = 0.20
# Height of the plotted AUROC window. Matches C2_evaluation.ipynb so the panels here
# and there are read on the same scale.
WIN = 0.05
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def out_dir(disease, src_name, k):
    return os.path.join(RESULTS_DIR, "C2_final_results", disease, src_name,
                        f"cf_{k}", "attributes_mlp_best")


def c0_threshold(disease, policy):
    p = os.path.join(RESULTS_DIR, "C0_final", f"multilabel_{policy}", disease, "threshold.txt")
    return float(open(p).read().strip())


def msp_feats(df, disease, t):
    """(n, 2): threshold MSP and the predicted-class flag -- the two branches of the
    threshold MSP live on different ranges, so the flag lets a linear skip offset them."""
    p = df[f"{disease}_prob"].values.astype(np.float64)
    return np.column_stack([msp_threshold(p, t), (p >= t).astype(np.float64)])


class Net(nn.Module):
    def __init__(self, d, n_skip, drop, zero_init=False):
        super().__init__()
        layers, w = [], d
        for h in (64, 32, 16):
            layers += [nn.Linear(w, h), nn.ReLU()] + ([nn.Dropout(drop)] if drop else [])
            w = h
        self.body = nn.Sequential(*layers, nn.Linear(w, 1))
        if zero_init:
            nn.init.zeros_(self.body[-1].weight)
            nn.init.zeros_(self.body[-1].bias)
        self.skip = nn.Linear(n_skip, 1) if n_skip else None

    def forward(self, x, s):
        out = self.body(x).squeeze(1)
        return out + self.skip(s).squeeze(1) if self.skip is not None else out


def prepare(blocks_fit, blocks_other, pca_dim):
    """Standardise each block on fit rows; PCA a block if it is wider than pca_dim.
    blocks_fit: list of (n_fit, d_b); blocks_other: list of such lists (val, test slots)."""
    fit_out, other_out = [], [[] for _ in blocks_other]
    for b, Xb in enumerate(blocks_fit):
        tf = [StandardScaler().fit(Xb)]
        Z = tf[0].transform(Xb)
        if pca_dim and Xb.shape[1] > pca_dim:
            tf.append(PCA(pca_dim, random_state=RANDOM_SEED).fit(Z))
            Z = tf[1].transform(Z)
        fit_out.append(Z)
        for i, other in enumerate(blocks_other):
            Y = other[b]
            for f in tf:
                Y = f.transform(Y)
            other_out[i].append(Y)
    return np.hstack(fit_out), [np.hstack(o) for o in other_out]


def fit_predict(fit, val, tests, y_fit, y_val, variant, max_epochs=200, patience=10, bs=200):
    """fit/val/tests: (blocks, skip) pairs. Returns (per-test probs, best epoch)."""
    weighted, use_skip, pca_dim, drop, wd, zero_init, n_seeds = VARIANTS[variant]
    # the MSP skip is only an allowed input where the config already has C0's prob
    use_skip = use_skip and fit[1] is not None
    if not use_skip:                  # no prob -> no skip, so PCA and zero-init only hurt
        pca_dim, zero_init = None, False
    blocks = list(fit[0]) + ([fit[1]] if use_skip else [])
    others = [[b for b in o[0]] + ([o[1]] if use_skip else []) for o in [val] + tests]
    X_fit, X_oth = prepare(blocks, others, pca_dim)
    if use_skip:
        ss = StandardScaler().fit(fit[1])
        S_fit, S_oth = ss.transform(fit[1]), [ss.transform(o[1]) for o in [val] + tests]
    else:
        S_fit, S_oth = np.zeros((len(X_fit), 0)), [np.zeros((len(x), 0)) for x in X_oth]
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    Xf, Sf, yf = T(X_fit), T(S_fit), T(y_fit)
    Xv, Sv = T(X_oth[0]), T(S_oth[0])
    runs = [_train(Xf, Sf, yf, Xv, Sv, y_val, [(T(x), T(s)) for x, s in zip(X_oth[1:], S_oth[1:])],
                   weighted, drop, wd, zero_init, RANDOM_SEED + i, max_epochs, patience, bs)
            for i in range(n_seeds)]
    probs = [np.mean([r[0][j] for r in runs], axis=0) for j in range(len(tests))]
    return probs, float(np.mean([r[1] for r in runs]))


def _train(Xf, Sf, yf, Xv, Sv, y_val, tests, weighted, drop, wd, zero_init, seed,
           max_epochs, patience, bs):
    torch.manual_seed(seed)
    y_fit = yf.cpu().numpy()
    net = Net(Xf.shape[1], Sf.shape[1], drop, zero_init).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    pw = (len(y_fit) - y_fit.sum()) / y_fit.sum() if weighted else 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=DEV))

    best, best_ep, best_state, bad = -1.0, 0, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(max_epochs):
        net.train()
        for idx in torch.randperm(len(yf), generator=g).split(bs):
            idx = idx.to(DEV)
            opt.zero_grad()
            loss_fn(net(Xf[idx], Sf[idx]), yf[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            auc = roc_auc_score(y_val, net(Xv, Sv).cpu().numpy())
        if auc > best:
            best, best_ep, bad = auc, ep + 1, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        probs = [torch.sigmoid(net(x, s)).cpu().numpy().astype(np.float64) for x, s in tests]
    return probs, best_ep


def fit_predict_hgb(fit, val, tests, y_fit, y_val, max_iter=500):
    """Gradient boosting on raw blocks (+ MSP features where allowed). Trees are
    grown to max_iter once; the stage with the best val AUC is used for test."""
    cat = lambda o: np.hstack(list(o[0]) + ([o[1]] if o[1] is not None else []))
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=max_iter,
                                       max_leaf_nodes=15, min_samples_leaf=50,
                                       l2_regularization=1.0, early_stopping=False,
                                       class_weight="balanced", random_state=RANDOM_SEED)
    m.fit(cat(fit), y_fit)
    val_auc = [roc_auc_score(y_val, d) for d in m.staged_decision_function(cat(val))]
    best = int(np.argmax(val_auc))
    probs = []
    for o in tests:
        for i, d in enumerate(m.staged_decision_function(cat(o))):
            if i == best:
                probs.append(1 / (1 + np.exp(-d.astype(np.float64))))
                break
    return probs, best + 1


def run(args, disease):
    src = cf_source_from_args(args)
    od = out_dir(disease, src.name, args.cf_count)
    os.makedirs(od, exist_ok=True)
    configs = [c.strip() for c in args.configs.split(",")] if args.configs else CONFIGS
    t = c0_threshold(disease, args.policy)
    print(f"\n== {disease} | {src.describe()} | K={args.cf_count} | t={t:.4f} | {DEV}\n   -> {od}")

    rows = []
    for fold in load_folds(disease, src, args.cf_count, n_folds=args.n_folds):
        tr = fold.train
        y = tr["correct"].values.astype(np.float32)
        cols = feature_columns(tr, disease)
        # same patient-grouped holdout for every variant and config
        fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC,
                                                  random_state=RANDOM_SEED)
                                .split(tr, groups=tr["patient_id"]))
        slots = [fold.test_slot(j) for j in range(fold.n_slots)]
        y_test = slots[0]["correct"].values
        pred = slots[0][["path", "patient_id", f"{disease}_prob", "correct"]].copy()
        S_tr = msp_feats(tr, disease, t)
        S_te = [msp_feats(s, disease, t) for s in slots]
        pred["msp_threshold"] = S_te[0][:, 0]
        rows.append({"fold": fold.index, "config": "-", "model": "MSP_threshold",
                     "auc": roc_auc_score(y_test, S_te[0][:, 0]), "best_epoch": 0})
        print(f"  fold {fold.index}: fit {len(fit_idx):,} val {len(val_idx):,} "
              f"test {len(fold):,} x{fold.n_slots}  pos rate {y.mean():.3f}  "
              f"MSP_thr {rows[-1]['auc']:.4f}")

        for cfg in configs:
            if not config_available(cfg, cols):
                continue
            B = [block_matrix(tr, b, cols, disease) for b in CONFIG_BLOCKS[cfg]]
            use = range(len(slots)) if cfg in CF_DEPENDENT else range(1)
            B_te = [[block_matrix(slots[j], b, cols, disease) for b in CONFIG_BLOCKS[cfg]]
                    for j in use]
            has_prob = "prob" in CONFIG_BLOCKS[cfg]      # B2, B3, M1: no skip
            fit = ([b[fit_idx] for b in B], S_tr[fit_idx] if has_prob else None)
            val = ([b[val_idx] for b in B], S_tr[val_idx] if has_prob else None)
            tests = [(B_te[i], S_te[j] if has_prob else None) for i, j in enumerate(use)]
            line = f"    {cfg:<5} dim={sum(b.shape[1] for b in B):<5}"
            for v in MODELS:
                t0 = time.time()
                probs, ep = (fit_predict_hgb(fit, val, tests, y[fit_idx], y[val_idx]) if v == "HGB"
                             else fit_predict(fit, val, tests, y[fit_idx], y[val_idx], v))
                p = np.mean(probs, axis=0)
                auc = roc_auc_score(y_test, p)
                pred[f"{v}_{cfg}_prob"] = p
                rows.append({"fold": fold.index, "config": cfg, "model": v,
                             "auc": auc, "best_epoch": ep})
                line += f"  {v} {auc:.4f} ({ep:>3.0f}ep {time.time()-t0:.0f}s)"
            print(line, flush=True)
            del B, B_te
        pred.to_csv(os.path.join(od, f"fold_{fold.index}_predictions.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, "fold_aucs.csv"), index=False)
    s = df.groupby(["model", "config"]).agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                                            epoch_mean=("best_epoch", "mean")).reset_index()
    s.to_csv(os.path.join(od, "cv_summary.csv"), index=False)
    print(f"  MSP_threshold {s.loc[s.model == 'MSP_threshold', 'auc_mean'].iloc[0]:.4f}")
    print(s[s.model != "MSP_threshold"].pivot(index="config", columns="model", values="auc_mean")
          .reindex(configs)[MODELS].round(4).to_string())


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = cf_source_from_args(args)
    diseases = [d.strip() for d in args.diseases.split(",")]
    # Panel sizes and palette come from the thesis design system, so this figure reads
    # like its homologues in C2_evaluation.ipynb.
    from thesis_style import (FULL, MM, FS, LW, GREY, MUTED, INK, FAMILY, ENS,
                              msp_line, msp_handle, line_handle, apply)
    apply()
    # One colour per MODEL FAMILY: the existing sklearn MLP and the new torch MLP are
    # both attribute models (blue, light -> dark); HGB is a different learner (neutral).
    STYLE = {"MLP (sklearn, existing)": ("#7CB4FC", "o"),
             "MLP":                     (FAMILY["attr"], "s"),
             "HGB":                     (ENS, "D")}
    fig, axes = plt.subplots(1, len(diseases), figsize=(FULL / 3 * len(diseases), 76 * MM),
                             squeeze=False)
    table = []
    for ax, d in zip(axes[0], diseases):
        od = out_dir(d, src.name, args.cf_count)
        new = pd.read_csv(os.path.join(od, "cv_summary.csv"))
        old = pd.read_csv(os.path.join(os.path.dirname(od), "attributes", "cv_summary.csv"))
        old = old[old["model"] == "MLP"].assign(model="MLP (sklearn, existing)")
        s = pd.concat([old[["model", "config", "auc_mean"]], new[["model", "config", "auc_mean"]]])
        s = s[s["config"].isin(CONFIGS)]
        piv = s.pivot(index="config", columns="model", values="auc_mean").reindex(CONFIGS)
        x = np.arange(len(CONFIGS))
        m = new.loc[new["model"] == "MSP_threshold", "auc_mean"]
        msp = float(m.iloc[0]) if len(m) else None

        # Y WINDOW. Without one, B2 (~0.70) and M1 (~0.76) set the floor and the whole
        # figure spans ~0.17 AUROC -- which renders the differences that matter here
        # (~0.001) as a few pixels. A fixed-height window anchored at the top of the data
        # keeps the scale comparable across panels; points below it are CLIPPED to the
        # floor and named under the panel, never silently dropped.
        vals = [v for c in STYLE if c in piv for v in piv[c].dropna().values]
        top = max(vals + ([msp] if msp is not None else []))
        lo = top - WIN
        hi = top + 0.06 * WIN

        for col, (colour, mk) in STYLE.items():
            if col not in piv:
                continue
            y = piv[col].values.astype(float)
            clipped = y < lo
            ax.plot(x, np.clip(y, lo, None), marker=mk, ms=4, lw=LW["line"], color=colour,
                    mec="white", mew=0.6, label=LABELS.get(col, col), zorder=3)
            # Hollow the markers that are not really there, so a clipped point cannot be
            # read as a real value sitting on the axis floor.
            if clipped.any():
                ax.plot(x[clipped], np.full(clipped.sum(), lo), linestyle="none", marker=mk,
                        ms=4, mfc="white", mec=colour, mew=0.9, zorder=4)
        if msp is not None:
            msp_line(ax, msp)
            piv["MSP_threshold"] = msp

        below = sorted({CONFIGS[i] for c in STYLE if c in piv
                        for i, v in enumerate(piv[c].values.astype(float))
                        if np.isfinite(v) and v < lo})
        if below:
            ax.annotate(f"below axis: {', '.join(below)}  (min {min(vals):.3f})",
                        xy=(0, 0), xycoords="axes fraction", xytext=(0, -34),
                        textcoords="offset points", fontsize=FS["tiny"], color=MUTED)

        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.5, len(CONFIGS) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(CONFIGS, rotation=90, fontsize=FS["tiny"])
        ax.set_title(d)
        ax.set_ylabel("AUROC (mean of 5 folds)" if ax is axes[0][0] else "")
        ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.set_box_aspect(1)
        table.append(piv.assign(disease=d))

    handles = [line_handle(c, LABELS.get(k, k), mk) for k, (c, mk) in STYLE.items()]
    fig.legend(handles=handles + [msp_handle("MSP (threshold)")], loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"C2 attribute models: MLP + HGB | {src.name} | K={args.cf_count}", y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    root = os.path.join(RESULTS_DIR, "C2_final_results", f"mlp_best_{src.name}_cf{args.cf_count}")
    fig.savefig(root + ".png", dpi=150)
    t = pd.concat(table)
    for v in MODELS:
        if {"MSP_threshold", v} <= set(t.columns):
            t[f"{v}_minus_msp"] = t[v] - t["MSP_threshold"]
    t.round(4).to_csv(root + ".csv")
    print(t.round(4).to_string())
    print(f"\nwrote {root}.png / .csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", default="")
    ap.add_argument("--diseases", default="atelectasis,cardiomegaly,consolidation")
    ap.add_argument("--configs", default="")
    ap.add_argument("--policy", default="ignore", help="C0 label policy, for the threshold")
    ap.add_argument("--plot", action="store_true")
    add_cf_args(ap)
    args = ap.parse_args()
    if args.plot:
        plot(args)
    else:
        for d in ([args.disease] if args.disease else args.diseases.split(",")):
            run(args, d.strip())


if __name__ == "__main__":
    main()
