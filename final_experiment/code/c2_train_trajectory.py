# c2_train_trajectory.py
"""
C2 models trained on the diffusion TRAJECTORIES, prob-free.

While a diffusion counterfactual is generated, C0's probability is recorded every
5 denoising steps (cf_generate_levels.py): 30 values per counterfactual, and three
counterfactuals per image (levels low / mid / high). These models predict whether
C0 was correct from how that probability moved. Exploration:
final_experiment/trajectory_fingerprint.ipynb.

Prob-free: no model is given C0's probability on the original image (`orig_prob`),
and there is no MSP skip connection. They see the trajectory and nothing else.

Models
------
MLP  on the trajectory FINGERPRINT (32 hand-made features, cf_extract_diverse_features.py),
     split into four blocks, trained on each block alone and on ALL of them (the x-axis
     of the evaluation plot):

        LEVEL   where the trajectory sits and whether it crosses C0's threshold
        MONO    how steadily it moves in one direction
        SHAPE   the shape of the curve (cubic fit)
        DYN     how much and how fast it moves (velocity, acceleration, path length)

CNN  a small 1D CNN on the whole trajectory (feature map flattened, so timing is kept)
RNN  a small bidirectional GRU on the whole trajectory
     Both read two channels per step: the logit of C0's probability, sign-flipped by
     C0's prediction so every trajectory moves up, and its step-to-step change.

All three are trained the same way as the attribute MLP in c2_block_ablation.ipynb:
class-balanced BCE, Adam (weight decay 1e-4), batch 200, early stopping on the AUC of
a 10% validation split of the training fold, grouped by PATIENT (patience 10).

K (how the three levels are combined)
-------------------------------------
Same rule as the other C2 trainers (c2_train_attributes.py): the model is TRAINED on
the high level only (slot 0, the most extreme counterfactual), one trajectory per row.
At test time it is applied to each level and the K probabilities are averaged. Levels
are ordered high, mid, low, so K=1 = high alone and K=3 = all three. The model never
sees mid/low trajectories in training, so K=3 is expected to be weaker; K=1 is the
reported number.

Folds / outputs
---------------
Folds are the diffusion arm's own (patient-disjoint), so results line up with the
attribute and CNN ladders on the same rows. For every disease and K it writes

    C2_final_results/<disease>/diffusion_hml_sub20k/cf_<K>/trajectory/
        fold_<i>_predictions.csv   one <model>_<config>_prob column per model
        fold_<i>_slot_probs.npz    the per-level probabilities (high, mid, low)
        cv_summary.csv             same columns as the attribute arm's
        run_config.json

Usage
-----
    python c2_train_trajectory.py                               # all diseases
    python c2_train_trajectory.py --diseases cardiomegaly
    python c2_train_trajectory.py --smoke --out-root /tmp/test  # quick check, small subset
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
SOURCE      = "diffusion_hml_sub20k"                 # the diffusion arm's folds
DISEASES    = ["atelectasis", "cardiomegaly", "consolidation"]
N_FOLDS     = 5
SEED        = 42

LEVELS   = [3, 2, 1]        # slot order used everywhere: high, mid, low
K_VALUES = [1, 3]

# The fingerprint blocks. Every feature is in exactly one block.
BLOCKS = {
    "LEVEL": ["traj_start", "traj_end", "net_delta", "p0_level",
              "first_cross_step", "ever_crossed", "steps_to_flip", "ends_flipped", "n_crossings"],
    "MONO":  ["mono_ratio", "backtrack", "n_reversals", "reversal_rate", "spearman"],
    "SHAPE": ["p1_slope", "p2_curvature", "p3_sshape", "poly_resid"],
    "DYN":   ["total_variation", "tv_ratio",
              "vel_mean", "vel_max", "vel_min", "vel_std", "vel_absmax", "vel_argmax_pos",
              "acc_mean", "acc_max", "acc_min", "acc_std", "acc_absmax", "acc_argmax_pos"],
}
# The MLP configs (x-axis of the plot): each block alone, then all of them together.
MLP_CONFIGS = ["LEVEL", "MONO", "SHAPE", "DYN", "ALL"]
# The sequence models see the whole raw trajectory: one config each.
SEQ_MODELS = ["CNN", "RNN"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def block_columns(config):
    """'MONO' -> that block's features; 'ALL' -> every block."""
    names = list(BLOCKS) if config == "ALL" else [config]
    return [f for b in names for f in BLOCKS[b]]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data(disease):
    """Everything for one disease, one row per image.

    Returns
        img  DataFrame: path, patient_id, orig_prob, orig_pred, correct, fold
             (orig_prob is only written to the prediction files, for MSP and the
             plots; no model reads it)
        P    (n_images, 3, 30) C0 probability along each trajectory; [:, 0] is level 1 (low)
        thr  C0's decision threshold
    """
    cf_dir = os.path.join(RESULTS_DIR, "CF_levels", disease)
    thr = json.load(open(os.path.join(cf_dir, "levels.json")))["threshold"]

    # The manifest has one row per (image, level). Some diseases were generated in
    # shards, each with its own manifest file.
    manifests = [f for f in os.listdir(cf_dir) if f.startswith("manifest") and f.endswith(".csv")]
    man = pd.concat([pd.read_csv(os.path.join(cf_dir, f)) for f in manifests])
    img = man.drop_duplicates("path")[["path", "key", "orig_pred"]]

    # Correctness (the label) comes from the C2 table, like every other C2 model.
    c2 = pd.read_csv(os.path.join(RESULTS_DIR, "C2_final", disease, "c2_data.csv"),
                     usecols=["path", "patient_id", "prob", "correct"])
    c2 = c2[c2["correct"].notna()].rename(columns={"prob": "orig_prob"})
    img = img.merge(c2, on="path")
    img["correct"] = img["correct"].astype(int)

    # Fold of an image = the fold in which it is a TEST row of the diffusion arm.
    fold_dir = os.path.join(RESULTS_DIR, "C2_final_folds", disease, SOURCE, "cf_3")
    folds = pd.concat([
        pd.read_parquet(os.path.join(fold_dir, f"pairing_fold_{k}.parquet"), columns=["split", "path"])
          .query("split == 'test'").assign(fold=k)
        for k in range(N_FOLDS)])
    img = img.merge(folds[["path", "fold"]], on="path").sort_values("path").reset_index(drop=True)

    # Each trajectory file holds level_1..level_3, each an array of (t, prob) rows.
    P = np.stack([
        np.stack([np.load(os.path.join(cf_dir, "trajectory", f"{key}.npz"))[f"level_{l}"][:, 1]
                  for l in (1, 2, 3)])
        for key in img["key"]]).astype(np.float64)
    return img, P, thr


# ══════════════════════════════════════════════════════════════════════════════
# 2. INPUTS
#    Computed for ONE trajectory at a time, so every level has the same columns.
# ══════════════════════════════════════════════════════════════════════════════

def summary_stats(v, name):
    """Scalar summaries of a velocity / acceleration vector."""
    return {f"{name}_mean": v.mean(), f"{name}_max": v.max(), f"{name}_min": v.min(),
            f"{name}_std": v.std(), f"{name}_absmax": np.abs(v).max(),
            # where along the run the largest move happened, 0 = start, 1 = end
            f"{name}_argmax_pos": np.argmax(np.abs(v)) / max(len(v) - 1, 1)}


def fingerprint(probs, pred, thr):
    """The trajectory fingerprint (cf_extract_diverse_features.py), in probability space.
    `pred` is C0's prediction, so `sign` points in the intended direction of travel
    (towards the other class)."""
    sign = -1.0 if pred else 1.0
    vel, acc = np.diff(probs), np.diff(probs, n=2)
    towards = vel * sign                              # > 0 = moving the intended way
    total = np.abs(vel).sum()
    net = probs[-1] - probs[0]

    # Flip bookkeeping: `crossed` = the trajectory is on the other side of the threshold
    crossed = (probs >= thr) != bool(pred)
    first_cross = np.argmax(crossed) + 1 if crossed.any() else np.nan
    if crossed[-1]:
        # the step from which it STAYED crossed until the end
        trailing = len(crossed) if crossed.all() else np.argmin(crossed[::-1])
        steps_to_flip = len(crossed) - trailing + 1
    else:
        steps_to_flip = np.nan

    # Cubic fit over normalised time: level, slope, curvature, S-shape, misfit
    x = np.linspace(0, 1, len(probs))
    c = np.polyfit(x, probs, 3)

    f = {
        "traj_start": probs[0], "traj_end": probs[-1],
        "first_cross_step": first_cross, "ever_crossed": float(crossed.any()),
        "steps_to_flip": steps_to_flip, "ends_flipped": float(crossed[-1]),
        "n_crossings": float((np.diff(crossed.astype(int)) != 0).sum()),
        "mono_ratio": towards[towards > 0].sum() / total if total > 0 else np.nan,
        "backtrack": np.abs(towards[towards < 0]).sum(),
        "n_reversals": float((np.diff(np.sign(vel)) != 0).sum()),
        "reversal_rate": (np.diff(np.sign(vel)) != 0).sum() / max(len(vel) - 1, 1),
        "tv_ratio": total / abs(net) if abs(net) > 0.02 else np.nan,
        "spearman": spearmanr(np.arange(len(probs)), probs).correlation * sign,
        "net_delta": net * sign, "total_variation": total,
        "p0_level": c[3], "p1_slope": c[2], "p2_curvature": c[1], "p3_sshape": c[0],
        "poly_resid": np.sqrt(np.mean((probs - np.polyval(c, x)) ** 2)),
    }
    f.update(summary_stats(vel, "vel"))
    f.update(summary_stats(acc, "acc"))
    return f


def level_fingerprints(img, P, thr):
    """{level: DataFrame of the 32 fingerprint features}, one row per image."""
    pred = img["orig_pred"].values.astype(int)
    return {level: pd.DataFrame([fingerprint(P[j, i], pred[j], thr) for j in range(len(P))]).astype(float)
            for i, level in enumerate((1, 2, 3))}


def level_sequences(img, P):
    """{level: (n, 2 * 30)} the sequence models' input, two channels side by side:

        value     the logit of C0's probability along the trajectory. Logit, because the
                  probabilities pile up near 0 and 1 where a network sees almost no variation.
                  Sign-flipped by C0's prediction, so every trajectory moves UP (towards the
                  other class) -- the same "intended direction" the fingerprint uses. The
                  flipped logit is simply the logit of P(the class the CF is heading to).
        velocity  the step-to-step change of that value (0 at the first step).
    """
    p = np.clip(P, 1e-6, 1 - 1e-6)
    L = np.log(p / (1 - p))
    sign = np.where(img["orig_pred"].values == 1, -1.0, 1.0)[:, None]
    out = {}
    for i, level in enumerate((1, 2, 3)):
        value = L[:, i] * sign
        velocity = np.diff(value, axis=1, prepend=value[:, :1])
        out[level] = np.hstack([value, velocity])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODELS
# ══════════════════════════════════════════════════════════════════════════════

def mlp(n_in):
    """Same body as the attribute MLP in c2_block_ablation.ipynb."""
    return nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                         nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))


N_CHANNELS = 2          # value + velocity, see level_sequences


def as_channels(x):
    """(batch, 2 * steps) -> (batch, 2, steps)."""
    return x.view(x.shape[0], N_CHANNELS, -1)


class CNN(nn.Module):
    """Two 1D convolutions over time. The feature map is FLATTENED, not pooled, so the
    classifier knows WHEN a pattern occurs (a jump at the start vs. at the end)."""

    def __init__(self, n_steps):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(N_CHANNELS, 16, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(16, 32, 5, padding=2), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * n_steps, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):                           # x: (batch, 2 * steps)
        return self.head(self.conv(as_channels(x)))    # conv out: (batch, 32, steps)


class RNN(nn.Module):
    """A bidirectional GRU: one pass reads the trajectory start -> end, the other
    end -> start, so both ends are equally 'fresh' in memory. The two final hidden
    states are concatenated and classified."""

    def __init__(self, hidden=32):
        super().__init__()
        self.gru = nn.GRU(input_size=N_CHANNELS, hidden_size=hidden, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):                           # x: (batch, 2 * steps)
        _, h = self.gru(as_channels(x).transpose(1, 2))   # input (batch, steps, 2); h (2, batch, hidden)
        return self.head(torch.cat([h[0], h[1]], 1))      # forward + backward final states


def train(net, X, y, groups, epochs=500, patience=10, batch=200, lr=1e-3):
    """The training loop of c2_block_ablation.ipynb's fit_mlp, without the MSP skip:
    class-balanced BCE, keep the epoch with the best AUC on a patient-grouped 10%
    validation split of the training rows (an epoch must improve it by > 1e-4)."""
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
                    .split(X, y, groups))
    tensor = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    X_fit, y_fit, X_val = tensor(X[fit]), tensor(y[fit]), tensor(X[val])

    pos_weight = torch.tensor((y[fit] == 0).sum() / (y[fit] == 1).sum(), device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    order = torch.Generator().manual_seed(SEED)

    best_auc, best_epoch, wait = -1.0, 0, 0
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    for epoch in range(1, epochs + 1):
        net.train()
        for idx in torch.randperm(len(fit), generator=order).split(batch):
            opt.zero_grad()
            loss_fn(net(X_fit[idx]).squeeze(1), y_fit[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            auc = roc_auc_score(y[val], net(X_val).squeeze(1).cpu().numpy())
        if auc > best_auc + 1e-4:
            best_auc, best_epoch, wait = auc, epoch, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    return net, best_epoch


def fit_and_predict(model, X_train, y_train, g_train, X_tests, epochs):
    """Preprocess, train, and return P(correct) for each test matrix."""
    torch.manual_seed(SEED)
    if model == "MLP":
        # Some fingerprint features are NaN by design (e.g. steps_to_flip when the
        # trajectory never flips): median-fill and add a "was missing" indicator.
        prep = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
        Z = prep.fit_transform(X_train)
        net = mlp(Z.shape[1])
    else:
        # One mean/std per CHANNEL over all time steps (not per step), so the shape of
        # each trajectory is kept intact.
        ch = X_train.reshape(len(X_train), N_CHANNELS, -1)
        mu, sd = ch.mean(axis=(0, 2)), ch.std(axis=(0, 2))
        prep = lambda X: ((X.reshape(len(X), N_CHANNELS, -1) - mu[:, None]) / sd[:, None]).reshape(len(X), -1)
        Z = prep(X_train)
        net = CNN(n_steps=Z.shape[1] // N_CHANNELS) if model == "CNN" else RNN()
    transform = prep.transform if model == "MLP" else prep

    net, epoch = train(net.to(DEVICE), Z, y_train, g_train, epochs)
    net.eval()
    with torch.no_grad():
        probs = [torch.sigmoid(net(torch.tensor(transform(X), dtype=torch.float32, device=DEVICE)))
                 .squeeze(1).cpu().numpy().astype(np.float64) for X in X_tests]
    return probs, epoch


# ══════════════════════════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def run_disease(disease, args):
    t_start = time.time()
    out_dirs = {K: os.path.join(args.out_root, disease, SOURCE, f"cf_{K}", "trajectory")
                for K in K_VALUES}
    if not args.overwrite and all(os.path.exists(os.path.join(d, "cv_summary.csv"))
                                  for d in out_dirs.values()):
        print(f"{disease}: results already exist (--overwrite to rebuild)")
        return
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"\n{'=' * 72}\n{disease}\n{'=' * 72}")
    img, P, thr = load_data(disease)
    if args.smoke:                                    # 400 images per fold
        keep = img.groupby("fold").sample(400, random_state=SEED).index.sort_values()
        img, P = img.loc[keep].reset_index(drop=True), P[keep]
    print(f"{len(img):,} images, {img['patient_id'].nunique():,} patients, "
          f"{img['correct'].mean():.1%} correct, trajectories {P.shape}")

    feats = level_fingerprints(img, P, thr)
    seqs = level_sequences(img, P)

    # Every (model, config) pair that gets trained, in plot order.
    runs = [("MLP", c) for c in MLP_CONFIGS] + [(m, "TRAJ") for m in SEQ_MODELS]

    def inputs(model, config, level):
        """The (n_images, n_cols) input for one model/config at one level."""
        if model == "MLP":
            return feats[level][block_columns(config)].values
        return seqs[level]

    y, groups = img["correct"].values, img["patient_id"].values
    aucs = {(K, m, c): [] for K in K_VALUES for m, c in runs}
    epochs = {(m, c): [] for m, c in runs}

    for fold in range(N_FOLDS):
        train_rows, test_rows = img["fold"].values != fold, img["fold"].values == fold
        print(f"\nfold {fold + 1}/{N_FOLDS}: train {train_rows.sum():,} images (high level), "
              f"test {test_rows.sum():,} x 3 levels")

        base = img.loc[test_rows, ["path", "patient_id", "orig_prob", "correct"]] \
                  .rename(columns={"orig_prob": f"{disease}_prob"}).reset_index(drop=True)
        predictions = {K: base.copy() for K in K_VALUES}
        slot_probs = {}

        for m, c in runs:
            t0 = time.time()
            X = {level: inputs(m, c, level) for level in LEVELS}

            # train on the high level only (slot 0), one row per image
            X_train, y_train, g_train = X[LEVELS[0]][train_rows], y[train_rows], groups[train_rows]

            # predict each level separately -> (3, n_test), rows = high, mid, low
            per_level, epoch = fit_and_predict(
                m, X_train, y_train, g_train, [X[level][test_rows] for level in LEVELS], args.epochs)
            per_level = np.stack(per_level)
            slot_probs[f"{m}_{c}"] = per_level
            epochs[(m, c)].append(epoch)

            # K = average of the first K levels (K=1: high only, K=3: all three)
            line = []
            for K in K_VALUES:
                p = per_level[:K].mean(0)
                auc = roc_auc_score(y[test_rows], p)
                aucs[(K, m, c)].append(auc)
                predictions[K][f"{m}_{c}_prob"] = p
                line.append(f"K={K} {auc:.4f}")
            print(f"  {m} {c:<15} best epoch {epoch:<3} {' | '.join(line)}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

        # written after every fold, so a killed job still leaves the finished folds
        for K in K_VALUES:
            predictions[K].to_csv(os.path.join(out_dirs[K], f"fold_{fold}_predictions.csv"), index=False)
            np.savez_compressed(os.path.join(out_dirs[K], f"fold_{fold}_slot_probs.npz"),
                                paths=base["path"].values.astype(str), labels=y[test_rows],
                                slot_levels=np.array(["high", "mid", "low"]), **slot_probs)

    # ── summary per K ─────────────────────────────────────────────────────────
    for K in K_VALUES:
        rows = [{"model": m, "config": c, "n_folds": N_FOLDS, "k_used": K,
                 "auc_mean": np.mean(aucs[(K, m, c)]), "auc_std": np.std(aucs[(K, m, c)]),
                 "auc_folds": ";".join(f"{a:.4f}" for a in aucs[(K, m, c)])}
                for m, c in runs]
        summary = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
        summary.to_csv(os.path.join(out_dirs[K], "cv_summary.csv"), index=False)
        json.dump({"disease": disease, "cf_source": SOURCE, "cf_count": K,
                   "k_combination": "mean of the per-level probabilities of the first K levels",
                   "slot_order": ["high", "mid", "low"],
                   "train_on": "high level only (slot 0)",
                   "prob_free": True, "blocks": BLOCKS, "mlp_configs": MLP_CONFIGS,
                   "seq_models": SEQ_MODELS, "seed": SEED, "max_epochs": args.epochs,
                   "smoke": args.smoke, "n_images": len(img),
                   "best_epoch_mean": {f"{m}_{c}": float(np.mean(e)) for (m, c), e in epochs.items()}},
                  open(os.path.join(out_dirs[K], "run_config.json"), "w"), indent=2)
        print(f"\nK={K}:\n{summary[['model', 'config', 'auc_mean', 'auc_std']].to_string(index=False)}")
    print(f"\n{disease} done in {(time.time() - t_start) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diseases", nargs="+", default=DISEASES)
    ap.add_argument("--out-root", default=os.path.join(RESULTS_DIR, "C2_final_results"))
    ap.add_argument("--epochs", type=int, default=500, help="maximum epochs (early stopping ends sooner)")
    ap.add_argument("--smoke", action="store_true", help="400 images per fold, 2 epochs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 2

    # use every core LSF gave the job
    if os.environ.get("LSB_DJOB_NUMPROC"):
        torch.set_num_threads(int(os.environ["LSB_DJOB_NUMPROC"]))
    print(f"device {DEVICE}, {torch.get_num_threads()} threads, writing to {args.out_root}")

    # one disease failing must not cost the others
    failed = []
    for disease in args.diseases:
        try:
            run_disease(disease, args)
        except Exception:
            import traceback
            traceback.print_exc()
            failed.append(disease)
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
