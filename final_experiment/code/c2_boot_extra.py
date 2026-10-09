"""The E-AURC bootstrap -- the part of figure 2a the main bootstrap lacks.

The main bootstrap (bootstrapping*.ipynb) resamples patients and records mean-fold AUROC. This
module adds mean-fold E-AURC on the SAME draws (same seed, same patient set), so every
replicate is paired with the main export. ``FoldMean(..., WeightedEAURC)`` is exact: a row
drawn w times is w consecutive rows of the ranking, so with unit weights it is the notebooks'
e_aurc to the last bit (ties broken by mergesort, as there).

The notebooks call ``extra_bootstrap`` and write its output next to the main export.
"""
import numpy as np
import pandas as pd


# ── weighted AUROC (verbatim from bootstrapping.ipynb) ──────────────────────────────────────
class WeightedAUC:
    def __init__(self, table, cols):
        self.cols  = list(cols)
        self.order = np.argsort(table[self.cols].to_numpy(), axis=0, kind="mergesort")
        self.yes   = np.asarray(table["correct"], bool)[self.order]

    def __call__(self, w):
        W = w[self.order]; pos = W * self.yes; neg = W * ~self.yes
        below = np.cumsum(neg, axis=0) - neg
        return (pos * below).sum(0) / (pos.sum(0) * neg.sum(0))


# ── weighted E-AURC ─────────────────────────────────────────────────────────────────────────
def _aurc(W, err):
    """AURC of one ranking (rows already in most-confident-first order) under weights W.

    A row with weight w fills w consecutive positions; its j-th copy sits at risk
    (E + j*e) / (N + j), with E, N the weighted errors and count before it. Summed exactly,
    copy by copy -- the loop runs to max(W), and only over the rows that still have copies."""
    N_prev = np.cumsum(W) - W
    E_prev = np.cumsum(W * err) - W * err
    total, idx, j = 0.0, np.flatnonzero(W >= 1), 1
    while idx.size:
        total += ((E_prev[idx] + j * err[idx]) / (N_prev[idx] + j)).sum()
        j += 1
        idx = idx[W[idx] >= j]
    return total / W.sum()


class WeightedEAURC:
    """E-AURC of every score column of one fold: AURC minus the AURC of a perfect ranking."""
    def __init__(self, table, cols):
        self.cols = list(cols)
        wrong = ~np.asarray(table["correct"], bool)
        # Most confident first; the last order is the perfect one (every correct row first).
        self.orders = [np.argsort(-table[c].to_numpy(float), kind="mergesort") for c in self.cols]
        self.orders.append(np.argsort(wrong, kind="mergesort"))
        self.errs = [wrong[o].astype(float) for o in self.orders]

    def __call__(self, w):
        a = [_aurc(w[o], e) for o, e in zip(self.orders, self.errs)]
        return np.array(a[:-1]) - a[-1]


class FoldMean:
    """Mean over folds of a per-fold scorer -- the statistic the evaluation notebooks report."""
    def __init__(self, table, cols, scorer):
        fold = table["fold"].to_numpy()
        self.cols  = list(cols)
        self.rows  = [np.flatnonzero(fold == f) for f in np.unique(fold)]
        self.parts = [scorer(table.iloc[r], self.cols) for r in self.rows]

    def __call__(self, w):
        return np.mean([s(w[r]) for s, r in zip(self.parts, self.rows)], axis=0)


def e_aurc(correct, score):
    """Unweighted E-AURC, mergesort ties -- the check WeightedEAURC must reproduce."""
    t = pd.DataFrame({"s": np.asarray(score, float), "correct": np.asarray(correct)})
    return float(WeightedEAURC(t, ["s"])(np.ones(len(t)))[0])


META = ("correct", "patient_id", "fold")   # table columns that are not scores


# ── the replay ──────────────────────────────────────────────────────────────────────────────
def extra_bootstrap(tables, auc_cols, n_boot, seed, progress=1000):
    """Replay the main bootstrap's draws: AUROC of `auc_cols` (the pairing check -- replay a
    column the main export has and compare) and E-AURC of every score column of each table.
    `tables` must be the main bootstrap's tables -- same diseases, same rows -- or the draws
    differ.

    Returns two DataFrames, replicate x (disease, column): mean-fold AUROC and mean-fold E-AURC.
    """
    rng      = np.random.default_rng(seed)
    patients = pd.Index(sorted(set().union(*(set(t.patient_id) for t in tables.values()))))
    codes    = {d: patients.get_indexer(t.patient_id) for d, t in tables.items()}
    auc      = {d: FoldMean(t, auc_cols, WeightedAUC) for d, t in tables.items()}
    eau      = {d: FoldMean(t, [c for c in t.columns if c not in META], WeightedEAURC)
                for d, t in tables.items()}
    ra, re   = [], []
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, len(patients), len(patients)),
                             minlength=len(patients)).astype(float)
        ra.append({(d, c): v for d, s in auc.items() for c, v in zip(s.cols, s(counts[codes[d]]))})
        re.append({(d, c): v for d, s in eau.items() for c, v in zip(s.cols, s(counts[codes[d]]))})
        if progress and (b + 1) % progress == 0:
            print(f"  {b + 1}/{n_boot} replicates", flush=True)
    out = []
    for r in (ra, re):
        df = pd.DataFrame(r)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["disease", "model"])
        df.index.name = "replicate"
        out.append(df)
    return tuple(out)


def summarise(x):
    """Mean, SE, 95% percentile interval, P(>0) and the two-sided bootstrap p (floored at
    1/n), as in bootstrapping.ipynb."""
    x = np.asarray(x, float)
    p = 2 * min((x <= 0).mean(), (x >= 0).mean())
    return {"delta": float(x.mean()), "se": float(x.std(ddof=1)),
            "lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5)),
            "P(delta>0)": float((x > 0).mean()), "p": float(min(1.0, max(p, 1.0 / len(x))))}


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m); out[order] = np.minimum(adj, 1.0)
    return out


def extra_deltas(eaurc_reps, picks, diseases):
    """E-AURC gain of the final system's best model over the adjusted MSP, per disease and on
    average: E-AURC(MSP) - E-AURC(best), so positive = better (lower E-AURC). Holm runs across
    the diseases; the AVERAGE row, averaged inside each replicate, is alone in its family and
    is left as is."""
    per = {d: (eaurc_reps[d]["MSP"] - eaurc_reps[d][picks[d]["best_all"]]).to_numpy()
           for d in diseases}
    out = pd.DataFrame([{"metric": "E-AURC", "comparison": "MSP - best overall", "disease": d,
                         **summarise(x)} for d, x in per.items()])
    out["p_adj"] = holm(out.p)
    avg = {"metric": "E-AURC", "comparison": "MSP - best overall", "disease": "AVERAGE",
           **summarise(np.mean(list(per.values()), axis=0))}
    avg["p_adj"] = avg["p"]
    return pd.concat([out, pd.DataFrame([avg])], ignore_index=True)
