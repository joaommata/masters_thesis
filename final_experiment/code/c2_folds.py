# c2_folds.py
"""
Computes the CV folds and their counterfactual pairing once, and cache THE PAIRING ONLY.
The expensive thing about a fold is not its feature table, it is deciding which
counterfactual each query is matched to: a nearest-neighbour search over the
standardised attribute space, ~2.3 hours per fold.

What is saved:
--------------------------------
    fold        which of the K folds this row belongs to
    split       'train' or 'test' within that fold
    path        the query image
    cf_paths    the K pipe-separated counterfactual paths   <- the expensive part
    cf_prob     mean C0 probability over those K
    cf_probs    the K individual C0 probabilities, pipe-separated, same order

That is ~455k (5x91) rows x 5 columns, tens of MB. Materialising it (addint the deltas, the embeddings, the attributes) instead would cost
about 12.4 GB per (disease, cf_source, K) and ~62 GB across the five competition tasks. 
-> So we cache only the pairing ("what fold each image belongs to and who it was matched to")

The full fold data is materialized when the trainer needs it.

Note that cf_paths genuinely differs per fold for the same image: the CF pool IS
the train partition, so a row's counterfactual changes depending on the split. Hence the
(fold, split) key rather than one pairing per image.
.
Reproducing the features (at training time) is cheap:
------------------------
`materialize_fold` rebuilds the full frame. The only subtlety is `delta_*`: those
are differences in the SCALED attribute space, where the StandardScaler was fit
on the fold's TRAIN attributes. Rather than store the scaler, it is refit at load
time on the same rows and columns -- deterministic, and it cannot drift out of
sync with a stored column order.

Fold construction
-----------------
StratifiedGroupKFold on patient_id, stratified on `correct`. 
Grouping by patient is not optional! Otherwise patient's images land on both sides of the split. 

    python c2_folds.py --disease effusion --cf-source knn --cf-count 1

More than one counterfactual per query
--------------------------------------
cf_paths is NEAREST FIRST, and nothing else about a fold depends on K: the split
is seeded, the scaler is fit on the same train rows, and the pools are routed by
prediction. So the K=5 pairing is literally the first five paths of the K=10 one.

Pair once at the largest K I intend to study and then just slice the rest out of it:

    python c2_folds.py --disease effusion --cf-count 10                # ~11.5h
    python c2_folds.py --disease effusion --cf-count 5  --derive       # seconds
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_cf_sources import build_cf_source, CF_SOURCE_KINDS, KNN_STRATEGIES  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code", "c2"))
from c2_prepare_data_simulated_cf import add_clinical_ratios     # noqa: E402
from c2_feature_spec import C0_DERIVED_COLS, RSNA_META_COLS      # noqa: E402

RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")
N_FOLDS = 5
RANDOM_SEED = 42

PAIRING_COLS = ["fold", "split", "path", "cf_paths", "cf_prob", "cf_probs"]

# Pairing files written before 2026-09-09 have no cf_probs column. It is
# recoverable for a KNN source -- every CF is a real row of c2_data.csv -- so it
# is backfilled on read rather than forcing a rebuild of the cf_1 pairings.
LEGACY_PAIRING_COLS = ["fold", "split", "path", "cf_paths", "cf_prob"]


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

def c2_data_path(disease):
    return os.path.join(RESULTS_DIR, "C2_final", disease, "c2_data.csv")


def fold_dir(disease, source_name, cf_count):
    return os.path.join(RESULTS_DIR, "C2_final_folds", disease, source_name,
                        f"cf_{cf_count}")


def pairing_path(disease, source_name, cf_count, fold=None):
    """Saves each fold separately
    To ensure that we keep the results in case the job is killed."""
    
    d = fold_dir(disease, source_name, cf_count)
    
    # Save as parquet for less space and faster to read
    return os.path.join(d, "pairing.parquet" if fold is None
                        else f"pairing_fold_{fold}.parquet")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD BASE DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_c2_data(disease):
    """Read c2_data.csv and rename C0's columns to their disease-qualified names.

    The CF code addresses them as <disease>_prob / _pred / _true; c2_build_dataset
    writes them as plain prob / pred / true.
    """
    path = c2_data_path(disease)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Build it first:  python c2_build_dataset.py --disease {disease}")

    df = pd.read_csv(path)
    df = df.rename(columns={"prob": f"{disease}_prob",
                            "pred": f"{disease}_pred",
                            "true": f"{disease}_true"})

    # Rows with an uncertain (-1) CheXpert label have true = NaN and therefore no
    # correctness label. A NaN target is not a class -- drop them here rather than
    # letting each trainer discover it separately.
    n_before = len(df)
    df = df[df["correct"].notna()].reset_index(drop=True)
    if len(df) < n_before:
        print(f"  dropped {n_before - len(df):,} rows with undefined correctness")
    df["correct"] = df["correct"].astype(int)
    return df


def _relevant_cols(df, disease):
    """The attribute space, defined exactly as the CF matcher defines it."""
    meta = {f"{disease}_prob", f"{disease}_pred", f"{disease}_true",
            "correct", "path", "cam_path", "patient_id",
            *C0_DERIVED_COLS, *RSNA_META_COLS}
    return [c for c in df.columns
            if c not in meta and not c.startswith("emb_")]


# ══════════════════════════════════════════════════════════════════════════════
# K > 1 SUPPORT
# ══════════════════════════════════════════════════════════════════════════════
# cf_paths holds the K retrieved counterfactuals NEAREST FIRST, so the pairing at
# K is a strict prefix of the pairing at any larger K': same folds (seed 42), same
# scaler (fit on the same train rows and columns), same pools, and kneighbors
# returns its neighbours sorted by distance. That is what makes `derive_pairing`
# sound, and it is what makes K a free axis: pair ONCE at the largest K you intend
# to study (~11.5h per disease) and slice every smaller K out of it in seconds
# instead of re-running the search per K.


def _split_cells(series):
    return [str(cell).split("|") for cell in series]


def _per_cf_probs(cf_paths, prob_lut):
    """Pipe-separated per-CF C0 probabilities, in cf_paths order."""
    out = []
    for cell in cf_paths:
        parts = str(cell).split("|")
        missing = [p for p in parts if p not in prob_lut]
        if missing:
            raise KeyError(
                f"CF path has no C0 probability in c2_data.csv: {missing[0]!r}. "
                f"A generated (diffusion) CF is not a row of c2_data.csv and needs "
                f"its own probability table -- see DiffusionCFSource.")
        out.append("|".join(repr(float(prob_lut[p])) for p in parts))
    return out


def _mean_of_cf_probs(series):
    return np.array([np.mean([float(v) for v in str(cell).split("|")])
                     for cell in series])


def _truncate_cell(cell, k):
    return "|".join(str(cell).split("|")[:k])


def read_pairing(disease, source_name, cf_count, fold, prob_lut=None):
    """One fold's pairing, with cf_probs backfilled if the file predates it.

    Pairings written before 2026-09-09 stored only the mean cf_prob. Rather than
    invalidate them -- the cf_1 results for all five diseases were produced from
    those exact files -- the per-CF column is reconstructed from c2_data.csv,
    which is lossless for a KNN source because every CF is a real row there.
    """
    f = pd.read_parquet(pairing_path(disease, source_name, cf_count, fold))
    if "cf_probs" not in f.columns:
        if prob_lut is None:
            base = pd.read_csv(c2_data_path(disease), usecols=["path", "prob"])
            prob_lut = dict(zip(base["path"], base["prob"]))
        f["cf_probs"] = _per_cf_probs(f["cf_paths"], prob_lut)
    return f[PAIRING_COLS]


def available_ks(disease, source_name, n_folds=N_FOLDS):
    """Every K whose pairing is complete on disk, ascending."""
    root = os.path.dirname(fold_dir(disease, source_name, 1))
    if not os.path.isdir(root):
        return []
    ks = []
    for name in os.listdir(root):
        if not name.startswith("cf_"):
            continue
        try:
            k = int(name[3:])
        except ValueError:
            continue
        if all(os.path.exists(pairing_path(disease, source_name, k, i))
               for i in range(n_folds)):
            ks.append(k)
    return sorted(ks)


def derive_pairing(disease, source_name, k_to, k_from=None, n_folds=N_FOLDS,
                   overwrite=False):
    """Write the K=k_to pairing by truncating an existing larger-K pairing.

    Returns the output directory, or None if no usable source pairing exists.
    Costs seconds; the search it replaces costs ~11.5 hours per disease.
    """
    if k_from is None:
        bigger = [k for k in available_ks(disease, source_name, n_folds) if k > k_to]
        if not bigger:
            return None
        k_from = min(bigger)          # the tightest superset, cheapest to read
    if k_from <= k_to:
        raise ValueError(f"cannot derive K={k_to} from K={k_from}: "
                         f"the source pairing must hold MORE counterfactuals")

    out_dir = fold_dir(disease, source_name, k_to)
    os.makedirs(out_dir, exist_ok=True)
    if (all(os.path.exists(pairing_path(disease, source_name, k_to, i))
            for i in range(n_folds)) and not overwrite):
        print(f"  all {n_folds} folds already present for K={k_to}"
              f"  (--overwrite to rebuild)")
        return out_dir

    print(f"\n{'='*72}")
    print(f"  DERIVING FOLDS: {disease} | {source_name} | K={k_to} <- K={k_from}")
    print(f"{'='*72}")

    n_rows = 0
    for i in range(n_folds):
        f = read_pairing(disease, source_name, k_from, i)
        counts = np.array([len(c) for c in _split_cells(f["cf_paths"])])
        if counts.min() < k_to:
            raise ValueError(
                f"fold {i}: a row carries only {counts.min()} counterfactuals, "
                f"fewer than the K={k_to} being derived")
        f["cf_paths"] = [_truncate_cell(c, k_to) for c in f["cf_paths"]]
        f["cf_probs"] = [_truncate_cell(c, k_to) for c in f["cf_probs"]]
        f["cf_prob"] = _mean_of_cf_probs(f["cf_probs"])
        # Write-then-rename: the attribute and CNN jobs start in parallel off the
        # same folds dependency, and either may be the one that triggers a
        # derivation. A half-written parquet read by the other is a far worse
        # failure than doing the truncation twice.
        final = pairing_path(disease, source_name, k_to, i)
        tmp = f"{final}.tmp{os.getpid()}"
        f[PAIRING_COLS].to_parquet(tmp, index=False)
        os.replace(tmp, final)
        n_rows += len(f)
        print(f"  fold {i}: {len(f):,} rows")

    size_mb = sum(os.path.getsize(pairing_path(disease, source_name, k_to, i))
                  for i in range(n_folds)) / 1e6
    src_manifest = os.path.join(fold_dir(disease, source_name, k_from),
                                "manifest.json")
    desc = source_name
    if os.path.exists(src_manifest):
        with open(src_manifest) as fh:
            desc = json.load(fh).get("cf_source_description", source_name)

    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({"disease": disease, "cf_source": source_name,
                   "cf_source_description": desc,
                   "cf_count": k_to, "n_folds": n_folds, "seed": RANDOM_SEED,
                   "derived_from_k": k_from,
                   "n_pairing_rows": int(n_rows),
                   "source_csv": c2_data_path(disease),
                   "pairing_mb": round(size_mb, 1)}, fh, indent=2)

    print(f"\n  wrote {n_folds} pairing files to {out_dir} "
          f"({n_rows:,} rows, {size_mb:.1f} MB)")
    return out_dir


def _manifest_distance(disease, source_name, k):
    """The metric the pairing was built with, read back from its manifest."""
    p = os.path.join(fold_dir(disease, source_name, k), "manifest.json")
    if os.path.exists(p):
        with open(p) as fh:
            desc = json.load(fh).get("cf_source_description", "")
        for m in ("l1", "l2", "cosine"):
            if f"distance={m}" in desc:
                return m
    return "l1"


class _AttrOnlyContext:
    """The scaled attribute space alone, for checks that never touch embeddings.

    A full FoldContext also materialises the 1,024 embedding columns and a copy
    of the frame holding them, about 2 GB per fold. The tie check compares
    distances in the attribute space only, so reading c2_data.csv without the
    emb_* columns cuts that to a few hundred MB and keeps the check runnable on
    a login node.
    """

    def __init__(self, disease, pair_train):
        path = c2_data_path(disease)
        head = pd.read_csv(path, nrows=0)
        keep = [c for c in head.columns if not c.startswith("emb_")]
        df = pd.read_csv(path, usecols=keep)
        df = df.rename(columns={"prob": f"{disease}_prob",
                                "pred": f"{disease}_pred",
                                "true": f"{disease}_true"})
        df = df[df["correct"].notna()].reset_index(drop=True)
        df = add_clinical_ratios(df)
        rel = _relevant_cols(df, disease)
        df[rel] = df[rel].fillna(0)

        self.row_of = {p: i for i, p in enumerate(df["path"])}
        attr = df[rel].values.astype(float)
        sc = StandardScaler()
        sc.fit(attr[[self.row_of[p] for p in pair_train["path"].values]])
        self.attr_scaled_all = sc.transform(attr)


def _is_tie(ctx, query, cf_a, cf_b, distance):
    """Are two counterfactuals exactly equidistant from the query?"""
    q, a, b = (ctx.attr_scaled_all[ctx.row_of[x]] for x in (query, cf_a, cf_b))
    if distance == "l1":
        da, db = np.abs(q - a).sum(), np.abs(q - b).sum()
    elif distance == "l2":
        da, db = np.linalg.norm(q - a), np.linalg.norm(q - b)
    else:
        cos = lambda u, v: 1 - u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
        da, db = cos(q, a), cos(q, b)
    return abs(float(da) - float(db)) <= 1e-9 * max(1.0, abs(float(da)))


def check_nesting(disease, source_name, k_small, k_large, n_folds=N_FOLDS,
                  distance=None):
    """Assert the K=k_small pairing on disk is the prefix of the K=k_large one.

    Run this once after building a large-K pairing: it is the empirical proof of
    the prefix property the derivation relies on, and it also proves the older
    cf_1 results and the new large-K results sit on the same folds. A mismatch
    means something moved -- the fold seed, the feature spec, or c2_data.csv.

    Exact distance ties are the one legitimate exception, and they do occur: in
    461 standardised dimensions two training images can sit at identically the
    same L1 distance from a query, and scikit-learn's neighbour search breaks
    that tie differently depending on n_neighbors, because n_neighbors is one of
    the inputs to its algorithm choice. Measured rate is 1-10 rows in ~90,000
    (<= 0.012%). Rather than wave that through as "close enough", every differing
    row is checked individually: a tie passes, anything else fails.
    """
    if distance is None:
        distance = _manifest_distance(disease, source_name, k_large)
    real, ties = [], 0

    for i in range(n_folds):
        a = read_pairing(disease, source_name, k_small, i)
        b = read_pairing(disease, source_name, k_large, i)
        a = a.set_index(["split", "path"])
        b = b.set_index(["split", "path"])
        if not a.index.equals(b.index):
            # Row ORDER carries no meaning, but membership does: a differing set
            # means the two pairings were built on different folds, and then no
            # comparison across K is honest.
            missing = a.index.difference(b.index)
            extra = b.index.difference(a.index)
            if len(missing) or len(extra):
                raise AssertionError(
                    f"fold {i}: K={k_small} and K={k_large} do not cover the same "
                    f"rows ({len(missing)} only in K={k_small}, {len(extra)} only "
                    f"in K={k_large}) -- the folds themselves differ")
            b = b.loc[a.index]

        pre = np.array([_truncate_cell(c, k_small) for c in b["cf_paths"]])
        diff = np.where(pre != a["cf_paths"].values)[0]
        if len(diff) == 0:
            print(f"  fold {i}: identical ({len(a):,} rows)")
            continue

        # Built per fold and dropped afterwards: the scaler is fold-specific, and
        # holding five at once is what pushed this over a login node's cap.
        tr = read_pairing(disease, source_name, k_small, i)
        ctx = _AttrOnlyContext(disease, tr[tr["split"] == "train"])

        fold_ties, fold_real = 0, []
        for j in diff:
            _, q = a.index[j]
            # Compare only the first counterfactual: if the nearest differs the
            # rest follow, and the nearest is the one training uses.
            ca = str(a["cf_paths"].values[j]).split("|")[0]
            cb = str(pre[j]).split("|")[0]
            if ca != cb and _is_tie(ctx, q, ca, cb, distance):
                fold_ties += 1
            else:
                fold_real.append((q, ca, cb))
        ties += fold_ties
        real += fold_real
        print(f"  fold {i}: {len(diff):,} / {len(a):,} rows differ "
              f"({100*len(diff)/len(a):.3f}%) -- {fold_ties} exact distance "
              f"tie{'s' if fold_ties != 1 else ''}, {len(fold_real)} real")

    if real:
        q, ca, cb = real[0]
        raise AssertionError(
            f"K={k_small} is not a prefix of K={k_large}: {len(real)} rows differ "
            f"at a DIFFERENT distance, not a tie. First: {q!r} matched {ca!r} at "
            f"K={k_small} and {cb!r} at K={k_large}. The two pairings were built "
            f"against different data or a different feature spec.")
    if ties:
        print(f"  OK: K={k_small} is the K={k_large} prefix except for {ties} "
              f"exact distance tie{'s' if ties != 1 else ''}, where two "
              f"counterfactuals are equidistant and either is equally valid")
    else:
        print(f"  OK: K={k_small} is exactly the K={k_large} prefix on all folds")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════════════════════════

def build_folds(disease, cf_source, cf_count, n_folds=N_FOLDS, seed=RANDOM_SEED,
                overwrite=False):
    """Compute every fold's CF pairing and cache it. Returns the pairing path."""
    out_dir = fold_dir(disease, cf_source.name, cf_count)
    os.makedirs(out_dir, exist_ok=True)

    done = [i for i in range(n_folds)
            if os.path.exists(pairing_path(disease, cf_source.name, cf_count, i))]
    if len(done) == n_folds and not overwrite:
        print(f"  all {n_folds} folds already paired in {out_dir}"
              f"  (--overwrite to rebuild)")
        return out_dir
    if done and not overwrite:
        print(f"  resuming: folds {done} already paired, skipping them")

    print(f"\n{'='*72}")
    print(f"  BUILDING FOLDS: {disease} | {cf_source.describe()} | K={cf_count}")
    print(f"{'='*72}")

    df = load_c2_data(disease)
    print(f"  {len(df):,} samples | correct {int((df['correct']==1).sum()):,} "
          f"| incorrect {int((df['correct']==0).sum()):,}")

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = df["correct"].values
    groups = df["patient_id"].values

    # Per-CF C0 probabilities. cf_prob alone is the mean over the K, which cannot
    # be un-averaged: deriving a smaller K from this pairing, and the CNN's scalar
    # head, both need the individual values. Storing them here also removes the
    # trainers' dependency on re-reading c2_data.csv to recover them.
    prob_lut = dict(zip(df["path"], df[f"{disease}_prob"]))

    fold_meta = []
    for i, (tr_idx, te_idx) in enumerate(skf.split(df, y, groups=groups)):
        fold_out = pairing_path(disease, cf_source.name, cf_count, i)
        if os.path.exists(fold_out) and not overwrite:
            continue
        t0 = time.time()
        print(f"\n  fold {i}: train {len(tr_idx):,} | test {len(te_idx):,}")

        fold_train = df.iloc[tr_idx].reset_index(drop=True)
        fold_test = df.iloc[te_idx].reset_index(drop=True)

        overlap = set(fold_train["patient_id"]) & set(fold_test["patient_id"])
        assert not overlap, (
            f"fold {i}: {len(overlap)} patients on both sides of the split -- "
            f"StratifiedGroupKFold grouping is broken")

        print(f"    pairing counterfactuals ({cf_source.describe()})...")
        train_out, test_out = cf_source.pair(fold_train, fold_test, disease, cf_count)

        # Keep the pairing; discard the ~3,400-column materialisation.
        parts = []
        for split, frame in (("train", train_out), ("test", test_out)):
            part = frame[["path", "cf_paths", "cf_prob"]].copy()
            part["cf_probs"] = _per_cf_probs(part["cf_paths"], prob_lut)
            # The source averaged the probabilities of the rows it retrieved; the
            # lookup averages the probabilities of the paths it wrote down. They
            # agree only if the path bookkeeping is right, which is exactly the
            # thing that silently broke the last time the feature spec changed.
            drift = float(np.abs(_mean_of_cf_probs(part["cf_probs"])
                                 - part["cf_prob"].values).max())
            assert drift < 1e-6, (
                f"fold {i} {split}: cf_paths disagree with cf_prob by {drift:.2e} "
                f"-- the retrieved rows and the recorded paths are not the same set")
            part.insert(0, "split", split)
            part.insert(0, "fold", i)
            parts.append(part)
        pd.concat(parts, ignore_index=True)[PAIRING_COLS].to_parquet(
            fold_out, index=False)

        mins = (time.time() - t0) / 60
        print(f"    paired {len(train_out):,} train + {len(test_out):,} test "
              f"[{mins:.1f} min]")
        fold_meta.append({"fold": i, "n_train": int(len(train_out)),
                          "n_test": int(len(test_out)),
                          "build_minutes": round(mins, 2)})

    size_mb = sum(os.path.getsize(pairing_path(disease, cf_source.name, cf_count, i))
                  for i in range(n_folds)) / 1e6
    n_rows = int(len(df)) * n_folds

    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({"disease": disease, "cf_source": cf_source.name,
                   "cf_source_description": cf_source.describe(),
                   "cf_count": cf_count, "n_folds": n_folds, "seed": seed,
                   "derived_from_k": None,
                   "n_samples": int(len(df)), "n_pairing_rows": n_rows,
                   "source_csv": c2_data_path(disease),
                   "pairing_mb": round(size_mb, 1), "folds": fold_meta}, fh, indent=2)

    print(f"\n  wrote {n_folds} pairing files to {out_dir}  "
          f"({n_rows:,} rows total, {size_mb:.1f} MB)")
    return out_dir


# ══════════════════════════════════════════════════════════════════════════════
# MATERIALIZE
# ══════════════════════════════════════════════════════════════════════════════

def _row_ids(paths, row_of, what):
    """Map paths to row positions in c2_data.csv, naming the offender on a miss."""
    try:
        return np.fromiter((row_of[p] for p in paths), dtype=np.int64,
                           count=len(paths))
    except KeyError as exc:
        raise ValueError(
            f"{what} path absent from c2_data.csv: {exc.args[0]!r}") from None


def _cf_row_ids(cf_paths, row_of):
    """Flat CF row positions, the query each belongs to, and K per query."""
    cells = [str(c).split("|") for c in cf_paths]
    k_per_row = np.array([len(c) for c in cells], dtype=np.int64)
    flat = [p for cell in cells for p in cell]
    return (_row_ids(flat, row_of, "CF"),
            np.repeat(np.arange(len(cells)), k_per_row),
            k_per_row)


def _mean_over_cfs(mats, cf_rows, row_ids, k_per_row, n_queries,
                   max_gather=250_000):
    """Per-query mean of each matrix over that query's counterfactuals.

    Chunked on purpose. A fold has ~73k train queries, so at K=16 the CF gather
    is ~1.2M rows; pulling all of them from the 1,478-column attribute+embedding
    space at once materialises ~14 GB of float64 and pushes the job past its
    memory reservation. Accumulating `max_gather` rows at a time bounds the peak
    at a few hundred MB and returns bit-identical means.
    """
    outs = [np.zeros((n_queries, m.shape[1]), dtype=np.float64) for m in mats]
    uniform_k = int(k_per_row[0]) if len(k_per_row) else 0
    constant_k = bool(len(k_per_row)) and bool((k_per_row == uniform_k).all())

    if constant_k:
        # Every query carries the same K, so the flat array reshapes cleanly and
        # the mean is a vectorised reduction rather than a scatter-add.
        rows_per_chunk = max(1, max_gather // max(uniform_k, 1))
        for s0 in range(0, n_queries, rows_per_chunk):
            s1 = min(s0 + rows_per_chunk, n_queries)
            idx = cf_rows[s0 * uniform_k:s1 * uniform_k]
            for out, m in zip(outs, mats):
                out[s0:s1] = m[idx].reshape(s1 - s0, uniform_k, -1).mean(axis=1)
        return outs

    # Ragged K (a source that returns fewer CFs for some queries): scatter-add.
    counts = np.bincount(row_ids, minlength=n_queries).astype(float)
    for s0 in range(0, len(cf_rows), max_gather):
        sl = slice(s0, s0 + max_gather)
        for out, m in zip(outs, mats):
            np.add.at(out, row_ids[sl], m[cf_rows[sl]])
    return [out / counts[:, None] for out in outs]


class FoldContext:
    """Everything a fold's feature frames are built from, computed once.

    The scaler, the dense attribute/embedding matrices and the path index do not
    depend on WHICH counterfactual a query is paired with, only on the fold. Score
    ensembling asks for K frames per fold that differ solely in that pairing, so
    hoisting the shared work out of the frame builder turns K materialisations
    into K gathers rather than K full rebuilds.
    """

    def __init__(self, base, pair_train, disease):
        base = add_clinical_ratios(base.copy())
        self.disease = disease
        self.rel = _relevant_cols(base, disease)
        self.emb = [c for c in base.columns if c.startswith("emb_")]
        base[self.rel] = base[self.rel].fillna(0)

        self.lut = base.set_index("path")
        assert self.lut.index.is_unique, "c2_data.csv has duplicate paths"
        self.row_of = {p: i for i, p in enumerate(self.lut.index)}

        # Dense views. Every lookup below is a row gather, and doing those through
        # pandas .loc on a 1,478-column frame costs a full copy each time.
        self.attr_all = self.lut[self.rel].values.astype(float)
        self.emb_all = self.lut[self.emb].values.astype(float)

        # Scaler fit on the fold's TRAIN attributes, exactly as the matcher did.
        self.scaler = StandardScaler()
        self.scaler.fit(self.attr_all[
            _row_ids(pair_train["path"].values, self.row_of, "query")])
        self.attr_scaled_all = self.scaler.transform(self.attr_all)


def materialize(ctx, pair):
    """The feature frame for one pairing: query rows joined to their CF columns.

    Reproduces exactly what the matcher produced in-line:
      * attributes/embeddings joined from c2_data.csv on `path`
      * clinical ratios added, attribute NaNs filled with 0 -- in that order,
        matching the matcher, since the ratios are computed before the fill
      * delta_<attr> = scaled(query) - scaled(CF), scaler fit on TRAIN only
      * cf_attr_/cf_emb_/delta_prob dereferenced from the CF paths

    A pairing carrying several CFs per row still yields ONE row per query, with
    the CF columns averaged. Nothing calls it that way any more -- both trainers
    ensemble scores rather than features, and feed it one CF at a time via
    `slot_pairing` -- but the averaging is kept because it is what makes a
    single-CF pairing a special case rather than a separate code path.
    """
    rel, emb = ctx.rel, ctx.emb
    q_rows = _row_ids(pair["path"].values, ctx.row_of, "query")
    frame = ctx.lut.iloc[q_rows].reset_index()
    frame["cf_paths"] = pair["cf_paths"].values
    frame["cf_prob"] = pair["cf_prob"].values

    q_scaled = ctx.attr_scaled_all[q_rows]

    cf_rows, row_ids, k_per_row = _cf_row_ids(pair["cf_paths"].values, ctx.row_of)
    cf_attr_mean, cf_emb_mean, cf_scaled_mean = _mean_over_cfs(
        (ctx.attr_all, ctx.emb_all, ctx.attr_scaled_all),
        cf_rows, row_ids, k_per_row, len(pair))

    new = {f"delta_{c}": q_scaled[:, j] - cf_scaled_mean[:, j]
           for j, c in enumerate(rel)}
    new.update({f"cf_attr_{c}": cf_attr_mean[:, j] for j, c in enumerate(rel)})
    new.update({f"cf_emb_{c}": cf_emb_mean[:, j] for j, c in enumerate(emb)})
    new["delta_prob"] = (frame[f"{ctx.disease}_prob"].values
                         - frame["cf_prob"].values)

    return pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)


def materialize_fold(base, pair_train, pair_test, disease, cf_source=None):
    """Both sides of one fold. Kept for callers that want the whole fold at once."""
    ctx = FoldContext(base, pair_train, disease)
    return materialize(ctx, pair_train), materialize(ctx, pair_test)


def slot_pairing(pair, j):
    """The pairing restricted to each query's j-th nearest counterfactual.

    This is the unit of score ensembling: query x paired with CF_j is one
    prediction, and the K predictions for x are averaged afterwards. Slot j is a
    K=1 pairing in every respect, so it flows through `materialize` unchanged.
    """
    out = pair.copy()
    cells = [str(c).split("|") for c in pair["cf_paths"]]
    if any(len(c) <= j for c in cells):
        raise IndexError(
            f"slot {j} requested but a row carries only "
            f"{min(len(c) for c in cells)} counterfactuals")
    out["cf_paths"] = [c[j] for c in cells]
    out["cf_probs"] = [str(c).split("|")[j] for c in pair["cf_probs"]]
    out["cf_prob"] = [float(v) for v in out["cf_probs"]]
    return out


class MaterializedFold:
    """One CV fold, with its test side addressable per counterfactual slot.

    Score ensembling applies ONE model to each (query, CF_j) pair and averages
    the K resulting probabilities, so the test side is not a frame but K frames
    that share their query rows and differ only in which counterfactual they were
    paired with. They are built on first use and then cached: rebuilding one is
    ~15 s, and the trainer revisits every slot once per (config, model).

    `train` is slot 0 -- training pairs each query with its nearest
    counterfactual only, which is exactly what the CNN ladder does.
    """

    def __init__(self, index, ctx, pair_train, pair_test):
        self.index = index
        self.ctx = ctx
        self._pair_train = pair_train
        self._pair_test = pair_test
        self._train = None
        self._slots = {}
        counts = [len(str(c).split("|")) for c in pair_test["cf_paths"]]
        self.n_slots = min(counts)
        if max(counts) != self.n_slots:
            raise ValueError(
                f"fold {index}: test rows carry between {self.n_slots} and "
                f"{max(counts)} counterfactuals. Score ensembling averages a "
                f"fixed number of predictions per query, so a ragged pairing "
                f"would weight queries unequally.")

    @property
    def train(self):
        if self._train is None:
            self._train = materialize(self.ctx, slot_pairing(self._pair_train, 0))
        return self._train

    def test_slot(self, j):
        if j not in self._slots:
            self._slots[j] = materialize(self.ctx, slot_pairing(self._pair_test, j))
        return self._slots[j]

    @property
    def test_paths(self):
        return self._pair_test["path"].values

    @property
    def test_cf_paths(self):
        """All K counterfactuals per test query, as the pipe-separated cell.

        The slot frames each hold one; this is what goes into the predictions
        CSV so a row can be traced back to the full set it was scored against.
        """
        return self._pair_test["cf_paths"].values

    def __len__(self):
        return len(self._pair_test)


# ══════════════════════════════════════════════════════════════════════════════
# READ
# ══════════════════════════════════════════════════════════════════════════════

def load_folds(disease, cf_source, cf_count, n_folds=N_FOLDS, materialize=True,
               columns=None, build_if_missing=True):
    """Yield one fold at a time.

    materialize=True yields a MaterializedFold: the full attribute/delta/CF
    feature space, with the test side addressable per counterfactual slot.
    materialize=False yields (fold_idx, train_df, test_df) carrying just the
    pairing joined to c2_data.csv's metadata columns -- what the CNN needs, and a
    few MB rather than 2.5 GB per fold. `columns` restricts that metadata.
    """
    missing = [i for i in range(n_folds)
               if not os.path.exists(pairing_path(disease, cf_source.name, cf_count, i))]
    if missing:
        # A larger-K pairing already contains this one: cf_paths is nearest-first,
        # so slicing it is exact and takes seconds. Prefer that over an ~11.5h
        # re-run of the neighbour search.
        if derive_pairing(disease, cf_source.name, cf_count, n_folds=n_folds):
            missing = []
    if missing:
        if not build_if_missing:
            raise FileNotFoundError(
                f"folds {missing} not paired under "
                f"{fold_dir(disease, cf_source.name, cf_count)}\n"
                f"Build them:  python c2_folds.py --disease {disease} "
                f"--cf-count {cf_count}")
        build_folds(disease, cf_source, cf_count, n_folds=n_folds)

    base = load_c2_data(disease)
    if not materialize:
        need = set(columns or []) | {"path"}
        base = base[[c for c in base.columns if c in need]]

    for i in range(n_folds):
        f = read_pairing(disease, cf_source.name, cf_count, i)
        tr = f[f["split"] == "train"]
        te = f[f["split"] == "test"]
        if materialize:
            yield MaterializedFold(i, FoldContext(base, tr, disease), tr, te)
        else:
            # cf_probs comes along: the CNN's scalar head needs the K individual
            # probabilities, not their mean.
            lut = base.set_index("path")
            yield (i,
                   lut.loc[tr["path"].values].reset_index().assign(
                       cf_paths=tr["cf_paths"].values, cf_prob=tr["cf_prob"].values,
                       cf_probs=tr["cf_probs"].values),
                   lut.loc[te["path"].values].reset_index().assign(
                       cf_paths=te["cf_paths"].values, cf_prob=te["cf_prob"].values,
                       cf_probs=te["cf_probs"].values))


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def add_cf_args(p):
    """Shared by both trainers so a fold set is named identically everywhere."""
    p.add_argument("--cf-source", default="knn", choices=CF_SOURCE_KINDS)
    p.add_argument("--cf-strategy", default="correct_cf", choices=KNN_STRATEGIES,
                   help="knn only: how the counterfactual pool is routed")
    p.add_argument("--cf-count", type=int, default=1, help="K counterfactuals per query")
    p.add_argument("--distance", default="l1", choices=["l1", "l2", "cosine"])
    p.add_argument("--k-offset", type=int, default=0,
                   help="knn/further only: take the (k_offset + k)-th neighbour")
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    return p


def cf_source_from_args(args):
    return build_cf_source(args.cf_source, strategy=args.cf_strategy,
                           distance=args.distance, k_offset=args.k_offset)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute the pairing even if it is cached")
    ap.add_argument("--derive-from", type=int, default=None, metavar="K",
                    help="slice --cf-count out of an existing larger-K pairing "
                         "instead of re-running the neighbour search (seconds "
                         "rather than ~11.5h). Omit the value and any larger K "
                         "on disk is used automatically.")
    ap.add_argument("--derive", action="store_true",
                    help="derive from whichever larger K is already on disk")
    ap.add_argument("--check-nesting", type=int, default=None, metavar="K_SMALL",
                    help="verify the K_SMALL pairing on disk is exactly the "
                         "prefix of the --cf-count one, then exit")
    add_cf_args(ap)
    args = ap.parse_args()

    src = cf_source_from_args(args)

    if args.check_nesting is not None:
        check_nesting(args.disease, src.name, args.check_nesting, args.cf_count,
                      n_folds=args.n_folds)
    elif args.derive or args.derive_from is not None:
        got = derive_pairing(args.disease, src.name, args.cf_count,
                             k_from=args.derive_from, n_folds=args.n_folds,
                             overwrite=args.overwrite)
        if got is None:
            raise SystemExit(
                f"nothing to derive K={args.cf_count} from. "
                f"Complete pairings on disk: "
                f"{available_ks(args.disease, src.name, args.n_folds) or 'none'}")
    else:
        build_folds(args.disease, src, args.cf_count, n_folds=args.n_folds,
                    overwrite=args.overwrite)
