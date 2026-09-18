# c2_train_attributes.py
"""
Stage 5b: attribute-based C2 models on the cached folds.

Trains the full config ladder with LR / RF / MLP on identical folds, so every
model has its own baseline measured on exactly the same rows.

    B1    prob                         C0 confidence alone -- the baseline to beat
    B1.2  H(prob)                      binary entropy of the same number
    B2    attrs                        image attributes, no C0 signal
    B3    emb                          C0 embedding
    B4    prob + attrs
    B5    prob + emb
    M1    delta                        query-minus-CF attribute deltas alone
    M2    prob + delta
    M3    prob + delta + attrs
    M4    prob + delta + emb
    M5    prob + delta + attrs + emb
    M6    prob + delta + attrs + cf_prob
    MCF1  prob + cf_prob
    MCF2  prob + cf_prob + attrs + cf_attrs
    MCF3  MCF2 + delta
    MCF4  MCF3 + emb + cf_emb
    MCF5  MCF4 + delta_emb

USAGE:

    python c2_train_attributes.py --disease effusion --cf-source knn --cf-count 1

Folds come from c2_folds.py and are built on first use if absent. Nothing here
recomputes the CF pairing.

More than one counterfactual per query
--------------------------------------
K is combined by ENSEMBLING SCORES, never by averaging features -- the same
protocol the CNN ladder uses, so the two are comparable at every K:

    train   each query is paired with its NEAREST counterfactual only. One row
            per query, one model fit. Training cost is flat in K.
    test    the fitted model is applied K times, once per (query, CF_j) pair,
            and the K predicted probabilities are averaged into one score per
            query. K predictions in, one AUC out, on the same query rows.

Feature-space averaging was the earlier design and it is gone. Averaging K
counterfactuals into one mean vector destroys what K is meant to add: a mean
attribute vector over ten neighbours is a smoother, more central point than any
real counterfactual, so delta_* shrinks toward the pool mean as K grows and the
model sees less disagreement rather than more. Ensembling keeps every pair intact
and combines ten opinions instead of ten inputs.

The feature space is therefore identical at every K -- one CF per row, always --
and no config changes width. What changes is how many predictions get averaged.

Configs with no CF block (B1-B5) score identically on every slot, so they are
evaluated once and reported with k_used=1 in cv_summary.csv. That is a claim, not
a shortcut: the baselines are K-invariant, which is what lets a difference across
K be read as coming from the counterfactual side alone.

Results land in .../<source>/cf_<K>/attributes, so different K never overwrite
each other.
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import (                                          # noqa: E402
    RESULTS_DIR, RANDOM_SEED, N_FOLDS,
    load_folds, add_cf_args, cf_source_from_args, fold_dir,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code", "c2"))
from c2_feature_spec import C0_DERIVED_COLS, RSNA_META_COLS     # noqa: E402

CONFIGS = ["B1", "B1.2", "B2", "B3", "B4", "B5",
           "M1", "M2", "M3", "M4", "M5", "M6",
           "MCF1", "MCF2", "MCF3", "MCF4", "MCF5"]
MODEL_TYPES = ["LR", "RF", "MLP"]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BLOCKS
# ══════════════════════════════════════════════════════════════════════════════
# A config is a list of blocks, concatenated in the order written here. Spelling
# it out (rather than hstack-ing inline) buys two things: the column order of
# every config is one readable line, and CF_DEPENDENT below falls out of the
# definitions instead of being a second list that can drift from them.

CONFIG_BLOCKS = {
    "B1":   ("prob",),
    "B1.2": ("ent",),
    "B2":   ("attr",),
    "B3":   ("emb",),
    "B4":   ("prob", "attr"),
    "B5":   ("prob", "emb"),
    "M1":   ("delta",),
    "M2":   ("prob", "delta"),
    "M3":   ("prob", "delta", "attr"),
    "M4":   ("prob", "delta", "emb"),
    "M5":   ("prob", "delta", "attr", "emb"),
    "M6":   ("prob", "delta", "attr", "cf_prob"),
    "MCF1": ("prob", "cf_prob"),
    "MCF2": ("prob", "cf_prob", "attr", "cf_attr"),
    "MCF3": ("prob", "cf_prob", "attr", "cf_attr", "delta"),
    "MCF4": ("prob", "cf_prob", "attr", "cf_attr", "delta", "emb", "cf_emb"),
    "MCF5": ("prob", "cf_prob", "attr", "cf_attr", "delta", "emb", "cf_emb",
             "delta_emb"),
}

# Blocks whose values depend on WHICH counterfactual the query was paired with.
CF_BLOCKS = {"delta", "cf_attr", "cf_emb", "cf_prob", "delta_emb"}
CF_DEPENDENT = {c for c, b in CONFIG_BLOCKS.items() if CF_BLOCKS & set(b)}


def feature_columns(df, disease):
    """The column list behind each block, derived once per fold from the train frame.

    The attribute space is whatever is left after removing the named metadata and
    the prefixed groups. That makes `meta_cols` load-bearing: a column that
    belongs to C0 but is not named here silently becomes an attribute and inflates
    every attribute config. C0_DERIVED_COLS exists because `margin` and the seven
    SDN prob_* columns did exactly that.

    Derived from the TRAIN frame and then applied by name to every test slot, so
    a slot cannot silently reorder or drop a column.
    """
    meta_cols = {f"{disease}_prob", f"{disease}_pred", f"{disease}_true",
                 "correct", "path", "cam_path", "patient_id",
                 "cf_prob", "cf_paths", "cf_probs", "delta_prob",
                 "label_raw", "certain",
                 *C0_DERIVED_COLS, *RSNA_META_COLS}

    cols = {
        "delta":   [c for c in df.columns
                    if c.startswith("delta_") and c != "delta_prob"],
        "emb":     [c for c in df.columns if c.startswith("emb_")],
        "cf_attr": [c for c in df.columns if c.startswith("cf_attr_")],
        "cf_emb":  [c for c in df.columns if c.startswith("cf_emb_")],
        "attr":    [c for c in df.columns
                    if c not in meta_cols
                    and not c.startswith("delta_")
                    and not c.startswith("emb_")
                    and not c.startswith("cf_attr_")
                    and not c.startswith("cf_emb_")],
    }

    non_numeric = [c for c in cols["attr"]
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"non-numeric columns reached the attribute space: {non_numeric[:5]}\n"
            f"They must be added to meta_cols -- a string column crashes "
            f"StandardScaler the way cam_path did.")
    return cols


def _entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return (-p * np.log(p) - (1 - p) * np.log(1 - p)).reshape(-1, 1)


def block_matrix(df, block, cols, disease):
    """One feature block as a float64 matrix."""
    if block == "prob":
        return df[[f"{disease}_prob"]].values.astype(np.float64)
    if block == "ent":
        return _entropy(df[f"{disease}_prob"].values.astype(np.float64))
    if block == "cf_prob":
        return df[["cf_prob"]].values.astype(np.float64)
    if block == "delta_emb":
        # derived, not stored: emb and cf_emb are aligned by construction
        return (df[cols["emb"]].values.astype(np.float64)
                - df[cols["cf_emb"]].values.astype(np.float64))
    return df[cols[block]].values.astype(np.float64)


def config_matrix(df, cfg, cols, disease):
    """The design matrix for one config, built one config at a time.

    Building all seventeen at once for both sides of a fold is ~18 GB, and with
    K test slots it would be ~18 GB plus K times the test half. One at a time
    keeps the peak at the widest single config (MCF5, ~4,460 columns).
    """
    blocks = [block_matrix(df, b, cols, disease) for b in CONFIG_BLOCKS[cfg]]
    return np.hstack(blocks) if len(blocks) > 1 else blocks[0]


def config_available(cfg, cols):
    """MCF* need the CF's own attributes and embeddings to exist."""
    need = set(CONFIG_BLOCKS[cfg])
    if "cf_attr" in need and not cols["cf_attr"]:
        return False
    if ("cf_emb" in need or "delta_emb" in need) and not cols["cf_emb"]:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

def fit_model(X_train, y_train, model_type="LR"):
    """Fit one model. Returns (model, scaler); scaler is None for RF."""
    if model_type == "LR":
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=5000, class_weight="balanced",
                                   random_state=RANDOM_SEED)
        model.fit(scaler.fit_transform(X_train), y_train)
        return model, scaler

    if model_type == "RF":
        model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        return model, None

    if model_type == "MLP":
        scaler = StandardScaler()
        model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500,
                              early_stopping=True, validation_fraction=0.05,
                              random_state=RANDOM_SEED)
        model.fit(scaler.fit_transform(X_train.astype(np.float32)), y_train)
        return model, scaler

    raise ValueError(f"unknown model type: {model_type}")


def predict_model(model, scaler, X, model_type):
    if scaler is None:
        return model.predict_proba(X)[:, 1]
    if model_type == "MLP":
        X = X.astype(np.float32)
    return model.predict_proba(scaler.transform(X))[:, 1]


def score_cf_anchor(train_df, test_slots):
    """Baseline: score a (query, CF) pair by the mean correctness of training rows
    that retrieved the same counterfactual. Returns the per-slot matrix.

    Measures how much of the signal is carried by WHICH CF was picked, before any
    feature is looked at. Returned per slot, and averaged by the caller, so it is
    ensembled exactly like every model above and stays comparable as K moves
    rather than quietly changing definition.
    """
    anchor = {}
    for a, corr in zip(train_df["cf_paths"], train_df["correct"]):
        anchor.setdefault(str(a), []).append(corr)
    anchor = {a: float(np.mean(v)) for a, v in anchor.items()}
    return np.stack([np.array([anchor.get(str(a), 0.0) for a in slot["cf_paths"]])
                     for slot in test_slots])


# ══════════════════════════════════════════════════════════════════════════════
# CV
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    src = cf_source_from_args(args)
    configs_wanted = ([c.strip() for c in args.configs.split(",")]
                      if args.configs else CONFIGS)
    unknown = [c for c in configs_wanted if c not in CONFIG_BLOCKS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; expected {CONFIGS}")
    models_wanted = [m.strip() for m in args.models.split(",")]

    out_dir = os.path.join(RESULTS_DIR, "C2_final_results", args.disease,
                           src.name, f"cf_{args.cf_count}", "attributes")
    os.makedirs(out_dir, exist_ok=True)

    # A finished run is ~4.75h and is deterministic (seed 42), so repeating it
    # only overwrites identical numbers. Guard it: a resubmitted chain, or a
    # job whose disease was misrouted, must not silently redo hours of work.
    if os.path.exists(os.path.join(out_dir, "cv_summary.csv")) and not args.overwrite:
        print(f"  cv_summary.csv already present in {out_dir}"
              f"  (--overwrite to rebuild)")
        return

    print(f"\n{'='*72}")
    print(f"  ATTRIBUTE MODELS: {args.disease} | {src.describe()} | K={args.cf_count}")
    print(f"{'='*72}")
    print(f"  folds:   {fold_dir(args.disease, src.name, args.cf_count)}")
    print(f"  results: {out_dir}")
    print(f"  configs: {', '.join(configs_wanted)}")
    print(f"  models:  {', '.join(models_wanted)}")
    print(f"  K combination: train on the nearest CF, average the K test "
          f"probabilities")

    results = {m: {c: [] for c in configs_wanted} for m in models_wanted}
    results["cf_anchor"] = []
    dims_seen = {}

    for fold in load_folds(args.disease, src, args.cf_count, n_folds=args.n_folds):
        train_df = fold.train
        n_slots = fold.n_slots
        print(f"\n{'-'*72}\n  FOLD {fold.index + 1}/{args.n_folds}"
              f"  (train {len(train_df):,} | test {len(fold):,} x {n_slots} CF)"
              f"\n{'-'*72}")

        y_train = train_df["correct"].values
        cols = feature_columns(train_df, args.disease)
        print(f"  feature groups: attrs={len(cols['attr'])} "
              f"delta={len(cols['delta'])} emb={len(cols['emb'])} "
              f"cf_attr={len(cols['cf_attr'])} cf_emb={len(cols['cf_emb'])}")

        # Build every test slot up front: each is revisited once per (config,
        # model), and rebuilding one costs ~15s against ~0.5 GB to keep it.
        slots = [fold.test_slot(j) for j in range(n_slots)]
        y_test = slots[0]["correct"].values
        for j, sl in enumerate(slots[1:], 1):
            assert (sl["path"].values == slots[0]["path"].values).all(), (
                f"fold {fold.index}: slot {j} has different query rows than slot 0 "
                f"-- the K predictions being averaged would not belong to the "
                f"same queries")

        anchor_slots = score_cf_anchor(train_df, slots)
        anchor_auc = float(roc_auc_score(y_test, anchor_slots.mean(axis=0)))
        results["cf_anchor"].append(anchor_auc)
        print(f"  CF-anchor baseline AUC: {anchor_auc:.4f}")

        pred_df = slots[0][["path", "patient_id", f"{args.disease}_prob",
                            "correct"]].copy()
        pred_df["cf_paths"] = fold.test_cf_paths
        pred_df["cf_prob"] = np.mean(
            [sl["cf_prob"].values for sl in slots], axis=0)
        pred_df["cf_anchor_prob"] = anchor_slots.mean(axis=0)

        # Per-slot probabilities, kept so a smaller K can be read off this run.
        # The model is fit on the nearest CF regardless of K, so slot j's
        # prediction does not depend on K at all: mean(slots 0..4) IS the K=5
        # answer, exactly. c2_derive_k.py does that averaging.
        slot_probs = {"cf_anchor": anchor_slots}

        for cfg in configs_wanted:
            if not config_available(cfg, cols):
                print(f"  {cfg}: unavailable for this fold set (needs "
                      f"cf_attr/cf_emb) -- skipped")
                continue

            X_tr = config_matrix(train_df, cfg, cols, args.disease)
            dims_seen.setdefault(cfg, int(X_tr.shape[1]))

            # A config with no CF block scores the same on every slot, so it is
            # evaluated once. That is not an optimisation only: it states that
            # the baselines are K-invariant, which is what lets a K sweep be read
            # as a change in the counterfactual side alone.
            use_slots = slots if cfg in CF_DEPENDENT else slots[:1]
            X_te = [config_matrix(sl, cfg, cols, args.disease) for sl in use_slots]

            for mt in models_wanted:
                t0 = time.time()
                model, scaler = fit_model(X_tr, y_train, model_type=mt)
                per_slot = np.stack(
                    [predict_model(model, scaler, X, mt) for X in X_te])
                # float64 on purpose. RF probabilities are multiples of 1/200
                # and full of exact ties; storing them as float32 reorders those
                # ties and moves the derived AUC in the fourth decimal, which
                # would make a derived K merely close to a computed one instead
                # of equal to it.
                slot_probs[f"{mt}_{cfg}"] = per_slot
                y_prob = per_slot.mean(axis=0)
                auc = float(roc_auc_score(y_test, y_prob))
                results[mt][cfg].append(auc)
                pred_df[f"{mt}_{cfg}_prob"] = y_prob
                print(f"  {mt:<4} {cfg:<5} dim={X_tr.shape[1]:<5} "
                      f"x{len(X_te)} CF   AUC = {auc:.4f}   "
                      f"[{time.time()-t0:.0f}s]")

                if args.save_models:
                    md = os.path.join(out_dir, "models")
                    os.makedirs(md, exist_ok=True)
                    payload = ({"model": model, "scaler": scaler}
                               if scaler is not None else model)
                    joblib.dump(payload,
                                os.path.join(md, f"{mt}_{cfg}_fold{fold.index}.pkl"))
            del X_tr, X_te

        # Written per fold: a wall-clock kill after fold 3 still leaves 3 usable folds.
        pred_df.to_csv(os.path.join(out_dir, f"fold_{fold.index}_predictions.csv"),
                       index=False)
        np.savez_compressed(
            os.path.join(out_dir, f"fold_{fold.index}_slot_probs.npz"),
            paths=slots[0]["path"].values.astype(str),
            patient_id=slots[0]["patient_id"].values.astype(str),
            labels=y_test.astype(np.int8),
            cf_paths=fold.test_cf_paths.astype(str),
            **slot_probs)

    # ── summary ───────────────────────────────────────────────────────────────
    rows = []
    for mt in models_wanted:
        for cfg in configs_wanted:
            aucs = results[mt][cfg]
            if aucs:
                rows.append({"model": mt, "config": cfg, "n_folds": len(aucs),
                             "k_used": args.cf_count if cfg in CF_DEPENDENT else 1,
                             "auc_mean": float(np.mean(aucs)),
                             "auc_std": float(np.std(aucs)),
                             "auc_folds": ";".join(f"{a:.4f}" for a in aucs)})
    if results["cf_anchor"]:
        rows.append({"model": "baseline", "config": "cf_anchor",
                     "n_folds": len(results["cf_anchor"]),
                     "k_used": args.cf_count,
                     "auc_mean": float(np.mean(results["cf_anchor"])),
                     "auc_std": float(np.std(results["cf_anchor"])),
                     "auc_folds": ";".join(f"{a:.4f}" for a in results["cf_anchor"])})

    summary = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    summary_path = os.path.join(out_dir, "cv_summary.csv")
    summary.to_csv(summary_path, index=False)

    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump({"disease": args.disease, "cf_source": src.name,
                   "cf_source_description": src.describe(),
                   "cf_count": args.cf_count, "train_k": 1,
                   "k_combination": "mean of the K predicted probabilities",
                   "n_folds": args.n_folds,
                   "configs": configs_wanted, "models": models_wanted,
                   "cf_dependent_configs": sorted(CF_DEPENDENT),
                   "seed": RANDOM_SEED, "feature_dims": dims_seen}, fh, indent=2)

    print(f"\n{'='*72}\n  SUMMARY\n{'='*72}")
    for _, r in summary.iterrows():
        print(f"  {r['model']:<9} {r['config']:<9} "
              f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}")
    print(f"\n  wrote {summary_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--configs", default="",
                    help=f"comma-separated subset; default all ({','.join(CONFIGS)})")
    ap.add_argument("--models", default=",".join(MODEL_TYPES))
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if cv_summary.csv already exists")
    add_cf_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
