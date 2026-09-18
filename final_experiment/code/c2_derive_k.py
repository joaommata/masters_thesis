# c2_derive_k.py
"""
Stage 5e: read a smaller K off a larger-K run, instead of re-running it.

Why this is exact, not an approximation
---------------------------------------
Both ladders combine K by averaging the K predicted probabilities, and in both
ladders the MODEL is trained on the nearest counterfactual only. So the score a
model gives to (query, CF_j) does not depend on how many counterfactuals the run
was configured with. Column j of a K=10 run is the same number it would be in a
K=5 run, and

    K=5 prediction  ==  mean(columns 0..4)  of the K=10 per-slot matrix

with nothing left over. Both trainers write that matrix to
`fold_<i>_slot_probs.npz`; this script averages a prefix of it and writes a
complete cf_<K> results directory in the usual layout.

    python c2_derive_k.py --disease effusion --from-k 10 --to-k 5
    python c2_derive_k.py --disease effusion --from-k 10 --to-k 1 --check

--check compares against the cf_<to-k> results already on disk rather than
writing anything. Run it against the K=1 results the thesis already reports: if
they come back identical, the K=10 run sits on the same folds, the same models
and the same protocol as everything published so far, and the K sweep is one
experiment rather than three.

What it does NOT do
-------------------
Derive a LARGER K from a smaller one. The extra counterfactuals were never
scored, and no averaging invents them.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import (                                          # noqa: E402
    RESULTS_DIR, RANDOM_SEED, N_FOLDS, read_pairing, pairing_path,
    derive_pairing, add_cf_args, cf_source_from_args,
)
from c2_train_attributes import CONFIGS, CF_DEPENDENT           # noqa: E402
from c2_train_images import MODEL_KEYS, MODEL_REGISTRY          # noqa: E402

META_KEYS = ("paths", "patient_id", "labels", "cf_paths", "probs")


def results_root(disease, source_name, k):
    return os.path.join(RESULTS_DIR, "C2_final_results", disease, source_name,
                        f"cf_{k}")


def _prefix_mean(mat, k):
    """Mean of the first k slots. A K-invariant series has one slot and stays put."""
    if mat.ndim == 1:
        mat = mat[None, :]
    if mat.shape[0] < k and mat.shape[0] != 1:
        raise ValueError(
            f"asked for {k} slots but only {mat.shape[0]} were scored -- a larger "
            f"K cannot be derived from a smaller one")
    return mat[:min(k, mat.shape[0])].mean(axis=0)


def _summary_rows(aucs_by_key, k, extra=None):
    rows = []
    for key, aucs in aucs_by_key.items():
        row = {"n_folds": len(aucs),
               "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
               "auc_folds": ";".join(f"{a:.4f}" for a in aucs)}
        row.update((extra or {}).get(key, {}))
        rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# ATTRIBUTES
# ══════════════════════════════════════════════════════════════════════════════

def derive_attributes(disease, source_name, from_k, to_k, n_folds, write=True):
    src = os.path.join(results_root(disease, source_name, from_k), "attributes")
    dst = os.path.join(results_root(disease, source_name, to_k), "attributes")
    aucs, per_fold = {}, {}

    for i in range(n_folds):
        f = os.path.join(src, f"fold_{i}_slot_probs.npz")
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"{f} not found. Per-slot probabilities are written by "
                f"c2_train_attributes.py; a run that predates them cannot be "
                f"sliced and has to be repeated at K={from_k}.")
        z = np.load(f, allow_pickle=False)
        y = z["labels"]
        cols = {"path": z["paths"], "patient_id": z["patient_id"],
                "correct": y}
        for key in z.files:
            if key in META_KEYS:
                continue
            p = _prefix_mean(z[key], to_k)
            cols[f"{key}_prob"] = p
            aucs.setdefault(key, []).append(float(roc_auc_score(y, p)))
        per_fold[i] = pd.DataFrame(cols)

    if not write:
        return aucs

    os.makedirs(dst, exist_ok=True)
    # The target K's PAIRING supplies cf_paths / cf_prob below. It is a prefix of
    # the source K's, costs seconds, and is normally already there -- but nothing
    # else in this chain creates it, so make sure.
    if not all(os.path.exists(pairing_path(disease, source_name, to_k, i))
               for i in range(n_folds)):
        derive_pairing(disease, source_name, to_k, n_folds=n_folds)

    for i, df in per_fold.items():
        # The C0 probability is the MSP baseline every downstream comparison is
        # drawn against, so it has to survive the derivation. It is a property of
        # the query, not of K, and is copied from the source run's predictions.
        src_pred = pd.read_csv(os.path.join(src, f"fold_{i}_predictions.csv"),
                               usecols=["path", f"{disease}_prob"])
        lut = dict(zip(src_pred["path"], src_pred[f"{disease}_prob"]))
        missing = [p for p in df["path"] if p not in lut]
        if missing:
            raise ValueError(
                f"fold {i}: {len(missing)} rows in the per-slot file are absent "
                f"from {src}/fold_{i}_predictions.csv (first: {missing[0]!r})")
        df.insert(2, f"{disease}_prob", [lut[p] for p in df["path"]])

        # cf_paths / cf_prob describe the K being derived, so they come from that
        # K's pairing rather than from the source run's wider one.
        pair = read_pairing(disease, source_name, to_k, i)
        pair = pair[pair["split"] == "test"].set_index("path")
        df["cf_paths"] = pair.loc[df["path"], "cf_paths"].values
        df["cf_prob"] = pair.loc[df["path"], "cf_prob"].values
        df.to_csv(os.path.join(dst, f"fold_{i}_predictions.csv"), index=False)

    rows = []
    for key, a in aucs.items():
        if key == "cf_anchor":
            model, config = "baseline", "cf_anchor"
        else:
            model, config = key.split("_", 1)
        rows.append({"model": model, "config": config, "n_folds": len(a),
                     "k_used": to_k if (config in CF_DEPENDENT
                                        or config == "cf_anchor") else 1,
                     "auc_mean": float(np.mean(a)), "auc_std": float(np.std(a)),
                     "auc_folds": ";".join(f"{x:.4f}" for x in a)})
    summary = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    summary.to_csv(os.path.join(dst, "cv_summary.csv"), index=False)
    _write_provenance(dst, disease, source_name, from_k, to_k, n_folds,
                      "attributes")
    return aucs


# ══════════════════════════════════════════════════════════════════════════════
# IMAGES
# ══════════════════════════════════════════════════════════════════════════════

def derive_images(disease, source_name, from_k, to_k, n_folds, write=True):
    src = os.path.join(results_root(disease, source_name, from_k), "images")
    dst = os.path.join(results_root(disease, source_name, to_k), "images")
    aucs = {}

    for key in MODEL_KEYS:
        d = os.path.join(src, key)
        if not os.path.isdir(d):
            continue
        files = [os.path.join(d, f"fold_{i}_slot_probs.npz") for i in range(n_folds)]
        if not all(os.path.exists(f) for f in files):
            print(f"  {key}: no per-slot probabilities -- skipped")
            continue

        fold_aucs, frames = [], {}
        for i, f in enumerate(files):
            z = np.load(f, allow_pickle=False)
            y = z["labels"]
            # (n, K) here: the CNN scores one query against K CFs, so slots are
            # the second axis, unlike the attribute matrices.
            p = _prefix_mean(z["probs"].T, to_k)
            fold_aucs.append(float(roc_auc_score(y, p)))
            frames[i] = (pd.DataFrame({"path": z["paths"],
                                       "patient_id": z["patient_id"],
                                       "correct": y, f"{key}_prob": p}), y, p)
        aucs[key] = fold_aucs

        if write:
            md = os.path.join(dst, key)
            os.makedirs(md, exist_ok=True)
            for i, (df, y, p) in frames.items():
                df.to_csv(os.path.join(md, f"fold_{i}_predictions.csv"), index=False)
                fpr, tpr, _ = roc_curve(y, p)
                np.savez(os.path.join(md, f"fold_{i}_roc.npz"),
                         fpr=fpr, tpr=tpr, probs=p, labels=y)
            pd.DataFrame({"fold": list(frames), "auc": fold_aucs,
                          "n_test": [len(frames[i][0]) for i in frames]}).to_csv(
                os.path.join(md, "fold_aucs.csv"), index=False)

    if write and aucs:
        os.makedirs(dst, exist_ok=True)
        rows = [{"model": k, "name": MODEL_REGISTRY[k]["name"],
                 "auc_mean": float(np.mean(v)), "auc_std": float(np.std(v)),
                 "auc_folds": ";".join(f"{a:.4f}" for a in v)}
                for k, v in aucs.items()]
        pd.DataFrame(rows).sort_values("auc_mean", ascending=False).to_csv(
            os.path.join(dst, "cv_summary.csv"), index=False)
        _write_provenance(dst, disease, source_name, from_k, to_k, n_folds, "images")
    return aucs


def _write_provenance(dst, disease, source_name, from_k, to_k, n_folds, kind):
    with open(os.path.join(dst, "run_config.json"), "w") as fh:
        json.dump({"disease": disease, "cf_source": source_name,
                   "cf_count": to_k, "train_k": 1, "n_folds": n_folds,
                   "kind": kind, "derived_from_k": from_k,
                   "k_combination": "mean of the K predicted probabilities",
                   "how": f"mean of slots 0..{to_k-1} of the cf_{from_k} "
                          f"per-slot probabilities; no model was refit",
                   "seed": RANDOM_SEED}, fh, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check(disease, source_name, from_k, to_k, n_folds, kinds):
    """Compare the derived numbers against the cf_<to_k> results already on disk."""
    ok = True
    for kind in kinds:
        fn = derive_attributes if kind == "attributes" else derive_images
        try:
            got = fn(disease, source_name, from_k, to_k, n_folds, write=False)
        except FileNotFoundError as e:
            print(f"  {kind}: cannot derive -- {e}")
            ok = False
            continue
        p = os.path.join(results_root(disease, source_name, to_k), kind,
                         "cv_summary.csv")
        if not os.path.exists(p):
            print(f"  {kind}: nothing to compare against ({p} absent)")
            continue
        have = pd.read_csv(p)
        keycol = "config" if kind == "attributes" else "model"
        have_map = {}
        for _, r in have.iterrows():
            k = (r[keycol] if kind == "images"
                 else ("cf_anchor" if r["config"] == "cf_anchor"
                       else f"{r['model']}_{r['config']}"))
            have_map[k] = float(r["auc_mean"])

        print(f"\n  {kind}: comparing {len(got)} series against cf_{to_k}")
        worst, worst_key = 0.0, None
        for k, a in sorted(got.items()):
            if k not in have_map:
                continue
            d = abs(float(np.mean(a)) - have_map[k])
            if d > worst:
                worst, worst_key = d, k
        if worst_key is None:
            print("    no overlapping series")
        else:
            verdict = "OK" if worst < 5e-4 else "MISMATCH"
            print(f"    largest |difference| {worst:.6f} on {worst_key}   {verdict}")
            ok &= worst < 5e-4
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--from-k", type=int, required=True)
    ap.add_argument("--to-k", type=int, required=True)
    ap.add_argument("--kinds", default="attributes,images")
    ap.add_argument("--check", action="store_true",
                    help="compare against the cf_<to-k> results on disk instead "
                         "of writing anything")
    add_cf_args(ap)
    args = ap.parse_args()
    if args.to_k >= args.from_k:
        raise SystemExit(f"--to-k {args.to_k} must be smaller than --from-k "
                         f"{args.from_k}; a larger K cannot be derived")

    src = cf_source_from_args(args)
    kinds = [k.strip() for k in args.kinds.split(",")]

    print(f"\n{'='*72}")
    print(f"  {'CHECK' if args.check else 'DERIVE'}: {args.disease} | {src.name} "
          f"| K={args.to_k} <- K={args.from_k}")
    print(f"{'='*72}")

    if args.check:
        raise SystemExit(0 if check(args.disease, src.name, args.from_k,
                                    args.to_k, args.n_folds, kinds) else 1)

    for kind in kinds:
        fn = derive_attributes if kind == "attributes" else derive_images
        aucs = fn(args.disease, src.name, args.from_k, args.to_k, args.n_folds)
        print(f"\n  {kind}: {len(aucs)} series")
        for k, v in sorted(aucs.items(), key=lambda kv: -np.mean(kv[1]))[:8]:
            print(f"    {k:<22} {np.mean(v):.4f} ± {np.std(v):.4f}")
    print(f"\n  wrote {results_root(args.disease, src.name, args.to_k)}")


if __name__ == "__main__":
    main()
