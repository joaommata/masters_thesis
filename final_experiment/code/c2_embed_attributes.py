# c2_embed_attributes.py
"""
Hidden-layer embeddings of the attribute C2 MLP, for clustering in
attribute_explainability.ipynb.

The CV runs of c2_train_attributes.py keep no weights, and the five fold models
do not share an embedding space (their hidden units are not aligned), so the
embeddings cannot be pooled across folds. Instead ONE model is fit on the train
side of one fold (fold 4 by default: trained on folds 0-3) and only that fold's
held-out queries are embedded. Training rows are never embedded: their
embeddings would be overfit.

Two variants of the I+C+D model, same MLP and same training as the CV runs:

    full    exactly the C2 I+C+D model: attributes AND C0 embeddings, per block.
            Its probabilities should reproduce fold_<i>_predictions.csv.
    attrs   attributes only (attr, cf_attr, delta): every input is a named
            measurement, so a cluster's attribute vector describes all of what
            put its images there.

Per variant, per query and per counterfactual slot (K of them): the 32- and
16-unit hidden activations (after ReLU; the 16 units are what the output layer
reads) and P(correct). The notebook averages the K slots, as C2 averages its
scores.

    python c2_embed_attributes.py --disease cardiomegaly --cf-source knn --cf-count 10

Results land in .../<source>/cf_<K>/attributes_embeddings/fold_<i>/.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_folds import RESULTS_DIR, load_folds, add_cf_args, cf_source_from_args  # noqa: E402
from c2_train_attributes import DEVICE, MODEL, config_matrix, feature_columns, fit  # noqa: E402

CONFIG = "I+C+D"
VARIANTS = ["full", "attrs"]


def attr_matrix(df, cols):
    """I+C+D without the C0 embeddings: attr, cf_attr, delta."""
    return np.hstack([df[cols["attr"]].values, df[cols["cf_attr"]].values,
                      df[cols["delta"]].values]).astype(np.float32)


def design(df, cols, variant):
    return config_matrix(df, cols, CONFIG) if variant == "full" else attr_matrix(df, cols)


def embed(net, scaler, X):
    """(h32, h16, prob) of the rows of X; net is the 64-32-16-1 Sequential."""
    x = torch.tensor(scaler.transform(X), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        h32 = net[:4](x)
        h16 = net[4:6](h32)
        prob = torch.sigmoid(net[6](h16)).squeeze(1)
    return h32.cpu().numpy(), h16.cpu().numpy(), prob.cpu().numpy().astype(np.float64)


def run(args):
    src = cf_source_from_args(args)
    base_dir = os.path.join(RESULTS_DIR, "C2_final_results", args.disease, src.name,
                            f"cf_{args.cf_count}")
    out_dir = os.path.join(base_dir, "attributes_embeddings", f"fold_{args.fold}")
    os.makedirs(out_dir, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",")]
    print(f"  {args.disease} | {src.describe()} | K={args.cf_count} | fold {args.fold}"
          f" | variants {variants} | device {DEVICE}")

    for fold in load_folds(args.disease, src, args.cf_count, n_folds=args.n_folds):
        if fold.index != args.fold:
            continue
        train_df = fold.train
        cols = feature_columns(train_df, args.disease)
        y_train, groups = train_df["correct"].values, train_df["patient_id"].values
        slots = [fold.test_slot(j) for j in range(fold.n_slots)]
        q = slots[0]
        y_test = q["correct"].values
        print(f"  train {len(train_df):,} | test {len(q):,} x {fold.n_slots} CF")

        for v in variants:
            net, scaler = fit(design(train_df, cols, v), y_train, groups)
            h32, h16, prob = map(np.stack, zip(*[embed(net, scaler, design(sl, cols, v))
                                                 for sl in slots]))   # (K, n, d)
            score = prob.mean(axis=0)
            auc = float(roc_auc_score(y_test, score))
            dead = int((h16.max(axis=(0, 1)) == 0).sum())
            print(f"  {v:<6} dim={len(scaler.mean_):<5} AUC = {auc:.4f}"
                  f"   dead 16-units: {dead}/16", flush=True)

            if v == "full":   # must match the CV run of the same fold
                pub = pd.read_csv(os.path.join(base_dir, "attributes",
                                               f"fold_{args.fold}_predictions.csv"))
                pub = pub.set_index("path").loc[q["path"].values, f"{MODEL}_{CONFIG}_prob"]
                print(f"         vs published fold {args.fold}: AUC "
                      f"{roc_auc_score(y_test, pub):.4f}, r = "
                      f"{np.corrcoef(score, pub)[0, 1]:.4f}, max |diff| "
                      f"{np.abs(score - pub.values).max():.4f}")

            np.savez_compressed(
                os.path.join(out_dir, f"{v}.npz"),
                paths=q["path"].values.astype(str),
                patient_id=q["patient_id"].values.astype(str),
                correct=y_test.astype(np.int8),
                pred=q[f"{args.disease}_pred"].values.astype(np.int8),
                cf_paths=fold.test_cf_paths.astype(str),
                h32=np.moveaxis(h32, 0, 1).astype(np.float32),   # (n, K, 32)
                h16=np.moveaxis(h16, 0, 1).astype(np.float32),   # (n, K, 16)
                slot_probs=prob.T,                               # (n, K)
                prob=score)
            torch.save({"state_dict": net.state_dict(), "scaler_mean": scaler.mean_,
                        "scaler_scale": scaler.scale_, "auc": auc},
                       os.path.join(out_dir, f"{v}_model.pt"))
            if DEVICE != "cpu":
                torch.cuda.empty_cache()

        with open(os.path.join(out_dir, "columns.json"), "w") as fh:
            json.dump({"full": [f"{p}{c}" for p, c in
                                [("", c) for c in cols["attr"] + cols["emb"]]
                                + [("", c) for c in cols["cf_attr"] + cols["cf_emb"]]
                                + [("", c) for c in cols["delta"]]
                                + [("emb_minus_cf_", c) for c in cols["emb"]]],
                       "attrs": cols["attr"] + cols["cf_attr"] + cols["delta"]},
                      fh, indent=1)
        print(f"  wrote {out_dir}")
        return
    raise SystemExit(f"fold {args.fold} not found")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--fold", type=int, default=4, help="held-out fold to embed")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    add_cf_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
