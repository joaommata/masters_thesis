# c2_embed_images.py
"""
Hidden-layer embeddings of an image (CNN) C2 model, for clustering in
saliency_explainability.ipynb. The image-model counterpart of
c2_embed_attributes.py.

Unlike the attribute MLP, the CV runs of c2_train_images.py keep their weights
(fold_<i>_best.pt), so nothing is retrained: fold FOLD's published checkpoint is
loaded and run on that fold's held-out queries. The five fold models do not
share an embedding space, so only one fold is embedded (fold 4 by default, as
for the attributes).

The embedding is what the output layer reads: the input of the classifier's
last Linear. For the dual / quad encoders that is the 256-unit hidden layer
after ReLU; for the single-encoder models (xi, cf) the 1024-dim pooled
DenseNet feature. Each query is paired with its NEAREST counterfactual only
(slot 0), the same pair the notebook shows, so the probability written here is
checked against column 0 of fold_<i>_slot_probs.npz.

    python c2_embed_images.py --disease cardiomegaly --cf-source knn --cf-count 10 \
        --models dual_sal

Results land in .../<source>/cf_<K>/images_embeddings/<model>/fold_<i>.npz.
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_train_images as ti                                                  # noqa: E402
from c2_folds import RESULTS_DIR, load_folds, add_cf_args, cf_source_from_args  # noqa: E402


def last_linear(model):
    """The classifier's final Linear: its input is the embedding."""
    head = model.classifier
    return head[-1] if isinstance(head, torch.nn.Sequential) else head


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ti.CAM_BASE = os.path.join(RESULTS_DIR, "C0_final", f"multilabel_{args.policy}",
                               args.disease, "gradcam")
    src = cf_source_from_args(args)
    base_dir = os.path.join(RESULTS_DIR, "C2_final_results", args.disease, src.name,
                            f"cf_{args.cf_count}")
    keys = [k.strip() for k in args.models.split(",")]
    print(f"  {args.disease} | {src.describe()} | K={args.cf_count} | fold {args.fold}"
          f" | models {keys} | device {device}")

    prob_col = f"{args.disease}_prob"
    test_df = next(te for i, _, te in load_folds(args.disease, src, args.cf_count,
                                                 materialize=False,
                                                 columns=ti.FOLD_COLS + [prob_col])
                   if i == args.fold)
    test_df = ti._truncate_cfs(test_df, 1)                      # nearest CF only
    print(f"  fold {args.fold}: {len(test_df):,} queries")

    for key in keys:
        cfg = ti.MODEL_REGISTRY[key]
        if cfg["mode"] == "quad_scalars":
            ti._attach_scalar_cols([test_df], args.disease, prob_col)
        model_dir = os.path.join(base_dir, "images", key)
        model = cfg["make_model"]().to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, f"fold_{args.fold}_best.pt"),
                                         map_location=device))
        model.eval()

        feats = []
        hook = last_linear(model).register_forward_hook(
            lambda m, inp, out: feats.append(inp[0].float().cpu().numpy()))
        loader = DataLoader(cfg["dataset"](test_df, ti.DATA_ROOT, ti.transform),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        probs = []
        with torch.no_grad(), autocast(device.type, enabled=device.type == "cuda"):
            for batch in loader:
                logits, b, k, _ = ti._run_batch(model, batch, device, cfg["mode"])
                probs.append(torch.sigmoid(logits.float()).view(b, k)[:, 0].cpu().numpy())
        hook.remove()
        h, prob = np.concatenate(feats), np.concatenate(probs).astype(np.float64)
        y = test_df["correct"].values

        # Must reproduce the CV run: slot 0 of the published per-slot probabilities.
        pub = np.load(os.path.join(model_dir, f"fold_{args.fold}_slot_probs.npz"))
        assert (pub["paths"] == test_df["path"].values.astype(str)).all(), "row order differs"
        print(f"  {key:<17} dim={h.shape[1]:<5} slot-0 AUC = {roc_auc_score(y, prob):.4f}"
              f" | published {roc_auc_score(y, pub['probs'][:, 0]):.4f}, max |diff| "
              f"{np.abs(prob - pub['probs'][:, 0]).max():.4f}"
              f" | dead units {(h.max(axis=0) <= 0).sum()}/{h.shape[1]}", flush=True)

        out_dir = os.path.join(base_dir, "images_embeddings", key)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, f"fold_{args.fold}.npz"),
            paths=test_df["path"].values.astype(str),
            patient_id=test_df["patient_id"].values.astype(str),
            correct=y.astype(np.int8),
            cf_path=test_df["cf_paths"].values.astype(str),
            h=h.astype(np.float32),                             # (n, d)
            prob=prob)                                          # (n,) slot-0 P(correct)
        print(f"  wrote {out_dir}/fold_{args.fold}.npz")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--policy", default="ignore")
    ap.add_argument("--models", default="dual_sal")
    ap.add_argument("--fold", type=int, default=4, help="held-out fold to embed")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    add_cf_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
