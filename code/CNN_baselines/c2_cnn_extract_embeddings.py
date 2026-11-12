"""
c2_cnn_extract_embeddings.py
============================
Extract-only companion to c2_cnn_all.py.

For configs that were trained BEFORE per-(xi, cf) embedding saving was added,
this reloads each fold's saved best checkpoint and writes the missing
`fold_{i}_pair_embeddings.npz` files WITHOUT retraining. The embeddings are
bit-for-bit what the training path would have produced, because the weights are
identical — only the extra `_embed_pairs` call is run.

It reuses everything from c2_cnn_all.py (models, datasets, `_embed_pairs`, the
`MODEL_REGISTRY`, and the same CV_DIR fold CSVs), so there is a single source of
truth for how a config is built. To keep c2_cnn_all.py's module-level argparse
happy at import time, we prime sys.argv before importing it.

Usage mirrors the training script's selectors:

    python c2_cnn_extract_embeddings.py --disease effusion --cf_count 16 \
        --models dual_sal_scalars

Only configs whose fold_{i}_best.pt checkpoints already exist are processed;
missing ones are reported and skipped (train them normally — they'll emit
embeddings on their own now).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── Parse our own args first, then hand a clean argv to c2_cnn_all's parser ─────
_p = argparse.ArgumentParser()
_p.add_argument('--disease', type=str, default='effusion')
_p.add_argument('--cf_count', type=int, default=16)
_p.add_argument('--models', type=str, nargs='+', default=None,
                help="Configs to extract (default: all in the registry).")
_p.add_argument('--batch_size', type=int, default=32)
_p.add_argument('--overwrite', action='store_true',
                help="Re-extract even if the npz already exists.")
_args = _p.parse_args()

# Prime argv so importing c2_cnn_all builds CV_DIR / paths for the right disease
# and cf_count. --models is intentionally left at its default (all) inside the
# imported module; we select configs ourselves below.
sys.argv = [
    'c2_cnn_all.py',
    '--disease', _args.disease,
    '--cf_count', str(_args.cf_count),
    '--batch_size', str(_args.batch_size),
]

import c2_cnn_all as C  # noqa: E402  (import after argv priming, by design)


def main():
    disease  = _args.disease
    cf_count = _args.cf_count
    selected = _args.models or list(C.MODEL_REGISTRY.keys())

    # Same fold CSVs the training run consumed — the embeddings must be keyed
    # against the identical test rows, so we load from C.CV_DIR verbatim.
    fold_dfs = [
        pd.read_csv(os.path.join(C.CV_DIR, f'fold_{i}_predictions.csv'))
        for i in range(C.N_FOLDS)
    ]

    # Guard the split assumption exactly as c2_cnn_all.py does at startup.
    for i in range(C.N_FOLDS):
        for j in range(i + 1, C.N_FOLDS):
            overlap = set(fold_dfs[i]['patient_id']) & set(fold_dfs[j]['patient_id'])
            assert len(overlap) == 0, (
                f"Patient overlap between outer folds {i} and {j}: {len(overlap)}."
            )
    print(f"Verified: zero patient overlap across {C.N_FOLDS} outer folds")

    # Scalar configs need per-CF C0 probabilities attached, because their
    # dataset (__getitem__) reads query_prob / cf_probs even though _embed_pairs
    # ignores the scalar tensor itself.
    if any(k.endswith('_scalars') for k in selected):
        C._attach_scalar_cols(fold_dfs)

    for key in selected:
        if key not in C.MODEL_REGISTRY:
            print(f"[skip] unknown config '{key}'")
            continue
        cfg         = C.MODEL_REGISTRY[key]
        output_dir  = cfg['output_dir']
        dataset_cls = cfg['dataset_cls']
        make_model  = cfg['make_model']
        mode        = cfg['mode']
        encoder_names = {
            'single': ['enc'],
            'dual':   ['query_enc', 'cf_enc'],
            'quad':   ['query_enc', 'cf_enc', 'qcam_enc', 'cfcam_enc'],
            'quad_scalars': ['query_enc', 'cf_enc', 'qcam_enc', 'cfcam_enc'],
        }[mode]

        print(f"\n{'='*60}\n[{cfg['name']}]  ({key}, cf_{cf_count})\n{'='*60}")
        if not os.path.isdir(output_dir):
            print(f"[skip] no output dir yet: {output_dir}")
            continue

        for fold in range(C.N_FOLDS):
            ckpt = os.path.join(output_dir, f'fold_{fold}_best.pt')
            npz  = os.path.join(output_dir, f'fold_{fold}_pair_embeddings.npz')
            if not os.path.exists(ckpt):
                print(f"  fold {fold}: [skip] missing checkpoint {os.path.basename(ckpt)}")
                continue
            if os.path.exists(npz) and not _args.overwrite:
                print(f"  fold {fold}: [skip] embeddings exist (use --overwrite)")
                continue

            # Rebuild the TEST loader exactly as run_cv does: full K (test_df is
            # never passed through _nearest_cf_only) and shuffle=False, so the
            # (sample_path, cf_path) keys _embed_pairs reconstructs stay aligned.
            test_df     = fold_dfs[fold]
            test_loader = DataLoader(
                dataset_cls(test_df, C.DATA_DIR, C.transform),
                batch_size=C.BATCH_SIZE, shuffle=False,
                num_workers=4, persistent_workers=True, pin_memory=True,
            )

            model = make_model().to(C.DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=C.DEVICE))

            emb, emb_lbl, s_paths, c_paths = C._embed_pairs(
                model, test_loader, encoder_names, test_df, C.DEVICE)

            # Store embeddings as float16 to halve on-disk size (~12GB -> ~6GB
            # for cf_16): these are fusion features, and ~3-4 significant digits
            # is far below the noise the downstream model cares about. Keys and
            # labels stay full precision.
            emb = emb.astype(np.float16)

            np.savez(
                npz,
                emb=emb, labels=emb_lbl, sample_paths=s_paths, cf_paths=c_paths,
                mode=mode,
                encoder_names=np.asarray(encoder_names, dtype=object),
                emb_dim=emb.shape[1], cf_count=cf_count, disease=disease,
                config=cfg['name'], fold=fold,
            )
            print(f"  fold {fold}: wrote {emb.shape[0]} rows x {emb.shape[1]} dims "
                  f"-> {os.path.basename(npz)}")


if __name__ == '__main__':
    main()
