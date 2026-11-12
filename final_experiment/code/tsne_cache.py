"""t-SNE maps cached on disk, keyed by the exact input matrix.

The explainability notebooks redraw the same C2 maps on every run (sklearn t-SNE, minutes per map).
`cached_tsne` stores each map as <cache_dir>/<tag>_<hash>.npz, the hash covering the input values,
their shape and the t-SNE settings, so a changed embedding, sample or setting computes a new map and
an unchanged one is read back: identical, as t-SNE runs with a fixed random_state.
"""
import os, hashlib
import numpy as np
from sklearn.manifold import TSNE


def cached_tsne(X, cache_dir, tag, perplexity=30, init="pca", random_state=42):
    """2-D t-SNE of X (n x d), read from cache_dir when this exact X was mapped before."""
    X = np.ascontiguousarray(X)                  # dtype kept: t-SNE runs on X exactly as given
    h = hashlib.sha1(X.tobytes())
    h.update(repr((X.shape, str(X.dtype), perplexity, init, random_state)).encode())
    fp = os.path.join(cache_dir, f"{tag}_{h.hexdigest()[:16]}.npz")
    if os.path.exists(fp):
        print(f"t-SNE {tag}: cached ({fp})")
        return np.load(fp)["emb"]
    emb = TSNE(n_components=2, perplexity=perplexity, init=init,
               random_state=random_state).fit_transform(X)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(fp, emb=emb)
    print(f"t-SNE {tag}: computed, saved to {fp}")
    return emb
