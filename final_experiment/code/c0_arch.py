# c0_arch.py
"""
Which C0 a results root belongs to, and what follows from it.

The architecture-generalisation experiment reruns the whole C2 pipeline once per
C0 backbone. Each backbone gets its own results root, so no output of one C0 can
land on top of another's and every script runs unchanged with THESIS_RESULTS
pointed at it:

    densenet121  /work3/s251710/thesis_results               (the main C0)
    vit_b_16     /work3/s251710/thesis_results/arch_vit_b_16
    resnet50     /work3/s251710/thesis_results/arch_resnet50

jobs/arch_setup.sh builds an arch root: C0_final -> ../C0_final_<arch>, so the
scripts' hardcoded C0_final/multilabel_ignore path resolves to that backbone.

The backbone is read from the root's own C0 labels.json rather than from a
separate setting, so the embedding width can never disagree with the C0 whose
embeddings are on disk. The DenseNet labels.json predates the "arch" key.
"""
import json
import os

BASE_RESULTS = "/work3/s251710/thesis_results"

# Width of the penultimate feature C0's head reads (emb_0 .. emb_<D-1>).
EMB_DIM = {"densenet121": 1024, "resnet50": 2048, "vit_b_16": 768}


def arch_root(arch):
    """The results root of one C0 backbone."""
    return BASE_RESULTS if arch == "densenet121" else os.path.join(BASE_RESULTS, f"arch_{arch}")


def c0_arch(results_dir, policy="ignore"):
    """Backbone of the C0 under <results_dir>/C0_final/multilabel_<policy>."""
    p = os.path.join(results_dir, "C0_final", f"multilabel_{policy}", "labels.json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found -- is THESIS_RESULTS an arch root built by "
                         f"jobs/arch_setup.sh?")
    with open(p) as f:
        return json.load(f).get("arch", "densenet121")


def n_emb(results_dir, policy="ignore"):
    return EMB_DIM[c0_arch(results_dir, policy)]
