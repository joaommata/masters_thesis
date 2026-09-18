# c2_cf_sources.py
"""
Pluggable counterfactual sources for the final_experiment C2 pipelines.

A CF source answers one question: for each query row, which other image(s) is it
paired against, and what do the resulting delta features look like? Everything
downstream -- the attribute ladder and the CNN ladder alike -- consumes the same
columns regardless of where the pairing came from:

    cf_paths    K pipe-separated image paths, nearest-first
    cf_prob     C0 probability of the CF (mean over the K, if K > 1)
    delta_<c>   query attribute minus CF attribute, for each of the 454 attrs
    cf_attr_<c> the CF's own attribute value  (added by attach_cf_features)
    cf_emb_<c>  the CF's own embedding        (added by attach_cf_features)
    delta_prob  query prob minus cf_prob      (added by attach_cf_features)

Why a class instead of the old boolean chain
--------------------------------------------
The previous pipeline selected its CF strategy with a five-way if/elif over
--correct_cf / --gt_routing / --unmatched / --k_offset, and every consumer had to
re-derive which branch had run in order to know what the output meant. Adding
diffusion CFs to that shape would mean touching every consumer. Here a source is
named once (--cf-source) and the trainers never ask again.

The KNN matching itself is NOT reimplemented here. It is imported from
code/c2/c2_prepare_data_simulated_cf.py so there is exactly one copy: a silent
divergence between two implementations is precisely what shifted every CF pair
when the feature spec was last corrected.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_C2 = os.path.join(REPO, "code", "c2")
if _C2 not in sys.path:
    sys.path.insert(0, _C2)

from c2_prepare_data_simulated_cf import (            # noqa: E402
    compute_cf_for_split_correct_cf,
    compute_cf_for_split_gt_routing,
    compute_cf_for_split_unmatched,
    compute_cf_for_split_matched_train_unmatched_test,
    compute_cf_for_split_further,
    attach_cf_features,
)
from c2_feature_spec import C0_DERIVED_COLS, RSNA_META_COLS   # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# BASE
# ══════════════════════════════════════════════════════════════════════════════

class CFSource:
    """Base class. A source pairs one CV fold and names itself for the cache path."""

    name = "base"

    def pair(self, train_df, test_df, disease, cf_count):
        """Return (train_out, test_out) with cf_paths / cf_prob / delta_* attached.

        train_df is the CF POOL as well as the training partition: CF paths must
        point into train_df only, or the pairing leaks test rows into training.
        """
        raise NotImplementedError

    def attach(self, train_out, test_out, disease):
        """Dereference cf_paths -> cf_attr_*, cf_emb_*, delta_prob.

        Shared by every source: once cf_paths exists the lookup is identical, and
        the CF pool is always the train fold.
        """
        meta = {f"{disease}_prob", f"{disease}_pred", f"{disease}_true",
                "correct", "path", "cam_path", "patient_id",
                "cf_prob", "cf_paths",
                *C0_DERIVED_COLS, *RSNA_META_COLS}
        emb_cols = [c for c in train_out.columns if c.startswith("emb_")]
        rel_cols = [c for c in train_out.columns
                    if c not in meta
                    and not c.startswith("emb_")
                    and not c.startswith("delta_")]

        lookup = train_out                      # CF paths always point into train
        train_out = attach_cf_features(train_out, lookup, rel_cols, emb_cols, disease)
        test_out  = attach_cf_features(test_out,  lookup, rel_cols, emb_cols, disease)
        return train_out, test_out

    def describe(self):
        return self.name


# ══════════════════════════════════════════════════════════════════════════════
# KNN
# ══════════════════════════════════════════════════════════════════════════════

class KnnCFSource(CFSource):
    """Nearest-opposite-neighbour retrieval in the standardised attribute space.

    strategy:
      correct_cf  query pred=1 -> nearest TN, pred=0 -> nearest TP, CF pool
                  restricted to correctly-classified train rows. Routing uses the
                  query's PREDICTION only -- never its label or its correctness --
                  so it is computable at test time. This is the thesis default.
      gt_routing  routes on ground truth instead. Leaky at test time; diagnostic only.
      unmatched   nearest opposite-prediction neighbour, no correctness filter.
      matched     matched train / unmatched test (the original scheme).
      further     correct_cf but taking the (k_offset + k)-th neighbour, to test
                  whether the signal is really about proximity.
    """

    _STRATEGIES = {
        "correct_cf": compute_cf_for_split_correct_cf,
        "gt_routing": compute_cf_for_split_gt_routing,
        "unmatched":  compute_cf_for_split_unmatched,
        "matched":    compute_cf_for_split_matched_train_unmatched_test,
        "further":    compute_cf_for_split_further,
    }

    def __init__(self, strategy="correct_cf", distance="l1", k_offset=0):
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"unknown knn strategy {strategy!r}; "
                f"expected one of {sorted(self._STRATEGIES)}")
        if k_offset and strategy != "further":
            raise ValueError("k_offset only applies to strategy='further'")
        self.strategy = strategy
        self.distance = distance
        self.k_offset = k_offset

        self.name = f"knn_{strategy}"
        if distance != "l1":
            self.name += f"_{distance}"
        if k_offset:
            self.name += f"_k{k_offset}"

    def pair(self, train_df, test_df, disease, cf_count):
        fn = self._STRATEGIES[self.strategy]
        kwargs = dict(train_df=train_df, test_df=test_df,
                      cf_count=cf_count, disease=disease, distance=self.distance)
        if self.strategy == "further":
            kwargs["k_offset"] = self.k_offset
        train_out, test_out, _scaler = fn(**kwargs)
        return train_out, test_out

    def describe(self):
        return (f"KNN / {self.strategy} / distance={self.distance}"
                + (f" / k_offset={self.k_offset}" if self.k_offset else ""))


# ══════════════════════════════════════════════════════════════════════════════
# DIFFUSION  (placeholder -- deliberately not implemented yet)
# ══════════════════════════════════════════════════════════════════════════════

class DiffusionCFSource(CFSource):
    """Generated counterfactuals along a diffusion trajectory.

    Not implemented. The contract it must satisfy when it is:

    1. Emit the same columns a KNN source does -- cf_paths, cf_prob, delta_<attr>
       for the same 454 attributes, in the same order. Everything downstream is
       built by column-name subtraction, so a missing or extra name silently
       changes the feature space rather than raising.

    2. Be fold-safe. KNN pairing is naturally fold-local because the pool IS the
       train partition. Generated CFs are not: they are produced offline per
       query image and carry no notion of a fold. If a generated CF was produced
       using information from rows that later land in test, that leaks. Whatever
       fills this in has to state explicitly why it does not.

    3. `attach()` above assumes CF paths resolve inside the train fold, because
       it builds its lookup from train_out. Generated CFs live in their own
       directory and will need either their own attribute/embedding table or an
       override of attach().

    Point it at the existing grids under $THESIS_RESULTS/diffusion_cf and
    diverse_cf when the attributes for those are rebuilt on the final split.
    """

    name = "diffusion"

    def __init__(self, grid_dir=None):
        self.grid_dir = grid_dir

    def pair(self, train_df, test_df, disease, cf_count):
        raise NotImplementedError(
            "DiffusionCFSource is a placeholder. See the class docstring for the "
            "three things an implementation must guarantee -- in particular that "
            "generated CFs do not leak across the fold boundary.")


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def build_cf_source(kind, strategy="correct_cf", distance="l1", k_offset=0,
                    grid_dir=None):
    """Factory used by both trainers so they stay identical in this respect."""
    if kind == "knn":
        return KnnCFSource(strategy=strategy, distance=distance, k_offset=k_offset)
    if kind == "diffusion":
        return DiffusionCFSource(grid_dir=grid_dir)
    raise ValueError(f"unknown cf source {kind!r}; expected 'knn' or 'diffusion'")


CF_SOURCE_KINDS = ("knn", "diffusion")
KNN_STRATEGIES = tuple(KnnCFSource._STRATEGIES)
