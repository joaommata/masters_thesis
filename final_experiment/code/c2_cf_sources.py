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

import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402

RESULTS_DIR = os.environ.get("THESIS_RESULTS", "/work3/s251710/thesis_results")


# ══════════════════════════════════════════════════════════════════════════════
# SUBSETS
# ══════════════════════════════════════════════════════════════════════════════
# A subset restricts BOTH the fold membership and the KNN counterfactual pool to
# the same rows. It exists so the diffusion arm -- which only has counterfactuals
# for the 20k images that were generated -- can be compared against a KNN arm
# trained on exactly those rows, leaving the counterfactual source as the only
# difference between the two.
#
# The list is a frozen file, not a query. Re-deriving it from the CF manifest at
# each run would let a later top-up of the generation campaign silently move the
# subset under results that are already finished.

def subset_path(disease, name):
    return os.path.join(RESULTS_DIR, "C2_final", disease, "subsets", f"{name}.txt")


def load_subset(disease, name):
    """The frozen path list for a named subset, or None when name is empty."""
    if not name:
        return None
    p = subset_path(disease, name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"subset {name!r} not frozen for {disease}: {p} does not exist.\n"
            f"Write it with cf_levels_export_images.py, or drop --subset.")
    with open(p) as fh:
        paths = [ln.strip() for ln in fh if ln.strip()]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{p} has duplicate paths")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# BASE
# ══════════════════════════════════════════════════════════════════════════════

class CFSource:
    """Base class. A source pairs one CV fold and names itself for the cache path."""

    name = "base"
    subset = ""

    # ── hooks a source overrides only if it needs to ──────────────────────────

    def subset_paths(self, disease):
        """Rows c2_data.csv is restricted to, or None for the whole table.

        Read at the single choke point in c2_folds.load_c2_data, so the fold
        membership and the KNN pool cannot go out of sync -- subsetting one but
        not the other would let the KNN arm retrieve from rows it never trains
        on, which is both a leak and an unfair advantage over the diffusion arm.
        """
        return load_subset(disease, self.subset)

    def cf_prob_lut(self, disease):
        """{cf_path: C0 probability} for counterfactuals not in c2_data.csv.

        Empty for a retrieval source, where every counterfactual IS a row there.
        """
        return {}

    def aux_table(self, disease, cf_count, columns=None):
        """Extra c2_data-shaped rows describing this source's counterfactuals.

        None for a retrieval source. A generative source returns one row per
        counterfactual, with exactly `columns` in exactly that order, so the
        attribute ladder can dereference a CF path the same way it dereferences
        a query path.
        """
        return None

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

    def __init__(self, strategy="correct_cf", distance="l1", k_offset=0, subset=""):
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"unknown knn strategy {strategy!r}; "
                f"expected one of {sorted(self._STRATEGIES)}")
        if k_offset and strategy != "further":
            raise ValueError("k_offset only applies to strategy='further'")
        self.strategy = strategy
        self.distance = distance
        self.k_offset = k_offset
        self.subset = subset

        self.name = f"knn_{strategy}"
        if distance != "l1":
            self.name += f"_{distance}"
        if k_offset:
            self.name += f"_k{k_offset}"
        # The subset goes in the NAME, which is the only directory component of
        # both the fold cache and the results tree. A subset arm therefore lands
        # beside the full-set arm rather than overwriting it, and no consumer
        # needs to learn what a subset is.
        if subset:
            self.name += f"_sub{subset}"

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
                + (f" / k_offset={self.k_offset}" if self.k_offset else "")
                + (f" / subset={self.subset}" if self.subset else ""))


# ══════════════════════════════════════════════════════════════════════════════
# DIFFUSION  (placeholder -- deliberately not implemented yet)
# ══════════════════════════════════════════════════════════════════════════════

class DiffusionCFSource(CFSource):
    """Counterfactuals GENERATED for each query along a guided diffusion trajectory.

    Produced by cf_generate_levels.py: three per query, each its own complete
    FastDiME trajectory guided to a different soft target probability, so the
    three differ in how far they push the classifier and not in how finished the
    image is.

    Slot order is HIGH, MID, LOW -- strongest edit first
    -------------------------------------------------------------------
    Nothing in the pipeline knows what a "level" is; it knows that cf_paths is
    ordered and that slot 0 is the one training uses. Putting `high` first makes
    K=1 the strongest counterfactual, which is the retrieval arm's analogue of
    the NEAREST neighbour: the single most informative pairing. It also makes the
    K=1 pairing a strict PREFIX of the K=3 pairing, which is what `derive_pairing`
    and `check_nesting` assume and what lets the K=1 numbers be reproduced from
    the K=3 pairing exactly.

    The level set is deliberately absent from `name`. cf_1 and cf_3 must share a
    source name or they are two unrelated experiments as far as the fold cache,
    the K derivation and the ensembler are concerned.

    Why this does not leak across the fold boundary
    -----------------------------------------------
    The base class pairs a query against another REAL row, so it has to keep the
    counterfactual pool inside the train partition. That constraint does not
    transfer here, and the reason is worth stating rather than assuming:

      * the counterfactual for query x is a function of x alone, the frozen C0
        classifier and the frozen diffusion UNet. No other row of c2_data.csv
        participates in generating it. A test query's counterfactual therefore
        carries no information about any training row, and vice versa;
      * the direction of the edit is chosen from C0's PREDICTION on x, never
        from its label or its correctness -- the same property that makes the
        `correct_cf` routing computable at test time;
      * consequently `attach()`'s assumption that CF paths resolve inside
        train_out is not merely unnecessary here, it is wrong. It is overridden
        below rather than left as a live trap.

    Two dependencies that are NOT fold-local, and are reported as limitations:

      * the guidance targets in levels.json are the 5th/95th percentiles of C0's
        probability over the CORRECTLY-CLASSIFIED rows of the whole C2 set,
        test rows included. That is six scalars per disease, identical for every
        query and carrying no per-row information -- the same category as
        choosing a hyperparameter on the full dataset -- but it is a real
        dependency on test-row labels and cf_levels_check_targets.py measures how
        far the targets move when they are estimated from train rows only;
      * the FastDiME UNet was pretrained on a CheXpert split that probably
        overlaps these rows. It does not distinguish folds, but it means the
        generator may partly reconstruct images it has seen.
    """

    #: strongest edit first -- see the class docstring.
    LEVEL_ORDER = ("high", "mid", "low")

    def __init__(self, levels_root=None, subset="", disease=None):
        self.levels_root = levels_root or os.path.join(RESULTS_DIR, "CF_levels")
        self.subset = subset
        self.name = "diffusion_" + "".join(lv[0] for lv in self.LEVEL_ORDER)
        if subset:
            self.name += f"_sub{subset}"
        self._pairs_cache = {}

    def describe(self):
        return (f"diffusion / levels={'>'.join(self.LEVEL_ORDER)}"
                + (f" / subset={self.subset}" if self.subset else ""))

    # ── on-disk tables ───────────────────────────────────────────────────────

    def _dir(self, disease):
        return os.path.join(self.levels_root, disease)

    def pairs_path(self, disease):
        return os.path.join(self._dir(disease), "cf_pairs.csv")

    def table_path(self, disease):
        return os.path.join(self._dir(disease), "c2_cf_table.parquet")

    def _pairs(self, disease):
        """query_path -> {level_name: (cf_path, prob)}, plus the flat prob lut.

        Small (60k x 4); the 1,486-column feature table is only read by
        aux_table(), and only for the levels a given K actually uses.
        """
        got = self._pairs_cache.get(disease)
        if got is not None:
            return got

        p = self.pairs_path(disease)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found -- the diffusion counterfactuals have not been "
                f"assembled yet.\n"
                f"Run, in order:  cf_levels_export_images.py, "
                f"cf_levels_c0_predict.py, cf_levels_c1_attributes.py, "
                f"cf_levels_assemble.py  (--disease {disease})")
        df = pd.read_csv(p, usecols=["query_path", "level_name", "cf_path", "prob"])

        by_query = {}
        for q, lv, cf, pr in zip(df["query_path"], df["level_name"],
                                 df["cf_path"], df["prob"]):
            by_query.setdefault(q, {})[lv] = (cf, float(pr))
        prob_lut = dict(zip(df["cf_path"], df["prob"].astype(float)))

        got = (by_query, prob_lut)
        self._pairs_cache[disease] = got
        return got

    # ── the CFSource contract ────────────────────────────────────────────────

    def pair(self, train_df, test_df, disease, cf_count):
        if not 1 <= cf_count <= len(self.LEVEL_ORDER):
            raise ValueError(
                f"cf_count={cf_count}: only {len(self.LEVEL_ORDER)} intensity "
                f"levels were generated ({', '.join(self.LEVEL_ORDER)})")
        levels = self.LEVEL_ORDER[:cf_count]
        by_query, _ = self._pairs(disease)

        out = []
        for frame in (train_df, test_df):
            cells, means = [], []
            for q in frame["path"].values:
                got = by_query.get(q)
                if got is None:
                    raise ValueError(
                        f"no generated counterfactual for query {q!r}. Every row "
                        f"of the fold must have one -- restrict the folds to the "
                        f"generated subset with --subset, or regenerate.")
                try:
                    cfs = [got[lv] for lv in levels]
                except KeyError as exc:
                    # A ragged K would only surface later, inside MaterializedFold
                    # or slot_pairing, by which point the query is no longer named.
                    raise ValueError(
                        f"query {q!r} is missing the {exc.args[0]!r} level "
                        f"(has {sorted(got)})") from None
                cells.append("|".join(c for c, _ in cfs))
                means.append(float(np.mean([pr for _, pr in cfs])))
            f = frame.copy()
            f["cf_paths"] = cells
            f["cf_prob"] = means
            out.append(f)
        return out[0], out[1]

    def cf_prob_lut(self, disease):
        return self._pairs(disease)[1]

    def aux_table(self, disease, cf_count, columns=None):
        """One c2_data-shaped row per generated counterfactual, for the used levels.

        Restricting to `LEVEL_ORDER[:cf_count]` is not cosmetic: it is what keeps
        a K=1 fold two-thirds smaller in memory than a K=3 one.
        """
        levels = set(self.LEVEL_ORDER[:cf_count])
        by_query, _ = self._pairs(disease)
        wanted = {cf for got in by_query.values()
                  for lv, (cf, _) in got.items() if lv in levels}

        p = self.table_path(disease)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found -- run cf_levels_assemble.py --disease {disease}")
        tab = pd.read_parquet(p)
        tab = tab[tab["path"].isin(wanted)].reset_index(drop=True)
        # c2_cf_table.parquet mirrors c2_data.csv on disk, and load_c2_data
        # renames these three on read. The aux rows have to arrive in the same
        # shape as the frame they are concatenated onto, not the shape they were
        # written in.
        tab = tab.rename(columns={"prob": f"{disease}_prob",
                                  "pred": f"{disease}_pred",
                                  "true": f"{disease}_true"})

        if columns is not None:
            missing = [c for c in columns if c not in tab.columns]
            extra = [c for c in tab.columns if c not in columns]
            # An extra column silently becomes a 455th attribute in every
            # attribute config, because the feature blocks are built by
            # subtracting the known meta names. A missing one becomes a
            # column of zeros. Both are silent; neither is acceptable.
            if missing or extra:
                raise ValueError(
                    f"{p} does not match c2_data.csv's schema: "
                    f"{len(missing)} missing (first {missing[:3]}), "
                    f"{len(extra)} extra (first {extra[:3]})")
            tab = tab[list(columns)]
        return tab

    def attach(self, train_out, test_out, disease):
        raise NotImplementedError(
            "CFSource.attach() builds its lookup from train_out, because a "
            "retrieved counterfactual is a row of the train fold. A generated "
            "counterfactual is not in any fold; it is dereferenced through "
            "aux_table(), which c2_folds.load_folds concatenates onto the base "
            "frame before building the FoldContext.")


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def build_cf_source(kind, strategy="correct_cf", distance="l1", k_offset=0,
                    levels_root=None, subset=""):
    """Factory used by both trainers so they stay identical in this respect."""
    if kind == "knn":
        return KnnCFSource(strategy=strategy, distance=distance,
                           k_offset=k_offset, subset=subset)
    if kind == "diffusion":
        return DiffusionCFSource(levels_root=levels_root, subset=subset)
    raise ValueError(f"unknown cf source {kind!r}; expected 'knn' or 'diffusion'")


CF_SOURCE_KINDS = ("knn", "diffusion")
KNN_STRATEGIES = tuple(KnnCFSource._STRATEGIES)
