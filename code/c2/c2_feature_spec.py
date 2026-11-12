"""
c2_feature_spec.py
==================
Single source of truth for which columns are C0-DERIVED and must therefore be
kept OUT of the C1 attribute space.

Why this exists
---------------
The attribute space (`relevant_cols` / `attr_cols` in the C2 pipelines) is built
by subtraction: everything in c2_data.csv that is not in `meta_cols` and not an
emb_* column becomes an attribute, and hence also feeds the delta vectors.

That is only correct if `meta_cols` names EVERY non-attribute column. It did not:
`margin` and the seven `prob_*` internal-classifier columns were silently landing
in the attribute space in every pipeline (8 of 469 columns).

Both are functions of C0's own output, not properties of the image:
  * `margin`   = |prob - threshold|. Perfectly correlated with C0 confidence
                 (corr = 1.0000 with |prob - 0.5634| on effusion). On its own it
                 predicts `correct` at AUC 0.731 (effusion) / 0.717 (RSNA).
  * `prob_*`   = the SDN internal-classifier probabilities (Kaya et al. 2019).
                 prob_denseblock4 alone predicts `correct` at AUC 0.737 (effusion).

Leaving them in defeats the point of the config ladder: B1 is supposed to be the
"no C0 information" anchor, but it was receiving a ~0.73-AUC confidence feature,
which compresses the measured gap between the baselines and the ΔA models. It
also survives the --entropy switch: entropy replaces `prob`, but `margin` carried
the same raw-confidence signal through untouched.

The attribute space must contain ONLY image-derived properties:
demographic (age/sex/race), geometric (bbox/area/perimeter/ratios) and
radiomic (firstorder/shape2D) features.

C0-derived signal is not forbidden -- it is opt-in. `prob` already enters the
design matrices explicitly via the M*/MCF* configs. `margin` and the internal
probabilities should likewise appear only as explicit, named feature sets, so
their contribution is measured rather than assumed.

Usage
-----
    from c2_feature_spec import C0_DERIVED_COLS

    meta_cols = {prob_col, pred_col, true_col,
                 'correct', 'path', 'cam_path', 'patient_id',
                 *C0_DERIVED_COLS}
"""

# The SDN internal-classifier taps, as attached in
# c0_custom_predictions_densenet121*.py (IC_LAYERS).
INTERNAL_CLASSIFIER_COLS = {
    "prob_denseblock1",
    "prob_transition1",
    "prob_denseblock2",
    "prob_transition2",
    "prob_denseblock3",
    "prob_transition3",
    "prob_denseblock4",
}

# |prob - threshold|, written by apply_threshold() in the C0 prediction scripts.
CONFIDENCE_COLS = {"margin"}

# Everything that is a function of C0's output rather than of the image.
C0_DERIVED_COLS = INTERNAL_CLASSIFIER_COLS | CONFIDENCE_COLS

# Dataset-specific ground-truth / grouping columns that are not C1 attributes.
# RSNA carries these; CheXpert cohorts do not. Both are strings, so they would
# also crash StandardScaler, and `label_name` is the true class (direct leakage).
RSNA_META_COLS = {"view", "label_name"}


def is_c0_derived(col: str) -> bool:
    """True if `col` is a function of the C0 model's output, not of the image."""
    return col in C0_DERIVED_COLS
