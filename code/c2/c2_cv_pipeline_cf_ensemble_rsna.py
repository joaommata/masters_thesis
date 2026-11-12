"""
c2_cv_pipeline_cf_ensemble_rsna.py

RSNA Pneumonia wrapper around c2_cv_pipeline_cf_ensemble.py.

The CF-ensemble pipeline itself is NOT reimplemented here: this module imports it
and patches the three things that are CheXpert-specific, so the modelling code,
the configs and the fold logic stay in one place and cannot drift.

What differs for RSNA
---------------------
1. Column names. The pipeline expects '{disease}_prob/_pred/_true' and renames
   'prob'/'pred'/'true' into that shape using the --disease string. RSNA's
   c2_data.csv lives at C2_custom/rsna_pneumonia/, so passing
   disease='rsna_pneumonia' makes that rename work unchanged.

2. meta_cols. Handled centrally now: the base pipeline's meta_cols includes
   c2_feature_spec.C0_DERIVED_COLS (margin + the SDN internal-classifier probs)
   and RSNA_META_COLS ('view', 'label_name'), so nothing needs patching here.
   Those RSNA columns are strings -- without the exclusion they would reach
   StandardScaler and raise "could not convert string to float" (the failure
   cam_path once caused) -- and 'label_name' is the ground-truth class, i.e.
   direct target leakage.

3. patient_id. RSNA has no patient identifiers; c1's FeatureVectorBuilder parses
   one out of the DICOM path, so it ends up as the StudyInstanceUID -- unique per
   row (verified: 5,204 unique for 5,204 rows). Since RSNA is one image per
   study, StratifiedGroupKFold on it degenerates to StratifiedKFold, which is the
   correct behaviour here (no patient-level leakage is possible). No patch needed
   -- this note exists so the grouping is not mistaken for the CheXpert leak.

Keeping in step with the base pipeline
--------------------------------------
This module forwards straight into base.run_cv, so its argument list has to
track that signature. It previously passed `extended` and the three `balance_*`
arguments, which run_cv no longer accepts -- any RSNA run would have died with a
TypeError before touching the data. `--strategy` and `--search_space` were added
to the base after the RSNA runs of 2026-07-17 and are forwarded here so the two
datasets can be launched with identical flags.

Usage (16 CFs + entropy, the config this was written for):
    python c2_cv_pipeline_cf_ensemble_rsna.py --cf_count 16 --entropy \
        --distance l1 --strategy correct_cf
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))

import c2_cv_pipeline_cf_ensemble as base
from c2_feature_spec import C0_DERIVED_COLS, RSNA_META_COLS

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="C2 CF-ensemble CV for RSNA Pneumonia (pneumonia vs rest)"
    )
    parser.add_argument("--cf_count", type=int, default=16,
                        help="Number of counterfactuals (k) averaged at test time")
    parser.add_argument("--distance", type=str, default="l1",
                        choices=["l1", "l2", "cosine"])
    parser.add_argument("--entropy", action="store_true",
                        help="ADD binary entropy H(p) to the M and MCF configs "
                             "only; baselines untouched. Results go to a "
                             "TEST_entropy_added(M-only)/ subdirectory.")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["LR", "RF", "MLP"])
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--save_folds", action="store_true")
    parser.add_argument("--strategy", type=str, default="correct_cf",
                        choices=["correct_cf", "gt_routing"],
                        help="CF routing strategy")
    parser.add_argument("--search_space", type=str, default="attr",
                        choices=["attr", "emb"],
                        help="Feature space the CF neighbour search runs in")
    args = parser.parse_args()

    print(f"Excluded from the attribute space (via c2_feature_spec):")
    print(f"  C0-derived: {sorted(C0_DERIVED_COLS)}")
    print(f"  RSNA meta : {sorted(RSNA_META_COLS)}\n")

    base.run_cv(
        # 'disease' is used both as the c2_data.csv subdirectory and as the
        # prob/pred/true column prefix -- rsna_pneumonia matches the layout
        # written by c1_match_to_new_split_rsna.py.
        disease="rsna_pneumonia",
        cf_count=args.cf_count,
        backbone="densenet",      # C2_custom
        save_fold_data=args.save_folds,
        distance=args.distance,
        entropy=args.entropy,
        models=args.models,
        configs=args.configs,
        strategy=args.strategy,
        search_space=args.search_space,
    )
