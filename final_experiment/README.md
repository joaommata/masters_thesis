# final_experiment

The clean rebuild (2026-09-04). Everything the final pipeline needs lives here:
split definitions, the scripts that consume them, the LSF jobs that run those
scripts, and links to the results they produced.

Nothing in here reads the pre-rebuild splits. Anything that does is legacy and
lives under `../code/`, `../jobs/` and `/work3/s251710/thesis_results/C0_custom/`.

## Layout

```
C0_train.csv, C0_checkpoint_selection.csv,   split definitions — scripts locate
C2_dataset.csv, Original_Test.csv            them via REPO/final_experiment/
C0_custom_split.ipynb                        builds the four CSVs above
c0_analyse_predictions.ipynb                 analysis
code/                                        the finalized scripts
jobs/                                        the LSF jobs that run them
results/                                     symlinks -> /work3 (bytes are too big for the repo)
figures/
```

Both notebooks stay at the top level on purpose: `C0_custom_split.ipynb` writes
its CSVs with **relative** paths, so re-running it from a subdirectory would
scatter them.

## The split

Patient-level, frontal only, seed 42. Blanks were filled with `0.0` upstream, so
the CSVs hold only `1 / 0 / -1` and contain no NaN.

| file | images | patients | role |
|---|---|---|---|
| `C0_train.csv` | 85,664 | 29,040 | fit C0 |
| `C0_checkpoint_selection.csv` | 9,528 | 3,227 | early stopping + best checkpoint |
| `C2_dataset.csv` | 95,835 | 32,267 | downstream meta-classifier; C0 never sees it |
| `Original_Test.csv` | 202 | 200 | official CheXpert valid; scored once |

Verified 0 patient and 0 path overlap across all six pairs. This is the fix for
the earlier patient leak, applied at the source rather than in the CV splitter.
`c0_train_multilabel_final.py` re-asserts train/selection disjointness at
startup, so a future notebook edit fails loudly instead of leaking silently.

## Uncertainty policy: U-Ignore

`-1` is masked **per label**: an uncertain cell contributes zero gradient for
that label while the row still trains the other 13. You lose cells, not rows.

Chosen because the previous `hybrid` run's two worst columns were exactly the two
labels carrying the most `-1`s, each given a policy that injects wrong targets —
Atelectasis (U-Ones roughly doubled the positive set: 14,975 uncertain vs 14,765
real positives) and Consolidation (U-Zeros buried ~2/3 of mentions as negative:
12,232 uncertain vs 6,496 positives). Masking cannot create that failure mode and
keeps the sigmoid calibrated, which matters because C2 consumes those
probabilities as features.

AUC is always computed on confidently-labelled rows only, under **every** policy,
so the five policies are directly comparable and reported discrimination never
depends on the choice.

## Pipeline — four stages

Two of the four run **once in total**; two run **once per disease**. C1
attributes describe the *image* (demographics, per-anatomy geometry, radiomics),
not the question asked about it, so the same 95,825-row attribute table is reused
by every disease. Only the C0 half changes.

| # | stage | scope | command | status |
|---|---|---|---|---|
| 1 | train C0 | **once** | `bsub < final_experiment/jobs/train_c0_multilabel_final.sh` | done (U-Ignore) |
| 2 | C0 predictions | per disease | `DISEASE=x bsub < final_experiment/jobs/c0_final_predictions.sh` | effusion done |
| 3 | C1 attributes | **once** | `bsub < final_experiment/jobs/build_c1_c2_dataset.sh` | done |
| 4 | build `c2_data.csv` | per disease | `python final_experiment/code/c2_build_dataset.py --disease x` | effusion done |

### Adding a disease

Stages 2 and 4 are chained in one job, so a new disease is a single submit:

```bash
DISEASE=pneumothorax bsub < final_experiment/jobs/run_disease.sh
DISEASE=cardiomegaly bsub < final_experiment/jobs/run_disease.sh
```

Available: `effusion pneumothorax cardiomegaly atelectasis consolidation edema
pneumonia`. It skips stage 2 if the predictions already exist (`FORCE=1` to
redo), then always rebuilds `c2_data.csv`.

### Then train C2 (legacy path)

```bash
python code/c2/c2_cv_pipeline_new_split.py --backbone final --disease pneumothorax
```

`--backbone final` resolves to `C2_final/`, registered in `BACKBONE_MAP` across
the c2 pipelines. It is a **new** tree: the legacy `C2_custom/` results are left
untouched, so old and new never overwrite each other.

### What stage 4 produces

`$THESIS_RESULTS/C2_final/<disease>/c2_data.csv` — 95,825 rows x 1,486 columns:

```
path, prob, true, cam_path, pred, correct, margin, patient_id,   ← 8 meta
<454 attribute columns>,                                          ← the feature space
emb_0 .. emb_1023                                                 ← 1024 C0 embeddings
```

The 454 attribute columns are asserted identical, in the same order, to the
legacy `C2_custom/effusion/c2_data.csv` attribute space, so the existing C2
pipelines run unchanged. The schema is the legacy one minus the 7 SDN
`prob_<layer>` columns, which were dropped from the project — harmless, since
`c2_feature_spec.C0_DERIVED_COLS` is only ever subtracted to form `meta_cols`.

### Leakage guards in stage 4

The pipelines build the attribute space by **subtraction**: anything not named in
`meta_cols` silently becomes a feature. Four groups are dropped and then asserted
absent from the output:

- `label_raw` — the raw 1/0/-1 CheXpert value for the disease. It IS the target,
  and it is **not** in `meta_cols`, so it would otherwise become a feature and
  every attribute config would score near 1.0.
- `certain` — same origin, same problem.
- the 14 CheXpert observation columns on the attribute side — one is the ground
  truth, the other 13 are radiologist labels rather than image features.
- `Sex`, `Frontal/Lateral`, `AP/PA` — strings that crash `StandardScaler`. Their
  numeric forms (`age_pred`, `sex_male`, `sex_female`) are inside the 454 and stay.

## More than one counterfactual per query (K)

K is the number of counterfactuals retrieved for each query. Everything so far
ran at K=1. K=5 and K=10 are the next axis, and the pipeline is set up so that
adding them costs one pairing job, not one per K.

### Why K is nearly free

`cf_paths` is written **nearest first**, and nothing else about a fold depends on
K: the split is seeded (42), the scaler is fit on the same train rows and the
same columns, and the CF pools are routed by prediction. So the K=5 pairing is
literally the first five entries of the K=10 pairing, and K=1 is the first one.

`c2_folds.py --derive` exploits that: it truncates an existing larger-K pairing
and re-averages `cf_prob` from the per-CF probabilities. Seconds, against the
~1.7h the neighbour search costs (measured ~21 min per fold at K=10).

**One exception, and it is benign: exact distance ties.** In 461 standardised
dimensions two training images occasionally sit at *identically* the same L1
distance from a query, and scikit-learn breaks that tie differently at
`n_neighbors=1` than at `n_neighbors=10`, because `n_neighbors` feeds its
algorithm choice. Measured on the real folds: 1 to 3 rows per fold out of ~90,000
(at most 0.012%). `--check-nesting` does not wave these through on a tolerance.
It recomputes the distance for every differing row and passes only if the two
counterfactuals are genuinely equidistant, in which case either is equally
correct. Anything at a different distance fails loudly.

```bash
# pair ONCE at the largest K
DISEASE=effusion CF_COUNT=10 bsub < final_experiment/jobs/c2_folds.sh

# every smaller K falls out of it
DISEASE=effusion CF_COUNT=5 DERIVE=1 bsub < final_experiment/jobs/c2_folds.sh
```

`c2_folds.sh` then runs `--check-nesting 1` automatically: it asserts the K=1
pairing already on disk — the one **every current result was produced from** — is
exactly the prefix of the new one. That is what makes the K sweep one experiment
rather than three runs that share a directory name. `load_folds` also derives on
demand, so a trainer asked for a K it can slice will never re-run the search.

### Running a K: one pass, everything else derived

K only enters at inference. Both ladders train on the nearest counterfactual, so
the model is the same object at K=1, K=5 and K=10, and the score it gives to
(query, CF_j) does not depend on how many counterfactuals the run was configured
with. That makes the sweep one pass at the largest K plus arithmetic.

```bash
# 1. the only genuinely new computation: neighbours 2..10
DISEASE=effusion CF_COUNT=10 bsub < final_experiment/jobs/c2_folds.sh

# 2. one pass at K=10, then K=5 and K=1 are derived from it
WAIT_ON="done(<folds jobid>)" ./final_experiment/jobs/submit_k_sweep.sh effusion
```

What each step actually costs:

| step | cost | why |
|---|---|---|
| folds at K=10 | ~11.5h CPU per disease | the neighbour search really does have to find CFs 2..10 |
| attributes at K=10 | ~5h CPU per disease | refit. Identical to the K=1 fit, but the K=1 run did not persist its sklearn models (`--save-models` was off), so there is no object to load |
| images at K=10 | inference only | **no retraining.** `--eval-from` loads the K=1 checkpoints, which already are the K=10 models |
| K=5, K=1 | seconds | `c2_derive_k.py` averages a prefix of the stored per-slot probabilities |

Both trainers write `fold_<i>_slot_probs.npz`, holding the score for every
(query, CF_j) pair. Because slot j is K-independent, `mean(slots 0..4)` **is** the
K=5 prediction, exactly, with no model refit. Stored as float64 on purpose:
Random Forest probabilities are multiples of 1/200 and full of exact ties, and
float32 rounding reorders them enough to move the derived AUC in the fourth
decimal.

Deriving K=1 and comparing it against the K=1 results already on disk is the
audit that ties the sweep to everything published so far:

```bash
DISEASE=effusion FROM_K=10 TO_K=1 CHECK=1 bsub < final_experiment/jobs/c2_derive_k.sh
bsub < final_experiment/jobs/check_nesting.sh   # the pairing-level audit
```

Reusing the K=1 checkpoints also fixes model selection across K, which is what
makes the comparison clean: the only thing that moves between K=1 and K=10 is how
many counterfactuals a fixed model is scored against.

`CF_COUNT` is in every output path (`.../<source>/cf_<K>/...`) and in the job
names (`at_effusion_k10`), so two K for one disease never collide in the results
tree or in `bjobs`.

### How K is combined: score ensembling, in both ladders

One model, K predictions, averaged. Never an average of features.

    train   each query is paired with its NEAREST counterfactual only. One row
            per query, one model fit. Training cost is flat in K.
    test    the fitted model is applied K times, once per (query, CF_j) pair,
            and the K predicted probabilities are averaged into one score per
            query.

The attribute ladder and the CNN ladder do exactly this, which is what makes
their numbers comparable at every K. In the CNN it is `_ensemble_probs`, the mean
of the K sigmoids. In the attribute ladder it is the mean of the K
`predict_proba` outputs.

**Averaging features instead would be a different and worse experiment.** A mean
attribute vector over ten neighbours is a smoother, more central point than any
real counterfactual, so `delta_*` would shrink toward the pool mean as K grows
and the model would see *less* disagreement rather than more. Ensembling keeps
every pair intact and combines ten opinions instead of ten inputs. The earlier
draft of this pipeline averaged features; it does not any more.

| stage | at K > 1 | cost |
|---|---|---|
| folds | `cf_paths` holds K paths; `cf_probs` holds their K probabilities | one build, then free |
| attributes | one row per (query, CF_j); K predictions averaged per query | fit once, predict K times |
| CNN | each (query, CF_j) pair scored separately, K sigmoids averaged | train once, eval K times |
| ensemble | unchanged, both sides are already one score per query | unchanged |

The feature space is identical at every K, one counterfactual per row always, so
no config changes width and the ladder is unchanged. What changes is how many
predictions get averaged.

Configs and models with no counterfactual input are K-invariant and are evaluated
once: B1 through B5 on the attribute side (reported as `k_used=1` in
`cv_summary.csv`), `xi` and `xi_sal` on the CNN side (copied from the K=1 run
with a `reused_from.json` marker). That is a claim and not just a saving: the
baselines do not move with K, which is what lets a difference across K be read as
coming from the counterfactual side alone.

### CNN cost control

Three things keep the CNN ladder affordable at K=10 rather than 10x:

- **`--train-k` (default 1).** Training uses the nearest CF only; evaluation uses
  all K. Cost per epoch is flat in K. `--train-k 0` trains on all K, but it
  breaks the symmetry with the attribute ladder, which cannot train on an
  ensembled probability the way the CNN loss can. Diagnostic only.
- **`--eval-batch-size` defaults to `--batch-size // K`.** Peak GPU memory is
  (batch x CFs x encoders) images, so a fixed batch at K=10 OOMs the
  quad-encoder configs immediately. Inference batches are independent, so this
  changes throughput and not predictions.
- **`xi` and `xi_sal` are copied, not retrained.** Neither opens a counterfactual
  image, so on identical folds their result is identical at every K. The copy
  drops a `reused_from.json` next to it so this is visible later.
  `--no-reuse-k-invariant` retrains them.

### Memory

Each frame holds one counterfactual per row, so a frame is the same size at K=10
as at K=1. What grows is the *number* of test frames: the attribute trainer keeps
all K per fold (~0.5 GB each) because every one is revisited once per (config,
model), and it holds K design matrices for the config it is currently fitting.

Measured 11.3 GB for one fold's materialisation at K=1. The peak is the widest
config, MCF5 at ~4,460 columns: roughly 20 GB at K=10, against the ~14 GB the
K=1 run reached. Both fit the 8-slot reservation (64 GB) the jobs already ask
for. The change that pays for this is building design matrices one config at a
time instead of all seventeen at once, which the previous version did.

### Pool sizes

The CF pool is one prediction class of the train fold, so K cannot exceed it. The
smallest pool across the five diseases is consolidation's TP pool at ~3,880 rows
per fold, so K=10 has ample room. `NearestNeighbors` raises if it does not.

## Where the counterfactual columns come from

`c2_data.csv` carries no `delta_*` or `cf_*` columns — neither did the legacy
table. They are not stored anywhere: `c2_folds.py` caches only the CF *pairing*
(~8 MB per disease at K=1) and `materialize_fold` rebuilds the ~3,400-column
frame from it in about 40 seconds. Storing it instead would cost ~12 GB per
(disease, source, K).

KNN counterfactuals on this split are done for all five diseases at K=1; the
generated (diffusion) source is still the placeholder described in
`c2_cf_sources.DiffusionCFSource`.

## Where results actually live

The bytes stay on work3 (`C2_dataset_c0_effusion.csv` alone is 1.7 GB) and are
reachable through `results/`:

- `results/C0_final` -> `/work3/s251710/thesis_results/C0_final`
- `results/C1_attributes` -> `/work3/s251710/thesis_results/C1_attributes/final_experiment`

The prediction and assembly scripts symlink their own output back into
`results/` automatically (`--link-into`), which is where `C0_ignore_effusion`
and `C2_dataset_attributes.csv` come from.

## Results so far

**C0, U-Ignore** (job 29335635) — best epoch 3, selection mean competition AUC
**0.8116**, `Original_Test` **0.8504**. Beats the pre-rebuild hybrid run (0.8091)
but only slightly: Atelectasis 0.706 -> 0.712, Consolidation 0.741 -> 0.744.
Selection loss turns upward after epoch 3, same overfitting shape as before —
`--augment` (light rotation/scale jitter, off by default to match Irvin et al.)
is the first thing to try.

**Effusion predictions** (job 29338668) — AUC 0.8819 on `C0_checkpoint_selection`
(Youden threshold 0.3932), 0.8842 on `C2_dataset`, 0.9091 on `Original_Test`.

Predictions are per-disease because C0 is multi-label (14 logits) but C1/C2 are
single-disease: they need one `prob`, one `true`, one Grad-CAM and one operating
threshold. `c0_final_predictions.py` slices one observation out of the 14 by name
(effusion = logit 10, read from `labels.json`) and backprops **only** that logit
for Grad-CAM. All 14 probabilities are still written to `all_probs_<split>.csv`.

## External dependencies (2)

1. `jobs/stage_chexpert.sh` is a **copy** of `../../jobs/stage_chexpert.sh`,
   brought in because `jobs/` is gitignored and this folder should be
   self-contained under version control. The legacy copy still serves the
   pre-rebuild jobs; keep the two in sync only if you change staging.

2. `code/c1_build_attribute_vector_subset.py` imports `FeatureVectorBuilder`
   from `../../code/c1/c1_build_attribute_vector.py` via a `sys.path` insert.
   That module is deliberately **not** copied: it is the single canonical
   attribute builder shared with four other c1 scripts, and duplicating it would
   fork the feature schema.
