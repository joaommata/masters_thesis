# Training C2 on diffusion counterfactuals

How the generated counterfactuals in `CF_levels/` become something the attribute
ladder and the CNN ladder can train on, and how the result is made comparable
against the retrieval (KNN) arm.

---

## The problem this solves

Every stage of C2 assumes a counterfactual path resolves to a **row of
`c2_data.csv`** — for its 454 attributes, its 1024-d embedding, its C0
probability, its Grad-CAM and its pixels. That is true by construction for a
retrieved counterfactual, which *is* another real image. It is true of nothing a
diffusion model produces.

So a generated counterfactual has to be given all five of those, and then the
two arms have to be put on the same rows before any number is read off them.

---

## The two arms

| arm | `src.name` | counterfactual |
|---|---|---|
| retrieval | `knn_correct_cf_sub20k` | nearest correctly-classified train row of the opposite prediction |
| generative | `diffusion_hml_sub20k` | FastDiME trajectory from the query itself, guided to a target probability |

`src.name` is the only directory component of both the fold cache and the
results tree, so encoding the arm and the subset there means **nothing
downstream needed to change** — not the trainers, not the ensembler, not
`c2_derive_k.py`. The pre-existing full-table arm (`knn_correct_cf`) is
untouched.

```
$THESIS_RESULTS/C2_final_folds/<disease>/<src.name>/cf_<K>/
$THESIS_RESULTS/C2_final_results/<disease>/<src.name>/cf_<K>/{attributes,images,ensemble}/
```

### Slot order is `high, mid, low`

`cf_paths` is ordered, and slot 0 is what training uses. Putting the strongest
edit first makes K=1 the most informative single pairing — the generative
analogue of the *nearest* neighbour — and makes the **K=1 pairing a strict
prefix of the K=3 one**, which is what `derive_pairing` and `check_nesting`
require and what lets the K=1 numbers be reproduced exactly from the K=3 run.

The level set is deliberately *not* in `src.name`: `cf_1` and `cf_3` must share
a source name or they are two unrelated experiments as far as the fold cache and
the K derivation are concerned.

---

## The subset

Counterfactuals exist for 20,000 queries per disease. Both arms are restricted
to exactly those rows, so the counterfactual source is the only difference
between them.

The list is **frozen** at `C2_final/<disease>/subsets/20k.txt` rather than
re-derived from the CF manifest at each run, so a later top-up of the generation
campaign cannot move the subset under finished results.

Restriction happens at exactly one place — `c2_folds.load_c2_data(subset=...)`
— because that single frame determines both the fold membership *and* the KNN
retrieval pool. Subsetting one but not the other would let the retrieval arm
draw counterfactuals from rows it never trains on: both a leak and an unfair
advantage.

Measured: the subset is a plain random sample, and class balance survives it.

| disease | subset correct-rate | full | patients |
|---|---|---|---|
| atelectasis | 0.5922 | 0.5946 | 12,511 |
| cardiomegaly | 0.8102 | 0.8103 | 12,505 |
| consolidation | 0.6138 | 0.6113 | 12,605 |

Side effect: the KNN neighbour search drops from ~2.3 h per fold to **0.8 min**.

---

## Materialising a counterfactual — four stages

Run by `jobs/cf_levels_materialize.sh`, each stage resumable.

```
bsub -env "all, DISEASE=atelectasis, LEVELS=high" < jobs/cf_levels_materialize.sh
```

`LEVELS` restricts the *attribute* and *assemble* stages only. Export and C0
always cover all three levels: they are cheap, and a partial `cf_index.csv`
would make the frozen subset depend on which stage ran last.

### 1. `cf_levels_export_images.py` — CPU, ~20 min

`cf/<key>.npz` holds three float32 arrays under one key. Identity has to move
into the path, or the three intensities are indistinguishable downstream:

```
cf_levels/<disease>/<level_name>/<stem>.png      relative to $THESIS_DATA
```

Also writes `cf_index.csv` (the driver for every later stage) and freezes the
subset file.

### 2. `cf_levels_c0_predict.py` — GPU, ~15 min

Probability, 1024-d embedding and the two-directional Grad-CAM, per PNG.

**Recomputed, not repackaged.** `manifest.csv` already has a `cf_prob` and
`saliency/` already has maps, but those describe the *float tensor the sampler
produced*, while every model downstream sees a **uint8 PNG**. Recomputing here
means `cf_prob`, `cf_emb_*` and the CNN's pixels all describe the same image —
and makes `build_folds`'s `drift < 1e-6` assert worth something, since both
sides of it then come from this one table.

It also makes `cf_regen_saliency.py` a figures-only concern rather than a
blocking dependency.

`--verify-originals N` re-scores N *real* images and requires them to reproduce
`c2_data.csv` — the guard that the model, transform and logit index copied out
of `c0_final_predictions.py` are the ones that produced the baseline. Tolerance
is device-aware: the baseline was scored on GPU, and re-scoring on CPU moves a
probability by ~1e-4 through float32 reassociation alone.

Grad-CAMs land in `gradcam/CF_levels/`, which is why `CAM_SPLITS` in
`c2_train_images.py` gained a third entry. It is searched **last**, so a query
image still resolves on the first `os.path.exists`.

### 3. `cf_levels_c1_attributes.py` — GPU, ~0.24 s/image (~80 min per level)

The same 454 attributes, from the same extractors, in the same order. Sharded
parquet, atomic writes, finished shards skipped before the models load —
deliberately *not* the rolling-CSV pattern that lost progress twice.

`--calibrate N` measures the one known asymmetry: query attributes came through
`XRayResizer` (skimage, anti-aliased), counterfactual attributes come from a
224×224 PNG with no resize. `delta_*` inherits that difference on the diffusion
arm and not on the KNN arm, where both sides took the identical path.

### 4. `cf_levels_assemble.py` — CPU, minutes

Joins the three into:

- `cf_pairs.csv` — the small pairing index `pair()` and `cf_prob_lut()` read;
- `c2_cf_table.parquet` — one row per counterfactual with **exactly**
  `c2_data.csv`'s columns, in its order.

The schema assertion is the point of the script. Nothing downstream validates
the column set: `_relevant_cols` builds the attribute space by subtracting the
known meta names, so a leftover column silently becomes a 455th attribute and a
missing one silently becomes zeros — in both cases only in the arm being
measured.

---

## How a generated CF is dereferenced

Three small hooks on `CFSource`, all no-ops for a retrieval source:

| hook | retrieval | generative |
|---|---|---|
| `subset_paths()` | the frozen list | the frozen list |
| `cf_prob_lut()` | `{}` | `{cf_path: prob}` |
| `aux_table(k)` | `None` | the parquet rows for `LEVEL_ORDER[:k]` |

`build_folds` merges `cf_prob_lut` into `prob_lut`, so `_per_cf_probs` is
untouched and still raises for a genuinely unknown path. `load_folds`
concatenates `aux_table` onto the base frame before constructing the
`FoldContext` — so `FoldContext` and `materialize` needed **no changes at all**,
and the scaler is still fit on **train queries only**, which is what keeps the
two arms on one scale.

`DiffusionCFSource.attach()` raises. The inherited version builds its lookup
from `train_out` because a retrieved CF is a train row; leaving it in place
would have been a live trap.

---

## Fold safety

**Clean per row.** The counterfactual for query *x* is a function of *x*, the
frozen C0 and the frozen UNet. No other row of `c2_data.csv` participates. The
edit direction comes from C0's *prediction*, never its label — the same property
that makes `correct_cf` routing computable at test time.

**Two dependencies that are not fold-local, both reported:**

1. **The guidance targets.** `levels.json` holds the 5th/95th percentiles of
   C0's probability over the *correctly-classified rows of the whole C2 set*,
   test rows included. Six scalars per disease, identical for every query — but
   a real dependency on test-row labels.

   `cf_levels_check_targets.py` measures it: recompute the targets from each
   fold's **train partition only** and express the shift in units of the spacing
   *between* levels.

   | disease | direction | mean shift | max |
   |---|---|---|---|
   | atelectasis | to_negative | 0.008 | 0.014 |
   | atelectasis | to_positive | 0.048 | 0.063 |
   | cardiomegaly | to_negative | 0.008 | 0.017 |
   | cardiomegaly | to_positive | 0.008 | 0.010 |
   | consolidation | to_negative | 0.007 | 0.008 |
   | consolidation | to_positive | 0.076 | 0.098 |

   Worst case **0.098 level-steps**. A fold-clean run would have aimed at
   targets a tenth of one level-gap away from the ones used. Report as a stated
   limitation with this table as the evidence.

2. **The UNet's pretraining corpus.** FastDiME's checkpoint was trained on a
   CheXpert split that probably overlaps these rows. It does not distinguish
   folds — train and test queries are treated identically — but the generator
   may partly reconstruct images it has seen.

**A third limitation, on the other arm.** The KNN arm retrieves from the 20k
subset's own train partition (~16k rows) rather than the full ~68k pool, so part
of any diffusion advantage may be a smaller retrieval pool rather than a better
counterfactual.

**Definitional asymmetry, to state in the headline comparison.** A KNN test
counterfactual is a *different patient's image drawn from train*. A diffusion
test counterfactual is *a counterfactual of the test image itself*. That is the
comparison being run, not a defect — but it is why `attach()` is overridden.

---

## `cf_anchor` is N/A for the generative arm

`score_cf_anchor` scores a pair by the mean correctness of train rows that
retrieved the **same** counterfactual. Every generated counterfactual is unique
to its query, so no test CF is ever in the train dict, every score falls back to
0.0, and `roc_auc_score` returns exactly 0.5 — for every fold, silently, looking
like a measured null result. It is now detected and reported as unavailable.

---

## Running it

```bash
# 1. materialise (per disease). LEVELS=high is enough for K=1.
bsub -env "all, DISEASE=atelectasis, LEVELS=high" < jobs/cf_levels_materialize.sh

# 2. the retrieval arm: pair once at K=3, derive K=1 in seconds
bsub -env "all, DISEASE=atelectasis, CF_COUNT=3, SUBSET=20k" < jobs/c2_folds.sh
bsub -env "all, DISEASE=atelectasis, CF_COUNT=1, SUBSET=20k, DERIVE=1" < jobs/c2_folds.sh

# 3. both ladders, both arms, K=1
CF_SOURCE=knn       SUBSET=20k CF_COUNT=1 ./jobs/submit_chain.sh atelectasis
CF_SOURCE=diffusion SUBSET=20k CF_COUNT=1 ./jobs/submit_chain.sh atelectasis

# 4. check the arms are comparable, then read the scores
python code/c2_compare_arms.py --disease atelectasis --cf-count 1 --scores
```

`c2_compare_arms.py` asserts, before any number is read, that the two arms score
the **identical `(split, path)` index** fold by fold, that slot 0 really is the
`high` level, that every retrieved CF is a train row of its own fold, and that
every generated CF belongs to its own query.

For K=3: rerun stage 1 without `LEVELS`, re-run `cf_levels_assemble.py`, then
the chains at `CF_COUNT=3`. The CNN can reuse the K=1 weights via `EVAL_FROM`,
since training is K-independent.

---

## Out of scope for this pass

`bootstrapping.ipynb` and `C2_evaluation.ipynb` each hold the arm in a single
constant (`SOURCE, POLICY = "knn_correct_cf", "ignore"`). Turning that into a
source *axis* is a follow-up once the results exist.
