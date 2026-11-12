# Pre-submission review: results, headline claims, bootstrapping

Scope, as agreed: (1) results numbers in Chapter 4 against Appendix B/C and the result files; (2) the main claims in the Abstract and the Conclusion; (3) how the thesis describes the bootstrap against what the code does. The Chapter 3 methods-vs-code audit (C0/C1/C2 hyperparameters, CF generation) was **not** done.

Paths: thesis = `MSc_Dissertation-13/Chapters/`; results = `/work3/s251710/thesis_results/C2_final_results/` (`R/` below); generated tables = `final_experiment/tables/`. All recomputed values come from read-only pandas reads of those CSVs.

---

## Summary

| # | Severity | Where | Issue |
|---|---|---|---|
| 1 | HIGH | Ch4:396 vs Ch4:367, Abstract:9 | "None of the baselines beat … the adjusted MSP": ConfidNet beats it significantly on effusion (AUROC and E-AURC) |
| 2 | LOW (corrected) | AppB:217 vs AppB:233-234 | Caption's "14k–20k" is a log-interpolated crossing, the table's 16k/32k are grid values; both are correct but the caption doesn't say it interpolates |
| 3 | MEDIUM | Abstract:9, Ch4:417/427/532 vs Ch4:450, AppC table, fig:xai-groups | Best cardiomegaly group is "35 %" in some places and "37 %" in others; effusion is "5–42 %" vs "6.0–42.9 %" |
| 4 | MEDIUM | Discussion:91 (Conclusion) | "error rates differ significantly": no test of group error rates exists in the code |
| 5 | MEDIUM | Ch4:367 | ConfidNet effusion gain given as +0.003; data say +0.0037 (rounds to +0.004) |
| 6 | MEDIUM | Ch4:84, 86, 293 | "+0.004 (effusion) to +0.011 (atelectasis)": the unrounded gains are +0.0046 and +0.0116, i.e. +0.005 to +0.012 |
| 7 | MEDIUM | Ch4:191, 274 | "With K=3 (K≫1) retrieved beats generated": that comparison was never tested; ensemble gap is 0.001 |
| 8 | MEDIUM | Abstract:9, Ch4:332 | ViT-B/16 transfer described as "not fully"/"partially"; the ensemble is significantly *worse* than adjusted MSP on 2/5 ViT tasks |
| 9 | MEDIUM | Ch4:325, AppB:233 | "needs 70,000 to 80,000 images" is the full pool ("all"), not a measured threshold; thesis table dropped the "all" label the generated table has |
| 10 | MEDIUM | Ch3:682 (bootstrap) | Bootstrap holds the trained models fixed (test-set resampling only); not stated |
| 11 | LOW | Ch3:682 + every table | p-value floor 1/5000 × Holm(5) ⇒ p_adj ≥ 0.001, so per-disease cells can never get `***`; not explained |
| 12 | LOW | AppB:10 vs `bootstrapping.ipynb` | "Mean rows … tested once": in the main bootstrap the mean rows are Holm-adjusted together (9 claims); in the arch/baseline scripts they are not adjusted |
| 13 | LOW | Ch3:682 vs tables | Text says it reports the bootstrap *mean* of δ; tables and prose report point estimates (e.g. 0.0049 vs bootstrap mean 0.0048) |
| 14 | LOW | Ch3:684 | "the patient-level bootstrap described below" – it is described in the paragraph above |
| 15 | LOW | Ch4:351 | "significantly matches threshold-adjusted MSP": should be "beats" |
| 16 | LOW | Ch4:125 | Result box: "does not improve either family significantly"; Ch4 text and Table prob_knn show a significant CNN gain on atelectasis (+0.0027**) |
| 17 | LOW | Ch4:191 / 198 | "diffusion consistently beats NUNs": effusion CNN is −0.0001 and 3/15 comparisons are n.s. |
| 18 | LOW | Ch4:351, 393 | E-AURC reduction "2.4 %": unrounded is 0.001071 / 0.043458 = 2.46 % (→ 2.5 %) |
| 19 | LOW | AppB:233 vs AppB tab:diffusion_vs_knn | Same diffusion attribute model at K=1 reported as 0.910 (cardiomegaly) in one table and 0.913 in another (pooled-OOF vs fold-mean AUROC); not explained |
| 20 | LOW | Ch4:214 area (trajectory) | Same sentence repeated twice ("All trajectory models remain below …") |

---

## 1. Results: thesis ↔ appendix ↔ data

### Table-level check
Every number in each thesis table was diffed against the generated `.tex` from `TABLES.ipynb`:

- **Identical numbers:** `tab:c0_split` (Ch4), `tab:attr_knn_k`, `tab:cnn_knn_k` (AppB).
- **Identical apart from layout:** `tab:prob_knn`, `tab:ens_knn`, `tab:diffusion_vs_knn`, `tab:k_arm_sub20k`, `tab:baselines_knn`, `tab:c0_arch`, `tab:k_diffusion_sub20k`. Layout means column spacing, the Δ columns dropped in the Ch4 copy, and row order. Labels stay with their rows.
- **No generated source:** `tab:risk80`, checked arithmetically below.
- **Different content:** `tab:top_attributes_cardiomegaly_effusion` / `tab:attr_group_profiles` / `tab:sal_group_profiles` (AppC) are a hand-merged layout. Their group sizes and error rates match the group-profile figures (see finding 3), not the generated `.tex`.

Stars were re-derived from `R/bootstrap_deltas.csv`, `R/bootstrap_diffusion_k_deltas.csv`, `R/bootstrap_confidnet_vs_msp.csv` and `C2_arch_compare/deltas.csv`. They match every table checked: prob_knn, ens_knn, k_arm_sub20k, k_diffusion_sub20k, baselines_knn, c0_arch.

### Finding 1 – HIGH – "None of the baselines beat the adjusted MSP"
- Thesis: `Chapter_4.tex:396` "None of the baseline beat our model or the adjusted \ac{MSP} … [ConfidNet] falls significantly below adjusted MSP on every task except effusion". `Chapter_4.tex:367` itself says "ConfidNet achieves a small but significant gain of +0.003** AUROC in effusion only". Abstract `:9`: "our ensemble was the only method to beat it".
- Evidence: `R/bootstrap_confidnet_vs_msp.csv`, effusion rows:
  - AUROC: ConfidNet 0.7854 vs MSP 0.7817, δ = +0.0037, p_adj = 0.001.
  - E-AURC: δ = +0.0014 better, p_adj = 0.001.
  - AppB `tab:baselines_knn` row Effusion also shows ConfidNet 0.785 > adjusted MSP 0.782.
- Fix: say "the only method to beat it **on average**" (abstract and Ch4:332 result box), and change Ch4:396 to "none beat our model, and only ConfidNet on effusion beat adjusted MSP". The ensemble does beat ConfidNet on effusion (`R/bootstrap_ensemble_vs_confidnet.csv`: +0.0020, p_adj 0.011), so the first half of Ch4:396 is fine.

### Finding 2 – LOW (corrected after re-check) – K=20 crossing point
- Thesis: `AppendixB.tex:217` "With K=20, retrieval matches diffusion at 14k–20k rows (dotted line)". Table `AppendixB.tex:229-234` gives 16k/32k at K=20.
- Evidence: `SubsetsC2.ipynb` cell 7 draws the dotted line at a crossing **interpolated on log N** between grid points.
  - Recomputed from `sweep_pooled.csv`: K=20 crosses at 14.2k (atelectasis), 15.2k (cardiomegaly), 20.3k (consolidation), 18.6k (edema), 15.4k (effusion). K=20 crosses first on every disease.
  - The table instead reports the first grid value reaching diffusion.
  - Both are correct; the caption should say it interpolates. A rewritten caption was supplied in chat.
- Earlier version of this finding (HIGH, "caption wrong") was a mistake on my part.

### Finding 5 – MEDIUM – ConfidNet effusion gain
- Thesis: `Chapter_4.tex:367` "+0.003**".
- Evidence: `R/bootstrap_confidnet_vs_msp.csv` effusion AUROC: gain 0.0037 (point) and δ 0.0037 (bootstrap). Both round to **+0.004**.

### Finding 6 – MEDIUM – Attribute I+C − I range
- Thesis: `Chapter_4.tex:84` and `:86` "+0.004 in effusion to +0.011 in atelectasis"; `:293` "at most +0.011".
- Evidence, recomputed as mean-fold AUROC from `R/<d>/knn_correct_cf/cf_10/attributes/fold_*_predictions.csv`:
  - effusion I 0.7786 → I+C 0.7833 = **+0.0046**
  - atelectasis 0.8496 → 0.8612 = **+0.0116**
  - Bootstrap means agree (`R/bootstrap_deltas.csv`: 0.0046, 0.0116).
- The text subtracted the rounded table values. It should read +0.005 to +0.012.

### Finding 7 – MEDIUM – "With K≫1 retrieved beat generated" is untested
- Thesis: `Chapter_4.tex:191` (result box) "with K=3 for both sources, NUNs beat diffusion for all three models"; `:274` "(3) With K>>1, retrieved beat generated".
- Evidence:
  - `tab:k_arm_sub20k` (AppB:183-184) Mean rows at K=3: attribute 0.859 vs 0.851; CNN 0.856 vs 0.854; ensemble **0.863 vs 0.862**.
  - No NUN-vs-diffusion test exists at K=3. `bootstrap_diffusion_deltas.csv` and `bootstrap_diffusion_k_deltas.csv` only hold K3−K1 within each source and NUN-vs-diffusion at K=1.
  - Per disease, diffusion is ahead at K=3 for edema CNN (0.823 vs 0.820) and tied for atelectasis CNN.
  - Also, "K≫1" is only K=3 on the 20k subset.
- Fix: soften the wording ("on average, without a direct test"), or add the comparison to the bootstrap.

### Finding 9 – MEDIUM – "needs 70,000–80,000 images"
- Thesis: `Chapter_4.tex:325` "With one NUN, the attribute model needs 70,000 to 80,000 training images to match diffusion". `AppendixB.tex:229-234` prints "79k", "78k" … as if they were measured thresholds.
- Evidence:
  - `tables/subsets_knn_nstar.tex` prints "all (79k)" etc. Its caption defines "all" as the full pool.
  - The grid jumps from 32k to all (`sweep_pooled.csv`). K=1 fails at 32k on every disease (e.g. cardiomegaly 0.9061 < 0.9101) and passes at all.
  - So the true N* lies between 32k and 70–80k. The figure's own log-interpolation puts the K=1 crossing at 34.6k / 55.7k / 42.8k / 46.3k / 34.0k (atelectasis / cardiomegaly / consolidation / edema / effusion), roughly half the "70,000 to 80,000" in Ch4:325.
- Fix: restore "all (…)" in the table and write "more than 32k (reached only with the full pool)".

### Finding 3 – MEDIUM – Aim 3 group error rates reported two ways
- Thesis:
  - Abstract `:9` "ranged from 35 % to 78 % wrong"; `Chapter_4.tex:417`, `:427`, `:532` "78 % … 35 %"; effusion "5 % to 42 %".
  - But `Chapter_4.tex:450` "In the best group A (37 % wrong)", and the AppC table (`AppendixC.tex` ~l.44) has cardiomegaly positive A 37.1 %, B 69.3 %, C 78.3 % and effusion A 6.0 %, C 42.9 %.
- Evidence:
  - `Images/Results/xai/Embeddings_C2.pdf` (fig:xai-clusters) corner box: A 35 %, B 69 %, C 78 %. Per `attribute_explainability.ipynb`, these are error rates on the **held-out half** of fold 4.
  - `cardiomegaly_knn_correct_cf_embedding_group_profiles.pdf` (fig:xai-groups), extracted with `pdftotext`: "n 1,244 · 37 % wrong", "69 %", "78 %". Effusion: 6 % / 19 % / 43 %. These match AppC.
  - So both numbers exist, but the text uses them interchangeably for the same group A.
- Fix: pick one (the held-out values are the more defensible), or say which half each number comes from. The effusion "42 %" against the profile's 42.9 % → 43 % has the same cause.

### Finding 16 – LOW – Probability result box
- `Chapter_4.tex:125` "does not improve either family significantly". `R/bootstrap_deltas.csv`, "CNN dual_sal_scalars − dual_sal", atelectasis: +0.0027, p_adj 0.001.
- The text at `:131` and AppB `tab:prob_knn` already report this as **. Note too that the attribute Mean-row change is significantly negative (−0.0004, p_adj 0.0056), which no table shows.

### Finding 17 – LOW – "consistently"
- `Chapter_4.tex:191` "With a single counterfactual, diffusion consistently beats NUNs, by +0.003/+0.012".
- `tab:diffusion_vs_knn`: effusion CNN −0.0001, and 3/15 n.s. The text at `:198` describes this correctly.

### Finding 19 – LOW – Two AUROCs for the same diffusion model
- `AppendixB.tex:229-234` "Diffusion AUROC" column (0.854/0.910/0.941/0.824/0.771) vs `tab:diffusion_vs_knn` Diff. attribute column (0.855/0.913/0.940/0.827/0.772).
- The first is pooled out-of-fold AUROC averaged over 5 seeds (`sweep_pooled.csv`); the second is the mean over folds. One sentence in the note would prevent an examiner question.

### Finding 18 – LOW – E-AURC percentage
- `Chapter_4.tex:351` and `:393` "2.4 %". From `R/bootstrap_baselines_summary.csv`: mean MSP E-AURC 0.043458, ensemble 0.042387, giving a reduction of 0.001071, which is **2.46 %** (→ 2.5 %).

### Risk at 80 % coverage (`tab:risk80`, no generated source)
- The C0 error column matches 1 − Acc in `tab:c0_arch`: 40.5 / 19.0 / 38.9 / 25.2 / 20.6.
- "Extra errors per 1,000" is consistent with (risk_MSP − risk_ens) × 0.8 × 10. Atelectasis gives −0.72 against a printed −0.8, which is within the rounding of the risk columns.
- **Recomputed and verified (follow-up)** from per-sample scores (ensemble `oof_predictions.csv` with the per-disease combiner, adjusted MSP from `c2_msp.msp_score(..., 'threshold')`, top 80 % per fold, averaged over folds). Every risk value and every "extra errors" value matches (−0.77 / 0.47 / −0.67 / 1.68 / 2.21, mean 0.58). The C0 error column differs only in the 2nd decimal for cardiomegaly (18.98 vs 18.97) and consolidation (38.88 vs 38.87).

---

## 2. Abstract and Conclusion claims

| Claim | Location | Verdict | Evidence |
|---|---|---|---|
| "Threshold-adjusting MSP raised its error-detection AUROC by +0.022" | Abstract:9 | ✔ | `R/bootstrap_deltas.csv`: (best−plain MSP) − (best−adj. MSP), mean = 0.0247 − 0.0030 = 0.0217. Per disease, +0.2 % (effusion) to +5.5 % (atelectasis) as in Ch4:340 ✔ |
| "ensemble … beat it: by +0.003 on average, significantly on three of five findings" | Abstract:9, Discussion:24, Conclusion:91 | ✔ numbers | AVERAGE δ = 0.0030 (p_adj 0.0018). Effusion, cardiomegaly and edema have p_adj 0.001; atelectasis (−0.0007, 0.31) and consolidation (−0.0007, 0.29) are n.s. |
| "our ensemble was the only method to beat it" | Abstract:9 | ✘ as worded | Finding 1: ConfidNet beats it significantly on effusion. True only "on average" |
| "Counterfactuals added signal over the input image-only on every finding" | Abstract:9 | ✔ | I+C−I and CNN dual−xi: all 5 diseases p_adj ≤ 0.001 (`R/bootstrap_deltas.csv`) |
| "a single generated counterfactual carrying more signal than a single retrieved one" | Abstract:9 | ✔ with a caveat | 12/15 significant (`tab:diffusion_vs_knn`). Holds only at equal training size (16k). With the full pool, one NUN beats diffusion (`sweep_pooled.csv`, K=1, all). Also, the diffusion ensemble never beats adjusted MSP on the subset and is significantly worse on atelectasis/consolidation (`R/bootstrap_diffusion_deltas.csv`, "best overall − MSP"). Neither point is in the abstract |
| "This gain held for ResNet-50 but not fully for ViT-B/16" | Abstract:9 | ⚠ understated (Finding 8) | `C2_arch_compare/deltas.csv` via `tab:c0_arch`: ViT mean −0.0004 n.s.; significantly worse on atelectasis (−0.0028) and consolidation (−0.0046). ResNet: +0.0061***, 4/5 significant ✔ |
| "Clusters of positive cardiomegaly predictions ranged from 35 % to 78 % wrong" | Abstract:9 | ⚠ | Finding 3: 35 % (held-out half) vs 37.1 % (profile/AppC) |
| "most trusted cluster: hearts enlarged and the CF shrank them; least trusted: near-normal and barely changed" | Abstract:9 | ✔ | AppC table, heart pixel surface: group A I +1.13 → C +0.28 (Δ −0.85); group C +0.10 → −0.22 (Δ −0.32) |
| "Effusion showed the same pattern for right-lung intensity" | Abstract:9 | ✔ (roughly) | Right-lung skewness: A −0.59 → +0.01 (Δ +0.60); C −0.13 → +0.10 (Δ +0.23) |
| "groups whose error rates differ **significantly**" | Conclusion, Discussion:91 | ✘ not verifiable | No significance test of group error rates in `attribute_explainability.ipynb`, `saliency_explainability.ipynb` or `case_card_combined.ipynb`. The only related note (attribute notebook) says \|r\| is "an effect size, not a test". Either drop "significantly" or add a χ² test |
| "beats … threshold-adjusted MSP on three of five CheXpert tasks, by a small but significant margin" | Discussion:24 | ✔ | as above |
| Portuguese abstract | Abstract-PT-Resumo | ✔ | Same numbers as the English (461, +0,022, 0,003, 35 %, 78 %), so Findings 1/3/8 apply to it too |

---

## 3. Bootstrapping: thesis description vs code

Thesis: `Chapter_3.tex:677-686` (sec:stats), `AppendixB.tex:10`, and the figure/table notes. Code: `final_experiment/bootstrapping.ipynb` (cells 9, 11), `final_experiment/code/c2_boot_extra.py`, and `c2_{confidnet,trustscore,scce,arch}_compare.py`. Metadata: `R/bootstrap_meta.json`, `R/bootstrap_diffusion_meta.json`, `C2_arch_compare/run_config.json`.

### Matches the code ✔
- **5,000 replicates, seed 0.** All three metadata files say `n_boot: 5000, seed: 0`.
- **Patient-level, paired.** `cmp_bootstrap` / `extra_bootstrap` draw patients with replacement (`rng.integers` → `np.bincount` gives counts used as row weights). Every model column is scored with the same weights, so the comparisons are paired. The baseline and arch scripts **replay the same draws** (asserted: `c2_confidnet_compare.py`, `c2_boot_extra.py`; replay check in `bootstrapping.ipynb` cell 20).
- **Metric = mean over the five folds.** `FoldMeanAUC` scores each fold with weights and averages.
- **95 % percentile interval; two-sided p = 2·min{P(δ≤0), P(δ≥0)}, floored at 1/5000.** `summarise()` in cell 11 and `c2_boot_extra.summarise`.
- **Holm–Bonferroni.** `holm()`. Families: "claims" are Holm-adjusted across the 5 diseases per comparison. The ensemble-combiner figure is one family of 15, as `Chapter_4.tex:176` and AppB:134 state; confirmed by p_adj = 15 × 0.0002 = 0.003.
- **E-AURC uses the same draws** (`bootstrap_eaurc_deltas.csv`).

### Should be clarified
- **Finding 10 – MEDIUM – models are not refit.**
  - The bootstrap reweights fixed out-of-fold scores (`WeightedAUC` sorts once and only the weights change).
  - So intervals and p-values cover **test-sample variability only**, not variability from C2 training (seed, fold split, early stopping) or from the leave-one-fold-out LR combiner.
  - With gains of 0.003, an examiner will ask. Add one sentence to Ch3:682 and to Limitations.
- **Finding 11 – LOW – why `***` never appears on single diseases.**
  - With p ≥ 1/5000 and Holm over 5, the smallest attainable p_adj is 0.001, which is not < 0.001.
  - So per-disease cells max out at `**`. Only Mean rows, or families of ≤ 4 tests, can show `***` (e.g. 3-model Mean rows: 3 × 0.0002 = 0.0006).
  - Readers will take `**` against `***` as a difference in evidence. Say this in the AppB:10 reading note.
- **Finding 12 – LOW – Mean-row family differs between scripts.** AppB:10 says "Mean rows average the five diseases and are tested once".
  - `bootstrapping.ipynb` cell 11 Holm-adjusts the AVERAGE rows **together within each group**: claims = 9 comparisons, p_adj 0.0018 = 9 × 0.0002; prob = 2.
  - `c2_arch_compare.py:157-160` and `:181` set `avg["p_adj"] = avg["p"]` (unadjusted), matching the fig:c2-arch caption "the average over diseases is a single test".
  - No star changes, but the description should match one convention.
- **Pooled patient draw across diseases** (clarity). One draw resamples the union of patients over all five diseases (`patients = union(...)`), which is what makes the Mean-row test valid. Worth one clause in Ch3:682.
- **Finding 13 – LOW.** Ch3:682 says "report its mean". The tables report point-estimate Δ ("from unrounded values", AppB:164), e.g. ens LR effusion +0.0049 (point) vs bootstrap mean 0.0048. State that Δ is the point estimate and the bootstrap gives the CI and p.
- **Finding 14 – LOW.** `Chapter_3.tex:684` "described below": the bootstrap is described in the preceding paragraph (`:682`). Swap the order or change the word.
- **AURC.** `Chapter_3.tex:662` and `:675` say Aim 2 reports AURC and E-AURC. Only E-AURC appears in Ch4/AppB tables and bootstrap files (no AURC column in `bootstrap_baselines_summary.csv`). Either drop "AURC" or add it.

---

## Checked and consistent (no action)
- C0 split table (Ch4) = generated table: thresholds 0.189/0.160/0.067/0.285/0.393; held-out AUROC 0.715–0.884.
- Aim 1 prose:
  - CNN dual−xi +0.042 avg; Sal +0.014 avg; edema +0.027 (0.804→0.831); consolidation +0.004
  - CNN I mean 0.809; attribute I mean 0.860
  - Effusion 0.783±0.006 (I+C); consolidation 0.945±0.003 (I+C+D)
  - K gains (+0.014 D effusion, +0.030 CNN-C edema, +0.002/+0.005 and +0.001/+0.004); K5→K10 ≤ 0.001
- Ensembles: 11/15 significant, range +0.0002…+0.0049, combiner bolding matches `C2_arch_compare/run_config.json` (rank for consolidation, lr elsewhere); attribute beats CNN on every task.
- Diffusion K=3 drop: 14/15 and all Mean rows; −0.0106 / −0.0046 / −0.0041; every star in `tab:k_diffusion_sub20k` and `tab:k_arm_sub20k`.
- Trajectory gaps to adjusted MSP 0.008 (consolidation) … 0.094 (effusion): recomputed 0.008 and 0.094 against `msp_auc` in `sweep_pooled.csv`.
- Aim 2:
  - ensemble +0.003 / −0.0011 E-AURC; 3/5 significant on both metrics (`bootstrap_eaurc_deltas.csv`)
  - TrustScore and SC-CE significantly below MSP everywhere; ConfidNet below everywhere except effusion
  - ResNet +0.006 (4/5); ViT −0.0004, image model −0.014 vs attribute −0.003
- Appendix B tables `tab:attr_knn_k` and `tab:cnn_knn_k` are byte-for-byte the generated numbers.

## Not covered / not verifiable here
- Chapter 3 methods vs code (hyperparameter tables, C1 attribute counts, CF generation settings, baseline adaptations): out of the agreed scope.
- Whether the figure PDFs in `Images/Results/c2*` are the latest versions in `final_experiment/figures/`: not compared (md5/mtime check).
- Saliency group-profile numbers (AppC `tab:sal_group_profiles`): not cross-checked against `saliency_explainability.ipynb`.

---

## 4. Overview of all bootstraps and Holm families

### Settings shared by every run
- 5,000 replicates, seed 0, paired, patient-level.
- One patient draw per replicate across all five diseases, with the same draws replayed in every script.
- Fold-mean AUROC / E-AURC; percentile 95 % CI; two-sided p floored at 1/5000 = 0.0002.

That floor caps the star level by family size m: the smallest possible p_adj is m × 0.0002.

| family size m | best possible p_adj | highest star |
|---|---|---|
| 1–4 | 0.0002–0.0008 | `***` |
| 5–49 | 0.001–0.0098 | `**` |
| 50–249 | ≥ 0.01 | `*` |

So in practice: every family over five diseases tops out at `**`, and only single tests or 2–3-test families can show `***`. **The number of stars therefore mostly reflects the family size, not the strength of evidence.**

### Inventory: what was tested, how it was grouped, where it appears

| # | Script → file | Comparisons | Holm family (size) | Mean row | Used in thesis |
|---|---|---|---|---|---|
| A1 | `bootstrapping.ipynb` → `bootstrap_deltas.csv`, group `claims` (KNN, K=10, full set) | 9 claims: I+C−I, I+C+D−I+C, I+C+D−I, CNN dual−xi, dual_sal−dual, dual_sal−xi, best attr−best CNN, ensemble−best single, best overall−adj. MSP | each claim across the 5 diseases (m=5) | Holm across the 9 claim means (m=9 → best ** ) | fig:aim1-attr, fig:aim1-cnn (Ch4:92), `tab:baselines_knn` AUROC stars incl. Mean (AppB:304), abstract / conclusion "3 of 5" |
| A2 | same file, group `2b` | ensemble (mean / rank / lr) − better member | all 15 combiner×disease in **one** family (m=15) | none | fig:ens_knn (Ch4:176), `tab:ens_knn` (AppB:134) — stated correctly |
| A3 | same file, group `prob` | I+C+D+P − I+C+D; CNN dual_sal_scalars − dual_sal | each across 5 diseases (m=5) | Holm across the 2 means (m=2) | fig:aim1_prob, `tab:prob_knn` (AppB:37); mean not shown |
| A4 | same file, group `msp_plain` | best overall − **plain** MSP | across 5 diseases (m=5) | single test | not shown anywhere (fig:msp-adjustment has no test) |
| B | `bootstrapping.ipynb` cell 20 → `bootstrap_eaurc_deltas.csv` | E-AURC: adj. MSP − best overall | across 5 diseases (m=5) | **single test (unadjusted)** | `tab:baselines_knn` E-AURC stars incl. Mean `***` (AppB:312), fig:aim2-baselines (b) |
| C | `c2_confidnet_compare.py`, `c2_trustscore_compare.py`, `c2_scce_compare.py` → `bootstrap_{confidnet,trustscore,scce}_vs_msp.csv`, `bootstrap_ensemble_vs_*.csv` | baseline − adj. MSP; ensemble − baseline; AUROC and E-AURC | per comparison × metric, across 5 diseases (m=5) | single test | Ch4:367 ("+0.003**" ConfidNet), fig:aim2-baselines; `ensemble_vs_*` used only for bolding in TABLES cell 41 |
| D | `c2_arch_compare.py` → `C2_arch_compare/deltas.csv` | Attr / CNN / Ensemble − adj. MSP, per backbone, AUROC and E-AURC | per backbone × comparison × metric, across 5 diseases (m=5) | **single test (unadjusted)** | `tab:c0_arch` (AppB:395), fig:c2-arch (Ch4:376) — stated correctly |
| D' | same script → `arch_gain_deltas.csv` | (backbone gain) − (DenseNet gain) | across 5 (m=5) | single | **not used** |
| E | `bootstrapping_diffusion.ipynb` → `bootstrap_diffusion_deltas.csv` (diffusion ladder, sub20k) | same 9 claims + 2b + msp_plain as A, on diffusion CFs | same as A | same as A (m=9) | **not used** in the thesis (no figure or table reads it) |
| E' | same notebook → `bootstrap_diffusion_eaurc_deltas.csv` | E-AURC adj. MSP − best (diffusion) | m=5 | single | **not used** |
| F | same notebook → `bootstrap_diffusion_external.csv` (diffusion vs NUN at K=1, sub20k) | `matched`: 12 same-config pairs (attr I, C, D, I+C, I+C+D; CNN xi, xi_sal, cf, cf_sal, dual, dual_sal, dual_sal_scalars); `family`: 2 averages; `best`: 3 (best attr, CNN, ens) | **per disease, across comparisons**: matched m=12 (best `**`), family m=2, best m=3 (`***` possible) | Holm across comparisons | `tab:diffusion_vs_knn` (AppB:164), fig:aim1-diffusion_vs_knn (Ch4:207) |
| G | same notebook cell 22 → `bootstrap_diffusion_k_deltas.csv` (K3 − K1, both sources, sub20k) | per source × model (attr, CNN, ens) | per source × model, across 5 diseases (m=5) | Holm across the 3 models within source (m=3) | `tab:k_diffusion_sub20k` (Ch4:242), `tab:k_arm_sub20k` (AppB:201) — stated correctly |
| H | `C2_evaluation_diffusion.ipynb` cell 17 → `bootstrap_diffusion_trajectory_K1_replicates.csv` | none (intervals only) | — | — | fig:trajectory whiskers. **No test exists**, yet Ch4:252 says sequence models give "no significant improvement" |
| — | `bootstrap_diffusion_internal.csv` (2026-09-29) | old version | — | — | stale; nothing reads it |

### Inconsistencies this creates in the thesis

1. **Same test, different stars (AppB:304 vs AppB:375).**
   - DenseNet ensemble − adjusted MSP, five-disease mean, AUROC, on identical draws (p = 0.0002).
   - `tab:baselines_knn` shows `0.871**`, because the Mean row comes from A1 (Holm over 9 claim means, p_adj 0.0018).
   - `tab:c0_arch` shows `0.871***` / `+0.0030***`, because it comes from D (unadjusted, p_adj 0.0002).
   - Within `tab:baselines_knn` itself, the AUROC Mean is `**` but the E-AURC Mean is `***` (B, unadjusted).
2. **Three different Mean-row rules.**
   - Holm across the means of one analysis (A1 m=9, A3 m=2, G m=3).
   - Unadjusted single test (B, C, D).
   - AppB:10 says "tested once", which matches only the second.
3. **Two different directions of correction.**
   - Everywhere except F, a family is *one comparison across the five diseases*.
   - In F (`tab:diffusion_vs_knn`), a family is *all comparisons within one disease*.
   - So one table mixes m=12 (attribute/CNN columns, max `**`) with m=3 (ensemble column, `***` possible). The ensemble's `***` and the others' `**` reflect the family, not the evidence.
4. **F's family is inflated by non-comparisons.**
   - The `matched` family of 12 includes attr I, CNN xi and CNN xi_sal, which use no counterfactual and are identical in both arms (δ = 0, p = 1).
   - It also includes dual_sal_scalars, which is not in the table.
   - These still raise the Holm factor from 5 to 12.
5. **The grouping choice changes a conclusion.** Re-running F with the convention used everywhere else (per comparison across 5 diseases) changes the stars as follows:

   | Comparison | Disease | Now | Under per-comparison Holm |
   |---|---|---|---|
   | attr I+C+D | consolidation | n.s. (p 0.0076) | **`**`** |
   | CNN dual_sal | cardiomegaly | `*` | `**` |
   | best ensemble | atelectasis, cardiomegaly, edema | `***` | `**` |
   | best ensemble | effusion | n.s. | n.s. |

   The Ch4:198 text ("12 significant … exceptions are consolidation in the attribute family …") would become 13 significant, with consolidation-attribute no longer an exception.
6. **Star rendering differs by figure.** fig:aim2-baselines labels use a single `*` for p_adj < 0.05 (`C2_evaluation.ipynb` cell 33, by design). Every other figure and table uses `*`/`**`/`***`, and the fig:aim2-baselines caption does not say this.
7. **Claims with no bootstrap behind them** (Section 1 lists several of these too):
   - NUN vs diffusion at K=3 (Ch4:191, 274)
   - trajectory sequence vs feature models "no significant improvement" (Ch4:252)
   - group error rates "differ significantly" (Conclusion)
   - adjusted vs plain MSP (Ch4:340, no significance claimed, so acceptable)

### Proposed single convention (smallest change)
- **Family = one comparison across the five diseases (m=5). Mean row = one separate test, unadjusted.** This is already what B, C, D and G's per-disease rows do and what AppB:10 says.
- Changes needed:
  - **A1/A3 Mean rows:** switch from Holm across means to unadjusted. Every claim mean except I+C+D−I+C goes `**` → `***`; `tab:baselines_knn` AUROC Mean becomes `***`, matching E-AURC and `tab:c0_arch`.
  - **G Mean rows:** unadjusted gives the same `***`, so nothing visible changes.
  - **F:** re-group per comparison across diseases, and drop the no-CF and `_scalars` rows. Stars change as in point 5; update Ch4:198 and the AppB:164 note.
  - **A2 (15-test combiner family):** either keep it and say why (one figure = one family), or split it into three m=5 families (per combiner). Splitting would turn consolidation-LR (+0.0006, p 0.0152) significant: it is the largest p in its m=5 family, so p_adj = 0.0152. Atelectasis-rank (p 0.177) and consolidation-mean (0.0392 × 2 = 0.078) stay n.s., so "11 of 15" would become "12 of 15". Decide before re-running.
- State the star ceiling once, in Ch3 sec:stats (`**` is the maximum for any five-disease family with 5,000 draws). Otherwise readers will compare `**` with `***` as evidence.
- All of this is post-processing of the saved replicate files (`bootstrap_*_replicates.csv`). No new draws or model runs are needed.
