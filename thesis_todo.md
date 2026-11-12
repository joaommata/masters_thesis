# Thesis to-do list (from the pre-submission review, 2026-10-29)

Line numbers refer to `MSc_Dissertation-13/Chapters/` as of 2026-10-29. Tick each box when done.
"PT" = mirror the change in `Abstract-PT-Resumo.tex`.

---

## 0. Files to upload to Overleaf

- [ ] `final_experiment/figures/xai_key/` – the 4 group-profile PDFs (held-out error rates in the panel headers) → `Images/Results/xai/<disease>/`
- [ ] `final_experiment/figures/bootstrap_key/` – 5 regenerated statistics figures (new star convention); `README.txt` there says where each one goes (4 → `Images/Results/c2_key/`, `knn_aim2d_baselines.pdf` → `Images/Results/c2/`)

---

## 1. Abstract (EN + PT)

- [ ] **"our ensemble was the only method to beat it"** → "…the only method to beat it **on average**…" (ConfidNet beats adjusted MSP on effusion).
- [ ] **"with a single generated counterfactual carrying more signal than a single retrieved one"** → add "**at equal training size, an advantage retrieval recovers with more data**".
- [ ] **"This gain held for ResNet-50 but not fully for ViT-B/16"** → "This gain held for ResNet-50 but not for ViT-B/16, where the ensemble only matched adjusted MSP on average."
- [ ] Keep "35 % to 78 %" (it is the held-out value, now used everywhere).

## 2. Chapter 3

- [ ] **Ch3:684** "the patient-level bootstrap described below" → "described above".
- [ ] **Ch3 sec:stats, after the Holm paragraph (~l.686)**, add:
  > "Following common practice for comparing methods across several datasets \cite{demsar2006statistical, dror2017replicability}, each comparison forms one Holm family across the five tasks, and the five-task mean is tested once. Because the bootstrap p-value cannot fall below $1/5{,}000$, $p_{\mathrm{adj}}<0.01$ (**) is the strongest level a five-task family can reach; *** appears only on single tests such as the five-task mean. The bootstrap resamples test patients with the trained models fixed, so its intervals do not include training variability."
- [ ] **Ch3:662 / sec:metrics** say Aim 2 reports AURC *and* E-AURC; only E-AURC is reported. Drop "AURC" there, or add it to the tables.

## 3. Chapter 4

### Aim 1
- [ ] **Ch4:84, 86, 293**: "+0.004 (effusion) … +0.011 (atelectasis)" / "at most +0.011" → **+0.005 … +0.012** (unrounded 0.0046, 0.0116).
- [ ] **fig:aim1-attr caption (~l.92)** and the CNN figure ("tests as in…"): the star key must include *** (the average panels now show *** after regeneration): "*** $p_{\mathrm{adj}}<0.001$, ** $p_{\mathrm{adj}}<0.01$, * $p_{\mathrm{adj}}<0.05$".
- [ ] **Ch4:125 result box (probability)** → replace with:
  > "Adding $C_0$'s probability and entropy does not improve either family on average: the attribute model loses $0.0004$ \acs{AUROC} (significant) and the \acs{CNN} gains $0.0003$ (not significant). The only significant per-task gain is the \acs{CNN} on atelectasis ($+0.0027$). We therefore keep all models ``probability-free'', in the sense that we do not feed them $C_0$'s probability directly."
- [ ] **Ch4:165 result box + Ch4:181 text (ensembles)**: "11 of 15 gains significant" → **"12 of 15"** (consolidation LR is now *, after splitting the family per combiner).
- [ ] **fig:ens_knn caption (Ch4:176)**: "Holm-adjusted across the 15 combiner--finding tests; ** …, n.s." → "Holm-adjusted across the five tasks within each combiner, the average over tasks a single test; *** $p_{\mathrm{adj}}<0.001$, ** $p_{\mathrm{adj}}<0.01$, * $p_{\mathrm{adj}}<0.05$, n.s. not significant" (the regenerated figure shows *** on the average).
- [ ] **Ch4:191 result box (CF sources)** → replace with:
  > "With a single counterfactual, diffusion beats \acp{NUN} in 13 of 15 model--task comparisons, by $+0.003$ to $+0.012$ \acs{AUROC}; the exceptions are effusion for the \acs{CNN} and the ensemble. Averaging over three counterfactuals reverses this: it lowers \acs{AUROC} for diffusion but raises it for \acp{NUN}, so with $K=3$ for both sources, \acp{NUN} beat diffusion on average for all three models (attribute $+0.008^{***}$, \acs{CNN} $+0.002^{*}$, ensemble $+0.001^{*}$), though for the \acs{CNN} and the ensemble the gain is significant on few or no individual tasks."
- [ ] **Ch4:198 text**: "diffusion shows 12 significant improvements … exceptions are consolidation in the attribute family, effusion in the ensembles and CNNs" → "**13** significant improvements … the exceptions are effusion for the CNN and the ensemble".
- [ ] **Fig. 4.7 caption (~l.207)**: "(paired bootstrap, Holm-adjusted; …)" → "(paired bootstrap, Holm-adjusted across the five tasks within each model; the average over tasks is a single test; …)". The regenerated figure has the new stars (per-task ensemble ** instead of ***, consolidation-attr **, cardiomegaly-CNN **) and new *** brackets in the average panel.
- [ ] **tab:k_diffusion_sub20k note (Ch4:242)**: "for the Mean row, … Holm-adjusted across the three models" → "for the Mean row, the bootstrap of the five-disease average, a single test". Cells are unchanged.
- [ ] **Ch4:252 (trajectory)**: "provides no significant improvement" → "provides no improvement" (no test was run). Also delete the duplicated sentence "All trajectory models remain below the adjusted MSP…".
- [ ] **Ch4:274**: "With $K>>1$, retrieved beat generated" → "With $K=3$, retrieved beat generated".
- [ ] **Ch4:325**: "needs 70,000 to 80,000 training images to match diffusion, about four times the subset" → "matches diffusion only with the full training pool (about 35,000 to 55,000 rows by interpolation, two to three times the subset); with three or more, it needs 16,000 to 32,000".

### Aim 2
- [ ] **Ch4:332 result box**: "Our best-models generalize to ResNet-50 and partially to ViT-B/16" → "…generalize to ResNet-50 but not to ViT-B/16, where they only match adjusted \acs{MSP} on average".
- [ ] **Ch4:335 (Aim 2 intro)**, after the sentence naming the ensemble, add:
  > "Diffusion counterfactuals were not carried into the benchmark: they were generated only for the 20{,}000-image subset, where no diffusion-based model beats adjusted \acs{MSP}, whereas retrieved counterfactuals, although individually weaker, scale to the full dataset."
- [ ] **Ch4:351**: "significantly matches" → "significantly beats"; "(2.4\%)" → "(2.5\%)".
- [ ] **Ch4:367**: ConfidNet "+0.003**" → "+0.004**".
- [ ] **Ch4:393**: "2.4\%" → "2.5\%".
- [ ] **Ch4:396**: "None of the baseline beat our model or the adjusted MSP" → "None of the baselines beat our model, and only ConfidNet on effusion beats adjusted \acs{MSP}".

### Aim 3
- [ ] **Ch4:450**: "best group A (37\% wrong)" → "(35\% wrong)".

## 4. Appendix B

- [ ] **AppB:10 reading note**: add "Because the bootstrap p-value cannot fall below $1/5{,}000$, ** is the strongest level for a per-disease row; *** appears only on Mean rows."
- [ ] **tab:prob_knn** (optional): add a Mean row `\textit{Mean} & $0.868$ & $0.868$ & $-0.0004^{**}$ & $0.865$ & $0.865$ & $+0.0003$ \\` and to the note "Mean: the five-disease average, tested once."
- [ ] **tab:ens_knn**: Consolidation LR `$+0.0006$` → `$+0.0006^{*}$`. Note (AppB:134): "Holm family: the 15 combiner--disease tests" → "Holm family: the five diseases, within each combiner".
- [ ] **tab:diffusion_vs_knn**: copy the new rows from `final_experiment/tables/diffusion_vs_knn.tex`:
  atelectasis ens `+0.0075^{**}`; cardiomegaly CNN `+0.0038^{**}`, ens `+0.0059^{**}`; consolidation attr `+0.0028^{**}`; edema ens `+0.0064^{**}`.
  Note (AppB:164): "Holm family, within each disease: the 12 matched … or the 3 best-model comparisons" → "Holm family: the five diseases, within each model".
- [ ] **tab:k_arm_sub20k note (AppB:201)**: "Mean rows: the three models, within each source" → "Mean rows: the five-disease average, a single test". Cells are unchanged.
- [ ] **fig:subsets_n caption (AppB:217)**: replace with the interpolation caption (in the chat log of 2026-10-29; key sentence: "The dotted vertical line marks where retrieval first reaches the diffusion level, interpolated on the log scale between grid points: in every panel this happens at $K=20$, at 14k--20k rows.").
- [ ] **tab:subsets_knn_nstar**: K=1 cells "79k" → "all (79k)" etc.; replace the note with the version that explains "all", the interpolated 34k–56k, and the pooled-vs-fold-mean AUROC difference.
- [ ] **tab:baselines_knn**: AUROC Mean `0.871 ± 0.002^{**}` → `^{***}`.
- [ ] **tab:risk80** (optional, cosmetic): C0 error 18.97 → 18.98 (cardiomegaly), 38.87 → 38.88 (consolidation).

## 5. Appendix C

- [ ] Replace **tab:top_attributes_cardiomegaly_effusion** and **tab:sal_group_profiles** with the corrected versions (held-out error rates and held-out n in the "wrong (n)" rows; note updated).

## 6. Discussion / Conclusion

- [ ] **Discussion:91 (Conclusion)**: "groups whose error rates differ significantly" → "groups whose error rates differ widely, from 35\% to 78\% wrong among positive cardiomegaly calls" (no test was run).
- [ ] **Discussion:24** (optional): "groups whose error rates differ" → "differ widely (35\% to 78\% wrong for positive cardiomegaly calls)".

## 7. Bibliography

- [ ] Add to `Bibliography.bib` (check details against the publisher pages):
```bibtex
@article{demsar2006statistical,
  author  = {Dem{\v{s}}ar, Janez},
  title   = {Statistical Comparisons of Classifiers over Multiple Data Sets},
  journal = {Journal of Machine Learning Research},
  volume  = {7},
  pages   = {1--30},
  year    = {2006}
}
@article{dror2017replicability,
  author  = {Dror, Rotem and Baumer, Gili and Bogomolov, Marina and Reichart, Roi},
  title   = {Replicability Analysis for Natural Language Processing: Testing Significance with Multiple Datasets},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {5},
  pages   = {471--486},
  year    = {2017}
}
```

## 8. Code housekeeping (no thesis change)

- Notebooks edited in place on 2026-10-29: `attribute_explainability.ipynb` (cell 16), `saliency_explainability.ipynb` (cell 18), `bootstrapping.ipynb` (cells 1, 10, 11; `RERUN = False`), `bootstrapping_diffusion.ipynb` (cells 9, 13, 22; new `bootstrap_diffusion_k3_sources.csv`), `TABLES.ipynb` (table notes). Backups: `/tmp/claude-321493/-zhome-d0-a-221493-thesis/79c420fc-69a2-4d08-aea9-968f6da0963e/scratchpad/boot_backup/`.
- `TABLES.ipynb` stops at cell 46 (top attributes for all 5 diseases, needs atelectasis/consolidation/edema explainability runs). Not caused by these changes; every table before it regenerated.
- `bootstrap_diffusion_internal.csv` (29 Sep) is stale and read by nothing; it can be deleted.
