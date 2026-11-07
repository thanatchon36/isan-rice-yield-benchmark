# Internal review ("Reviewer 2" simulation) and responses
Manuscript: *Leakage-safe, explainable multi-source machine learning for district-level rice yield forecasting:
a ground-truth simulation benchmark for Northeast Thailand.*  Recommendation to self: **major revision before submission.**

## Major comments
1. **External validity / circularity.** The DGP was written by the same authors who evaluate the models; e.g. satellite
   indices are generated from the visible yield, which builds in their value. *Response:* stated in abstract,
   introduction and limitations; framed as a methods benchmark. *To do:* replicate with real OAE/DOAE yields or
   calibrate DGP parameters to real data before claiming agronomic conclusions.
2. **Model selection and post-hoc analysis (HARKing risk).** The validation-selected ensemble lost to LightGBM on
   test seasons; the season-blocked Ridge remedy was found afterwards. *Response:* the ensemble remains the reported
   main model; Sec. 5.8 and Table 8 are labelled exploratory; limitation added that the check was applied to worlds
   selected because they failed. *To do:* prospective test of season-blocked tuning on fresh seeds and all scenarios.
3. **Statistical power.** Eight test seasons; no Holm-adjusted DM test significant. *Response:* reported plainly;
   evidence rests on bootstrap CIs and ten replicate worlds. *To do:* longer simulated panels or pooled tests.
4. **Recovery metric.** Correlation ignores scale; the component-to-group mapping (OBSERVED_BY) is a modelling
   choice. *Response:* limitation added. *To do:* regression-based recovery, conditional/observational SHAP.
5. **Ablation with fixed hyper-parameters** and ablation of an unstable learner mixes information content with
   stability. *Response:* LightGBM ablation treated as primary; all learners reported (Table B2).
6. **Baseline fairness.** MLP untuned; panel FE without dynamic terms; RF defaults. *Response:* limitation added.
7. **Leakage audit depends on DGP choices** (e.g. price feedback kappa = 0.25). *To do:* sensitivity over kappa.
8. **Intervals.** Calibration set grows from 2014 and mixes regimes; coverage failures follow model failures.
   *To do:* conformalised quantile regression / adaptive conformal inference.

## Minor comments
- Author added (Thanatchon Chongmankhong, arXiv preprint style). Optionally add affiliation, e-mail, ORCID, funding and
  competing-interest statements.
- **Verify every reference** (compiled from memory; see references_to_verify.md); complete "and others" author lists.
- Consider moving Table 8 (post-hoc) to the appendix if a journal requires a confirmatory main text.
- Copy-edit; check journal template and word limits.

## Checks performed
- All numbers come from `paper/tables/macros.tex` generated from `results/` (no hand-typed results).
- LaTeX compiles without errors; 0 undefined references or citations.
- Leakage tests: 6/6 pass (including tests that deliberately leaky variants are detected).
- Abstract about 230 words.
- Automatic text-overlap check (`src/isanrice/figcheck.py`) on all 10 figures: no overlapping text elements
  (`results/figure_text_check.json`); legends and annotations re-placed and checked visually.
