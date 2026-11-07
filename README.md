# isan-rice-yield-benchmark

**Leakage-safe, explainable multi-source machine learning for district-level rice yield forecasting:
a ground-truth simulation benchmark for Northeast Thailand.**

Author: **Thanatchon Chongmankhong**

> **All data in this repository are synthetic.** They come from a documented data-generating process (DGP)
> whose true yield components are stored, so model explanations can be checked against the truth.
> Results are methodological findings about forecasting pipelines, not empirical statements about Thai rice
> yields. To run an empirical study, replace `data/raw/` with real data (see `docs/DATA_SCHEMA.md`).

📄 Manuscript: [`paper/paper.pdf`](paper/paper.pdf) (LaTeX source in `paper/`; every number is generated from `results/`).

## What is inside
- **Simulator**: 200 districts x 18 seasons (2008-2025); drought, flood, heat, disease, salinity, price and
  competing-crop (cassava, sugarcane, maize) effects, management persistence, satellite-only pest outbreaks;
  emits weather, cloud-contaminated satellite indices (NDVI, EVI, LSWI, LST), prices and noisy map layers.
- **Features**: 81 leakage-safe features in 8 groups (history, moisture, water, climate, satellite, prices,
  geography, map/spatial lags).
- **Learners**: persistence, climatology, panel fixed effects, Ridge, random forest, MLP, LightGBM,
  LightGBM + Ridge ensemble.
- **Experiments**: E1 accuracy, E2 leakage audit, E3 ablation, E4 ground-truth recovery of SHAP,
  E5 in-season forecasting, E6 conformal intervals, E7 robustness (10 replicate worlds, 4 scenarios).

## Key findings (synthetic benchmark, test seasons 2018-2025)
| Topic | Result |
|---|---|
| Accuracy | LightGBM RMSE 27.4 kg/rai vs. climatology 47.1 (-42%); replicated in 10 worlds |
| Leakage | random cross-validation understates error by 10-21% and hides model failures |
| Inner-loop leakage | row-wise LOO tuning of Ridge caused catastrophic seasons (post-hoc finding) |
| Satellite | only feature group whose removal matters (+25% RMSE); value appears from October |
| Explanations | SHAP recovers drought (r = 0.84) but misattributes heat and absorbs the trend into prices |
| Uncertainty | 80% conformal intervals cover 78.3% of outcomes |

## Quick start
```bash
git clone https://github.com/thanatchon36/isan-rice-yield-benchmark.git
cd isan-rice-yield-benchmark
pip install -r requirements.txt
python run_all.py   # tests -> experiments (~7 min, 1 CPU) -> figures/tables -> paper/main.pdf (needs LaTeX)
```
Single steps: `python -m pytest -q tests` · `python experiments.py [--only e3 --force | --quick]` ·
`python figures.py` · `cd paper && latexmk -pdf main.tex`

## Repository layout
```
config.py          study design (crop, region, seasons, validation years, feature groups)
simulate.py        data-generating process + true components + scenarios
features.py        leakage-safe feature engineering (+ deliberately leaky variants for the audit)
models.py          learners (panel FE, Ridge, RF, MLP, LightGBM, ensemble)
evaluation.py      rolling origin, random-CV audit, metrics, season bootstrap, DM-HLN + Holm, conformal, SHAP
experiments.py     E1-E7 (cached, resumable)
diagnostics.py     post-hoc diagnosis of the Ridge failures (exploratory)
diag_worlds.py     same diagnosis in other worlds
figures.py         figures, LaTeX tables and paper/tables/macros.tex
figcheck.py        automatic check that no text overlaps in any figure
make_arxiv.py      builds arxiv_source.zip (LaTeX + .bbl + figures) for arXiv submission
thailand_map.py    study-area figure of the paper (`--paper`) and standalone maps (Natural Earth, UTM 47N)
gee_extract.py     Google Earth Engine template for real MODIS / CHIRPS / ERA5-Land / SRTM inputs
tests/             leakage, determinism and shape tests
data/raw/          synthetic inputs (same schema as real data)
results/           experiment outputs (CSV/JSON)
paper/             manuscript source, figures, tables, paper.pdf
docs/              data schema, internal review, references to verify
.github/workflows/ continuous integration: leakage/determinism tests on every push
```

## Submitting to arXiv
`python make_arxiv.py` creates `arxiv_source.zip` (main.tex, main.bbl, sections, tables, figures) ready to upload.
The compiled preprint is `paper/paper.pdf`.

## Using real data
1. Run `gee_extract.py` with your Earth Engine account and district polygons, then `--split` the exports.
2. Add yields, prices, reservoir storage and map layers following `docs/DATA_SCHEMA.md`.
3. `python run_all.py`.

## Status
Draft research code and manuscript. Before submission: verify every reference in
`docs/references_to_verify.md` (compiled without database access), add affiliation/e-mail if desired, and address
`docs/review.md`. Developed with the assistance of Claude (Anthropic).

## License
Code: MIT (see `LICENSE`). Choose a licence for the manuscript text and figures (e.g. CC BY 4.0) before publishing.
