# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Zenodo archive of every release (concept DOI [10.5281/zenodo.24036674](https://doi.org/10.5281/zenodo.24036674)), cited in the README, `CITATION.cff`,
  the paper, the web reader and the Hugging Face dataset card.

## [0.1.0] - 2026-11-07

### Added
- Simulator of main-season rice in 200 districts of Northeast Thailand over 2008-2025 with drought, flood, heat,
  disease, salinity, price and competing-crop effects, persistent management and satellite-only pest outbreaks,
  emitting weather, cloud-contaminated satellite indices, prices, reservoir storage, noisy map layers and the true
  yield components.
- 81 leakage-safe features in 8 groups, with deliberately leaky variants for the audit, and tests that fail if a
  feature sees a later season.
- Eight learners (persistence, climatology, panel fixed effects, Ridge, random forest, MLP, LightGBM, LightGBM + Ridge
  ensemble) and experiments E1-E7: accuracy, leakage audit, ablation, ground-truth recovery of SHAP, in-season
  forecasting, conformal intervals and robustness over 10 worlds and 4 scenarios; post-hoc Ridge diagnostics.
- The preprint (`paper/paper.pdf`), with every number, table and figure generated from `results/`.
- Google Earth Engine template for real MODIS, CHIRPS, ERA5-Land and SRTM inputs, and the real-data schema.
- Web reader of the paper on GitHub Pages, <https://thanatchon36.github.io/isan-rice-yield-benchmark/paper/>:
  pandoc renders `paper/main.tex` with `paper/reader.html`, takes section, table, figure and equation numbers from
  the LaTeX build (`paper/main.aux`) and shows a results strip whose numbers come from `paper/tables/macros.tex`.
- Dataset on Hugging Face, [`thanatchon36/isan-rice-yield-benchmark`](https://huggingface.co/datasets/thanatchon36/isan-rice-yield-benchmark):
  the simulated main world with its true yield components and the experiment results (CC BY 4.0), staged and uploaded
  by `scripts/hf_dataset.py` (`make dataset`).
- README banner (`docs/assets/banner.png`, `make banner`) drawn from the simulated districts.
- Package layout: `src/isanrice/` with `pyproject.toml`, a command line (`python -m isanrice.cli`) and a `Makefile`
  (`make setup`, `make all`, `make test`, ...); helper scripts in `scripts/`; CI with uv (`make setup`, `make test`).
- Paper build with tectonic (`make paper`) and the arXiv source bundle `paper/arxiv.tar.gz` (`make arxiv`), checked
  to compile on its own.
