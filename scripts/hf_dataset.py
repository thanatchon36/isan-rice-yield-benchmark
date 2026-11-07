"""Stage (and optionally upload) the Hugging Face dataset thanatchon36/isan-rice-yield-benchmark.

Copies the simulated main world (data/raw/) and the experiment results (results/) into build/hf_dataset/,
writes the dataset card and SHA256SUMS. ``--upload`` pushes the folder with the ``hf`` CLI (log in first
with ``hf auth login``).

Usage: ``python scripts/hf_dataset.py [--upload]`` (or ``make dataset [UPLOAD=1]``).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "build" / "hf_dataset"
REPO = "thanatchon36/isan-rice-yield-benchmark"
RESULTS = ["e1_metrics.csv", "e1_predictions.csv", "e1_by_season.csv", "e1_best.json", "e2_leakage.csv",
           "e3_ablation.csv", "e4_recovery.csv", "e4_corr.csv", "e5_leadtime.csv", "e6_conformal_season.csv",
           "e6_conformal_summary.csv", "e7_robustness.csv", "diag_ridge_season.csv", "diag_ridge_worlds.csv",
           "feature_groups.csv"]

CARD = """---
license: cc-by-4.0
pretty_name: "Isan rice yield benchmark: a ground-truth simulation of district-level rice yields in Northeast Thailand"
language:
  - en
size_categories:
  - 10K<n<100K
task_categories:
  - tabular-regression
tags:
  - agriculture
  - crop-yield
  - rice
  - thailand
  - remote-sensing
  - explainable-ai
  - data-leakage
  - conformal-prediction
  - synthetic
configs:
  - config_name: yields
    data_files: data/yields.csv
    default: true
  - config_name: districts
    data_files: data/districts.csv
  - config_name: climate_monthly
    data_files: data/climate_monthly.csv
  - config_name: satellite_monthly
    data_files: data/satellite_monthly.csv
  - config_name: prices_annual
    data_files: data/prices_annual.csv
  - config_name: reservoir_annual
    data_files: data/reservoir_annual.csv
  - config_name: truth_components
    data_files: data/truth_components.csv
  - config_name: predictions
    data_files: results/e1_predictions.csv
---

# Isan rice yield benchmark

> **All data in this dataset are synthetic.** They come from a documented data-generating process whose true
> yield components are stored next to the observations, so forecasts *and their explanations* can be checked
> against the truth. Nothing here is an empirical statement about Thai rice yields.

Data for the preprint *Leakage-safe, explainable multi-source machine learning for district-level rice yield
forecasting: a ground-truth simulation benchmark for Northeast Thailand* by Thanatchon Chongmankhong
([read online](https://thanatchon36.github.io/isan-rice-yield-benchmark/paper/) ·
[PDF](https://github.com/thanatchon36/isan-rice-yield-benchmark/blob/main/paper/paper.pdf)).
Main-season rice in 200 simulated districts of Northeast Thailand (Isan) over 18 seasons, 2008-2025, with drought,
flood, heat, disease, salinity, price and competing-crop effects, persistent management and satellite-only pest
outbreaks. Code: <https://github.com/thanatchon36/isan-rice-yield-benchmark>.

*Independent personal research. Does not represent the views of any employer or institution.*

## Files

| path | content |
|---|---|
| `data/yields.csv` | district-season rice yield, `yield_kg_rai` (kg per rai; 1 rai = 0.16 ha, multiply by 0.00625 for t/ha); 200 districts x 18 seasons |
| `data/districts.csv` | static district attributes: province, coordinates, basin, terrain, soil, salinity, irrigation and paddy share (noisy map layers), flood frequency, rainfall normal, distances to river, sugar mill and market |
| `data/climate_monthly.csv` | May-November weather: rainfall, mean and maximum temperature, relative humidity, soil moisture |
| `data/satellite_monthly.csv` | May-November NDVI, EVI, LSWI and land-surface temperature; `NaN` = no cloud-free composite |
| `data/prices_annual.csv` | 2006-2025 farm-gate prices of rice, cassava, sugarcane and maize, and the urea price |
| `data/reservoir_annual.csv` | reservoir storage (% of capacity) by basin and year |
| `data/truth_components.csv` | **the ground truth**: the additive contribution of every mechanism (base, trend, drought, flood, heat, disease, salinity, price, management, pest, noise) to each district-season yield, in kg/rai |
| `data/meta.json` | seed, scenario and every parameter of the data-generating process |
| `results/e1_*` | E1 accuracy: metrics with season-block bootstrap CIs and Holm-adjusted Diebold-Mariano tests, out-of-sample predictions for 2018-2025 with conformal 80%/90% intervals, errors by season, the selected configuration |
| `results/e2_leakage.csv` | E2: the leakage-safe rolling origin against random 5-fold CV, same-season prices, current neighbour yields and anomalies from the full-period mean |
| `results/e3_ablation.csv` | E3: change in RMSE when each of the eight feature groups is removed |
| `results/e4_*` | E4: how well grouped SHAP values recover the true components |
| `results/e5_leadtime.csv` | E5: in-season forecasts from July to November, with and without satellite features |
| `results/e6_*` | E6: conformal interval coverage and width, overall and by season |
| `results/e7_robustness.csv` | E7: the model race repeated in 10 replicate worlds of the base scenario and in 4 alternative scenarios (nonlinear, clouds, map error, no pest) |
| `results/diag_*` | post-hoc diagnosis of the Ridge failures (row-wise vs season-blocked penalty tuning) |
| `results/feature_groups.csv` | the 81 features and their groups |
| `SHA256SUMS` | checksums of every file above |

Test seasons are 2018-2025 under a rolling origin: the forecast for season *t* is trained on seasons before *t* only.
`docs/DATA_SCHEMA.md` in the code repository gives the schema that real data must follow to replace these files.

## Licence

CC BY 4.0.

## Citation

```bibtex
@misc{chongmankhong2026isanrice,
  author = {Chongmankhong, Thanatchon},
  title  = {Leakage-safe, explainable multi-source machine learning for district-level rice yield forecasting:
            a ground-truth simulation benchmark for Northeast Thailand},
  year   = {2026},
  url    = {https://thanatchon36.github.io/isan-rice-yield-benchmark/paper/},
  note   = {Preprint; all results are from a synthetic benchmark}
}
```
"""


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "data").mkdir(parents=True)
    (STAGE / "results").mkdir()
    for f in sorted((ROOT / "data" / "raw").glob("*")):
        shutil.copyfile(f, STAGE / "data" / f.name)
    for name in RESULTS:
        shutil.copyfile(ROOT / "results" / name, STAGE / "results" / name)
    (STAGE / "README.md").write_text(CARD, encoding="utf-8")
    files = sorted(p for p in STAGE.rglob("*") if p.is_file() and p.name != "README.md")
    (STAGE / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(STAGE)}\n" for p in files))
    print(f"staged {len(files)} files in {STAGE.relative_to(ROOT)}")
    return STAGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help=f"upload to huggingface.co/datasets/{REPO}")
    a = ap.parse_args()
    folder = stage()
    if a.upload:
        subprocess.run(["hf", "repo", "create", REPO, "--repo-type", "dataset", "--exist-ok"], check=True)
        subprocess.run(["hf", "upload", REPO, str(folder), ".", "--repo-type", "dataset",
                        "--commit-message", "Upload synthetic benchmark data and results"], check=True)


if __name__ == "__main__":
    main()
