"""Train, export, verify and (optionally) upload the model repository thanatchon36/isan-rice-yield-benchmark.

The final models are the paper's main model, the validation-selected LightGBM + Ridge ensemble (E1), refitted on
every season of the simulated main world with features (2011-2025; 2008-2010 are yield history only) with the tuned hyperparameters in results/e1_best.json.
Nothing is pickled: LightGBM is saved in its native text format and Ridge (median imputation, standardisation,
penalised linear fit) as JSON coefficients, so ``predict.py`` loads them with numpy, pandas and lightgbm only.
Split-conformal intervals use the ensemble's out-of-sample errors on every season it could forecast (2011-2025).
The export is checked against the in-memory models before anything is written to build/hf_model/.

Usage: ``python scripts/hf_model.py [--upload]`` (or ``make model [UPLOAD=1]``).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from isanrice import __version__
from isanrice import config as C
from isanrice import evaluation as E
from isanrice.features import build_features, feature_list, load_raw
from isanrice.models import make_model

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "build" / "hf_model"
REPO = "thanatchon36/isan-rice-yield-benchmark"
DOI = "10.5281/zenodo.24036674"
Y = C.YIELD_COL

PREDICT_PY = '''"""Load the exported isan-rice-yield-benchmark models and predict (numpy, pandas and lightgbm only).

    from predict import load
    model = load(".")                    # folder with lightgbm.txt, ridge.json, features.json, conformal.json
    out = model.predict(X)               # X: DataFrame with the 81 columns listed in features.json
    out[["ensemble", "lo80", "hi80"]]    # kg/rai; multiply by 0.00625 for t/ha
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


class IsanRiceModel:
    def __init__(self, folder="."):
        d = Path(folder)
        self.features = json.loads((d / "features.json").read_text())["features"]
        self.booster = lgb.Booster(model_file=str(d / "lightgbm.txt"))
        r = json.loads((d / "ridge.json").read_text())
        self.med, self.mu, self.sd = (np.asarray(r[k], float) for k in ("impute_median", "scaler_mean", "scaler_scale"))
        self.coef, self.intercept = np.asarray(r["coef"], float), float(r["intercept"])
        self.conformal = json.loads((d / "conformal.json").read_text())["quantiles"]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise KeyError(f"missing feature columns: {missing}")
        A = X[self.features].to_numpy(float)
        p_lgb = self.booster.predict(A)
        Z = np.where(np.isnan(A), self.med, A)
        p_ridge = ((Z - self.mu) / self.sd) @ self.coef + self.intercept
        out = pd.DataFrame({"lightgbm": p_lgb, "ridge": p_ridge, "ensemble": 0.5 * (p_lgb + p_ridge)}, index=X.index)
        for lvl, (qlo, qhi) in self.conformal.items():
            out[f"lo{lvl}"], out[f"hi{lvl}"] = out.ensemble + qlo, out.ensemble + qhi
        return out


def load(folder="."):
    return IsanRiceModel(folder)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    X = pd.read_csv(here / "sample_input.csv")
    print(load(here).predict(X).head().round(1))
'''


def ridge_json(pipe, feats):
    imp, sc, reg = pipe[0], pipe[1], pipe[-1]
    return {"model": "median imputation -> standardisation -> ridge (penalty chosen by RidgeCV, efficient LOO)",
            "features": feats, "impute_median": np.nan_to_num(imp.statistics_).tolist(),
            "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
            "alpha": float(reg.alpha_), "coef": reg.coef_.tolist(), "intercept": float(reg.intercept_)}


def card(best, metrics, conf, n_train):
    m = metrics.set_index("model")
    row = lambda k, lab: (f"| {lab} | {m.at[k, 'RMSE']:.1f} [{m.at[k, 'RMSE_lo']:.1f}, {m.at[k, 'RMSE_hi']:.1f}] | "
                          f"{m.at[k, 'MAE']:.1f} | {m.at[k, 'MAPE']:.1f} | {m.at[k, 'R2']:.2f} | {m.at[k, 'Skill']:+.2f} |")
    params = ", ".join(f"`{k}={v}`" for k, v in best["params"].items())
    return f"""---
license: mit
library_name: lightgbm
pipeline_tag: tabular-regression
datasets:
  - thanatchon36/isan-rice-yield-benchmark
tags:
  - tabular-regression
  - crop-yield
  - rice
  - agriculture
  - thailand
  - remote-sensing
  - lightgbm
  - ridge
  - conformal-prediction
  - synthetic
model-index:
  - name: isan-rice-yield-benchmark (LightGBM + Ridge ensemble)
    results:
      - task:
          type: tabular-regression
          name: District-level rice yield forecasting (rolling origin, test seasons 2018-2025)
        dataset:
          type: thanatchon36/isan-rice-yield-benchmark
          name: Isan rice yield benchmark (synthetic)
        metrics:
          - type: rmse
            value: {m.at['Ensemble', 'RMSE']:.1f}
            name: RMSE (kg/rai)
          - type: mae
            value: {m.at['Ensemble', 'MAE']:.1f}
            name: MAE (kg/rai)
          - type: r_squared
            value: {m.at['Ensemble', 'R2']:.2f}
            name: R2
---

# Isan rice yield benchmark: forecasting models

> **Trained on synthetic data.** These models were fitted to a simulated Northeast Thailand whose data-generating
> process is documented and whose true yield components are known. They show how the paper's pipeline behaves; they
> are **not** forecasts of Thai rice yields and must not be used as such. To forecast real yields, rebuild the
> features from real inputs and retrain with the code (`make all`).

Main-season rice yield (kg/rai) of a district, forecast at the end of the season (November information set) from
81 leakage-safe features in eight groups: previous yields, humidity and soil moisture, water (rain, reservoirs,
irrigation), climate, satellite indices (NDVI, EVI, LSWI, LST), prices of rice and competing crops, geography, and
map layers with spatial lags. Companion models to the preprint *Leakage-safe, explainable multi-source machine learning
for district-level rice yield forecasting: a ground-truth simulation benchmark for Northeast Thailand* by Thanatchon
Chongmankhong ([read online](https://thanatchon36.github.io/isan-rice-yield-benchmark/paper/) ·
[code](https://github.com/thanatchon36/isan-rice-yield-benchmark) ·
[data](https://huggingface.co/datasets/thanatchon36/isan-rice-yield-benchmark) · DOI [{DOI}](https://doi.org/{DOI})).

*Independent personal research. Does not represent the views of any employer or institution.*

## The model

The paper's main model, selected on the validation seasons 2015-2017: the equal-weight average of
- **LightGBM** (`lightgbm.txt`, native format), with the hyperparameters tuned on 2015-2017: {params};
- **Ridge regression** (`ridge.json`): median imputation, standardisation and a ridge fit whose penalty RidgeCV chose
  (alpha = {conf['ridge_alpha']:.3g}).

Both are refitted here on all {n_train:,} district-seasons with features in the main world (200 districts,
seasons 2011-2025; 2008-2010 serve only as yield history), the information a forecaster would have before the 2026
season. The level of yield is the target.

**Prediction intervals.** `conformal.json` holds split-conformal offsets: empirical quantiles of the ensemble's
out-of-sample errors on every season it could forecast with at least three seasons of training data
({conf['calibration_seasons'][0]}-{conf['calibration_seasons'][-1]}, {conf['n_calibration']:,} forecasts). Added
to a point forecast they give 80% and 90% intervals. In the paper's rolling evaluation the 80% intervals covered
78.3% of outcomes; the worst season (2018) covered 40%.

## Accuracy (from the paper, synthetic benchmark)

Rolling origin: each test season 2018-2025 is forecast by models trained on earlier seasons only (1,600 forecasts).
RMSE and MAE in kg/rai (multiply by 0.00625 for t/ha); brackets: 95% season-block bootstrap interval; Skill =
1 - MSE / MSE of climatology.

| Model | RMSE [95% CI] | MAE | MAPE (%) | R2 | Skill |
|---|---|---|---|---|---|
{row('Ensemble', 'LightGBM + Ridge ensemble (this model)')}
{row('LightGBM', 'LightGBM alone (`lightgbm.txt`)')}
{row('Ridge', 'Ridge alone (`ridge.json`)')}
{row('Climatology', 'Climatology (previous 3-season mean)')}

LightGBM alone had the lowest test error, but the ensemble was the model chosen on the validation seasons, before
the test seasons were seen; the paper reports both. Holm-adjusted Diebold-Mariano tests over eight test seasons
were not significant. Ridge failed badly in some seasons, which a post-hoc analysis traced to its row-wise
leave-one-out penalty choice (Section 5.8 of the paper); `predict.py` returns each component so you can use
LightGBM alone.

## Files

| file | content |
|---|---|
| `lightgbm.txt` | LightGBM booster (native text format) |
| `ridge.json` | Ridge pipeline as numbers: imputation medians, scaler means and scales, penalty, coefficients, intercept |
| `features.json` | the 81 input columns in order, their groups, target and units |
| `conformal.json` | 80% and 90% split-conformal offsets (kg/rai) and the calibration seasons |
| `metrics.json` | the rolling-origin test metrics above, for every model in the paper |
| `predict.py` | loader: `load(folder).predict(X)` -> LightGBM, Ridge, ensemble and interval columns |
| `sample_input.csv`, `sample_output.csv` | the 200 districts of season 2025 and their predictions (a check of the export, not a forecast) |

## Usage

```python
from huggingface_hub import snapshot_download
import sys, pandas as pd

folder = snapshot_download("{REPO}")
sys.path.insert(0, folder)
from predict import load

model = load(folder)
X = pd.read_csv(f"{{folder}}/sample_input.csv")
print(model.predict(X)[["ensemble", "lo80", "hi80"]].head())
```

To build the features for new seasons, use the code repository: put the raw tables in `data/raw/` (schema in
`docs/DATA_SCHEMA.md`), then

```python
from isanrice.features import build_features, load_raw
df, groups = build_features(load_raw())          # one row per district-season
X = df[df.year == 2026]                           # columns as in features.json
```

Features are built so that a season's row uses only information available by the end of that season; the
repository's tests check this.

## Limitations

- Synthetic training data: the simulator decides which mechanisms exist and how strong they are. The model has
  learned that world, not Thailand.
- One region, one crop (main-season rice), district level, end-of-season information set. Earlier cut-offs
  (July-October) need models trained on the matching features (E5 in the paper).
- The explanations of these models are only partly faithful: grouped SHAP recovered drought (r = 0.84) but
  misattributed heat stress to moisture and absorbed the technology trend into prices.

## Reproduce

`make setup && make all` in the code repository runs the experiments; `make model` retrains these files
(isanrice {__version__}, LightGBM {lightgbm.__version__}, scikit-learn {sklearn.__version__}).

## Citation

```bibtex
@misc{{chongmankhong2026isanrice,
  author = {{Chongmankhong, Thanatchon}},
  title  = {{Leakage-safe, explainable multi-source machine learning for district-level rice yield forecasting:
            a ground-truth simulation benchmark for Northeast Thailand}},
  year   = {{2026}},
  doi    = {{{DOI}}},
  url    = {{https://thanatchon36.github.io/isan-rice-yield-benchmark/paper/}},
  note   = {{Preprint; all results are from a synthetic benchmark}}
}}
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help=f"upload to huggingface.co/{REPO}")
    a = ap.parse_args()
    best = json.loads((C.RESULTS / "e1_best.json").read_text())
    assert best["main"] == "Ensemble" and best["target"] == "level", best
    df, groups = build_features(load_raw())
    feats = feature_list(groups)
    train = df[df[Y].notna()]

    ens = make_model("Ensemble", best["params"]).fit(train[feats], train[Y])
    # out-of-sample errors of every season with >= MIN_TRAIN_SEASONS seasons of history -> conformal offsets
    first = int(df.year.min()) + C.MIN_TRAIN_SEASONS
    cal_years = list(range(first, int(df.year.max()) + 1))
    oos, _ = E.rolling_cv(df, feats, "Ensemble", best["params"], test_years=cal_years, target="level")
    resid = (oos[Y] - oos.pred).to_numpy()
    quant = {str(int(round(100 * (1 - al)))): [float(np.quantile(resid, al / 2)), float(np.quantile(resid, 1 - al / 2))]
             for al in C.ALPHAS}

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    ens.lgb.booster_.save_model(str(STAGE / "lightgbm.txt"))
    (STAGE / "ridge.json").write_text(json.dumps(ridge_json(ens.ridge, feats)))
    (STAGE / "features.json").write_text(json.dumps({
        "features": feats, "groups": groups, "target": Y, "unit": C.YIELD_UNIT,
        "information_set": f"end of season (month {C.FORECAST_MONTH})", "id_columns": ["district_id", "year"],
        "training_seasons": [int(train.year.min()), int(train.year.max())], "n_train": int(len(train))}, indent=1))
    conf = {"method": "split-conformal on out-of-sample ensemble errors (rolling origin)", "quantiles": quant,
            "calibration_seasons": cal_years, "n_calibration": int(len(resid)),
            "ridge_alpha": float(ens.ridge[-1].alpha_)}
    (STAGE / "conformal.json").write_text(json.dumps(conf, indent=1))
    metrics = pd.read_csv(C.RESULTS / "e1_metrics.csv")
    metrics.to_json(STAGE / "metrics.json", orient="records", indent=1)
    (STAGE / "predict.py").write_text(PREDICT_PY)
    sample = df[df.year == df.year.max()][["district_id", "year", *feats]]
    sample.to_csv(STAGE / "sample_input.csv", index=False)

    # the exported files must reproduce the fitted models
    spec = importlib.util.spec_from_file_location("hf_predict", STAGE / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hub = mod.load(STAGE)
    err = 0.0
    for X in (train[feats], pd.read_csv(STAGE / "sample_input.csv")[feats]):  # every training row, then the CSV
        o = hub.predict(X)
        err = max(err, np.abs(o.lightgbm.to_numpy() - ens.lgb.predict(X)).max(),
                  np.abs(o.ridge.to_numpy() - ens.ridge.predict(X)).max())
    assert err < 1e-6, f"exported model differs from the fitted one by {err}"
    out = hub.predict(pd.read_csv(STAGE / "sample_input.csv"))
    shutil.rmtree(STAGE / "__pycache__", ignore_errors=True)
    pd.concat([sample[["district_id", "year"]].reset_index(drop=True), out.reset_index(drop=True).round(3)],
              axis=1).to_csv(STAGE / "sample_output.csv", index=False)
    (STAGE / "README.md").write_text(card(best, metrics, conf, len(train)), encoding="utf-8")
    print(f"staged {len(list(STAGE.iterdir()))} files in {STAGE.relative_to(ROOT)}; export matches (max |diff| {err:.1e}); "
          f"80% offsets {quant['80'][0]:+.1f}/{quant['80'][1]:+.1f} kg/rai from {len(resid)} forecasts")

    if a.upload:
        subprocess.run(["hf", "repo", "create", REPO, "--repo-type", "model", "--exist-ok"], check=True)
        subprocess.run(["hf", "upload", REPO, str(STAGE), ".", "--repo-type", "model",
                        "--commit-message", f"Upload models (isanrice {__version__})"], check=True)


if __name__ == "__main__":
    main()
