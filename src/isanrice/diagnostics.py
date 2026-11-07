"""POST-HOC diagnostics (not part of the pre-specified protocol; reported as such in the paper).
1) Why did Ridge fail in the first test season? Compare RidgeCV's row-wise LOO penalty with a
   season-blocked (leave-one-season-out) penalty, season by season.
2) Attribution faithfulness of LightGBM alone (TreeSHAP) for comparison with the main model."""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import evaluation as E
from .features import build_features, feature_list, load_raw
from .models import make_model
from .simulate import COMPONENTS, OBSERVED_BY

warnings.filterwarnings("ignore")
R, Y, ALPHAS = C.RESULTS, C.YIELD_COL, np.logspace(-2, 3, 30)   # same grid as RidgeCV in models.py


def ridge(a):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=a))


def loso_alpha(tr, feats, y):
    errs = []
    for a in ALPHAS:
        se = 0.0
        for s in sorted(tr.year.unique()):
            fit = (tr.year != s).to_numpy()
            m = ridge(a).fit(tr.loc[fit, feats], y[fit])
            se += ((m.predict(tr.loc[~fit, feats]) - y[~fit]) ** 2).sum()
        errs.append(se)
    return float(ALPHAS[int(np.argmin(errs))])


def main():
    best = json.loads((R / "e1_best.json").read_text())
    params, target = best["params"], best["target"]
    raw = load_raw()
    df, groups = build_features(raw)
    feats = feature_list(groups)
    rows, shap = [], []
    for yr in C.TEST_YEARS:
        tr, te = df[df.year < yr], df[df.year == yr]
        y = (tr[Y] if target == "level" else tr.target_anom).to_numpy()
        off = 0.0 if target == "level" else te.y_ref.to_numpy()
        m1 = make_model("Ridge").fit(tr[feats], y)
        a2 = loso_alpha(tr, feats, y)
        m2 = ridge(a2).fit(tr[feats], y)
        lg = make_model("LightGBM", params).fit(tr[feats], y)
        pl, p1, p2 = lg.predict(te[feats]), m1.predict(te[feats]), m2.predict(te[feats])
        rm = lambda p: float(np.sqrt(np.mean((off + p - te[Y].to_numpy()) ** 2)))
        rows.append({"year": yr, "alpha_rowloo": float(m1[-1].alpha_), "alpha_loso": a2, "rmse_ridge_rowloo": rm(p1),
                     "rmse_ridge_loso": rm(p2), "rmse_lgb": rm(pl), "rmse_ens_rowloo": rm(0.5 * (pl + p1)),
                     "rmse_ens_loso": rm(0.5 * (pl + p2))})
        shap.append(pd.DataFrame(lg.predict(te[feats], pred_contrib=True)[:, :-1], columns=feats, index=te.index))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(R / "diag_ridge_season.csv", index=False)
    sh = pd.concat(shap)
    gs = E.group_shap(sh, groups).reset_index(drop=True)
    keys = df.loc[sh.index, ["district_id", "year"]].reset_index(drop=True)
    tr_ = keys.merge(pd.read_csv(C.DATA_RAW / "truth_components.csv"), on=["district_id", "year"], how="left")
    rec = [{"component": c, "r_observed_lgb": tr_[c].corr(gs[[g for g in OBSERVED_BY[c] if g in gs]].sum(axis=1))
            if OBSERVED_BY[c] else np.nan} for c in COMPONENTS]
    pd.DataFrame(rec).to_csv(R / "e4_recovery_lgb.csv", index=False)
    f2g = {f: g for g, fs in groups.items() for f in fs}
    imp = sh.abs().mean()
    (100 * imp.groupby(imp.index.map(f2g)).sum() / imp.sum()).rename("share_pct").to_csv(R / "e4_shap_share_lgb.csv")


if __name__ == "__main__":
    main()
