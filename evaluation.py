"""Rolling-origin evaluation, leakage audit, metrics, inference, conformal intervals, SHAP."""
import numpy as np
import pandas as pd
from scipy import stats

import config as C
from models import LGB_BASE, LGB_SPACE, PanelFE, make_model

Y = C.YIELD_COL


def _target(tr, te, target):
    if target == "anomaly":
        return tr.target_anom, te.y_ref.to_numpy()
    return tr[Y], 0.0


def fit_predict(model, tr, te, feats, params=None, target="level", seed=C.SEED):
    if model == "Persistence":
        return te.yield_lag1.to_numpy(), None
    if model == "Climatology":
        return te.y_ref.to_numpy(), None
    if model == "PanelFE":
        f = [c for c in feats if not c.startswith(("yield_", "nbr_yield"))]
        m = PanelFE().fit(tr[f], tr[Y], tr.district_id, tr.year)
        return m.predict(te[f], te.district_id, te.year), m
    y, off = _target(tr, te, target)
    m = make_model(model, params, seed).fit(tr[feats], y)
    return off + m.predict(te[feats]), m


def explain(model, m, X):
    """Per-prediction SHAP values: TreeSHAP (LightGBM), exact linear SHAP (Ridge); the ensemble's
    SHAP is the average because Shapley values are additive in the model output."""
    lin = lambda p: p[-1].coef_ * p[1].transform(p[0].transform(X))
    if model == "LightGBM":
        return m.predict(X, pred_contrib=True)[:, :-1]
    if model == "Ridge":
        return lin(m)
    if model == "Ensemble":
        return 0.5 * m.lgb.predict(X, pred_contrib=True)[:, :-1] + 0.5 * lin(m.ridge)
    raise ValueError(model)


def add_conformal(pred, test_years, alphas):
    """Rolling split-conformal: empirical quantiles of PAST out-of-sample errors (seasons < t)."""
    resid, yrs, pv = (pred[Y] - pred.pred).to_numpy(), pred.year.to_numpy(), pred.pred.to_numpy()
    for a in alphas:
        lvl = int(round(100 * (1 - a)))
        lo, hi = np.full(len(pred), np.nan), np.full(len(pred), np.nan)
        for yr in test_years:
            cal, sel = resid[yrs < yr], yrs == yr
            lo[sel], hi[sel] = pv[sel] + np.quantile(cal, a / 2), pv[sel] + np.quantile(cal, 1 - a / 2)
        pred[f"lo{lvl}"], pred[f"hi{lvl}"] = lo, hi
    return pred[pred.year.isin(test_years)].reset_index(drop=True)


def rolling_cv(df, feats, model, params=None, test_years=C.TEST_YEARS, target="level", alphas=(),
               explain_=False, seed=C.SEED):
    """Season t is predicted by a model trained on seasons < t only."""
    years = list(test_years)
    if alphas:   # earlier out-of-sample seasons, used only to calibrate the intervals
        years = sorted(set(range(int(df.year.min()) + C.MIN_TRAIN_SEASONS, years[0])) | set(years))
    out, shap = [], []
    for yr in years:
        tr, te = df[df.year < yr], df[df.year == yr]
        if te.empty or tr.empty:
            continue
        p, m = fit_predict(model, tr, te, feats, params, target, seed)
        out.append(te[["district_id", "year", Y, "y_ref"]].assign(pred=p))
        if explain_ and yr in test_years and m is not None:
            shap.append(pd.DataFrame(explain(model, m, te[feats]), columns=feats, index=te.index))
    pred = pd.concat(out, ignore_index=True)
    if alphas:
        pred = add_conformal(pred, test_years, alphas)
    return pred, (pd.concat(shap) if shap else None)


def _trio(tr, te, feats, params, target, seed):
    y, off = _target(tr, te, target)
    pl = make_model("LightGBM", params, seed).fit(tr[feats], y).predict(te[feats])
    pr = make_model("Ridge").fit(tr[feats], y).predict(te[feats])
    base = te[["district_id", "year", Y, "y_ref"]]
    return {k: base.assign(pred=off + p) for k, p in (("LightGBM", pl), ("Ridge", pr), ("Ensemble", 0.5 * (pl + pr)))}


def rolling_trio(df, feats, params, test_years=C.TEST_YEARS, target="level", seed=C.SEED):
    """One LightGBM and one Ridge fit per fold -> LightGBM, Ridge and Ensemble predictions."""
    parts = [_trio(df[df.year < yr], df[df.year == yr], feats, params, target, seed) for yr in test_years]
    return {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in parts[0]}


def random_kfold(df, feats, params, target="level", k=5, eval_years=C.TEST_YEARS, seed=C.SEED):
    """Naive random k-fold CV over district-seasons (future seasons and same-season neighbours
    leak into training) - evaluated on the same test seasons for comparability."""
    fold = np.random.default_rng(seed).integers(0, k, len(df))
    parts = [_trio(df[fold != f], df[(fold == f) & df.year.isin(eval_years).to_numpy()], feats, params, target, seed)
             for f in range(k)]
    return {m: pd.concat([p[m] for p in parts], ignore_index=True) for m in parts[0]}


def score(pred):
    y, p, r = pred[Y].to_numpy(), pred.pred.to_numpy(), pred.y_ref.to_numpy()
    e, a = p - y, y - r
    mse = np.mean(e ** 2)
    return {"RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(np.abs(e))), "MAPE": float(100 * np.mean(np.abs(e) / y)),
            "R2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "R2_anomaly": float(1 - (e ** 2).sum() / ((a - a.mean()) ** 2).sum()),
            "Skill": float(1 - mse / np.mean(a ** 2))}


def _season_sse(p, seasons):
    e2 = (p.pred - p[Y]) ** 2
    g = e2.groupby(p.year)
    return g.sum().reindex(seasons).to_numpy(), g.size().reindex(seasons).to_numpy()


def season_bootstrap(preds, ref, B=C.BOOT_B, seed=C.SEED):
    """Season-block bootstrap (resample whole test seasons) for RMSE and RMSE - RMSE_ref."""
    seasons = sorted(preds[ref].year.unique())
    S = len(seasons)
    W = np.random.default_rng(seed).multinomial(S, np.full(S, 1 / S), size=B)
    s0, n0 = _season_sse(preds[ref], seasons)
    r0 = np.sqrt((W @ s0) / (W @ n0))
    rows = []
    for k, p in preds.items():
        s, n = _season_sse(p, seasons)
        r = np.sqrt((W @ s) / (W @ n))
        d = r - r0
        rows.append({"model": k, "RMSE_lo": np.percentile(r, 2.5), "RMSE_hi": np.percentile(r, 97.5),
                     "dRMSE": float(np.sqrt(s.sum() / n.sum()) - np.sqrt(s0.sum() / n0.sum())),
                     "dRMSE_lo": np.percentile(d, 2.5), "dRMSE_hi": np.percentile(d, 97.5)})
    return pd.DataFrame(rows)


def dm_test(pa, pb):
    """Diebold-Mariano test on season-level MSE differentials with the Harvey-Leybourne-Newbold
    small-sample correction (h = 1; t distribution with T-1 df). Positive stat: b is more accurate."""
    ma = ((pa.pred - pa[Y]) ** 2).groupby(pa.year).mean()
    mb = ((pb.pred - pb[Y]) ** 2).groupby(pb.year).mean()
    d = (ma - mb).to_numpy()
    T = len(d)
    if T < 3 or d.std() == 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(np.mean((d - d.mean()) ** 2) / T)
    stat = dm * np.sqrt((T + 1 - 2 + 0) / T)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=T - 1))


def holm(p):
    p = np.asarray(p, float)
    adj, ok = np.full(len(p), np.nan), np.where(~np.isnan(p))[0]
    run, m = 0.0, len(ok)
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def tune_lightgbm(df, feats, n_trials=C.N_TUNE_TRIALS, valid_years=C.TUNE_VALID_YEARS, seed=C.SEED):
    """Random search over hyper-parameters AND target formulation on validation seasons only."""
    rng = np.random.default_rng(seed)
    cands = [{}] + [{k: v[rng.integers(len(v))] for k, v in LGB_SPACE.items()} for _ in range(n_trials - 1)]
    trials = []
    for i, p in enumerate(cands):
        for target in ("level", "anomaly"):
            s = score(rolling_cv(df, feats, "LightGBM", p, valid_years, target=target)[0])
            trials.append({"trial": i, "target": target, **{k: p.get(k, LGB_BASE[k]) for k in LGB_SPACE}, **s})
    trials = pd.DataFrame(trials).sort_values("RMSE").reset_index(drop=True)
    top = trials.iloc[0]
    ints = ("num_leaves", "n_estimators", "min_child_samples", "max_depth")
    best = {k: int(top[k]) if k in ints else float(top[k]) for k in LGB_SPACE}
    return best, str(top["target"]), trials


def select_model(df, feats, params, target, valid_years=C.TUNE_VALID_YEARS):
    rows = [{"model": m, **score(rolling_cv(df, feats, m, params, valid_years, target=target)[0])}
            for m in ("LightGBM", "Ridge", "Ensemble")]
    t = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return str(t.model.iloc[0]), t


def group_shap(shap, groups):
    return pd.DataFrame({g: shap[[f for f in fs if f in shap.columns]].sum(axis=1) for g, fs in groups.items()})
