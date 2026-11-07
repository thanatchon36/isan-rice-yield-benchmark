"""Model zoo: econometric, linear, tree-based, neural and ensemble learners."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C

LGB_BASE = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, max_depth=-1, n_jobs=1, verbose=-1)
LGB_SPACE = dict(num_leaves=[7, 15, 31, 63], learning_rate=[0.01, 0.02, 0.03, 0.05, 0.08],
                 n_estimators=[200, 400, 700, 1000], min_child_samples=[10, 20, 40, 80],
                 subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.3, 0.5, 0.7, 0.9],
                 reg_lambda=[0.0, 1.0, 5.0, 20.0], max_depth=[-1, 4, 6, 8])


def _imputer():
    return SimpleImputer(strategy="median", keep_empty_features=True)


def make_model(name, params=None, seed=C.SEED):
    if name == "LightGBM":
        return lgb.LGBMRegressor(**{**LGB_BASE, "random_state": seed, **(params or {})})
    if name == "Ridge":
        return make_pipeline(_imputer(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    if name == "RandomForest":
        return make_pipeline(_imputer(), RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                                               max_features=0.33, n_jobs=-1, random_state=seed))
    if name == "MLP":
        net = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, early_stopping=True,
                           validation_fraction=0.15, n_iter_no_change=20, max_iter=600, random_state=seed)
        return TransformedTargetRegressor(regressor=make_pipeline(_imputer(), StandardScaler(), net),
                                          transformer=StandardScaler())
    if name == "Ensemble":
        return LgbRidgeEnsemble(params, seed)
    raise ValueError(name)


class LgbRidgeEnsemble:
    """Equal-weight average of LightGBM (thresholds, interactions) and Ridge (smooth, extrapolates)."""

    def __init__(self, params=None, seed=C.SEED):
        self.lgb, self.ridge = make_model("LightGBM", params, seed), make_model("Ridge")

    def fit(self, X, y):
        self.lgb.fit(X, y)
        self.ridge.fit(X, y)
        return self

    def predict(self, X):
        return 0.5 * self.lgb.predict(X) + 0.5 * self.ridge.predict(X)


class PanelFE:
    """Econometric baseline: district fixed effects + linear trend + time-varying covariates
    (within estimator with a light ridge penalty). Lagged yields are excluded (Nickell bias)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, district, year):
        X = X.assign(_trend=year.to_numpy() - 2008)
        g = district.to_numpy()
        within = X - X.groupby(g).transform("mean")
        self.cols = [c for c in X.columns if np.nanstd(within[c].to_numpy()) > 1e-9]
        Z = X[self.cols]
        self.med = Z.median()
        Z = Z.fillna(self.med)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1)
        Z = (Z - self.mu) / self.sd
        Zd = Z - Z.groupby(g).transform("mean")
        yv = y.to_numpy()
        yd = yv - pd.Series(yv).groupby(g).transform("mean").to_numpy()
        self.reg = Ridge(alpha=self.alpha, fit_intercept=False).fit(Zd.to_numpy(), yd)
        resid = yv - Z.to_numpy() @ self.reg.coef_
        self.fe = pd.Series(resid).groupby(g).mean()
        return self

    def predict(self, X, district, year):
        X = X.assign(_trend=year.to_numpy() - 2008)
        Z = (X[self.cols].fillna(self.med) - self.mu) / self.sd
        fe = pd.Series(district.to_numpy()).map(self.fe).fillna(self.fe.mean()).to_numpy()
        return fe + Z.to_numpy() @ self.reg.coef_
