"""Leakage, shape and determinism tests:  make test"""
import numpy as np
import pandas as pd
import pytest

from isanrice import config as C
from isanrice.features import build_features, feature_list
from isanrice.simulate import COMPONENTS, simulate

T0 = 2016
PRICES = ["rice", "cassava", "sugarcane", "maize", "urea"]


@pytest.fixture(scope="module")
def world():
    return simulate(seed=7)


def _perturb_future(w):
    """Change every observation of seasons > T0 and every price of seasons >= T0."""
    w2 = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in w.items()}
    for k in ("yields", "climate_monthly", "satellite_monthly"):
        m = w2[k].year > T0
        cols = [c for c in w2[k].select_dtypes("number").columns if c not in ("year", "month")]
        w2[k].loc[m, cols] = w2[k].loc[m, cols] * 1.5 + 7
    w2["prices_annual"].loc[w2["prices_annual"].year >= T0, PRICES] *= 2
    return w2


def _past_rows(w, leak=None):
    df, g = build_features(w, leak=leak)
    return df[df.year <= T0].set_index(["district_id", "year"])[feature_list(g)]


def test_shapes_and_truth(world):
    assert len(world["yields"]) == C.N_DISTRICTS * len(C.YEARS)
    assert set(COMPONENTS) <= set(world["truth"].columns)
    _, g = build_features(world)
    f = feature_list(g)
    assert len(f) == len(set(f)) and not {C.YIELD_COL, "target_anom", "y_ref"} & set(f)


def test_determinism():
    pd.testing.assert_frame_equal(simulate(seed=3)["yields"], simulate(seed=3)["yields"])


def test_no_future_information(world):
    """Features of seasons <= T0 must not change when later seasons change."""
    pd.testing.assert_frame_equal(_past_rows(world), _past_rows(_perturb_future(world)))


@pytest.mark.parametrize("leak", ["future_anomaly", "same_year_prices", "nbr_current_yield"])
def test_leaky_variants_are_detected(world, leak):
    """Sanity check of the test itself: deliberately leaky feature sets MUST change."""
    w2 = _perturb_future(world)
    if leak == "nbr_current_yield":
        w2["yields"].loc[w2["yields"].year == T0, C.YIELD_COL] += 50
    a, b = _past_rows(world, leak), _past_rows(w2, leak)
    assert not np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
