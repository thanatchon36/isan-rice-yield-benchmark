"""Leakage-safe feature engineering: raw tables -> one row per (district, season) + 8 named groups.

Rules: yield history and anomalies use previous seasons only; prices are lagged one season
(known at sowing); weather and satellite data are truncated at `cutoff_month`; spatial lags use
neighbours' past yields or current-season observations. `leak=` builds deliberately leaky
variants used only in the leakage audit (experiment E2).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import config as C

KEY = ["district_id", "year"]
RAW_TABLES = ["districts", "climate_monthly", "satellite_monthly", "reservoir_annual", "prices_annual", "yields"]
SOIL_SUIT_ORDER = {"S1": 3, "S2": 2, "S3": 1, "N": 0}
LEAKS = (None, "same_year_prices", "nbr_current_yield", "future_anomaly")


def load_raw(path=C.DATA_RAW):
    return {t: pd.read_csv(path / f"{t}.csv") for t in RAW_TABLES}


def feature_list(groups):
    return [f for fs in groups.values() for f in fs]


def to_km(lon, lat, lat0=16.0):
    return np.column_stack([np.asarray(lon, float) * 111.32 * np.cos(np.radians(lat0)),
                            np.asarray(lat, float) * 110.57])


def _past_mean(s, min_periods=2):
    return s.shift(1).expanding(min_periods=min_periods).mean()


def _slope(a):
    a = a[~np.isnan(a)]
    return np.polyfit(np.arange(len(a)), a, 1)[0] if len(a) >= 3 else np.nan


def _vpd_kpa(t, rh):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3)) * (1 - rh / 100)


def build_features(raw, cutoff_month=C.FORECAST_MONTH, leak=None):
    assert leak in LEAKS, leak
    groups = {g: [] for g in C.GROUP_LABELS}
    dist = raw["districts"].copy()
    years = sorted(raw["yields"].year.unique())
    grid = pd.MultiIndex.from_product([dist.district_id, years], names=KEY).to_frame(index=False)
    df = (grid.merge(raw["yields"], on=KEY, how="left").merge(dist, on="district_id", how="left")
              .sort_values(KEY).reset_index(drop=True))
    ref_mean = (lambda s: s.transform("mean")) if leak == "future_anomaly" else (lambda s: s.transform(_past_mean))

    def merge(frame):
        nonlocal df
        df = df.merge(frame.reset_index(), on=KEY, how="left")

    # 1) previous yields
    g = df.groupby("district_id")[C.YIELD_COL]
    df["yield_lag1"], df["yield_lag2"] = g.shift(1), g.shift(2)
    df["yield_mean3"] = g.transform(lambda s: s.shift(1).rolling(C.HISTORY_YEARS).mean())
    df["yield_std3"] = g.transform(lambda s: s.shift(1).rolling(C.HISTORY_YEARS).std())
    df["yield_trend5"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=3).apply(_slope, raw=True))
    groups["history"] += ["yield_lag1", "yield_lag2", "yield_mean3", "yield_std3", "yield_trend5"]

    # 2) weather and satellite by growth stage
    cl = raw["climate_monthly"]
    cl = cl[cl.month.isin(C.SEASON_MONTHS) & (cl.month <= cutoff_month)].copy()
    cl["vpd_kpa"] = _vpd_kpa(cl.tmean_c, cl.rh_pct)
    cl["gdd"] = np.maximum(cl.tmean_c - 10.0, 0) * 30.4
    sa = raw["satellite_monthly"]
    sa = sa[sa.month.isin(C.SEASON_MONTHS) & (sa.month <= cutoff_month)].copy()
    for st, months in C.STAGES.items():
        ms = [m for m in months if m <= cutoff_month]
        if not ms:
            continue
        a = cl[cl.month.isin(ms)].groupby(KEY).agg(
            rain=("rain_mm", "sum"), sm_mean=("soil_moist", "mean"), sm_min=("soil_moist", "min"),
            rh=("rh_pct", "mean"), vpd=("vpd_kpa", "mean"), tmean=("tmean_c", "mean"), tmax=("tmax_c", "max"))
        s = sa[sa.month.isin(ms)].groupby(KEY).agg(ndvi_max=("ndvi", "max"), evi_max=("evi", "max"),
                                                   lswi_mean=("lswi", "mean"), lst_mean=("lst_c", "mean"))
        a.columns = [f"{c}_{st}" for c in a.columns]
        s.columns = [f"{c}_{st}" for c in s.columns]
        merge(a)
        merge(s)
        groups["water"] += [f"rain_{st}"]
        groups["moisture"] += [f"sm_mean_{st}", f"sm_min_{st}", f"rh_{st}", f"vpd_{st}"]
        groups["climate"] += [f"tmean_{st}", f"tmax_{st}"]
        groups["satellite"] += list(s.columns)

    # 3) season totals, water supply, climate normals
    cl["hot"] = cl.tmax_c > 35
    cl["dry"] = cl.month.isin([6, 7, 8, 9]) & (cl.rain_mm < 100)
    merge(cl.groupby(KEY).agg(rain_total=("rain_mm", "sum"), gdd_total=("gdd", "sum"),
                              heat_months=("hot", "sum"), dry_months=("dry", "sum")))
    df["rain_anom_pct"] = 100 * (df.rain_total / ref_mean(df.groupby("district_id").rain_total) - 1)
    df = df.merge(raw["reservoir_annual"], on=["basin", "year"], how="left")
    df["irr_supply"] = df.irrigation_share * df.storage_pct / 100
    groups["water"] += ["rain_total", "rain_anom_pct", "dry_months", "storage_pct", "irr_supply"]
    groups["climate"] += ["gdd_total", "heat_months", "rain_normal_mm"]

    # 4) satellite season metrics (cloud gaps filled by linear interpolation)
    cols = [m for m in C.SEASON_MONTHS if m <= cutoff_month]
    nd = sa.pivot_table(index=KEY, columns="month", values="ndvi", dropna=False).reindex(columns=cols)
    fil = nd.interpolate(axis=1, limit_direction="both")
    v = fil.to_numpy()
    allnan = np.isnan(v).all(1)
    ss = pd.DataFrame(index=fil.index)
    ss["ndvi_peak"] = fil.max(axis=1)
    ss["ndvi_auc"] = fil.sum(axis=1, min_count=1)
    ss["ndvi_peak_month"] = np.where(allnan, np.nan, np.array(cols)[np.where(np.isnan(v), -np.inf, v).argmax(1)])
    ss["ndvi_greenup"] = (ss.ndvi_peak - fil.iloc[:, 0]) / (ss.ndvi_peak_month - cols[0] + 1)
    ss["clear_frac"] = nd.notna().mean(axis=1)
    merge(ss)
    gd = df.groupby("district_id")
    df["ndvi_peak_anom"] = df.ndvi_peak - ref_mean(gd.ndvi_peak)
    df["ndvi_auc_anom"] = df.ndvi_auc - ref_mean(gd.ndvi_auc)
    groups["satellite"] += ["ndvi_peak", "ndvi_auc", "ndvi_peak_month", "ndvi_greenup", "clear_frac",
                            "ndvi_peak_anom", "ndvi_auc_anom"]

    # 5) prices: lag-1 local farm-gate prices = national - transport cost x distance
    lag = 0 if leak == "same_year_prices" else 1
    pr = raw["prices_annual"].set_index("year")
    prev = pr.reindex(df.year.to_numpy() - lag).reset_index(drop=True)
    prev2 = pr.reindex(df.year.to_numpy() - lag - 1).reset_index(drop=True)
    t, local = C.TARGET_CROP, {}
    for c in [t] + C.COMPETING_CROPS:
        local[c] = prev[c].to_numpy() - C.TRANSPORT_COST[c] * df[C.PRICE_DISTANCE_COL[c]].to_numpy()
        df[f"price_{c}_lag1"] = local[c]
        groups["price"].append(f"price_{c}_lag1")
    for c in C.COMPETING_CROPS:
        df[f"ratio_{c}_{t}"] = local[c] / local[t]
        groups["price"].append(f"ratio_{c}_{t}")
    rev_comp = np.mean([local[c] * C.TYPICAL_YIELD[c] for c in C.COMPETING_CROPS], axis=0)
    df["comp_revenue_index"] = np.log(rev_comp / (local[t] * C.TYPICAL_YIELD[t]))
    df[f"price_{t}_change"] = np.log(prev[t].to_numpy() / prev2[t].to_numpy())
    groups["price"] += ["comp_revenue_index", f"price_{t}_change"]
    if "urea" in pr.columns:
        df["price_urea_lag1"] = prev["urea"].to_numpy()
        df[f"{t}_urea_ratio"] = local[t] / df.price_urea_lag1
        groups["price"] += ["price_urea_lag1", f"{t}_urea_ratio"]

    # 6) geography & map layers
    df["soil_suit_ord"] = df.soil_suit.map(SOIL_SUIT_ORDER)
    groups["geography"] += ["elevation_m", "slope_deg", "sand_pct", "soc_pct", "soil_ph", "saline_share",
                            "soil_suit_ord", "lat", "lon"]
    groups["map"] += ["dist_river_km", "dist_mill_km", "dist_market_km", "irrigation_share", "paddy_share",
                      "flood_freq"]

    # 7) spatial lags over the k nearest districts
    order = dist.district_id.to_numpy()
    xy = to_km(dist.lon, dist.lat)
    nbr = NearestNeighbors(n_neighbors=C.K_NEIGHBORS + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    lags = [("yield_lag1", "nbr_yield_lag1"), ("ndvi_peak_anom", "nbr_ndvi_peak_anom")]
    if leak == "nbr_current_yield":
        lags.append((C.YIELD_COL, "nbr_yield_current"))
    for src, new in lags:
        mat = df.pivot(index="district_id", columns="year", values=src).reindex(order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nb = np.nanmean(mat.to_numpy()[nbr], axis=1)
        long = (pd.DataFrame(nb, index=pd.Index(order, name="district_id"), columns=mat.columns)
                  .reset_index().melt(id_vars="district_id", var_name="year", value_name=new))
        long["year"] = long.year.astype(int)
        df = df.merge(long, on=KEY, how="left")
        groups["map"].append(new)

    # 8) target and reference (previous-3-season mean)
    df["y_ref"] = df.yield_mean3
    df["target_anom"] = df[C.YIELD_COL] - df.y_ref
    df = df[df.y_ref.notna() & df[C.YIELD_COL].notna()].reset_index(drop=True)
    groups = {g: list(dict.fromkeys(f for f in fs if f in df.columns)) for g, fs in groups.items()}
    return df, {g: fs for g, fs in groups.items() if fs}
