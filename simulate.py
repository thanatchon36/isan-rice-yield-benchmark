"""Process-informed synthetic benchmark with a KNOWN data-generating process (DGP) for
district-level main-season rice yields in Northeast Thailand.

Writes exactly the schema expected by features.py (so real data can replace it) plus the
true additive yield components (truth_components.csv) that enable ground-truth evaluation
of model explanations. Structural parameters live in PARAMS; SCENARIOS override them.
"""
import json

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

import config as C

REGION_OUTLINE = np.array([
    (101.3, 17.6), (101.8, 18.1), (102.1, 17.9), (102.7, 17.9), (103.4, 18.4),
    (104.0, 18.3), (104.7, 17.6), (104.8, 16.6), (105.4, 16.0), (105.6, 15.0),
    (105.2, 14.3), (104.0, 14.3), (103.0, 14.3), (102.4, 14.0), (101.8, 14.2),
    (101.3, 14.5), (101.4, 15.2), (101.3, 16.0), (101.4, 16.8), (101.3, 17.6),
])
RIVERS = {
    "mekong": [(101.8, 18.1), (102.7, 17.9), (103.4, 18.4), (104.0, 18.3),
               (104.7, 17.6), (104.8, 16.6), (105.4, 16.0), (105.6, 15.2)],
    "mun": [(102.0, 14.9), (103.0, 15.2), (103.5, 15.3), (104.3, 15.1), (104.9, 15.25), (105.5, 15.3)],
    "chi": [(101.9, 16.0), (102.8, 16.2), (103.3, 16.2), (103.7, 16.0), (104.2, 15.8), (104.6, 15.25)],
}
TOWNS = {
    "Nakhon Ratchasima": (102.10, 14.97), "Khon Kaen": (102.83, 16.43), "Udon Thani": (102.79, 17.41),
    "Ubon Ratchathani": (104.85, 15.24), "Buriram": (103.10, 15.00), "Surin": (103.49, 14.88),
    "Si Sa Ket": (104.32, 15.12), "Roi Et": (103.65, 16.05), "Maha Sarakham": (103.30, 16.18),
    "Kalasin": (103.51, 16.43), "Sakon Nakhon": (104.15, 17.16), "Nakhon Phanom": (104.78, 17.40),
    "Mukdahan": (104.72, 16.54), "Yasothon": (104.15, 15.79), "Amnat Charoen": (104.62, 15.86),
    "Chaiyaphum": (102.03, 15.81), "Loei": (101.72, 17.49), "Nong Bua Lamphu": (102.43, 17.20),
    "Nong Khai": (102.74, 17.88), "Bueng Kan": (103.65, 18.36),
}
SUGAR_MILLS = np.array([(102.75, 16.35), (102.55, 17.20), (102.15, 15.05), (101.95, 15.85), (103.45, 16.55),
                        (103.15, 15.10), (101.75, 16.65), (102.30, 16.90), (103.95, 16.95)])
RAIN_FRAC = {5: .11, 6: .13, 7: .14, 8: .18, 9: .20, 10: .09, 11: .02}
T_CLIM = {5: 29.6, 6: 28.9, 7: 28.5, 8: 28.1, 9: 27.9, 10: 27.3, 11: 25.6}
RH_CLIM = {5: 71, 6: 76, 7: 78, 8: 81, 9: 83, 10: 78, 11: 70}
CLOUD_P = {5: .15, 6: .35, 7: .45, 8: .50, 9: .45, 10: .25, 11: .10}
ENSO_HINT = {2010: .3, 2011: -1.4, 2015: 1.8, 2016: .6, 2017: -.7, 2019: 1.3, 2020: .8,
             2021: -.8, 2022: -1.2, 2023: 1.5, 2024: .4}          # + = dry/hot phase

PARAMS = dict(
    base_mean=365.0, b_soil=30.0, b_irrig=115.0, b_unobs_field=18.0, b_unobs_iid=12.0, trend=2.2,
    est_thr=0.24, est_coef=1100.0, est_irr=0.12, late_thr=0.22, late_coef=900.0, late_irr=0.10,
    flood_thr=1.30, flood_coef=160.0, flood_cap=150.0, heat_thr=34.0, heat_coef=6.0,
    dis_thr=84.0, dis_coef=4.0, sal_thr=0.26, sal_coef=250.0,
    price_comp=12.0, price_input=10.0, price_feedback=0.25, price_phi=0.6,
    mgmt_rho=0.6, mgmt_sd=14.0, pest_amp=75.0, pest_radius=70.0, noise_sd=16.0,
    nl_mult=1.0, compound=0.0, map_err_mult=1.0, cloud_mult=1.0,
)
SCENARIOS = {
    "base": {},
    "nonlinear": {"nl_mult": 1.8, "compound": 60.0},   # stronger thresholds + compound heat-drought
    "map_error": {"map_err_mult": 3.0},                 # 3x measurement error in map layers
    "clouds": {"cloud_mult": 1.7},                      # 1.7x monsoon cloud contamination
    "no_pest": {"pest_amp": 0.0},                       # no satellite-only signal
}
COMPONENTS = ["base", "trend", "drought", "flood", "heat", "disease", "salinity",
              "price", "mgmt", "pest", "noise"]
OBSERVED_BY = {   # feature groups that (noisily) observe each true component
    "base": ["geography", "map", "history"], "trend": ["history"],
    "drought": ["moisture", "water", "satellite"], "flood": ["water", "map", "satellite"],
    "heat": ["climate"], "disease": ["moisture"], "salinity": ["geography", "moisture"],
    "price": ["price", "map"], "mgmt": ["history", "satellite"], "pest": ["satellite"], "noise": [],
}


def to_km(lon, lat, lat0=16.0):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat0)), lat * 110.57])


def dist_to_polyline_km(lonlat, poly):
    P, V = to_km(lonlat[:, 0], lonlat[:, 1]), to_km(*np.asarray(poly, float).T)
    d = np.full(len(P), np.inf)
    for a, b in zip(V[:-1], V[1:]):
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    return d


def dist_to_points_km(lonlat, pts):
    P, Q = to_km(lonlat[:, 0], lonlat[:, 1]), to_km(pts[:, 0], pts[:, 1])
    return np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1))


def sample_districts(n, rng, n_cand=20):
    """Evenly spread district centroids inside the region (Mitchell's best-candidate)."""
    path, lo, hi = MplPath(REGION_OUTLINE), REGION_OUTLINE.min(0), REGION_OUTLINE.max(0)

    def draw(k):
        out = np.empty((0, 2))
        while len(out) < k:
            c = rng.uniform(lo, hi, size=(4 * k, 2))
            out = np.vstack([out, c[path.contains_points(c)]])
        return out[:k]

    pts = draw(1)
    for _ in range(n - 1):
        cand = draw(n_cand)
        pts = np.vstack([pts, cand[dist_to_points_km(cand, pts).min(1).argmax()]])
    return pts


class FieldSampler:
    """Spatially correlated Gaussian fields (kernel-smoothed white noise, unit variance)."""

    def __init__(self, xy_km, rng):
        self.d2, self.rng, self.cache = ((xy_km[:, None, :] - xy_km[None, :, :]) ** 2).sum(-1), rng, {}

    def __call__(self, length_km):
        if length_km not in self.cache:
            self.cache[length_km] = np.exp(-self.d2 / (2 * length_km ** 2))
        f = self.cache[length_km] @ self.rng.standard_normal(len(self.d2))
        return (f - f.mean()) / f.std()


def simulate(seed=C.SEED, scenario="base", out_dir=None):
    P = {**PARAMS, **SCENARIOS[scenario]}
    rng = np.random.default_rng(seed)
    n, years, months = C.N_DISTRICTS, [int(y) for y in C.YEARS], C.SEASON_MONTHS
    mi = {m: j for j, m in enumerate(months)}
    ll = sample_districts(n, rng)
    lon, lat = ll[:, 0], ll[:, 1]
    ids = np.array([f"D{i + 1:03d}" for i in range(n)])
    field = FieldSampler(to_km(lon, lat), rng)
    z = lambda a: (a - a.mean()) / a.std()

    # ---- map layers (true values) -----------------------------------------------------
    d_riv = {k: dist_to_polyline_km(ll, v) for k, v in RIVERS.items()}
    D = np.column_stack(list(d_riv.values()))
    dist_river, basin = D.min(1), np.array(list(d_riv))[D.argmin(1)]
    d_town = dist_to_points_km(ll, np.array(list(TOWNS.values())))
    province, dist_market = np.array(list(TOWNS))[d_town.argmin(1)], d_town.min(1)
    dist_mill = dist_to_points_km(ll, SUGAR_MILLS).min(1)
    elev = np.clip(170 + 45 * field(80) + 280 * np.exp(-((lon - 101.25) / 0.35) ** 2)
                   + 160 * np.exp(-((lat - 14.15) / 0.25) ** 2)
                   + 140 * np.exp(-((lon - 103.9) / 0.45) ** 2 - ((lat - 17.0) / 0.25) ** 2)
                   - 30 * np.exp(-dist_river / 15), 110, 900)
    slope = np.clip(0.6 + 0.022 * (elev - 150) + 0.8 * np.abs(field(40)), 0.2, 25)
    sand = np.clip(62 + 10 * field(80) + 0.03 * (elev - 170) - 8 * np.exp(-dist_river / 12), 30, 92)
    soc = np.clip(0.75 - 0.012 * (sand - 62) + 0.12 * field(40) + 0.25 * np.exp(-dist_river / 10), 0.2, 2.2)
    ph = np.clip(5.3 + 0.35 * field(80), 4.3, 7.0)
    saline = np.clip((0.25 * np.exp(-((lon - 102.9) / 0.9) ** 2 - ((lat - 15.9) / 0.6) ** 2)
                      + 0.18 * np.exp(-((lon - 103.8) / 0.6) ** 2 - ((lat - 17.3) / 0.35) ** 2)
                      + 0.04 * field(40)) * (elev < 260), 0, 0.5)
    irrig = np.clip(0.04 + 0.45 * np.exp(-dist_river / 10) + 0.05 * field(40), 0, 0.85)
    paddy = np.clip(0.62 - 0.0012 * (elev - 170) - 0.012 * slope + 0.08 * field(80), 0.08, 0.92)
    flood_freq = np.clip(0.35 * np.exp(-dist_river / 8) * (elev < 200) + 0.03 * field(40) + 0.02, 0, 0.6)
    rain_normal = 1050 + 120 * (lon - 101.5) + 85 * (lat - 14.5) + 70 * field(150)
    suit = 0.6 * z(soc) - 0.6 * z(sand) - 0.5 * z(saline) - 0.4 * z(slope) + 0.2 * field(40)
    q = np.quantile(suit, [.15, .45, .80])
    soil_suit = np.select([suit >= q[2], suit >= q[1], suit >= q[0]], ["S1", "S2", "S3"], "N")
    me = P["map_err_mult"]                      # observed layers carry measurement error
    obs = lambda a, sd, lo, hi: np.clip(a + me * sd * rng.standard_normal(n), lo, hi)
    districts = pd.DataFrame({
        "district_id": ids, "province": province, "lon": lon.round(4), "lat": lat.round(4), "basin": basin,
        "elevation_m": elev.round(1), "slope_deg": slope.round(2), "sand_pct": obs(sand, 4, 20, 95).round(1),
        "soc_pct": obs(soc, 0.12, 0.1, 2.5).round(3), "soil_ph": ph.round(2),
        "saline_share": obs(saline, 0.03, 0, 0.6).round(3), "soil_suit": soil_suit,
        "irrigation_share": obs(irrig, 0.05, 0, 0.9).round(3), "paddy_share": paddy.round(3),
        "flood_freq": flood_freq.round(3), "rain_normal_mm": rain_normal.round(0),
        "dist_river_km": dist_river.round(2), "dist_mill_km": dist_mill.round(2),
        "dist_market_km": dist_market.round(2)})

    # ---- year drivers: ENSO-like index, reservoirs, prices ----------------------------
    enso = {y: float(rng.normal(0, .5)) for y in range(years[0] - 1, years[-1] + 1)}
    enso.update({y: v for y, v in ENSO_HINT.items() if y in enso})
    basin_eff = {"mekong": -8.0, "mun": 0.0, "chi": 6.0}
    reservoir = pd.DataFrame([{"basin": b, "year": y, "storage_pct": float(np.clip(
        58 - 14 * enso[y - 1] + basin_eff[b] + rng.normal(0, 7), 12, 95))} for b in RIVERS for y in years])
    res_lookup = reservoir.set_index(["basin", "year"]).storage_pct
    names = ["rice", "cassava", "sugarcane", "maize", "urea"]
    ix = {k: i for i, k in enumerate(names)}
    mu, sig = np.log([10.5, 2.2, 1.0, 8.0, 14.0]), np.array([.11, .17, .11, .13, .13])
    corr = np.full((5, 5), .35)
    corr[4, :] = corr[:, 4] = .2
    np.fill_diagonal(corr, 1)
    L = np.linalg.cholesky(corr)
    shocks = {2008: {"urea": .35, "rice": .15}, 2022: {"urea": .55}, 2023: {"urea": .25, "rice": .12}}
    price, lp = {}, mu.copy()
    for y in range(years[0] - 2, years[-1] + 1):
        lp = mu + P["price_phi"] * (lp - mu) + sig * (L @ rng.standard_normal(5))
        price[y] = np.exp(lp + np.array([shocks.get(y, {}).get(k, 0.0) for k in names]))
    dist_for = {"rice": dist_market, "cassava": dist_market, "sugarcane": dist_mill, "maize": dist_market}
    w_cane = np.exp(-dist_mill / 40)                   # cane only pays near a sugar mill
    TY = C.TYPICAL_YIELD

    def econ(y):                                       # indices from final prices of season y-1
        p = price[y - 1]
        loc = {c: p[ix[c]] - C.TRANSPORT_COST[c] * dist_for[c] for c in names[:4]}
        comp = np.log((loc["cassava"] * TY["cassava"] + w_cane * loc["sugarcane"] * TY["sugarcane"]
                       + loc["maize"] * TY["maize"]) / (2 + w_cane) / (loc["rice"] * TY["rice"]))
        return comp, np.log(loc["rice"] / p[ix["urea"]])

    ref = np.array([econ(y) for y in years])           # normalising constants (pre-feedback)
    cmu, csd, fmu, fsd = ref[:, 0].mean(), ref[:, 0].std(), ref[:, 1].mean(), ref[:, 1].std()
    early = [y for y in years if years[0] + 3 <= y < C.TEST_YEARS[0]]
    pest_years = [int(rng.choice(early))] + [int(y) for y in rng.choice(C.TEST_YEARS, 2, replace=False)]
    pest, pest_events = {}, []
    for y in pest_years:                                # local outbreaks (e.g. planthopper)
        c = int(rng.integers(n))
        dd = dist_to_points_km(ll, ll[[c]])[:, 0]
        pest[y] = pest.get(y, 0.0) - P["pest_amp"] * np.exp(-dd ** 2 / (2 * P["pest_radius"] ** 2))
        pest_events.append({"year": y, "centre": ids[c]})

    # ---- seasons ------------------------------------------------------------------------
    soil_q = z(0.6 * z(soc) - 0.6 * z(sand) - 0.5 * z(saline) - 0.3 * z(slope))
    base = (P["base_mean"] + P["b_soil"] * soil_q + P["b_irrig"] * irrig
            + P["b_unobs_field"] * field(60) + P["b_unobs_iid"] * rng.standard_normal(n))
    mgmt = P["mgmt_sd"] * rng.standard_normal(n)
    clim_rows, sat_rows, yld_rows, truth_rows = [], [], [], []
    nl = P["nl_mult"]
    for y in years:
        e, A, TA = enso[y], field(150), field(150)
        rain, tmean, tmax, rh, sm = (np.zeros((len(months), n)) for _ in range(5))
        sm_prev = np.clip(0.15 + 0.02 * rng.standard_normal(n), 0.08, 0.3)
        for j, m in enumerate(months):
            zr = (-0.20 * e * (1.4 if m in (6, 7) else 1.0) + 0.18 * A + 0.30 * field(80)
                  + 0.08 * rng.standard_normal(n))
            rain[j] = rain_normal * RAIN_FRAC[m] * np.exp(zr - 0.065)
            zs, tanom = np.clip(zr / 0.36, -3, 3), 0.30 * e + 0.25 * TA
            tmean[j] = (T_CLIM[m] - 0.0065 * (elev - 170) + 0.035 * (y - 2008) + tanom - 0.35 * zs
                        + 0.15 * rng.standard_normal(n))
            tmax[j] = tmean[j] + 6.0 + 0.8 * np.maximum(0, -zs) + 0.3 * rng.standard_normal(n)
            rh[j] = np.clip(RH_CLIM[m] + 4.0 * zs - 1.5 * tanom + 1.5 * rng.standard_normal(n), 40, 98)
            sm[j] = 0.45 * sm_prev + 0.55 * (0.08 + 0.42 * (1 - np.exp(-rain[j] / 200)) * (1.30 - sand / 100))
            sm_prev = sm[j]

        mgmt = P["mgmt_rho"] * mgmt + P["mgmt_sd"] * rng.standard_normal(n)
        irr_supply = irrig * np.array([res_lookup[(b, y)] for b in basin]) / 100
        sm_jj, sm_so = sm[[mi[6], mi[7]]].mean(0), sm[[mi[9], mi[10]]].mean(0)
        drought = nl * (-P["est_coef"] * np.maximum(0, P["est_thr"] - (sm_jj + P["est_irr"] * irr_supply))
                        - P["late_coef"] * np.maximum(0, P["late_thr"] - (sm_so + P["late_irr"] * irr_supply)))
        drought = drought - P["compound"] * ((sm_jj < 0.21) & (tmax[mi[7]] > 35.5))
        wet = (rain[mi[8]] + rain[mi[9]]) / (rain_normal * (RAIN_FRAC[8] + RAIN_FRAC[9]))
        flood = nl * np.maximum(-P["flood_cap"], -P["flood_coef"] * (flood_freq / 0.3) * np.maximum(0, wet - P["flood_thr"]))
        heat = -nl * P["heat_coef"] * (np.maximum(0, tmax[mi[9]] - P["heat_thr"]) + np.maximum(0, tmax[mi[10]] - P["heat_thr"]))
        disease = -nl * P["dis_coef"] * np.maximum(0, rh[[mi[8], mi[9]]].mean(0) - P["dis_thr"])
        salinity = -nl * P["sal_coef"] * saline * np.maximum(0, P["sal_thr"] - sm_jj)
        comp, fert = econ(y)
        price_eff = (-P["price_comp"] * (comp - cmu) / csd * (1 - irrig)
                     + P["price_input"] * (fert - fmu) / fsd * (0.6 + irrig))
        trend = P["trend"] * (y - 2008)
        pest_y = pest.get(y, np.zeros(n))
        noise = P["noise_sd"] * rng.standard_normal(n)
        visible = base + trend + drought + flood + heat + disease + salinity + price_eff + mgmt + pest_y
        yld = np.clip(visible + noise, 120, 850)
        price[y][ix["rice"]] *= np.exp(-P["price_feedback"] * (yld.mean() - (base.mean() + trend)) / 100)
        truth_rows.append(pd.DataFrame({
            "district_id": ids, "year": y, "base": base - base.mean(), "trend": trend, "drought": drought,
            "flood": flood, "heat": heat, "disease": disease, "salinity": salinity, "price": price_eff,
            "mgmt": mgmt, "pest": pest_y, "noise": noise}).round(3))

        sm_mj = sm[[mi[5], mi[6]]].mean(0)               # satellite: canopy driven by visible yield
        peak_m = 9.4 + 0.8 * np.clip((0.25 - sm_mj) / 0.05, 0, 1.5)
        ndvi_peak = 0.42 + 0.38 / (1 + np.exp(-(visible - 380) / 85))
        evi_peak = 0.22 + 0.00075 * (visible - 250)
        flood_sev = np.clip(-flood / 120, 0, 1)
        for j, m in enumerate(months):
            shape = np.exp(-((m - peak_m) / 1.7) ** 2)
            nd = 0.20 + (ndvi_peak - 0.20) * shape
            if m in (9, 10):
                nd = nd * (1 - 0.25 * flood_sev)
            ndvi = 0.85 * nd + 0.15 * 0.62 + 0.025 * rng.standard_normal(n)
            evi = 0.12 + (evi_peak - 0.12) * shape + 0.02 * rng.standard_normal(n)
            lswi = (-0.10 + 0.9 * (sm[j] - 0.2) + 0.03 * rng.standard_normal(n)
                    + (0.12 * (0.5 + irr_supply) if m in (6, 7) else 0) + (0.25 * flood_sev if m in (9, 10) else 0))
            lst = tmax[j] + 1.5 + 25 * (0.28 - sm[j]) - 6 * (ndvi - 0.5) + 0.8 * rng.standard_normal(n)
            pc = min(0.9, CLOUD_P[m] * P["cloud_mult"])
            u = rng.random(n)
            gone, cont = u < 0.25 * pc, (u >= 0.25 * pc) & (u < pc)
            ndvi = np.where(cont, ndvi - rng.uniform(.05, .18, n), ndvi)
            evi = np.where(cont, evi - rng.uniform(.03, .10, n), evi)
            lswi = np.where(cont, lswi + rng.uniform(0, .08, n), lswi)
            lst = np.where(cont, lst - rng.uniform(2, 6, n), lst)
            na = lambda a, d: np.where(gone, np.nan, a).round(d)
            sat_rows.append(pd.DataFrame({"district_id": ids, "year": y, "month": m, "ndvi": na(ndvi, 4),
                                          "evi": na(evi, 4), "lswi": na(lswi, 4), "lst_c": na(lst, 2)}))
            clim_rows.append(pd.DataFrame({
                "district_id": ids, "year": y, "month": m, "rain_mm": rain[j].round(1),
                "tmean_c": tmean[j].round(2), "tmax_c": tmax[j].round(2), "rh_pct": rh[j].round(1),
                "soil_moist": np.clip(sm[j] + 0.025 * rng.standard_normal(n), 0.03, 0.55).round(3)}))
        yld_rows.append(pd.DataFrame({"district_id": ids, "year": y, C.YIELD_COL: yld.round(1)}))

    prices = pd.DataFrame([[y, *price[y]] for y in sorted(price)], columns=["year", *names]).round(3)
    tables = {"districts": districts, "climate_monthly": pd.concat(clim_rows, ignore_index=True),
              "satellite_monthly": pd.concat(sat_rows, ignore_index=True), "reservoir_annual": reservoir.round(1),
              "prices_annual": prices, "yields": pd.concat(yld_rows, ignore_index=True),
              "truth": pd.concat(truth_rows, ignore_index=True)}
    meta = {"seed": seed, "scenario": scenario, "params": P, "enso": enso, "pest_events": pest_events}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, t in tables.items():
            t.to_csv(out_dir / ("truth_components.csv" if k == "truth" else f"{k}.csv"), index=False)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    tables["meta"] = meta
    return tables


if __name__ == "__main__":
    t = simulate(out_dir=C.DATA_RAW)
    print({k: v.shape for k, v in t.items() if hasattr(v, "shape")}, t["meta"]["pest_events"])
