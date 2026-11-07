"""Experiments E1-E7. Each caches its outputs in results/ and is skipped once done.
    python experiments.py                  # everything
    python experiments.py --only e3 --force
    python experiments.py --quick          # smoke test, tiny settings -> results_quick/
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import config as C
import evaluation as E
from features import build_features, feature_list, load_raw
from simulate import COMPONENTS, OBSERVED_BY, simulate

warnings.filterwarnings("ignore")
T0, Y = time.time(), C.YIELD_COL


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def settings(quick):
    if quick:
        return dict(out=C.ROOT / "results_quick", trials=2, test=[2024, 2025], valid=[2016, 2017], seeds=1,
                    scen=["nonlinear"], lead=[9, 11], B=200)
    return dict(out=C.RESULTS, trials=C.N_TUNE_TRIALS, test=C.TEST_YEARS, valid=C.TUNE_VALID_YEARS,
                seeds=C.N_SEEDS, scen=C.SCENARIOS, lead=C.LEAD_TIME_MONTHS, B=C.BOOT_B)


def e1(ctx):
    """Tuning, model selection (validation seasons) and model comparison (test seasons)."""
    df, feats, S, out = ctx["df"], ctx["feats"], ctx["S"], ctx["out"]
    best, target, trials = E.tune_lightgbm(df, feats, S["trials"], S["valid"])
    main, sel = E.select_model(df, feats, best, target, S["valid"])
    log(f"E1 tuned: target={target} params={best}; validation -> main={main}")
    alt = "anomaly" if target == "level" else "level"
    runs = {"Persistence": dict(model="Persistence"), "Climatology": dict(model="Climatology"),
            "PanelFE": dict(model="PanelFE"), "Ridge": dict(model="Ridge", target=target),
            "RandomForest": dict(model="RandomForest", target=target), "MLP": dict(model="MLP", target=target),
            "LightGBM_alt": dict(model="LightGBM", params=best, target=alt),
            "LightGBM": dict(model="LightGBM", params=best, target=target),
            "Ensemble": dict(model="Ensemble", params=best, target=target)}
    runs[main].update(alphas=C.ALPHAS, explain_=True)
    preds, shap = {}, None
    for k, kw in runs.items():
        p, sv = E.rolling_cv(df, feats, test_years=S["test"], **kw)
        preds[k], shap = p, (sv if sv is not None else shap)
        log(f"E1 {k:14s} RMSE {E.score(p)['RMSE']:.2f}")
    long = pd.concat([p.assign(model=k) for k, p in preds.items()], ignore_index=True)
    long.to_csv(out / "e1_predictions.csv", index=False)
    met = pd.DataFrame([{"model": k, **E.score(p)} for k, p in preds.items()])
    met = met.merge(E.season_bootstrap(preds, main, S["B"]), on="model")
    dm = pd.DataFrame([(k, *E.dm_test(p, preds[main])) for k, p in preds.items() if k != main],
                      columns=["model", "DM_stat", "DM_p"])
    dm["DM_p_holm"] = E.holm(dm.DM_p.to_numpy())
    met.merge(dm, on="model", how="left").to_csv(out / "e1_metrics.csv", index=False)
    (((long.pred - long[Y]) ** 2).groupby([long.model, long.year]).mean() ** 0.5).unstack(0).to_csv(out / "e1_by_season.csv")
    shap.to_csv(out / "e1_shap.csv")
    trials.to_csv(out / "e1_tuning.csv", index=False)
    sel.to_csv(out / "e1_selection.csv", index=False)
    (out / "e1_best.json").write_text(json.dumps({"main": main, "target": target, "alt_target": alt,
                                                  "params": best}, indent=2))


def e2(ctx):
    """Leakage audit: random k-fold CV and three leaky feature variants vs. rolling origin."""
    raw, df, feats, S, out, b = (ctx[k] for k in ("raw", "df", "feats", "S", "out", "best"))
    variants = [("Rolling origin (leakage-safe)", E.rolling_trio(df, feats, b["params"], S["test"], b["target"])),
                ("Random 5-fold CV", E.random_kfold(df, feats, b["params"], b["target"], eval_years=S["test"]))]
    for leak, label in [("same_year_prices", "Same-season prices"), ("nbr_current_yield", "Current neighbour yields"),
                        ("future_anomaly", "Anomalies vs. full-period mean")]:
        dl, gl = build_features(raw, leak=leak)
        variants.append((label, E.rolling_trio(dl, feature_list(gl), b["params"], S["test"], b["target"])))
        log(f"E2 {label} done")
    t = pd.DataFrame([{"protocol": lab, "model": k, **E.score(p)} for lab, tri in variants for k, p in tri.items()])
    honest = t[t.protocol == variants[0][0]].set_index("model").RMSE
    t["apparent_gain_pct"] = 100 * (1 - t.RMSE / t.model.map(honest))
    t.to_csv(out / "e2_leakage.csv", index=False)


def e3(ctx):
    """Drop-one-group ablation (history removed -> level target), season-bootstrap CIs."""
    df, groups, feats, S, out, b = (ctx[k] for k in ("df", "groups", "feats", "S", "out", "best"))
    full = E.rolling_trio(df, feats, b["params"], S["test"], b["target"])
    rows = []
    for g, fs in groups.items():
        drop = set(fs)
        tri = E.rolling_trio(df, [f for f in feats if f not in drop], b["params"], S["test"],
                             "level" if g == "history" else b["target"])
        for k in tri:
            bs = E.season_bootstrap({"full": full[k], "ablated": tri[k]}, "full", S["B"]).set_index("model").loc["ablated"]
            sa = E.score(tri[k])
            rows.append({"group": g, "model": k, "RMSE_full": E.score(full[k])["RMSE"], "RMSE_ablated": sa["RMSE"],
                         "R2_anomaly_ablated": sa["R2_anomaly"], "dRMSE": bs.dRMSE, "dRMSE_lo": bs.dRMSE_lo,
                         "dRMSE_hi": bs.dRMSE_hi})
        log(f"E3 drop {g} done")
    t = pd.DataFrame(rows)
    for c in ("dRMSE", "dRMSE_lo", "dRMSE_hi"):
        t[c + "_pct"] = 100 * t[c] / t.RMSE_full
    t.to_csv(out / "e3_ablation.csv", index=False)


def e4(ctx):
    """Ground-truth recovery: true DGP components vs. grouped out-of-sample SHAP values."""
    df, groups, out, truth = (ctx[k] for k in ("df", "groups", "out", "truth"))
    shap = pd.read_csv(out / "e1_shap.csv", index_col=0)
    gs = E.group_shap(shap, groups).reset_index(drop=True)
    keys = df.loc[shap.index, ["district_id", "year"]].reset_index(drop=True)
    tr = keys.merge(truth, on=["district_id", "year"], how="left")
    corr = pd.DataFrame({g: [tr[c].corr(gs[g]) for c in COMPONENTS] for g in gs.columns}, index=COMPONENTS)
    corr.to_csv(out / "e4_corr.csv")
    rec = []
    for c in COMPONENTS:
        obs = [g for g in OBSERVED_BY[c] if g in gs.columns]
        row = corr.loc[c]
        bg = row.abs().idxmax() if row.notna().any() else ""
        rec.append({"component": c, "sd": tr[c].std(), "observed_by": "+".join(obs),
                    "r_observed": tr[c].corr(gs[obs].sum(axis=1)) if obs else np.nan,
                    "best_group": bg, "r_best": row[bg] if bg else np.nan})
    pd.DataFrame(rec).to_csv(out / "e4_recovery.csv", index=False)
    X = df.loc[shap.index].reset_index(drop=True)
    pd.DataFrame({"year": keys.year, "sm_mean_est": X.sm_mean_est, "ndvi_peak_anom": X.ndvi_peak_anom,
                  "comp_revenue_index": X.comp_revenue_index, "tmax_rep": X.tmax_rep,
                  "shap_drought": gs.moisture + gs.water, "shap_pest": gs.satellite, "shap_price": gs.price,
                  "shap_heat": gs.climate, "true_drought": tr.drought, "true_pest": tr.pest,
                  "true_price": tr.price, "true_heat": tr.heat}).to_csv(out / "e4_dependence.csv", index=False)


def e5(ctx):
    """In-season forecasting: features truncated at each cut-off month, with/without satellite."""
    raw, S, out, b = (ctx[k] for k in ("raw", "S", "out", "best"))
    rows = []
    for m in S["lead"]:
        dm_, gm = build_features(raw, cutoff_month=m)
        fa = feature_list(gm)
        fn = [f for f in fa if f not in set(gm.get("satellite", []))]
        for lab, fl in (("All features", fa), ("Without satellite", fn)):
            for k, p in E.rolling_trio(dm_, fl, b["params"], S["test"], b["target"]).items():
                rows.append({"cutoff_month": m, "features": lab, "model": k, **E.score(p)})
        rows.append({"cutoff_month": m, "features": "Climatology", "model": "Climatology",
                     **E.score(E.rolling_cv(dm_, fa, "Climatology", test_years=S["test"])[0])})
        log(f"E5 cut-off month {m} done")
    pd.DataFrame(rows).to_csv(out / "e5_leadtime.csv", index=False)


def e6(ctx):
    """Conformal interval calibration overall, by season and in shock vs. normal seasons."""
    df, out, b = ctx["df"], ctx["out"], ctx["best"]
    p = pd.read_csv(out / "e1_predictions.csv")
    p = p[p.model == b["main"]].copy()
    seas = df.groupby("year").target_anom.mean()
    thr = seas.mean() - seas.std()
    p["shock"] = p.year.map(seas) < thr
    rows = []
    for yr, d in p.groupby("year"):
        r = {"year": yr, "regional_anomaly": seas[yr], "shock": bool(d.shock.iloc[0]),
             "RMSE": float(np.sqrt(((d.pred - d[Y]) ** 2).mean()))}
        for lvl in (80, 90):
            r[f"cov{lvl}"] = float(((d[Y] >= d[f"lo{lvl}"]) & (d[Y] <= d[f"hi{lvl}"])).mean())
            r[f"width{lvl}"] = float((d[f"hi{lvl}"] - d[f"lo{lvl}"]).mean())
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "e6_conformal_season.csv", index=False)
    summ = []
    for name, d in (("All seasons", p), ("Shock seasons", p[p.shock]), ("Other seasons", p[~p.shock])):
        if d.empty:
            continue
        r = {"subset": name, "n": len(d), "n_seasons": d.year.nunique()}
        for lvl in (80, 90):
            r[f"cov{lvl}"] = float(((d[Y] >= d[f"lo{lvl}"]) & (d[Y] <= d[f"hi{lvl}"])).mean())
            r[f"width{lvl}"] = float((d[f"hi{lvl}"] - d[f"lo{lvl}"]).mean())
        summ.append(r)
    pd.DataFrame(summ).to_csv(out / "e6_conformal_summary.csv", index=False)


def e7(ctx):
    """Robustness: replicate worlds (seeds) and DGP scenarios with the tuned configuration.
    Resumable: worlds already in e7_robustness.csv are skipped; E7_MAX_WORLDS limits new worlds per call."""
    S, out, b = ctx["S"], ctx["out"], ctx["best"]
    path = out / "e7_robustness.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["scenario", "seed"])
    rows, done = prev.to_dict("records"), set(zip(prev.scenario, prev.seed))
    worlds = [("base", 1000 + i) for i in range(S["seeds"])] + [(sc, C.SEED) for sc in S["scen"]]
    todo = [w for w in worlds if w not in done]
    max_new = int(os.environ.get("E7_MAX_WORLDS", len(todo)))
    for sc, seed in todo[:max_new]:
        d, g = build_features(simulate(seed=seed, scenario=sc))
        f = feature_list(g)
        fn = [x for x in f if x not in set(g["satellite"])]
        res = {k: E.score(p) for k, p in E.rolling_trio(d, f, b["params"], S["test"], b["target"], seed).items()}
        res["Climatology"] = E.score(E.rolling_cv(d, f, "Climatology", test_years=S["test"])[0])
        res["Ensemble_no_satellite"] = E.score(E.rolling_trio(d, fn, b["params"], S["test"], b["target"], seed)["Ensemble"])
        res["Ensemble_random_cv"] = E.score(E.random_kfold(d, f, b["params"], b["target"], eval_years=S["test"],
                                                           seed=seed)["Ensemble"])
        rows += [{"scenario": sc, "seed": seed, "model": k, **m} for k, m in res.items()]
        pd.DataFrame(rows).to_csv(path, index=False)
        log(f"E7 world {sc}/{seed} done")
    ctx["e7_complete"] = len(todo) <= max_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["e1", "e2", "e3", "e4", "e5", "e6", "e7"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    S = settings(a.quick)
    out = S["out"]
    out.mkdir(parents=True, exist_ok=True)
    if not (C.DATA_RAW / "yields.csv").exists():
        simulate(C.SEED, "base", out_dir=C.DATA_RAW)
        log("main world simulated -> data/raw/")
    raw, truth = load_raw(), pd.read_csv(C.DATA_RAW / "truth_components.csv")
    df, groups = build_features(raw)
    feats = feature_list(groups)
    pd.DataFrame([(g, f) for g, fs in groups.items() for f in fs], columns=["group", "feature"]).to_csv(
        out / "feature_groups.csv", index=False)
    log(f"features: {len(df)} district-seasons x {len(feats)} features in {len(groups)} groups")
    ctx = dict(raw=raw, df=df, groups=groups, feats=feats, truth=truth, S=S, out=out)
    for name in ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]:
        if name not in a.only:
            continue
        done = out / f"{name}.done"
        if done.exists() and not a.force:
            log(f"{name} cached - skipped")
            continue
        if name != "e1":
            ctx["best"] = json.loads((out / "e1_best.json").read_text())
        t1 = time.time()
        globals()[name](ctx)
        tp = out / "timings.json"
        tim = json.loads(tp.read_text()) if tp.exists() else {}
        tim[name] = tim.get(name, 0.0) + time.time() - t1
        tp.write_text(json.dumps(tim, indent=1))
        if not ctx.pop("e7_complete", True):
            log(f"{name} partially done - run again to continue")
            continue
        done.touch()
        log(f"{name} finished")


if __name__ == "__main__":
    main()
