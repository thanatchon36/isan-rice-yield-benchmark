"""Publication figures (F1-F10), LaTeX tables and number macros - all generated from results/.
Every number quoted in the paper comes from paper/tables/macros.tex written here."""
import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import NullFormatter, ScalarFormatter
from matplotlib.patches import FancyBboxPatch, Polygon as MplPolygon, Rectangle

import config as C
from figcheck import text_overlaps
from matplotlib.lines import Line2D
from simulate import COMPONENTS, OBSERVED_BY, PARAMS, REGION_OUTLINE, RIVERS, SUGAR_MILLS, TOWNS

R, FIG, TAB, Y = C.RESULTS, C.FIGS, C.TABLES, C.YIELD_COL
GC = C.GROUP_COLORS
MON = {7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov"}
MODEL_LABEL = {"Persistence": "Persistence", "Climatology": "Climatology (3-season mean)",
               "PanelFE": "Panel fixed effects", "Ridge": "Ridge regression", "RandomForest": "Random forest",
               "MLP": "Multilayer perceptron", "LightGBM_alt": "LightGBM (alternative target)",
               "LightGBM": "LightGBM", "Ensemble": "Ensemble (LightGBM + Ridge)"}
GROUP_SHORT = {"history": "History", "moisture": "Moisture", "water": "Water", "climate": "Climate",
               "satellite": "Satellite", "price": "Prices", "geography": "Geography", "map": "Map/spatial"}
COMP_LABEL = {"base": "Base level", "trend": "Technology trend", "drought": "Drought", "flood": "Flood",
              "heat": "Heat stress", "disease": "Humid-season disease", "salinity": "Salinity x drought",
              "price": "Prices & competition", "mgmt": "Management (AR(1))", "pest": "Pest outbreaks",
              "noise": "Unobservable noise"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
                     "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42, "axes.titleweight": "bold"})


CHECK = {}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    CHECK[name] = text_overlaps(fig)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=160)
    plt.close(fig)


def load():
    r = {"best": json.loads((R / "e1_best.json").read_text())}
    fg = pd.read_csv(R / "feature_groups.csv")
    r["groups"] = {g: list(fg[fg.group == g].feature) for g in dict.fromkeys(fg.group)}
    for k, f in [("met", "e1_metrics"), ("sel", "e1_selection"), ("tune", "e1_tuning"), ("e2", "e2_leakage"),
                 ("e3", "e3_ablation"), ("rec", "e4_recovery"), ("dep", "e4_dependence"), ("e5", "e5_leadtime"),
                 ("e6s", "e6_conformal_season"), ("e6", "e6_conformal_summary"), ("e7", "e7_robustness")]:
        r[k] = pd.read_csv(R / f"{f}.csv")
    r["bys"] = pd.read_csv(R / "e1_by_season.csv", index_col=0)
    r["corr"] = pd.read_csv(R / "e4_corr.csv", index_col=0)
    r["shap"] = pd.read_csv(R / "e1_shap.csv", index_col=0)
    r["meta"] = json.loads((C.DATA_RAW / "meta.json").read_text())
    return r


# ------------------------------------------------------------------------------- figures
def f1_study_area():
    from shapely.geometry import MultiPoint, Point, Polygon
    from shapely.ops import voronoi_diagram
    d = pd.read_csv(C.DATA_RAW / "districts.csv")
    d["mean_yield"] = d.district_id.map(pd.read_csv(C.DATA_RAW / "yields.csv").groupby("district_id")[Y].mean())
    region, pts = Polygon(REGION_OUTLINE), d[["lon", "lat"]].to_numpy()
    vor = voronoi_diagram(MultiPoint([tuple(p) for p in pts]), envelope=region.buffer(1.0))
    cells = [None] * len(pts)
    for poly in vor.geoms:
        for i, p in enumerate(pts):
            if cells[i] is None and poly.contains(Point(p)):
                cells[i] = poly.intersection(region)
                break
    layers = [("elevation_m", "(a) Elevation (m)", "terrain"), ("rain_normal_mm", "(b) Rainfall normal (mm/yr)", "Blues"),
              ("irrigation_share", "(c) Irrigated share and sugar mills", "PuBu"),
              ("mean_yield", f"(d) Mean yield 2008-2025 ({C.YIELD_UNIT})", "YlGn")]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.1))
    for ax, (col, title, cmap) in zip(axes.flat, layers):
        vals = d[col].to_numpy(float)
        norm = Normalize(np.nanpercentile(vals, 2), np.nanpercentile(vals, 98))
        patches, cv = [], []
        for c, v in zip(cells, vals):
            for g in ([] if c is None else getattr(c, "geoms", [c])):
                if g.geom_type == "Polygon" and not g.is_empty:
                    patches.append(MplPolygon(np.asarray(g.exterior.coords), closed=True))
                    cv.append(v)
        pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.25)
        pc.set_array(np.asarray(cv))
        ax.add_collection(pc)
        ax.plot(*REGION_OUTLINE.T, color="0.3", lw=0.6)
        for nm, r in RIVERS.items():
            ax.plot(*np.asarray(r).T, color="#1f5fa8", lw=1.0, label="Rivers (Mekong, Mun, Chi)" if nm == "mekong" else None)
        if col == "irrigation_share":
            ax.scatter(*SUGAR_MILLS.T, marker="^", s=16, c="#D55E00", edgecolor="k", lw=0.3, zorder=5, label="Sugar mill")
        if col == "mean_yield":
            ax.scatter(*np.array(list(TOWNS.values())).T, s=7, c="k", zorder=5, label="Provincial town (market)")
        if col in ("irrigation_share", "mean_yield", "elevation_m"):
            ax.legend(loc="lower left", frameon=True, framealpha=0.85, edgecolor="none", fontsize=5.8)
        ax.set_xlim(101.1, 105.8)
        ax.set_ylim(13.85, 18.6)
        ax.set_aspect(1 / np.cos(np.radians(16)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc="left")
        fig.colorbar(pc, ax=ax, shrink=0.72, pad=0.01)
    fig.tight_layout()
    save(fig, "fig1_study_area")


def f2_workflow(n_feat):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    cols = [("Inputs\n(8 groups)", ["Yield statistics", "Weather: rain, T, RH,\nsoil moisture", "Satellite: NDVI, EVI,\nLSWI, LST",
                                   "Prices: rice, cassava,\nsugarcane, maize,\nurea", "Maps: terrain, soils,\nirrigation, floods"], "#E8F1FA"),
            ("Leakage-safe\nfeatures", [f"{n_feat} features", "growth-stage\naggregates", "past-only anomalies",
                                       "lag-1 local prices", "k-NN spatial lags"], "#EAF6F0"),
            ("Learners", ["persistence,\nclimatology", "panel fixed effects", "Ridge, RF, MLP", "LightGBM",
                          "Ensemble\n(LGBM + Ridge)"], "#FDF3E1"),
            ("Evaluation", ["rolling origin", "tune 2015-2017", "test 2018-2025", "bootstrap CIs,\nDM tests",
                            "leakage audit"], "#F7EAF3"),
            ("Evidence", ["accuracy (E1)", "leakage (E2)", "attribution & truth\nrecovery (E3-E4)",
                          "in-season skill (E5)", "intervals, robustness\n(E6-E7)"], "#EEEEEE")]
    for i, (title, items, fc) in enumerate(cols):
        x = 0.1 + i * 2.0
        ax.add_patch(FancyBboxPatch((x, 0.35), 1.8, 4.3, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec="0.35", lw=0.7))
        ax.text(x + 0.9, 4.22, title, ha="center", va="center", fontweight="bold", fontsize=6.8, linespacing=1.0)
        for j, it in enumerate(items):
            ax.text(x + 0.9, 3.4 - j * 0.68, it, ha="center", va="center", fontsize=5.8, linespacing=1.0)
        if i < len(cols) - 1:
            ax.annotate("", xy=(x + 2.0, 2.45), xytext=(x + 1.84, 2.45),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="0.25"))
    ax.text(5, 0.1, "Benchmark: known data-generating process with saved true yield components (truth for E4)",
            ha="center", va="center", fontsize=6.4, style="italic", color="0.25")
    save(fig, "fig2_workflow")


def f3_models(met, main):
    t = met.set_index("model").sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    yy = np.arange(len(t))
    ax.barh(yy, t.RMSE, xerr=[t.RMSE - t.RMSE_lo, t.RMSE_hi - t.RMSE],
            color=["#0072B2" if m == main else "#A8A8A8" for m in t.index], error_kw=dict(lw=0.8, capsize=2))
    ax.set_yticks(yy, [MODEL_LABEL[m] for m in t.index])
    ax.invert_yaxis()
    for i, (hi, r, mape, sk) in enumerate(zip(t.RMSE_hi, t.RMSE, t.MAPE, t.Skill)):
        ax.text(hi + 1.0, i, f"{r:.1f}  |  MAPE {mape:.1f}%  |  skill {sk:+.2f}", va="center", fontsize=6.3)
    ax.set_xlim(0, t.RMSE_hi.max() * 1.62)
    ax.set_xlabel(f"RMSE ({C.YIELD_UNIT}), test seasons {C.TEST_YEARS[0]}-{C.TEST_YEARS[-1]}, 95% season-bootstrap CI")
    ax.grid(axis="y", visible=False)
    save(fig, "fig3_model_comparison")


def f4_seasons(bys, e6s, main):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 3.9), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
    styles = {"Persistence": ("#999999", ":"), "Climatology": ("#000000", "--"), "PanelFE": ("#E69F00", "-"),
              "Ridge": ("#56B4E9", "-"), "LightGBM": ("#009E73", "-"), "Ensemble": ("#0072B2", "-")}
    for m, (c, ls) in styles.items():
        a1.plot(bys.index, bys[m], color=c, ls=ls, lw=2.3 if m == main else 1.1, marker="o", ms=2.5,
                label=MODEL_LABEL[m], zorder=3 if m == main else 2)
    for yr in e6s[e6s.shock].year:
        for ax in (a1, a2):
            ax.axvspan(yr - 0.45, yr + 0.45, color="#D55E00", alpha=0.10, lw=0)
    a1.set_ylabel(f"RMSE ({C.YIELD_UNIT})")
    a1.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6.2, borderaxespad=0.3)
    a1.set_title("(a) Out-of-sample error by test season (shaded: shock seasons)", loc="left", pad=27)
    a2.bar(e6s.year, e6s.regional_anomaly, color=["#D55E00" if s else "#A8A8A8" for s in e6s.shock], width=0.6)
    a2.axhline(0, color="k", lw=0.6)
    a2.set_ylabel(f"Anomaly\n({C.YIELD_UNIT})")
    a2.set_title("(b) Regional mean yield anomaly vs. previous-3-season mean", loc="left")
    a2.set_xticks(e6s.year)
    save(fig, "fig4_error_by_season")


def f5_leakage(e2, main):
    protos = list(dict.fromkeys(e2.protocol))
    models = ["LightGBM", "Ridge", "Ensemble"]
    cols = {"LightGBM": "#009E73", "Ridge": "#56B4E9", "Ensemble": "#0072B2"}
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x, w = np.arange(len(protos)), 0.26
    honest = e2[(e2.model == main) & (e2.protocol == protos[0])].RMSE.iloc[0]
    for i, m in enumerate(models):
        s = e2[e2.model == m].set_index("protocol").reindex(protos)
        ax.bar(x + (i - 1) * w, s.RMSE, w, color=cols[m], label=MODEL_LABEL[m])
        if m == main:
            for xi, (r, gpc) in enumerate(zip(s.RMSE, s.apparent_gain_pct)):
                if xi:
                    ax.text(xi + (i - 1) * w, max(r, honest) + 0.9, f"{-gpc:+.0f}%", ha="center", fontsize=6.3,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    honest = e2[(e2.model == main) & (e2.protocol == protos[0])].RMSE.iloc[0]
    ax.axhline(honest, color="k", lw=0.7, ls="--")
    ax.set_xticks(x, [textwrap.fill(p, 18) for p in protos])
    ax.set_ylabel(f"Reported RMSE ({C.YIELD_UNIT})")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_ylim(0, e2.RMSE.max() * 1.25)
    ax.grid(axis="x", visible=False)
    save(fig, "fig5_leakage")


def f6_attribution(e3, shap, groups, main):
    order = list(e3[e3.model == "LightGBM"].set_index("group").sort_values("dRMSE_pct", ascending=False).index)
    f2g = {f: g for g, fs in groups.items() for f in fs}
    imp = shap.abs().mean()
    share = (100 * imp.groupby(imp.index.map(f2g)).sum() / imp.sum()).reindex(order)
    models = list(dict.fromkeys(["LightGBM", main]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
    yy, h = np.arange(len(order)), 0.8 / len(models)
    for k, m in enumerate(models):
        t = e3[e3.model == m].set_index("group").reindex(order)
        a1.barh(yy + (k - (len(models) - 1) / 2) * h, t.dRMSE_pct, h,
                xerr=[t.dRMSE_pct - t.dRMSE_lo_pct, t.dRMSE_hi_pct - t.dRMSE_pct],
                color="#009E73" if m == "LightGBM" else "#0072B2", error_kw=dict(lw=0.6, capsize=1.2), label=MODEL_LABEL[m])
    a1.set_xscale("symlog", linthresh=10)
    a1.set_yticks(yy, [GROUP_SHORT[g] for g in order])
    a1.invert_yaxis()
    a1.axvline(0, color="k", lw=0.6)
    a1.set_xlabel("RMSE change when group removed (%, symlog axis)")
    a1.set_title("(a) Drop-one-group ablation", loc="left")
    a1.legend(frameon=False, fontsize=5.8, loc="lower right")
    a2.barh(yy, share.values, color=[GC[g] for g in order])
    for i, v in enumerate(share.values):
        a2.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=6.3)
    a2.set_yticks(yy, [])
    a2.invert_yaxis()
    a2.set_xlim(0, share.max() * 1.3)
    a2.set_xlabel("Share of mean |SHAP| (%)")
    a2.set_title(f"(b) Grouped SHAP, {'ensemble' if main == 'Ensemble' else MODEL_LABEL[main]}", loc="left")
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig6_attribution")
    full = imp.groupby(imp.index.map(f2g)).sum()
    return 100 * full / full.sum()


def t_ablation(e3):
    order = list(e3[e3.model == "LightGBM"].set_index("group").sort_values("dRMSE_pct", ascending=False).index)
    body = []
    for g in order:
        cells = []
        for m in ("LightGBM", "Ridge", "Ensemble"):
            r = e3[(e3.group == g) & (e3.model == m)].iloc[0]
            cells.append(f"{r.dRMSE_pct:+.1f} [{r.dRMSE_lo_pct:+.1f}, {r.dRMSE_hi_pct:+.1f}]")
        body.append(f"{tex_escape(C.GROUP_LABELS[g])} & " + " & ".join(cells) + r" \\")
    full = " & ".join(f"{e3[e3.model == m].RMSE_full.iloc[0]:.1f}" for m in ("LightGBM", "Ridge", "Ensemble"))
    write("tab_ablation.tex", "\n".join([
        r"\begin{table}[H]\centering\scriptsize",
        r"\caption{Drop-one-group ablation: change in test RMSE (\%) when a feature group is removed and the learner re-trained in every fold; brackets: 95\% season-block bootstrap interval. Positive values mean the group is useful. The first row gives the RMSE (kg\,rai$^{-1}$) with all features.}\label{tab:ablation}",
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Removed group & LightGBM & Ridge & Ensemble \\", r"\midrule",
        f"(none: RMSE with all features) & {full} \\\\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"]))


def t_byseason(bys, met):
    order = met.sort_values("RMSE").model.tolist()
    yrs = list(bys.index)
    best = bys[order].min(axis=1)
    overall = met.set_index("model").RMSE
    body = []
    for m in order:
        cells = [(f"\\textbf{{{bys.loc[y, m]:.1f}}}" if np.isclose(bys.loc[y, m], best[y]) else f"{bys.loc[y, m]:.1f}")
                 for y in yrs]
        body.append(f"{MODEL_LABEL[m]} & " + " & ".join(cells) + f" & {overall[m]:.1f} \\\\")
    write("tab_byseason.tex", "\n".join([
        r"\begin{table}[H]\centering\footnotesize",
        r"\caption{Test RMSE (kg\,rai$^{-1}$) of every learner in every test season (bold: lowest error of the season) and over all 1{,}600 forecasts.}\label{tab:byseason}",
        r"\begin{tabular}{@{}l" + "c" * (len(yrs) + 1) + r"@{}}", r"\toprule",
        "Model & " + " & ".join(str(y) for y in yrs) + r" & All \\", r"\midrule", *body,
        r"\bottomrule", r"\end{tabular}", r"\end{table}"]))


def t_worlds(e7):
    piv = e7.pivot_table(index=["scenario", "seed"], columns="model", values="RMSE")
    cols = ["Climatology", "Ridge", "LightGBM", "Ensemble", "Ensemble_no_satellite", "Ensemble_random_cv"]
    names = {"nonlinear": "Stronger non-linearity", "map_error": "3$\\times$ map-layer error",
             "clouds": "1.7$\\times$ cloud contamination", "no_pest": "No satellite-only pests"}

    def row(lab, r):
        cells = []
        for c in cols:
            v = f"{r[c]:.1f}"
            if c in ("Ridge", "Ensemble") and r[c] > 1.5 * r["LightGBM"]:
                v += r"$^{\dagger}$"
            cells.append(v)
        return f"{lab} & " + " & ".join(cells) + r" \\"

    reps = [row(f"Replicate world {sd}", piv.loc[("base", sd)]) for sd in sorted(piv.loc["base"].index)]
    scen = [row(names[sc], piv.loc[(sc, C.SEED)]) for sc in C.SCENARIOS if (sc, C.SEED) in piv.index]
    write("tab_worlds.tex", "\n".join([
        r"\begin{table}[H]\centering\footnotesize",
        r"\caption{Test RMSE (kg\,rai$^{-1}$) in every replicate world (seeds 1000--1009) and scenario world (seed 42), using the configuration tuned once in the main world. $^{\dagger}$: RMSE more than 1.5 times that of LightGBM in the same world. No sat.: ensemble without satellite features; Random CV: the same ensemble scored by random 5-fold cross-validation.}\label{tab:worlds}",
        r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
        r"World & Climatology & Ridge & LightGBM & Ensemble & No sat. & Random CV \\", r"\midrule",
        *reps, r"\midrule", *scen, r"\bottomrule", r"\end{tabular}", r"\end{table}"]))


def f7_dependence(dep):
    panels = [("sm_mean_est", "shap_drought", "true_drought", "(a) Drought", "Soil moisture May-Jul (m$^3$ m$^{-3}$)", "Moisture+water"),
              ("ndvi_peak_anom", "shap_pest", "true_pest", "(b) Pest outbreaks", "Peak-NDVI anomaly vs. past seasons", "Satellite"),
              ("comp_revenue_index", "shap_price", "true_price", "(c) Prices and competition", "Competing-revenue index (log ratio)", "Prices"),
              ("tmax_rep", "shap_heat", "true_heat", "(d) Heat around flowering", "Max. temperature Oct-Nov (°C)", "Climate")]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.1))
    rs = {}
    for ax, (x, s, t, title, xl, gl) in zip(axes.flat, panels):
        ok = dep[[x, s, t]].notna().all(axis=1)
        xs, sv, tv = dep.loc[ok, x].to_numpy(), dep.loc[ok, s].to_numpy(), dep.loc[ok, t].to_numpy()
        sv, tv = sv - sv.mean(), tv - tv.mean()
        ax.scatter(xs, sv, s=2.5, alpha=0.22, color="#0072B2", rasterized=True)
        edges = np.unique(np.quantile(xs, np.linspace(0, 1, 13)))
        idx = np.clip(np.digitize(xs, edges[1:-1]), 0, len(edges) - 2)
        mid, ms, mt = [], [], []
        for b in range(len(edges) - 1):
            sel = idx == b
            if sel.sum() >= 10:
                mid.append(np.median(xs[sel]))
                ms.append(sv[sel].mean())
                mt.append(tv[sel].mean())
        ax.plot(mid, ms, color="#0072B2", lw=1.6)
        ax.plot(mid, mt, color="#D55E00", lw=1.4, ls="--", marker="o", ms=2.3)
        rs[t] = np.corrcoef(sv, tv)[0, 1]
        ax.text(0.97, 0.05, f"r = {rs[t]:.2f}", transform=ax.transAxes, ha="right", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
        ax.set_xlabel(xl)
        ax.set_ylabel(f"Contribution ({C.YIELD_UNIT})")
        ax.set_title(f"{title}: {gl} SHAP", loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles=[Line2D([], [], color="#0072B2", marker="o", ls="", ms=3, alpha=0.5, label="Group SHAP (one point per forecast)"),
                        Line2D([], [], color="#0072B2", lw=1.6, label="Group SHAP, binned mean"),
                        Line2D([], [], color="#D55E00", lw=1.4, ls="--", marker="o", ms=2.3, label="True component, binned mean")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    save(fig, "fig7_dependence_vs_truth")
    return rs


def f8_recovery(corr):
    cols = list(corr.columns)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i, c in enumerate(corr.index):
        for j, g in enumerate(cols):
            v = corr.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.6 else "k")
            if g in OBSERVED_BY.get(c, []):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=1.1))
    ax.set_xticks(range(len(cols)), [GROUP_SHORT[g] for g in cols], rotation=90, ha="center")
    ax.set_yticks(range(len(corr)), [COMP_LABEL[c] for c in corr.index])
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson r: true component vs. group SHAP")
    ax.set_title("Ground-truth recovery (boxes: groups that observe the mechanism)", loc="left")
    save(fig, "fig8_truth_recovery")


def f9_leadtime(e5, main):
    months = sorted(e5.cutoff_month.unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})
    get = lambda m, f: e5[(e5.model == m) & (e5.features == f)].set_index("cutoff_month").RMSE.reindex(months)
    for m, c, lw in (("LightGBM", "#009E73", 1.9), ("Ridge", "#56B4E9", 1.0), ("Ensemble", "#0072B2", 1.3)):
        a1.plot(months, get(m, "All features"), color=c, lw=lw, marker="o", ms=2.6, label=MODEL_LABEL[m])
    a1.plot(months, get("LightGBM", "Without satellite"), color="#009E73", lw=1.2, ls="--", marker="o", ms=2.6,
            label="LightGBM without satellite")
    a1.plot(months, get("Climatology", "Climatology"), color="k", lw=0.9, ls=":", label="Climatology")
    a1.set_yscale("log")
    a1.set_yticks([20, 30, 50, 100, 150])
    a1.yaxis.set_major_formatter(ScalarFormatter())
    a1.yaxis.set_minor_formatter(NullFormatter())
    a1.set_xticks(months, [MON[m] for m in months])
    a1.set_xlabel("Data cut-off (end of month)")
    a1.set_ylabel(f"RMSE ({C.YIELD_UNIT}, log scale)")
    a1.set_title("(a) Skill vs. forecast date", loc="left")
    w = 0.36
    for i, (m, c) in enumerate((("LightGBM", "#009E73"), ("Ensemble", "#0072B2"))):
        a2.bar(np.arange(len(months)) + (i - 0.5) * w, get(m, "Without satellite") - get(m, "All features"), w,
               color=c, label=MODEL_LABEL[m])
    a2.axhline(0, color="k", lw=0.6)
    a2.set_xticks(range(len(months)), [MON[m] for m in months])
    a2.set_ylabel(f"RMSE reduction ({C.YIELD_UNIT})")
    a2.set_title("(b) Value of satellite features", loc="left")
    a2.legend(frameon=False, fontsize=5.8, loc="upper left")
    a2.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.2, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig9_leadtime")


def f10_conformal(e6s):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for yr in e6s[e6s.shock].year:
        for ax in (a1, a2):
            ax.axvspan(yr - 0.45, yr + 0.45, color="#D55E00", alpha=0.10, lw=0)
    a1.plot(e6s.year, 100 * e6s.cov80, "o-", color="#0072B2", ms=3, label="80% interval")
    a1.plot(e6s.year, 100 * e6s.cov90, "s-", color="#E69F00", ms=3, label="90% interval")
    a1.axhline(80, color="#0072B2", ls="--", lw=0.7)
    a1.axhline(90, color="#E69F00", ls="--", lw=0.7)
    a1.set_ylim(max(0, 100 * min(e6s.cov80.min(), e6s.cov90.min()) - 10), 101)
    a1.set_ylabel("Empirical coverage (%)")
    a1.set_title("(a) Coverage by test season", loc="left")
    a1.legend(frameon=False, loc="lower center", ncol=2)
    a2.bar(e6s.year - 0.18, e6s.width80, 0.36, color="#0072B2", label="80%")
    a2.bar(e6s.year + 0.18, e6s.width90, 0.36, color="#E69F00", label="90%")
    a2.set_ylabel(f"Mean width ({C.YIELD_UNIT})")
    a2.set_title("(b) Interval width", loc="left")
    a2.set_ylim(0, max(e6s.width80.max(), e6s.width90.max()) * 1.3)
    a2.legend(frameon=False, loc="upper center", ncol=2)
    for a in (a1, a2):
        a.set_xticks(e6s.year, [str(y)[2:] for y in e6s.year])
        a.set_xlabel("Test season ('yy)")
    fig.tight_layout()
    save(fig, "fig10_conformal")


# ------------------------------------------------------------------------------- tables & macros
def tex_escape(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write(name, txt):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text(txt, encoding="utf-8")


def fmt_p(p):
    return "--" if not np.isfinite(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def t_dgp():
    rows = [
        ("Base level", r"$b_i=\beta_0+\beta_q q_i+\beta_I I_i+\sigma_u u_i+\sigma_\eta\eta_i$",
         r"$\beta_0=@base_mean@$, $\beta_q=@b_soil@$, $\beta_I=@b_irrig@$, $\sigma_u=@b_unobs_field@$, $\sigma_\eta=@b_unobs_iid@$", "Geography, map, history"),
        ("Technology trend", r"$\tau\,(t-2008)$", r"$\tau=@trend@$", "History"),
        ("Establishment drought", r"$-\gamma_E\max\{0,\theta_E-(\bar s^{JJ}_{it}+\kappa_E S_{it})\}$",
         r"$\gamma_E=@est_coef@$, $\theta_E=@est_thr@$, $\kappa_E=@est_irr@$", "Moisture, water, satellite"),
        ("Late-season drought", r"$-\gamma_L\max\{0,\theta_L-(\bar s^{SO}_{it}+\kappa_L S_{it})\}$",
         r"$\gamma_L=@late_coef@$, $\theta_L=@late_thr@$, $\kappa_L=@late_irr@$", "Moisture, water, satellite"),
        ("Flood", r"$-\min\{c_F,\ \gamma_F (f_i/0.3)\max(0,w_{it}-\theta_F)\}$",
         r"$\gamma_F=@flood_coef@$, $\theta_F=@flood_thr@$, $c_F=@flood_cap@$", "Water, map, satellite"),
        ("Heat at flowering", r"$-\gamma_H\sum_{m\in\{9,10\}}\max(0,T^{\max}_{itm}-\theta_H)$",
         r"$\gamma_H=@heat_coef@$, $\theta_H=@heat_thr@$\,$^\circ$C", "Climate"),
        ("Humid-season disease", r"$-\gamma_D\max(0,\overline{RH}^{AS}_{it}-\theta_D)$",
         r"$\gamma_D=@dis_coef@$, $\theta_D=@dis_thr@$\,\%", "Moisture"),
        ("Salinity $\\times$ drought", r"$-\gamma_S\,\sigma_i\max(0,\theta_S-\bar s^{JJ}_{it})$",
         r"$\gamma_S=@sal_coef@$, $\theta_S=@sal_thr@$", "Geography, moisture"),
        ("Prices \\& competition", r"$-\gamma_C\,\tilde c_{it}(1-I_i)+\gamma_P\,\tilde p_{it}(0.6+I_i)$",
         r"$\gamma_C=@price_comp@$, $\gamma_P=@price_input@$", "Prices, map"),
        ("Harvest-price feedback", r"$\ln P^{\mathrm{rice}}_t \leftarrow \ln P^{\mathrm{rice}}_t-\kappa\,\bar a_t/100$",
         r"$\kappa=@price_feedback@$", "(price endogeneity)"),
        ("Management", r"$m_{it}=\rho\,m_{i,t-1}+\sigma_m\xi_{it}$", r"$\rho=@mgmt_rho@$, $\sigma_m=@mgmt_sd@$", "History, satellite"),
        ("Pest outbreaks", r"$-A\exp\{-d_{ic}^2/(2r^2)\}$ in 3 seasons", r"$A=@pest_amp@$, $r=@pest_radius@$\,km", "Satellite only"),
        ("Noise", r"$\varepsilon_{it}\sim\mathcal N(0,\sigma_\varepsilon^2)$", r"$\sigma_\varepsilon=@noise_sd@$", "None (irreducible)"),
    ]
    body = []
    for a, b, c, d in rows:
        for k, v in PARAMS.items():
            c = c.replace(f"@{k}@", f"{v:g}")
        body.append(f"{a} & {b} & {c} & {d} \\\\")
    write("tab_dgp.tex", "\n".join([
        r"\begin{table}[t]\centering\scriptsize",
        r"\caption{Additive yield mechanisms of the benchmark DGP (contributions in kg\,rai$^{-1}$), parameter values used in the base world, and the feature groups that observe each mechanism. $q_i$: soil-quality index; $I_i$: irrigated share; $\bar s$: true soil moisture (JJ: Jun--Jul, SO: Sep--Oct); $S_{it}$: irrigation supply (irrigated share $\times$ reservoir storage); $f_i$: flood frequency; $w_{it}$: Aug--Sep rainfall relative to normal; $\sigma_i$: saline share; $\tilde c_{it}$, $\tilde p_{it}$: standardised competing-revenue and rice/urea indices from lag-1 local prices; $\bar a_t$: regional mean yield anomaly. Scenario variants multiply all threshold terms by 1.8 and add a compound heat--drought penalty (non-linear), triple map-layer measurement error, raise cloud frequency 1.7-fold, or remove pests.}\label{tab:dgp}",
        r"\begin{tabularx}{\textwidth}{@{}p{2.3cm}Xp{3.9cm}p{2.3cm}@{}}", r"\toprule",
        r"Mechanism & Contribution to $y_{it}$ & Parameters & Observed by \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabularx}", r"\end{table}"]))


def t_features(groups):
    info = {"history": ("lags $t-1$, $t-2$; previous-3-season mean and s.d.; 5-season trend", "OAE / DOAE statistics"),
            "moisture": ("RH, VPD, soil-moisture mean and minimum per growth stage", "ERA5-Land; SMAP"),
            "water": ("rain per stage and season; anomaly vs. past; dry months; reservoir storage; irrigation supply", "CHIRPS; TMD; RID reservoirs"),
            "climate": ("$T_{\\mathrm{mean}}$, $T_{\\max}$ per stage; GDD; hot months; rainfall normal", "ERA5-Land; TMD normals"),
            "satellite": ("NDVI/EVI maxima, LSWI/LST means per stage; NDVI peak, area, timing, green-up; clear-sky fraction; anomalies", "MODIS MOD13Q1, MOD09A1, MOD11A2; Sentinel-1/2"),
            "price": ("lag-1 local prices of rice, cassava, sugarcane, maize; ratios; competing-revenue index; rice/urea ratio", "OAE farm-gate prices; OCSB cane price"),
            "geography": ("elevation, slope, sand, SOC, pH, saline share, suitability class, coordinates", "SRTM; LDD soil maps; SoilGrids"),
            "map": ("distances to river, mill and market; irrigated and paddy share; flood frequency; $k$-NN spatial lags", "RID; LDD land use; GISTDA floods; OSM")}
    body = [f"{tex_escape(C.GROUP_LABELS[g])} & {len(fs)} & {info[g][0]} & {info[g][1]} \\\\" for g, fs in groups.items()]
    write("tab_features.tex", "\n".join([
        r"\begin{table}[t]\centering\scriptsize",
        r"\caption{Feature groups, number of features, content, and the real-data sources that can replace the synthetic inputs without code changes (OAE: Office of Agricultural Economics; DOAE: Department of Agricultural Extension; TMD: Thai Meteorological Department; RID: Royal Irrigation Department; OCSB: Office of the Cane and Sugar Board; LDD: Land Development Department; GISTDA: Geo-Informatics and Space Technology Development Agency).}\label{tab:features}",
        r"\begin{tabularx}{\textwidth}{@{}p{3.3cm}cXp{3.4cm}@{}}", r"\toprule",
        r"Group & $n$ & Features & Real-data sources \\", r"\midrule", *body,
        rf"\midrule Total & {sum(len(v) for v in groups.values())} & & \\",
        r"\bottomrule", r"\end{tabularx}", r"\end{table}"]))


def t_models(met, main):
    t = met.set_index("model").sort_values("RMSE")
    body = []
    for m, r in t.iterrows():
        name = MODEL_LABEL[m] + (r" $^{\star}$" if m == main else "")
        body.append(f"{name} & {r.RMSE:.1f} [{r.RMSE_lo:.1f}, {r.RMSE_hi:.1f}] & {r.MAE:.1f} & {r.MAPE:.1f} & "
                    f"{r.R2:.2f} & {r.R2_anomaly:.2f} & {r.Skill:+.2f} & {fmt_p(r.DM_p_holm)} \\\\")
    write("tab_models.tex", "\n".join([
        r"\begin{table}[t]\centering\scriptsize",
        r"\caption{Out-of-sample accuracy on the test seasons 2018--2025 (rolling origin; 1{,}600 district-season forecasts). RMSE and MAE in kg\,rai$^{-1}$ (multiply by 0.00625 for t\,ha$^{-1}$); brackets: 95\% season-block bootstrap CI. $R^2_{\mathrm{anom}}$: skill on deviations from the previous-3-season mean; Skill: $1-\mathrm{MSE}/\mathrm{MSE}_{\mathrm{clim}}$. $p_{\mathrm{DM}}$: Holm-adjusted Diebold--Mariano (HLN) test against the main model ($^{\star}$).}\label{tab:models}",
        r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
        r"Model & RMSE [95\% CI] & MAE & MAPE (\%) & $R^2$ & $R^2_{\mathrm{anom}}$ & Skill & $p_{\mathrm{DM}}$ \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"]))


def t_leakage(e2):
    protos = list(dict.fromkeys(e2.protocol))
    body = []
    for p in protos:
        cells = []
        for m in ("LightGBM", "Ridge", "Ensemble"):
            r = e2[(e2.protocol == p) & (e2.model == m)].iloc[0]
            cells.append(f"{r.RMSE:.1f}" + ("" if p == protos[0] else f" ({-r.apparent_gain_pct:+.0f}\\%)"))
        body.append(f"{tex_escape(p)} & " + " & ".join(cells) + r" \\")
    write("tab_leakage.tex", "\n".join([
        r"\begin{table}[t]\centering\footnotesize",
        r"\caption{Leakage audit: RMSE (kg\,rai$^{-1}$) reported on the same 1{,}600 test forecasts under the leakage-safe protocol and under four flawed protocols; parentheses: change relative to the leakage-safe RMSE (negative = optimistic bias).}\label{tab:leakage}",
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Protocol & LightGBM & Ridge & Ensemble \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"]))


def t_recovery(rec):
    lg = pd.read_csv(R / "e4_recovery_lgb.csv").set_index("component").r_observed_lgb if (R / "e4_recovery_lgb.csv").exists() else None
    body = []
    for _, r in rec.iterrows():
        obs = ", ".join(GROUP_SHORT[g] for g in str(r.observed_by).split("+") if g in GROUP_SHORT) or "--"
        ro = "--" if not np.isfinite(r.r_observed) else f"{r.r_observed:.2f}"
        rb = "--" if not isinstance(r.best_group, str) else f"{GROUP_SHORT[r.best_group]} ({r.r_best:.2f})"
        rl = "--" if lg is None or not np.isfinite(lg[r.component]) else f"{lg[r.component]:.2f}"
        lab = tex_escape(COMP_LABEL[r.component]).replace(" x ", " $\\times$ ")
        body.append(f"{lab} & {r.sd:.1f} & {obs} & {ro} & {rl} & {rb} \\\\")
    write("tab_recovery.tex", "\n".join([
        r"\begin{table}[t]\centering\footnotesize",
        r"\caption{Ground-truth recovery on the test forecasts. s.d.: standard deviation of the true component (kg\,rai$^{-1}$); $r_{\mathrm{obs}}$: Pearson correlation between the true component and the summed SHAP values of the groups that observe it, for the main model and (post hoc) for LightGBM alone; last column (main model): the single group whose SHAP correlates most strongly (absolute value) with the component.}\label{tab:recovery}",
        r"\begin{tabular}{@{}lcp{3.6cm}ccc@{}}", r"\toprule",
        r"True component & s.d. & Observed by & $r_{\mathrm{obs}}$ main & $r_{\mathrm{obs}}$ LightGBM & Best single group ($r$) \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"]))


def robust_tables(e7, met, e2, e3, main):
    cols = ["Climatology", "Ridge", "LightGBM", "Ensemble", "Ensemble_no_satellite", "Ensemble_random_cv"]
    piv = e7.pivot_table(index=["scenario", "seed"], columns="model", values="RMSE").reset_index()
    rep = piv[piv.scenario == "base"]
    base42 = {"Climatology": met.set_index("model").RMSE["Climatology"], "Ridge": met.set_index("model").RMSE["Ridge"],
              "LightGBM": met.set_index("model").RMSE["LightGBM"], "Ensemble": met.set_index("model").RMSE["Ensemble"],
              "Ensemble_no_satellite": e3[(e3.model == "Ensemble") & (e3.group == "satellite")].RMSE_ablated.iloc[0],
              "Ensemble_random_cv": e2[(e2.model == "Ensemble") & (e2.protocol == "Random 5-fold CV")].RMSE.iloc[0]}
    lines = [f"Main world (seed {C.SEED}) & " + " & ".join(f"{base42[c]:.1f}" for c in cols) + r" \\",
             f"Replicates, mean $\\pm$ s.d. ($n={len(rep)}$) & " + " & ".join(
                 f"{rep[c].mean():.1f} $\\pm$ {rep[c].std():.1f}" for c in cols) + r" \\", r"\midrule"]
    names = {"nonlinear": "Stronger non-linearity", "map_error": "3$\\times$ map-layer error",
             "clouds": "1.7$\\times$ cloud contamination", "no_pest": "No satellite-only pests"}
    for sc in C.SCENARIOS:
        r = piv[piv.scenario == sc]
        if not r.empty:
            lines.append(f"{names[sc]} & " + " & ".join(f"{r[c].iloc[0]:.1f}" for c in cols) + r" \\")
    write("tab_robust.tex", "\n".join([
        r"\begin{table}[t]\centering\scriptsize",
        r"\caption{Robustness: test RMSE (kg\,rai$^{-1}$) in the main world, across ten independently simulated replicate worlds, and in four DGP scenarios (seed 42), with the configuration tuned once in the main world. ``No sat.'': ensemble without satellite features; ``Random CV'': the same ensemble scored with naive random 5-fold cross-validation.}\label{tab:robust}",
        r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
        r"World & Climatology & Ridge & LightGBM & Ensemble & No sat. & Random CV \\", r"\midrule",
        *lines, r"\bottomrule", r"\end{tabular}", r"\end{table}"]))
    return piv


def t_hparams(best, sel, tune):
    from models import LGB_SPACE
    body = [f"\\texttt{{{k.replace('_', chr(92) + '_')}}} & {', '.join(str(v) for v in vals)} & {best['params'][k]:g} \\\\"
            for k, vals in LGB_SPACE.items()]
    selrows = " ; ".join(f"{MODEL_LABEL[m]} {r:.1f}" for m, r in zip(sel.model, sel.RMSE))
    write("tab_hparams.tex", "\n".join([
        r"\begin{table}[H]\centering\footnotesize",
        rf"\caption{{LightGBM random search ({tune.trial.nunique()} configurations $\times$ 2 target formulations, scored by rolling origin on 2015--2017) and the selected configuration (target: {best['target']}). Validation RMSE of the candidate main models (kg\,rai$^{{-1}}$): {selrows}.}}\label{{tab:hparams}}",
        r"\begin{tabular}{@{}llc@{}}", r"\toprule", r"Hyper-parameter & Search grid & Selected \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"]))


def macros(r, share, rs, piv):
    met, main, b = r["met"].set_index("model"), r["best"]["main"], r["best"]
    M = {}
    put = lambda k, v: M.__setitem__(k, v)
    put("nDistricts", C.N_DISTRICTS)
    put("nSeasons", len(C.YEARS))
    put("nFeatures", sum(len(v) for v in r["groups"].values()))
    put("nTrials", r["tune"].trial.nunique())
    put("mainModel", MODEL_LABEL[main])
    put("mainTarget", b["target"])
    put("altTarget", b["alt_target"])
    for key, m in [("Main", main), ("Clim", "Climatology"), ("Pers", "Persistence"), ("FE", "PanelFE"),
                   ("Ridge", "Ridge"), ("RF", "RandomForest"), ("MLP", "MLP"), ("LGB", "LightGBM"),
                   ("LGBalt", "LightGBM_alt"), ("Ens", "Ensemble")]:
        x = met.loc[m]
        put(f"rmse{key}", f"{x.RMSE:.1f}")
        put(f"rmseLo{key}", f"{x.RMSE_lo:.1f}")
        put(f"rmseHi{key}", f"{x.RMSE_hi:.1f}")
        put(f"tha{key}", f"{x.RMSE * C.KG_RAI_TO_T_HA:.3f}")
        put(f"mae{key}", f"{x.MAE:.1f}")
        put(f"mape{key}", f"{x.MAPE:.1f}")
        put(f"rtwo{key}", f"{x.R2:.2f}")
        put(f"rtwoanom{key}", f"{x.R2_anomaly:.2f}")
        put(f"skill{key}", f"{x.Skill:.2f}")
        put(f"dmp{key}", fmt_p(x.DM_p_holm) if m != main else "--")
        put(f"drmse{key}", f"{x.dRMSE:.1f}")
        put(f"drmseLo{key}", f"{x.dRMSE_lo:.1f}")
        put(f"drmseHi{key}", f"{x.dRMSE_hi:.1f}")
    for key, m in [("Clim", "Climatology"), ("Pers", "Persistence"), ("FE", "PanelFE"), ("LGB", "LightGBM"),
                   ("Ridge", "Ridge"), ("RF", "RandomForest"), ("MLP", "MLP")]:
        put(f"red{key}Pct", f"{100 * (1 - met.RMSE[main] / met.RMSE[m]):.0f}")
    sel = r["sel"].set_index("model").RMSE
    for key, m in [("LGB", "LightGBM"), ("Ridge", "Ridge"), ("Ens", "Ensemble")]:
        put(f"valRmse{key}", f"{sel[m]:.1f}")
    e2 = r["e2"]
    g = lambda p, m: e2[(e2.protocol == p) & (e2.model == m)].iloc[0]
    for key, p in [("Random", "Random 5-fold CV"), ("Price", "Same-season prices"), ("Nbr", "Current neighbour yields"),
                   ("Future", "Anomalies vs. full-period mean")]:
        for mk, m in [("", main), ("LGB", "LightGBM"), ("Ridge", "Ridge")]:
            put(f"gain{key}{mk}", f"{g(p, m).apparent_gain_pct:.0f}")
            put(f"rmseLeak{key}{mk}", f"{g(p, m).RMSE:.1f}")
    e3 = r["e3"][r["e3"].model == main].set_index("group")
    for gname in e3.index:
        k = gname.capitalize()
        put(f"abl{k}", f"{e3.dRMSE_pct[gname]:+.1f}")
        put(f"ablLo{k}", f"{e3.dRMSE_lo_pct[gname]:+.1f}")
        put(f"ablHi{k}", f"{e3.dRMSE_hi_pct[gname]:+.1f}")
        put(f"shap{k}", f"{share[gname]:.1f}")
    rec = r["rec"].set_index("component")
    for c in COMPONENTS:
        k = c.capitalize()
        put(f"rec{k}", "--" if not np.isfinite(rec.r_observed[c]) else f"{rec.r_observed[c]:.2f}")
        put(f"sd{k}", f"{rec.sd[c]:.1f}")
    for t, v in rs.items():
        put("dep" + t.replace("true_", "").capitalize(), f"{v:.2f}")
    e5 = r["e5"]
    for m in sorted(e5.cutoff_month.unique()):
        a = e5[(e5.model == main) & (e5.features == "All features") & (e5.cutoff_month == m)].RMSE.iloc[0]
        n = e5[(e5.model == main) & (e5.features == "Without satellite") & (e5.cutoff_month == m)].RMSE.iloc[0]
        put(f"lead{MON[m]}", f"{a:.1f}")
        put(f"leadNoSat{MON[m]}", f"{n:.1f}")
        put(f"satGain{MON[m]}", f"{n - a:.1f}")
        put(f"satGainPct{MON[m]}", f"{100 * (n - a) / n:.0f}")
    put("leadClim", f"{e5[e5.model == 'Climatology'].RMSE.iloc[0]:.1f}")
    e6 = r["e6"].set_index("subset")
    for key, s in [("", "All seasons"), ("Shock", "Shock seasons"), ("Other", "Other seasons")]:
        if s in e6.index:
            put(f"covEighty{key}", f"{100 * e6.cov80[s]:.1f}")
            put(f"covNinety{key}", f"{100 * e6.cov90[s]:.1f}")
            put(f"widthEighty{key}", f"{e6.width80[s]:.0f}")
            put(f"widthNinety{key}", f"{e6.width90[s]:.0f}")
    e6s = r["e6s"]
    put("nShock", int(e6s.shock.sum()))
    put("shockYears", ", ".join(str(y) for y in e6s[e6s.shock].year) or "none")
    put("covEightyMin", f"{100 * e6s.cov80.min():.0f}")
    put("covEightyMinYear", int(e6s.year[e6s.cov80.idxmin()]))
    rep = piv[piv.scenario == "base"]
    for key, c in [("Clim", "Climatology"), ("Ridge", "Ridge"), ("LGB", "LightGBM"), ("Ens", "Ensemble"),
                   ("NoSat", "Ensemble_no_satellite"), ("RandomCV", "Ensemble_random_cv")]:
        put(f"rep{key}", f"{rep[c].mean():.1f}")
        put(f"repSd{key}", f"{rep[c].std():.1f}")
    put("nRep", len(rep))
    put("repWinsEnsLGB", int((rep.Ensemble < rep.LightGBM).sum()))
    put("repWinsEnsRidge", int((rep.Ensemble < rep.Ridge).sum()))
    put("repWinsLGBRidge", int((rep.LightGBM < rep.Ridge).sum()))
    gain_rand = 100 * (1 - rep.Ensemble_random_cv / rep.Ensemble)
    put("repGainRandom", f"{gain_rand.mean():.0f}")
    put("repGainRandomSd", f"{gain_rand.std():.0f}")
    sat = 100 * (rep.Ensemble_no_satellite / rep.Ensemble - 1)
    put("repSatLoss", f"{sat.mean():.1f}")
    put("repSatLossSd", f"{sat.std():.1f}")
    put("repRedClim", f"{(100 * (1 - rep.Ensemble / rep.Climatology)).mean():.0f}")
    for sc in C.SCENARIOS:
        s = piv[piv.scenario == sc]
        if s.empty:
            continue
        k = {"nonlinear": "Nonlin", "map_error": "MapErr", "clouds": "Clouds", "no_pest": "NoPest"}[sc]
        for mk, c in [("Ridge", "Ridge"), ("LGB", "LightGBM"), ("Ens", "Ensemble"), ("NoSat", "Ensemble_no_satellite"),
                      ("Clim", "Climatology")]:
            put(f"scen{k}{mk}", f"{s[c].iloc[0]:.1f}")
        put(f"scen{k}SatLoss", f"{100 * (s.Ensemble_no_satellite.iloc[0] / s.Ensemble.iloc[0] - 1):.1f}")
    put("pestYears", ", ".join(str(e["year"]) for e in r["meta"]["pest_events"]))
    for k, v in b["params"].items():
        put("hp" + "".join(w.capitalize() for w in k.split("_")), f"{v:g}")
    for mname, mk in (("LightGBM", "LGB"), ("Ridge", "Ridge")):
        e3m = r["e3"][r["e3"].model == mname].set_index("group")
        for gname in e3m.index:
            put(f"abl{mk}{gname.capitalize()}", f"{e3m.dRMSE_pct[gname]:+.1f}")
            put(f"ablLo{mk}{gname.capitalize()}", f"{e3m.dRMSE_lo_pct[gname]:+.1f}")
            put(f"ablHi{mk}{gname.capitalize()}", f"{e3m.dRMSE_hi_pct[gname]:+.1f}")
    tim = json.loads((R / "timings.json").read_text())
    put("runMinutes", f"{sum(tim.values()) / 60:.0f}")
    if (R / "diag_ridge_season.csv").exists():
        dg = pd.read_csv(R / "diag_ridge_season.csv")
        f0 = dg.iloc[0]
        put("diagYear", int(f0.year))
        put("diagAlphaRow", f"{f0.alpha_rowloo:.2g}")
        put("diagAlphaLoso", f"{f0.alpha_loso:.3g}")
        put("diagRidgeRowFirst", f"{f0.rmse_ridge_rowloo:.1f}")
        put("diagRidgeLosoFirst", f"{f0.rmse_ridge_loso:.1f}")
        pool = lambda c: f"{np.sqrt((dg[c] ** 2).mean()):.1f}"
        put("diagRidgeLosoAll", pool("rmse_ridge_loso"))
        put("diagRidgeRowAll", pool("rmse_ridge_rowloo"))
        put("diagEnsLosoAll", pool("rmse_ens_loso"))
        put("diagLGBAll", pool("rmse_lgb"))
        put("diagAlphaRowMed", f"{dg.alpha_rowloo.median():.2g}")
        put("diagAlphaLosoMed", f"{dg.alpha_loso.median():.3g}")
    if (R / "e4_recovery_lgb.csv").exists():
        rl = pd.read_csv(R / "e4_recovery_lgb.csv").set_index("component").r_observed_lgb
        for c in COMPONENTS:
            put(f"recLGB{c.capitalize()}", "--" if not np.isfinite(rl[c]) else f"{rl[c]:.2f}")
        sl = pd.read_csv(R / "e4_shap_share_lgb.csv", index_col=0).share_pct
        for gname, v in sl.items():
            put(f"shapLGB{gname.capitalize()}", f"{v:.1f}")
    put("redClimLGBPct", f"{100 * (1 - met.RMSE['LightGBM'] / met.RMSE['Climatology']):.0f}")
    for key, m in [("FE", "PanelFE"), ("Ridge", "Ridge"), ("Ens", "Ensemble"), ("LGB", "LightGBM"), ("Clim", "Climatology")]:
        put(f"worst{key}", f"{r['bys'][m].max():.1f}")
        put(f"worstYear{key}", int(r["bys"][m].idxmax()))
    fail = lambda c: int((rep[c] > 1.5 * rep.LightGBM).sum())
    put("repFailRidge", fail("Ridge"))
    put("repFailEns", fail("Ensemble"))
    for key, c in [("Ridge", "Ridge"), ("LGB", "LightGBM"), ("Ens", "Ensemble"), ("RandomCV", "Ensemble_random_cv"),
                   ("NoSat", "Ensemble_no_satellite"), ("Clim", "Climatology")]:
        put(f"repMed{key}", f"{rep[c].median():.1f}")
        put(f"repMin{key}", f"{rep[c].min():.1f}")
        put(f"repMax{key}", f"{rep[c].max():.1f}")
    lsat = 100 * (e5_lgb_nosat := None) if False else None
    if (R / "diag_ridge_worlds.csv").exists():
        for tag, (_, w) in zip("ABC", pd.read_csv(R / "diag_ridge_worlds.csv").iterrows()):
            for col, nm in (("ridge_rowloo", "RidgeRow"), ("ridge_loso", "RidgeLoso"), ("lgb", "LGB"),
                            ("ens_rowloo", "EnsRow"), ("ens_loso", "EnsLoso")):
                put(f"diagW{tag}{nm}", f"{w[col]:.1f}")
    e5 = r["e5"]
    for mname in sorted(e5.cutoff_month.unique()):
        a_ = e5[(e5.model == "LightGBM") & (e5.features == "All features") & (e5.cutoff_month == mname)].RMSE.iloc[0]
        n_ = e5[(e5.model == "LightGBM") & (e5.features == "Without satellite") & (e5.cutoff_month == mname)].RMSE.iloc[0]
        r_ = e5[(e5.model == "Ridge") & (e5.features == "All features") & (e5.cutoff_month == mname)].RMSE.iloc[0]
        put(f"leadLGB{MON[mname]}", f"{a_:.1f}")
        put(f"leadLGBNoSat{MON[mname]}", f"{n_:.1f}")
        put(f"satGainLGB{MON[mname]}", f"{n_ - a_:.1f}")
        put(f"leadRidge{MON[mname]}", f"{r_:.1f}")
    txt = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items())
    write("macros.tex", "% auto-generated by figures.py - do not edit\n" + txt + "\n")
    return M


def main():
    r = load()
    main_m = r["best"]["main"]
    f1_study_area()
    f2_workflow(sum(len(v) for v in r["groups"].values()))
    f3_models(r["met"], main_m)
    f4_seasons(r["bys"], r["e6s"], main_m)
    f5_leakage(r["e2"], main_m)
    share = f6_attribution(r["e3"], r["shap"], r["groups"], main_m)
    rs = f7_dependence(r["dep"])
    f8_recovery(r["corr"])
    f9_leadtime(r["e5"], main_m)
    f10_conformal(r["e6s"])
    t_dgp()
    t_features(r["groups"])
    t_models(r["met"], main_m)
    t_leakage(r["e2"])
    t_recovery(r["rec"])
    t_ablation(r["e3"])
    t_byseason(r["bys"], r["met"])
    t_worlds(r["e7"])
    piv = robust_tables(r["e7"], r["met"], r["e2"], r["e3"], main_m)
    t_hparams(r["best"], r["sel"], r["tune"])
    M = macros(r, share, rs, piv)
    (R / "figure_text_check.json").write_text(json.dumps(CHECK, indent=1))
    print(f"{len(M)} macros; figures -> {FIG}; tables -> {TAB}")
    print("text-overlap check:", {k: (v if v else "OK") for k, v in CHECK.items()})


if __name__ == "__main__":
    main()
