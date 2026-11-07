"""Publication-style maps of Thailand.
(a) study-area locator map; (b) province choropleths of a satellite indicator (small multiples).

Boundaries and rivers: Natural Earth (public domain), downloaded to data/external/ on first run.
Panel (b) uses the benchmark's SIMULATED peak-season NDVI, aggregated from simulated districts to provinces
(by nearest provincial capital). Replace `indicator()` with real satellite statistics (e.g. from gee_extract.py)
and amphoe polygons to map observations.          python thailand_map.py --out paper/figures/thailand_map
"""
import argparse
import urllib.request
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from shapely.geometry import box

from figcheck import text_overlaps

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
EXT = ROOT / "data" / "external"
URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/{}.geojson"
LAYERS = {"adm1": "ne_10m_admin_1_states_provinces", "adm0": "ne_50m_admin_0_countries",
          "rivers": "ne_10m_rivers_lake_centerlines"}
CRS = "EPSG:32647"                                   # WGS 84 / UTM zone 47N
ISAN = [f"TH-{i}" for i in range(30, 50)]           # ISO 3166-2 codes of the 20 northeastern provinces
SYN_TO_ISO = {"Nakhon Ratchasima": 30, "Buriram": 31, "Surin": 32, "Si Sa Ket": 33, "Ubon Ratchathani": 34,
              "Yasothon": 35, "Chaiyaphum": 36, "Amnat Charoen": 37, "Bueng Kan": 38, "Nong Bua Lamphu": 39,
              "Khon Kaen": 40, "Udon Thani": 41, "Loei": 42, "Nong Khai": 43, "Maha Sarakham": 44, "Roi Et": 45,
              "Kalasin": 46, "Sakon Nakhon": 47, "Nakhon Phanom": 48, "Mukdahan": 49}
CITIES = {"Bangkok": (100.50, 13.75), "Chiang Mai": (98.99, 18.79), "Khon Kaen": (102.83, 16.43),
          "Nakhon Ratchasima": (102.10, 14.97), "Ubon Ratchathani": (104.85, 15.24), "Udon Thani": (102.79, 17.41)}
SEA, LAND, THAI, ISANC, RIVER = "#dbe7f1", "#f3f1ec", "#e3dfd4", "#a9cfaa", "#2f78b5"
HALO = [pe.withStroke(linewidth=2, foreground="white")]
TR = Transformer.from_crs(4326, CRS, always_xy=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "pdf.fonttype": 42})


def layer(key):
    EXT.mkdir(parents=True, exist_ok=True)
    f = EXT / f"{LAYERS[key]}.geojson"
    if not f.exists():
        urllib.request.urlretrieve(URL.format(LAYERS[key]), f)
    g = gpd.read_file(f)
    return g.rename(columns={c: c.lower() for c in g.columns if c != "geometry"})


def indicator(years):
    raw = ROOT / "data" / "raw"
    sat, dist = pd.read_csv(raw / "satellite_monthly.csv"), pd.read_csv(raw / "districts.csv")
    pk = sat.groupby(["district_id", "year"]).ndvi.max().rename("peak").reset_index()
    pk["anom"] = pk.peak - pk.groupby("district_id").peak.transform("mean")
    pk["iso"] = pk.district_id.map(dist.set_index("district_id").province).map(lambda p: f"TH-{SYN_TO_ISO[p]}")
    return pk[pk.year.isin(years)].groupby(["iso", "year"]).anom.mean().unstack()


def graticule(ax, lons=range(94, 110, 2), lats=range(0, 26, 2), size=6.5):
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xt, xl, yt, yl = [], [], [], []
    for lon in lons:
        X, Y = TR.transform(np.full(400, lon), np.linspace(-5, 30, 400))
        ax.plot(X, Y, color="#8a97a3", lw=0.35, ls=(0, (4, 3)), zorder=0.5)
        k = np.argmin(np.abs(Y - y0))
        if x0 < X[k] < x1:
            xt.append(X[k]); xl.append(f"{lon}°E")
    for lat in lats:
        X, Y = TR.transform(np.linspace(85, 115, 400), np.full(400, lat))
        ax.plot(X, Y, color="#8a97a3", lw=0.35, ls=(0, (4, 3)), zorder=0.5)
        k = np.argmin(np.abs(X - x0))
        if y0 < Y[k] < y1:
            yt.append(Y[k]); yl.append(f"{lat}°N")
    ax.set_xticks(xt, xl)
    ax.set_yticks(yt, yl)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.tick_params(length=2, labelsize=size, pad=1.5)


def scalebar(ax, x, y, km=200, segs=4, h=11e3, fs=6.3):
    seg = km * 1000 / segs
    for i in range(segs):
        ax.add_patch(Rectangle((x + i * seg, y), seg, h, fc="k" if i % 2 == 0 else "w", ec="k", lw=0.5, zorder=9))
    labs = ((0, "0"), (segs, f"{km} km")) if segs <= 2 else ((0, "0"), (segs // 2, f"{km // 2}"), (segs, f"{km} km"))
    for i, lab in labs:
        ax.text(x + i * seg, y + 2.2 * h, lab, ha="center", va="bottom", fontsize=fs, zorder=9, path_effects=HALO)


def north(ax, x, y, size=70e3, fs=9):
    ax.annotate("", xy=(x, y + size), xytext=(x, y), zorder=9,
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7", lw=1.4, color="k"))
    ax.text(x, y + size + 12e3, "N", ha="center", va="bottom", fontweight="bold", fontsize=fs, path_effects=HALO)


def main(out):
    a1, a0, rv = layer("adm1"), layer("adm0"), layer("rivers")
    th = a1[a1.adm0_a3 == "THA"]
    isan = th[th.iso_3166_2.isin(ISAN)]
    region = box(88, -4, 116, 28)
    ctx = a0[a0.intersects(region)].clip(region)
    rv = rv.cx[95:108, 4:23].clip(box(95, 4, 108, 23))
    th_p, is_p, ctx_p, rv_p = (g.to_crs(CRS) for g in (th, isan, ctx, rv))
    isan_line, th_line = is_p.dissolve().boundary, th_p.dissolve().boundary
    years = [2019, 2021, 2023, 2025]
    val = indicator(years)

    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1, 1], left=0.045, right=0.99, top=0.88, bottom=0.13,
                          wspace=0.06, hspace=0.14)
    # ---------------------------------------------------------------- (a) locator map
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(SEA)
    ctx_p.plot(ax=ax, color=LAND, edgecolor="#b8b2a6", lw=0.4, zorder=1)
    th_p.plot(ax=ax, color=THAI, edgecolor="#a59f92", lw=0.25, zorder=2)
    is_p.plot(ax=ax, color=ISANC, edgecolor="#5f8361", lw=0.3, zorder=3)
    th_line.plot(ax=ax, color="#4a4640", lw=0.7, zorder=4)
    isan_line.plot(ax=ax, color="#1f4d27", lw=1.1, zorder=4)
    rv_p.plot(ax=ax, color=RIVER, lw=0.7, zorder=5)
    x0, y0, x1, y1 = th_p.total_bounds
    pad = 70e3
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    graticule(ax)
    for name, (lon, lat) in CITIES.items():
        x, y = TR.transform(lon, lat)
        cap = name == "Bangkok"
        ax.plot(x, y, marker="*" if cap else "o", ms=8 if cap else 3, color="#b2182b" if cap else "k", zorder=6,
                markeredgecolor="white", markeredgewidth=0.5)
        ax.text(x + 16e3, y + 9e3, name, fontsize=6.5, zorder=6, path_effects=HALO)
    for txt, lon, lat, it in (("MYANMAR", 97.75, 16.3, False), ("LAOS", 101.9, 18.9, False),
                              ("CAMBODIA", 104.2, 12.9, False), ("MALAYSIA", 101.7, 5.4, False),
                              ("Andaman\nSea", 97.4, 9.6, True), ("Gulf of\nThailand", 101.3, 10.8, True)):
        ax.text(*TR.transform(lon, lat), txt, ha="center", va="center", fontsize=6.8, color="#5e6b78" if it else "#77726a",
                style="italic" if it else "normal", zorder=6)
    ax.text(*TR.transform(103.7, 15.75), "NORTHEAST\n(ISAN)", ha="center", va="center", fontsize=7.5,
            fontweight="bold", color="#173d1e", zorder=6, path_effects=HALO)
    scalebar(ax, x0 - pad + 35e3, y0 - pad + 40e3)
    north(ax, x0 - pad + 70e3, y1 + pad - 190e3)
    ax.legend(handles=[Patch(fc=ISANC, ec="#1f4d27", label="Northeastern provinces (20)"),
                       Patch(fc=THAI, ec="#a59f92", label="Other Thai provinces"),
                       Line2D([], [], color=RIVER, lw=1.2, label="Major rivers"),
                       Line2D([], [], color="#b2182b", marker="*", ls="", ms=8, label="Bangkok")],
              loc="upper right", fontsize=6, frameon=True, framealpha=0.92, edgecolor="#999")
    ins = ax.inset_axes([0.6, 0.012, 0.385, 0.25])
    ins.set_facecolor(SEA)
    ctx.plot(ax=ins, color=LAND, edgecolor="#b8b2a6", lw=0.2)
    th.dissolve().plot(ax=ins, color="#b2182b")
    ins.set_xlim(89, 116)
    ins.set_ylim(-3, 27)
    ins.set_aspect(1 / np.cos(np.radians(12)))
    ins.set_xticks([])
    ins.set_yticks([])
    ins.text(0.04, 0.92, "Southeast Asia", transform=ins.transAxes, fontsize=5.8, color="#444")
    ax.set_title("(a) Study area: Northeast Thailand (Isan)", loc="left")
    for a_ in (ax, ins):
        a_.set_xlabel("")
        a_.set_ylabel("")
    # ---------------------------------------------------------------- (b) small multiples
    vmax = float(np.ceil(np.nanmax(np.abs(val.to_numpy())) * 100) / 100)
    bx0, by0, bx1, by1 = is_p.total_bounds
    bp = 30e3
    for k, yr in enumerate(years):
        a = fig.add_subplot(gs[k // 2, 1 + k % 2])
        a.set_facecolor(SEA)
        ctx_p.plot(ax=a, color=LAND, edgecolor="#cfc9bd", lw=0.3)
        th_p.plot(ax=a, color=THAI, edgecolor="#bdb7aa", lw=0.25)
        is_p.assign(v=is_p.iso_3166_2.map(val[yr])).plot(
            ax=a, column="v", cmap="BrBG", vmin=-vmax, vmax=vmax, edgecolor="white", lw=0.6,
            missing_kwds={"color": "#d9d9d9", "hatch": "////", "edgecolor": "white"})
        isan_line.plot(ax=a, color="#333", lw=0.8)
        rv_p.plot(ax=a, color=RIVER, lw=0.8)
        a.set_xlim(bx0 - bp, bx1 + bp)
        a.set_ylim(by0 - bp, by1 + bp)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(str(yr), loc="left")
        a.set_xlabel("")
        a.set_ylabel("")
        m = val[yr].mean()
        a.text(0.97, 0.04, f"regional mean {m:+.3f}", transform=a.transAxes, ha="right", fontsize=6.5, path_effects=HALO)
        if k == 3:
            scalebar(a, bx0 - bp + 15e3, by0 - bp + 18e3, km=100, segs=2, h=7e3)
    cax = fig.add_axes([0.43, 0.075, 0.5, 0.016])
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(-vmax, vmax), "BrBG"), cax=cax, orientation="horizontal")
    cb.set_label("Peak-season NDVI anomaly vs. 2008-2025 district mean (-)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.text(0.415, 0.935, "(b) Peak-season NDVI anomaly by province", fontweight="bold", fontsize=9)
    fig.text(0.415, 0.912, "SYNTHETIC benchmark values (simulated, aggregated to provinces) - not satellite observations",
             fontsize=7, color="#b2182b")
    fig.text(0.5, 0.012, "Boundaries and rivers: Natural Earth (public domain). Projection: WGS 84 / UTM zone 47N "
             "(EPSG:32647). Graticule every 2°.", ha="center", fontsize=6.5, color="0.35")
    print("text-overlap check:", text_overlaps(fig) or "OK")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    print("saved", out.with_suffix(".png"), out.with_suffix(".pdf"))


def wrap(name):
    if len(name) <= 9 or " " not in name:
        return name
    k = min((i for i, ch in enumerate(name) if ch == " "), key=lambda i: abs(i - len(name) / 2))
    return name[:k] + "\n" + name[k + 1:]


def paper_figure(out):
    """Print-size (6.5 in) study-area figure for the manuscript + LaTeX macros for the text."""
    from simulate import TOWNS
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.2, "axes.titleweight": "bold"})
    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    a1, a0, rv = layer("adm1"), layer("adm0"), layer("rivers")
    th = a1[a1.adm0_a3 == "THA"]
    isan = th[th.iso_3166_2.isin(ISAN)]
    region = box(88, -4, 116, 28)
    ctx = a0[a0.intersects(region)].clip(region)
    rv = rv.cx[95:108, 4:23].clip(box(95, 4, 108, 23))
    th_p, is_p, ctx_p, rv_p = (g.to_crs(CRS) for g in (th, isan, ctx, rv))
    isan_line, th_line = is_p.dissolve().boundary, th_p.dissolve().boundary
    years = [2019, 2021, 2023, 2025]
    val = indicator(years)
    fig = plt.figure(figsize=(6.5, 5.95))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.7, 0.8], hspace=0.11, left=0.055, right=0.995, top=0.972,
                             bottom=0.085)
    top = outer[0].subgridspec(1, 2, width_ratios=[0.75, 1.25], wspace=0.12)
    bot = outer[1].subgridspec(1, 4, wspace=0.04)
    # (a) location
    ax = fig.add_subplot(top[0])
    ax.set_facecolor(SEA)
    ctx_p.plot(ax=ax, color=LAND, edgecolor="#b8b2a6", lw=0.3, zorder=1)
    th_p.plot(ax=ax, color=THAI, edgecolor="#a59f92", lw=0.15, zorder=2)
    is_p.plot(ax=ax, color=ISANC, edgecolor="#5f8361", lw=0.2, zorder=3)
    th_line.plot(ax=ax, color="#4a4640", lw=0.5, zorder=4)
    isan_line.plot(ax=ax, color="#1f4d27", lw=0.8, zorder=4)
    rv_p.plot(ax=ax, color=RIVER, lw=0.45, zorder=5)
    x0, y0, x1, y1 = th_p.total_bounds
    pad = 60e3
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    graticule(ax, size=5.3)
    bkx, bky = TR.transform(100.50, 13.75)
    ax.plot(bkx, bky, marker="*", ms=6, color="#b2182b", markeredgecolor="white", markeredgewidth=0.4, zorder=6)
    ax.text(bkx + 25e3, bky - 12e3, "Bangkok", fontsize=5.3, va="top", zorder=6, path_effects=halo)
    for txt, lon, lat, it in (("MYANMAR", 97.75, 16.6, False), ("LAOS", 101.75, 19.3, False),
                              ("CAMBODIA", 104.3, 13.0, False), ("MALAYSIA", 101.4, 5.4, False),
                              ("Andaman\nSea", 97.5, 9.3, True), ("Gulf of\nThailand", 101.3, 10.9, True)):
        ax.text(*TR.transform(lon, lat), txt, ha="center", va="center", fontsize=4.9, zorder=6,
                color="#5e6b78" if it else "#77726a", style="italic" if it else "normal")
    ax.text(*TR.transform(103.6, 15.9), "ISAN", ha="center", va="center", fontsize=6, fontweight="bold",
            color="#173d1e", zorder=6, path_effects=halo)
    scalebar(ax, x0 - pad + 30e3, y0 - pad + 35e3, km=200, segs=2, h=14e3, fs=5)
    north(ax, x0 - pad + 75e3, y1 + pad - 240e3, size=90e3, fs=7)
    ins = ax.inset_axes([0.61, 0.05, 0.37, 0.2])
    ins.set_facecolor(SEA)
    ctx.plot(ax=ins, color=LAND, edgecolor="#b8b2a6", lw=0.15)
    th.dissolve().plot(ax=ins, color="#b2182b")
    ins.set_xlim(89, 116)
    ins.set_ylim(-3, 27)
    ins.set_aspect(1 / np.cos(np.radians(12)))
    ins.set_xticks([])
    ins.set_yticks([])
    ax.set_title("(a) Location", loc="left")
    # (b) provinces, rivers, capitals
    bx = fig.add_subplot(top[1])
    bx.set_facecolor(SEA)
    ctx_p.plot(ax=bx, color=LAND, edgecolor="#cfc9bd", lw=0.4, zorder=1)
    th_p.plot(ax=bx, color="#ece9e2", edgecolor="#cfc9bd", lw=0.3, zorder=2)
    is_p.plot(ax=bx, color="#e3efdf", edgecolor="#8fa98c", lw=0.5, zorder=3)
    isan_line.plot(ax=bx, color="#1f4d27", lw=1.0, zorder=4)
    rv_p.plot(ax=bx, color=RIVER, lw=0.9, zorder=5)
    caps = {f"TH-{SYN_TO_ISO[n]}": TR.transform(lon, lat) for n, (lon, lat) in TOWNS.items()}
    for cx, cy in caps.values():
        bx.plot(cx, cy, "o", ms=2.3, color="#222", zorder=6)
    for _, r in is_p.iterrows():
        p = r.geometry.representative_point()
        lx, ly = p.x, p.y
        cx, cy = caps[r.iso_3166_2]
        lab = wrap(r["name"])
        halfw = 0.5 * max(len(t) for t in lab.split("\n")) * 5.6e3 + 3e3     # ~5.6 km per character at 4.9 pt
        halfh = (5e3 if "\n" not in lab else 10e3) + 3e3
        if abs(cx - lx) < halfw and abs(cy - ly) < halfh:          # label would cover its capital: put it below
            lx, ly = cx, cy - halfh - 4e3
        bx.text(lx, ly, lab, fontsize=4.9, ha="center", va="center", color="#2b2b2b", linespacing=0.95,
                zorder=7, path_effects=halo)
    bx0, by0, bx1, by1 = is_p.total_bounds
    bp = 25e3
    lim = ((bx0 - bp, bx1 + bp), (by0 - bp, by1 + bp))
    bx.set_xlim(*lim[0])
    bx.set_ylim(*lim[1])
    graticule(bx, lons=range(100, 108), lats=range(13, 20), size=5.3)
    for txt, lon, lat in (("LAOS", 102.35, 18.55), ("CAMBODIA", 103.4, 14.05)):
        bx.text(*TR.transform(lon, lat), txt, ha="center", va="center", fontsize=5, color="#77726a", zorder=6)
    scalebar(bx, lim[0][0] + 20e3, lim[1][0] + 18e3, km=100, segs=2, h=7e3, fs=5)
    bx.legend(handles=[Patch(fc="#e3efdf", ec="#1f4d27", label="Northeastern province"),
                       Line2D([], [], color=RIVER, lw=1.2, label="Mekong, Mun and Chi rivers"),
                       Line2D([], [], color="#222", marker="o", ls="", ms=2.5, label="Provincial capital")],
              loc="upper right", fontsize=5, frameon=True, framealpha=0.92, edgecolor="#bbb")
    bx.set_title("(b) Provinces, rivers and provincial capitals", loc="left")
    # (c-f) simulated satellite indicator
    vmax = float(np.ceil(np.nanmax(np.abs(val.to_numpy())) * 100) / 100)
    for k, yr in enumerate(years):
        c = fig.add_subplot(bot[k])
        c.set_facecolor(SEA)
        ctx_p.plot(ax=c, color=LAND, edgecolor="#d5cfc4", lw=0.25)
        th_p.plot(ax=c, color="#ece9e2", edgecolor="#d5cfc4", lw=0.2)
        is_p.assign(v=is_p.iso_3166_2.map(val[yr])).plot(
            ax=c, column="v", cmap="BrBG", vmin=-vmax, vmax=vmax, edgecolor="white", lw=0.35,
            missing_kwds={"color": "#d9d9d9", "hatch": "////", "edgecolor": "white"})
        isan_line.plot(ax=c, color="#333", lw=0.6)
        rv_p.plot(ax=c, color=RIVER, lw=0.5)
        c.set_xlim(*lim[0])
        c.set_ylim(*lim[1])
        c.set_xticks([])
        c.set_yticks([])
        c.set_xlabel("")
        c.set_ylabel("")
        c.set_title(f"({'cdef'[k]}) {yr}  (mean {val[yr].mean():+.3f})", loc="left", fontsize=6.6)
    for a_ in (ax, bx, ins):
        a_.set_xlabel("")
        a_.set_ylabel("")
    cax = fig.add_axes([0.3, 0.078, 0.4, 0.011])
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(-vmax, vmax), "BrBG"), cax=cax, orientation="horizontal")
    cb.set_label("Simulated peak-season NDVI anomaly relative to the 2008-2025 mean (-)", fontsize=6)
    cb.ax.tick_params(labelsize=5.3, length=2)
    print("text-overlap check:", text_overlaps(fig) or "OK")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    m = [f"\\newcommand{{\\mapNprov}}{{{val.shape[0]}}}"]
    for k, yr in enumerate(years):
        L = "ABCD"[k]
        m += [f"\\newcommand{{\\mapYear{L}}}{{{yr}}}", f"\\newcommand{{\\mapAnom{L}}}{{{val[yr].mean():+.3f}}}",
              f"\\newcommand{{\\mapNeg{L}}}{{{int((val[yr] < 0).sum())}}}",
              f"\\newcommand{{\\mapPos{L}}}{{{int((val[yr] > 0).sum())}}}"]
    (ROOT / "paper" / "tables" / "macros_map.tex").write_text("% generated by thailand_map.py --paper\n" + "\n".join(m) + "\n")
    print("saved", out.with_suffix(".pdf"), "|", " ".join(x.split("{")[1].strip("\\}") + "=" + x.split("}{")[1].strip("}") for x in m))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures" / "thailand_map"))
    ap.add_argument("--paper", action="store_true", help="print-size manuscript figure + LaTeX macros")
    a = ap.parse_args()
    if a.paper:
        paper_figure(ROOT / "paper" / "figures" / "fig_thailand")
    else:
        main(a.out)
