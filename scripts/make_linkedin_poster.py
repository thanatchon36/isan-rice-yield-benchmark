"""LinkedIn launch poster -> docs/assets/linkedin_poster.png (1440 x 1800, 4:5), in the layout of the thainowcast
poster: one headline number, the simulated study region, the ground-truth check of the explanations, four key
figures and the links. Every number is read from paper/tables/macros.tex (the file the paper uses).

Fonts: Fira Sans (SIL Open Font License), downloaded once into data/external/fonts/; falls back to DejaVu Sans.
Usage: ``python scripts/make_linkedin_poster.py`` (or ``make poster``).
"""
import re
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager, patheffects  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

from isanrice import config as C  # noqa: E402
from isanrice.figures import district_cells  # noqa: E402
from isanrice.simulate import REGION_OUTLINE, RIVERS, TOWNS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "linkedin_poster.png"
FONTS = ROOT / "data" / "external" / "fonts"
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/firasans/FiraSans-{}.ttf"
DOI = "10.5281/zenodo.24036674"
BG, PANEL, RULE = "#04070d", "#0a1222", "#1c2740"
AMBER, GOLD, INK, MUTED, DIM = "#f5a623", "#ffe8a3", "#eef0f4", "#a9b4c8", "#6f7a90"
GOOD, BAD = "#f5a623", "#e0603a"
PADDY = LinearSegmentedColormap.from_list("paddy", ["#1b2516", "#4d5a1e", "#a8861d", "#f5a623", "#ffe8a3"])


def fonts():
    family = "DejaVu Sans"
    try:
        FONTS.mkdir(parents=True, exist_ok=True)
        for w in ("Regular", "Medium", "SemiBold", "Bold"):
            f = FONTS / f"FiraSans-{w}.ttf"
            if not f.exists():
                urllib.request.urlretrieve(FONT_URL.format(w), f)
            font_manager.fontManager.addfont(str(f))
        family = "Fira Sans"
    except OSError as e:  # offline: keep the fallback
        print("Fira Sans unavailable, using DejaVu Sans:", e)
    plt.rcParams.update({"font.family": family, "savefig.bbox": None})


def macros():
    text = (ROOT / "paper" / "tables" / "macros.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}", text))


def region_map(ax):
    d = pd.read_csv(C.DATA_RAW / "districts.csv")
    d["y"] = d.district_id.map(pd.read_csv(C.DATA_RAW / "yields.csv").groupby("district_id")[C.YIELD_COL].mean())
    vals, patches, cv = d.y.to_numpy(float), [], []
    for c, v in zip(district_cells(d), vals):
        for g in ([] if c is None else getattr(c, "geoms", [c])):
            if g.geom_type == "Polygon" and not g.is_empty:
                patches.append(Polygon(np.asarray(g.exterior.coords), closed=True))
                cv.append(v)
    # soft glow under the region
    ax.add_patch(Polygon(REGION_OUTLINE, closed=True, facecolor=AMBER, alpha=0.16, edgecolor=AMBER, lw=16,
                         joinstyle="round"))
    pc = PatchCollection(patches, cmap=PADDY, norm=Normalize(*np.nanpercentile(vals, [2, 98])), edgecolor=BG,
                         linewidth=0.7)
    pc.set_array(np.asarray(cv))
    ax.add_collection(pc)
    for r in RIVERS.values():
        ax.plot(*np.asarray(r).T, color="#4a8fe0", lw=2.2, alpha=0.95, solid_capstyle="round")
    ax.set_xlim(101.0, 105.9)
    ax.set_ylim(13.8, 18.65)
    ax.set_aspect(1 / np.cos(np.radians(16)))
    ax.axis("off")
    return d


def main():
    fonts()
    M = macros()
    W, H = 1440, 1800
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)

    def px(x, y):  # pixel coordinates from the top-left -> figure fraction
        return x / W, 1 - y / H

    def text(x, y, s, size, color=INK, weight="regular", **kw):
        fig.text(*px(x, y), s, fontsize=size * 0.72, color=color, fontweight=weight, va="top", **kw)

    # vignette behind the map
    bg = fig.add_axes([0, 0, 1, 1], zorder=-2)
    yy, xx = np.mgrid[0:1:300j, 0:1:240j]
    glow = np.exp(-(((xx - 0.72) / 0.32) ** 2 + ((yy - 0.66) / 0.24) ** 2))
    bg.imshow(glow, cmap=LinearSegmentedColormap.from_list("v", [BG, "#0d1a33"]), extent=[0, 1, 0, 1],
              aspect="auto", origin="lower")
    bg.axis("off")

    # header
    text(94, 66, "ISAN-RICE-YIELD-BENCHMARK", 27, AMBER, "bold")
    text(560, 68, "open-source benchmark for crop-yield machine learning", 25, MUTED)

    # headline
    text(96, 150, "up to", 34, GOLD)
    text(78, 188, f"{M['gainRandom']}%", 230, AMBER, "bold")
    text(104, 482, "That is how much better a rice-yield\nmodel looks under random cross-validation\n"
                   "than when it must forecast seasons\nit has never seen.", 31, INK, linespacing=1.45)
    text(104, 722, "So we built a simulated Isan\nwhere the truth is known, and\nchecked the forecasts and\n"
                   "the explanations against it.", 31, AMBER, "semibold", linespacing=1.45)

    # map
    ax = fig.add_axes([0.50, 0.385, 0.48, 0.535])
    region_map(ax)
    halo = [patheffects.withStroke(linewidth=3, foreground=BG)]
    for name, dx, ha in (("Khon Kaen", 8, "left"), ("Udon Thani", 8, "left"), ("Ubon Ratchathani", -8, "right"),
                         ("Nakhon Ratchasima", 8, "left")):
        lon, lat = TOWNS[name]
        ax.scatter([lon], [lat], s=30, color=INK, edgecolor=BG, lw=1.2, zorder=5)
        ax.annotate(name, (lon, lat), xytext=(dx, 7), textcoords="offset points", color=INK, ha=ha,
                    fontsize=13, zorder=6, fontweight="medium", path_effects=halo)
    text(1346, 1012, f"200 simulated districts · mean rice yield {C.YEARS[0]}-{C.YEARS[-1]}", 17, DIM, ha="right")

    # ground-truth check of the explanations
    text(94, 1050, "Do the explanations match the truth?", 30, INK, "semibold")
    text(94, 1094, "correlation between each mechanism's grouped SHAP value and its true contribution",
         19, MUTED)
    rows = [("Drought", float(M["recDrought"]), "recovered"),
            ("Salinity", float(M["recSalinity"]), "partly recovered"),
            ("Technology trend", float(M["recTrend"]), "absorbed into prices"),
            ("Heat stress", float(M["recHeat"]), "misattributed to moisture")]
    x0, x1, zero = 380, 1060, 720  # bar axis in pixels: r = -1 .. 1
    y = 1150
    for name, r, verdict in rows:
        color = GOOD if r >= 0.3 else BAD
        text(94, y, name, 25, INK, "medium")
        fig.patches.append(FancyBboxPatch(px(x0, y + 30), (x1 - x0) / W, 2 / H, boxstyle="square,pad=0",
                                          transform=fig.transFigure, color=RULE))
        bx = zero + min(r, 0) * (x1 - zero)
        fig.patches.append(FancyBboxPatch(px(bx, y + 40), abs(r) * (x1 - zero) / W, 22 / H,
                                          boxstyle="round,pad=0,rounding_size=0.002", transform=fig.transFigure,
                                          color=color))
        text(x1 + 24, y - 2, f"r = {r:+.2f}".replace("-", "\u2212"), 25, color, "semibold")
        text(x1 + 24, y + 30, verdict, 18, MUTED)
        y += 82
    fig.patches.append(FancyBboxPatch(px(zero - 1, y - 20), 2 / W, (y - 20 - 1160) / H, boxstyle="square,pad=0",
                                      transform=fig.transFigure, color=DIM))

    # key figures
    fig.patches.append(FancyBboxPatch(px(94, 1488), (W - 188) / W, 1 / H, boxstyle="square,pad=0",
                                      transform=fig.transFigure, color=RULE))
    lgb, clim = float(M["rmseLGB"]), float(M["rmseClim"])
    stats = [(f"{M['nDistricts']}×{M['nSeasons']}", "districts × seasons,\ntrue yield components"),
             (M["nFeatures"], "leakage-safe features\nin 8 groups"),
             (f"−{round(100 * (1 - lgb / clim))}%", "forecast error vs.\nclimatology (LightGBM)"),
             (f"{M['covEighty']}%", "coverage of 80%\nconformal intervals")]
    for i, (big, small) in enumerate(stats):
        x = 94 + i * 318
        text(x, 1510, big, 58, AMBER, "bold")
        text(x, 1590, small, 20, MUTED, linespacing=1.4)

    # links
    fig.patches.append(FancyBboxPatch(((86 + 8) / W, 1 - 1752 / H), (W - 188) / W, 70 / H,
                                      boxstyle="round,pad=0.006,rounding_size=0.012", transform=fig.transFigure,
                                      facecolor=PANEL, edgecolor=RULE, lw=1.4))
    text(124, 1690, "github.com/thanatchon36/isan-rice-yield-benchmark", 21, INK, "medium")
    text(124, 1724, "code · paper · web reader", 16, DIM)
    text(756, 1690, "hf.co/datasets/thanatchon36/isan-rice-yield-benchmark", 21, INK, "medium")
    text(756, 1724, "dataset with the true yield components", 16, DIM)

    # footer
    text(94, 1772, f"Thanatchon Chongmankhong  ·  views are my own  ·  DOI {DOI}", 17, MUTED)
    text(1346, 1772, "All data are synthetic: the benchmark tests methods, not Thai rice yields.", 15, DIM,
         ha="right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=BG, dpi=100)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
