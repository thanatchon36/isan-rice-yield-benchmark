"""README banner -> docs/assets/banner.png: the synthetic study region (mean rice yield of the 200 simulated
districts, 2008-2025) beside the project name, in the dark layout of the thainowcast banner.

Usage: ``python scripts/make_banner.py`` (or ``make banner``). Needs data/raw/ (``make simulate``).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from isanrice import config as C  # noqa: E402
from isanrice.figures import district_cells  # noqa: E402
from isanrice.simulate import REGION_OUTLINE, RIVERS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "banner.png"
BG, AMBER, INK, MUTED = "#04070d", "#f5a623", "#e8e9ec", "#8f9bb3"
# dry -> ripe paddy: dark olive through amber to pale gold
PADDY = LinearSegmentedColormap.from_list("paddy", ["#1b2516", "#4d5a1e", "#a8861d", "#f5a623", "#ffe8a3"])


def main():
    d = pd.read_csv(C.DATA_RAW / "districts.csv")
    d["y"] = d.district_id.map(pd.read_csv(C.DATA_RAW / "yields.csv").groupby("district_id")[C.YIELD_COL].mean())
    cells = district_cells(d)
    vals = d.y.to_numpy(float)
    patches, cv = [], []
    for c, v in zip(cells, vals):
        for g in ([] if c is None else getattr(c, "geoms", [c])):
            if g.geom_type == "Polygon" and not g.is_empty:
                patches.append(Polygon(np.asarray(g.exterior.coords), closed=True))
                cv.append(v)

    fig = plt.figure(figsize=(16.8, 6.24), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.035, 0.05, 0.30, 0.90], facecolor=BG)
    pc = PatchCollection(patches, cmap=PADDY, norm=Normalize(*np.nanpercentile(vals, [2, 98])),
                         edgecolor=BG, linewidth=0.6)
    pc.set_array(np.asarray(cv))
    ax.add_collection(pc)
    ax.plot(*REGION_OUTLINE.T, color="#2b3550", lw=1.0)
    for r in RIVERS.values():
        ax.plot(*np.asarray(r).T, color="#3d7cc9", lw=1.6, alpha=0.9)
    ax.set_xlim(101.0, 105.9)
    ax.set_ylim(13.8, 18.65)
    ax.set_aspect(1 / np.cos(np.radians(16)))
    ax.axis("off")

    x = 0.36
    fig.text(x, 0.60, "isan-rice-yield-benchmark", color=AMBER, fontsize=44, fontweight="bold")
    fig.text(x, 0.40, "Leakage-safe, explainable rice yield forecasting:\n"
             "a ground-truth simulation benchmark for Northeast Thailand",
             color=INK, fontsize=19, linespacing=1.55)
    fig.text(x, 0.25, "satellite · weather · water · prices · maps  →  81 features · 8 learners · SHAP checked against truth",
             color=MUTED, fontsize=13.5)
    fig.text(0.985, 0.035, "Simulated mean main-season rice yield 2008-2025, 200 districts (synthetic data) · "
             "rivers Mekong, Mun and Chi (schematic)", color=MUTED, fontsize=9, ha="right")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=BG, dpi=100)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
