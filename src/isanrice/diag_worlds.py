"""POST-HOC: do Ridge failures in other worlds share the row-wise-LOO-penalty mechanism?"""
import json
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .diagnostics import loso_alpha, ridge
from .features import build_features, feature_list
from .models import make_model
from .simulate import simulate

warnings.filterwarnings("ignore")
Y = C.YIELD_COL


def main():
    best = json.loads((C.RESULTS / "e1_best.json").read_text())
    P, target = best["params"], best["target"]
    rows = []
    for sc, seed in [("base", 1002), ("base", 1008), ("nonlinear", 42)]:
        df, g = build_features(simulate(seed=seed, scenario=sc))
        f = feature_list(g)
        se, n = dict.fromkeys(("ridge_rowloo", "ridge_loso", "lgb", "ens_rowloo", "ens_loso"), 0.0), 0
        for yr in C.TEST_YEARS:
            tr, te = df[df.year < yr], df[df.year == yr]
            y = (tr[Y] if target == "level" else tr.target_anom).to_numpy()
            off = 0.0 if target == "level" else te.y_ref.to_numpy()
            p1 = make_model("Ridge").fit(tr[f], y).predict(te[f])
            p2 = ridge(loso_alpha(tr, f, y)).fit(tr[f], y).predict(te[f])
            pl = make_model("LightGBM", P).fit(tr[f], y).predict(te[f])
            for k, p in (("ridge_rowloo", p1), ("ridge_loso", p2), ("lgb", pl), ("ens_rowloo", .5 * (pl + p1)), ("ens_loso", .5 * (pl + p2))):
                se[k] += float(((off + p - te[Y].to_numpy()) ** 2).sum())
            n += len(te)
        rows.append({"scenario": sc, "seed": seed, **{k: float(np.sqrt(v / n)) for k, v in se.items()}})
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS / "diag_ridge_worlds.csv", index=False)


if __name__ == "__main__":
    main()
