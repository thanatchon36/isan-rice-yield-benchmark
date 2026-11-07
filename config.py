"""Central configuration of the benchmark study (edit here to adapt crop / region / design)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figures"
TABLES = ROOT / "paper" / "tables"

SEED = 42

# ---------------------------------------------------------------- study design
TARGET_CROP = "rice"                       # main-season (wet-season) rice
COMPETING_CROPS = ["cassava", "sugarcane", "maize"]
YIELD_COL = "yield_kg_rai"
YIELD_UNIT = "kg/rai"
KG_RAI_TO_T_HA = 6.25 / 1000               # 1 rai = 0.16 ha

YEARS = list(range(2008, 2026))
SEASON_MONTHS = [5, 6, 7, 8, 9, 10, 11]    # May (sowing) ... Nov (harvest)
STAGES = {"est": [5, 6, 7], "veg": [8, 9], "rep": [10, 11]}
FORECAST_MONTH = 11
LEAD_TIME_MONTHS = [7, 8, 9, 10, 11]

# ---------------------------------------------------------------- validation
TEST_YEARS = list(range(2018, 2026))       # rolling origin: season t uses seasons < t only
TUNE_VALID_YEARS = [2015, 2016, 2017]      # tuning + model selection (never sees test seasons)
N_TUNE_TRIALS = 15                         # x 2 target formulations
HISTORY_YEARS = 3
MIN_TRAIN_SEASONS = 3
ALPHAS = (0.2, 0.1)                        # 80 % and 90 % conformal intervals
BOOT_B = 2000
N_SEEDS = 10
SCENARIOS = ["nonlinear", "map_error", "clouds", "no_pest"]

# ---------------------------------------------------------------- economics
TYPICAL_YIELD = {"rice": 400, "cassava": 3500, "sugarcane": 11000, "maize": 700}       # kg/rai
TRANSPORT_COST = {"rice": 0.010, "cassava": 0.006, "sugarcane": 0.004, "maize": 0.010}  # THB/kg/km
PRICE_DISTANCE_COL = {"rice": "dist_market_km", "cassava": "dist_market_km",
                      "sugarcane": "dist_mill_km", "maize": "dist_market_km"}

# ---------------------------------------------------------------- spatial
N_DISTRICTS = 200
K_NEIGHBORS = 8

# ---------------------------------------------------------------- feature groups
GROUP_LABELS = {
    "history": "Previous yields", "moisture": "Humidity & soil moisture",
    "water": "Water (rain, reservoir, irrigation)", "climate": "Climate (temperature, GDD)",
    "satellite": "Satellite (NDVI, EVI, LSWI, LST)", "price": "Prices (own & competing crops)",
    "geography": "Geography (terrain, soil)", "map": "Map layers & spatial lags",
}
GROUP_COLORS = {   # Okabe-Ito (colour-blind safe)
    "history": "#999999", "moisture": "#56B4E9", "water": "#0072B2", "climate": "#E69F00",
    "satellite": "#009E73", "price": "#CC79A7", "geography": "#D55E00", "map": "#000000",
}
