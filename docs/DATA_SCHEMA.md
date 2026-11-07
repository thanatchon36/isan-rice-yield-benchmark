# Input schema (data/raw/*.csv)
| File | Columns | Real-data source (Thailand) |
|---|---|---|
| `districts.csv` | district_id, province, lon, lat, basin, elevation_m, slope_deg, sand_pct, soc_pct, soil_ph, saline_share, soil_suit (S1/S2/S3/N), irrigation_share, paddy_share, flood_freq, rain_normal_mm, dist_river_km, dist_mill_km, dist_market_km | DOPA/GISTDA boundaries; SRTM; LDD soil & suitability maps; SoilGrids; RID irrigation areas; GISTDA flood maps; OSM / mill registers |
| `climate_monthly.csv` | district_id, year, month (5-11), rain_mm, tmean_c, tmax_c, rh_pct, soil_moist (m3/m3) | CHIRPS; ERA5-Land; TMD stations (scripts/gee_extract.py) |
| `satellite_monthly.csv` | district_id, year, month, ndvi, evi, lswi, lst_c (NaN = no clear composite) | MODIS MOD13Q1, MOD09A1, MOD11A2 over a paddy mask (scripts/gee_extract.py) |
| `reservoir_annual.csv` | basin, year, storage_pct (at sowing) | RID reservoir storage |
| `prices_annual.csv` | year, rice, cassava, sugarcane, maize, urea (THB/kg farm-gate; include the two years before the first season) | OAE farm-gate prices; OCSB cane price |
| `yields.csv` | district_id, year, yield_kg_rai | OAE / DOAE statistics |
| `truth_components.csv` | synthetic only: true additive yield components | - |
Units: yield in kg/rai (x 6.25 = kg/ha). Basins in `districts.csv` must match `reservoir_annual.csv`.
