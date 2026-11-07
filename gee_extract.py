"""Google Earth Engine template: real monthly satellite + climate features per district.

NOT executed in the demo (it needs your own Earth Engine account):
    pip install earthengine-api pandas
    earthengine authenticate
    python gee_extract.py --project YOUR_GCP_PROJECT --asset projects/YOUR_GCP_PROJECT/assets/districts

Exports to Google Drive (folder 'crop_yield_gee'):
    gee_monthly_<year>.csv : district_id, year, month, ndvi, evi, lswi, lst_c,
                             rain_mm, tmean_c, tmax_c, rh_pct, soil_moist
    gee_static.csv         : district_id, elevation_m, slope_deg
Download them, then split into the pipeline's schema:
    python gee_extract.py --split gee_monthly_*.csv
    -> data/raw/satellite_monthly.csv and data/raw/climate_monthly.csv

Datasets (check IDs / band names in the Earth Engine catalog before a long run):
    MODIS/061/MOD13Q1  NDVI, EVI (250 m, 16-day)     MODIS/061/MOD09A1  LSWI (500 m, 8-day)
    MODIS/061/MOD11A2  day LST (1 km, 8-day)         UCSB-CHG/CHIRPS/DAILY  rainfall (~5 km)
    ECMWF/ERA5_LAND/DAILY_AGGR  temperature, dew point -> RH, soil moisture (~11 km)
    ESA/WorldCover/v200  cropland mask (class 40)    USGS/SRTMGL1_003  elevation, slope
For rice, a paddy-specific mask (e.g. a national land-use map) is better than generic cropland.
For field-level work use Sentinel-2 (10 m) and Sentinel-1 SAR (sees through monsoon clouds).
"""
import argparse
import glob

MONTHS = [5, 6, 7, 8, 9, 10, 11]
BANDS = ["ndvi", "evi", "lswi", "lst_c", "rain_mm", "tmean_c", "tmax_c", "rh_pct", "soil_moist"]
SCALE = 250  # metres


def monthly_image(ee, year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    def vi(img):  # keep good/marginal pixels (SummaryQA <= 1)
        return img.select(["NDVI", "EVI"]).multiply(0.0001).updateMask(img.select("SummaryQA").lte(1))

    def lswi(img):  # (NIR - SWIR) / (NIR + SWIR); cloud state bits 0-1 == 0
        clear = img.select("StateQA").bitwiseAnd(3).eq(0)
        return (img.normalizedDifference(["sur_refl_b02", "sur_refl_b06"]).rename("lswi").updateMask(clear))

    def lst(img):  # Kelvin*50 -> Celsius, mandatory QA bits 0-1 <= 1
        good = img.select("QC_Day").bitwiseAnd(3).lte(1)
        return img.select("LST_Day_1km").multiply(0.02).subtract(273.15).rename("lst_c").updateMask(good)

    modis_vi = (ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, end).map(vi)
                .max().rename(["ndvi", "evi"]))                    # max-value composite vs clouds
    modis_lswi = ee.ImageCollection("MODIS/061/MOD09A1").filterDate(start, end).map(lswi).median()
    modis_lst = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(start, end).map(lst).mean()
    rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end).sum().rename("rain_mm")

    era = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, end)
    t = era.select("temperature_2m").mean().subtract(273.15)
    td = era.select("dewpoint_temperature_2m").mean().subtract(273.15)
    rh = t.expression(                                             # Magnus formula
        "100 * exp(17.625 * td / (243.04 + td)) / exp(17.625 * t / (243.04 + t))",
        {"t": t, "td": td}).rename("rh_pct")
    tmax = era.select("temperature_2m_max").mean().subtract(273.15).rename("tmax_c")
    sm = era.select("volumetric_soil_water_layer_1").mean().rename("soil_moist")
    return ee.Image.cat([modis_vi, modis_lswi, modis_lst, rain, t.rename("tmean_c"), tmax, rh, sm])


def run(project, asset, start, end):
    import ee
    ee.Initialize(project=project)
    districts = ee.FeatureCollection(asset)        # polygons with a 'district_id' property
    crop = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(40)

    for year in range(start, end + 1):
        fcs = []
        for month in MONTHS:
            img = monthly_image(ee, year, month).updateMask(crop)
            fc = img.reduceRegions(collection=districts, reducer=ee.Reducer.mean(), scale=SCALE, tileScale=4)
            fcs.append(fc.map(lambda f, y=year, m=month: f.set({"year": y, "month": m})))
        task = ee.batch.Export.table.toDrive(
            collection=ee.FeatureCollection(fcs).flatten(), description=f"gee_monthly_{year}",
            folder="crop_yield_gee", fileFormat="CSV", selectors=["district_id", "year", "month", *BANDS])
        task.start()
        print("started", f"gee_monthly_{year}")

    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    terrain = ee.Image.cat([dem.rename("elevation_m"), ee.Terrain.slope(dem).rename("slope_deg")]).updateMask(crop)
    static = terrain.reduceRegions(collection=districts, reducer=ee.Reducer.mean(), scale=90, tileScale=4)
    ee.batch.Export.table.toDrive(collection=static, description="gee_static", folder="crop_yield_gee",
                                  fileFormat="CSV", selectors=["district_id", "elevation_m", "slope_deg"]).start()
    print("started gee_static - check progress at https://code.earthengine.google.com/tasks")


def split(paths, out_dir="data/raw"):
    import pandas as pd
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    key = ["district_id", "year", "month"]
    df[key + ["ndvi", "evi", "lswi", "lst_c"]].to_csv(f"{out_dir}/satellite_monthly.csv", index=False)
    df[key + ["rain_mm", "tmean_c", "tmax_c", "rh_pct", "soil_moist"]].to_csv(
        f"{out_dir}/climate_monthly.csv", index=False)
    print(f"wrote {len(df)} rows to {out_dir}/satellite_monthly.csv and climate_monthly.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project")
    ap.add_argument("--asset", help="FeatureCollection of district polygons with property 'district_id'")
    ap.add_argument("--start", type=int, default=2008)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--split", nargs="+", help="downloaded gee_monthly_*.csv files to convert")
    a = ap.parse_args()
    if a.split:
        split([p for pat in a.split for p in glob.glob(pat)])
    else:
        run(a.project, a.asset, a.start, a.end)
