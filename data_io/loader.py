"""
Workspace loader. Tries, in order:
  1. CSV + GeoPackage (spatial join of PS points to building polygons)
  2. Exported Parquet / Pickle in ./insar_export/
  3. Synthetic demo data
"""

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import re

from analysis.temporal import TemporalRateFitter
from app_config import EXPORT_DIR, HEIGHT_INFO_PATH
from data_io.summary import summarize_building_fast
from data_io.synthetic import generate_synthetic_workspace
from data_io.temperature import load_temperature_series

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    gpd = None
    HAS_GEOPANDAS = False


def _normalize_bag_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    return digits


def _parse_height(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"-?\d+(?:[.,]\d+)?", str(value).replace(",", "."))
    return float(match.group(0)) if match else None


def _match_overpass_building_heights(
    gdf_poly: "gpd.GeoDataFrame",
) -> dict[str, float]:
    """Match Overpass max heights to GPKG polygons by BAG ID, then geometry."""
    if not HEIGHT_INFO_PATH.exists():
        return {}

    gdf_height = gpd.read_file(HEIGHT_INFO_PATH)
    if gdf_height.empty:
        return {}
    if gdf_height.crs is not None:
        gdf_height = gdf_height.to_crs(gdf_poly.crs)

    max_height_by_key: dict[str, float] = {}

    bag_id_col = next(
        (c for c in gdf_poly.columns if c.lower() in {"identificatie", "pand_id", "bag_id", "bldg_id"}),
        None,
    )
    poly_bag: dict[str, str] = {}
    if bag_id_col is not None:
        poly_bag = {
            _normalize_bag_id(value): key
            for value, key in zip(gdf_poly[bag_id_col], gdf_poly["poly_key"])
            if _normalize_bag_id(value)
        }
        for _, row in gdf_height.iterrows():
            bag_id = row["ref:bag"] if "ref:bag" in gdf_height.columns else None
            height = _parse_height(row["height"]) if "height" in gdf_height.columns else None
            poly_key = poly_bag.get(_normalize_bag_id(bag_id))
            if poly_key is not None and height is not None:
                max_height_by_key[poly_key] = height

    unmatched_height = []
    for _, row in gdf_height.iterrows():
        bag_id = row["ref:bag"] if "ref:bag" in gdf_height.columns else None
        height = _parse_height(row["height"]) if "height" in gdf_height.columns else None
        poly_key = poly_bag.get(_normalize_bag_id(bag_id))
        if height is not None and poly_key not in max_height_by_key:
            unmatched_height.append(row)

    if unmatched_height:
        height_candidates = gpd.GeoDataFrame(
            [
                {
                    "height": _parse_height(row["height"]) if "height" in gdf_height.columns else None,
                    "geometry": row.geometry,
                }
                for row in unmatched_height
                if row.geometry is not None and not row.geometry.is_empty
                and (_parse_height(row["height"]) if "height" in gdf_height.columns else None) is not None
            ],
            geometry="geometry",
            crs=gdf_height.crs,
        )
        if not height_candidates.empty:
            spatial = gpd.sjoin(
                height_candidates,
                gdf_poly[["poly_key", "geometry"]],
                how="inner",
                predicate="intersects",
            )
            for _, match in spatial.iterrows():
                poly_key = match["poly_key"]
                if poly_key not in max_height_by_key:
                    max_height_by_key[poly_key] = float(match["height"])

    return max_height_by_key


@st.cache_data(show_spinner="Filtering CSV & joining with building polygons (GeoPackage)...")
def load_csv_and_gpkg_workspace(
    csv_path_str: str,
    gpkg_path_str: str,
    filter_height_wrt_dem: bool,
    min_h_dem: float,
    max_h_dem: float,
    filter_coherence: bool,
    min_coh: float,
    max_coh: float,
    min_pts_per_building: int,
) -> tuple[dict[int, dict], pd.DataFrame, pd.DatetimeIndex, np.ndarray, str, bool]:
    csv_path = Path(csv_path_str)
    gpkg_path = Path(gpkg_path_str)

    if csv_path.exists() and gpkg_path.exists() and HAS_GEOPANDAS:
        data = pd.read_csv(csv_path, dtype={"pnt_flags": str})

        d_cols = sorted([c for c in data.columns if c.startswith("d_")])
        if not d_cols:
            raise ValueError(f"No `d_YYYYMMDD` displacement columns found in {csv_path}.")
        dates_dt = pd.to_datetime([c[2:] for c in d_cols], format="%Y%m%d")

        # Ground-relative height is physical and must always be positive for a
        # PS point. Apply this independently of the optional DEM range filter.
        if "pnt_height" in data.columns and "pnt_demheight_highres" in data.columns:
            data = data.copy()
            data["h_above_dem"] = data["pnt_height"] - data["pnt_demheight_highres"]
        elif "h_above_dem" not in data.columns:
            raise ValueError("Missing `pnt_height` / `pnt_demheight_highres` columns for DEM height calculation.")
        data = data[np.isfinite(data["h_above_dem"]) & (data["h_above_dem"] > 0.0)]

        if filter_height_wrt_dem:
            data = data[(data["h_above_dem"] >= min_h_dem) & (data["h_above_dem"] <= max_h_dem)]

        if filter_coherence:
            if "pnt_enscoh" in data.columns:
                data = data[(data["pnt_enscoh"] >= min_coh) & (data["pnt_enscoh"] <= max_coh)]
            if "pnt_quality" in data.columns:
                data = data[(data["pnt_quality"] >= min_coh) & (data["pnt_quality"] <= max_coh)]

        if data.empty:
            raise ValueError("0 PS points remained after applying Height WRT DEM and Coherence filters.")

        gdf_poly = gpd.read_file(gpkg_path)
        if gdf_poly.crs is not None and gdf_poly.crs.to_epsg() != 4326:
            gdf_poly = gdf_poly.to_crs("EPSG:4326")
        elif gdf_poly.crs is None:
            gdf_poly = gdf_poly.set_crs("EPSG:4326")

        id_candidates = [
            "identificatie", "pand_id", "building_id", "bldg_id",
            "gml_id", "id", "fid", "objectid", "name",
        ]
        poly_cols_lower = {c.lower(): c for c in gdf_poly.columns}
        bld_id_col = next((poly_cols_lower[c] for c in id_candidates if c in poly_cols_lower), None)
        year_col = next((poly_cols_lower[c] for c in ["bouwjaar", "construction_year", "year"] if c in poly_cols_lower), None)

        gdf_poly = gdf_poly.reset_index(drop=True)
        gdf_poly["poly_key"] = (
            gdf_poly[bld_id_col].astype(str)
            if bld_id_col is not None
            else [f"BLD_{i:04d}" for i in range(len(gdf_poly))]
        )
        max_height_by_poly = _match_overpass_building_heights(gdf_poly)

        # Simplified GeoJSON geometry per building for crisp 2D footprint rendering
        poly_geom_lookup: dict[str, dict] = {}
        for p_key, geom in zip(gdf_poly["poly_key"], gdf_poly["geometry"]):
            if geom is not None and not geom.is_empty:
                simp = geom.simplify(1e-6, preserve_topology=True)
                poly_geom_lookup[p_key] = simp.__geo_interface__

        keep_poly_cols = ["poly_key", "geometry"] + ([year_col] if year_col else [])
        gdf_pts = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(data["pnt_lon"], data["pnt_lat"]),
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(gdf_pts, gdf_poly[keep_poly_cols], how="inner", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]

        filtered_groups: list[pd.DataFrame] = []
        for poly_key, grp in joined.groupby("poly_key", sort=False):
            poly_key = str(grp["poly_key"].iloc[0])
            max_height = max_height_by_poly.get(poly_key)
            if max_height is None:
                filtered_groups.append(grp)
            else:
                filtered_groups.append(grp[grp["h_above_dem"] <= max_height])
        joined = (
            pd.concat(filtered_groups, axis=0)
            if filtered_groups
            else joined.iloc[0:0]
        )

        temp_c, temp_is_measured = load_temperature_series(dates_dt)
        fitter = TemporalRateFitter(dates_dt, trend_order=1, seasonal=True)

        valid_groups = [
            (poly_key, grp)
            for poly_key, grp in joined.groupby("poly_key", sort=False)
            if len(grp) >= min_pts_per_building
        ]

        if not valid_groups:
            raise ValueError(
                f"No buildings inside `{gpkg_path.name}` had >= {min_pts_per_building} filtered PS points."
            )

        building_datasets: dict[int, dict] = {}
        summary_rows: list[dict] = []

        # Sequential integer building IDs from 1 to max
        for bld_num, (poly_key, grp) in enumerate(valid_groups, start=1):
            disp_mm = grp[d_cols].to_numpy(dtype=float) * 1e3
            bdata: dict[str, Any] = dict(
                displacement=disp_mm,
                h_above_dem=grp["h_above_dem"].to_numpy(dtype=float),
                lon=grp["pnt_lon"].to_numpy(dtype=float),
                lat=grp["pnt_lat"].to_numpy(dtype=float),
                polygon_geojson=poly_geom_lookup.get(poly_key),
            )
            max_height = max_height_by_poly.get(poly_key)
            if max_height is not None:
                bdata["max_height_m"] = max_height
            if "pnt_enscoh" in grp.columns:
                bdata["pnt_enscoh"] = grp["pnt_enscoh"].to_numpy(dtype=float)
            if "pnt_incidangle" in grp.columns:
                bdata["pnt_incidangle"] = grp["pnt_incidangle"].to_numpy(dtype=float)
            if year_col is not None:
                yr_vals = grp[year_col].dropna()
                bdata["construction_year"] = int(yr_vals.iloc[0]) if len(yr_vals) > 0 else None

            building_datasets[bld_num] = bdata
            summary_rows.append(summarize_building_fast(bld_num, bdata, dates_dt, temp_c, fitter))

        intra_df = pd.DataFrame(summary_rows)
        source_label = f"CSV + GeoPackage (`{csv_path.name}` + `{gpkg_path.name}`)"
        return building_datasets, intra_df, dates_dt, temp_c, source_label, temp_is_measured

    df_path = EXPORT_DIR / "intra_df.parquet"
    ds_path = EXPORT_DIR / "building_datasets.pkl"
    dt_path = EXPORT_DIR / "dates.npy"
    if df_path.exists() and ds_path.exists() and dt_path.exists():
        raw_df = pd.read_parquet(df_path)
        with open(ds_path, "rb") as f:
            raw_datasets = pickle.load(f)
        dates_dt = pd.DatetimeIndex(np.load(dt_path, allow_pickle=True))
        temp_c, temp_is_measured = load_temperature_series(dates_dt)

        rename_map = {"dh_theilsen_mm_yr_per_m": "dh_theilsen", "therm_grad_mm_per_m": "therm_grad"}
        raw_df = raw_df.rename(columns={k: v for k, v in rename_map.items() if k in raw_df.columns})

        building_datasets = {}
        new_rows = []
        for bld_num, row in enumerate(raw_df.itertuples(index=False), start=1):
            old_id = row.building
            bdata = raw_datasets[old_id]
            h_arr = np.asarray(bdata["h_above_dem"], dtype=float)
            keep = h_arr > 0.0
            if not np.all(keep):
                bdata = bdata.copy()
                bdata["h_above_dem"] = h_arr[keep]
                bdata["displacement"] = np.asarray(bdata["displacement"], dtype=float)[keep]
                bdata["lon"] = np.asarray(bdata["lon"], dtype=float)[keep]
                bdata["lat"] = np.asarray(bdata["lat"], dtype=float)[keep]
                for key in ("pnt_enscoh", "pnt_incidangle", "coherence", "enscoh"):
                    if key in bdata:
                        bdata[key] = np.asarray(bdata[key], dtype=float)[keep]
            building_datasets[bld_num] = bdata
            r_dict = row._asdict()
            r_dict["building"] = bld_num
            h_arr = np.asarray(bdata["h_above_dem"], dtype=float)
            r_dict.setdefault("h_min_m", float(np.nanmin(h_arr)))
            r_dict.setdefault("h_max_m", float(np.nanmax(h_arr)))
            r_dict.setdefault("lat_center", float(np.nanmean(bdata["lat"])))
            r_dict.setdefault("lon_center", float(np.nanmean(bdata["lon"])))
            new_rows.append(r_dict)

        intra_df = pd.DataFrame(new_rows)
        return building_datasets, intra_df, dates_dt, temp_c, "Exported Parquet/Pickle (`./insar_export/`)", temp_is_measured

    datasets, intra_df, dates_dt, temp_c, temp_is_measured = generate_synthetic_workspace()
    return datasets, intra_df, dates_dt, temp_c, "Synthetic Demo Dataset", temp_is_measured
