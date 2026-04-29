"""
data_pipeline.py — Load dashboard-ready data from `data_processed/`.

This module prioritises parquet outputs produced by the analysis pipeline
(especially scripts 04, 05 and 06) and only derives lightweight secondary
products needed by the dashboard views.
"""

from __future__ import annotations

import json
import os
import warnings
import numpy as np
import pandas as pd
from shapely.geometry import Point
from utils.coords import PORT_COORDS, PORT_META

warnings.filterwarnings("ignore")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data_processed"))


def _p(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.parquet")


def _read(name: str) -> pd.DataFrame:
    path = _p(name)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def _title_to_norm(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.upper()
        .str.replace("Á", "A")
        .str.replace("É", "E")
        .str.replace("Í", "I")
        .str.replace("Ó", "O")
        .str.replace("Ú", "U")
        .str.replace("Ç", "C")
    )


def _ensure_local_norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "local_norm" not in out.columns:
        if "municipality_context_norm" in out.columns:
            out["local_norm"] = out["municipality_context_norm"]
        elif "locality_norm" in out.columns:
            out["local_norm"] = out["locality_norm"]
        elif "municipality_norm" in out.columns:
            out["local_norm"] = out["municipality_norm"]
        elif "local_canonical" in out.columns:
            out["local_norm"] = _title_to_norm(out["local_canonical"])
    return out


def _prepare_species(species_processed: pd.DataFrame, species_raw: pd.DataFrame) -> pd.DataFrame:
    if not species_processed.empty:
        out = _ensure_local_norm(species_processed)
        if "species" not in out.columns and "species_canonical" in out.columns:
            out["species"] = out["species_canonical"]
        keep = [
            c for c in [
                "local_norm", "local_canonical", "year", "species", "species_canonical",
                "sp_production_ton", "production_ton", "production_per_trip_ton",
                "production_per_fisher_ton", "species_richness", "shannon_species",
                "pielou_species",
            ] if c in out.columns
        ]
        return out[keep].copy()

    if species_raw.empty:
        return pd.DataFrame()

    out = _ensure_local_norm(species_raw)
    if "species" not in out.columns and "species_canonical" in out.columns:
        out["species"] = out["species_canonical"]
    return out


def _prepare_gear(gear_processed: pd.DataFrame, gear_raw: pd.DataFrame) -> pd.DataFrame:
    if not gear_processed.empty:
        out = _ensure_local_norm(gear_processed)
        if "gear_type" not in out.columns and "gear_type_canonical" in out.columns:
            out["gear_type"] = out["gear_type_canonical"]
        if "gear_group" not in out.columns and "gear_group_canonical" in out.columns:
            out["gear_group"] = out["gear_group_canonical"]
        if "gear_group" in out.columns:
            out["gear_group"] = out["gear_group"].astype("string").str.lower()
        keep = [
            c for c in [
                "local_norm", "local_canonical", "year", "gear_type", "gear_group",
                "gear_type_canonical", "gear_group_canonical", "gear_production_ton",
            ] if c in out.columns
        ]
        return out[keep].copy()

    if gear_raw.empty:
        return pd.DataFrame()

    out = _ensure_local_norm(gear_raw)
    if "gear_group" in out.columns:
        out["gear_group"] = out["gear_group"].astype("string").str.lower()
    if "gear_type" not in out.columns and "gear_type_canonical" in out.columns:
        out["gear_type"] = out["gear_type_canonical"]
    return out


def load_all() -> dict[str, pd.DataFrame]:
    analysis = _read("04_analysis_locality_year")
    diversity = _read("06_diversity_table")
    species_processed = _read("06_composition_long")
    gear_processed = _read("04_gear_locality_year")
    species_raw = _read("03_species_landings_canonical")
    gear_raw = _read("03_gear_production_canonical")
    pmdp_raw = _read("03_pmdp_master_canonical")
    prod_value = _read("03_production_value_canonical")
    socioeco = _read("03_socioeconomic_canonical")
    fleet_raw = _read("03_fleet_composition_canonical")

    if not prod_value.empty and "municipality_norm" in prod_value.columns:
        prod_value["local_norm"] = prod_value["municipality_norm"]

    dfs = {
        "analysis": analysis,
        "diversity": diversity,
        "species": _prepare_species(species_processed, species_raw),
        "gear": _prepare_gear(gear_processed, gear_raw),
        "prod_value": prod_value,
        "pmdp": pmdp_raw,
        "socioeco": socioeco,
        "fleet": fleet_raw,
        "species_raw": species_raw,
        "gear_raw": gear_raw,
        "species_processed": species_processed,
        "gear_processed": gear_processed,
    }
    return dfs


def compute_cpue(dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gear = dfs["gear"].copy()
    analysis = _ensure_local_norm(dfs.get("analysis", pd.DataFrame()).copy())
    pmdp = _ensure_local_norm(dfs.get("pmdp", pd.DataFrame()).copy())

    if gear.empty:
        return pd.DataFrame(), pd.DataFrame()

    trips_source = analysis if not analysis.empty else pmdp
    if trips_source.empty or "assisted_trips" not in trips_source.columns:
        return pd.DataFrame(), pd.DataFrame()

    merged = gear.merge(
        trips_source[["local_norm", "year", "assisted_trips"]],
        on=["local_norm", "year"],
        how="left",
    )
    merged["cpue"] = merged["gear_production_ton"] / merged["assisted_trips"].replace(0, np.nan)

    cpue_port = (
        merged.groupby(["local_norm", "year"], dropna=False)
        .agg(
            total_production_ton=("gear_production_ton", "sum"),
            total_trips=("assisted_trips", "mean"),
            cpue=("cpue", "mean"),
        )
        .reset_index()
    )

    group_col = "gear_group" if "gear_group" in merged.columns else "gear_group_canonical"
    type_col = "gear_type" if "gear_type" in merged.columns else "gear_type_canonical"
    cpue_gear = (
        merged.groupby(["local_norm", "year", type_col, group_col], dropna=False)
        .agg(production_ton=("gear_production_ton", "sum"), cpue=("cpue", "mean"))
        .reset_index()
        .rename(columns={type_col: "gear_type", group_col: "gear_group"})
    )
    if "gear_group" in cpue_gear.columns:
        cpue_gear["gear_group"] = cpue_gear["gear_group"].astype("string").str.lower()
    return cpue_port, cpue_gear


def compute_biodiversity(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    div = dfs.get("diversity", pd.DataFrame()).copy()
    if not div.empty:
        out = _ensure_local_norm(div)
        if "shannon_index" not in out.columns and "shannon_species" in out.columns:
            out["shannon_index"] = out["shannon_species"]
        if "pielou_index" not in out.columns and "pielou_species" in out.columns:
            out["pielou_index"] = out["pielou_species"]
        if "sp_production_ton" not in out.columns and "production_ton" in out.columns:
            out["sp_production_ton"] = out["production_ton"]
        keep = [c for c in ["local_norm", "year", "species_richness", "shannon_index", "pielou_index", "sp_production_ton"] if c in out.columns]
        return out[keep].copy()

    sp = dfs.get("species", pd.DataFrame()).copy()
    if sp.empty:
        return pd.DataFrame()

    def _shannon(values: np.ndarray) -> float:
        v = values[values > 0]
        if len(v) == 0:
            return np.nan
        p = v / v.sum()
        return float(-np.sum(p * np.log(p)))

    rows = []
    for (local_norm, year), grp in sp.groupby(["local_norm", "year"], dropna=False):
        values = grp["sp_production_ton"].dropna().to_numpy(dtype=float)
        s = int((values > 0).sum())
        h = _shannon(values)
        j = h / np.log(s) if s > 1 and pd.notna(h) else np.nan
        rows.append({
            "local_norm": local_norm,
            "year": year,
            "species_richness": s,
            "shannon_index": h,
            "pielou_index": j,
            "sp_production_ton": float(np.nansum(values)),
        })
    return pd.DataFrame(rows)


def build_master(dfs: dict[str, pd.DataFrame], cpue_port: pd.DataFrame, biodiv: pd.DataFrame) -> pd.DataFrame:
    analysis = _ensure_local_norm(dfs.get("analysis", pd.DataFrame()).copy())
    if analysis.empty:
        pmdp = _ensure_local_norm(dfs.get("pmdp", pd.DataFrame()).copy())
        if pmdp.empty:
            return pd.DataFrame()
        master = pmdp.copy()
        if "production_per_trip_ton" not in master.columns and "assisted_trips" in master.columns:
            master["production_per_trip_ton"] = master["production_ton"] / master["assisted_trips"].replace(0, np.nan)
    else:
        master = analysis.copy()

    if "total_vessels" not in master.columns and "vessels_monitored_total" in master.columns:
        master["total_vessels"] = master["vessels_monitored_total"]
    if "fleet_production_ton" not in master.columns and "fleet_production_ton_total" in master.columns:
        master["fleet_production_ton"] = master["fleet_production_ton_total"]
    if "cpue" not in master.columns and "production_per_trip_ton" in master.columns:
        master["cpue"] = master["production_per_trip_ton"]
    if "shannon_index" not in master.columns and "shannon_species" in master.columns:
        master["shannon_index"] = master["shannon_species"]
    if "pielou_index" not in master.columns and "pielou_species" in master.columns:
        master["pielou_index"] = master["pielou_species"]

    if not cpue_port.empty:
        for col in ["cpue", "total_production_ton"]:
            if col in master.columns:
                master = master.drop(columns=[col])
        master = master.merge(cpue_port[["local_norm", "year", "cpue", "total_production_ton"]], on=["local_norm", "year"], how="left")

    if not biodiv.empty:
        bio_cols = ["local_norm", "year", "species_richness", "shannon_index", "pielou_index", "sp_production_ton"]
        for col in [c for c in bio_cols if c not in {"local_norm", "year"} and c in master.columns]:
            master = master.drop(columns=[col])
        master = master.merge(biodiv[bio_cols], on=["local_norm", "year"], how="left")

    if "fishermen_per_vessel" not in master.columns:
        socio = _ensure_local_norm(dfs.get("socioeco", pd.DataFrame()).copy())
        if not socio.empty and "fishermen_per_vessel" in socio.columns:
            master = master.merge(socio[["local_norm", "year", "fishermen_per_vessel"]], on=["local_norm", "year"], how="left")

    if "port_name" not in master.columns:
        master["port_name"] = master["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("name", x))
    master["lat"] = master["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lat"))
    master["lon"] = master["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lon"))
    master["region"] = master["local_norm"].map(lambda x: PORT_META.get(x, {}).get("region", ""))
    return master


def to_geodataframe(master: pd.DataFrame) -> pd.DataFrame:
    out = master.copy()
    out["geometry"] = [Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None for lat, lon in zip(out.get("lat", []), out.get("lon", []))]
    return out


def _df_to_geojson(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon", exclude_cols=None) -> dict:
    exclude_cols = set(exclude_cols or [])
    features = []
    for _, row in df.iterrows():
        lat, lon = row.get(lat_col), row.get(lon_col)
        if pd.isna(lat) or pd.isna(lon):
            continue
        props = {
            k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
            for k, v in row.items()
            if k not in exclude_cols and k not in {lat_col, lon_col}
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def export_geojson(gdf: pd.DataFrame, cpue_gear: pd.DataFrame, dfs: dict[str, pd.DataFrame], output_dir: str = "outputs/geojson") -> None:
    os.makedirs(output_dir, exist_ok=True)

    agg_spec = {
        "lat": "first",
        "lon": "first",
        "port_name": "first",
        "region": "first",
        "cpue": "mean",
        "shannon_index": "mean",
        "species_richness": "mean",
        "estimated_fishermen": "mean",
        "production_ton": "sum",
        "total_vessels": "mean",
    }
    agg_spec = {k: v for k, v in agg_spec.items() if k in gdf.columns}
    ports_avg = gdf.groupby("local_norm", dropna=False).agg(agg_spec).reset_index()
    with open(os.path.join(output_dir, "ports_indicators.geojson"), "w", encoding="utf-8") as f:
        json.dump(_df_to_geojson(ports_avg), f, ensure_ascii=False, indent=2)

    if not cpue_gear.empty:
        cpue_export = cpue_gear.groupby(["local_norm", "gear_type", "gear_group"], dropna=False).agg(cpue=("cpue", "mean"), production_ton=("production_ton", "sum")).reset_index()
        cpue_export["lat"] = cpue_export["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lat"))
        cpue_export["lon"] = cpue_export["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lon"))
        with open(os.path.join(output_dir, "cpue_by_gear.geojson"), "w", encoding="utf-8") as f:
            json.dump(_df_to_geojson(cpue_export), f, ensure_ascii=False, indent=2)

    drop_cols = [c for c in ["geometry"] if c in gdf.columns]
    gdf.drop(columns=drop_cols).to_csv(os.path.join(output_dir, "master_timeseries.csv"), index=False)


def compute_correlations(master: pd.DataFrame):
    from scipy import stats as scipy_stats

    cols = [
        "cpue", "shannon_index", "species_richness", "pielou_index",
        "estimated_fishermen", "fishermen_per_vessel", "total_vessels",
        "production_ton", "fleet_production_ton", "assisted_trips",
    ]
    use_cols = [c for c in cols if c in master.columns]
    sub = master[use_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pearson = sub.corr(method="pearson")
    spearman = sub.corr(method="spearman")
    pairs = []
    for i, c1 in enumerate(pearson.columns):
        for j, c2 in enumerate(pearson.columns):
            if j <= i:
                continue
            r = pearson.loc[c1, c2]
            rs = spearman.loc[c1, c2]
            pair_df = sub[[c1, c2]].dropna()
            n = len(pair_df)
            if n > 3:
                _, p = scipy_stats.pearsonr(pair_df[c1], pair_df[c2])
            else:
                p = np.nan
            pairs.append({
                "var1": c1,
                "var2": c2,
                "pearson_r": round(r, 4),
                "spearman_r": round(rs, 4),
                "p_value": round(p, 4) if pd.notna(p) else None,
                "significant": bool(pd.notna(p) and p < 0.05),
            })
    top_pairs = pd.DataFrame(pairs).sort_values("pearson_r", key=np.abs, ascending=False)
    return pearson, spearman, top_pairs


def build_all(export: bool = True, output_dir: str = "outputs/geojson") -> dict[str, pd.DataFrame]:
    dfs = load_all()
    cpue_port, cpue_gear = compute_cpue(dfs)
    biodiv = compute_biodiversity(dfs)
    master = build_master(dfs, cpue_port, biodiv)
    gdf = to_geodataframe(master)
    pearson, spearman, top_pairs = compute_correlations(master)
    if export:
        export_geojson(gdf, cpue_gear, dfs, output_dir=output_dir)
    return {
        "dfs": dfs,
        "cpue_port": cpue_port,
        "cpue_gear": cpue_gear,
        "biodiv": biodiv,
        "master": master,
        "gdf": gdf,
        "pearson": pearson,
        "spearman": spearman,
        "top_pairs": top_pairs,
    }
