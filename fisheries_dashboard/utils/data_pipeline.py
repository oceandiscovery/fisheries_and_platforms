"""
data_pipeline.py — Carga, integración y cálculo de métricas derivadas:
  - CPUE por arte y puerto
  - Índices de biodiversidad (Shannon, riqueza)
  - GeoDataFrame con todos los indicadores
  - Exportación GeoJSON para QGIS
"""

import os
import math
import warnings
import json
import numpy as np
import pandas as pd
from shapely.geometry import Point, mapping
from utils.coords import PORT_COORDS, PORT_META

warnings.filterwarnings("ignore")

# Parquets are in the data/ folder at the repo root (3 levels up from utils/ on Streamlit Cloud)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data_processed"))

# ─────────────────────────────────────────────
# 1. CARGA DE DATASETS
# ─────────────────────────────────────────────
def load_all():
    """Carga los 6 datasets canónicos y devuelve un dict de DataFrames."""
    base = DATA_DIR
    dfs = {
        "fleet":      pd.read_parquet(os.path.join(base, "03_fleet_composition_canonical.parquet")),
        "gear":       pd.read_parquet(os.path.join(base, "03_gear_production_canonical.parquet")),
        "pmdp":       pd.read_parquet(os.path.join(base, "03_pmdp_master_canonical.parquet")),
        "prod_value": pd.read_parquet(os.path.join(base, "03_production_value_canonical.parquet")),
        "socioeco":   pd.read_parquet(os.path.join(base, "03_socioeconomic_canonical.parquet")),
        "species":    pd.read_parquet(os.path.join(base, "03_species_landings_canonical.parquet")),
    }
    # Normalizar columna de localidad a local_norm en prod_value
    dfs["prod_value"]["local_norm"] = dfs["prod_value"]["municipality_norm"]
    return dfs


# ─────────────────────────────────────────────
# 2. CPUE (Captura Por Unidad de Esfuerzo)
# ─────────────────────────────────────────────
def compute_cpue(dfs):
    """
    CPUE = producción total (ton) / viajes asistidos
    Calculado por (local, year, gear_type).
    """
    gear = dfs["gear"].copy()
    pmdp = dfs["pmdp"][["local_norm", "year", "assisted_trips"]].copy()

    # Unir gear con viajes
    merged = gear.merge(pmdp, left_on=["local_norm", "year"], right_on=["local_norm", "year"], how="left")
    merged["cpue"] = merged["gear_production_ton"] / merged["assisted_trips"].replace(0, np.nan)
    # CPUE anual por localidad (todas las artes)
    cpue_port = merged.groupby(["local_norm", "year"]).agg(
        total_production_ton=("gear_production_ton", "sum"),
        total_trips=("assisted_trips", "mean"),
        cpue=("cpue", "mean"),
    ).reset_index()
    # CPUE por arte y puerto
    # 03_gear_production_canonical.parquet expone gear_type_canonical / gear_group_canonical.
    # Usamos el nombre canónico si existe, con fallback al nombre bruto del Excel.
    gear_type_col  = "gear_type_canonical"  if "gear_type_canonical"  in merged.columns else "gear_type"
    gear_group_col = "gear_group_canonical" if "gear_group_canonical" in merged.columns else "gear_group"
    cpue_gear = merged.groupby(["local_norm", "year", gear_type_col, gear_group_col]).agg(
        production_ton=("gear_production_ton", "sum"),
        cpue=("cpue", "mean"),
    ).reset_index()
    # Normalizar nombres para que el resto del código vea siempre gear_type / gear_group
    cpue_gear = cpue_gear.rename(columns={gear_type_col: "gear_type", gear_group_col: "gear_group"})
    return cpue_port, cpue_gear


# ─────────────────────────────────────────────
# 3. ÍNDICES DE BIODIVERSIDAD
# ─────────────────────────────────────────────
def shannon_index(counts):
    """Índice de Shannon-Wiener H' a partir de un array de abundancias."""
    counts = counts[counts > 0]
    total = counts.sum()
    proportions = counts / total
    return -np.sum(proportions * np.log(proportions))


def compute_biodiversity(dfs):
    """
    Calcula por (local, year):
      - Riqueza de especies (S)
      - Índice de Shannon-Wiener (H')
      - Índice de Pielou (J' = H'/ln(S))
      - Producción total de especies
    """
    sp = dfs["species"].copy()
    results = []
    for (local, year), grp in sp.groupby(["local_norm", "year"]):
        counts = grp["sp_production_ton"].dropna().values
        # S counts only species with valid (non-NaN, non-zero) production,
        # consistent with how shannon_index filters its input.
        S = int((counts > 0).sum())
        H = shannon_index(counts)
        J = H / math.log(S) if S > 1 else 0.0
        results.append({
            "local_norm": local,
            "year": year,
            "species_richness": S,
            "shannon_index": round(H, 4),
            "pielou_index": round(J, 4),
            "sp_production_ton": grp["sp_production_ton"].sum(),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 4. INTEGRACIÓN MASTER
# ─────────────────────────────────────────────
def build_master(dfs, cpue_port, biodiv):
    """Tabla maestra por (local, year) con todos los indicadores."""
    pmdp  = dfs["pmdp"].copy()
    fleet = dfs["fleet"].groupby(["local_norm", "year"]).agg(
        total_vessels=("vessels_monitored", "sum"),
        fleet_production_ton=("fleet_production_ton", "sum"),
    ).reset_index()
    socio = dfs["socioeco"].copy()

    master = pmdp.merge(fleet, on=["local_norm", "year"], how="left")
    master = master.merge(socio[["local_norm", "year", "fishermen_per_vessel"]], on=["local_norm", "year"], how="left")
    master = master.merge(cpue_port[["local_norm", "year", "cpue", "total_production_ton"]], on=["local_norm", "year"], how="left")
    master = master.merge(biodiv, on=["local_norm", "year"], how="left")

    # Añadir coordenadas
    master["lat"] = master["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lat"))
    master["lon"] = master["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lon"))
    master["region"] = master["local_norm"].map(lambda x: PORT_META.get(x, {}).get("region", ""))
    master["port_name"] = master["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("name", x))
    return master


# ─────────────────────────────────────────────
# 5. GeoDataFrame liviano + EXPORTACIÓN QGIS
# (sin geopandas/fiona — compatible con Streamlit Cloud)
# ─────────────────────────────────────────────
def to_geodataframe(master):
    """Devuelve el master DataFrame con columna geometry de Shapely (sin geopandas)."""
    master = master.copy()
    master["geometry"] = [
        Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
        for lat, lon in zip(master["lat"], master["lon"])
    ]
    return master


def _df_to_geojson(df, lat_col="lat", lon_col="lon", exclude_cols=None):
    """Convierte un DataFrame con columnas lat/lon a dict GeoJSON FeatureCollection."""
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


def export_geojson(gdf, cpue_gear, dfs, output_dir="outputs/geojson"):
    """Exporta capas GeoJSON para uso en QGIS (sin dependencia de fiona/GDAL)."""
    os.makedirs(output_dir, exist_ok=True)

    # Capa 1: puertos con indicadores promedio
    agg_cols = {c: "first" for c in ["lat", "lon", "port_name", "region"]}
    agg_cols.update({"cpue": "mean", "shannon_index": "mean", "species_richness": "mean",
                     "estimated_fishermen": "mean", "production_ton": "sum", "total_vessels": "mean"})
    ports_avg = gdf.groupby("local_norm").agg(
        {k: v for k, v in agg_cols.items() if k in gdf.columns}
    ).reset_index()
    geojson1 = _df_to_geojson(ports_avg)
    with open(os.path.join(output_dir, "ports_indicators.geojson"), "w", encoding="utf-8") as f:
        json.dump(geojson1, f, ensure_ascii=False, indent=2)

    # Capa 2: CPUE por arte de pesca
    cpue_export = cpue_gear.groupby(["local_norm", "gear_type", "gear_group"]).agg(
        cpue=("cpue", "mean"), production_ton=("production_ton", "sum")
    ).reset_index()
    cpue_export["lat"] = cpue_export["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lat"))
    cpue_export["lon"] = cpue_export["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lon"))
    geojson2 = _df_to_geojson(cpue_export)
    with open(os.path.join(output_dir, "cpue_by_gear.geojson"), "w", encoding="utf-8") as f:
        json.dump(geojson2, f, ensure_ascii=False, indent=2)

    # Capa 3: Serie temporal completa (CSV)
    drop_cols = [c for c in ["geometry"] if c in gdf.columns]
    gdf.drop(columns=drop_cols).to_csv(os.path.join(output_dir, "master_timeseries.csv"), index=False)
    print(f"[OK] GeoJSON exportados en: {output_dir}")


# ─────────────────────────────────────────────
# 6. ANÁLISIS ESTADÍSTICO DE CORRELACIONES
# ─────────────────────────────────────────────
def compute_correlations(master):
    """
    Matriz de correlaciones de Pearson y Spearman entre variables clave.
    Devuelve (pearson_df, spearman_df, top_pairs).
    """
    from scipy import stats as scipy_stats

    cols = [
        "cpue", "shannon_index", "species_richness", "pielou_index",
        "estimated_fishermen", "fishermen_per_vessel", "total_vessels",
        "production_ton", "fleet_production_ton", "assisted_trips",
    ]
    sub = master[[c for c in cols if c in master.columns]].dropna()

    pearson  = sub.corr(method="pearson")
    spearman = sub.corr(method="spearman")

    # Top pares significativos
    pairs = []
    for i, c1 in enumerate(pearson.columns):
        for j, c2 in enumerate(pearson.columns):
            if j <= i:
                continue
            r = pearson.loc[c1, c2]
            rs = spearman.loc[c1, c2]
            n = len(sub[[c1, c2]].dropna())
            if n > 3:
                _, p = scipy_stats.pearsonr(sub[c1].dropna(), sub[c2].dropna())
            else:
                p = np.nan
            pairs.append({"var1": c1, "var2": c2, "pearson_r": round(r, 4),
                          "spearman_r": round(rs, 4), "p_value": round(p, 4) if not np.isnan(p) else None,
                          "significant": bool(p < 0.05) if not np.isnan(p) else False})
    top_pairs = pd.DataFrame(pairs).sort_values("pearson_r", key=abs, ascending=False)
    return pearson, spearman, top_pairs


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def build_all(export=True, output_dir="outputs/geojson"):
    """Ejecuta el pipeline completo y devuelve todos los artefactos."""
    dfs       = load_all()
    cpue_port, cpue_gear = compute_cpue(dfs)
    biodiv    = compute_biodiversity(dfs)
    master    = build_master(dfs, cpue_port, biodiv)
    gdf       = to_geodataframe(master)
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
