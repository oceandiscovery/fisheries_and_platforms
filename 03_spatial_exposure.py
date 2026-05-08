"""
03_spatial_exposure.py
======================
Calculates spatial exposure variables for every landing point:
  • Distance to nearest offshore oil/gas platform
  • Distance to nearest MPA (RDS + APA combined)
  • Inside / outside MPA status
  • Distance to MPA boundary
  • Municipality assignment (spatial join)

Also assigns platform-exposure and MPA-exposure classes, and builds
combined platform × MPA exposure groups.

All distance calculations are performed in the projected CRS (EPSG:31984,
SIRGAS 2000 / UTM zone 24S, metres).

MPA layers are delivered as ordered point sequences (polygon vertices).
The polygon is reconstructed by sorting on the numeric part of the point
label before creating the Shapely Polygon.

For the 14 locals that have no explicit landing-point coordinates, the
centroid of the corresponding RN municipality polygon is used as a fallback.
These rows are tagged coord_source='municipality_centroid'.

Outputs (data/interim/):
  - landing_points_exposure.csv   Landing points with all exposure variables
  - local_exposure.csv            Local-level aggregated exposure
  - xwalk_local_updated.csv       xwalk_local.csv with municipality filled in
  - platforms_projected.gpkg      Platforms reprojected (diagnostic)
  - mpas_combined.gpkg            Both MPAs merged and reprojected (diagnostic)

Run:
  python 03_spatial_exposure.py
"""

import logging
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_00 as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT,
    handlers=[
        logging.FileHandler(cfg.LOGS / "03_spatial_exposure.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("03_spatial_exposure")
warnings.filterwarnings("ignore", category=UserWarning)

# Locals whose name differs from the official RN municipality name
LOCAL_TO_MUN_OVERRIDE = {
    "ARES":        "Arez",
    "CEARA MIRIM": "Ceará-Mirim",
}


def _check(path: Path) -> None:
    if not path.exists():
        log.error("Missing input: %s", path)
        sys.exit(1)


def _save_csv(df: pd.DataFrame, name: str) -> None:
    out = cfg.DATA_INTERIM / name
    df.to_csv(out, index=False)
    log.info("Saved %s  (%d rows)", out.name, len(df))


def _save_gpkg(gdf: gpd.GeoDataFrame, name: str) -> None:
    out = cfg.DATA_INTERIM / name
    gdf.to_file(out, driver="GPKG")
    log.info("Saved %s  (%d features)", out.name, len(gdf))


# ─── Load spatial layers ──────────────────────────────────────────────────────

def load_layer(key: str, label: str, prefer_pattern: str = "") -> gpd.GeoDataFrame:
    """
    Read a spatial layer from a zip archive.
    If prefer_pattern is given (e.g. '*polygon*'), files whose name matches
    the pattern are tried first; remaining files are tried as fallback.
    """
    import fnmatch
    import tempfile
    import zipfile as zf_mod

    path = cfg.FILES[key]
    _check(path)

    with tempfile.TemporaryDirectory() as tmpdir:
        with zf_mod.ZipFile(path) as zf:
            zf.extractall(tmpdir)

        tmppath = Path(tmpdir)
        candidates: list[Path] = []
        for ext in [".shp", ".geojson", ".gpkg", ".json"]:
            candidates += list(tmppath.rglob(f"*{ext}"))

        if prefer_pattern:
            preferred = [c for c in candidates if fnmatch.fnmatch(c.name, prefer_pattern)]
            others    = [c for c in candidates if c not in preferred]
            candidates = preferred + others

        for c in candidates:
            gdf = gpd.read_file(c)
            log.info(
                "Loaded %s from %s: %d features, CRS=%s",
                label, c.name, len(gdf), gdf.crs,
            )
            return gdf

    raise FileNotFoundError(f"No readable spatial file found in zip: {path}")


def load_platforms() -> gpd.GeoDataFrame:
    gdf = load_layer("platforms", "platforms")
    gdf = gdf.to_crs(cfg.CRS_PROJ)
    log.info("Platforms reprojected to %s", cfg.CRS_PROJ)
    return gdf


def load_mpas() -> gpd.GeoDataFrame:
    """
    Load the two MPA polygon layers, preferring the *polygon* shapefile
    in each zip (both zips also contain a *points* shapefile).
    """
    rds = load_layer("rds", "RDS Ponta do Tubarão", prefer_pattern="*polygon*")
    apa = load_layer("apa", "APA Dunas do Rosado",  prefer_pattern="*polygon*")

    rds["mpa_name"] = cfg.MPA_RDS
    apa["mpa_name"] = cfg.MPA_APA

    if rds.crs != apa.crs:
        apa = apa.to_crs(rds.crs)

    combined = gpd.GeoDataFrame(
        pd.concat([rds[["mpa_name", "geometry"]], apa[["mpa_name", "geometry"]]],
                  ignore_index=True),
        crs=rds.crs,
    )
    combined = combined.to_crs(cfg.CRS_PROJ)
    log.info(
        "MPAs loaded as polygons and reprojected to %s: %d features",
        cfg.CRS_PROJ, len(combined),
    )
    return combined


def load_municipalities() -> gpd.GeoDataFrame:
    gdf = load_layer("municipalities", "RN municipalities")
    gdf = gdf.to_crs(cfg.CRS_PROJ)
    return gdf


# ─── Landing points as GeoDataFrame ──────────────────────────────────────────

def landing_points_to_gdf(lp: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        lp.copy(),
        geometry=gpd.points_from_xy(lp["longitude"], lp["latitude"]),
        crs=cfg.CRS_GEO,
    )
    gdf = gdf.to_crs(cfg.CRS_PROJ)
    gdf["coord_source"] = "exact"
    log.info("Landing points GeoDataFrame: %d points, CRS=%s", len(gdf), gdf.crs)
    return gdf


def _norm(s: str) -> str:
    """Strip accents, lower-case, collapse hyphens/spaces."""
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[-\s]+", "", s)


def expand_with_municipality_centroids(
    lp_gdf: gpd.GeoDataFrame,
    xwalk_local: pd.DataFrame,
    munic_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    For every local that has no landing-point entry, add a synthetic point
    at the centroid of its RN municipality polygon.
    New rows are tagged coord_source='municipality_centroid'.
    """
    covered   = set(lp_gdf["local"].str.upper())
    all_local = set(xwalk_local["local"].str.upper())
    missing   = sorted(all_local - covered)

    if not missing:
        log.info("All locals already have landing-point coordinates.")
        return lp_gdf

    log.info(
        "Adding municipality-centroid fallbacks for %d locals: %s",
        len(missing), missing,
    )

    mun_col = next(
        (c for c in munic_gdf.columns if "NM_MUN" in c.upper()),
        munic_gdf.columns[0],
    )
    munic_gdf = munic_gdf.copy()
    munic_gdf["_norm"] = munic_gdf[mun_col].apply(_norm)

    new_rows = []
    for local in missing:
        override = LOCAL_TO_MUN_OVERRIDE.get(local.upper())
        if override:
            mask = munic_gdf[mun_col].str.lower() == override.lower()
        else:
            mask = munic_gdf["_norm"] == _norm(local)

        match = munic_gdf[mask]
        if match.empty:
            log.warning(
                "No municipality match for local '%s' — skipping centroid fallback",
                local,
            )
            continue

        centroid = match.iloc[0].geometry.centroid
        new_rows.append({
            "local":              local,
            "landing_point_mean": f"{local.title()}_centroid",
            "coord_source":       "municipality_centroid",
            "geometry":           centroid,
        })

    if not new_rows:
        return lp_gdf

    new_gdf = gpd.GeoDataFrame(new_rows, crs=cfg.CRS_PROJ)
    combined = gpd.GeoDataFrame(
        pd.concat([lp_gdf, new_gdf], ignore_index=True),
        crs=cfg.CRS_PROJ,
    )
    log.info(
        "Landing points after expansion: %d exact + %d centroid = %d total",
        len(lp_gdf), len(new_gdf), len(combined),
    )
    return combined


# ─── Distance calculations ────────────────────────────────────────────────────

def distance_to_nearest(
    points_gdf: gpd.GeoDataFrame,
    targets_gdf: gpd.GeoDataFrame,
    col_prefix: str,
) -> gpd.GeoDataFrame:
    """
    Distance (m / km) from each point to the nearest geometry in targets_gdf.
    For polygon targets, distance is 0 for points inside the polygon.
    """
    assert points_gdf.crs == targets_gdf.crs, "CRS mismatch"
    targets_union = targets_gdf.geometry.union_all()
    dists = points_gdf.geometry.apply(lambda geom: geom.distance(targets_union))
    points_gdf = points_gdf.copy()
    points_gdf[f"{col_prefix}_dist_m"]  = dists
    points_gdf[f"{col_prefix}_dist_km"] = dists / 1_000
    return points_gdf


def inside_mpa(
    points_gdf: gpd.GeoDataFrame,
    mpas_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Determine whether each landing point falls inside any MPA polygon."""
    joined = gpd.sjoin(
        points_gdf[["geometry"]].reset_index(),
        mpas_gdf[["mpa_name", "geometry"]],
        how="left",
        predicate="within",
    ).set_index("index")

    points_gdf = points_gdf.copy()
    points_gdf["inside_mpa_name"] = joined["mpa_name"]
    points_gdf["inside_any_mpa"]  = points_gdf["inside_mpa_name"].notna()
    return points_gdf


def distance_to_mpa_boundary(
    points_gdf: gpd.GeoDataFrame,
    mpas_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Distance from each point to the nearest MPA boundary ring.
    Points inside an MPA receive a negative value (depth inside polygon).
    """
    boundary = mpas_gdf.geometry.boundary.union_all()
    dists    = points_gdf.geometry.apply(lambda g: g.distance(boundary))
    sign     = points_gdf["inside_any_mpa"].map({True: -1, False: 1})
    points_gdf = points_gdf.copy()
    points_gdf["mpa_boundary_dist_m"]  = dists * sign
    points_gdf["mpa_boundary_dist_km"] = points_gdf["mpa_boundary_dist_m"] / 1_000
    return points_gdf


# ─── Municipality spatial join ────────────────────────────────────────────────

def assign_municipality(
    points_gdf: gpd.GeoDataFrame,
    munic_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assign NM_MUN to each landing point via spatial join.
    Falls back to nearest polygon for points outside all municipalities.
    Centroid-fallback rows already sit inside a municipality polygon, so
    this call simply confirms / labels them.
    """
    mun_col = next(
        (c for c in munic_gdf.columns
         if "NM_MUN" in c.upper() or "NOME" in c.upper() or "NAME" in c.upper()),
        munic_gdf.columns[0],
    )

    within = gpd.sjoin(
        points_gdf.reset_index(),
        munic_gdf[[mun_col, "geometry"]],
        how="left",
        predicate="within",
    ).set_index("index")

    points_gdf = points_gdf.copy()
    points_gdf["municipality"] = within[mun_col]

    unmatched = points_gdf["municipality"].isna()
    if unmatched.any():
        for idx in points_gdf[unmatched].index:
            pt          = points_gdf.loc[idx, "geometry"]
            nearest_idx = munic_gdf.distance(pt).idxmin()
            points_gdf.loc[idx, "municipality"] = munic_gdf.loc[nearest_idx, mun_col]
        log.info(
            "Municipality assigned by nearest-polygon for %d points", unmatched.sum()
        )

    return points_gdf


# ─── Exposure classification ──────────────────────────────────────────────────

def classify_platform_exposure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["platform_exposure_class"] = pd.cut(
        df["platform_dist_km"],
        bins=cfg.PLATFORM_BREAKS_KM,
        labels=cfg.PLATFORM_LABELS,
        right=False,
    )
    return df


def classify_mpa_exposure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat = pd.cut(
        df["mpa_dist_km"],
        bins=cfg.MPA_BREAKS_KM,
        labels=cfg.MPA_LABELS,
        right=False,
    ).astype(str)
    df["mpa_exposure_class"] = np.where(df["inside_any_mpa"], "inside", cat)
    return df


def build_combined_exposure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combined_exposure_class"] = (
        df["platform_exposure_class"].astype(str)
        + " × "
        + df["mpa_exposure_class"].astype(str)
    )
    return df


# ─── Local-level spatial summary ─────────────────────────────────────────────

def aggregate_to_local(points_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Aggregate landing-point exposure values to local level.
    Uses mean distance; modal class for categoricals.
    coord_source is 'exact' when all points for that local are exact,
    otherwise 'municipality_centroid'.
    """
    df  = pd.DataFrame(points_gdf.drop(columns="geometry"))
    grp = df.groupby("local")

    def modal(s):
        m = s.mode()
        return m.iloc[0] if len(m) > 0 else pd.NA

    agg = grp.agg(
        platform_dist_km_mean     =("platform_dist_km",       "mean"),
        platform_dist_km_min      =("platform_dist_km",       "min"),
        mpa_dist_km_mean          =("mpa_dist_km",            "mean"),
        mpa_dist_km_min           =("mpa_dist_km",            "min"),
        mpa_boundary_dist_km_mean =("mpa_boundary_dist_km",   "mean"),
        inside_any_mpa_any        =("inside_any_mpa",         "any"),
        n_landing_points          =("landing_point_mean",     "count"),
        municipality              =("municipality",           modal),
        coord_source              =("coord_source",           modal),
    )

    agg["platform_exposure_class"] = grp["platform_exposure_class"].agg(modal)
    agg["mpa_exposure_class"]      = grp["mpa_exposure_class"].agg(modal)
    agg["combined_exposure_class"] = grp["combined_exposure_class"].agg(modal)
    return agg.reset_index()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg.setup_dirs()
    _check(cfg.DATA_INTERIM / "landing_points_clean.csv")
    _check(cfg.DATA_INTERIM / "xwalk_local.csv")

    lp          = pd.read_csv(cfg.DATA_INTERIM / "landing_points_clean.csv")
    xwalk_local = pd.read_csv(cfg.DATA_INTERIM / "xwalk_local.csv")

    platforms = load_platforms()
    mpas      = load_mpas()
    munic     = load_municipalities()

    lp_gdf = landing_points_to_gdf(lp)
    lp_gdf = expand_with_municipality_centroids(lp_gdf, xwalk_local, munic)

    log.info("Computing distances to platforms …")
    lp_gdf = distance_to_nearest(lp_gdf, platforms, "platform")

    log.info("Computing MPA inside/outside …")
    lp_gdf = inside_mpa(lp_gdf, mpas)

    log.info("Computing distances to MPA …")
    lp_gdf = distance_to_nearest(lp_gdf, mpas, "mpa")
    lp_gdf = distance_to_mpa_boundary(lp_gdf, mpas)

    log.info("Assigning municipalities …")
    lp_gdf = assign_municipality(lp_gdf, munic)

    lp_gdf = classify_platform_exposure(lp_gdf)
    lp_gdf = classify_mpa_exposure(lp_gdf)
    lp_gdf = build_combined_exposure(lp_gdf)

    # Backfill latitude/longitude from geometry for centroid-fallback rows
    geo = lp_gdf.to_crs(cfg.CRS_GEO)
    missing_ll = lp_gdf["latitude"].isna() | lp_gdf["longitude"].isna() \
                 if "latitude" in lp_gdf.columns else pd.Series(True, index=lp_gdf.index)
    lp_gdf = lp_gdf.copy()
    if missing_ll.any():
        lp_gdf.loc[missing_ll, "latitude"]  = geo.loc[missing_ll, "geometry"].y
        lp_gdf.loc[missing_ll, "longitude"] = geo.loc[missing_ll, "geometry"].x

    lp_exposure = pd.DataFrame(lp_gdf.drop(columns="geometry"))
    _save_csv(lp_exposure, "landing_points_exposure.csv")

    log.info("Aggregating exposure to local level …")
    local_exposure = aggregate_to_local(lp_gdf)
    _save_csv(local_exposure, "local_exposure.csv")

    xwalk_updated = xwalk_local.merge(
        local_exposure[["local", "municipality"]],
        on="local", how="left", suffixes=("_old", ""),
    )
    if "municipality_old" in xwalk_updated.columns:
        xwalk_updated = xwalk_updated.drop(columns=["municipality_old"])
    _save_csv(xwalk_updated, "xwalk_local_updated.csv")

    _save_gpkg(platforms, "platforms_projected.gpkg")
    _save_gpkg(mpas,      "mpas_combined.gpkg")

    log.info("03_spatial_exposure.py complete.")


if __name__ == "__main__":
    main()
