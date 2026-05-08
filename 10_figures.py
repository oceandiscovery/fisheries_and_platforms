"""
10_figures.py
=============
Produces all maps, plots, and visualisation outputs.

Figure list:
  Maps
    F01  Study area map — municipalities, landing points, platforms, MPAs
    F02  Platform distance gradient map (choropleth over landing points)
    F03  MPA exposure map (inside/outside + buffer zones)

  Productivity & diversity
    F04  CPUE over time by platform exposure class (line plot)
    F05  CPUE over time by MPA exposure class
    F06  CPUE by platform exposure class — boxplot
    F07  Species richness by platform × MPA exposure (heatmap / bar)
    F08  Shannon H' over time by platform exposure class
    F09  CPUE × platform distance scatter (mean ± se per class)

  Gear & boat effort
    F10  Gear-type proportion stacked bars by year
    F11  Gear shares by platform exposure class (grouped bar)
    F12  Gear shares by MPA exposure class
    F13  Boat-type proportion stacked bars by year
    F14  Boat shares across exposure gradients

  Species composition
    F15  Top-20 species shares by year (stacked area)
    F16  Species share by platform exposure class (heatmap)
    F17  SIMPER chart — top species driving platform class differences
    F18  Bray-Curtis turnover between periods (bar)

  Temporal comparison
    F19  Regional production time series with period bands
    F20  CPUE time series: regional vs platform class vs MPA class (multi-panel)
    F21  Period-comparison heatmap (CPUE × platform × MPA class)

Outputs (outputs/figures/):
  PNG files F01–F21 at 300 dpi.

Run:
  python 10_figures.py
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_00 as cfg  # noqa: E402

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False
    warnings.warn("geopandas not available — map figures will be skipped")

logging.basicConfig(
    level=logging.INFO, format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT,
    handlers=[
        logging.FileHandler(cfg.LOGS / "10_figures.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("10_figures")

DPI     = 300
FIGSIZE = (10, 7)

# Ordered exposure classes for consistent axis ordering
PLATFORM_ORDER = cfg.PLATFORM_LABELS
MPA_ORDER      = ["inside"] + cfg.MPA_LABELS


def _fig_path(name: str) -> Path:
    return cfg.FIGURES / name


def _load(name: str) -> pd.DataFrame | None:
    for base in [cfg.DATA_PROCESSED, cfg.DATA_INTERIM]:
        p = base / name
        if p.exists():
            return pd.read_csv(p)
    log.warning("File not found: %s — figure may be skipped", name)
    return None


def _save_fig(fig: plt.Figure, name: str) -> None:
    path = _fig_path(name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", path.name)


# ─── Map figures (require geopandas) ─────────────────────────────────────────

def fig_study_area() -> None:
    if not HAS_GPD:
        return
    try:
        munic     = gpd.read_file(f"zip://{cfg.FILES['municipalities']}").to_crs(cfg.CRS_GEO)
        rds       = gpd.read_file(f"zip://{cfg.FILES['rds']}",  layer="rds_polygon_sirgas2000").to_crs(cfg.CRS_GEO)
        apa       = gpd.read_file(f"zip://{cfg.FILES['apa']}",  layer="apa_dunas_rosado_polygon").to_crs(cfg.CRS_GEO)
        platforms = gpd.read_file(f"zip://{cfg.FILES['platforms']}").to_crs(cfg.CRS_GEO)
        lp_exp    = pd.read_csv(cfg.DATA_INTERIM / "landing_points_exposure.csv")
        valid_ll  = lp_exp["longitude"].notna() & lp_exp["latitude"].notna()
        lp_gdf    = gpd.GeoDataFrame(
            lp_exp[valid_ll],
            geometry=gpd.points_from_xy(lp_exp.loc[valid_ll, "longitude"],
                                         lp_exp.loc[valid_ll, "latitude"]),
            crs=cfg.CRS_GEO,
        )
    except Exception as e:
        log.warning("F01 skipped: %s", e)
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    munic.boundary.plot(ax=ax, color="grey", linewidth=0.5)
    rds.plot(ax=ax, color="#1a9850", alpha=0.3, label=cfg.MPA_RDS)
    apa.plot(ax=ax, color="#66bd63", alpha=0.3, label=cfg.MPA_APA)
    platforms.plot(ax=ax, color="red", marker="^", markersize=40, label="Platforms")
    lp_gdf.plot(ax=ax, color="navy", markersize=15, label="Landing points")
    ax.set_title("Study area: RN municipalities, landing points, platforms, and MPAs")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc="lower left", fontsize=8)
    _save_fig(fig, "F01_study_area.png")


def fig_platform_distance_map() -> None:
    if not HAS_GPD:
        return
    lp_exp = _load("landing_points_exposure.csv")
    if lp_exp is None:
        return
    try:
        munic = gpd.read_file(f"zip://{cfg.FILES['municipalities']}").to_crs(cfg.CRS_GEO)
    except Exception as e:
        log.warning("F02 skipped: %s", e)
        return

    lp_gdf = gpd.GeoDataFrame(
        lp_exp, geometry=gpd.points_from_xy(lp_exp["longitude"], lp_exp["latitude"]),
        crs=cfg.CRS_GEO,
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    munic.boundary.plot(ax=ax, color="grey", linewidth=0.4)
    lp_gdf.plot(ax=ax, column="platform_dist_km", cmap="RdYlBu_r",
                legend=True, markersize=30,
                legend_kwds={"label": "Distance to platform (km)"})
    ax.set_title("Platform distance gradient at landing points")
    _save_fig(fig, "F02_platform_distance_map.png")


def fig_mpa_exposure_map() -> None:
    if not HAS_GPD:
        return
    lp_exp = _load("landing_points_exposure.csv")
    if lp_exp is None:
        return
    try:
        munic = gpd.read_file(f"zip://{cfg.FILES['municipalities']}").to_crs(cfg.CRS_GEO)
        rds   = gpd.read_file(f"zip://{cfg.FILES['rds']}",  layer="rds_polygon_sirgas2000").to_crs(cfg.CRS_GEO)
        apa   = gpd.read_file(f"zip://{cfg.FILES['apa']}",  layer="apa_dunas_rosado_polygon").to_crs(cfg.CRS_GEO)
    except Exception as e:
        log.warning("F03 skipped: %s", e)
        return

    valid = lp_exp["longitude"].notna() & lp_exp["latitude"].notna()
    lp_gdf = gpd.GeoDataFrame(
        lp_exp[valid],
        geometry=gpd.points_from_xy(lp_exp.loc[valid, "longitude"],
                                     lp_exp.loc[valid, "latitude"]),
        crs=cfg.CRS_GEO,
    )
    # Colour by mpa_exposure_class
    colours = {c: v for c, v in cfg.PALETTE_MPA.items()}
    lp_gdf["colour"] = lp_gdf["mpa_exposure_class"].map(colours).fillna("grey")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    munic.boundary.plot(ax=ax, color="grey", linewidth=0.4)
    rds.plot(ax=ax, color="#1a9850", alpha=0.25)
    apa.plot(ax=ax, color="#66bd63", alpha=0.25)
    for cls, grp in lp_gdf.groupby("mpa_exposure_class"):
        colour = colours.get(cls, "grey")
        grp.plot(ax=ax, color=colour, markersize=25, label=cls)
    ax.set_title("MPA exposure classes at landing points")
    ax.legend(loc="lower left", fontsize=8)
    _save_fig(fig, "F03_mpa_exposure_map.png")


# ─── Productivity figures ─────────────────────────────────────────────────────

def fig_cpue_platform_time() -> None:
    df = _load("timeseries_platform.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cls in PLATFORM_ORDER:
        sub = df[df["platform_exposure_class"] == cls].sort_values("year")
        ax.plot(sub["year"], sub["cpue"], marker="o", markersize=3,
                label=cls, color=cfg.PALETTE_PLATFORM.get(cls, "grey"))
    ax.set_xlabel("Year"); ax.set_ylabel("CPUE (ton/trip)")
    ax.set_title("CPUE over time by platform exposure class")
    ax.legend(title="Platform distance", fontsize=8)
    _save_fig(fig, "F04_cpue_platform_time.png")


def fig_cpue_mpa_time() -> None:
    df = _load("timeseries_mpa.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cls in MPA_ORDER:
        sub = df[df["mpa_exposure_class"] == cls].sort_values("year")
        ax.plot(sub["year"], sub["cpue"], marker="o", markersize=3,
                label=cls, color=cfg.PALETTE_MPA.get(cls, "grey"))
    ax.set_xlabel("Year"); ax.set_ylabel("CPUE (ton/trip)")
    ax.set_title("CPUE over time by MPA exposure class")
    ax.legend(title="MPA exposure", fontsize=8)
    _save_fig(fig, "F05_cpue_mpa_time.png")


def fig_cpue_platform_boxplot() -> None:
    df = _load("productivity_local_year.csv")
    if df is None or "platform_exposure_class" not in df.columns:
        return
    classes = [c for c in PLATFORM_ORDER if c in df["platform_exposure_class"].values]
    data    = [df.loc[df["platform_exposure_class"] == c, "cpue_ton_per_trip"].dropna()
               for c in classes]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bp = ax.boxplot(data, labels=classes, patch_artist=True)
    for patch, cls in zip(bp["boxes"], classes):
        patch.set_facecolor(cfg.PALETTE_PLATFORM.get(cls, "grey"))
    ax.set_xlabel("Platform exposure class"); ax.set_ylabel("CPUE (ton/trip)")
    ax.set_title("CPUE distribution by platform exposure class")
    plt.xticks(rotation=20)
    _save_fig(fig, "F06_cpue_platform_boxplot.png")


def fig_diversity_platform_year() -> None:
    df = _load("productivity_platform_year.csv")
    if df is None or "shannon_h" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cls in PLATFORM_ORDER:
        sub = df[df["platform_exposure_class"] == cls].sort_values("year")
        ax.plot(sub["year"], sub["shannon_h"], marker="o", markersize=3,
                label=cls, color=cfg.PALETTE_PLATFORM.get(cls, "grey"))
    ax.set_xlabel("Year"); ax.set_ylabel("Shannon H'")
    ax.set_title("Species diversity (Shannon H') by platform exposure class")
    ax.legend(title="Platform distance", fontsize=8)
    _save_fig(fig, "F08_diversity_platform_time.png")


# ─── Gear & boat figures ──────────────────────────────────────────────────────

def fig_gear_stacked_time() -> None:
    df = _load("timeseries_gear_year.csv")
    if df is None:
        return
    gear_types = df["gear_type"].dropna().unique()
    pivot = df.pivot_table(
        index="year", columns="gear_type", values="share", fill_value=0,
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pivot.plot(kind="area", stacked=True, ax=ax, colormap="tab10", alpha=0.85)
    ax.set_xlabel("Year"); ax.set_ylabel("Share of production")
    ax.set_title("Gear type composition (share of total production) over time")
    ax.legend(title="Gear type", loc="upper left", fontsize=8)
    _save_fig(fig, "F10_gear_stacked_time.png")


def fig_gear_shares_platform() -> None:
    df = _load("gear_share_platform_year.csv")
    if df is None:
        return
    # Pivot: rows = platform class, cols = gear_type, values = mean share
    pivot = df.groupby(["platform_exposure_class", "gear_type"])["gear_share"].mean().unstack(fill_value=0)
    pivot = pivot.reindex([c for c in PLATFORM_ORDER if c in pivot.index])
    pivot.plot(kind="bar", stacked=False, figsize=FIGSIZE, colormap="tab10")
    plt.xlabel("Platform exposure class"); plt.ylabel("Mean gear production share")
    plt.title("Gear shares by platform exposure class")
    plt.xticks(rotation=20); plt.tight_layout()
    fig = plt.gcf()
    _save_fig(fig, "F11_gear_shares_platform.png")


def fig_boat_stacked_time() -> None:
    df = _load("timeseries_boat_year.csv")
    if df is None:
        return
    pivot = df.pivot_table(
        index="year", columns="propulsion", values="share", fill_value=0,
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pivot.plot(kind="area", stacked=True, ax=ax, colormap="Set2", alpha=0.85)
    ax.set_xlabel("Year"); ax.set_ylabel("Share of vessels monitored")
    ax.set_title("Boat propulsion composition over time")
    ax.legend(title="Propulsion", loc="upper left", fontsize=8)
    _save_fig(fig, "F13_boat_stacked_time.png")


# ─── Species figures ──────────────────────────────────────────────────────────

def fig_species_share_time() -> None:
    df = _load("timeseries_species_top20.csv")
    if df is None:
        return
    top_sp = df.groupby("species")["sp_production_ton"].sum().nlargest(10).index
    pivot = df[df["species"].isin(top_sp)].pivot_table(
        index="year", columns="species", values="species_share", fill_value=0,
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pivot.plot(kind="area", stacked=True, ax=ax, colormap="tab20", alpha=0.85)
    ax.set_xlabel("Year"); ax.set_ylabel("Share of total production")
    ax.set_title("Top-10 species share of total production over time")
    ax.legend(title="Species", loc="upper left", fontsize=7, ncol=2)
    _save_fig(fig, "F15_species_share_time.png")


def fig_simper_platform() -> None:
    df = _load("species_SIMPER_platform.csv")
    if df is None:
        return
    # Plot top 10 species by mean contribution across all pairs
    top = (
        df.groupby("species")["contribution"].mean()
        .nlargest(10).reset_index()
        .sort_values("contribution")
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["species"], top["contribution"], color="steelblue")
    ax.set_xlabel("Mean contribution to Bray-Curtis dissimilarity")
    ax.set_title("Top species driving platform-class differences (SIMPER)")
    _save_fig(fig, "F17_SIMPER_platform.png")


def fig_turnover_period() -> None:
    df = _load("species_turnover_period.csv")
    if df is None:
        return
    # Overall (all locals pooled)
    overall = df[df["local"] == "ALL"]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = overall.apply(lambda r: f"{r['period_a']}→{r['period_b']}", axis=1)
    ax.bar(labels, overall["bray_curtis"], color="coral")
    ax.set_ylabel("Bray-Curtis dissimilarity")
    ax.set_title("Species compositional turnover between periods")
    ax.set_ylim(0, 1)
    _save_fig(fig, "F18_turnover_period.png")


# ─── Temporal comparison ─────────────────────────────────────────────────────

def fig_regional_production_time() -> None:
    df = _load("timeseries_regional.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(df["year"], 0, df["production_ton_sum"], alpha=0.3, color="steelblue")
    ax.plot(df["year"], df["production_ton_sum"], color="steelblue", linewidth=1.5)
    # Period bands
    colours_p = {"early": "#ffffcc", "middle": "#c7e9b4", "recent": "#7fcdbb"}
    for period, (a, b) in cfg.PERIOD_BREAKS.items():
        ax.axvspan(a, b, alpha=0.15, color=colours_p.get(period, "grey"), label=period)
    ax.set_xlabel("Year"); ax.set_ylabel("Total production (ton)")
    ax.set_title("Regional total production over time")
    ax.legend(fontsize=8)
    _save_fig(fig, "F19_regional_production_time.png")


def fig_cpue_multipanel() -> None:
    regional = _load("timeseries_regional.csv")
    platform = _load("timeseries_platform.csv")
    mpa      = _load("timeseries_mpa.csv")
    if any(df is None for df in [regional, platform, mpa]):
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Panel 1: regional
    ax = axes[0]
    ax.plot(regional["year"], regional["cpue_regional"], color="black", linewidth=1.5)
    ax.set_title("Regional CPUE"); ax.set_xlabel("Year"); ax.set_ylabel("CPUE (ton/trip)")

    # Panel 2: platform
    ax = axes[1]
    for cls in PLATFORM_ORDER:
        sub = platform[platform["platform_exposure_class"] == cls].sort_values("year")
        ax.plot(sub["year"], sub["cpue"], label=cls,
                color=cfg.PALETTE_PLATFORM.get(cls, "grey"), linewidth=1)
    ax.set_title("CPUE by platform class"); ax.set_xlabel("Year")
    ax.legend(fontsize=7, title="Platform dist.")

    # Panel 3: MPA
    ax = axes[2]
    for cls in MPA_ORDER:
        sub = mpa[mpa["mpa_exposure_class"] == cls].sort_values("year")
        ax.plot(sub["year"], sub["cpue"], label=cls,
                color=cfg.PALETTE_MPA.get(cls, "grey"), linewidth=1)
    ax.set_title("CPUE by MPA class"); ax.set_xlabel("Year")
    ax.legend(fontsize=7, title="MPA exposure")

    fig.suptitle("CPUE time series: regional vs spatial exposure contexts", y=1.01)
    plt.tight_layout()
    _save_fig(fig, "F20_cpue_multipanel.png")


def fig_period_heatmap() -> None:
    df = _load("period_comparison.csv")
    if df is None or "cpue_agg" not in df.columns:
        return
    # Pivot: rows = platform class, cols = period
    pivot = df.pivot_table(
        index="platform_exposure_class", columns="period",
        values="cpue_agg", aggfunc="mean",
    )
    pivot = pivot.reindex([c for c in PLATFORM_ORDER if c in pivot.index])
    pivot = pivot[["early", "middle", "recent"] if all(
        c in pivot.columns for c in ["early", "middle", "recent"]) else pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label="CPUE (ton/trip)")
    ax.set_title("CPUE heatmap: platform exposure × temporal period")
    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    _save_fig(fig, "F21_period_heatmap.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg.setup_dirs()
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)

    log.info("Generating maps …")
    fig_study_area()
    fig_platform_distance_map()
    fig_mpa_exposure_map()

    log.info("Generating productivity figures …")
    fig_cpue_platform_time()
    fig_cpue_mpa_time()
    fig_cpue_platform_boxplot()
    fig_diversity_platform_year()

    log.info("Generating effort figures …")
    fig_gear_stacked_time()
    fig_gear_shares_platform()
    fig_boat_stacked_time()

    log.info("Generating species figures …")
    fig_species_share_time()
    fig_simper_platform()
    fig_turnover_period()

    log.info("Generating temporal comparison figures …")
    fig_regional_production_time()
    fig_cpue_multipanel()
    fig_period_heatmap()

    log.info("10_figures.py complete.")


if __name__ == "__main__":
    main()
