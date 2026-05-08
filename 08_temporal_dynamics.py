"""
08_temporal_dynamics.py
=======================
Builds temporal summary outputs that allow comparison between:
  • general (regional) temporal trends
  • trends within each platform exposure class
  • trends within each MPA exposure class
  • trends within each combined exposure class
  • changes in gear, boat, and species composition over time

Outputs (data/processed/):
  - timeseries_regional.csv        Regional totals and means by year
  - timeseries_platform.csv        Time series by platform exposure class
  - timeseries_mpa.csv             Time series by MPA exposure class
  - timeseries_combined.csv        Time series by combined exposure class
  - timeseries_gear_year.csv       Gear-type production shares by year
  - timeseries_boat_year.csv       Boat-type vessel shares by year
  - timeseries_species_top20.csv   Top-20 species shares by year
  - period_comparison.csv          Period-level summary comparing all contexts

Run:
  python 08_temporal_dynamics.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_00 as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT,
    handlers=[
        logging.FileHandler(cfg.LOGS / "08_temporal_dynamics.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("08_temporal_dynamics")


def _check(path: Path) -> None:
    if not path.exists():
        log.error("Missing input: %s", path)
        sys.exit(1)


def _save(df: pd.DataFrame, name: str) -> None:
    out = cfg.DATA_PROCESSED / name
    df.to_csv(out, index=False)
    log.info("Saved %s  (%d rows × %d cols)", out.name, len(df), df.shape[1])


# ─── Regional time series ─────────────────────────────────────────────────────

def regional_timeseries(master: pd.DataFrame) -> pd.DataFrame:
    """
    Year-level regional aggregation: sums, means, and CPUE.
    """
    grp = master.groupby("year", dropna=False)
    ts = grp.agg(
        production_ton_sum      =("production_ton",       "sum"),
        production_ton_mean     =("production_ton",       "mean"),
        assisted_trips_sum      =("assisted_trips",        "sum"),
        fleet_monitored_sum     =("fleet_monitored",       "sum"),
        estimated_fishermen_sum =("estimated_fishermen",   "sum"),
        cpue_mean               =("cpue_ton_per_trip",     "mean"),
        cpue_median             =("cpue_ton_per_trip",     "median"),
        n_locals                =("local",                 "nunique"),
    ).reset_index()

    ts["cpue_regional"] = np.where(
        ts["assisted_trips_sum"] > 0,
        ts["production_ton_sum"] / ts["assisted_trips_sum"],
        np.nan,
    )
    return ts


def exposure_timeseries(master: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Time series of key metrics stratified by an exposure class column.
    """
    grp = master.groupby([group_col, "year"], dropna=False)
    ts = grp.agg(
        production_ton      =("production_ton",       "sum"),
        assisted_trips      =("assisted_trips",        "sum"),
        fleet_monitored     =("fleet_monitored",       "sum"),
        estimated_fishermen =("estimated_fishermen",   "sum"),
        cpue_mean           =("cpue_ton_per_trip",     "mean"),
        n_locals            =("local",                 "nunique"),
    ).reset_index()
    ts["cpue"] = np.where(
        ts["assisted_trips"] > 0,
        ts["production_ton"] / ts["assisted_trips"],
        np.nan,
    )
    return ts


# ─── Gear and boat time series ────────────────────────────────────────────────

def gear_timeseries(prod: pd.DataFrame) -> pd.DataFrame:
    agg = prod.groupby(["gear_type", "year"], dropna=False)["gear_production_ton"] \
              .sum().reset_index()
    total = agg.groupby("year")["gear_production_ton"].sum().rename("year_total")
    agg = agg.merge(total, on="year", how="left")
    agg["share"] = agg["gear_production_ton"] / agg["year_total"].replace(0, np.nan)
    return agg.sort_values(["year", "gear_type"])


def boat_timeseries(comp: pd.DataFrame, xb: pd.DataFrame) -> pd.DataFrame:
    comp_xb = comp.merge(xb[["boat_type", "propulsion"]], on="boat_type", how="left")
    agg = comp_xb.groupby(["propulsion", "year"], dropna=False)["vessels_monitored"] \
                 .sum().reset_index()
    total = agg.groupby("year")["vessels_monitored"].sum().rename("year_total")
    agg = agg.merge(total, on="year", how="left")
    agg["share"] = agg["vessels_monitored"] / agg["year_total"].replace(0, np.nan)
    return agg.sort_values(["year", "propulsion"])


# ─── Species time series (top 20 by overall catch) ───────────────────────────

def species_timeseries(land: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    # Identify top N species by total catch across all years
    top_sp = (
        land.groupby("species")["sp_production_ton"].sum()
        .nlargest(n).index.tolist()
    )
    sub = land[land["species"].isin(top_sp)]
    agg = sub.groupby(["species", "year"], dropna=False)["sp_production_ton"] \
             .sum().reset_index()
    total_yr = land.groupby("year")["sp_production_ton"].sum().rename("year_total")
    agg = agg.merge(total_yr, on="year", how="left")
    agg["species_share"] = agg["sp_production_ton"] / agg["year_total"].replace(0, np.nan)
    return agg.sort_values(["year", "species"])


# ─── Period comparison table ──────────────────────────────────────────────────

def period_comparison(master: pd.DataFrame) -> pd.DataFrame:
    """
    Compare productivity metrics across temporal periods and exposure classes.
    """
    def assign_period(y):
        for period, (a, b) in cfg.PERIOD_BREAKS.items():
            if pd.notna(y) and a <= int(y) <= b:
                return period
        return pd.NA

    df = master.copy()
    df["period"] = df["year"].apply(assign_period)

    records = []
    for (period, platform_cls), sub in df.groupby(
        ["period", "platform_exposure_class"], dropna=False
    ):
        records.append({
            "period":                   period,
            "platform_exposure_class":  platform_cls,
            "mpa_exposure_class":       sub["mpa_exposure_class"].mode().iloc[0]
                                        if len(sub) > 0 and len(sub["mpa_exposure_class"].mode()) > 0 else pd.NA,
            "production_ton":           sub["production_ton"].sum(),
            "assisted_trips":           sub["assisted_trips"].sum(),
            "cpue_mean":                sub["cpue_ton_per_trip"].mean(),
            "n_locals":                 sub["local"].nunique(),
        })

    period_df = pd.DataFrame(records)
    period_df["cpue_agg"] = np.where(
        period_df["assisted_trips"] > 0,
        period_df["production_ton"] / period_df["assisted_trips"],
        np.nan,
    )
    return period_df.sort_values(["period", "platform_exposure_class"])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg.setup_dirs()

    for f in [
        "productivity_local_year.csv",
        "production_clean.csv",
        "composition_clean.csv",
        "landings_clean.csv",
        "xwalk_boat.csv",
        "local_exposure.csv",
    ]:
        path = cfg.DATA_PROCESSED / f if (cfg.DATA_PROCESSED / f).exists() \
               else cfg.DATA_INTERIM / f
        _check(path)

    def _load(name: str) -> pd.DataFrame:
        p = cfg.DATA_PROCESSED / name
        if not p.exists():
            p = cfg.DATA_INTERIM / name
        return pd.read_csv(p)

    master   = _load("productivity_local_year.csv")
    prod     = _load("production_clean.csv")
    comp     = _load("composition_clean.csv")
    land     = _load("landings_clean.csv")
    xb       = _load("xwalk_boat.csv")
    exposure = _load("local_exposure.csv")

    # Enrich production and landings with exposure
    prod = prod.merge(
        exposure[["local", "platform_exposure_class", "mpa_exposure_class",
                  "combined_exposure_class"]],
        on="local", how="left",
    )
    comp = comp.merge(
        exposure[["local", "platform_exposure_class", "mpa_exposure_class"]],
        on="local", how="left",
    )
    land = land.merge(
        exposure[["local", "platform_exposure_class", "mpa_exposure_class",
                  "combined_exposure_class"]],
        on="local", how="left",
    )

    # ── Time series ───────────────────────────────────────────────────────────
    _save(regional_timeseries(master),                       "timeseries_regional.csv")
    _save(exposure_timeseries(master, "platform_exposure_class"), "timeseries_platform.csv")
    _save(exposure_timeseries(master, "mpa_exposure_class"),      "timeseries_mpa.csv")
    _save(exposure_timeseries(master, "combined_exposure_class"), "timeseries_combined.csv")
    _save(gear_timeseries(prod),                             "timeseries_gear_year.csv")
    _save(boat_timeseries(comp, xb),                         "timeseries_boat_year.csv")
    _save(species_timeseries(land, n=20),                    "timeseries_species_top20.csv")
    _save(period_comparison(master),                         "period_comparison.csv")

    log.info("08_temporal_dynamics.py complete.")


if __name__ == "__main__":
    main()
