"""
04_productivity_diversity.py
============================
Computes productivity and diversity metrics at multiple aggregation levels,
then joins spatial exposure variables so outputs are ready for modelling
and visualisation.

Metrics computed:
  • CPUE (ton/trip) — from PMDP_MASTER: production_ton / assisted_trips
  • Species richness (S)
  • Shannon entropy (H')
  • Simpson diversity (1 − D)
  • Pielou's evenness (J' = H' / ln S)
  • Total production and total effort (assisted_trips)

Aggregation levels:
  • local × year              (finest resolution)
  • local × period            (early / middle / recent)
  • platform_exposure_class × year
  • mpa_exposure_class × year
  • combined_exposure_class × year
  • municipality × year

Outputs (data/processed/):
  - productivity_local_year.csv
  - productivity_local_period.csv
  - productivity_platform_year.csv
  - productivity_mpa_year.csv
  - productivity_combined_year.csv
  - productivity_municipality_year.csv

Run:
  python 04_productivity_diversity.py
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
        logging.FileHandler(cfg.LOGS / "04_productivity_diversity.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("04_productivity_diversity")


def _check(path: Path) -> None:
    if not path.exists():
        log.error("Missing input: %s", path)
        sys.exit(1)


def _save(df: pd.DataFrame, name: str) -> None:
    out = cfg.DATA_PROCESSED / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Saved %s  (%d rows × %d cols)", out.name, len(df), df.shape[1])


# ─── Diversity functions ──────────────────────────────────────────────────────

def shannon_entropy(counts: np.ndarray) -> float:
    """H' = -Σ p_i ln p_i; ignores zeros."""
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def simpson_diversity(counts: np.ndarray) -> float:
    """1 - D = 1 - Σ p_i²"""
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    p = counts / counts.sum()
    return float(1 - np.sum(p ** 2))


def pielou_evenness(counts: np.ndarray) -> float:
    """J' = H' / ln S; undefined for S=1."""
    counts = counts[counts > 0]
    s = len(counts)
    if s <= 1:
        return np.nan
    h = shannon_entropy(counts)
    return float(h / np.log(s))


def diversity_from_species_df(
    land: pd.DataFrame,
    groupby: list[str],
) -> pd.DataFrame:
    """
    Compute diversity metrics from PMDP_LANDINGS.
    land must have columns: groupby columns + 'species', 'sp_production_ton'
    Returns one row per group with richness, H', 1-D, J'.
    """
    records = []
    for keys, sub in land.groupby(groupby):
        vals = sub.groupby("species")["sp_production_ton"].sum().values
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(groupby, keys))
        rec["richness"]          = int((vals > 0).sum())
        rec["shannon_h"]         = shannon_entropy(vals)
        rec["simpson_1d"]        = simpson_diversity(vals)
        rec["pielou_j"]          = pielou_evenness(vals)
        rec["total_production_div"] = float(vals.sum())
        records.append(rec)

    return pd.DataFrame(records)


# ─── CPUE table ───────────────────────────────────────────────────────────────

def cpue_from_master(master: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute CPUE with threshold filter; return tidy table.
    """
    df = master.copy()
    df["cpue_ton_per_trip"] = np.where(
        df["assisted_trips"] >= cfg.MIN_TRIPS_CPUE,
        df["production_ton"] / df["assisted_trips"],
        np.nan,
    )
    return df


# ─── Aggregated productivity tables ──────────────────────────────────────────

def agg_productivity(
    master: pd.DataFrame,
    diversity: pd.DataFrame,
    groupby: list[str],
    label: str,
) -> pd.DataFrame:
    """
    Aggregate master-level productivity and merge diversity metrics.
    groupby: list of columns shared by both master and diversity DataFrames.
    """
    grp_m = master.groupby(groupby, dropna=False)
    agg_m = grp_m.agg(
        production_ton      =("production_ton",       "sum"),
        assisted_trips      =("assisted_trips",        "sum"),
        fleet_monitored     =("fleet_monitored",       "sum"),
        estimated_fishermen =("estimated_fishermen",   "sum"),
        n_locals            =("local",                 "nunique"),
    ).reset_index()

    agg_m["cpue_ton_per_trip"] = np.where(
        agg_m["assisted_trips"] >= cfg.MIN_TRIPS_CPUE,
        agg_m["production_ton"] / agg_m["assisted_trips"],
        np.nan,
    )

    # Merge diversity
    merged = agg_m.merge(diversity, on=groupby, how="left")
    log.info("Productivity table '%s': %d rows", label, len(merged))
    return merged


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg.setup_dirs()

    for f in ["locality_year.csv", "local_exposure.csv", "landings_clean.csv"]:
        _check(cfg.DATA_INTERIM / f)
    _check(cfg.DATA_INTERIM / "xwalk_local_updated.csv")

    master      = pd.read_csv(cfg.DATA_INTERIM / "locality_year.csv")
    exposure    = pd.read_csv(cfg.DATA_INTERIM / "local_exposure.csv")
    land        = pd.read_csv(cfg.DATA_INTERIM / "landings_clean.csv")
    xwalk_local = pd.read_csv(cfg.DATA_INTERIM / "xwalk_local_updated.csv")

    # Join exposure to master and landings
    master = master.merge(
        exposure[["local", "platform_dist_km_mean", "mpa_dist_km_mean",
                  "inside_any_mpa_any", "platform_exposure_class",
                  "mpa_exposure_class", "combined_exposure_class", "municipality"]],
        on="local", how="left",
        suffixes=("", "_exp"),
    )
    # Fill municipality from exposure if missing
    if "municipality_exp" in master.columns:
        master["municipality"] = master["municipality"].fillna(master["municipality_exp"])
        master = master.drop(columns=["municipality_exp"])

    # Join exposure classes to landings
    land = land.merge(
        exposure[["local", "platform_exposure_class", "mpa_exposure_class",
                  "combined_exposure_class", "municipality"]],
        on="local", how="left",
    )

    master = cpue_from_master(master)

    # ── local × year ──────────────────────────────────────────────────────────
    div_ly = diversity_from_species_df(land, ["local", "year"])
    prod_ly = master.merge(div_ly, on=["local", "year"], how="left")
    _save(prod_ly, "productivity_local_year.csv")

    # ── local × period ────────────────────────────────────────────────────────
    def assign_period(y):
        for period, (a, b) in cfg.PERIOD_BREAKS.items():
            if a <= int(y) <= b:
                return period
        return pd.NA

    land["period"]   = land["year"].apply(assign_period)
    master["period"] = master["year"].apply(assign_period)

    div_lp = diversity_from_species_df(land, ["local", "period"])
    prod_lp = agg_productivity(master, div_lp, ["local", "period"], "local×period")
    _save(prod_lp, "productivity_local_period.csv")

    # ── platform_exposure_class × year ────────────────────────────────────────
    div_pf_yr = diversity_from_species_df(land, ["platform_exposure_class", "year"])
    prod_pf_yr = agg_productivity(
        master, div_pf_yr, ["platform_exposure_class", "year"], "platform×year"
    )
    _save(prod_pf_yr, "productivity_platform_year.csv")

    # ── mpa_exposure_class × year ─────────────────────────────────────────────
    div_mpa_yr = diversity_from_species_df(land, ["mpa_exposure_class", "year"])
    prod_mpa_yr = agg_productivity(
        master, div_mpa_yr, ["mpa_exposure_class", "year"], "mpa×year"
    )
    _save(prod_mpa_yr, "productivity_mpa_year.csv")

    # ── combined_exposure_class × year ────────────────────────────────────────
    div_cmb_yr = diversity_from_species_df(land, ["combined_exposure_class", "year"])
    prod_cmb_yr = agg_productivity(
        master, div_cmb_yr, ["combined_exposure_class", "year"], "combined×year"
    )
    _save(prod_cmb_yr, "productivity_combined_year.csv")

    # ── municipality × year ───────────────────────────────────────────────────
    div_mun_yr = diversity_from_species_df(land, ["municipality", "year"])
    prod_mun_yr = agg_productivity(
        master, div_mun_yr, ["municipality", "year"], "municipality×year"
    )
    _save(prod_mun_yr, "productivity_municipality_year.csv")

    log.info("04_productivity_diversity.py complete.")


if __name__ == "__main__":
    main()
