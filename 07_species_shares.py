"""
07_species_shares.py
====================
Computes species-level proportional contributions to total catch across all
exposure contexts, gear structures, boat structures, and time.

Complements the community-level outputs of 06 with species-level summaries
that can be directly used in plots and reporting.

Outputs (data/processed/):
  - species_share_local_year.csv      species × local × year: share of total catch
  - species_share_platform_year.csv   species × platform_exposure_class × year
  - species_share_mpa_year.csv        species × mpa_exposure_class × year
  - species_share_combined_year.csv   species × combined_exposure_class × year
  - species_share_gear_year.csv       species × gear_type × year
  - species_share_boat_year.csv       species × propulsion × year
  - species_rank_platform.csv         top-N species per platform class (overall)
  - species_rank_mpa.csv              top-N species per MPA class (overall)
  - species_rank_combined.csv         top-N species per combined class (overall)
  - species_year.csv                  species-level time series (all localities)

Run:
  python 07_species_shares.py
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
        logging.FileHandler(cfg.LOGS / "07_species_shares.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("07_species_shares")

TOP_N_SPECIES = 20   # number of top species to retain in ranking tables


def _check(path: Path) -> None:
    if not path.exists():
        log.error("Missing input: %s", path)
        sys.exit(1)


def _save(df: pd.DataFrame, name: str) -> None:
    out = cfg.DATA_PROCESSED / name
    df.to_csv(out, index=False)
    log.info("Saved %s  (%d rows × %d cols)", out.name, len(df), df.shape[1])


# ─── Core share calculator ────────────────────────────────────────────────────

def species_share(
    land: pd.DataFrame,
    groupby: list[str],
) -> pd.DataFrame:
    """
    Compute proportional share of each species within each group defined by groupby.

    Returns long DataFrame:
      groupby columns | species | sp_production_ton | total_production_ton | species_share
    """
    grp_cols = groupby + ["species"]
    agg = land.groupby(grp_cols, dropna=False)["sp_production_ton"].sum().reset_index()
    total = agg.groupby(groupby, dropna=False)["sp_production_ton"] \
               .sum().rename("total_production_ton")
    agg = agg.merge(total, on=groupby, how="left")
    agg["species_share"] = np.where(
        agg["total_production_ton"] > 0,
        agg["sp_production_ton"] / agg["total_production_ton"],
        np.nan,
    )
    return agg.sort_values(groupby + ["species_share"], ascending=False).reset_index(drop=True)


def top_species_per_group(df: pd.DataFrame, groupby: list[str], n: int) -> pd.DataFrame:
    """
    Retain the top-N species by mean species_share within each exposure group.
    """
    mean_share = (
        df.groupby(groupby + ["species"], dropna=False)["species_share"]
        .mean()
        .reset_index()
        .rename(columns={"species_share": "mean_species_share"})
    )
    mean_share["rank"] = (
        mean_share.groupby(groupby, dropna=False)["mean_species_share"]
        .rank(ascending=False, method="first")
    )
    return mean_share[mean_share["rank"] <= n].sort_values(groupby + ["rank"])


# ─── Gear/boat enrichment ─────────────────────────────────────────────────────

def join_gear_to_landings(
    land: pd.DataFrame,
    prod: pd.DataFrame,
) -> pd.DataFrame:
    """
    Associate each local × year species record with dominant gear type.
    Dominant gear = gear_type with highest gear_production_ton in that local × year.
    """
    dominant_gear = (
        prod.groupby(["local", "year", "gear_type"])["gear_production_ton"]
        .sum()
        .reset_index()
        .sort_values("gear_production_ton", ascending=False)
        .drop_duplicates(subset=["local", "year"])
        .rename(columns={"gear_type": "dominant_gear_type"})
    )[["local", "year", "dominant_gear_type"]]

    return land.merge(dominant_gear, on=["local", "year"], how="left")


def join_boat_to_landings(
    land: pd.DataFrame,
    comp: pd.DataFrame,
    xb: pd.DataFrame,
) -> pd.DataFrame:
    """
    Associate each local × year species record with dominant boat propulsion.
    Dominant propulsion = mode of boat_type weighted by vessels_monitored.
    """
    comp_xb = comp.merge(xb[["boat_type", "propulsion"]], on="boat_type", how="left")
    dominant_boat = (
        comp_xb.groupby(["local", "year", "propulsion"])["vessels_monitored"]
        .sum()
        .reset_index()
        .sort_values("vessels_monitored", ascending=False)
        .drop_duplicates(subset=["local", "year"])
        .rename(columns={"propulsion": "dominant_propulsion"})
    )[["local", "year", "dominant_propulsion"]]

    return land.merge(dominant_boat, on=["local", "year"], how="left")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg.setup_dirs()

    for f in ["landings_clean.csv", "local_exposure.csv",
              "production_clean.csv", "composition_clean.csv", "xwalk_boat.csv"]:
        _check(cfg.DATA_INTERIM / f)

    land     = pd.read_csv(cfg.DATA_INTERIM / "landings_clean.csv")
    exposure = pd.read_csv(cfg.DATA_INTERIM / "local_exposure.csv")
    prod     = pd.read_csv(cfg.DATA_INTERIM / "production_clean.csv")
    comp     = pd.read_csv(cfg.DATA_INTERIM / "composition_clean.csv")
    xb       = pd.read_csv(cfg.DATA_INTERIM / "xwalk_boat.csv")

    # Join exposure classes to landings
    land = land.merge(
        exposure[["local", "platform_exposure_class", "mpa_exposure_class",
                  "combined_exposure_class", "municipality"]],
        on="local", how="left",
    )

    # Gear and boat context
    land = join_gear_to_landings(land, prod)
    land = join_boat_to_landings(land, comp, xb)

    # Period
    def assign_period(y):
        for period, (a, b) in cfg.PERIOD_BREAKS.items():
            if pd.notna(y) and a <= int(y) <= b:
                return period
        return pd.NA

    land["period"] = land["year"].apply(assign_period)

    # ── Species-share tables ──────────────────────────────────────────────────
    tables = {
        "species_share_local_year.csv":      ["local", "year"],
        "species_share_platform_year.csv":   ["platform_exposure_class", "year"],
        "species_share_mpa_year.csv":        ["mpa_exposure_class", "year"],
        "species_share_combined_year.csv":   ["combined_exposure_class", "year"],
        "species_share_gear_year.csv":       ["dominant_gear_type", "year"],
        "species_share_boat_year.csv":       ["dominant_propulsion", "year"],
    }
    for fname, groupby in tables.items():
        df = species_share(land, groupby)
        _save(df, fname)

    # ── Top-N species per exposure class ─────────────────────────────────────
    ranking_tables = {
        "species_rank_platform.csv":  ("platform_exposure_class", ["platform_exposure_class"]),
        "species_rank_mpa.csv":       ("mpa_exposure_class",      ["mpa_exposure_class"]),
        "species_rank_combined.csv":  ("combined_exposure_class", ["combined_exposure_class"]),
    }
    for fname, (share_fname_prefix, groupby) in ranking_tables.items():
        share_fname = f"species_share_{share_fname_prefix.split('_')[0]}_year.csv"
        share_df = pd.read_csv(cfg.DATA_PROCESSED / share_fname)
        top = top_species_per_group(share_df, groupby, TOP_N_SPECIES)
        _save(top, fname)

    # ── Species time series (all localities pooled) ───────────────────────────
    species_yr = land.groupby(["species", "year"], dropna=False).agg(
        sp_production_ton   =("sp_production_ton", "sum"),
        n_locals            =("local", "nunique"),
    ).reset_index()
    total_yr = species_yr.groupby("year")["sp_production_ton"].sum().rename("year_total")
    species_yr = species_yr.merge(total_yr, on="year", how="left")
    species_yr["species_share_year"] = (
        species_yr["sp_production_ton"] / species_yr["year_total"]
    )
    _save(species_yr, "species_year.csv")

    log.info("07_species_shares.py complete.")


if __name__ == "__main__":
    main()
