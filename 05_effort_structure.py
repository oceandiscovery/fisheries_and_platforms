"""
05_effort_structure.py
======================
Analyses fishing effort composition along gear and boat dimensions.

Inputs:
  - production_clean.csv        gear × local × year production
  - composition_clean.csv       boat_type × local × year
  - local_exposure.csv          spatial exposure variables per local
  - xwalk_gear.csv              gear crosswalk
  - xwalk_boat.csv              boat crosswalk

Outputs (data/processed/):
  Gear shares:
  - gear_share_local_year.csv   gear proportional share by local × year
  - gear_share_platform_year.csv  gear share by platform exposure × year
  - gear_share_mpa_year.csv
  - gear_share_combined_year.csv
  Boat shares:
  - boat_share_local_year.csv
  - boat_share_platform_year.csv
  - boat_share_mpa_year.csv
  - boat_share_combined_year.csv
  Gear-year:
  - gear_year.csv               production, share, exposure by gear × year
  Boat-year:
  - boat_year.csv               vessel counts, share, exposure by boat × year

Run:
  python 05_effort_structure.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_00 as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT,
    handlers=[
        logging.FileHandler(cfg.LOGS / "05_effort_structure.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("05_effort_structure")


def _check(path: Path) -> None:
    if not path.exists():
        log.error("Missing input: %s", path)
        sys.exit(1)


def _save(df: pd.DataFrame, name: str) -> None:
    out = cfg.DATA_PROCESSED / name
    df.to_csv(out, index=False)
    log.info("Saved %s  (%d rows × %d cols)", out.name, len(df), df.shape[1])


# ─── Gear share computation ───────────────────────────────────────────────────

def compute_gear_shares(prod: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    """
    Compute proportional gear-production share within each group.
    groupby: non-gear columns defining the aggregation level.
    """
    # Total production per group
    prod_agg = prod.groupby(groupby + ["gear_cod", "gear_type", "gear_group"],
                             dropna=False)["gear_production_ton"].sum().reset_index()
    total = prod_agg.groupby(groupby, dropna=False)["gear_production_ton"] \
                    .sum().rename("total_production_ton")
    prod_agg = prod_agg.merge(total, on=groupby, how="left")
    prod_agg["gear_share"] = np.where(
        prod_agg["total_production_ton"] > 0,
        prod_agg["gear_production_ton"] / prod_agg["total_production_ton"],
        np.nan,
    )
    # Active vs passive share (broad)
    broad = prod_agg.groupby(groupby + ["gear_type"], dropna=False) \
                    ["gear_production_ton"].sum().reset_index()
    broad = broad.pivot_table(
        index=groupby, columns="gear_type",
        values="gear_production_ton", aggfunc="sum", fill_value=0,
    ).reset_index()
    broad.columns.name = None
    broad_total = broad.drop(columns=groupby).sum(axis=1)
    for col in ["active", "passive", "mixed"]:
        if col in broad.columns:
            broad[f"share_{col}"] = broad[col] / broad_total.replace(0, np.nan)
        else:
            broad[f"share_{col}"] = np.nan

    result = prod_agg.merge(
        broad[groupby + [c for c in broad.columns if c.startswith("share_")]],
        on=groupby, how="left",
    )
    return result


# ─── Boat share computation ───────────────────────────────────────────────────

def compute_boat_shares(comp: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    """
    Compute proportional boat-type vessel share within each group.
    Uses vessels_monitored as effort proxy.
    """
    comp_agg = comp.groupby(groupby + ["boat_type", "propulsion"],
                             dropna=False)["vessels_monitored"].sum().reset_index()
    total = comp_agg.groupby(groupby, dropna=False)["vessels_monitored"] \
                    .sum().rename("total_vessels")
    comp_agg = comp_agg.merge(total, on=groupby, how="left")
    comp_agg["boat_share"] = np.where(
        comp_agg["total_vessels"] > 0,
        comp_agg["vessels_monitored"] / comp_agg["total_vessels"],
        np.nan,
    )
    # Motorised vs non-motorised share
    broad = comp_agg.groupby(groupby + ["propulsion"], dropna=False) \
                    ["vessels_monitored"].sum().reset_index()
    broad = broad.pivot_table(
        index=groupby, columns="propulsion",
        values="vessels_monitored", aggfunc="sum", fill_value=0,
    ).reset_index()
    broad.columns.name = None
    broad_total = broad.drop(columns=groupby).sum(axis=1)
    for col in ["motorised", "sail/row", "other"]:
        col_key = col.replace("/", "_")
        if col in broad.columns:
            broad[f"share_{col_key}"] = broad[col] / broad_total.replace(0, np.nan)
        else:
            broad[f"share_{col_key}"] = np.nan

    result = comp_agg.merge(
        broad[groupby + [c for c in broad.columns if c.startswith("share_")]],
        on=groupby, how="left",
    )
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg.setup_dirs()

    for f in ["production_clean.csv", "composition_clean.csv",
              "local_exposure.csv", "xwalk_gear.csv", "xwalk_boat.csv"]:
        _check(cfg.DATA_INTERIM / f)

    prod     = pd.read_csv(cfg.DATA_INTERIM / "production_clean.csv")
    comp     = pd.read_csv(cfg.DATA_INTERIM / "composition_clean.csv")
    exposure = pd.read_csv(cfg.DATA_INTERIM / "local_exposure.csv")
    xg       = pd.read_csv(cfg.DATA_INTERIM / "xwalk_gear.csv")
    xb       = pd.read_csv(cfg.DATA_INTERIM / "xwalk_boat.csv")

    # Enrich with gear/boat metadata and exposure
    exp_cols = ["local", "platform_exposure_class", "mpa_exposure_class",
                "combined_exposure_class", "municipality"]

    extra_cols = [c for c in ["gear_type", "gear_group"] if c not in prod.columns]
    if extra_cols:
        prod = prod.merge(xg[["gear_cod"] + extra_cols], on="gear_cod", how="left")
    prod = prod.merge(exposure[exp_cols], on="local", how="left")

    comp = comp.merge(xb[["boat_type", "propulsion"]], on="boat_type", how="left")
    comp = comp.merge(exposure[exp_cols], on="local", how="left")

    # Period
    def assign_period(y):
        for period, (a, b) in cfg.PERIOD_BREAKS.items():
            if pd.notna(y) and a <= int(y) <= b:
                return period
        return pd.NA

    prod["period"] = prod["year"].apply(assign_period)
    comp["period"] = comp["year"].apply(assign_period)

    # ── Gear shares ───────────────────────────────────────────────────────────
    gs_ly  = compute_gear_shares(prod, ["local", "year"])
    gs_pf  = compute_gear_shares(prod, ["platform_exposure_class", "year"])
    gs_mpa = compute_gear_shares(prod, ["mpa_exposure_class", "year"])
    gs_cmb = compute_gear_shares(prod, ["combined_exposure_class", "year"])

    _save(gs_ly,  "gear_share_local_year.csv")
    _save(gs_pf,  "gear_share_platform_year.csv")
    _save(gs_mpa, "gear_share_mpa_year.csv")
    _save(gs_cmb, "gear_share_combined_year.csv")

    # Gear-year summary
    gear_year = prod.groupby(["gear_cod", "gear_type", "gear_group", "year"],
                              dropna=False).agg(
        gear_production_ton =("gear_production_ton", "sum"),
        n_locals            =("local", "nunique"),
    ).reset_index()
    total_yr = prod.groupby("year", dropna=False)["gear_production_ton"].sum() \
                   .rename("year_total_production")
    gear_year = gear_year.merge(total_yr, on="year", how="left")
    gear_year["gear_share_of_year"] = (
        gear_year["gear_production_ton"] / gear_year["year_total_production"]
    )
    _save(gear_year, "gear_year.csv")

    # ── Boat shares ───────────────────────────────────────────────────────────
    bs_ly  = compute_boat_shares(comp, ["local", "year"])
    bs_pf  = compute_boat_shares(comp, ["platform_exposure_class", "year"])
    bs_mpa = compute_boat_shares(comp, ["mpa_exposure_class", "year"])
    bs_cmb = compute_boat_shares(comp, ["combined_exposure_class", "year"])

    _save(bs_ly,  "boat_share_local_year.csv")
    _save(bs_pf,  "boat_share_platform_year.csv")
    _save(bs_mpa, "boat_share_mpa_year.csv")
    _save(bs_cmb, "boat_share_combined_year.csv")

    # Boat-year summary
    boat_year = comp.groupby(["boat_type", "propulsion", "year"],
                              dropna=False).agg(
        vessels_monitored = ("vessels_monitored", "sum"),
        n_locals          = ("local", "nunique"),
    ).reset_index()
    total_ves = comp.groupby("year", dropna=False)["vessels_monitored"].sum() \
                    .rename("year_total_vessels")
    boat_year = boat_year.merge(total_ves, on="year", how="left")
    boat_year["boat_share_of_year"] = (
        boat_year["vessels_monitored"] / boat_year["year_total_vessels"]
    )
    _save(boat_year, "boat_year.csv")

    log.info("05_effort_structure.py complete.")


if __name__ == "__main__":
    main()
