"""
02_crosswalks.py
================
Builds all reference crosswalk tables from the cleaned data.

Outputs (data/interim/):
  - xwalk_gear.csv       gear_cod → gear_type, gear_group, gear_group_broad
  - xwalk_boat.csv       boat_type code → description (manual or inferred)
  - xwalk_species.csv    species name → taxonomic group (broad)
  - xwalk_local.csv      local → municipality (NM_MUN from spatial layer)
  - xwalk_landing.csv    local → list of landing_point_mean names
  - locality_year.csv    analytical table: local × year with all key metrics joined
  - municipality_year.csv analytical table: municipality × year aggregated

Run:
  python 02_crosswalks.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_00 as cfg  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format=cfg.LOG_FORMAT, datefmt=cfg.LOG_DATEFMT,
    handlers=[
        logging.FileHandler(cfg.LOGS / "02_crosswalks.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("02_crosswalks")


def _check(path: Path) -> None:
    if not path.exists():
        log.error("Missing input: %s", path)
        sys.exit(1)


def _save(df: pd.DataFrame, name: str) -> None:
    out = cfg.DATA_INTERIM / name
    df.to_csv(out, index=False)
    log.info("Saved %s  (%d rows)", out.name, len(df))


# ─── Gear crosswalk ───────────────────────────────────────────────────────────

def build_gear_xwalk(prod: pd.DataFrame) -> pd.DataFrame:
    """
    Derive gear crosswalk from PMDP_PRODUCTION.
    One row per gear_cod with canonical gear_type and gear_group.
    """
    xwalk = (
        prod[["gear_cod", "gear_type", "gear_group"]]
        .dropna(subset=["gear_cod"])
        .drop_duplicates(subset=["gear_cod"])
        .sort_values("gear_cod")
        .reset_index(drop=True)
    )
    # gear_group_broad: simplify to three-level hierarchy
    xwalk["gear_group_broad"] = xwalk["gear_type"].map({
        "passive": "passive",
        "active":  "active",
        "mixed":   "mixed",
    }).fillna("unknown")
    return xwalk


# ─── Boat crosswalk ───────────────────────────────────────────────────────────

# Boat-type codes observed in PMDP_COMPOSITION.  Extend as needed.
BOAT_DESCRIPTIONS = {
    "BMM": "Motor boat — medium",
    "BMP": "Motor boat — small",
    "BMG": "Motor boat — large",
    "BOV": "Sailing boat",
    "BOM": "Rowing/other boat — medium",
    "CAM": "Canoe — motorised",
    "CAR": "Canoe — rowing",
    "CAV": "Canoe — sailing",
    "JAM": "Jangada — motorised",
    "JAV": "Jangada — sailing",
    "PED": "Wooden boat",
    "PQM": "Small motor vessel",
    "PQR": "Small rowing vessel",
    "PQV": "Small sailing vessel",
    "TNR": "Trawler",
    "TRF": "Tarrafa / cast-net vessel",
}


def build_boat_xwalk(comp: pd.DataFrame) -> pd.DataFrame:
    observed = comp["boat_type"].dropna().unique()
    xwalk = pd.DataFrame({"boat_type": sorted(observed)})
    xwalk["boat_description"] = xwalk["boat_type"].map(BOAT_DESCRIPTIONS).fillna("Unknown")
    # Broad category: motorised / non-motorised / unknown
    motor_codes = {"BMM", "BMP", "BMG", "CAM", "JAM", "PQM", "TNR"}
    sail_codes  = {"BOV", "CAV", "JAV", "PQV"}
    xwalk["propulsion"] = xwalk["boat_type"].map(
        lambda c: "motorised" if c in motor_codes
        else ("sail/row" if c in sail_codes else "other")
    )
    return xwalk


# ─── Species crosswalk ────────────────────────────────────────────────────────

# Broad taxonomic groups — extend with domain knowledge as needed.
SPECIES_GROUPS = {
    # Pelagic schooling
    "Sardinha":      "small_pelagic",
    "Agulha":        "small_pelagic",
    "Agulhao":       "small_pelagic",
    "Peixe Voador":  "small_pelagic",
    "Arabaiana":     "large_pelagic",
    "Albacora":      "large_pelagic",
    "Albacorinha":   "large_pelagic",
    "Cavala":        "large_pelagic",
    "Serra":         "large_pelagic",
    "Bonito":        "large_pelagic",
    # Demersal
    "Pescada":       "demersal",
    "Sirigado":      "demersal",
    "Garoupa":       "demersal",
    "Pargo":         "demersal",
    "Badejo":        "demersal",
    "Biquara":       "demersal",
    "Cioba":         "demersal",
    "Ariaco":        "demersal",
    "Xareu":         "demersal",
    "Tainha":        "demersal",
    "Beijupira":     "demersal",
    "Bicuda":        "demersal",
    "Espada":        "demersal",
    "Budiao":        "demersal",
    # Invertebrates
    "Lagosta":       "invertebrate",
    "Camarao":       "invertebrate",
    "Polvo":         "invertebrate",
    "Siri":          "invertebrate",
    "Siriboia":      "invertebrate",
    # Elasmobranch
    "Arraia":        "elasmobranch",
    "Cação":         "elasmobranch",
    # Other/aggregated
    "Outros":        "other",
}


def build_species_xwalk(land: pd.DataFrame) -> pd.DataFrame:
    observed = sorted(land["species"].dropna().unique())
    xwalk = pd.DataFrame({"species": observed})
    xwalk["taxon_group"] = xwalk["species"].map(SPECIES_GROUPS).fillna("other")
    return xwalk


# ─── Locality → municipality crosswalk ────────────────────────────────────────
# Built from landing_points (which carry local names).
# Municipality assignment is done spatially in 03_spatial_exposure.py;
# here we create a placeholder that can be updated after spatial join.

def build_local_xwalk(master: pd.DataFrame, lp: pd.DataFrame) -> pd.DataFrame:
    locals_master = set(master["local"].unique())
    locals_lp     = set(lp["local"].unique())
    all_locals = sorted(locals_master | locals_lp)

    xwalk = pd.DataFrame({"local": all_locals})
    # municipality will be filled by spatial join in 03_spatial_exposure.py
    xwalk["municipality"] = pd.NA
    # Note number of landing points per local
    lp_counts = lp.groupby("local")["landing_point_mean"].count().rename("n_landing_points")
    xwalk = xwalk.merge(lp_counts, on="local", how="left")
    xwalk["n_landing_points"] = xwalk["n_landing_points"].fillna(0).astype(int)
    return xwalk


# ─── Landing point → local crosswalk ─────────────────────────────────────────

def build_landing_xwalk(lp: pd.DataFrame) -> pd.DataFrame:
    return (
        lp[["local", "landing_point_mean", "latitude", "longitude"]]
        .drop_duplicates()
        .sort_values(["local", "landing_point_mean"])
        .reset_index(drop=True)
    )


# ─── Analytical tables: locality-year and municipality-year ──────────────────

def build_locality_year(master: pd.DataFrame, socio: pd.DataFrame,
                        xwalk_local: pd.DataFrame) -> pd.DataFrame:
    """
    Master analytical table at local × year resolution.
    Joins master + socioeconomic; leaves municipality column for
    update after spatial join (03_spatial_exposure.py).
    """
    df = master.merge(
        socio[["local", "year", "fishermen_per_vessel"]],
        on=["local", "year"], how="left",
    )
    df = df.merge(
        xwalk_local[["local", "municipality", "n_landing_points"]],
        on="local", how="left",
    )
    # Temporal period
    def assign_period(y):
        for period, (a, b) in cfg.PERIOD_BREAKS.items():
            if a <= y <= b:
                return period
        return pd.NA

    df["period"] = df["year"].apply(assign_period)
    return df.sort_values(["local", "year"]).reset_index(drop=True)


def build_municipality_year(locality_year: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate locality_year to municipality × year.
    production_ton summed; assisted_trips summed; CPUE recomputed;
    fleet_monitored and estimated_fishermen summed.
    """
    grp = locality_year.groupby(["municipality", "year"], dropna=False)
    agg = grp.agg(
        production_ton      =("production_ton",       "sum"),
        assisted_trips      =("assisted_trips",        "sum"),
        fleet_monitored     =("fleet_monitored",       "sum"),
        estimated_fishermen =("estimated_fishermen",   "sum"),
        n_locals            =("local",                 "nunique"),
    ).reset_index()

    import numpy as np
    agg["cpue_ton_per_trip"] = np.where(
        agg["assisted_trips"] >= cfg.MIN_TRIPS_CPUE,
        agg["production_ton"] / agg["assisted_trips"],
        np.nan,
    )
    return agg.sort_values(["municipality", "year"]).reset_index(drop=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg.setup_dirs()

    # Load cleaned intermediates
    for name in ["master_clean.csv", "composition_clean.csv", "production_clean.csv",
                 "landings_clean.csv", "socioeconomic_clean.csv", "landing_points_clean.csv"]:
        _check(cfg.DATA_INTERIM / name)

    master = pd.read_csv(cfg.DATA_INTERIM / "master_clean.csv")
    comp   = pd.read_csv(cfg.DATA_INTERIM / "composition_clean.csv")
    prod   = pd.read_csv(cfg.DATA_INTERIM / "production_clean.csv")
    land   = pd.read_csv(cfg.DATA_INTERIM / "landings_clean.csv")
    socio  = pd.read_csv(cfg.DATA_INTERIM / "socioeconomic_clean.csv")
    lp     = pd.read_csv(cfg.DATA_INTERIM / "landing_points_clean.csv")

    # Crosswalks
    xwalk_gear    = build_gear_xwalk(prod)
    xwalk_boat    = build_boat_xwalk(comp)
    xwalk_species = build_species_xwalk(land)
    xwalk_local   = build_local_xwalk(master, lp)
    xwalk_landing = build_landing_xwalk(lp)

    _save(xwalk_gear,    "xwalk_gear.csv")
    _save(xwalk_boat,    "xwalk_boat.csv")
    _save(xwalk_species, "xwalk_species.csv")
    _save(xwalk_local,   "xwalk_local.csv")
    _save(xwalk_landing, "xwalk_landing.csv")

    # Analytical tables
    locality_year     = build_locality_year(master, socio, xwalk_local)
    municipality_year = build_municipality_year(locality_year)

    _save(locality_year,     "locality_year.csv")
    _save(municipality_year, "municipality_year.csv")

    log.info("02_crosswalks.py complete.")


if __name__ == "__main__":
    main()
