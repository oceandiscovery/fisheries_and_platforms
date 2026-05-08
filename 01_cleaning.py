"""
01_cleaning.py
==============
Loads, cleans, and harmonises all five sheets from PMDP_DATABASE_clean.xlsx
plus local_landing_points.xlsx.

Outputs (data/interim/):
  - master_clean.csv            PMDP_MASTER cleaned
  - composition_clean.csv       Boat-type composition cleaned
  - production_clean.csv        Gear-level production cleaned
  - landings_clean.csv          Species landings cleaned
  - socioeconomic_clean.csv     Socioeconomic indicators cleaned
  - landing_points_clean.csv    Landing points with coordinates

Run:
  python 01_cleaning.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_00 as cfg  # noqa: E402 (import after sys.path insert)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format=cfg.LOG_FORMAT,
    datefmt=cfg.LOG_DATEFMT,
    handlers=[
        logging.FileHandler(cfg.LOGS / "01_cleaning.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("01_cleaning")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_file(path: Path) -> None:
    if not path.exists():
        log.error("Required input not found: %s", path)
        sys.exit(1)


def _normalise_local(series: pd.Series) -> pd.Series:
    """Uppercase, strip accents and extra spaces from locality names."""
    import unicodedata

    def clean(s: str) -> str:
        s = str(s).strip().upper()
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return " ".join(s.split())

    return series.map(clean)


def _save(df: pd.DataFrame, name: str) -> None:
    out = cfg.DATA_INTERIM / name
    df.to_csv(out, index=False)
    log.info("Saved %s  (%d rows × %d cols)", out.name, len(df), df.shape[1])


# ─── Sheet loaders ────────────────────────────────────────────────────────────

def load_master(xl: pd.ExcelFile) -> pd.DataFrame:
    """
    PMDP_MASTER: local × year summary.
    Columns: local, year, fleet_monitored, assisted_trips,
             estimated_fishermen, production_ton
    """
    df = xl.parse(cfg.SHEETS["master"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename to canonical names if needed
    rename = {
        "fleet_monitored":    "fleet_monitored",
        "assisted_trips":     "assisted_trips",
        "estimated_fishermen":"estimated_fishermen",
        "production_ton":     "production_ton",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["local"] = _normalise_local(df["local"])
    df["year"]  = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Drop rows missing key identifiers or with negative production
    n0 = len(df)
    df = df.dropna(subset=["local", "year"])
    df = df[df["production_ton"] >= 0]
    log.info("master: dropped %d rows with missing/negative values", n0 - len(df))

    # Numeric coercion
    for col in ["fleet_monitored", "assisted_trips", "estimated_fishermen", "production_ton"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived field: CPUE at master level
    df["cpue_ton_per_trip"] = np.where(
        df["assisted_trips"] >= cfg.MIN_TRIPS_CPUE,
        df["production_ton"] / df["assisted_trips"],
        np.nan,
    )

    return df.sort_values(["local", "year"]).reset_index(drop=True)


def load_composition(xl: pd.ExcelFile) -> pd.DataFrame:
    """
    PMDP_COMPOSITION: boat-type × local × year.
    Columns: local, year, boat_type, vessels_monitored, fleet_production_ton
    """
    df = xl.parse(cfg.SHEETS["composition"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["local"]     = _normalise_local(df["local"])
    df["year"]      = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["boat_type"] = df["boat_type"].str.strip().str.upper()

    for col in ["vessels_monitored", "fleet_production_ton"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["local", "year", "boat_type"])

    # Derived: CPUE by boat type — requires assisted_trips join later;
    # flag vessel-years with zero fleet production
    df["zero_production_flag"] = df["fleet_production_ton"].fillna(0) == 0

    return df.sort_values(["local", "year", "boat_type"]).reset_index(drop=True)


def load_production(xl: pd.ExcelFile) -> pd.DataFrame:
    """
    PMDP_PRODUCTION: gear × local × year.
    Columns: local, year, gear_cod, gear_type, gear_group, gear_production_ton
    """
    df = xl.parse(cfg.SHEETS["production"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["local"]    = _normalise_local(df["local"])
    df["year"]     = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["gear_cod"] = df["gear_cod"].str.strip().str.upper()
    df["gear_type"]= df["gear_type"].str.strip().str.lower()  # passive/active/mixed

    df["gear_production_ton"] = pd.to_numeric(df["gear_production_ton"], errors="coerce")
    df = df[df["gear_production_ton"] >= 0]
    df = df.dropna(subset=["local", "year", "gear_cod"])

    return df.sort_values(["local", "year", "gear_cod"]).reset_index(drop=True)


def load_landings(xl: pd.ExcelFile) -> pd.DataFrame:
    """
    PMDP_LANDINGS: species × local × year.
    Columns: local, year, species, sp_production_ton
    """
    df = xl.parse(cfg.SHEETS["landings"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["local"]   = _normalise_local(df["local"])
    df["year"]    = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["species"] = df["species"].str.strip().str.title()

    df["sp_production_ton"] = pd.to_numeric(df["sp_production_ton"], errors="coerce")
    df = df[df["sp_production_ton"] >= 0]
    df = df.dropna(subset=["local", "year", "species"])

    return df.sort_values(["local", "year", "species"]).reset_index(drop=True)


def load_socioeconomic(xl: pd.ExcelFile) -> pd.DataFrame:
    """
    PMDP_SOCIOECONOMIC: local × year socioeconomic summary.
    Columns: local, year, fleet_monitored, estimated_fishermen, fishermen_per_vessel
    """
    df = xl.parse(cfg.SHEETS["socioeconomic"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["local"] = _normalise_local(df["local"])
    df["year"]  = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    for col in ["fleet_monitored", "estimated_fishermen", "fishermen_per_vessel"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["local", "year"])
    return df.sort_values(["local", "year"]).reset_index(drop=True)


def load_landing_points() -> pd.DataFrame:
    """
    local_landing_points.xlsx:
    Columns: local, landing_point_mean, latitude, longitude
    """
    path = cfg.FILES["landing_pts"]
    _check_file(path)
    df = pd.read_excel(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["local"]             = _normalise_local(df["local"])
    df["landing_point_mean"]= df["landing_point_mean"].str.strip()
    df["latitude"]          = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"]         = pd.to_numeric(df["longitude"], errors="coerce")

    n0 = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    log.info("landing_points: %d points loaded (%d dropped for missing coords)",
             len(df), n0 - len(df))

    return df.sort_values(["local", "landing_point_mean"]).reset_index(drop=True)


# ─── Quality checks ───────────────────────────────────────────────────────────

def qc_report(master: pd.DataFrame, comp: pd.DataFrame,
              prod: pd.DataFrame, land: pd.DataFrame,
              socio: pd.DataFrame, lp: pd.DataFrame) -> None:
    """Log basic QC statistics across all tables."""
    log.info("=== QC REPORT ===")
    for name, df in [("master", master), ("composition", comp),
                     ("production", prod), ("landings", land),
                     ("socioeconomic", socio), ("landing_points", lp)]:
        log.info(
            "%-20s  rows=%-6d  cols=%-3d  years=%s",
            name, len(df), df.shape[1],
            str(sorted(df["year"].dropna().unique().tolist()))
            if "year" in df.columns else "n/a",
        )
        if "local" in df.columns:
            log.info("  └─ localities: %s", sorted(df["local"].unique().tolist()))

    # Cross-check: all locals in master should have landing-point coords
    master_locals = set(master["local"].unique())
    lp_locals     = set(lp["local"].unique())
    missing_coords = master_locals - lp_locals
    if missing_coords:
        log.warning(
            "Localities in master WITHOUT landing-point coordinates: %s",
            sorted(missing_coords),
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg.setup_dirs()
    _check_file(cfg.FILES["db"])

    log.info("Loading PMDP_DATABASE_clean.xlsx …")
    xl = pd.ExcelFile(cfg.FILES["db"])

    master = load_master(xl)
    comp   = load_composition(xl)
    prod   = load_production(xl)
    land   = load_landings(xl)
    socio  = load_socioeconomic(xl)
    lp     = load_landing_points()

    qc_report(master, comp, prod, land, socio, lp)

    _save(master, "master_clean.csv")
    _save(comp,   "composition_clean.csv")
    _save(prod,   "production_clean.csv")
    _save(land,   "landings_clean.csv")
    _save(socio,  "socioeconomic_clean.csv")
    _save(lp,     "landing_points_clean.csv")

    log.info("01_cleaning.py complete.")


if __name__ == "__main__":
    main()
