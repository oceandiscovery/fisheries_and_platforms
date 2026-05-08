"""
00_config.py
============
Central configuration for the pipeline:
  "Fish catches and oil platforms Brazil"
  Focus: small-scale fisheries × offshore platforms × MPAs — Rio Grande do Norte

All paths, constants, CRS definitions, and exposure thresholds live here.
No data processing in this module.
"""

from pathlib import Path

# ─── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ─── Input data ───────────────────────────────────────────────────────────────
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
LOGS = ROOT / "logs"

# Raw source files (place all inputs in data/raw/)
FILES = {
    "db":           DATA_RAW / "PMDP_DATABASE_clean.xlsx",
    "landing_pts":  DATA_RAW / "local_landing_points.xlsx",
    "lp_shp":       DATA_RAW / "pontos_desembarque_pmdp.zip",
    "municipalities": DATA_RAW / "RN_Municipios_2025.zip",
    "platforms":    DATA_RAW / "platforms_rio_grande_do_norte.zip",
    "rds":          DATA_RAW / "rds_ponta_tubarao_sirgas2000.zip",
    "apa":          DATA_RAW / "apa_dunas_rosado_sirgas2000.zip",
}

# ─── Sheet names in PMDP_DATABASE_clean.xlsx ─────────────────────────────────
SHEETS = {
    "master":       "PMDP_MASTER",        # local × year summary
    "composition":  "FLEET_COMPOSITION",   # boat type × local × year
    "production":   "GEAR_PRODUCTION",    # gear × local × year
    "landings":     "SPECIES_LANDING",       # species × local × year
    "socioeconomic":"PMDP_MASTER",   # fishermen/vessel metrics (merged into master sheet)
}

# ─── Coordinate reference systems ─────────────────────────────────────────────
CRS_GEO  = "EPSG:4674"   # SIRGAS 2000 geographic — source CRS for all layers
CRS_PROJ = "EPSG:31984"  # SIRGAS 2000 / UTM zone 24S — projected (metres) for distances

# ─── Spatial exposure thresholds ─────────────────────────────────────────────
# Platform distance classes (km)
PLATFORM_BREAKS_KM = [0, 20, 50, 100, 200, float("inf")]
PLATFORM_LABELS    = ["0-20 km", "20-50 km", "50-100 km", "100-200 km", ">200 km"]

# MPA distance classes (km)
MPA_BREAKS_KM = [0, 10, 25, 50, float("inf")]
MPA_LABELS    = ["0-10 km", "10-25 km", "25-50 km", ">50 km"]

# MPA names
MPA_RDS = "RDS Ponta do Tubarão"
MPA_APA = "APA Dunas do Rosado"

# ─── Analysis settings ────────────────────────────────────────────────────────
MIN_TRIPS_CPUE   = 5      # minimum assisted_trips to include a local-year in CPUE
MIN_RECORDS_COMP = 3      # minimum records for composition analyses
DIVERSITY_METHOD = "all"  # compute Shannon H', Simpson, Pielou J' and richness
CPUE_UNIT        = "ton/trip"

# ─── Temporal windows ─────────────────────────────────────────────────────────
PERIOD_BREAKS = {
    "early":  (2001, 2008),
    "middle": (2009, 2015),
    "recent": (2016, 2025),
}

# ─── Colour palettes (used by figures script) ────────────────────────────────
PALETTE_PLATFORM = {
    "0-20 km":    "#d73027",
    "20-50 km":   "#fc8d59",
    "50-100 km":  "#fee08b",
    "100-200 km": "#91bfdb",
    ">200 km":    "#4575b4",
}
PALETTE_MPA = {
    "inside":     "#1a9850",
    "0-10 km":    "#91cf60",
    "10-25 km":   "#d9ef8b",
    "25-50 km":   "#fee08b",
    ">50 km":     "#d73027",
}

# ─── Logging format ───────────────────────────────────────────────────────────
LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_dirs() -> None:
    """Create all required directories if they do not exist."""
    for d in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, OUTPUTS, FIGURES, LOGS]:
        d.mkdir(parents=True, exist_ok=True)