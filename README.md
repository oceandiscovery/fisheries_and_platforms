# Fish catches × Oil platforms Brazil — Analytical Pipeline

## Project
**Research question:** How are small-scale fisheries productivity, effort structure, and catch composition associated with spatial exposure to offshore oil platforms and marine protected areas over time?

**Study area:** Rio Grande do Norte, Brazil  
**Data:** PMDP monitoring programme, 2001–2025

---

## Repository structure

```
pipeline_fish_brazil/
│
├── 00_config.py                   Central configuration (paths, CRS, thresholds)
├── 01_cleaning.py                 Load and clean all 5 DB sheets + landing points
├── 02_crosswalks.py               Build all reference crosswalk tables
├── 03_spatial_exposure.py         Compute platform/MPA distance and exposure classes
├── 04_productivity_diversity.py   CPUE, richness, Shannon H', Simpson, Pielou J'
├── 05_effort_structure.py         Gear shares, boat shares, effort composition
├── 06_species_composition.py      Species matrices, SIMPER, turnover
├── 07_species_shares.py           Species-level proportional contributions
├── 08_temporal_dynamics.py        Time series and period comparisons
├── 09_diagnostics.py              QC checks and production reconciliation
├── 10_figures.py                  All maps, plots, and visualisations
│
├── run_pipeline.py                Master runner script
│
├── data/
│   ├── raw/                       ← Place all source files here
│   ├── interim/                   Cleaned and intermediate files
│   └── processed/                 Final analytical outputs
│
├── outputs/
│   └── figures/                   All figures (F01–F21)
│
└── logs/                          Per-script log files
```

---

## Source data (place in `data/raw/`)

| File | Description |
|------|-------------|
| `PMDP_DATABASE_clean.xlsx` | Main fisheries database (5 sheets) |
| `local_landing_points.xlsx` | Landing point coordinates |
| `pontos_desembarque_pmdp.zip` | Spatial layer of landing points |
| `RN_Municipios_2025.zip` | Rio Grande do Norte municipalities |
| `platforms_rio_grande_do_norte.zip` | Offshore oil/gas platforms |
| `rds_ponta_tubarao_sirgas2000.zip` | RDS Ponta do Tubarão MPA |
| `apa_dunas_rosado_sirgas2000.zip` | APA Dunas do Rosado MPA |

### Database sheets

| Sheet | Columns |
|-------|---------|
| `PMDP_MASTER` | local, year, fleet_monitored, assisted_trips, estimated_fishermen, production_ton |
| `PMDP_COMPOSITION` | local, year, boat_type, vessels_monitored, fleet_production_ton |
| `PMDP_PRODUCTION` | local, year, gear_cod, gear_type, gear_group, gear_production_ton, gear_cod_original |
| `PMDP_LANDINGS` | local, year, species, sp_production_ton |
| `PMDP_SOCIOECONOMIC` | local, year, fleet_monitored, estimated_fishermen, fishermen_per_vessel |

---

## Installation

```bash
pip install pandas numpy geopandas matplotlib scipy openpyxl
```

---

## Running the pipeline

### Full pipeline
```bash
python run_pipeline.py
```

### Re-run from a specific step
```bash
python run_pipeline.py --from 4
```

### Run a single script
```bash
python run_pipeline.py --only 10
python 04_productivity_diversity.py   # or run directly
```

---

## CRS policy

| Purpose | EPSG | Description |
|---------|------|-------------|
| Source / display | 4674 | SIRGAS 2000 geographic |
| Distance calculations | 31984 | SIRGAS 2000 / UTM zone 24S (metres) |

All spatial operations are performed in EPSG:31984. Results are stored in metres and kilometres.

---

## Exposure class definitions

### Platform distance classes
| Class | Range |
|-------|-------|
| `0-20 km` | Very near |
| `20-50 km` | Near |
| `50-100 km` | Intermediate |
| `100-200 km` | Far |
| `>200 km` | Very far |

### MPA exposure classes
| Class | Description |
|-------|-------------|
| `inside` | Within MPA boundary |
| `0-10 km` | Very near MPA |
| `10-25 km` | Near MPA |
| `25-50 km` | Intermediate |
| `>50 km` | Far from MPA |

### Combined class
`platform_class × mpa_class` — e.g. `"0-20 km × inside"`.

---

## Key outputs

### Cleaned tables (`data/interim/`)
- `master_clean.csv`, `composition_clean.csv`, `production_clean.csv`
- `landings_clean.csv`, `socioeconomic_clean.csv`, `landing_points_clean.csv`
- `xwalk_gear.csv`, `xwalk_boat.csv`, `xwalk_species.csv`, `xwalk_local.csv`
- `locality_year.csv`, `municipality_year.csv`
- `landing_points_exposure.csv`, `local_exposure.csv`

### Analytical tables (`data/processed/`)
- `productivity_local_year.csv` — CPUE, richness, H', J' by local × year
- `productivity_platform_year.csv` / `_mpa_year.csv` / `_combined_year.csv`
- `gear_share_*.csv`, `boat_share_*.csv` — effort structure
- `species_catch_matrix_*.csv`, `species_SIMPER_*.csv` — composition
- `species_share_*.csv`, `species_rank_*.csv` — species shares
- `timeseries_*.csv`, `period_comparison.csv` — temporal dynamics
- `diagnostics_summary.csv`, `production_reconciliation.csv` — QC

### Figures (`outputs/figures/`)
| Code | Description |
|------|-------------|
| F01 | Study area map |
| F02 | Platform distance gradient map |
| F03 | MPA exposure map |
| F04 | CPUE × time by platform class |
| F05 | CPUE × time by MPA class |
| F06 | CPUE boxplot by platform class |
| F08 | Shannon H' × time by platform class |
| F10 | Gear composition stacked area |
| F11 | Gear shares by platform class |
| F13 | Boat composition stacked area |
| F15 | Top-10 species share area chart |
| F17 | SIMPER bar chart — platform differences |
| F18 | Bray-Curtis turnover between periods |
| F19 | Regional production time series |
| F20 | CPUE multi-panel (regional / platform / MPA) |
| F21 | CPUE heatmap: platform × period |

---

## Configuration

All key parameters are in `00_config.py`:
- `MIN_TRIPS_CPUE = 5` — minimum trips to compute CPUE
- `PERIOD_BREAKS` — temporal period definitions
- `PLATFORM_BREAKS_KM`, `MPA_BREAKS_KM` — exposure thresholds
- `DPI = 300` — figure resolution

---

## Notes
- All scripts are idempotent: re-running overwrites existing outputs cleanly.
- Each script checks for its required inputs and exits with an error if any are missing.
- Logs are written to `logs/` alongside console output.
- The `combined_exposure_class` variable allows analysis of platform–MPA interaction contexts without fitting an explicit interaction term.
