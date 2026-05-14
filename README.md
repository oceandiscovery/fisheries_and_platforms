# Fish catches × Oil platforms — Rio Grande do Norte, Brazil

**Research question:** How are small-scale fisheries productivity, effort structure, and catch composition associated with spatial exposure to offshore oil platforms and marine protected areas over time?

**Study area:** Rio Grande do Norte, Brazil  
**Data:** PMDP fisheries monitoring programme, 2001–2025  
**Localities:** 8 coastal fishing communities (7 primary analysis + Areia Branca case study)

---

## Running the app

```bash
pip install -r requirements.txt
streamlit run app_explorer.py
```

---

## Repository contents

```
.
├── app_explorer.py          Interactive Streamlit data explorer
├── requirements.txt         Python dependencies
│
├── py_data/
│   ├── interim/             Cleaned and spatially-enriched source tables
│   └── processed/           Analytical outputs (CPUE, diversity, models, species)
│
└── .streamlit/
    └── config.toml          Streamlit server configuration
```

---

## Data: `py_data/interim/`

Cleaned and crosswalked tables derived from the PMDP database and spatial layers.

| File | Description |
|------|-------------|
| `master_clean.csv` | Locality × year summary (production, effort, fleet) |
| `composition_clean.csv` | Vessel type × locality × year |
| `production_clean.csv` | Gear × locality × year production |
| `landings_clean.csv` | Species × locality × year landings |
| `socioeconomic_clean.csv` | Fishermen and vessel metrics |
| `landing_points_clean.csv` | Landing point coordinates (lat/lon) |
| `landing_points_exposure.csv` | Landing points enriched with platform and MPA distances |
| `local_exposure.csv` | Locality-level spatial exposure summary |
| `locality_year.csv` | Balanced locality × year panel |
| `municipality_year.csv` | Municipality × year aggregation |
| `xwalk_gear.csv` | Gear type crosswalk |
| `xwalk_boat.csv` | Vessel type crosswalk |
| `xwalk_species.csv` | Species name crosswalk |
| `xwalk_local.csv` / `xwalk_local_updated.csv` | Locality name crosswalk |
| `xwalk_landing.csv` | Landing point crosswalk |
| `mpas_combined.gpkg` | MPA boundaries (RDS Ponta do Tubarão + APA Dunas do Rosado) |
| `platforms_projected.gpkg` | Offshore oil/gas platform locations |

---

## Data: `py_data/processed/`

Analytical tables ready for exploration and modelling.

### Productivity and CPUE
| File | Description |
|------|-------------|
| `productivity_local_year.csv` | CPUE, richness, Shannon H', Pielou J' by locality × year |
| `productivity_platform_year.csv` | Aggregated by platform exposure class × year |
| `productivity_mpa_year.csv` | Aggregated by MPA exposure class × year |
| `productivity_combined_year.csv` | Aggregated by combined exposure class × year |
| `productivity_municipality_year.csv` | Aggregated by municipality × year |
| `productivity_local_period.csv` | Locality means by time period |
| `cpue_std_data.csv` / `cpue_std_index.csv` | GAM-standardised CPUE index |
| `cpue_std_gam_partial.csv` / `cpue_std_gam_summary.csv` | GAM partial effects |
| `cpue_std_lmm.csv` | LMM coefficients for CPUE standardisation |
| `cpue_three_way_index.csv` | CPUE comparison: raw, per fisherman, standardised |

### Temporal dynamics
| File | Description |
|------|-------------|
| `timeseries_regional.csv` | Regional production and CPUE time series |
| `timeseries_platform.csv` | Time series by platform exposure class |
| `timeseries_mpa.csv` | Time series by MPA exposure class |
| `timeseries_combined.csv` | Time series by combined exposure class |
| `timeseries_gear_year.csv` | Gear composition over time |
| `timeseries_boat_year.csv` | Vessel composition over time |
| `timeseries_species_top20.csv` | Top-20 species production over time |
| `period_comparison.csv` | CPUE and diversity by exposure class × period |

### CPUE and diversity trends (research questions P1–P9)
| File | Description |
|------|-------------|
| `spatial_p1_local_means.csv` / `spatial_p1_spearman.csv` | P1: spatial gradients |
| `cpue_p2_mannkendall.csv` / `cpue_p2_gam_trend.csv` | P2: CPUE temporal trends |
| `diversity_p3a_mannkendall.csv` | P3a: diversity temporal trends |
| `effort_p3b_mannkendall.csv` | P3b: gear/vessel composition trends |
| `cpue_p4_trajectories.csv` / `cpue_p4_lmm.csv` | P4: CPUE trajectories by MPA status |
| `cpue_p4_trajectories_n8.csv` / `cpue_p4_lmm_n8.csv` | P4 sensitivity (n=8 incl. Areia Branca) |
| `diversity_p5_trajectories.csv` / `diversity_p5_lmm.csv` | P5: diversity trajectories by MPA status |
| `diversity_p5_trajectories_n8.csv` / `diversity_p5_lmm_n8.csv` | P5 sensitivity (n=8) |
| `effort_p6_spearman.csv` / `effort_p6_lmm.csv` | P6: CPUE ~ gear/vessel composition |
| `effort_p6_gam_partial.csv` / `effort_p6_year_smooth.csv` | P6 GAM partial effects |
| `within_p7_density_dep.csv` | P7: density dependence (CPUE ~ fishermen) |
| `within_p8_spearman.csv` / `within_p8_gam_partial.csv` | P8: CPUE ~ diversity within localities |
| `within_p8_year_smooth.csv` | P8 year smooth |

### Gear and vessel structure
| File | Description |
|------|-------------|
| `gear_year.csv` / `boat_year.csv` | Gear and vessel counts by year |
| `gear_share_local_year.csv` / `boat_share_local_year.csv` | Shares by locality × year |
| `gear_share_platform_year.csv` / `gear_share_mpa_year.csv` | Shares by exposure class |
| `boat_share_platform_year.csv` / `boat_share_mpa_year.csv` | Vessel shares by exposure class |
| `gear_share_combined_year.csv` / `boat_share_combined_year.csv` | Shares by combined class |
| `gear_shares_exposure.csv` / `boat_shares_exposure.csv` | Long-format exposure summaries |

### Species composition
| File | Description |
|------|-------------|
| `species_year.csv` | Species-level production by year |
| `species_share_local_year.csv` | Species shares by locality × year |
| `species_share_platform_year.csv` / `species_share_mpa_year.csv` | Shares by exposure class |
| `species_share_gear_year.csv` / `species_share_boat_year.csv` | Shares by gear/vessel type |
| `species_share_combined_year.csv` | Shares by combined exposure class |
| `species_rank_platform.csv` / `species_rank_mpa.csv` / `species_rank_combined.csv` | Species rankings by exposure class |
| `species_SIMPER_platform.csv` / `species_SIMPER_mpa.csv` / `species_SIMPER_combined.csv` | SIMPER dissimilarity contributions |
| `species_p9_simper_n7.csv` / `species_p9_simper_ab.csv` | P9a: SIMPER (n=7 and Areia Branca) |
| `species_p9_species_mk.csv` / `species_p9_species_mk_ab.csv` | P9b: Mann-Kendall per species |
| `species_catch_matrix_local_year.csv` | Catch matrix (wide) by locality × year |
| `species_catch_wide_local_year.csv` / `species_catch_wide_combined.csv` | Wide-format catch tables |
| `species_catch_wide_platform.csv` / `species_catch_wide_mpa.csv` | Catch matrices by exposure class |
| `species_rel_wide_local_year.csv` | Relative abundance matrix |
| `species_turnover_period.csv` | Bray-Curtis compositional turnover between periods |

### Areia Branca case study
| File | Description |
|------|-------------|
| `ab_case_metrics.csv` | Key metrics time series for Areia Branca vs. Porto do Mangue |
| `ab_case_species_top.csv` | Top species comparison between localities |

### Statistical models
| File | Description |
|------|-------------|
| `models_mann_kendall.csv` / `models_mann_kendall_local.csv` | Mann-Kendall trend tests |
| `models_spearman.csv` | Spearman correlations |
| `models_kruskal.csv` / `models_mann_whitney_pairwise.csv` | Non-parametric group tests |
| `models_permanova.csv` | PERMANOVA for species composition |
| `models_gear_mk.csv` / `models_boat_mk.csv` | Mann-Kendall for gear/vessel trends |
| `models_gear_kruskal.csv` / `models_gear_mann_whitney.csv` | Gear group comparisons |
| `models_boat_kruskal.csv` / `models_boat_mann_whitney.csv` | Vessel group comparisons |
| `models_gear_permanova.csv` | PERMANOVA for gear composition |

### Quality control
| File | Description |
|------|-------------|
| `diagnostics_summary.csv` | Pipeline QC check results |
| `diagnostics_report.txt` | Full diagnostics report |
| `production_reconciliation.csv` | Production cross-table reconciliation |

---

## Spatial exposure classes

### Platform distance
| Class | Range |
|-------|-------|
| `0-20 km` | Very near |
| `20-50 km` | Near |
| `50-100 km` | Intermediate |
| `100-200 km` | Far |
| `>200 km` | Very far |

### MPA exposure
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

## MPAs
- **RDS Ponta do Tubarão** — Reserva de Desenvolvimento Sustentável
- **APA Dunas do Rosado** — Área de Proteção Ambiental
