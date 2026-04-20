"""
analysis_loader.py — Carga los datasets de análisis (módulos 04–11).
Resuelve la ruta hacia /data/ que está al mismo nivel que fisheries_dashboard/.
"""

import os
import pandas as pd

# data/ está dos niveles arriba de utils/
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def _p(name):
    return os.path.join(DATA_DIR, f"{name}.parquet")


@pd.core.common.cache_readonly if False else lambda f: f  # no-op decorator placeholder
def load_analysis():
    """Carga todos los datasets de análisis en un dict de DataFrames."""
    return {
        # ── Módulo 06: Diversidad y matrices de composición ──
        "div_table":        pd.read_parquet(_p("06_diversity_table")),
        "composition_long": pd.read_parquet(_p("06_composition_long")),
        "top_species_long": pd.read_parquet(_p("06_top_species_long")),
        "sp_coverage":      pd.read_parquet(_p("06_species_coverage_summary")),
        "sp_abundance_mat": pd.read_parquet(_p("06_species_abundance_matrix")),
        "sp_relative_mat":  pd.read_parquet(_p("06_species_relative_matrix")),

        # ── Módulo 07: Asociaciones diversidad ↔ exposición ──
        "assoc_overall":    pd.read_parquet(_p("07_diversity_overall_associations")),
        "assoc_within":     pd.read_parquet(_p("07_diversity_within_locality_associations")),
        "assoc_screening":  pd.read_parquet(_p("07_diversity_screening_table")),
        "div_locality":     pd.read_parquet(_p("07_diversity_locality_summary")),
        "div_year":         pd.read_parquet(_p("07_diversity_year_summary")),

        # ── Módulo 08: Modelos GAM (pyGAM — LinearGAM + GCV) ──
        "gam_best":         pd.read_parquet(_p("08_best_gam_linear_models")),
        "gam_coef":         pd.read_parquet(_p("08_gam_linear_coefficients")),
        "gam_fitted":       pd.read_parquet(_p("08_gam_linear_fitted_values")),
        "gam_smooth":       pd.read_parquet(_p("08_gam_linear_smooth_predictions")),
        "gam_comparison":   pd.read_parquet(_p("08_gam_linear_model_comparison")),

        # ── Módulo 09: Diagnóstico GAM (flexibilidad, LOLO, LOYO, residuos) ──
        "rob_curves":       pd.read_parquet(_p("09_gam_alternative_curves")),
        "rob_comparison":   pd.read_parquet(_p("09_gam_alternative_model_comparison")),
        "rob_signature":    pd.read_parquet(_p("09_gam_curve_signature_table")),
        "rob_influence":    pd.read_parquet(_p("09_gam_influence_table")),
        "rob_lolo":         pd.read_parquet(_p("09_gam_leave_one_locality_out_curves")),
        "rob_loyo":         pd.read_parquet(_p("09_gam_leave_one_year_out_curves")),
        "rob_lolo_summary": pd.read_parquet(_p("09_gam_leave_one_locality_out_summary")),
        "rob_loyo_summary": pd.read_parquet(_p("09_gam_leave_one_year_out_summary")),

        # ── Módulo 10: Ordenación multivariante ──
        "pcoa_hell":        pd.read_parquet(_p("10_refined_pcoa_hellinger_scores")),
        "pcoa_rel":         pd.read_parquet(_p("10_refined_pcoa_relative_scores")),
        "nmds_rel":         pd.read_parquet(_p("10_refined_nmds_relative_scores")),
        "permanova":        pd.read_parquet(_p("10_refined_permanova_table_long")),
        "dispersion":       pd.read_parquet(_p("10_refined_dispersion_table_long")),
        "axis_exp":         pd.read_parquet(_p("10_refined_axis_exposure_associations")),
        "top_sp_axis":      pd.read_parquet(_p("10_refined_top_species_axis_associations")),
        "exp_bins":         pd.read_parquet(_p("10_refined_exposure_bins_long")),

        # ── Módulo 11: Gradiente primario ──
        "grad_summary":     pd.read_parquet(_p("11_primary_gradient_summary")),
        "grad_scores":      pd.read_parquet(_p("11_primary_composition_scores")),
        "grad_top_sp":      pd.read_parquet(_p("11_primary_axis_top_species")),
        "turnover_top":     pd.read_parquet(_p("11_species_turnover_primary_gradient_top")),
        "turnover_full":    pd.read_parquet(_p("11_species_turnover_primary_gradient")),
        "top_by_bin":       pd.read_parquet(_p("11_top_species_by_primary_bin")),
        "mean_abund_bin":   pd.read_parquet(_p("11_mean_relative_abundance_by_primary_bin")),
    }
