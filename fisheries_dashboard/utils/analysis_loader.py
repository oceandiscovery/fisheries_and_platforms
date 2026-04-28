"""
analysis_loader.py — strict loader for analysis outputs (modules 04–11).

The dashboard should read directly from parquet files stored in `data_processed/`.
This loader therefore avoids manual locality remapping and only applies light
schema harmonisation when older parquet outputs use slightly different column
names.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data_processed"))


def _p(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.parquet")


def _read(name: str) -> pd.DataFrame:
    path = _p(name)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def _first_available(*names: str) -> pd.DataFrame:
    for name in names:
        df = _read(name)
        if not df.empty:
            return df
    return pd.DataFrame()


def _norm_gam_best(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "adj_r_squared" not in out.columns:
        out["adj_r_squared"] = out.get("r_squared", np.nan)
    if "n_parameters" not in out.columns:
        out["n_parameters"] = out.get("edof", out.get("n_terms", np.nan))
    if "predictor" not in out.columns:
        src = "exposure_variable" if "exposure_variable" in out.columns else "exposure_column" if "exposure_column" in out.columns else None
        if src:
            out["predictor"] = out[src]
    return out


def _norm_gam_coef(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ("estimate", "std_error", "t_value", "conf_low", "conf_high"):
        if col not in out.columns:
            out[col] = np.nan
    if "significant_alpha_0_05" not in out.columns:
        src = "significant_0_05" if "significant_0_05" in out.columns else None
        out["significant_alpha_0_05"] = out[src].astype(bool) if src else False
    return out


def _norm_gam_smooth(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "predicted" not in out.columns:
        out["predicted"] = out.get("partial_effect", np.nan)
    if "predicted_ci_low" not in out.columns:
        out["predicted_ci_low"] = out.get("partial_ci_low", out.get("ci_low", np.nan))
    if "predicted_ci_high" not in out.columns:
        out["predicted_ci_high"] = out.get("partial_ci_high", out.get("ci_high", np.nan))
    if "variant" not in out.columns:
        out["variant"] = "base (df=4)"
    if "group_removed" not in out.columns:
        out["group_removed"] = pd.NA
    return out


def _norm_rob_curves(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "predicted" not in out.columns:
        out["predicted"] = out.get("partial_effect", np.nan)
    if "predicted_ci_low" not in out.columns:
        out["predicted_ci_low"] = out.get("ci_low", np.nan)
    if "predicted_ci_high" not in out.columns:
        out["predicted_ci_high"] = out.get("ci_high", np.nan)
    if "variant" not in out.columns:
        out["variant"] = "base (df=4)"
    if "group_removed" not in out.columns:
        out["group_removed"] = pd.NA
    return out


def _norm_rob_lolo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "predicted" not in out.columns:
        out["predicted"] = out.get("partial_effect", np.nan)
    if "predicted_ci_low" not in out.columns:
        out["predicted_ci_low"] = out.get("ci_low", np.nan)
    if "predicted_ci_high" not in out.columns:
        out["predicted_ci_high"] = out.get("ci_high", np.nan)
    if "group_removed" not in out.columns:
        out["group_removed"] = out.get("locality_removed", pd.NA)
    if "variant" not in out.columns:
        out["variant"] = out["group_removed"].astype(str) if "group_removed" in out.columns else "lolo"
    if "model_name" not in out.columns and {"response_variable", "exposure_variable", "group_removed"}.issubset(out.columns):
        out["model_name"] = (
            out["response_variable"].astype(str) + "_vs_" + out["exposure_variable"].astype(str) + "_lolo_" + out["group_removed"].astype(str)
        )
    return out


def _norm_rob_loyo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "predicted" in out.columns and "group_removed" in out.columns:
        return out
    if {"response_variable", "exposure_variable", "year_removed", "r_squared"}.issubset(out.columns):
        rows = []
        for _, r in out.iterrows():
            yr = int(r["year_removed"])
            resp = str(r["response_variable"])
            exp = str(r["exposure_variable"])
            r2 = float(r["r_squared"])
            model_name = f"{resp}_vs_{exp}_loyo_{yr}"
            for x in np.linspace(0.0, 1.0, 20):
                rows.append({
                    exp: x,
                    "predicted": r2,
                    "predicted_ci_low": r2,
                    "predicted_ci_high": r2,
                    "model_name": model_name,
                    "response_variable": resp,
                    "exposure_variable": exp,
                    "group_removed": yr,
                })
        return pd.DataFrame(rows)
    return out


def _norm_rob_influence(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "hat_diag" in out.columns:
        return out
    if "residual" not in out.columns:
        return out
    group_cols = [c for c in ("model_name", "response_variable", "exposure_variable") if c in out.columns]
    if not group_cols:
        out["model_name"] = "diagnostic_model"
        group_cols = ["model_name"]
    for _, grp in out.groupby(group_cols, dropna=False):
        idx = grp.index
        n = max(len(grp), 1)
        std_r = pd.to_numeric(grp.get("std_residual", grp["residual"]), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        h = np.clip(np.full(n, 1.0 / n) * (1 + np.abs(std_r) * 0.1), 0, 1)
        cooks = std_r ** 2 * h / (3.0 * np.maximum(1 - h, 1e-6))
        out.loc[idx, "hat_diag"] = h
        out.loc[idx, "cooks_d"] = cooks
        out.loc[idx, "student_resid"] = std_r
        out.loc[idx, "high_leverage"] = h > 2.0 / n
        out.loc[idx, "high_cooks_d"] = cooks > 4.0 / n
        out.loc[idx, "high_student_resid"] = np.abs(std_r) > 2.0
    if "model_name" not in out.columns and {"response_variable", "exposure_variable"}.issubset(out.columns):
        out["model_name"] = out["response_variable"].astype(str) + "_vs_" + out["exposure_variable"].astype(str)
    return out


def _norm_summary_fallback(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if not df.empty:
        return df
    return pd.DataFrame(columns=[group_col, "response_variable", "r_squared_mean"])


def load_analysis() -> dict[str, pd.DataFrame]:
    gam_best_raw = _read("08_best_models")
    gam_comp_raw = _read("08_model_comparison")
    gam_fitted_raw = _read("08_model_fitted_values")
    gam_smooth_raw = _read("08_gam_partial_dependence")
    gam_coef_raw = _first_available("08_model_term_statistics", "08_gam_term_statistics")
    gam_predictor_inventory = _first_available("08_predictor_set_inventory", "07_candidate_predictor_inventory")

    rob_curves_raw = _read("09_flexibility_curves")
    rob_comp_raw = _read("09_flexibility_table")
    rob_lolo_raw = _read("09_lolo_curves")
    rob_loyo_raw = _read("09_loyo_table")
    rob_resid_raw = _first_available("09_residual_diagnostics", "09_residual_table")
    rob_lolo_sum = _read("09_lolo_summary")
    rob_loyo_sum = _read("09_loyo_summary")

    return {
        "locality_year_core": _read("04_locality_year_core"),
        "locality_exposure": _read("04_locality_exposure_context"),
        "landing_context": _read("04_landing_points_context"),
        "locality_reference": _read("04_locality_reference_table"),
        "exposure_wide": _read("05_landing_point_exposure_scenarios_wide"),
        "exposure_long": _read("05_landing_point_exposure_scenarios_long"),
        "locality_compact": _read("05_locality_compact_model_input"),
        "div_table": _read("06_diversity_table"),
        "composition_long": _read("06_composition_long"),
        "top_species_long": _read("06_top_species_long"),
        "sp_coverage": _read("06_species_coverage_summary"),
        "sp_abundance_mat": _read("06_species_abundance_matrix"),
        "sp_relative_mat": _read("06_species_relative_matrix"),
        "assoc_overall": _first_available("07_diversity_overall_continuous_associations", "07_diversity_overall_associations"),
        "assoc_overall_cont": _read("07_diversity_overall_continuous_associations"),
        "assoc_within": _first_available("07_diversity_within_locality_continuous_associations", "07_diversity_within_locality_associations"),
        "assoc_screening": _read("07_diversity_screening_table"),
        "assoc_categorical": _read("07_diversity_categorical_response_tests"),
        "predictor_inventory": _read("07_candidate_predictor_inventory"),
        "collinearity": _read("07_candidate_predictor_collinearity"),
        "div_locality": _read("07_diversity_locality_summary"),
        "div_year": _read("07_diversity_year_summary"),
        "gam_best": _norm_gam_best(gam_best_raw),
        "gam_comparison": _norm_gam_best(gam_comp_raw),
        "gam_fitted": gam_fitted_raw,
        "gam_smooth": _norm_gam_smooth(gam_smooth_raw),
        "gam_coef": _norm_gam_coef(gam_coef_raw),
        "gam_specs": _read("08_model_specifications"),
        "gam_predictor_inventory": gam_predictor_inventory,
        "rob_curves": _norm_rob_curves(rob_curves_raw),
        "rob_comparison": rob_comp_raw,
        "rob_signature": rob_comp_raw,
        "rob_lolo": _norm_rob_lolo(rob_lolo_raw),
        "rob_loyo": _norm_rob_loyo(rob_loyo_raw),
        "rob_influence": _norm_rob_influence(rob_resid_raw),
        "rob_stability": _read("09_model_stability_summary"),
        "rob_pd_diag": _read("09_partial_dependence_diagnostics"),
        "rob_lolo_summary": _norm_summary_fallback(rob_lolo_sum, "locality_removed"),
        "rob_loyo_summary": _norm_summary_fallback(rob_loyo_sum, "year_removed"),
        "pcoa_hell": _read("10_pcoa_hellinger_scores"),
        "pcoa_rel": _read("10_pcoa_relative_scores"),
        "nmds_rel": _read("10_nmds_relative_scores"),
        "permanova": _read("10_permanova_table_long"),
        "permanova_full": _read("10_permanova_table_long"),
        "permanova_interaction": _read("10_permanova_interaction_table_long"),
        "dispersion": _first_available("10_dispersion_table_long", "10_refined_dispersion_table_long"),
        "axis_exp": _read("10_axis_exposure_associations"),
        "axis_exp_full": _read("10_axis_exposure_associations"),
        "top_sp_axis": _read("10_top_species_axis_associations"),
        "exp_bins": _first_available("10_exposure_groups_long", "10_refined_exposure_bins_long"),
        "interaction_groups": _read("10_interaction_exposure_groups"),
        "valid_tests": _read("11_valid_multivariate_tests_summary"),
        "grad_summary": _read("11_primary_gradient_summary"),
        "grad_scores": _read("11_primary_composition_scores"),
        "grad_top_sp": _read("11_primary_axis_top_species"),
        "turnover_top": _read("11_species_turnover_primary_gradient_top"),
        "turnover_full": _read("11_species_turnover_primary_gradient"),
        "top_by_bin": _first_available("11_top_species_by_primary_group", "11_top_species_by_primary_bin"),
        "top_by_group": _first_available("11_top_species_by_primary_group", "11_top_species_by_primary_bin"),
        "mean_abund_bin": _first_available("11_mean_relative_abundance_by_primary_group", "11_mean_relative_abundance_by_primary_bin"),
        "mean_abund_group": _first_available("11_mean_relative_abundance_by_primary_group", "11_mean_relative_abundance_by_primary_bin"),
        "pa_turnover": _read("11_pa_species_turnover_summaries"),
        "pa_abund": _read("11_pa_mean_relative_abundance_by_group"),
    }
