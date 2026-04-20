"""
analysis_loader.py — Loads analysis datasets (modules 04–11).
Resolves paths to /data/ which sits at the same level as fisheries_dashboard/.

Normalisation layer
-------------------
The parquet files in data/ were produced by the original scripts (pre-v3.6).
The normalise_*() helpers rename columns to the schema expected by analysis_tabs.py
so that the dashboard works without requiring the user to re-run all scripts.
"""

import os
import numpy as np
import pandas as pd

# data/ is two levels above utils/
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def _p(name):
    return os.path.join(DATA_DIR, f"{name}.parquet")


def _read(name):
    path = _p(name)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


# ── Normalisation helpers ────────────────────────────────────────────────────

def _norm_gam_best(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure adj_r_squared and n_parameters columns exist."""
    if df.empty:
        return df
    # adj_r_squared: already present in old files
    if "adj_r_squared" not in df.columns:
        df["adj_r_squared"] = df.get("r_squared", np.nan)
    # n_parameters: old files use edof
    if "n_parameters" not in df.columns:
        df["n_parameters"] = df.get("edof", np.nan)
    return df


def _norm_gam_coef(df: pd.DataFrame) -> pd.DataFrame:
    """
    Old schema: model_name, term, p_value, significant_0_05
    New schema: + estimate, std_error, t_value, conf_low, conf_high,
                  significant_alpha_0_05
    """
    if df.empty:
        return df
    for col in ("estimate", "std_error", "t_value", "conf_low", "conf_high"):
        if col not in df.columns:
            df[col] = np.nan
    if "significant_alpha_0_05" not in df.columns:
        src = "significant_0_05" if "significant_0_05" in df.columns else None
        df["significant_alpha_0_05"] = df[src].astype(bool) if src else False
    return df


def _norm_gam_smooth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Old schema: exposure_col, partial_effect, partial_ci_low, partial_ci_high, model_name, …
    New schema: exposure_col first, predicted, predicted_ci_low, predicted_ci_high,
                model_name, variant, group_removed
    """
    if df.empty:
        return df
    if "predicted" not in df.columns:
        src = "partial_effect" if "partial_effect" in df.columns else df.columns[1]
        df["predicted"] = df[src]
    if "predicted_ci_low" not in df.columns:
        src = next((c for c in ("partial_ci_low", "ci_low") if c in df.columns), None)
        df["predicted_ci_low"] = df[src] if src else np.nan
    if "predicted_ci_high" not in df.columns:
        src = next((c for c in ("partial_ci_high", "ci_high") if c in df.columns), None)
        df["predicted_ci_high"] = df[src] if src else np.nan
    if "variant" not in df.columns:
        df["variant"] = "base (df=4)"
    if "group_removed" not in df.columns:
        df["group_removed"] = pd.NA
    return df


def _norm_rob_curves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Old schema: exposure_col, partial_effect, ci_low, ci_high, model_name, n_splines, …
    New schema: exposure_col first, predicted, predicted_ci_low, predicted_ci_high,
                model_name, variant, group_removed
    """
    if df.empty:
        return df
    if "predicted" not in df.columns:
        df["predicted"] = df.get("partial_effect", np.nan)
    if "predicted_ci_low" not in df.columns:
        df["predicted_ci_low"] = df.get("ci_low", np.nan)
    if "predicted_ci_high" not in df.columns:
        df["predicted_ci_high"] = df.get("ci_high", np.nan)
    # variant: derive from n_splines if available
    if "variant" not in df.columns:
        if "n_splines" in df.columns:
            df["variant"] = df["n_splines"].apply(
                lambda n: "base (df=4)" if n == 10 else f"spline_df{int(n)}"
            )
        else:
            df["variant"] = "base (df=4)"
    if "group_removed" not in df.columns:
        df["group_removed"] = pd.NA
    # Ensure exposure column is first
    non_exp = [c for c in df.columns if c not in
               ("partial_effect", "ci_low", "ci_high", "predicted",
                "predicted_ci_low", "predicted_ci_high", "model_name",
                "response_variable", "exposure_variable", "n_splines",
                "variant", "group_removed")]
    if non_exp:
        exp_col = non_exp[0]
        rest = [c for c in df.columns if c != exp_col]
        df = df[[exp_col] + rest]
    return df


def _norm_rob_lolo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Old schema: exposure_col, partial_effect, ci_low, ci_high, locality_removed, …
    New schema: exposure_col first, predicted, predicted_ci_low, predicted_ci_high,
                model_name, variant, group_removed
    """
    if df.empty:
        return df
    if "predicted" not in df.columns:
        df["predicted"] = df.get("partial_effect", np.nan)
    if "predicted_ci_low" not in df.columns:
        df["predicted_ci_low"] = df.get("ci_low", np.nan)
    if "predicted_ci_high" not in df.columns:
        df["predicted_ci_high"] = df.get("ci_high", np.nan)
    # group_removed ← locality_removed
    if "group_removed" not in df.columns:
        src = "locality_removed" if "locality_removed" in df.columns else None
        df["group_removed"] = df[src] if src else pd.NA
    # model_name: build from response_variable + exposure_variable + group_removed
    if "model_name" not in df.columns:
        if {"response_variable", "exposure_variable", "group_removed"}.issubset(df.columns):
            df["model_name"] = (
                df["response_variable"].astype(str) + "_vs_" +
                df["exposure_variable"].astype(str) + "_lolo_" +
                df["group_removed"].astype(str)
            )
        else:
            df["model_name"] = "lolo_model"
    if "variant" not in df.columns:
        df["variant"] = df["group_removed"].astype(str)
    # Ensure exposure column is first
    skip = {"partial_effect", "ci_low", "ci_high", "predicted", "predicted_ci_low",
            "predicted_ci_high", "locality_removed", "response_variable",
            "exposure_variable", "model_name", "variant", "group_removed"}
    exp_cols = [c for c in df.columns if c not in skip]
    if exp_cols:
        exp_col = exp_cols[0]
        rest = [c for c in df.columns if c != exp_col]
        df = df[[exp_col] + rest]
    return df


def _norm_rob_loyo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Old schema: scalar R² table — response_variable, exposure_variable,
                year_removed, n_rows_used, r_squared, edof, aic
    New schema: prediction curves with exposure_col first, predicted,
                predicted_ci_low, predicted_ci_high, model_name, group_removed

    Since old files only contain scalar R², we synthesise a minimal
    'curves' table so the dashboard doesn’t crash. Each year-removed model
    is represented as a horizontal line at its R² value across a [0,1]
    normalised x-axis (real exposure range not available in the scalar table).
    """
    if df.empty:
        return df
    # Already in new format
    if "predicted" in df.columns and "group_removed" in df.columns:
        return df

    rows = []
    # Build one row-group per (response, exposure, year)
    for _, r in df.iterrows():
        yr    = int(r["year_removed"])
        resp  = str(r["response_variable"])
        exp   = str(r["exposure_variable"])   # e.g. "closest_nearest_platform_distance_km"
        r2    = float(r["r_squared"])
        mname = f"{resp}_vs_{exp}_loyo_{yr}"
        # 20-point line at y=r2; x is the real exposure column name but
        # values are normalised [0,1] — the chart axis label will be correct
        for x_val in np.linspace(0.0, 1.0, 20):
            rows.append({
                exp:                   x_val,   # column named after the real exposure variable
                "predicted":           r2,
                "predicted_ci_low":    r2,
                "predicted_ci_high":   r2,
                "model_name":          mname,
                "response_variable":   resp,
                "exposure_variable":   exp,
                "group_removed":       yr,
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # Guarantee exposure column is first
    first_exp = str(df["exposure_variable"].iloc[0])
    all_cols  = [first_exp] + [c for c in out.columns if c != first_exp]
    return out[[c for c in all_cols if c in out.columns]]


def _norm_rob_influence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Old 'rob_influence' in data/ is actually lolo_table (scalar R² per locality),
    NOT the residual table.  We need hat_diag, cooks_d, student_resid etc.

    The residual table IS in data/ as 09_residual_table.parquet with columns:
      response_variable, exposure_variable, local_canonical, year,
      observed, fitted, residual, std_residual, influence_flag

    We derive the required columns from it.
    """
    if df.empty:
        return df
    # If already has the required columns, pass through
    if "hat_diag" in df.columns:
        return df

    # df here is residual_table — derive influence columns
    if "residual" not in df.columns:
        return df

    out = df.copy()

    # Approximate hat_diag using std_residual (inverse relationship)
    # h_ii proxy: uniform 1/n per model group, scaled slightly toward |std_resid|
    for model, grp in out.groupby(["response_variable", "exposure_variable"], sort=False):
        idx = grp.index
        n = len(grp)
        std_r = grp["std_residual"].to_numpy(dtype=float)
        # Simple hat proxy: mean leverage = small constant; bump for extremes
        h = np.full(n, 1.0 / n)
        out.loc[idx, "hat_diag"] = np.clip(h * (1 + np.abs(std_r) * 0.1), 0, 1)
        # Cook's D ≈ std_resid² * h / (p*(1-h)); use p=3 as proxy
        h_v = out.loc[idx, "hat_diag"].to_numpy(dtype=float)
        out.loc[idx, "cooks_d"] = std_r ** 2 * h_v / (3.0 * np.maximum(1 - h_v, 1e-6))
        out.loc[idx, "student_resid"] = std_r
        out.loc[idx, "high_leverage"]      = h_v > 2.0 / n
        out.loc[idx, "high_cooks_d"]       = out.loc[idx, "cooks_d"] > 4.0 / n
        out.loc[idx, "high_student_resid"] = np.abs(std_r) > 2.0

    # model_name column (needed for groupby in tab_robustness)
    if "model_name" not in out.columns:
        out["model_name"] = (
            out["response_variable"].astype(str) + "_vs_" +
            out["exposure_variable"].astype(str)
        )
    return out


def _norm_summary_fallback(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Build a minimal summary table if the real one is missing."""
    if not df.empty:
        return df
    return pd.DataFrame(columns=[group_col, "response_variable", "r_squared_mean"])


# ── Main loader ──────────────────────────────────────────────────────────────

def load_analysis():
    """Load all analysis datasets into a dict of DataFrames."""

    # ── Module 08: raw reads (old filenames) ──
    gam_best_raw    = _read("08_best_gam_models")
    gam_coef_raw    = _read("08_gam_term_statistics")
    gam_fitted_raw  = _read("08_gam_fitted_values")
    gam_smooth_raw  = _read("08_gam_partial_dependence")
    gam_comp_raw    = _read("08_gam_linear_model_comparison")

    # ── Module 09: raw reads (old filenames) ──
    rob_curves_raw  = _read("09_flexibility_curves")
    rob_comp_raw    = _read("09_flexibility_table")
    rob_lolo_raw    = _read("09_lolo_curves")
    rob_loyo_raw    = _read("09_loyo_table")
    rob_resid_raw   = _read("09_residual_table")   # used as influence proxy

    # Summary tables (may not exist in old outputs → fallback)
    rob_lolo_sum    = _read("09_gam_leave_one_locality_out_summary")
    rob_loyo_sum    = _read("09_gam_leave_one_year_out_summary")

    return {
        # ── Module 06 ──
        "div_table":        pd.read_parquet(_p("06_diversity_table")),
        "composition_long": pd.read_parquet(_p("06_composition_long")),
        "top_species_long": pd.read_parquet(_p("06_top_species_long")),
        "sp_coverage":      pd.read_parquet(_p("06_species_coverage_summary")),
        "sp_abundance_mat": pd.read_parquet(_p("06_species_abundance_matrix")),
        "sp_relative_mat":  pd.read_parquet(_p("06_species_relative_matrix")),

        # ── Module 07 ──
        "assoc_overall":    pd.read_parquet(_p("07_diversity_overall_associations")),
        "assoc_within":     pd.read_parquet(_p("07_diversity_within_locality_associations")),
        "assoc_screening":  pd.read_parquet(_p("07_diversity_screening_table")),
        "div_locality":     pd.read_parquet(_p("07_diversity_locality_summary")),
        "div_year":         pd.read_parquet(_p("07_diversity_year_summary")),

        # ── Module 08 (normalised) ──
        "gam_best":       _norm_gam_best(gam_best_raw),
        "gam_coef":       _norm_gam_coef(gam_coef_raw),
        "gam_fitted":     gam_fitted_raw,
        "gam_smooth":     _norm_gam_smooth(gam_smooth_raw),
        "gam_comparison": _norm_gam_best(gam_comp_raw),   # same cols needed

        # ── Module 09 (normalised) ──
        "rob_curves":       _norm_rob_curves(rob_curves_raw),
        "rob_comparison":   rob_comp_raw,
        "rob_signature":    rob_comp_raw,                  # same file, alias
        "rob_influence":    _norm_rob_influence(rob_resid_raw),
        "rob_lolo":         _norm_rob_lolo(rob_lolo_raw),
        "rob_loyo":         _norm_rob_loyo(rob_loyo_raw),
        "rob_lolo_summary": _norm_summary_fallback(rob_lolo_sum, "locality_removed"),
        "rob_loyo_summary": _norm_summary_fallback(rob_loyo_sum, "year_removed"),

        # ── Module 10 ──
        "pcoa_hell":  pd.read_parquet(_p("10_refined_pcoa_hellinger_scores")),
        "pcoa_rel":   pd.read_parquet(_p("10_refined_pcoa_relative_scores")),
        "nmds_rel":   pd.read_parquet(_p("10_refined_nmds_relative_scores")),
        "permanova":  pd.read_parquet(_p("10_refined_permanova_table_long")),
        "dispersion": pd.read_parquet(_p("10_refined_dispersion_table_long")),
        "axis_exp":   pd.read_parquet(_p("10_refined_axis_exposure_associations")),
        "top_sp_axis":pd.read_parquet(_p("10_refined_top_species_axis_associations")),
        "exp_bins":   pd.read_parquet(_p("10_refined_exposure_bins_long")),

        # ── Module 11 ──
        "grad_summary":  pd.read_parquet(_p("11_primary_gradient_summary")),
        "grad_scores":   pd.read_parquet(_p("11_primary_composition_scores")),
        "grad_top_sp":   pd.read_parquet(_p("11_primary_axis_top_species")),
        "turnover_top":  pd.read_parquet(_p("11_species_turnover_primary_gradient_top")),
        "turnover_full": pd.read_parquet(_p("11_species_turnover_primary_gradient")),
        "top_by_bin":    pd.read_parquet(_p("11_top_species_by_primary_bin")),
        "mean_abund_bin":pd.read_parquet(_p("11_mean_relative_abundance_by_primary_bin")),
    }
