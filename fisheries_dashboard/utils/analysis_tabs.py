"""
analysis_tabs.py — The 5 analysis tabs of the dashboard:
  tab_exposure     → Module 07: Platform exposure
  tab_assoc        → Module 07: Associations (platform + AMP predictors)
  tab_gam          → Module 08: GAM models
  tab_robustness   → Module 09: GAM robustness
  tab_ordination   → Module 10: Multivariate ordination
  tab_gradient     → Module 11: Composition gradient
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.coords import PORT_COORDS

# ── Helpers ────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Converts '#rrggbb' to valid Plotly 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


EXPOSURE_LABELS = {
    "mean_nearest_platform_distance_km":     "Mean platform distance (km)",
    "closest_nearest_platform_distance_km":  "Closest platform distance (km)",
    "farthest_nearest_platform_distance_km": "Farthest platform distance (km)",
    "mean_n_platforms_within_20km":          "No. platforms within 20 km (mean)",
    "mean_n_platforms_within_10km":          "No. platforms within 10 km (mean)",
    "inv_distance_sum_mean":                 "Inverse distance sum (mean)",
    "mean_distance_to_nearest_protected_area_km": "Mean distance to nearest protected area (km)",
    "mean_distance_to_apa_dunas_do_rosado_km": "Mean distance to APA Dunas do Rosado (km)",
    "mean_distance_to_rds_ponta_do_tubarao_km": "Mean distance to RDS Ponta do Tubarao (km)",
    "share_landings_inside_apa_dunas_do_rosado": "Share of landings inside APA Dunas do Rosado",
    "share_landings_inside_rds_ponta_do_tubarao": "Share of landings inside RDS Ponta do Tubarao",
    "dominant_protected_area_relation":      "Dominant protected-area relation",
    "dominant_nearest_protected_area":       "Dominant nearest protected area",
    "inside_any_protected_area":             "Inside any protected area",
    "mean_nearest_platform_distance_km__x__inside_any_protected_area":
        "Mean platform distance x inside any protected area",
    "mean_nearest_platform_distance_km__x__mean_distance_to_nearest_protected_area_km":
        "Mean platform distance x mean protected-area distance",
    "mean_nearest_platform_distance_km__x__dominant_protected_area_relation":
        "Mean platform distance x protected-area relation",
    "mean_nearest_platform_distance_km__x__dominant_nearest_protected_area":
        "Mean platform distance x nearest protected area",
}

RESPONSE_LABELS = {
    "pielou_species":           "Pielou's J' index",
    "shannon_species":          "Shannon H' index",
    "species_richness":         "Species richness (S)",
    "production_ton":           "Total production (t)",
    "production_per_trip_ton":  "Production per trip (t)",
    "production_per_fisher_ton": "Production per fisher (t)",
    "production_value_per_ton":  "Production value per ton",
}

LOCALITY_COLORS = {
    "AREIA BRANCA":    "#e74c3c",
    "CAICARA DO NORTE":"#2980b9",
    "GUAMARE":         "#27ae60",
    "MACAU":           "#8e44ad",
    "PORTO DO MANGUE": "#e67e22",
}

def _loc_color(x, fallback="#95a5a6"):
    """Return locality colour regardless of case (UPPER, Title, lower)."""
    if not isinstance(x, str):
        return fallback
    return LOCALITY_COLORS.get(x.upper(), fallback)

# Color map keyed by display name (for color_discrete_map when column is port_name)
_LOCALITY_COLORS_BY_NAME = {
    PORT_COORDS[k]["name"]: v
    for k, v in LOCALITY_COLORS.items()
    if k in PORT_COORDS
}

_PAL_PRIMARY = "#2980b9"
_PAL_SECONDARY = "#e67e22"
_PAL_ACCENT = "#8e44ad"
_PAL_NEUTRAL = "#95a5a6"

# Script 04 creates these columns via pd.get_dummies(prefix="landing_relation").
# Script 05's rename (share_landing_relation_ → share_landings_relation_) never
# fires because the prefix it checks doesn't match, so the 04_ parquet columns
# remain as landing_relation_* throughout the pipeline.
_RELATION_COLS = [
    "landing_relation_inside_apa",
    "landing_relation_inside_rds",
    "landing_relation_inside_both",
    "landing_relation_outside_between_both",
    "landing_relation_outside_closer_to_apa",
    "landing_relation_outside_closer_to_rds",
]

_ZONE_LABELS = {
    "inside_apa": "Inside APA Dunas do Rosado",
    "inside_rds": "Inside RDS Ponta do Tubarao",
    "inside_both": "Inside both protected areas",
    "outside_between_both": "Outside, between both PAs",
    "outside_closer_to_apa": "Outside, closer to APA",
    "outside_closer_to_rds": "Outside, closer to RDS",
    "outside_unknown": "Outside, unknown relation",
    "APA_DUNAS_DO_ROSADO": "APA Dunas do Rosado",
    "RDS_PONTA_DO_TUBARAO": "RDS Ponta do Tubarao",
    "Q1": "Q1 (near)",
    "Q2": "Q2 (mid)",
    "Q3": "Q3 (far)",
}

_ZONE_COLORS = {
    "inside_apa": "#27ae60",
    "inside_rds": "#16a085",
    "inside_both": "#1abc9c",
    "outside_between_both": "#95a5a6",
    "outside_closer_to_apa": "#f39c12",
    "outside_closer_to_rds": "#e67e22",
    "outside_unknown": "#7f8c8d",
}

# Pre-build a case-insensitive lookup: upper-case key → display name
_PORT_NAME_UPPER = {k.upper(): v["name"] for k, v in PORT_COORDS.items()}

def _port_name(x):
    """Return the display name for a locality key (case-insensitive lookup)."""
    if not isinstance(x, str):
        return str(x)
    # Try exact match first, then upper-case match
    direct = PORT_COORDS.get(x)
    if direct:
        return direct["name"]
    return _PORT_NAME_UPPER.get(x.upper(), x)

def _elabel(col):
    return EXPOSURE_LABELS.get(col, col)

def _rlabel(col):
    return RESPONSE_LABELS.get(col, col)

_RESP_LABELS = RESPONSE_LABELS

FAMILY_LABELS = {
    "continuous":                  "Continuous predictor",
    "categorical":                 "Categorical predictor",
    "platform_inside_interaction": "Platform x inside-PA interaction",
    "tensor_interaction":          "Tensor interaction: platform distance x PA distance",
    "platform_pa_combined":        "Combined platform x protected-area groups",
}
_FAM_LABELS = FAMILY_LABELS

def _family_label(col):
    return FAMILY_LABELS.get(col, col)

def _model_type_label(col):
    return {
        "gam_penalised": "GAM spline",
        "linear_ols": "Linear",
        "spline_gam_like": "GAM spline",
        "linear": "Linear",
    }.get(col, col)

def _derive_inside_any_pa(df):
    """Return an inside-PA indicator from available protected-area columns."""
    if "inside_any_protected_area" in df.columns:
        return pd.to_numeric(df["inside_any_protected_area"], errors="coerce").fillna(0).astype(int)
    candidates = [
        "share_landings_inside_apa_dunas_do_rosado",
        "share_landings_inside_rds_ponta_do_tubarao",
        "mean_n_protected_areas_containing_landing",
        "n_protected_areas_containing_landing",
    ]
    present = [c for c in candidates if c in df.columns]
    if present:
        vals = df[present].apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1)
        return (vals > 0).astype(int)
    if "dominant_protected_area_relation" in df.columns:
        return df["dominant_protected_area_relation"].astype("string").str.contains("inside", na=False).astype(int)
    return pd.Series(0, index=df.index, dtype=int)

def _zone_label(value):
    if pd.isna(value):
        return "Unknown"
    return _ZONE_LABELS.get(str(value), str(value).replace("_", " ").title())

def _zone_color(value):
    if pd.isna(value):
        return _PAL_NEUTRAL
    return _ZONE_COLORS.get(str(value), _PAL_NEUTRAL)

def _derive_dominant_relation(row):
    best_key = None
    best_val = -np.inf
    for col in _RELATION_COLS:
        if col not in row.index:
            continue
        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        if pd.notna(val) and val > best_val:
            best_val = val
            best_key = col.replace("landing_relation_", "")
    return best_key or "outside_between_both"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXPOSICIÓN A PLATAFORMAS (módulo 07)
# ══════════════════════════════════════════════════════════════════════════════
def tab_exposure(ad):
    st.markdown('<h3 class="section-title">Associations between platform exposure and fisheries</h3>',
                unsafe_allow_html=True)

    # ── Panel de métricas clave ──────────────────────────────────────────
    screening = ad["assoc_screening"].copy()
    top1 = screening.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strongest association", f"ρ = {top1['spearman_corr']:.3f}",
              f"{_rlabel(top1['response_variable'])}")
    c2.metric("Response variable", _rlabel(top1['response_variable']))
    c3.metric("Exposure variable", _elabel(top1['exposure_variable']))
    c4.metric("Formal candidates",
              str(len(screening[screening["screening_note"] == "candidate_for_formal_followup"])))

    st.markdown("---")

    # ── Screening: ranking de correlaciones ──────────────────────────────
    col_l, col_r = st.columns([1.1, 1])

    with col_l:
        st.markdown("#### Global association ranking (Spearman |ρ|)")
        sc_plot = screening.copy()
        sc_plot["response"] = sc_plot["response_variable"].map(_rlabel)
        sc_plot["exposure"] = sc_plot["exposure_variable"].map(_elabel)
        sc_plot["par"] = sc_plot["response"] + " ↔ " + sc_plot["exposure"]
        sc_plot["color"] = sc_plot["spearman_corr"].apply(
            lambda x: "#e74c3c" if x < 0 else "#27ae60")
        sc_plot = sc_plot.sort_values("abs_spearman_corr", ascending=True)

        fig = go.Figure(go.Bar(
            x=sc_plot["spearman_corr"],
            y=sc_plot["par"],
            orientation="h",
            marker_color=sc_plot["color"],
            text=sc_plot["spearman_corr"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
            hovertemplate="ρ = %{x:.4f}<extra></extra>",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#aaa", line_width=1)
        fig.update_layout(
            height=420, margin=dict(t=20, l=10, r=30),
            xaxis_title="Spearman correlation (ρ)",
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Prioritized screening table")
        disp = screening[["priority_rank", "response_variable", "exposure_variable",
                           "spearman_corr", "pearson_corr", "n_complete", "screening_note"]].copy()
        disp["response_variable"] = disp["response_variable"].map(_rlabel)
        disp["exposure_variable"] = disp["exposure_variable"].map(_elabel)
        disp.columns = ["Rank", "Response", "Exposure", "ρ Spearman", "r Pearson",
                        "N", "Note"]
        disp["ρ Spearman"] = disp["ρ Spearman"].round(3)
        disp["r Pearson"]  = disp["r Pearson"].round(3)
        st.dataframe(
            disp.style.background_gradient(subset=["ρ Spearman"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True, height=420,
        )

    st.markdown("---")

    # ── Scatter interactivo ───────────────────────────────────────────────
    st.markdown("#### Explore exposure ↔ diversity/production relationship")
    div = ad["div_table"].copy()
    div["port_name"] = div["local_canonical"].map(_port_name)

    col_sel1, col_sel2, col_sel3 = st.columns(3)
    x_col = col_sel1.selectbox("X variable (exposure)",
        options=[c for c in EXPOSURE_LABELS if c in div.columns],
        format_func=_elabel)
    y_col = col_sel2.selectbox("Y variable (response)",
        options=[c for c in RESPONSE_LABELS if c in div.columns],
        format_func=_rlabel)
    color_by = col_sel3.selectbox("Color", ["Locality", "Year"])

    color_col = "port_name" if color_by == "Locality" else "year"
    fig2 = px.scatter(
        div.dropna(subset=[x_col, y_col]),
        x=x_col, y=y_col,
        color=color_col,
        trendline="ols",
        trendline_scope="overall",
        trendline_color_override="#2c3e50",
        height=420,
        labels={x_col: _elabel(x_col), y_col: _rlabel(y_col),
                "port_name": "Locality", "year": "Year"},
        color_discrete_map=_LOCALITY_COLORS_BY_NAME if color_by == "Locality" else None,
        hover_data=["port_name", "year"],
    )
    fig2.update_layout(margin=dict(t=20), legend_title_text=color_by)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Asociaciones within-locality ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Within-locality correlations (Spearman)")
    within = ad["assoc_within"].copy()
    within["port_name"] = within["local_canonical"].map(_port_name)
    within["response"] = within["response_variable"].map(_rlabel)
    within["exposure"] = within["exposure_variable"].map(_elabel)

    pivot = within.dropna(subset=["spearman_corr"]).pivot_table(
        index=["response", "exposure"], columns="port_name",
        values="spearman_corr", aggfunc="first"
    )
    if not pivot.empty:
        fig3 = px.imshow(
            pivot.round(3),
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
            height=max(350, len(pivot) * 28),
        )
        fig3.update_layout(
            margin=dict(t=20),
            xaxis_title="Locality",
            yaxis_title="",
            coloraxis_colorbar_title="ρ",
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Insufficient temporal data for within-locality correlations.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ASSOCIATIONS (módulo 07 — platform + AMP predictors)
# ══════════════════════════════════════════════════════════════════════════════
def tab_assoc(ad):
    st.markdown('<h3 class="section-title">Associations: platform exposure and protected area predictors</h3>',
                unsafe_allow_html=True)

    # ── Section 1: Continuous predictor associations (all) ────────────────
    st.markdown("### Continuous predictor associations (all)")
    st.caption(
        "Spearman correlations for all continuous predictors — both platform distance "
        "metrics and AMP (protected area) distance metrics."
    )

    assoc_cont = ad.get("assoc_overall_cont", pd.DataFrame())

    if assoc_cont.empty:
        st.info("Dataset `assoc_overall_cont` (07_diversity_overall_continuous_associations) "
                "not yet available.")
    else:
        # Build heatmap: rows = exposure_variable, columns = response_variable
        pivot_cont = assoc_cont.pivot_table(
            index="exposure_variable",
            columns="response_variable",
            values="spearman_corr",
            aggfunc="first",
        )

        # Nicer axis labels
        pivot_cont.index = [_elabel(i) for i in pivot_cont.index]
        pivot_cont.columns = [_rlabel(c) for c in pivot_cont.columns]

        fig_heat = px.imshow(
            pivot_cont.round(3),
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
            height=max(350, len(pivot_cont) * 36),
            labels={"color": "ρ Spearman"},
        )
        fig_heat.update_layout(
            margin=dict(t=20, b=10),
            xaxis_title="Response variable",
            yaxis_title="Predictor",
            coloraxis_colorbar_title="ρ",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Detailed table below the heatmap
        with st.expander("Full continuous associations table"):
            disp_cont = assoc_cont[[
                "response_variable", "exposure_variable", "spearman_corr",
                "pearson_corr", "spearman_p_value", "spearman_p_value_adj_fdr",
                "significant_fdr_05", "n_complete",
            ]].copy()
            disp_cont["response_variable"] = disp_cont["response_variable"].map(_rlabel)
            disp_cont["exposure_variable"] = disp_cont["exposure_variable"].map(_elabel)
            disp_cont = disp_cont.sort_values("spearman_corr", key=abs, ascending=False)
            disp_cont.columns = [
                "Response", "Predictor", "ρ Spearman", "r Pearson",
                "p-value", "p-value (FDR adj.)", "Sig. FDR 5%", "N",
            ]
            disp_cont["ρ Spearman"] = disp_cont["ρ Spearman"].round(3)
            disp_cont["r Pearson"]  = disp_cont["r Pearson"].round(3)
            disp_cont["p-value"]    = disp_cont["p-value"].apply(lambda v: f"{v:.4g}")
            disp_cont["p-value (FDR adj.)"] = disp_cont["p-value (FDR adj.)"].apply(
                lambda v: f"{v:.4g}")
            st.dataframe(
                disp_cont.style.background_gradient(subset=["ρ Spearman"], cmap="RdYlGn"),
                use_container_width=True, hide_index=True,
            )

    st.markdown("---")

    # ── Section 2: Categorical AMP predictor associations ─────────────────
    st.markdown("### Categorical AMP predictor associations (Kruskal-Wallis)")
    st.caption(
        "Kruskal-Wallis H tests for categorical AMP predictors: "
        "`dominant_protected_area_relation` and `dominant_nearest_protected_area`."
    )

    assoc_cat = ad.get("assoc_categorical", pd.DataFrame())

    if assoc_cat.empty:
        st.info("Dataset `assoc_categorical` (07_diversity_categorical_response_tests) "
                "not yet available.")
    else:
        disp_cat = assoc_cat[[
            "response_variable", "exposure_variable", "n_complete",
            "n_levels_tested", "kruskal_h", "kruskal_p_value",
            "p_value_adj_fdr", "significant_fdr_05",
        ]].copy()
        disp_cat["response_variable"] = disp_cat["response_variable"].map(_rlabel)
        disp_cat = disp_cat.sort_values("kruskal_p_value")
        disp_cat.columns = [
            "Response", "Predictor", "N", "Levels",
            "Kruskal H", "p-value", "p-value (FDR adj.)", "Sig. FDR 5%",
        ]
        disp_cat["Kruskal H"] = disp_cat["Kruskal H"].round(3)
        disp_cat["p-value"]   = disp_cat["p-value"].apply(lambda v: f"{v:.4g}")
        disp_cat["p-value (FDR adj.)"] = disp_cat["p-value (FDR adj.)"].apply(
            lambda v: f"{v:.4g}")

        # Highlight significant rows
        def _highlight_sig(row):
            style = "background-color: #d4edda;" if row["Sig. FDR 5%"] else ""
            return [style] * len(row)

        st.dataframe(
            disp_cat.style.apply(_highlight_sig, axis=1),
            use_container_width=True, hide_index=True,
        )

        # Visual: bar chart of H statistic by response × predictor
        cat_plot = assoc_cat.copy()
        cat_plot["response"] = cat_plot["response_variable"].map(_rlabel)
        cat_plot["pair"] = cat_plot["response"] + " ↔ " + cat_plot["exposure_variable"]
        cat_plot["color"] = cat_plot["significant_fdr_05"].map(
            {True: "#27ae60", False: "#bdc3c7"})
        cat_plot = cat_plot.sort_values("kruskal_h", ascending=True)

        fig_h = go.Figure(go.Bar(
            x=cat_plot["kruskal_h"],
            y=cat_plot["pair"],
            orientation="h",
            marker_color=cat_plot["color"],
            text=cat_plot["kruskal_h"].apply(lambda v: f"{v:.2f}"),
            textposition="outside",
            customdata=cat_plot["kruskal_p_value"].apply(lambda v: f"{v:.4g}"),
            hovertemplate="H = %{x:.3f}<br>p = %{customdata}<extra>%{y}</extra>",
        ))
        fig_h.update_layout(
            height=max(300, len(cat_plot) * 36),
            margin=dict(t=20, r=80),
            xaxis_title="Kruskal-Wallis H statistic",
            yaxis_title="",
            showlegend=False,
        )
        # Legend annotation
        fig_h.add_annotation(
            x=cat_plot["kruskal_h"].max() * 0.95,
            y=-0.8,
            text="Green = significant (FDR 5%)",
            showarrow=False,
            font=dict(size=10, color="#27ae60"),
            xanchor="right",
        )
        st.plotly_chart(fig_h, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODELOS GAM (módulo 08)
# ══════════════════════════════════════════════════════════════════════════════
def tab_gam(ad):
    st.markdown('<h3 class="section-title">GAM-spline models and comparison with linear models</h3>',
                unsafe_allow_html=True)

    best = ad["gam_best"].copy()
    comparison = ad["gam_comparison"].copy()
    smooth = ad["gam_smooth"].copy()
    fitted = ad["gam_fitted"].copy()
    coef = ad["gam_coef"].copy()
    div_table = ad.get("div_table", pd.DataFrame()).copy()

    if best.empty or comparison.empty:
        st.info("GAM model outputs are not available.")
        return

    if "predictor_family" in comparison.columns:
        st.markdown("#### Model family leaderboard")
        gam_only = comparison[comparison["model_type"].eq("gam_penalised")].copy()
        if not gam_only.empty:
            fam_summary = (
                gam_only.groupby("predictor_family", dropna=False)
                .agg(
                    models=("model_name", "nunique"),
                    mean_r2=("r_squared", "mean"),
                    best_r2=("r_squared", "max"),
                    best_aic=("aic", "min"),
                    mean_rmse=("rmse", "mean"),
                )
                .reset_index()
                .sort_values(["best_r2", "best_aic"], ascending=[False, True])
            )
            fam_summary["Predictor family"] = fam_summary["predictor_family"].map(_family_label)
            show = fam_summary[["Predictor family", "models", "mean_r2", "best_r2", "best_aic", "mean_rmse"]].copy()
            show.columns = ["Predictor family", "N models", "Mean R2", "Best R2", "Best AIC", "Mean RMSE"]
            st.dataframe(show.round(4), use_container_width=True, hide_index=True)

        st.caption(
            "The current best-response table is dominated by interaction models. "
            "Partial-dependence curves are only available for simple continuous/categorical GAMs; "
            "interaction models are shown with fitted/observed diagnostics and interaction-specific summaries."
        )

    # ── Selector de modelo: grouped by predictor_family ───────────────────
    # Step 1: choose predictor family
    families_available = sorted(best["predictor_family"].dropna().unique().tolist()) \
        if "predictor_family" in best.columns else []

    if families_available:
        sel_family = st.selectbox(
            "Predictor family",
            options=families_available,
            format_func=_family_label,
            key="gam_family_sel",
        )
        best_family = best[best["predictor_family"] == sel_family].copy()
    else:
        sel_family = None
        best_family = best.copy()

    # Step 2: choose response variable within family
    resp_in_family = sorted(best_family["response_variable"].dropna().unique().tolist())
    sel_resp_gam = st.selectbox(
        "Response variable",
        options=resp_in_family,
        format_func=_rlabel,
        key="gam_resp_sel",
    )
    best_resp = best_family[best_family["response_variable"] == sel_resp_gam].copy()

    # Step 3: select specific model
    model_names = best_resp["model_name"].tolist()

    def _model_label(model_name):
        row_ = best[best["model_name"] == model_name]
        if row_.empty:
            return model_name
        r_ = row_.iloc[0]
        pred = r_.get("predictor", r_.get("exposure_variable", ""))
        return f"{_rlabel(r_['response_variable'])} ↔ {_elabel(pred)} [GAM]"

    sel_model = st.selectbox(
        "Select model",
        options=model_names,
        format_func=_model_label,
        key="gam_model_sel",
    )

    row = best[best["model_name"] == sel_model].iloc[0]

    # ── KPIs del modelo seleccionado — now includes lam ──────────────────
    has_lam = "lam" in row.index and pd.notna(row.get("lam"))
    kpi_cols = st.columns(6 if has_lam else 5)
    kpi_cols[0].metric("R²", f"{row['r_squared']:.4f}")
    kpi_cols[1].metric("Adj. R²", f"{row['adj_r_squared']:.4f}" if pd.notna(row.get('adj_r_squared')) else "—")
    kpi_cols[2].metric("AIC", f"{row['aic']:.2f}" if pd.notna(row.get('aic')) else "—")
    kpi_cols[3].metric("RMSE", f"{row['rmse']:.4f}" if pd.notna(row.get('rmse')) else "—")
    edof_val = row.get("edof", row.get("n_parameters", None))
    kpi_cols[4].metric("eDoF (spline)", f"{float(edof_val):.2f}" if pd.notna(edof_val) else "—")
    if has_lam:
        kpi_cols[5].metric("λ (smoothing)", f"{float(row['lam']):.4g}")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # ── Curve or interaction diagnostic ───────────────────────────────────
    with col_l:
        st.markdown("#### Model effect / interaction diagnostic")
        sm = smooth[smooth["model_name"] == sel_model].copy()
        fit_sel = fitted[fitted["model_name"] == sel_model].copy() if not fitted.empty else pd.DataFrame()
        predictor_family = row.get("predictor_family", "")
        predictor_name = row.get("predictor", row.get("exposure_variable", ""))

        if not sm.empty and predictor_family in {"continuous", "categorical"}:
            # Determine the exposure column (column 0 of gam_smooth)
            exp_col = predictor_name
            x_col_smooth = exp_col if exp_col in sm.columns else sm.columns[0]
            sm_sorted = sm.dropna(subset=[x_col_smooth]).sort_values(x_col_smooth)

            fig = go.Figure()
            if "predicted_ci_high" in sm_sorted.columns and "predicted_ci_low" in sm_sorted.columns:
                fig.add_trace(go.Scatter(
                    x=sm_sorted[x_col_smooth], y=sm_sorted["predicted_ci_high"],
                    mode="lines", line=dict(width=0), showlegend=False,
                    name="CI 95% upper",
                ))
                fig.add_trace(go.Scatter(
                    x=sm_sorted[x_col_smooth], y=sm_sorted["predicted_ci_low"],
                    mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(41,128,185,0.18)", showlegend=False,
                    name="CI 95%",
                ))
            y_col_curve = "predicted" if "predicted" in sm_sorted.columns else "partial_effect"
            if y_col_curve in sm_sorted.columns:
                fig.add_trace(go.Scatter(
                    x=sm_sorted[x_col_smooth], y=sm_sorted[y_col_curve],
                    mode="lines", line=dict(color="#2980b9", width=2.5),
                    name="GAM predicted",
                ))

            if not fit_sel.empty and "local_canonical" in fit_sel.columns:
                fit_sel["port_name"] = fit_sel["local_canonical"].map(_port_name)
                obs_x = "observed_exposure" if "observed_exposure" in fit_sel.columns else fit_sel.columns[0]
                obs_y = "observed_response" if "observed_response" in fit_sel.columns else fit_sel.columns[1]
                fig.add_trace(go.Scatter(
                    x=fit_sel[obs_x], y=fit_sel[obs_y],
                    mode="markers",
                    marker=dict(color=[_loc_color(l) for l in fit_sel["local_canonical"]],
                                size=7, line=dict(width=1, color="white")),
                    name="Observed",
                    text=fit_sel["port_name"] + " " + fit_sel["year"].astype(str),
                    hovertemplate="%{text}<br>X=%{x:.2f} Y=%{y:.4f}<extra></extra>",
                ))

            fig.update_layout(
                height=400, margin=dict(t=20),
                xaxis_title=_elabel(exp_col),
                yaxis_title=_rlabel(row["response_variable"]),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)

        elif not fit_sel.empty and not div_table.empty and "local_canonical" in fit_sel.columns:
            merge_cols = [c for c in ["local_canonical", "year"] if c in fit_sel.columns and c in div_table.columns]
            interaction_df = fit_sel.merge(div_table, on=merge_cols, how="left", suffixes=("", "_div"))
            fit_sel["port_name"] = fit_sel["local_canonical"].map(_port_name)
            interaction_df["port_name"] = interaction_df["local_canonical"].map(_port_name)
            parts = str(predictor_name).split("__x__")
            platform_col = parts[0] if parts and parts[0] in interaction_df.columns else "mean_nearest_platform_distance_km"
            modifier_col = parts[1] if len(parts) > 1 else None

            if modifier_col == "inside_any_protected_area" or predictor_family == "platform_inside_interaction":
                interaction_df["inside_any_protected_area"] = _derive_inside_any_pa(interaction_df)
                color_col = "inside_any_protected_area"
                color_label = "Inside any PA"
            elif modifier_col in interaction_df.columns:
                color_col = modifier_col
                color_label = _elabel(modifier_col)
            else:
                color_col = "port_name"
                color_label = "Locality"

            fig = px.scatter(
                interaction_df.dropna(subset=[platform_col, "observed_response", "fitted"]),
                x=platform_col,
                y="observed_response",
                color=color_col,
                hover_data=["port_name", "year", "fitted", "residual"],
                height=400,
                labels={
                    platform_col: _elabel(platform_col),
                    "observed_response": _rlabel(row["response_variable"]),
                    color_col: color_label,
                },
            )
            fig.add_trace(go.Scatter(
                x=interaction_df[platform_col],
                y=interaction_df["fitted"],
                mode="markers",
                marker=dict(symbol="x", color="#2c3e50", size=8),
                name="Fitted",
                hovertemplate="Fitted=%{y:.4f}<extra></extra>",
            ))
            fig.update_layout(margin=dict(t=20), legend_title_text=color_label)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Partial-dependence curves are not available for this interaction model. "
                    "Use the observed-vs-fitted diagnostics and term table below.")

    # ── Observado vs Ajustado ─────────────────────────────────────────────
    with col_r:
        st.markdown("#### Observed vs Fitted + residuals")
        fit_sel2 = fitted[fitted["model_name"] == sel_model].copy() if not fitted.empty else pd.DataFrame()

        if not fit_sel2.empty and "local_canonical" in fit_sel2.columns:
            fit_sel2["port_name"] = fit_sel2["local_canonical"].map(_port_name)

            fig2 = px.scatter(
                fit_sel2, x="fitted", y="observed_response",
                color="port_name",
                color_discrete_map=_LOCALITY_COLORS_BY_NAME,
                height=250,
                labels={"fitted": "Fitted", "observed_response": "Observed",
                        "port_name": "Locality"},
                hover_data=["year"],
            )
            vmin = min(fit_sel2["fitted"].min(), fit_sel2["observed_response"].min())
            vmax = max(fit_sel2["fitted"].max(), fit_sel2["observed_response"].max())
            fig2.add_shape(type="line", x0=vmin, y0=vmin, x1=vmax, y1=vmax,
                           line=dict(dash="dash", color="#aaa", width=1))
            fig2.update_layout(margin=dict(t=20), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

            if "residual" in fit_sel2.columns:
                fig3 = px.scatter(
                    fit_sel2, x="fitted", y="residual",
                    color="port_name",
                    color_discrete_map=_LOCALITY_COLORS_BY_NAME,
                    height=130,
                    labels={"fitted": "Fitted", "residual": "Residual"},
                )
                fig3.add_hline(y=0, line_dash="dash", line_color="#aaa", line_width=1)
                fig3.update_layout(margin=dict(t=5, b=5), showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Fitted values not available for this model.")

    # ── Comparación GAM vs lineal — grouped by predictor_family ──────────
    st.markdown("---")
    st.markdown("#### Model comparison: GAM spline vs linear (AIC, R²) — by predictor family")

    resp_options_comp = comparison["response_variable"].unique().tolist() \
        if "response_variable" in comparison.columns else []
    if resp_options_comp:
        sel_resp_comp = st.selectbox("Response variable (comparison)",
                                     options=resp_options_comp,
                                     format_func=_rlabel,
                                     key="comp_resp")
        comp_sub = comparison[comparison["response_variable"] == sel_resp_comp].copy()
    else:
        comp_sub = comparison.copy()

    if not comp_sub.empty:
        # Group by predictor_family if available; compute delta_AIC vs best within each family
        if "predictor_family" in comp_sub.columns:
            family_best_aic = (
                comp_sub.groupby("predictor_family")["aic"]
                .min()
                .rename("family_best_aic")
            )
            comp_sub = comp_sub.merge(family_best_aic, on="predictor_family", how="left")
            comp_sub["delta_aic_within_family"] = comp_sub["aic"] - comp_sub["family_best_aic"]
            comp_sub["family_label"] = comp_sub["predictor_family"].map(
                lambda x: FAMILY_LABELS.get(x, x) if pd.notna(x) else "—"
            )
        else:
            comp_sub["delta_aic_within_family"] = np.nan
            comp_sub["family_label"] = "—"

        comp_sub["model_label"] = comp_sub["model_name"].str.replace("gam_", "GAM: ").str.replace("lin_", "LIN: ")
        model_type_col = "model_type" if "model_type" in comp_sub.columns else None
        if model_type_col:
            comp_sub["model_type_label"] = comp_sub[model_type_col].map(
                _model_type_label)
        else:
            comp_sub["model_type_label"] = "GAM spline"

        col_a, col_b = st.columns(2)
        adj_r2_col = "adj_r_squared" if "adj_r_squared" in comp_sub.columns else "r_squared"
        with col_a:
            fig_r2 = px.bar(
                comp_sub.sort_values(adj_r2_col, ascending=False),
                x=adj_r2_col, y="model_label", orientation="h",
                color="model_type_label", height=350,
                color_discrete_map={"GAM spline": "#2980b9", "Linear": "#e74c3c"},
                labels={adj_r2_col: "Adj. R²", "model_label": "",
                        "model_type_label": "Type"},
            )
            fig_r2.update_layout(margin=dict(t=20))
            st.plotly_chart(fig_r2, use_container_width=True)

        with col_b:
            fig_aic = px.bar(
                comp_sub.sort_values("aic"),
                x="aic", y="model_label", orientation="h",
                color="model_type_label", height=350,
                color_discrete_map={"GAM spline": "#2980b9", "Linear": "#e74c3c"},
                labels={"aic": "AIC (lower = better)", "model_label": "",
                        "model_type_label": "Type"},
            )
            fig_aic.update_layout(margin=dict(t=20))
            st.plotly_chart(fig_aic, use_container_width=True)

        # Delta-AIC table grouped by family
        if "predictor_family" in comp_sub.columns and comp_sub["delta_aic_within_family"].notna().any():
            st.markdown("##### ΔAIC within predictor family (vs best in each family)")
            delta_disp = comp_sub[[
                "family_label", "model_label", adj_r2_col, "aic", "delta_aic_within_family",
            ]].copy()
            delta_disp.columns = ["Predictor family", "Model", "Adj. R²", "AIC", "ΔAIC (vs best in family)"]
            delta_disp = delta_disp.sort_values(["Predictor family", "ΔAIC (vs best in family)"])
            delta_disp["Adj. R²"] = delta_disp["Adj. R²"].round(4)
            delta_disp["AIC"] = delta_disp["AIC"].round(2)
            delta_disp["ΔAIC (vs best in family)"] = delta_disp["ΔAIC (vs best in family)"].round(2)
            st.dataframe(
                delta_disp.style.background_gradient(
                    subset=["ΔAIC (vs best in family)"], cmap="YlOrRd"),
                use_container_width=True, hide_index=True,
            )

    # ── Coeficientes del modelo ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Model coefficients with 95% confidence intervals")
    coef_sel = coef[coef["model_name"] == sel_model].copy() if not coef.empty else pd.DataFrame()

    if not coef_sel.empty:
        if "term" in coef_sel.columns:
            coef_sel = coef_sel[~coef_sel["term"].str.startswith("bs(")].copy()
        coef_sel["sig"] = coef_sel.get("significant_alpha_0_05", pd.Series(dtype=bool)).map(
            {True: "p<0.05", False: "—"})
        coef_sel["color"] = coef_sel.get("significant_alpha_0_05", pd.Series(dtype=bool)).map(
            {True: "#27ae60", False: "#95a5a6"})

        has_estimates = "estimate" in coef_sel.columns and coef_sel["estimate"].notna().any()

        if has_estimates:
            fig_coef = go.Figure()
            for _, r in coef_sel.iterrows():
                fig_coef.add_trace(go.Scatter(
                    x=[r.get("conf_low", np.nan), r.get("conf_high", np.nan)],
                    y=[r["term"], r["term"]],
                    mode="lines", line=dict(color=r["color"], width=3),
                    showlegend=False,
                ))
                fig_coef.add_trace(go.Scatter(
                    x=[r["estimate"]], y=[r["term"]],
                    mode="markers",
                    marker=dict(color=r["color"], size=10, symbol="circle"),
                    showlegend=False,
                    hovertemplate=(
                        f"Estimate: {r['estimate']:.4f}<br>"
                        f"CI: [{r.get('conf_low', float('nan')):.4f}, "
                        f"{r.get('conf_high', float('nan')):.4f}]<br>"
                        f"p={r.get('p_value', float('nan')):.4f}"
                        f"<extra>{r['term']}</extra>"
                    ),
                ))
            fig_coef.add_vline(x=0, line_dash="dash", line_color="#aaa", line_width=1)
            fig_coef.update_layout(
                height=max(200, len(coef_sel) * 50),
                margin=dict(t=20),
                xaxis_title="Estimate (95% CI)",
            )
            st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.info("Point estimates and confidence intervals are not available in "
                    "the current output files. Re-run script 08 with the updated "
                    "version to unlock this chart.")

        disp_cols = [c for c in ["term", "estimate", "std_error", "t_value", "p_value", "sig"]
                     if c in coef_sel.columns]
        rename_map = {
            "term": "Term", "estimate": "Estimate", "std_error": "Std. error",
            "t_value": "t", "p_value": "p-value", "sig": "Sig.",
        }
        disp_coef = coef_sel[disp_cols].copy().rename(columns=rename_map)
        round_map = {}
        for c, r in [("Estimate", 5), ("Std. error", 5), ("t", 3), ("p-value", 4)]:
            if c in disp_coef.columns:
                round_map[c] = r
        if round_map:
            disp_coef = disp_coef.round(round_map)
        st.dataframe(disp_coef, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ROBUSTEZ DEL MODELO (módulo 09)
# ══════════════════════════════════════════════════════════════════════════════
def tab_robustness(ad):
    st.markdown('<h3 class="section-title">GAM model robustness — sensitivity analysis</h3>',
                unsafe_allow_html=True)

    rob_curves = ad["rob_curves"].copy()
    gam_smooth = ad["gam_smooth"].copy()
    rob_comp   = ad["rob_comparison"].copy()
    signature  = ad["rob_signature"].copy()
    influence  = ad["rob_influence"].copy()
    lolo       = ad["rob_lolo"].copy()
    loyo       = ad["rob_loyo"].copy()

    base_models = rob_curves["model_name"].str.replace("_alt_df3|_alt_df5|_quadratic", "",
                                                        regex=True).str.replace("_alt$","",
                                                        regex=True).unique().tolist()
    # Limpiar nombres
    all_base = sorted(set([m.rsplit("_alt", 1)[0].rsplit("_quadratic", 1)[0]
                            for m in rob_curves["model_name"].unique()]))
    sel_base = st.selectbox("Base model", all_base)

    # Extract the response_vs_exposure fragment used in LOLO/LOYO/influence model names.
    # rob_curves names: "gam_nsp{n}_{resp}_vs_{exp}"  →  rv_ev_base = "{resp}_vs_{exp}"
    # lolo/loyo/influence names: "{resp}_vs_{exp}_lolo_*" / "_loyo_*"
    import re as _re
    _stripped = _re.sub(r'^gam_nsp\d+_', '', sel_base)
    rv_ev_base = _stripped  # e.g. "shannon_species_vs_mean_nearest_platform_distance_km"

    st.markdown("---")

    # ── 1. Variantes de spline ────────────────────────────────────────────
    st.markdown("#### Curve stability: GAM variants")
    variants = rob_curves[rob_curves["model_name"].str.startswith(sel_base)].copy()

    # Añadir curva base del módulo 08
    base_name_exact = sel_base.replace("_alt_df3","").replace("_alt_df5","").replace("_quadratic","")
    # gam_smooth model names: "gam_{resp}_vs_{exp}"; rv_ev_base = "{resp}_vs_{exp}"
    base_key = [m for m in gam_smooth["model_name"].unique() if rv_ev_base in m]
    if base_key:
        base_sm = gam_smooth[gam_smooth["model_name"] == base_key[0]].copy()
        base_sm["variant"] = "base (df=4)"
        base_sm["group_removed"] = pd.NA
        # normalize column names
        variants = pd.concat([variants, base_sm], ignore_index=True)

    exp_col = variants.columns[0]  # first column is the exposure variable
    fitted_m = ad["gam_fitted"]
    base_fit = fitted_m[fitted_m["model_name"].isin(base_key)].copy() if base_key else pd.DataFrame()

    variant_colors = {
        "base (df=4)": "#2c3e50",
        "spline_df3":  "#e74c3c",
        "spline_df5":  "#27ae60",
        "quadratic":   "#8e44ad",
    }

    fig = go.Figure()

    # Observaciones
    if not base_fit.empty:
        base_fit["port_name"] = base_fit["local_canonical"].map(_port_name)
        fig.add_trace(go.Scatter(
            x=base_fit["observed_exposure"], y=base_fit["observed_response"],
            mode="markers",
            marker=dict(color=[_loc_color(l) for l in base_fit["local_canonical"]],
                        size=6, opacity=0.5),
            showlegend=False, name="Observed",
            text=base_fit["port_name"] + " " + base_fit["year"].astype(str),
            hovertemplate="%{text}<extra></extra>",
        ))

    for variant, grp in variants.groupby("variant"):
        grp_s = grp.dropna(subset=[exp_col]).sort_values(exp_col)
        color = variant_colors.get(variant, "#888")
        # IC
        if "predicted_ci_high" in grp_s.columns and "predicted_ci_low" in grp_s.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([grp_s[exp_col], grp_s[exp_col].iloc[::-1]]),
                y=pd.concat([grp_s["predicted_ci_high"], grp_s["predicted_ci_low"].iloc[::-1]]),
                fill="toself", fillcolor=_hex_to_rgba(color, 0.08),
                line=dict(width=0), showlegend=False,
            ))
        y_col_rob = "predicted" if "predicted" in grp_s.columns else grp_s.columns[1]
        fig.add_trace(go.Scatter(
            x=grp_s[exp_col], y=grp_s[y_col_rob],
            mode="lines", name=variant,
            line=dict(color=color, width=2 if variant == "base (df=4)" else 1.5,
                      dash="solid" if variant == "base (df=4)" else "dot"),
        ))

    row_resp = signature[signature["model_name"].str.contains(rv_ev_base, regex=False)].head(1)
    y_label = _rlabel(row_resp["response_variable"].values[0]) if not row_resp.empty else "Response"
    x_label = _elabel(row_resp["exposure_variable"].values[0]) if not row_resp.empty else "Exposure"

    fig.update_layout(height=420, margin=dict(t=20),
                      xaxis_title=x_label, yaxis_title=y_label,
                      legend=dict(orientation="h", y=-0.22))
    st.plotly_chart(fig, use_container_width=True)

    # ── 2. Leave-one-locality-out ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Leave-one-locality-out (LOLO) — stability by locality")
    lolo_m = lolo[lolo["model_name"].str.contains(rv_ev_base, regex=False)].copy()

    if not lolo_m.empty:
        fig2 = go.Figure()
        exp_lolo = lolo_m.columns[0]
        for locality, grp in lolo_m.groupby("group_removed"):
            grp_s = grp.dropna(subset=[exp_lolo]).sort_values(exp_lolo)
            color = _loc_color(str(locality))
            y_col_lolo = "predicted" if "predicted" in grp_s.columns else grp_s.columns[1]
            fig2.add_trace(go.Scatter(
                x=grp_s[exp_lolo], y=grp_s[y_col_lolo],
                mode="lines", name=_port_name(str(locality)),
                line=dict(color=color, width=1.5, dash="dot"),
            ))
        # Curva base
        if base_key:
            base_s = gam_smooth[gam_smooth["model_name"] == base_key[0]].copy()
            exp_b = base_key[0].split("_vs_")[-1] if "_vs_" in base_key[0] else exp_lolo
            x_b = exp_lolo if exp_lolo in base_s.columns else base_s.columns[0]
            base_s = base_s.dropna(subset=[x_b]).sort_values(x_b)
            y_b = "predicted" if "predicted" in base_s.columns else base_s.columns[1]
            fig2.add_trace(go.Scatter(
                x=base_s[x_b], y=base_s[y_b],
                mode="lines", name="Full",
                line=dict(color="#2c3e50", width=2.5),
            ))
        fig2.update_layout(height=380, margin=dict(t=20),
                           xaxis_title=x_label, yaxis_title=y_label,
                           legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig2, use_container_width=True)

    # ── 3. Leave-one-year-out ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Leave-one-year-out (LOYO) — temporal stability")
    loyo_m = loyo[loyo["model_name"].str.contains(rv_ev_base, regex=False)].copy()

    # Detect whether LOYO contains real prediction curves or synthetic R² lines
    _loyo_is_r2 = (
        not loyo_m.empty and
        "predicted_ci_low" in loyo_m.columns and
        loyo_m["predicted_ci_low"].equals(loyo_m["predicted"]) and
        loyo_m["predicted_ci_high"].equals(loyo_m["predicted"])
    ) if not loyo_m.empty else False
    if _loyo_is_r2:
        st.caption("LOYO prediction curves are not available in the current output files. "
                   "Each line represents the R² of the year-removed model — a flat line "
                   "per year. Re-run script 09 with the updated version for full curves.")

    if not loyo_m.empty:
        exp_loyo = loyo_m.columns[0]
        years = sorted(loyo_m["group_removed"].dropna().unique())
        fig3 = go.Figure()
        colorscale = px.colors.sequential.Viridis
        for i, yr in enumerate(years):
            grp = loyo_m[loyo_m["group_removed"] == yr].dropna(subset=[exp_loyo]).sort_values(exp_loyo)
            color = colorscale[int(i / len(years) * (len(colorscale) - 1))]
            y_col_loyo = "predicted" if "predicted" in grp.columns else grp.columns[1]
            fig3.add_trace(go.Scatter(
                x=grp[exp_loyo], y=grp[y_col_loyo],
                mode="lines", name=str(yr),
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.7,
            ))
        if base_key:
            base_s = gam_smooth[gam_smooth["model_name"] == base_key[0]].copy()
            x_b = exp_loyo if exp_loyo in base_s.columns else base_s.columns[0]
            base_s = base_s.dropna(subset=[x_b]).sort_values(x_b)
            y_b = "predicted" if "predicted" in base_s.columns else base_s.columns[1]
            fig3.add_trace(go.Scatter(
                x=base_s[x_b], y=base_s[y_b],
                mode="lines", name="Full",
                line=dict(color="#2c3e50", width=2.5),
            ))
        fig3.update_layout(height=380, margin=dict(t=20),
                           xaxis_title=x_label, yaxis_title=y_label,
                           legend=dict(orientation="h", y=-0.25, font_size=9))
        st.plotly_chart(fig3, use_container_width=True)

    # ── 4. Puntos de influencia ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Influence diagnostics — Cook's D and leverage")
    inf_m = influence[influence["model_name"].str.contains(rv_ev_base, regex=False)].copy()

    if not inf_m.empty:
        inf_m["port_name"] = inf_m["local_canonical"].map(_port_name)
        inf_m["flag"] = inf_m.apply(
            lambda r: ("Cook's D + Residuo" if r.get("high_cooks_d") and r.get("high_student_resid")
                       else ("Cook's D alto" if r.get("high_cooks_d")
                             else ("Leverage alto" if r.get("high_leverage")
                                   else "Normal"))),
            axis=1,
        )
        color_inf = {"Normal": "#bdc3c7", "Cook's D alto": "#e74c3c",
                     "Leverage alto": "#f39c12", "Cook's D + Residuo": "#8e44ad"}

        fig4 = px.scatter(
            inf_m, x="hat_diag", y="cooks_d",
            color="flag", symbol="port_name",
            color_discrete_map=color_inf,
            height=380,
            labels={"hat_diag": "Leverage (h)", "cooks_d": "Cook's D",
                    "port_name": "Locality", "flag": "Type"},
            hover_data=["year", "port_name", "student_resid"],
            size_max=12,
        )
        fig4.update_layout(margin=dict(t=20), legend_title_text="Diagnostic")
        st.plotly_chart(fig4, use_container_width=True)

        n_flag = len(inf_m[inf_m["flag"] != "Normal"])
        if n_flag:
            st.warning(f"{n_flag} observations flagged as influential.")
            with st.expander("View influential observations"):
                disp_inf = inf_m[inf_m["flag"] != "Normal"][
                    ["local_canonical", "year", "cooks_d", "hat_diag",
                     "student_resid", "flag"]].copy()
                disp_inf["local_canonical"] = disp_inf["local_canonical"].map(_port_name)
                disp_inf.columns = ["Locality", "Year", "Cook's D", "Leverage",
                                    "Studentized residual", "Diagnostic"]
                st.dataframe(disp_inf.round(4), hide_index=True, use_container_width=True)

    # ── 5. Model stability: primary vs comparator ─────────────────────────
    st.markdown("---")
    st.markdown("#### Model stability: primary vs comparator")
    st.caption(
        "Rows show the primary model vs each comparator within the same response variable. "
        "Rows where |ΔAIC| < 2 are highlighted — practically equivalent models."
    )

    rob_stability = ad.get("rob_stability", pd.DataFrame())

    if rob_stability.empty:
        st.info("Dataset `rob_stability` (09_model_stability_summary) not yet available.")
    else:
        stab_disp = rob_stability[[
            "response_variable",
            "primary_model_name",
            "primary_predictor_family",
            "comparator_predictor",
            "comparator_predictor_family",
            "delta_aic_primary_minus_comparator",
            "delta_r2_primary_minus_comparator",
            "delta_rmse_primary_minus_comparator",
        ]].copy()
        stab_disp["response_variable"] = stab_disp["response_variable"].map(_rlabel)
        stab_disp = stab_disp.sort_values("delta_aic_primary_minus_comparator",
                                          key=abs, ascending=True)
        stab_disp.columns = [
            "Response", "Primary model", "Primary family",
            "Comparator predictor", "Comparator family",
            "ΔAIC (primary − comp.)", "ΔR²", "ΔRMSE",
        ]
        stab_disp["ΔAIC (primary − comp.)"] = stab_disp["ΔAIC (primary − comp.)"].round(3)
        stab_disp["ΔR²"]   = stab_disp["ΔR²"].round(4)
        stab_disp["ΔRMSE"] = stab_disp["ΔRMSE"].round(4)

        def _highlight_equiv(row):
            """Highlight rows where |ΔAIC| < 2 (practically equivalent)."""
            if abs(row["ΔAIC (primary − comp.)"]) < 2:
                return ["background-color: #d4edda;"] * len(row)
            return [""] * len(row)

        st.dataframe(
            stab_disp.style.apply(_highlight_equiv, axis=1),
            use_container_width=True, hide_index=True,
        )

        # Visual: scatter of ΔAIC vs ΔR²
        fig_stab = px.scatter(
            stab_disp,
            x="ΔAIC (primary − comp.)", y="ΔR²",
            color="Response",
            hover_data=["Comparator predictor", "Comparator family"],
            height=320,
            labels={
                "ΔAIC (primary − comp.)": "ΔAIC (primary − comparator)",
                "ΔR²": "ΔR² (primary − comparator)",
            },
        )
        fig_stab.add_vline(x=-2, line_dash="dash", line_color="#e74c3c",
                           annotation_text="|ΔAIC| = 2", line_width=1)
        fig_stab.add_vline(x=2, line_dash="dash", line_color="#e74c3c", line_width=1)
        fig_stab.add_vline(x=0, line_dash="dot", line_color="#aaa", line_width=1)
        fig_stab.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1)
        fig_stab.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_stab, use_container_width=True)

    # ── 6. Partial dependence diagnostics ────────────────────────────────
    st.markdown("---")
    st.markdown("#### Partial dependence diagnostics")
    st.caption(
        "Low slope sign changes + high effect range = robust non-linear effect. "
        "Monotonicity proxy close to ±1 indicates near-monotone response."
    )

    rob_pd_diag = ad.get("rob_pd_diag", pd.DataFrame())

    if rob_pd_diag.empty:
        st.info("Dataset `rob_pd_diag` (09_partial_dependence_diagnostics) not yet available.")
    else:
        pd_disp = rob_pd_diag[[
            "model_name", "response_variable", "predictor", "predictor_family",
            "n_points", "effect_min", "effect_max", "effect_range",
            "slope_sign_changes", "monotonicity_proxy",
        ]].copy()
        pd_disp["response_variable"] = pd_disp["response_variable"].map(_rlabel)
        pd_disp = pd_disp.sort_values("effect_range", ascending=False)
        pd_disp.columns = [
            "Model", "Response", "Predictor", "Family",
            "N points", "Effect min", "Effect max", "Effect range",
            "Slope sign changes", "Monotonicity proxy",
        ]
        for c in ["Effect min", "Effect max", "Effect range", "Monotonicity proxy"]:
            pd_disp[c] = pd_disp[c].round(4)

        def _highlight_robust(row):
            """Highlight rows with low slope sign changes and high effect range."""
            if row["Slope sign changes"] <= 1 and row["Effect range"] >= pd_disp["Effect range"].quantile(0.5):
                return ["background-color: #cce5ff;"] * len(row)
            return [""] * len(row)

        st.dataframe(
            pd_disp.style.apply(_highlight_robust, axis=1),
            use_container_width=True, hide_index=True,
        )

        # Visual: bubble chart of effect_range vs slope_sign_changes
        col_pd1, col_pd2 = st.columns(2)
        with col_pd1:
            fig_pd1 = px.scatter(
                pd_disp,
                x="Slope sign changes", y="Effect range",
                color="Response",
                size="Effect range",
                hover_data=["Model", "Predictor", "Monotonicity proxy"],
                height=320,
                labels={
                    "Slope sign changes": "Slope sign changes (lower = more stable)",
                    "Effect range": "Effect range (max − min)",
                },
            )
            fig_pd1.update_layout(margin=dict(t=20))
            st.plotly_chart(fig_pd1, use_container_width=True)

        with col_pd2:
            fig_pd2 = px.bar(
                pd_disp.sort_values("Effect range", ascending=True),
                x="Effect range", y="Model", orientation="h",
                color="Response", height=320,
                labels={"Effect range": "Effect range (partial dependence)",
                        "Model": ""},
            )
            fig_pd2.update_layout(margin=dict(t=20))
            st.plotly_chart(fig_pd2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ORDENACIÓN MULTIVARIANTE (módulo 10)
# ══════════════════════════════════════════════════════════════════════════════
def tab_ordination(ad):
    st.markdown('<h3 class="section-title">Multivariate ordination of the fishing community</h3>',
                unsafe_allow_html=True)

    permanova = ad["permanova"].copy()
    dispersion = ad["dispersion"].copy()
    axis_exp = ad["axis_exp"].copy()
    top_sp_axis = ad["top_sp_axis"].copy()
    exp_bins = ad["exp_bins"].copy()

    # Disponibles
    ord_options = {
        "PCoA Hellinger": ad["pcoa_hell"],
        "PCoA Bray-Curtis": ad["pcoa_rel"],
        "NMDS Bray-Curtis": ad["nmds_rel"],
    }
    sel_ord = st.radio("Ordination method", list(ord_options.keys()), horizontal=True)
    scores = ord_options[sel_ord].copy()
    scores["port_name"] = scores["local_canonical"].map(_port_name)

    ax1_pct = scores["Axis1_explained"].iloc[0] * 100 if "Axis1_explained" in scores.columns else None
    ax2_pct = scores["Axis2_explained"].iloc[0] * 100 if "Axis2_explained" in scores.columns else None
    ax1_label = f"Axis 1 ({ax1_pct:.1f}%)" if ax1_pct else "Axis 1"
    ax2_label = f"Axis 2 ({ax2_pct:.1f}%)" if ax2_pct else "Axis 2"

    col_top = st.columns([1, 1, 1])
    perm_row = permanova[permanova["distance_basis"].str.contains("BrayCurtis")].iloc[0] \
        if "BrayCurtis" in permanova["distance_basis"].values[0] else permanova.iloc[0]
    col_top[0].metric("PERMANOVA R²", f"{perm_row['r2']:.4f}")
    col_top[1].metric("PERMANOVA p", f"{perm_row['p_value']:.3f}",
                      "sig." if perm_row["p_value"] <= 0.05 else "n.s.")
    col_top[2].metric("pseudo-F", f"{perm_row['pseudo_F']:.2f}")

    st.markdown("---")

    # ── Biplot interactivo ────────────────────────────────────────────────
    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        st.markdown("#### Ordination biplot")

        # Dynamically collect all available exposure_variable values from
        # permanova/dispersion/exp_bins — do NOT hardcode them
        ev_from_perm = permanova["exposure_variable"].dropna().unique().tolist() \
            if "exposure_variable" in permanova.columns else []
        ev_from_bins = exp_bins["exposure_variable"].dropna().unique().tolist() \
            if "exposure_variable" in exp_bins.columns else []
        ev_all = sorted(set(ev_from_perm + ev_from_bins))

        # Also include exposure cols present in scores themselves
        ev_in_scores = [c for c in ev_all if c in scores.columns]
        # Fall back to EXPOSURE_LABELS keys if ev_in_scores is empty
        if not ev_in_scores:
            ev_in_scores = [c for c in EXPOSURE_LABELS if c in scores.columns]

        color_options = ["Locality", "Year"] + [_elabel(c) for c in ev_in_scores]
        color_opt = st.selectbox("Color by", color_options, key="ord_color")

        if color_opt == "Locality":
            fig = px.scatter(
                scores, x="Axis1", y="Axis2",
                color="port_name",
                color_discrete_map=_LOCALITY_COLORS_BY_NAME,
                hover_data=["year"],
                height=450,
                labels={"Axis1": ax1_label, "Axis2": ax2_label, "port_name": "Locality"},
            )
        elif color_opt == "Year":
            fig = px.scatter(
                scores, x="Axis1", y="Axis2",
                color="year",
                color_continuous_scale="Viridis",
                hover_data=["port_name"],
                height=450,
                labels={"Axis1": ax1_label, "Axis2": ax2_label},
            )
        else:
            exp_sel = ev_in_scores[[_elabel(c) for c in ev_in_scores].index(color_opt)]
            fig = px.scatter(
                scores.dropna(subset=[exp_sel]), x="Axis1", y="Axis2",
                color=exp_sel,
                color_continuous_scale="RdYlGn_r",
                hover_data=["port_name", "year"],
                height=450,
                labels={"Axis1": ax1_label, "Axis2": ax2_label, exp_sel: color_opt},
            )

        # Elipses por localidad
        for local, grp in scores.groupby("port_name"):
            if len(grp) >= 3:
                cx, cy = grp["Axis1"].mean(), grp["Axis2"].mean()
                sx, sy = grp["Axis1"].std(), grp["Axis2"].std()
                theta = np.linspace(0, 2 * np.pi, 60)
                ex = cx + sx * np.cos(theta)
                ey = cy + sy * np.sin(theta)
                color = _loc_color(local)
                fig.add_trace(go.Scatter(
                    x=ex, y=ey, mode="lines",
                    line=dict(color=color, width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
        fig.add_hline(y=0, line_dash="dash", line_color="#ddd", line_width=0.8)
        fig.add_vline(x=0, line_dash="dash", line_color="#ddd", line_width=0.8)
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # ── Asociaciones eje ↔ exposición ─────────────────────────────────
        st.markdown("#### Axis ↔ exposure correlations (|ρ|)")
        ax_sub = axis_exp[axis_exp["ordination"].str.contains(
            "BrayCurtis" if "Bray" in sel_ord else "hellinger", case=False)].copy()
        ax_sub["exp_label"] = ax_sub["exposure_variable"].map(_elabel)
        ax_sub["ax_label"] = ax_sub["axis"] + " · " + ax_sub["exp_label"]
        ax_sub = ax_sub.sort_values("abs_spearman_corr", ascending=True)

        fig_ax = go.Figure(go.Bar(
            x=ax_sub["spearman_corr"],
            y=ax_sub["ax_label"],
            orientation="h",
            marker_color=ax_sub["spearman_corr"].apply(
                lambda x: "#e74c3c" if x < 0 else "#27ae60"),
            text=ax_sub["spearman_corr"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
        ))
        fig_ax.add_vline(x=0, line_dash="dash", line_color="#aaa", line_width=1)
        fig_ax.update_layout(height=200, margin=dict(t=20, b=10),
                             xaxis_title="ρ Spearman", yaxis_title="")
        st.plotly_chart(fig_ax, use_container_width=True)

        # ── Dispersión beta por tertil de exposición ───────────────────────
        st.markdown("#### β-dispersion by exposure tertile")
        disp_sub = dispersion[dispersion["ordination_basis"].str.contains(
            "BrayCurtis" if "Bray" in sel_ord else "hellinger", case=False)].copy()
        if not disp_sub.empty:
            fig_disp = go.Figure()
            for exp_var, grp in disp_sub.groupby("exposure_variable"):
                fig_disp.add_trace(go.Bar(
                    x=grp["exposure_bin"],
                    y=grp["dispersion_mean"],
                    error_y=dict(type="data", array=grp["dispersion_sd"].tolist()),
                    name=_elabel(exp_var),
                ))
            fig_disp.update_layout(height=200, margin=dict(t=20, b=10),
                                   barmode="group", yaxis_title="Mean dispersion",
                                   xaxis_title="Exposure tertile",
                                   legend=dict(font_size=9))
            st.plotly_chart(fig_disp, use_container_width=True)

    # ── Especies asociadas a los ejes ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Species most strongly associated with ordination axes (|ρ|)")
    sp_ax_sub = top_sp_axis[top_sp_axis["ordination"].str.contains(
        "BrayCurtis" if "Bray" in sel_ord else "hellinger", case=False)].copy()

    if not sp_ax_sub.empty:
        for axis, grp in sp_ax_sub.groupby("axis"):
            grp = grp.sort_values("abs_spearman_corr", ascending=False).head(10)
            fig_sp = go.Figure(go.Bar(
                x=grp["spearman_corr"],
                y=grp["species_name"],
                orientation="h",
                marker_color=grp["spearman_corr"].apply(
                    lambda x: "#e74c3c" if x < 0 else "#27ae60"),
                text=grp["spearman_corr"].apply(lambda x: f"{x:.3f}"),
                textposition="outside",
            ))
            fig_sp.add_vline(x=0, line_dash="dash", line_color="#aaa", line_width=1)
            fig_sp.update_layout(
                height=300, margin=dict(t=25, b=10),
                title=f"Species associations with {axis}",
                xaxis_title="ρ Spearman", yaxis_title="",
                font_size=11,
            )
            st.plotly_chart(fig_sp, use_container_width=True)

    # ── Platform x protected-area interaction PERMANOVA ──────────────────
    perm_inter = ad.get("permanova_interaction", pd.DataFrame()).copy()
    interaction_groups = ad.get("interaction_groups", pd.DataFrame()).copy()
    if not perm_inter.empty:
        st.markdown("---")
        st.markdown("#### Platform x protected-area composition screen")
        st.caption(
            "PERMANOVA on combined groups such as platform-distance tertile x "
            "protected-area relation. This is an explicit screen for joint "
            "platform and protected-area structure in species composition."
        )

        plot_df = perm_inter.copy()
        plot_df["interaction_label"] = (
            plot_df["platform_exposure"].map(_elabel)
            + " x "
            + plot_df["pa_exposure"].map(_elabel)
            + " | "
            + plot_df["ordination_basis"].astype(str)
        )
        plot_df = plot_df.sort_values("r2", ascending=True)
        fig_int = go.Figure(go.Bar(
            x=plot_df["r2"],
            y=plot_df["interaction_label"],
            orientation="h",
            marker_color=plot_df["p_value"].apply(lambda p: "#27ae60" if pd.notna(p) and p <= 0.05 else "#95a5a6"),
            text=plot_df["r2"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—"),
            textposition="outside",
            customdata=np.stack([
                plot_df["pseudo_F"],
                plot_df["p_value"],
                plot_df["n_groups"],
                plot_df["n_rows"],
            ], axis=-1),
            hovertemplate=(
                "R2=%{x:.4f}<br>pseudo-F=%{customdata[0]:.3f}<br>"
                "p=%{customdata[1]:.4f}<br>groups=%{customdata[2]}<br>"
                "N=%{customdata[3]}<extra></extra>"
            ),
        ))
        fig_int.update_layout(
            height=max(260, len(plot_df) * 36),
            margin=dict(t=20, r=80),
            xaxis_title="PERMANOVA R2",
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig_int, use_container_width=True)

        if not interaction_groups.empty:
            grp_counts = (
                interaction_groups.groupby(["exposure_variable", "group_level"], dropna=False)
                .size()
                .reset_index(name="n_locality_years")
            )
            grp_counts["Interaction"] = grp_counts["exposure_variable"].map(_elabel)
            grp_counts = grp_counts.sort_values(["Interaction", "group_level"])
            with st.expander("Combined platform x protected-area group sizes"):
                st.dataframe(
                    grp_counts[["Interaction", "group_level", "n_locality_years"]]
                    .rename(columns={"group_level": "Group", "n_locality_years": "N locality-years"}),
                    use_container_width=True,
                    hide_index=True,
                )

    # ── PERMANOVA completo ────────────────────────────────────────────────
    with st.expander("Full PERMANOVA table"):
        perm_disp = permanova.copy()
        if "exposure_variable" in perm_disp.columns:
            perm_disp["exposure_variable"] = perm_disp["exposure_variable"].map(_elabel)
        perm_disp.columns = [c.replace("_", " ").title() for c in perm_disp.columns]
        st.dataframe(perm_disp.round(4), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — GRADIENTE DE COMPOSICIÓN (módulo 11)
# ══════════════════════════════════════════════════════════════════════════════
def tab_gradient(ad):
    st.markdown('<h3 class="section-title">Primary gradient of catch composition</h3>',
                unsafe_allow_html=True)

    summary   = ad["grad_summary"].copy()
    scores    = ad["grad_scores"].copy()
    turnover  = ad["turnover_top"].copy()
    top_bin   = ad["top_by_bin"].copy()
    mean_ab   = ad["mean_abund_bin"].copy()

    if scores.empty:
        scores = ad.get("pcoa_rel", pd.DataFrame()).copy()
    if top_bin.empty and not mean_ab.empty:
        top_bin = mean_ab.copy()
    if summary.empty and not scores.empty:
        axis_exp = ad.get("axis_exp_full", ad.get("axis_exp", pd.DataFrame())).copy()
        perm = ad.get("permanova_full", ad.get("permanova", pd.DataFrame())).copy()

        if not axis_exp.empty:
            axis_exp = axis_exp[
                axis_exp["axis"].isin([c for c in ["Axis1", "Axis2"] if c in scores.columns])
                & axis_exp["exposure_variable"].isin(scores.columns)
            ].copy()
            if not axis_exp.empty:
                preferred = axis_exp[
                    axis_exp["ordination"].astype(str).str.contains("PCoA_BrayCurtis", na=False)
                    & axis_exp["exposure_variable"].eq("mean_nearest_platform_distance_km")
                ]
                best_axis = (
                    preferred.sort_values("abs_spearman_corr", ascending=False).iloc[0]
                    if not preferred.empty
                    else axis_exp.sort_values("abs_spearman_corr", ascending=False).iloc[0]
                )
                exp_var = best_axis["exposure_variable"]
                axis = best_axis["axis"]
                ordination = best_axis["ordination"]

                r2 = np.nan
                p_value = np.nan
                if not perm.empty:
                    basis_col = "ordination_basis" if "ordination_basis" in perm.columns else "distance_basis"
                    perm_match = perm[perm["exposure_variable"].eq(exp_var)].copy()
                    if basis_col in perm_match.columns:
                        perm_match = perm_match[
                            perm_match[basis_col].astype(str).str.contains("BrayCurtis", na=False)
                        ]
                    if not perm_match.empty:
                        r2 = perm_match.iloc[0].get("r2", np.nan)
                        p_value = perm_match.iloc[0].get("p_value", np.nan)

                summary = pd.DataFrame([{
                    "ordination": ordination,
                    "axis": axis,
                    "exposure_variable": exp_var,
                    "permanova_r2": r2,
                    "permanova_p_value": p_value,
                    "axis_spearman_corr": best_axis.get("spearman_corr", np.nan),
                    "axis_pearson_corr": best_axis.get("pearson_corr", np.nan),
                }])

    if turnover.empty and not mean_ab.empty:
        pivot = mean_ab.pivot_table(
            index="species_name",
            columns="exposure_bin",
            values="mean_relative_abundance",
            aggfunc="mean",
        )
        if {"Q1", "Q3"}.issubset(pivot.columns):
            turnover = pivot[["Q1", "Q3"]].fillna(0).reset_index()
            turnover = turnover.rename(columns={"Q1": "mean_rel_Q1", "Q3": "mean_rel_Q3"})
            turnover["near_group"] = "Q1"
            turnover["far_group"] = "Q3"
            turnover["difference_far_minus_near"] = turnover["mean_rel_Q3"] - turnover["mean_rel_Q1"]
            turnover["abs_difference"] = turnover["difference_far_minus_near"].abs()
            turnover["dominant_in"] = np.where(
                turnover["difference_far_minus_near"] >= 0, "Q3", "Q1"
            )
            turnover = turnover.sort_values("abs_difference", ascending=False).head(20)

    if summary.empty or scores.empty:
        st.info(
            "Primary-gradient outputs are not available. "
            "Upload the Module 11 parquet files or keep the Module 10 ordination outputs "
            "available so this section can be reconstructed."
        )
        return

    # ── KPIs del gradiente ────────────────────────────────────────────────
    row = summary.iloc[0]
    primary_axis_label = scores["primary_axis"].iloc[0] if "primary_axis" in scores.columns else row.get("axis", "Axis1")
    if "primary_axis_score" not in scores.columns and primary_axis_label in scores.columns:
        scores["primary_axis_score"] = scores[primary_axis_label]
    if "primary_axis_score" not in scores.columns:
        st.info("Primary-axis scores are not available for the selected gradient.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ordination", f"{row['ordination'].replace('_',' ')}")
    p_label = row.get("permanova_p_value", np.nan)
    p_delta = f"p={p_label:.3f}" if pd.notna(p_label) else None
    c2.metric("PERMANOVA R²", f"{row['permanova_r2']:.4f}" if pd.notna(row.get("permanova_r2")) else "n/a", p_delta)
    c3.metric("Axis ↔ distance correlation", f"ρ={row['axis_spearman_corr']:.4f}" if pd.notna(row.get("axis_spearman_corr")) else "n/a")
    c4.metric("Exposure variable", _elabel(row["exposure_variable"]))

    st.markdown("---")
    col_l, col_r = st.columns([1.2, 1])

    # ── Scores del gradiente vs exposición ───────────────────────────────
    with col_l:
        st.markdown("#### Primary axis score vs platform distance")
        scores["port_name"] = scores["local_canonical"].map(_port_name)
        exp_var = row["exposure_variable"]

        if exp_var not in scores.columns:
            st.info(f"Exposure variable `{exp_var}` is not available in the ordination scores.")
        else:
            fig = px.scatter(
                scores.dropna(subset=["primary_axis_score", exp_var]),
                x=exp_var,
                y="primary_axis_score",
                color="port_name",
                color_discrete_map=_LOCALITY_COLORS_BY_NAME,
                trendline="ols",
                trendline_scope="overall",
                trendline_color_override="#2c3e50",
                size_max=10,
                height=400,
                labels={
                    exp_var: _elabel(exp_var),
                    "primary_axis_score": f"Score {primary_axis_label}",
                    "port_name": "Locality",
                },
                hover_data=["year"],
            )
            fig.update_layout(margin=dict(t=20), legend_title_text="Locality")
            st.plotly_chart(fig, use_container_width=True)

    # ── Evolución temporal del score ──────────────────────────────────────
    with col_r:
        st.markdown("#### Temporal trend of gradient by locality")
        fig2 = px.line(
            scores, x="year", y="primary_axis_score",
            color="port_name",
            markers=True,
            color_discrete_map=_LOCALITY_COLORS_BY_NAME,
            height=400,
            labels={"primary_axis_score": f"Score {primary_axis_label}",
                    "year": "Year", "port_name": "Locality"},
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="#aaa", line_width=1)
        fig2.update_layout(margin=dict(t=20), legend_title_text="Locality")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Turnover de especies ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Species turnover along the gradient")
    st.caption("Change in relative abundance between the nearest (Q1) "
               "and farthest (Q3) tertile from oil platforms.")

    if turnover.empty:
        st.info("Species-turnover outputs are not available for the primary gradient.")
    else:
        turn_plot = turnover.sort_values("difference_far_minus_near").copy()
        turn_plot["color"] = turn_plot["difference_far_minus_near"].apply(
            lambda x: "#e74c3c" if x < 0 else "#27ae60")
        _has_corr = "axis_spearman_corr" in turn_plot.columns
        turn_plot["label"] = turn_plot["species_name"] + (
            turn_plot["axis_spearman_corr"].apply(lambda r: f"  (ρ={r:.2f})")
            if _has_corr else "")

        fig3 = go.Figure(go.Bar(
            x=turn_plot["difference_far_minus_near"],
            y=turn_plot["label"],
            orientation="h",
            marker_color=turn_plot["color"],
            text=turn_plot["difference_far_minus_near"].apply(lambda x: f"{x:+.3f}"),
            textposition="outside",
            customdata=np.stack([
                turn_plot["mean_rel_Q1"],
                turn_plot["mean_rel_Q3"],
                turn_plot["dominant_in"],
                turn_plot["species_total_ton"] if "species_total_ton" in turn_plot.columns
                else np.zeros(len(turn_plot)),
            ], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Change Q3−Q1: %{x:.4f}<br>"
                "Rel. abundance Q1: %{customdata[0]:.4f}<br>"
                "Rel. abundance Q3: %{customdata[1]:.4f}<br>"
                "Dominant in: %{customdata[2]}<br>"
                "Historical total: %{customdata[3]:.0f} t<extra></extra>"
            ),
        ))
        fig3.add_vline(x=0, line_dash="dash", line_color="#aaa", line_width=1)
        fig3.update_layout(
            height=max(420, len(turn_plot) * 28),
            margin=dict(t=20, r=80),
            xaxis_title="Relative abundance difference (Q3 − Q1)",
            yaxis_title="",
            annotations=[
                dict(x=turn_plot["difference_far_minus_near"].max() * 0.7,
                     y=len(turn_plot) - 0.5,
                     text="More abundant far from platforms",
                     showarrow=False, font=dict(size=10, color="#27ae60")),
                dict(x=turn_plot["difference_far_minus_near"].min() * 0.7,
                     y=len(turn_plot) - 0.5,
                     text="More abundant near platforms",
                     showarrow=False, font=dict(size=10, color="#e74c3c")),
            ],
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Abundancias por tertil ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Catch composition by exposure tertile")
    top_bin["species_label"] = top_bin["species_name"]

    col_bins = st.columns(3)
    bin_colors = {"Q1": "#e74c3c", "Q2": "#f39c12", "Q3": "#27ae60"}
    bin_labels = {
        "Q1": "Q1 — Closest to platforms",
        "Q2": "Q2 — Intermediate distance",
        "Q3": "Q3 — Farthest from platforms",
    }
    if top_bin.empty:
        st.info("Tertile-level species abundance outputs are not available.")
    else:
        for i, (bin_val, col) in enumerate(zip(["Q1", "Q2", "Q3"], col_bins)):
            grp = top_bin[top_bin["exposure_bin"] == bin_val].sort_values(
                "mean_relative_abundance", ascending=True)
            with col:
                st.markdown(f"**{bin_labels[bin_val]}**")
                if grp.empty:
                    st.info("No data for this tertile.")
                    continue
                fig_b = go.Figure(go.Bar(
                    x=grp["mean_relative_abundance"],
                    y=grp["species_label"],
                    orientation="h",
                    marker_color=bin_colors[bin_val],
                    text=grp["mean_relative_abundance"].apply(lambda x: f"{x:.3f}"),
                    textposition="outside",
                ))
                fig_b.update_layout(
                    height=350, margin=dict(t=10, b=10, l=10, r=40),
                    xaxis_title="Relative abundance",
                    yaxis_title="",
                    font_size=10,
                )
                st.plotly_chart(fig_b, use_container_width=True)

    # ── Heatmap abundancia completo por especie × tertil ─────────────────
    st.markdown("---")
    st.markdown("#### Relative abundance heatmap — all species × tertile")
    if mean_ab.empty:
        st.info("Mean relative abundance by primary bin is not available.")
    else:
        mean_ab_pivot = mean_ab.pivot_table(
            index="species_name", columns="exposure_bin",
            values="mean_relative_abundance", aggfunc="mean"
        ).fillna(0)
        mean_ab_pivot = mean_ab_pivot.loc[
            mean_ab_pivot.max(axis=1).sort_values(ascending=False).index
        ]

        fig_heat = px.imshow(
            mean_ab_pivot.values,
            x=[_zone_label(c) for c in mean_ab_pivot.columns],
            y=mean_ab_pivot.index.tolist(),
            color_continuous_scale="YlOrRd",
            text_auto=".3f",
            labels={"color": "Rel. abundance"},
            height=max(500, len(mean_ab_pivot) * 22),
            aspect="auto",
        )
        fig_heat.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    # ══ Protected Area gradient ═══════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Protected area (AMP) composition gradient")

    pa_turnover = ad.get("pa_turnover", pd.DataFrame())
    pa_abund    = ad.get("pa_abund", pd.DataFrame())

    # ── PA species turnover ───────────────────────────────────────────────
    st.markdown("#### Species turnover: near AMP (Q1) vs far AMP (Q3)")
    st.caption(
        "Top 10 species by absolute relative abundance difference between Q1 (near APA/RDS) "
        "and Q3 (far from APA/RDS). Colored by which group the species dominates."
    )

    if pa_turnover.empty:
        st.info("Dataset `pa_turnover` (11_pa_species_turnover_summaries) not yet available.")
    else:
        # Filter to AMP exposure variable rows if multiple exist
        pa_ev_options = pa_turnover["exposure_variable"].dropna().unique().tolist() \
            if "exposure_variable" in pa_turnover.columns else [None]
        if len(pa_ev_options) > 1:
            sel_pa_ev = st.selectbox(
                "AMP exposure variable",
                options=pa_ev_options,
                format_func=_elabel,
                key="pa_turnover_ev",
            )
            pa_turn_sub = pa_turnover[pa_turnover["exposure_variable"] == sel_pa_ev].copy()
        else:
            pa_turn_sub = pa_turnover.copy()

        if pa_turn_sub.empty:
            st.info("No data for the selected exposure variable.")
        else:
            # Top 10 species by abs_difference
            top10 = pa_turn_sub.nlargest(10, "abs_difference").copy()
            top10 = top10.sort_values("abs_difference", ascending=True)

            dom_color_map = {
                "Q1": "#e74c3c",
                "Q3": "#27ae60",
                "Q1 (near)": "#e74c3c",
                "Q3 (far)": "#27ae60",
                "near": "#e74c3c",
                "far":  "#27ae60",
            }
            top10["bar_color"] = top10["dominant_in"].map(
                lambda v: dom_color_map.get(str(v), "#8e44ad"))

            fig_pa_turn = go.Figure(go.Bar(
                x=top10["difference_far_minus_near"],
                y=top10["species_name"],
                orientation="h",
                marker_color=top10["bar_color"],
                text=top10["difference_far_minus_near"].apply(lambda v: f"{v:+.3f}"),
                textposition="outside",
                customdata=np.stack([
                    top10["mean_rel_Q1"],
                    top10["mean_rel_Q3"],
                    top10["dominant_in"].astype(str),
                    top10.get("abs_difference", top10["difference_far_minus_near"].abs()),
                ], axis=-1),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Far − Near: %{x:.4f}<br>"
                    "Rel. abund. Q1 (near): %{customdata[0]:.4f}<br>"
                    "Rel. abund. Q3 (far): %{customdata[1]:.4f}<br>"
                    "Dominant in: %{customdata[2]}<extra></extra>"
                ),
            ))
            fig_pa_turn.add_vline(x=0, line_dash="dash", line_color="#aaa", line_width=1)
            fig_pa_turn.update_layout(
                height=max(380, len(top10) * 36),
                margin=dict(t=20, r=80),
                xaxis_title="Relative abundance difference (far − near AMP)",
                yaxis_title="",
                annotations=[
                    dict(x=top10["difference_far_minus_near"].max() * 0.7 if top10["difference_far_minus_near"].max() > 0 else 0.01,
                         y=len(top10) - 0.5,
                         text="More abundant far from AMP",
                         showarrow=False, font=dict(size=10, color="#27ae60")),
                    dict(x=top10["difference_far_minus_near"].min() * 0.7 if top10["difference_far_minus_near"].min() < 0 else -0.01,
                         y=len(top10) - 0.5,
                         text="More abundant near AMP",
                         showarrow=False, font=dict(size=10, color="#e74c3c")),
                ],
            )
            st.plotly_chart(fig_pa_turn, use_container_width=True)

            # Sortable full table below the chart
            with st.expander("Full AMP species turnover table"):
                # Show per-APA columns if present
                pa_cols_base = [
                    "species_name", "mean_rel_Q1", "mean_rel_Q3",
                    "difference_far_minus_near", "abs_difference", "dominant_in",
                ]
                pa_cols_extra = [c for c in [
                    "near_group", "far_group",
                    "mean_rel_inside_apa",
                    "mean_rel_outside_closer_to_rds",
                    "mean_rel_APA_DUNAS_DO_ROSADO",
                    "mean_rel_RDS_PONTA_DO_TUBARAO",
                ] if c in pa_turn_sub.columns]
                pa_show_cols = [c for c in pa_cols_base + pa_cols_extra if c in pa_turn_sub.columns]
                pa_full = pa_turn_sub[pa_show_cols].copy()
                pa_full = pa_full.sort_values("abs_difference", ascending=False)
                # Round numeric columns
                for c in pa_full.select_dtypes(include="number").columns:
                    pa_full[c] = pa_full[c].round(4)
                st.dataframe(pa_full, use_container_width=True, hide_index=True)

    # ── PA mean relative abundance by group (Q1/Q2/Q3/Q4) ─────────────────
    st.markdown("#### Mean relative abundance by AMP exposure group")
    st.caption(
        "Mean relative abundance per species across Q1/Q2/Q3/Q4 quartile bins "
        "of the AMP exposure variable."
    )

    if pa_abund.empty:
        st.info("Dataset `pa_abund` (11_pa_mean_relative_abundance_by_group) not yet available.")
    else:
        # Selector for exposure_variable
        pa_abund_ev_options = pa_abund["exposure_variable"].dropna().unique().tolist() \
            if "exposure_variable" in pa_abund.columns else [None]
        if len(pa_abund_ev_options) > 1:
            sel_pa_abund_ev = st.selectbox(
                "AMP exposure variable (abundance)",
                options=pa_abund_ev_options,
                format_func=_elabel,
                key="pa_abund_ev",
            )
            pa_abund_sub = pa_abund[pa_abund["exposure_variable"] == sel_pa_abund_ev].copy()
        else:
            pa_abund_sub = pa_abund.copy()

        if not pa_abund_sub.empty and "species_name" in pa_abund_sub.columns:
            # Pivot: rows = species, columns = group_level (Q1/Q2/Q3/Q4)
            pa_abund_pivot = pa_abund_sub.pivot_table(
                index="species_name",
                columns="group_level",
                values="mean_relative_abundance",
                aggfunc="mean",
            ).fillna(0)
            # Order columns logically if present
            col_order = [c for c in ["Q1", "Q2", "Q3", "Q4"] if c in pa_abund_pivot.columns]
            if not col_order:
                col_order = pa_abund_pivot.columns.tolist()
            pa_abund_pivot = pa_abund_pivot[col_order]
            # Sort species by total abundance descending
            pa_abund_pivot = pa_abund_pivot.loc[
                pa_abund_pivot.sum(axis=1).sort_values(ascending=False).index
            ]

            fig_pa_heat = px.imshow(
                pa_abund_pivot.values,
                x=col_order,
                y=pa_abund_pivot.index.tolist(),
                color_continuous_scale="YlOrRd",
                text_auto=".3f",
                labels={"color": "Rel. abundance"},
                height=max(450, len(pa_abund_pivot) * 22),
                aspect="auto",
            )
            fig_pa_heat.update_layout(
                margin=dict(t=20),
                xaxis_title="AMP exposure quartile group",
                yaxis_title="",
                coloraxis_colorbar_title="Rel. abund.",
            )
            st.plotly_chart(fig_pa_heat, use_container_width=True)

            # Bar charts per group (top species per quartile)
            st.markdown("##### Top species per AMP exposure group")
            n_groups = len(col_order)
            grp_cols = st.columns(min(n_groups, 4))
            grp_pal  = ["#e74c3c", "#f39c12", "#27ae60", "#2980b9"]
            for gi, gname in enumerate(col_order):
                grp_data = pa_abund_sub[pa_abund_sub["group_level"] == gname].copy()
                grp_data = grp_data.sort_values("mean_relative_abundance", ascending=False).head(10)
                grp_data = grp_data.sort_values("mean_relative_abundance", ascending=True)
                with grp_cols[gi % len(grp_cols)]:
                    st.markdown(f"**{gname}**")
                    fig_g = go.Figure(go.Bar(
                        x=grp_data["mean_relative_abundance"],
                        y=grp_data["species_name"],
                        orientation="h",
                        marker_color=grp_pal[gi % len(grp_pal)],
                        text=grp_data["mean_relative_abundance"].apply(lambda v: f"{v:.3f}"),
                        textposition="outside",
                    ))
                    fig_g.update_layout(
                        height=300,
                        margin=dict(t=10, b=10, l=5, r=40),
                        xaxis_title="Rel. abundance",
                        yaxis_title="",
                        font_size=10,
                    )
                    st.plotly_chart(fig_g, use_container_width=True)
        else:
            st.info("Abundance data not in expected format — "
                    "expected columns: species_name, group_level, mean_relative_abundance.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — METHODS & RESULTS (academic synthesis)
# ══════════════════════════════════════════════════════════════════════════════

def tab_protected_areas(ad: dict) -> None:
    """Protected Areas analysis tab."""

    st.markdown(
        '<h3 class="section-title">Protected Areas: Proximity, Fishing Indicators &amp; Composition</h3>',
        unsafe_allow_html=True,
    )

    # ── Guard: required datasets ──────────────────────────────────────────────
    locality_exp  = ad.get("locality_exposure", pd.DataFrame())
    locality_year = ad.get("locality_year_core", pd.DataFrame())
    gam_comp      = ad.get("gam_comparison",    pd.DataFrame())
    pa_turn       = ad.get("pa_turnover",       pd.DataFrame())
    pa_abund      = ad.get("pa_abund",          pd.DataFrame())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Locality–PA proximity map
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Section 1 — Locality–PA proximity")
    st.caption(
        "Mean distance from each locality's landing points to each protected area, "
        "and the dominant spatial relationship category."
    )

    if locality_exp.empty:
        st.info("Locality exposure data not available.")
    else:
        le = locality_exp.copy()

        # Derive dominant_relation for every locality
        rel_cols_present = [c for c in _RELATION_COLS if c in le.columns]
        if rel_cols_present:
            le["dominant_relation"] = le.apply(_derive_dominant_relation, axis=1)
        else:
            le["dominant_relation"] = "outside_between_both"

        # ── Summary table ─────────────────────────────────────────────────────
        table_cols = {
            "local_canonical":                              "Locality",
            "distance_to_apa_dunas_do_rosado_km_mean":      "Mean dist. APA Dunas (km)",
            "distance_to_rds_ponta_do_tubarao_km_mean":     "Mean dist. RDS Tubarão (km)",
            "dominant_relation":                            "Dominant relation",
            "share_landings_inside_apa_dunas_do_rosado":    "% landings inside APA",
            "share_landings_inside_rds_ponta_do_tubarao":   "% landings inside RDS",
        }
        disp_cols = [c for c in table_cols if c in le.columns]
        tbl = le[disp_cols].copy()
        tbl = tbl.rename(columns={c: table_cols[c] for c in disp_cols})

        # Format share columns as percentages
        for pct_col in ["% landings inside APA", "% landings inside RDS"]:
            if pct_col in tbl.columns:
                tbl[pct_col] = (tbl[pct_col] * 100).round(1)

        # Human-readable dominant relation
        if "Dominant relation" in tbl.columns:
            tbl["Dominant relation"] = tbl["Dominant relation"].map(_zone_label)

        # Round numeric columns
        for num_col in ["Mean dist. APA Dunas (km)", "Mean dist. RDS Tubarão (km)"]:
            if num_col in tbl.columns:
                tbl[num_col] = tbl[num_col].round(2)

        tbl = tbl.sort_values("Mean dist. APA Dunas (km)") if "Mean dist. APA Dunas (km)" in tbl.columns else tbl
        st.dataframe(tbl, hide_index=True, use_container_width=True)

        # ── Bar chart: mean distance to nearest PA coloured by dominant_relation ──
        st.markdown("#### Mean distance to nearest protected area by locality")
        dist_col = "distance_to_nearest_protected_area_km_mean"
        if dist_col not in le.columns:
            # Fall back to APA distance if nearest-PA mean is absent
            dist_col = "distance_to_apa_dunas_do_rosado_km_mean"

        bar_df = le[["local_canonical", dist_col, "dominant_relation"]].dropna(subset=[dist_col]).copy()
        bar_df = bar_df.sort_values(dist_col, ascending=True)
        bar_df["zone_label"] = bar_df["dominant_relation"].map(_zone_label)
        bar_df["color"]      = bar_df["dominant_relation"].map(_zone_color)

        fig1 = go.Figure(go.Bar(
            x=bar_df[dist_col],
            y=bar_df["local_canonical"],
            orientation="h",
            marker_color=bar_df["color"],
            text=bar_df[dist_col].round(2).astype(str) + " km",
            textposition="outside",
            customdata=bar_df["zone_label"],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Mean distance to nearest PA: %{x:.2f} km<br>"
                "Dominant zone: %{customdata}<extra></extra>"
            ),
        ))
        fig1.update_layout(
            height=max(280, len(bar_df) * 52),
            margin=dict(t=20, b=20, r=120),
            xaxis_title="Mean distance to nearest protected area (km)",
            yaxis_title="",
            font_size=12,
        )

        # Manual legend via invisible scatter traces
        added_zones = set()
        for _, row in bar_df.iterrows():
            rel = row["dominant_relation"]
            if rel not in added_zones:
                fig1.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=10, color=_zone_color(rel), symbol="square"),
                    name=_zone_label(rel),
                    showlegend=True,
                ))
                added_zones.add(rel)
        fig1.update_layout(legend=dict(orientation="h", y=-0.25, x=0, font_size=11))
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Fishing indicators by PA zone
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Section 2 — Fishing indicators by PA zone")
    st.caption(
        "Catch metrics across locality-years, grouped by each locality's dominant "
        "protected-area relation (all years pooled)."
    )

    if locality_exp.empty or locality_year.empty:
        st.info("Locality exposure or locality-year core data not available.")
    else:
        # Build dominant_relation for each locality (re-use from section 1 or recompute)
        le2 = locality_exp.copy()
        rel_cols_present = [c for c in _RELATION_COLS if c in le2.columns]
        if rel_cols_present:
            le2["dominant_relation"] = le2.apply(_derive_dominant_relation, axis=1)
        else:
            le2["dominant_relation"] = "outside_between_both"

        zone_map = le2.set_index("local_canonical")["dominant_relation"].to_dict()

        # Join locality_year_core with zone
        ly = locality_year.copy()
        ly["dominant_relation"] = ly["local_canonical"].map(zone_map)
        ly = ly.dropna(subset=["dominant_relation"])
        ly["zone_label"] = ly["dominant_relation"].map(_zone_label)
        ly["zone_color"]  = ly["dominant_relation"].map(_zone_color)

        if ly.empty:
            st.info("No matching records after joining locality exposure zones.")
        else:
            # Ordered zone list
            zones_present = [z for z in _ZONE_COLORS if z in ly["dominant_relation"].unique()]
            zone_labels_ordered = [_zone_label(z) for z in zones_present]

            # 3-column layout: one boxplot per production metric
            metrics = [
                ("production_per_trip_ton",   "Production per trip (t)"),
                ("production_ton",            "Total production (t)"),
                ("production_per_fisher_ton", "Production per fisher (t)"),
            ]
            cols = st.columns(3)
            for (metric_col, metric_label), col in zip(metrics, cols):
                if metric_col not in ly.columns:
                    with col:
                        st.caption(f"{metric_label} — not available")
                    continue
                with col:
                    st.markdown(f"**{metric_label}**")
                    fig_box = go.Figure()
                    for zone_key in zones_present:
                        grp = ly[ly["dominant_relation"] == zone_key][metric_col].dropna()
                        if grp.empty:
                            continue
                        fig_box.add_trace(go.Box(
                            y=grp,
                            name=_zone_label(zone_key),
                            marker_color=_zone_color(zone_key),
                            boxmean="sd",
                            showlegend=False,
                        ))
                    fig_box.update_layout(
                        height=360,
                        margin=dict(t=15, b=10, l=10, r=10),
                        yaxis_title=metric_label,
                        xaxis_title="",
                        font_size=11,
                        xaxis=dict(tickangle=-30),
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

            # Kruskal–Wallis significance note (from assoc_categorical if available)
            assoc_cat = ad.get("assoc_categorical", pd.DataFrame())
            if not assoc_cat.empty:
                pa_assoc = assoc_cat[
                    assoc_cat["exposure_variable"].str.contains("protected_area", na=False)
                ].copy()
                if not pa_assoc.empty:
                    st.markdown("**Kruskal–Wallis tests (PA zone vs fishing metrics)**")
                    pa_assoc_disp = pa_assoc[[
                        "response_variable", "exposure_column", "n_complete",
                        "kruskal_h", "kruskal_p_value", "p_value_adj_fdr", "significant_fdr_05",
                    ]].copy()
                    pa_assoc_disp["response_variable"] = pa_assoc_disp["response_variable"].map(
                        lambda x: _RESP_LABELS.get(x, x)
                    )
                    pa_assoc_disp.columns = [
                        "Response variable", "Predictor column", "N",
                        "Kruskal H", "p-value", "p-adj (FDR)", "Sig. (FDR 0.05)",
                    ]
                    st.dataframe(pa_assoc_disp.round({"Kruskal H": 3, "p-value": 5, "p-adj (FDR)": 5}),
                                 hide_index=True, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — GAM model comparison: PA vs platform predictors
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Section 3 — GAM model comparison: PA vs platform predictors")
    st.caption(
        "Comparison of model fit (R², AIC) for the four key response variables across "
        "predictor families. Lower AIC indicates a better-fitting model."
    )

    if gam_comp.empty:
        st.info("GAM model comparison data not available.")
    else:
        target_rvs = ["pielou_species", "shannon_species", "production_per_trip_ton", "production_ton"]
        gc = gam_comp[gam_comp["response_variable"].isin(target_rvs)].copy()

        if gc.empty:
            st.info("No GAM comparison rows for the selected response variables.")
        else:
            # Compute delta-AIC vs best model per response variable
            best_aic = gc.groupby("response_variable")["aic"].min().rename("best_aic")
            gc = gc.join(best_aic, on="response_variable")
            gc["delta_aic"] = gc["aic"] - gc["best_aic"]
            gc["is_best"]   = gc["delta_aic"] < 1e-6  # flag best model(s)

            # Aggregate: best (lowest AIC) per response × predictor_family
            # Show both GAM and linear for context
            agg = (
                gc.sort_values("aic")
                  .groupby(["response_variable", "predictor_family", "model_type"], as_index=False)
                  .first()
            )

            # Display grouped table
            st.markdown("#### Model fit summary (best per response × family × model type)")
            disp = agg[[
                "response_variable", "predictor_family", "model_type",
                "r_squared", "aic", "delta_aic", "is_best",
            ]].copy()
            disp["response_variable"]  = disp["response_variable"].map(lambda x: _RESP_LABELS.get(x, x))
            disp["predictor_family"]   = disp["predictor_family"].map(lambda x: _FAM_LABELS.get(x, x))
            disp["model_type"]         = disp["model_type"].str.replace("_", " ").str.title()
            disp.columns = [
                "Response variable", "Predictor family", "Model type",
                "R²", "AIC", "ΔAIC", "Best model",
            ]
            disp["R²"]   = disp["R²"].round(3)
            disp["AIC"]  = disp["AIC"].round(2)
            disp["ΔAIC"] = disp["ΔAIC"].round(2)
            disp = disp.sort_values(["Response variable", "ΔAIC"])

            # Style to highlight best model
            def _highlight_best(row):
                if row["Best model"]:
                    return ["background-color: #d5f5e3"] * len(row)
                return [""] * len(row)

            st.dataframe(
                disp.style.apply(_highlight_best, axis=1),
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("#### R² by predictor family and response variable")
            # Use best GAM per family (lowest AIC within gam_penalised)
            gam_only = gc[gc["model_type"] == "gam_penalised"].copy()
            gam_best_per_fam = (
                gam_only.sort_values("aic")
                         .groupby(["response_variable", "predictor_family"], as_index=False)
                         .first()
            )

            r2_fig = go.Figure()
            families_ordered = [
                "categorical", "continuous",
                "platform_inside_interaction", "tensor_interaction",
            ]
            fam_colors = {
                "categorical":                 _PAL_NEUTRAL,
                "continuous":                  _PAL_PRIMARY,
                "platform_inside_interaction": _PAL_SECONDARY,
                "tensor_interaction":          _PAL_ACCENT,
            }

            rv_tick_labels = [_RESP_LABELS.get(rv, rv) for rv in target_rvs]

            for fam in families_ordered:
                sub = gam_best_per_fam[gam_best_per_fam["predictor_family"] == fam].copy()
                if sub.empty:
                    continue
                # Align to target_rvs order
                sub = sub.set_index("response_variable").reindex(target_rvs).reset_index()
                r2_fig.add_trace(go.Bar(
                    x=rv_tick_labels,
                    y=sub["r_squared"],
                    name=_FAM_LABELS.get(fam, fam),
                    marker_color=fam_colors.get(fam, _PAL_NEUTRAL),
                ))
            r2_fig.update_layout(
                barmode="group",
                height=380,
                margin=dict(t=20, b=60),
                yaxis_title="R² (GAM, penalised)",
                xaxis_title="",
                legend=dict(orientation="h", y=-0.28, font_size=11),
                font_size=12,
            )
            st.plotly_chart(r2_fig, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Species turnover along PA gradient
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Section 4 — Species turnover along PA gradient")
    st.caption(
        "Change in relative catch abundance between the nearest (Q1) and farthest (Q3) "
        "tertile of localities by distance to the nearest protected area. "
        "Blue bars = more abundant near protected areas (Q1); "
        "orange bars = more abundant far from protected areas (Q3)."
    )

    if pa_turn.empty:
        st.info("PA species turnover data not available.")
    else:
        # Filter to the PA distance gradient
        turn_pa = pa_turn[
            pa_turn["exposure_variable"] == "mean_distance_to_nearest_protected_area_km"
        ].copy()

        if turn_pa.empty:
            st.info("No turnover data for 'mean_distance_to_nearest_protected_area_km'.")
        else:
            # Top 15 species by absolute difference
            top15 = turn_pa.nlargest(15, "abs_difference").copy()
            # Sort by difference (negative = more in Q1/near, positive = more in Q3/far)
            top15 = top15.sort_values("difference_far_minus_near")

            # Colour by dominant_in: Q1 (near PA) → primary blue, Q3 (far) → secondary orange
            top15["bar_color"] = top15["dominant_in"].map(
                {"Q1": _PAL_PRIMARY, "Q3": _PAL_SECONDARY}
            ).fillna(_PAL_NEUTRAL)

            top15["text_val"] = top15["difference_far_minus_near"].apply(lambda x: f"{x:+.3f}")

            fig_turn = go.Figure(go.Bar(
                x=top15["difference_far_minus_near"],
                y=top15["species_name"],
                orientation="h",
                marker_color=top15["bar_color"],
                text=top15["text_val"],
                textposition="outside",
                customdata=np.stack([
                    top15["mean_rel_Q1"],
                    top15["mean_rel_Q3"],
                    top15["dominant_in"].fillna("—"),
                    top15["abs_difference"],
                ], axis=-1),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Difference (far − near): %{x:+.4f}<br>"
                    "Rel. abundance Q1 (near PA): %{customdata[0]:.4f}<br>"
                    "Rel. abundance Q3 (far PA): %{customdata[1]:.4f}<br>"
                    "Dominant in: %{customdata[2]}<br>"
                    "Abs. difference: %{customdata[3]:.4f}<extra></extra>"
                ),
            ))
            fig_turn.add_vline(x=0, line_dash="dash", line_color="#aaa", line_width=1.5)
            fig_turn.update_layout(
                height=max(420, len(top15) * 32),
                margin=dict(t=20, r=100, l=10),
                xaxis_title="Relative abundance difference (far − near)",
                yaxis_title="",
                font_size=12,
                annotations=[
                    dict(
                        x=top15["difference_far_minus_near"].max() * 0.7 if top15["difference_far_minus_near"].max() > 0 else 0.05,
                        y=len(top15) - 0.5,
                        text="↑ More abundant far from PA",
                        showarrow=False,
                        font=dict(size=10, color=_PAL_SECONDARY),
                    ),
                    dict(
                        x=top15["difference_far_minus_near"].min() * 0.7 if top15["difference_far_minus_near"].min() < 0 else -0.05,
                        y=0.5,
                        text="↑ More abundant near PA",
                        showarrow=False,
                        font=dict(size=10, color=_PAL_PRIMARY),
                    ),
                ],
            )
            st.plotly_chart(fig_turn, use_container_width=True)

            # ── Companion table: PA-specific mean relative abundances ──────────
            # These columns are non-NaN in the dominant_protected_area_relation subset,
            # but may be NaN here; include them when at least one is non-NaN.
            companion_cols = [
                "mean_rel_inside_apa",
                "mean_rel_outside_closer_to_rds",
                "mean_rel_APA_DUNAS_DO_ROSADO",
                "mean_rel_RDS_PONTA_DO_TUBARAO",
            ]
            companion_col_labels = {
                "species_name":                    "Species",
                "mean_rel_Q1":                     "Rel. abund. Q1 (near)",
                "mean_rel_Q3":                     "Rel. abund. Q3 (far)",
                "difference_far_minus_near":        "Diff. (far − near)",
                "dominant_in":                     "Dominant in",
                "mean_rel_inside_apa":             "Inside APA",
                "mean_rel_outside_closer_to_rds":  "Outside, closer RDS",
                "mean_rel_APA_DUNAS_DO_ROSADO":    "APA Dunas group",
                "mean_rel_RDS_PONTA_DO_TUBARAO":   "RDS Tubarão group",
            }
            # Only include companion cols that have at least one non-NaN value
            valid_companion = [
                c for c in companion_cols
                if c in top15.columns and top15[c].notna().any()
            ]
            base_display_cols = ["species_name", "mean_rel_Q1", "mean_rel_Q3",
                                  "difference_far_minus_near", "dominant_in"]
            all_display_cols = base_display_cols + valid_companion
            all_display_cols = [c for c in all_display_cols if c in top15.columns]

            comp_tbl = top15[all_display_cols].copy().sort_values(
                "difference_far_minus_near"
            )
            comp_tbl = comp_tbl.rename(columns={c: companion_col_labels.get(c, c) for c in all_display_cols})

            # Round numeric columns
            numeric_tbl_cols = [c for c in comp_tbl.columns if c not in ("Species", "Dominant in")]
            comp_tbl[numeric_tbl_cols] = comp_tbl[numeric_tbl_cols].round(4)

            with st.expander("Companion table — relative abundances for top-15 species"):
                st.dataframe(comp_tbl, hide_index=True, use_container_width=True)

            # ── PA-zone abundance breakdown (from pa_abund if available) ──────
            if not pa_abund.empty:
                pa_rel = pa_abund[
                    (pa_abund["exposure_variable"] == "dominant_protected_area_relation") &
                    (pa_abund["group_level"].isin(["inside_apa", "outside_closer_to_apa",
                                                    "outside_closer_to_rds"]))
                ].copy()

                if not pa_rel.empty:
                    # Limit to the species in top15 for relevance
                    top_sp = set(top15["species_name"].tolist())
                    pa_rel_top = pa_rel[pa_rel["species_name"].isin(top_sp)].copy()

                    if not pa_rel_top.empty:
                        st.markdown("#### Mean relative abundance by PA zone for top-15 turnover species")
                        pivot = pa_rel_top.pivot_table(
                            index="species_name",
                            columns="group_level",
                            values="mean_relative_abundance",
                            aggfunc="mean",
                        ).round(4)
                        pivot.index.name = "Species"
                        pivot.columns = [_zone_label(c) for c in pivot.columns]
                        st.dataframe(pivot, use_container_width=True)

def tab_methods_results(ad=None):
    """Synthesized Methods and Results in academic journal style."""

    st.markdown('<h3 class="section-title">Methods & Results</h3>', unsafe_allow_html=True)

    if ad is None:
        from utils.analysis_loader import load_analysis
        ad = load_analysis()

    div = ad.get("div_table", pd.DataFrame()).copy()
    best = ad.get("gam_best", pd.DataFrame()).copy()
    comparison = ad.get("gam_comparison", pd.DataFrame()).copy()
    perm_inter = ad.get("permanova_interaction", pd.DataFrame()).copy()
    perm_full = ad.get("permanova_full", ad.get("permanova", pd.DataFrame())).copy()
    axis_exp = ad.get("axis_exp_full", ad.get("axis_exp", pd.DataFrame())).copy()
    mean_ab = ad.get("mean_abund_bin", pd.DataFrame()).copy()
    pa_turnover = ad.get("pa_turnover", pd.DataFrame()).copy()

    n_obs = int(len(div)) if not div.empty else np.nan
    n_localities = int(div["local_canonical"].nunique()) if "local_canonical" in div.columns and not div.empty else np.nan
    year_min = int(div["year"].min()) if "year" in div.columns and div["year"].notna().any() else np.nan
    year_max = int(div["year"].max()) if "year" in div.columns and div["year"].notna().any() else np.nan

    richness_mean = div["species_richness"].mean() if "species_richness" in div.columns else np.nan
    richness_sd = div["species_richness"].std() if "species_richness" in div.columns else np.nan
    shannon_mean = div["shannon_species"].mean() if "shannon_species" in div.columns else np.nan
    shannon_sd = div["shannon_species"].std() if "shannon_species" in div.columns else np.nan
    pielou_mean = div["pielou_species"].mean() if "pielou_species" in div.columns else np.nan
    pielou_sd = div["pielou_species"].std() if "pielou_species" in div.columns else np.nan

    def _fmt(value, digits=3, na="n/a"):
        return na if pd.isna(value) else f"{value:.{digits}f}"

    def _fmt_int(value, na="n/a"):
        return na if pd.isna(value) else f"{int(value):,}"

    def _basis_col(df):
        if "ordination_basis" in df.columns:
            return "ordination_basis"
        if "distance_basis" in df.columns:
            return "distance_basis"
        return None

    def _best_perm_row(df, basis="BrayCurtis"):
        if df.empty:
            return pd.Series(dtype=object)
        out = df.copy()
        bcol = _basis_col(out)
        if bcol:
            subset = out[out[bcol].astype(str).str.contains(basis, na=False)].copy()
            if not subset.empty:
                out = subset
        if "r2" in out.columns:
            out = out.sort_values("r2", ascending=False)
        return out.iloc[0] if not out.empty else pd.Series(dtype=object)

    st.markdown("""
> *Synthesis of the main analytical findings. Statistical details and interactive
> visualizations are available in the preceding analysis tabs. Values in this section
> are read from the current parquet outputs whenever possible.*
""")

    # ── METHODS ──────────────────────────────────────────────────────────────
    st.markdown("## Methods")

    st.markdown("### Study area and analytical unit")
    st.markdown("""
Fisheries landings were analysed as a locality-year panel for coastal fishing
communities in Rio Grande do Norte, Brazil. Each observation links catch diversity,
production, fleet and effort descriptors, offshore oil-platform exposure metrics,
and spatial descriptors of nearby marine protected areas (MPAs). Platform exposure
was represented primarily by mean distance to the nearest platform and by platform
density within distance buffers. Protected-area exposure was represented by distance
to the nearest MPA, nearest/dominant protected-area identity, and categorical
inside/outside or edge-relation classes for APA Dunas do Rosado and RDS Ponta do
Tubarão.
""")

    st.markdown("### Diversity and GAM modelling")
    st.markdown("""
Taxonomic diversity was described using species richness, Shannon diversity and
Pielou evenness, alongside production metrics. Generalized Additive Models (GAMs)
were fitted to evaluate non-linear exposure-response relationships. The candidate
set included simple continuous predictors, categorical protected-area predictors,
platform distance x inside-MPA interactions, and tensor interactions between platform
distance and distance to the nearest protected area. Model comparison used R2, AIC,
RMSE and effective degrees of freedom, with linear models retained as baselines where
available.
""")

    st.markdown("### Community composition")
    st.markdown("""
Species composition was analysed using ordination and permutation tests. PERMANOVA
contrasted compositional differences across exposure groups, including explicit
combined groups for platform distance x protected-area relation. This provides a
multivariate test of whether industrial infrastructure and conservation geography
jointly structure catch composition. Species turnover was summarized from relative
abundance differences between near and far exposure groups when Module 11 outputs
were available.
""")

    st.markdown("---")

    # ── RESULTS ──────────────────────────────────────────────────────────────
    st.markdown("## Results")

    st.markdown("### Fisheries overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", _fmt_int(n_obs), "locality-years")
    col2.metric("Localities", _fmt_int(n_localities))
    col3.metric("Mean richness", f"{_fmt(richness_mean, 1)} ± {_fmt(richness_sd, 1)} spp")
    col4.metric("Study period", f"{_fmt_int(year_min)}-{_fmt_int(year_max)}")

    st.markdown(f"""
Across the current analytical table, mean Shannon diversity was
**{_fmt(shannon_mean, 2)} ± {_fmt(shannon_sd, 2)}** and mean Pielou evenness was
**{_fmt(pielou_mean, 2)} ± {_fmt(pielou_sd, 2)}**. The panel therefore captures
substantial variation in both species richness and catch evenness while preserving
the locality-year structure needed to compare platform and protected-area gradients.
""")

    if not div.empty and "local_canonical" in div.columns:
        loc_cols = [
            "local_canonical", "species_richness", "shannon_species", "pielou_species",
            "production_ton", "production_per_trip_ton", "production_per_fisher_ton",
            "mean_nearest_platform_distance_km", "mean_distance_to_nearest_protected_area_km",
        ]
        loc_cols = [c for c in loc_cols if c in div.columns]
        loc_summary = div[loc_cols].groupby("local_canonical", dropna=False).mean(numeric_only=True).reset_index()
        loc_summary["Locality"] = loc_summary["local_canonical"].map(_port_name)
        rename = {
            "species_richness": "Richness",
            "shannon_species": "Shannon H'",
            "pielou_species": "Pielou J'",
            "production_ton": "Production (t)",
            "production_per_trip_ton": "Production/trip (t)",
            "production_per_fisher_ton": "Production/fisher (t)",
            "mean_nearest_platform_distance_km": "Platform distance (km)",
            "mean_distance_to_nearest_protected_area_km": "Nearest MPA distance (km)",
        }
        show_cols = ["Locality"] + [rename[c] for c in loc_summary.columns if c in rename]
        loc_summary = loc_summary.rename(columns=rename)
        st.markdown("**Table 1.** Mean locality-level diversity, production and spatial exposure.")
        st.dataframe(loc_summary[show_cols].round(3), use_container_width=True, hide_index=True)


    st.markdown("### GAM model performance")
    if best.empty:
        st.info("Best-model outputs from Module 08 are not available.")
    else:
        fam_counts = best["predictor_family"].value_counts().to_dict() if "predictor_family" in best.columns else {}
        top_row = best.sort_values("r_squared", ascending=False).iloc[0]
        interaction_n = int(best["predictor_family"].isin(["platform_inside_interaction", "tensor_interaction"]).sum()) \
            if "predictor_family" in best.columns else 0

        st.markdown(f"""
The current best-model table is dominated by interaction formulations: **{interaction_n} of
{len(best)}** best response-specific models use either a platform x inside-MPA interaction
or a tensor interaction between platform distance and MPA distance. The strongest single
fit was for **{_rlabel(top_row['response_variable'])}**, explained by
**{_elabel(top_row['predictor'])}** (R2 = **{_fmt(top_row['r_squared'], 3)}**,
AIC = **{_fmt(top_row['aic'], 2)}**).
""")

        tbl3 = best[[
            "response_variable", "predictor_family", "predictor", "n_rows", "r_squared", "aic", "rmse"
        ]].copy()
        tbl3["Response"] = tbl3["response_variable"].map(_rlabel)
        tbl3["Predictor family"] = tbl3["predictor_family"].map(_family_label)
        tbl3["Best predictor"] = tbl3["predictor"].map(_elabel)
        tbl3 = tbl3[["Response", "Predictor family", "Best predictor", "n_rows", "r_squared", "aic", "rmse"]]
        tbl3 = tbl3.rename(columns={"n_rows": "N", "r_squared": "R2", "aic": "AIC", "rmse": "RMSE"})
        st.markdown("**Table 2.** Best GAM per response variable.")
        st.dataframe(tbl3.sort_values("R2", ascending=False).round(4), use_container_width=True, hide_index=True)

        fam_text = ", ".join(f"{_family_label(k)}: {v}" for k, v in fam_counts.items())
        st.caption(f"Best-model family counts: {fam_text}.")

    if not comparison.empty and {"model_type", "predictor_family", "r_squared"}.issubset(comparison.columns):
        gam_only = comparison[comparison["model_type"].eq("gam_penalised")].copy()
        if not gam_only.empty:
            fam_summary = (
                gam_only.groupby("predictor_family", dropna=False)
                .agg(
                    n_models=("model_name", "nunique"),
                    mean_r2=("r_squared", "mean"),
                    max_r2=("r_squared", "max"),
                    min_aic=("aic", "min"),
                )
                .reset_index()
                .sort_values("max_r2", ascending=False)
            )
            fam_summary["Predictor family"] = fam_summary["predictor_family"].map(_family_label)
            fam_summary = fam_summary[["Predictor family", "n_models", "mean_r2", "max_r2", "min_aic"]]
            fam_summary = fam_summary.rename(columns={
                "n_models": "N models", "mean_r2": "Mean R2",
                "max_r2": "Best R2", "min_aic": "Best AIC",
            })
            st.markdown("**Table 3.** GAM performance summarized by predictor family.")
            st.dataframe(fam_summary.round(4), use_container_width=True, hide_index=True)

    st.markdown("### Protected-area modulation")
    if not best.empty and "predictor_family" in best.columns:
        inside_best = best[best["predictor_family"].eq("platform_inside_interaction")]
        tensor_best = best[best["predictor_family"].eq("tensor_interaction")]
        inside_responses = ", ".join(_rlabel(v) for v in inside_best["response_variable"].tolist())
        tensor_responses = ", ".join(_rlabel(v) for v in tensor_best["response_variable"].tolist())
        st.markdown(f"""
Protected-area covariates do not behave as peripheral controls in the current run;
they are part of the best explanatory structure. Platform x inside-MPA interactions
were selected for **{inside_responses or 'no response variables'}**, while tensor
platform-distance x MPA-distance interactions were selected for
**{tensor_responses or 'no response variables'}**. This pattern supports the core
interpretation that MPA geography modulates the platform-catch relationship rather
than acting only as an independent spatial covariate.
""")

    st.markdown("### Community composition and interaction PERMANOVA")
    bray_inter = _best_perm_row(perm_inter, "BrayCurtis")
    hell_inter = _best_perm_row(perm_inter, "Euclidean")
    bray_full = _best_perm_row(perm_full, "BrayCurtis")

    if not bray_inter.empty:
        st.markdown(f"""
The explicit platform x protected-area grouping test detected strong community
structure. For Bray-Curtis composition, the interaction grouping explained
**R2 = {_fmt(bray_inter.get('r2'), 4)}** of compositional variation
(pseudo-F = **{_fmt(bray_inter.get('pseudo_F'), 3)}**, p = **{_fmt(bray_inter.get('p_value'), 3)}**,
N = **{_fmt_int(bray_inter.get('n_rows'))}**). The Hellinger version was consistent
(R2 = **{_fmt(hell_inter.get('r2'), 4)}**, pseudo-F = **{_fmt(hell_inter.get('pseudo_F'), 3)}**,
p = **{_fmt(hell_inter.get('p_value'), 3)}**) when available. These results indicate
that the combined spatial position relative to platforms and MPAs is associated with
marked differences in species composition.
""")
    elif not bray_full.empty:
        st.markdown(f"""
PERMANOVA detected significant compositional structure along the available spatial
exposure gradients. For Bray-Curtis composition, the strongest available test explained
R2 = **{_fmt(bray_full.get('r2'), 4)}** (pseudo-F = **{_fmt(bray_full.get('pseudo_F'), 3)}**,
p = **{_fmt(bray_full.get('p_value'), 3)}**).
""")
    else:
        st.info("PERMANOVA outputs are not available.")

    if not perm_inter.empty:
        tbl_perm = perm_inter[[
            "ordination_basis", "pa_exposure", "n_rows", "n_groups", "pseudo_F", "r2", "p_value"
        ]].copy()
        tbl_perm["Protected-area component"] = tbl_perm["pa_exposure"].map(_elabel)
        tbl_perm = tbl_perm.rename(columns={
            "ordination_basis": "Ordination basis",
            "n_rows": "N", "n_groups": "Groups",
            "pseudo_F": "Pseudo-F", "r2": "R2", "p_value": "p-value",
        })
        tbl_perm = tbl_perm[["Ordination basis", "Protected-area component", "N", "Groups", "Pseudo-F", "R2", "p-value"]]
        st.markdown("**Table 4.** PERMANOVA tests for combined platform x protected-area groups.")
        st.dataframe(tbl_perm.round(4), use_container_width=True, hide_index=True)

    if not axis_exp.empty and "abs_spearman_corr" in axis_exp.columns:
        axis_show = axis_exp.sort_values("abs_spearman_corr", ascending=False).head(8).copy()
        axis_show["Exposure"] = axis_show["exposure_variable"].map(_elabel)
        axis_show = axis_show.rename(columns={
            "ordination": "Ordination", "axis": "Axis",
            "spearman_corr": "Spearman rho", "pearson_corr": "Pearson r",
            "abs_spearman_corr": "|rho|",
        })
        st.markdown("**Table 5.** Strongest ordination-axis associations with spatial predictors.")
        st.dataframe(
            axis_show[["Ordination", "Axis", "Exposure", "n_complete", "Spearman rho", "Pearson r", "|rho|"]]
            .rename(columns={"n_complete": "N"})
            .round(4),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Species turnover")
    if not mean_ab.empty and {"species_name", "exposure_bin", "mean_relative_abundance"}.issubset(mean_ab.columns):
        pivot = mean_ab.pivot_table(
            index="species_name", columns="exposure_bin",
            values="mean_relative_abundance", aggfunc="mean"
        ).fillna(0)
        if {"Q1", "Q3"}.issubset(pivot.columns):
            turnover = pivot[["Q1", "Q3"]].reset_index()
            turnover["Delta Q3-Q1"] = turnover["Q3"] - turnover["Q1"]
            turnover["Abs delta"] = turnover["Delta Q3-Q1"].abs()
            turnover["Dominant group"] = np.where(turnover["Delta Q3-Q1"] >= 0, "Q3 (far)", "Q1 (near)")
            top_turn = turnover.sort_values("Abs delta", ascending=False).head(10)
            near_names = ", ".join(top_turn[top_turn["Dominant group"].eq("Q1 (near)")]["species_name"].head(4))
            far_names = ", ".join(top_turn[top_turn["Dominant group"].eq("Q3 (far)")]["species_name"].head(4))
            st.markdown(f"""
Species turnover along the primary platform-distance bins indicates a change in the
relative dominance of catch taxa between near and far exposure classes. The strongest
near-platform signals include **{near_names or 'no dominant near-platform taxa among the top contrasts'}**,
whereas far-platform bins are characterized by **{far_names or 'no dominant far-platform taxa among the top contrasts'}**.
This turnover result complements the PERMANOVA evidence by identifying the taxa that
contribute most to compositional separation.
""")
            st.markdown("**Table 6.** Largest species-level relative-abundance contrasts between Q3 and Q1.")
            st.dataframe(
                top_turn[["species_name", "Q1", "Q3", "Delta Q3-Q1", "Dominant group"]]
                .rename(columns={"species_name": "Species"})
                .round(4),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Primary-bin abundance data are available, but Q1/Q3 contrasts could not be computed.")
    else:
        st.info("Species-turnover outputs are not available.")

    if not pa_turnover.empty:
        st.markdown("### Protected-area turnover")
        pa_top = pa_turnover.sort_values("abs_difference", ascending=False).head(10).copy()
        st.markdown("""
Protected-area turnover summaries further indicate that MPA distance and MPA relation
are associated with changes in the relative contribution of individual taxa. These
taxon-level contrasts should be interpreted as compositional signatures of the
platform-MPA spatial system rather than as direct causal effects of protection.
""")
        st.dataframe(
            pa_top[[
                "exposure_variable", "species_name", "mean_rel_Q1", "mean_rel_Q3",
                "difference_far_minus_near", "dominant_in"
            ]]
            .rename(columns={
                "exposure_variable": "Exposure",
                "species_name": "Species",
                "mean_rel_Q1": "Mean rel. Q1",
                "mean_rel_Q3": "Mean rel. Q3",
                "difference_far_minus_near": "Delta Q3-Q1",
                "dominant_in": "Dominant in",
            })
            .assign(Exposure=lambda d: d["Exposure"].map(_elabel))
            .round(4),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.caption(
        "Interpretation note: these outputs quantify spatial associations in an observational "
        "fishery dataset. Interaction terms indicate modulation or joint spatial structure, "
        "but causal attribution requires additional design assumptions or external validation."
    )
