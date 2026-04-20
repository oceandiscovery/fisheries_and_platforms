"""
analysis_tabs.py — The 5 analysis tabs of the dashboard:
  tab_exposure     → Module 07: Platform exposure
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
}

RESPONSE_LABELS = {
    "pielou_species":           "Pielou's J' index",
    "shannon_species":          "Shannon H' index",
    "species_richness":         "Species richness (S)",
    "production_ton":           "Total production (t)",
    "production_per_trip_ton":  "Production per trip (t)",
}

LOCALITY_COLORS = {
    "AREIA BRANCA":    "#e74c3c",
    "CAICARA DO NORTE":"#2980b9",
    "GUAMARE":         "#27ae60",
    "MACAU":           "#8e44ad",
    "PORTO DO MANGUE": "#e67e22",
}

def _port_name(x):
    return PORT_COORDS.get(x, {}).get("name", x)

def _elabel(col):
    return EXPOSURE_LABELS.get(col, col)

def _rlabel(col):
    return RESPONSE_LABELS.get(col, col)


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
        color_discrete_map=LOCALITY_COLORS if color_by == "Locality" else None,
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
# TAB 2 — MODELOS GAM (módulo 08)
# ══════════════════════════════════════════════════════════════════════════════
def tab_gam(ad):
    st.markdown('<h3 class="section-title">GAM-spline models and comparison with linear models</h3>',
                unsafe_allow_html=True)

    best = ad["gam_best"].copy()
    comparison = ad["gam_comparison"].copy()
    smooth = ad["gam_smooth"].copy()
    fitted = ad["gam_fitted"].copy()
    coef = ad["gam_coef"].copy()

    # ── Selector de modelo ────────────────────────────────────────────────
    model_names = best["model_name"].tolist()
    sel_model = st.selectbox(
        "Select model",
        options=model_names,
        format_func=lambda x: f"{_rlabel(best[best['model_name']==x]['response_variable'].values[0])} "
                               f"↔ {_elabel(best[best['model_name']==x]['exposure_variable'].values[0])} "
                               f"[GAM spline]",
    )

    row = best[best["model_name"] == sel_model].iloc[0]

    # KPIs del modelo seleccionado
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R²", f"{row['r_squared']:.4f}")
    c2.metric("Adj. R²", f"{row['adj_r_squared']:.4f}")
    c3.metric("AIC", f"{row['aic']:.2f}")
    c4.metric("RMSE", f"{row['rmse']:.4f}")
    c5.metric("N parameters", str(int(row['n_parameters'])))

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # ── Curva GAM con IC ──────────────────────────────────────────────────
    with col_l:
        st.markdown("#### GAM curve with confidence interval (95%)")
        sm = smooth[smooth["model_name"] == sel_model].copy()
        exp_col = row["exposure_variable"]

        # Eje X es el exposure_variable del modelo
        x_col_smooth = exp_col if exp_col in sm.columns else sm.columns[0]

        sm_sorted = sm.dropna(subset=[x_col_smooth]).sort_values(x_col_smooth)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sm_sorted[x_col_smooth], y=sm_sorted["predicted_ci_high"],
            mode="lines", line=dict(width=0), showlegend=False,
            name="IC 95% sup.",
        ))
        fig.add_trace(go.Scatter(
            x=sm_sorted[x_col_smooth], y=sm_sorted["predicted_ci_low"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(41,128,185,0.18)", showlegend=False,
            name="IC 95%",
        ))
        fig.add_trace(go.Scatter(
            x=sm_sorted[x_col_smooth], y=sm_sorted["predicted"],
            mode="lines", line=dict(color="#2980b9", width=2.5),
            name="GAM predicted",
        ))

        # Observaciones
        fit_sel = fitted[fitted["model_name"] == sel_model].copy()
        fit_sel["port_name"] = fit_sel["local_canonical"].map(_port_name)
        fig.add_trace(go.Scatter(
            x=fit_sel["observed_exposure"], y=fit_sel["observed_response"],
            mode="markers",
            marker=dict(color=[LOCALITY_COLORS.get(l, "#888") for l in fit_sel["local_canonical"]],
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

    # ── Observado vs Ajustado ─────────────────────────────────────────────
    with col_r:
        st.markdown("#### Observed vs Fitted + residuals")
        fit_sel = fitted[fitted["model_name"] == sel_model].copy()
        fit_sel["port_name"] = fit_sel["local_canonical"].map(_port_name)

        fig2 = px.scatter(
            fit_sel, x="fitted", y="observed_response",
            color="port_name",
            color_discrete_map=LOCALITY_COLORS,
            height=250,
            labels={"fitted": "Fitted", "observed_response": "Observed",
                    "port_name": "Locality"},
            hover_data=["year"],
        )
        # Línea 1:1
        vmin = min(fit_sel["fitted"].min(), fit_sel["observed_response"].min())
        vmax = max(fit_sel["fitted"].max(), fit_sel["observed_response"].max())
        fig2.add_shape(type="line", x0=vmin, y0=vmin, x1=vmax, y1=vmax,
                       line=dict(dash="dash", color="#aaa", width=1))
        fig2.update_layout(margin=dict(t=20), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # Residuos
        fig3 = px.scatter(
            fit_sel, x="fitted", y="residual",
            color="port_name",
            color_discrete_map=LOCALITY_COLORS,
            height=130,
            labels={"fitted": "Fitted", "residual": "Residual"},
        )
        fig3.add_hline(y=0, line_dash="dash", line_color="#aaa", line_width=1)
        fig3.update_layout(margin=dict(t=5, b=5), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Comparación GAM vs lineal ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Model comparison: GAM spline vs linear (AIC, R²)")
    resp_options = comparison["response_variable"].unique().tolist()
    sel_resp = st.selectbox("Response variable",
                            options=resp_options, format_func=_rlabel, key="comp_resp")

    comp_sub = comparison[comparison["response_variable"] == sel_resp].copy()
    comp_sub["model_label"] = comp_sub["model_name"].str.replace("gam_", "GAM: ").str.replace("lin_", "LIN: ")
    comp_sub["model_type_label"] = comp_sub["model_type"].map(
        {"spline_gam_like": "GAM spline", "linear": "Linear"})

    col_a, col_b = st.columns(2)
    with col_a:
        fig_r2 = px.bar(
            comp_sub.sort_values("adj_r_squared", ascending=False),
            x="adj_r_squared", y="model_label", orientation="h",
            color="model_type_label", height=350,
            color_discrete_map={"GAM spline": "#2980b9", "Linear": "#e74c3c"},
            labels={"adj_r_squared": "Adj. R²", "model_label": "",
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

    # ── Coeficientes del modelo ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Model coefficients with 95% confidence intervals")
    coef_sel = coef[coef["model_name"] == sel_model].copy()
    coef_sel = coef_sel[~coef_sel["term"].str.startswith("bs(")].copy()  # excluir splines
    coef_sel["sig"] = coef_sel["significant_alpha_0_05"].map({True: "✅ p<0.05", False: "—"})
    coef_sel["color"] = coef_sel["significant_alpha_0_05"].map(
        {True: "#27ae60", False: "#95a5a6"})

    if not coef_sel.empty:
        fig_coef = go.Figure()
        for _, r in coef_sel.iterrows():
            fig_coef.add_trace(go.Scatter(
                x=[r["conf_low"], r["conf_high"]], y=[r["term"], r["term"]],
                mode="lines", line=dict(color=r["color"], width=3),
                showlegend=False,
            ))
            fig_coef.add_trace(go.Scatter(
                x=[r["estimate"]], y=[r["term"]],
                mode="markers",
                marker=dict(color=r["color"], size=10, symbol="circle"),
                name=r["sig"],
                showlegend=False,
                hovertemplate=f"Estimado: {r['estimate']:.4f}<br>"
                              f"IC: [{r['conf_low']:.4f}, {r['conf_high']:.4f}]<br>"
                              f"p={r['p_value']:.4f}<extra>{r['term']}</extra>",
            ))
        fig_coef.add_vline(x=0, line_dash="dash", line_color="#aaa", line_width=1)
        fig_coef.update_layout(height=max(200, len(coef_sel) * 50),
                               margin=dict(t=20), xaxis_title="Estimate (95% CI)")
        st.plotly_chart(fig_coef, use_container_width=True)

        disp_coef = coef_sel[["term", "estimate", "std_error", "t_value",
                               "p_value", "sig"]].copy()
        disp_coef.columns = ["Term", "Estimate", "Std. error", "t", "p-valor", "Sig."]
        disp_coef = disp_coef.round({"Estimate": 5, "Std. error": 5, "t": 3, "p-valor": 4})
        st.dataframe(disp_coef, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROBUSTEZ DEL MODELO (módulo 09)
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
            marker=dict(color=[LOCALITY_COLORS.get(l,"#aaa") for l in base_fit["local_canonical"]],
                        size=6, opacity=0.5),
            showlegend=False, name="Observed",
            text=base_fit["port_name"] + " " + base_fit["year"].astype(str),
            hovertemplate="%{text}<extra></extra>",
        ))

    for variant, grp in variants.groupby("variant"):
        grp_s = grp.dropna(subset=[exp_col]).sort_values(exp_col)
        color = variant_colors.get(variant, "#888")
        # IC
        fig.add_trace(go.Scatter(
            x=pd.concat([grp_s[exp_col], grp_s[exp_col].iloc[::-1]]),
            y=pd.concat([grp_s["predicted_ci_high"], grp_s["predicted_ci_low"].iloc[::-1]]),
            fill="toself", fillcolor=_hex_to_rgba(color, 0.08),
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=grp_s[exp_col], y=grp_s["predicted"],
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
            color = LOCALITY_COLORS.get(str(locality), "#aaa")
            fig2.add_trace(go.Scatter(
                x=grp_s[exp_lolo], y=grp_s["predicted"],
                mode="lines", name=_port_name(str(locality)),
                line=dict(color=color, width=1.5, dash="dot"),
            ))
        # Curva base
        if base_key:
            base_s = gam_smooth[gam_smooth["model_name"] == base_key[0]].copy()
            exp_b = base_key[0].split("_vs_")[-1] if "_vs_" in base_key[0] else exp_lolo
            x_b = exp_lolo if exp_lolo in base_s.columns else base_s.columns[0]
            base_s = base_s.dropna(subset=[x_b]).sort_values(x_b)
            fig2.add_trace(go.Scatter(
                x=base_s[x_b], y=base_s["predicted"],
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

    if not loyo_m.empty:
        exp_loyo = loyo_m.columns[0]
        years = sorted(loyo_m["group_removed"].dropna().unique())
        fig3 = go.Figure()
        colorscale = px.colors.sequential.Viridis
        for i, yr in enumerate(years):
            grp = loyo_m[loyo_m["group_removed"] == yr].dropna(subset=[exp_loyo]).sort_values(exp_loyo)
            color = colorscale[int(i / len(years) * (len(colorscale) - 1))]
            fig3.add_trace(go.Scatter(
                x=grp[exp_loyo], y=grp["predicted"],
                mode="lines", name=str(yr),
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.7,
            ))
        if base_key:
            base_s = gam_smooth[gam_smooth["model_name"] == base_key[0]].copy()
            x_b = exp_loyo if exp_loyo in base_s.columns else base_s.columns[0]
            base_s = base_s.dropna(subset=[x_b]).sort_values(x_b)
            fig3.add_trace(go.Scatter(
                x=base_s[x_b], y=base_s["predicted"],
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
                disp_inf.columns = ["Locality", "Year", "Cook's D", "Leverage", "Studentized residual", "Diagnostic"]
                st.dataframe(disp_inf.round(4), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ORDENACIÓN MULTIVARIANTE (módulo 10)
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
    ax1_label = f"Eje 1 ({ax1_pct:.1f}%)" if ax1_pct else "Eje 1"
    ax2_label = f"Eje 2 ({ax2_pct:.1f}%)" if ax2_pct else "Eje 2"

    col_top = st.columns([1, 1, 1])
    perm_row = permanova[permanova["distance_basis"].str.contains("BrayCurtis")].iloc[0] \
        if "BrayCurtis" in permanova["distance_basis"].values[0] else permanova.iloc[0]
    col_top[0].metric("PERMANOVA R²", f"{perm_row['r2']:.4f}")
    col_top[1].metric("PERMANOVA p", f"{perm_row['p_value']:.3f}",
                      "✅ sig." if perm_row["p_value"] <= 0.05 else "n.s.")
    col_top[2].metric("pseudo-F", f"{perm_row['pseudo_F']:.2f}")

    st.markdown("---")

    # ── Biplot interactivo ────────────────────────────────────────────────
    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        st.markdown("#### Ordination biplot")
        exp_cols_avail = [c for c in EXPOSURE_LABELS if c in scores.columns]
        color_opt = st.selectbox("Color by",
                                 ["Locality", "Year"] + [_elabel(c) for c in exp_cols_avail],
                                 key="ord_color")

        if color_opt == "Locality":
            fig = px.scatter(
                scores, x="Axis1", y="Axis2",
                color="port_name",
                color_discrete_map={_port_name(k): v for k, v in LOCALITY_COLORS.items()},
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
            # Color por variable de exposición
            exp_sel = exp_cols_avail[[_elabel(c) for c in exp_cols_avail].index(color_opt)]
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
                color = LOCALITY_COLORS.get(
                    next((k for k, v in {_port_name(k): k for k in LOCALITY_COLORS}.items()
                          if v == local), ""), "#aaa")
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

    # ── PERMANOVA completo ────────────────────────────────────────────────
    with st.expander("Full PERMANOVA table"):
        perm_disp = permanova.copy()
        perm_disp["exposure_variable"] = perm_disp["exposure_variable"].map(_elabel)
        perm_disp.columns = [c.replace("_", " ").title() for c in perm_disp.columns]
        st.dataframe(perm_disp.round(4), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GRADIENTE DE COMPOSICIÓN (módulo 11)
# ══════════════════════════════════════════════════════════════════════════════
def tab_gradient(ad):
    st.markdown('<h3 class="section-title">Primary gradient of catch composition</h3>',
                unsafe_allow_html=True)

    summary   = ad["grad_summary"].copy()
    scores    = ad["grad_scores"].copy()
    turnover  = ad["turnover_top"].copy()
    top_bin   = ad["top_by_bin"].copy()
    mean_ab   = ad["mean_abund_bin"].copy()

    # ── KPIs del gradiente ────────────────────────────────────────────────
    row = summary.iloc[0]
    # 'axis' en grad_summary, 'primary_axis' en grad_scores — usamos el de scores
    primary_axis_label = scores["primary_axis"].iloc[0] if "primary_axis" in scores.columns else row.get("axis", "Axis1")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ordination", f"{row['ordination'].replace('_',' ')}")
    c2.metric("PERMANOVA R²", f"{row['permanova_r2']:.4f}", "p=0.001")
    c3.metric("Axis ↔ distance correlation", f"ρ={row['axis_spearman_corr']:.4f}")
    c4.metric("Exposure variable", _elabel(row["exposure_variable"]))

    st.markdown("---")
    col_l, col_r = st.columns([1.2, 1])

    # ── Scores del gradiente vs exposición ───────────────────────────────
    with col_l:
        st.markdown("#### Primary axis score vs platform distance")
        scores["port_name"] = scores["local_canonical"].map(_port_name)
        exp_var = row["exposure_variable"]

        fig = px.scatter(
            scores.dropna(subset=["primary_axis_score", exp_var]),
            x=exp_var,
            y="primary_axis_score",
            color="port_name",
            color_discrete_map={_port_name(k): v for k, v in LOCALITY_COLORS.items()},
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
            color_discrete_map={_port_name(k): v for k, v in LOCALITY_COLORS.items()},
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

    turn_plot = turnover.sort_values("difference_Q3_minus_Q1").copy()
    turn_plot["color"] = turn_plot["difference_Q3_minus_Q1"].apply(
        lambda x: "#e74c3c" if x < 0 else "#27ae60")
    turn_plot["label"] = turn_plot["species_name"] + \
        turn_plot["axis_spearman_corr"].apply(lambda r: f"  (ρ={r:.2f})")

    fig3 = go.Figure(go.Bar(
        x=turn_plot["difference_Q3_minus_Q1"],
        y=turn_plot["label"],
        orientation="h",
        marker_color=turn_plot["color"],
        text=turn_plot["difference_Q3_minus_Q1"].apply(lambda x: f"{x:+.3f}"),
        textposition="outside",
        customdata=np.stack([
            turn_plot["mean_rel_Q1"],
            turn_plot["mean_rel_Q3"],
            turn_plot["dominant_in"],
            turn_plot["species_total_ton"],
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
            dict(x=turn_plot["difference_Q3_minus_Q1"].max() * 0.7,
                 y=len(turn_plot) - 0.5,
                 text="↑ More abundant far from platforms",
                 showarrow=False, font=dict(size=10, color="#27ae60")),
            dict(x=turn_plot["difference_Q3_minus_Q1"].min() * 0.7,
                 y=len(turn_plot) - 0.5,
                 text="↑ More abundant near platforms",
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
    for i, (bin_val, col) in enumerate(zip(["Q1", "Q2", "Q3"], col_bins)):
        grp = top_bin[top_bin["exposure_bin"] == bin_val].sort_values(
            "mean_relative_abundance", ascending=True)
        with col:
            st.markdown(f"**{bin_labels[bin_val]}**")
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
    mean_ab_pivot = mean_ab.pivot_table(
        index="species_name", columns="exposure_bin",
        values="mean_relative_abundance", aggfunc="mean"
    ).fillna(0)
    mean_ab_pivot = mean_ab_pivot.loc[
        mean_ab_pivot.max(axis=1).sort_values(ascending=False).index
    ]

    fig_heat = px.imshow(
        mean_ab_pivot.values,
        x=["Q1 (near)", "Q2 (mid)", "Q3 (far)"],
        y=mean_ab_pivot.index.tolist(),
        color_continuous_scale="YlOrRd",
        text_auto=".3f",
        labels={"color": "Rel. abundance"},
        height=max(500, len(mean_ab_pivot) * 22),
        aspect="auto",
    )
    fig_heat.update_layout(margin=dict(t=20))
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — METHODS & RESULTS (academic synthesis)
# ══════════════════════════════════════════════════════════════════════════════
def tab_methods_results():
    """Synthesized Methods and Results in academic journal style."""

    st.markdown('<h3 class="section-title">Methods & Results</h3>', unsafe_allow_html=True)

    st.markdown("""
> *Synthesis of the main analytical findings. Statistical details and interactive
> visualizations are available in the preceding analysis tabs.*
""")

    # ── METHODS ──────────────────────────────────────────────────────────────
    st.markdown("## Methods")

    st.markdown("### Study area and data")
    st.markdown("""
Fisheries landings were compiled for **five coastal localities** along the Rio Grande
do Norte coast, Brazil (Areia Branca, Caiçara do Norte, Guamaré, Macau, and Porto do
Mangue), yielding **112 locality-year observations** spanning 2001–2023. The dataset
records **59 species** across multiple fishing gears and vessel types. Offshore oil
platform coordinates were used to compute per-locality annual exposure metrics: mean,
closest, and farthest distances to the nearest platform, and the number of platforms
within 10 and 20 km radii.
""")

    st.markdown("### Exposure–response screening")
    st.markdown("""
All pairwise associations between platform exposure variables and fisheries response
variables (species richness, Shannon H', Pielou J', total production, and CPUE in
t/trip) were quantified with Spearman rank correlation across the full 112-observation
panel (Module 07). Pairs with |ρ| > 0.30 were retained as candidates for formal
modelling.
""")

    st.markdown("### GAM modelling")
    st.markdown("""
Generalized Additive Models (GAMs) were fitted for each candidate exposure–response
pair using a natural spline basis (df = 4, degree = 3) for the platform exposure term,
plus gear-type richness, boat-type richness, and year (centred) as linear covariates
(Module 08). Model fit was evaluated by R², adjusted R², AIC, and RMSE. Spline
degrees of freedom were varied (df = 3, 5) and a quadratic alternative was tested to
assess curve stability (Module 09).

Sensitivity was further evaluated via leave-one-locality-out (LOLO) and
leave-one-year-out (LOYO) cross-validation. Influential observations were identified
through Cook's D and studentized residuals.
""")

    st.markdown("### Community ordination")
    st.markdown("""
Species composition was analysed using Principal Coordinates Analysis (PCoA) with
Bray-Curtis and Hellinger dissimilarities, and Non-metric Multidimensional Scaling
(NMDS). Differences in community composition across platform-exposure tertiles were
tested with PERMANOVA (999 permutations). Beta-dispersion was assessed with a
multivariate homogeneity test. Associations between ordination axes and exposure
variables were quantified by Spearman correlation (Module 10).

Species turnover along the primary gradient was characterized by comparing mean
relative abundances between the nearest (Q1) and farthest (Q3) platform-exposure
tertiles (Module 11).
""")

    st.markdown("---")

    # ── RESULTS ──────────────────────────────────────────────────────────────
    st.markdown("## Results")

    st.markdown("### Fisheries overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", "112", "locality-years")
    col2.metric("Species recorded", "59")
    col3.metric("Mean richness", "41.7 ± 11.5 spp")
    col4.metric("Study period", "2001–2023")

    st.markdown("""
Mean Shannon diversity across the dataset was H' = 2.40 ± 0.67, with Pielou evenness
J' = 0.67 ± 0.17. Substantial inter-locality differences were evident: Areia Branca
had the highest mean CPUE (0.539 t/trip) and production (1,809 t/yr) but the lowest
diversity (H' = 1.35; J' = 0.40), whereas Porto do Mangue showed the highest diversity
(H' = 2.78; J' = 0.78) with a mean platform distance of 24.8 km (Table 1).
""")

    # Table 1
    st.markdown("**Table 1.** Summary statistics by fishing locality (2001–2023 means).")
    tbl1 = pd.DataFrame({
        "Locality":           ["Areia Branca", "Caiçara do Norte", "Guamaré", "Macau", "Porto do Mangue"],
        "Shannon H'":         [1.35, 2.62, 2.66, 2.44, 2.78],
        "Pielou J'":          [0.40, 0.72, 0.74, 0.66, 0.78],
        "Production (t/yr)":  [1809, 915, 206, 1065, 334],
        "CPUE (t/trip)":      [0.539, 0.127, 0.061, 0.092, 0.053],
        "Mean dist. (km)":    [31.2, 31.7, 15.5, 8.70, 24.8],
    })
    st.dataframe(tbl1, use_container_width=True, hide_index=True)

    st.markdown("### Platform exposure associations")
    st.markdown("""
The strongest global association identified in the screening was a **negative
correlation between CPUE and the number of platforms within 20 km**
(ρ = −0.611, closest-platform count; ρ = −0.521, mean count; n = 112). Total
production showed a similarly strong negative relationship with platform density
(ρ = −0.500). In contrast, **diversity indices were positively associated with
platform proximity** (Pielou J' ~ closest-platform count: ρ = +0.348; Shannon H':
ρ = +0.326), indicating that localities closer to platforms maintained higher
evenness despite lower yields.
""")

    # Table 2
    st.markdown("**Table 2.** Top-ranked Spearman associations (n = 112).")
    tbl2 = pd.DataFrame({
        "Rank": [1, 2, 3, 4, 6, 9],
        "Response":  ["CPUE (t/trip)", "CPUE (t/trip)", "Total production (t)",
                      "CPUE (t/trip)", "Pielou J'", "Shannon H'"],
        "Exposure":  ["No. platforms ≤20 km (closest)", "No. platforms ≤20 km (mean)",
                      "No. platforms ≤20 km (closest)", "Farthest platform distance",
                      "No. platforms ≤20 km (closest)", "No. platforms ≤20 km (closest)"],
        "Spearman ρ": [-0.611, -0.521, -0.500, +0.457, +0.348, +0.326],
    })
    st.dataframe(tbl2, use_container_width=True, hide_index=True)

    st.markdown("### GAM model performance")
    st.markdown("""
GAM-spline models (df = 4) consistently outperformed their linear equivalents by
ΔR² ≈ 0.40–0.50, confirming non-linear exposure–response relationships. The
best-performing models used the closest platform distance as the exposure variable
(Table 3). Shannon H' and Pielou J' models explained ~61% and ~60% of variance,
respectively; the CPUE model explained ~60%; and the production model ~57%. All four
models incorporated gear-type richness, boat-type richness, and year as linear
covariates.
""")

    # Table 3
    st.markdown("**Table 3.** Best-fit GAM models (spline, df = 4).")
    tbl3 = pd.DataFrame({
        "Response":         ["Shannon H'", "Pielou J'", "CPUE (t/trip)", "Production (t)"],
        "Exposure":         ["Closest platform dist. (km)", "Closest platform dist. (km)",
                             "Closest platform dist. (km)", "Mean platform dist. (km)"],
        "R²":               [0.614, 0.605, 0.603, 0.571],
        "Adj. R²":          [0.588, 0.578, 0.576, 0.542],
        "AIC":              [135.50, -164.97, -89.53, 1766.03],
    })
    st.dataframe(tbl3, use_container_width=True, hide_index=True)

    st.markdown("### Non-monotonic exposure–response curves")
    st.markdown("""
All GAM curves were non-monotonic, exhibiting hump-shaped or U-shaped responses to
platform distance. **Diversity indices (H' and J') peaked at intermediate distances**
(~17–18 km), whereas **CPUE showed an inverse pattern** — lowest at intermediate
distances and highest at greater distances — consistent with a trade-off between
ecological and extractive productivity. Production peaked at ~29 km from the nearest
platform. The predicted range of the Pielou J' curve was 0.56 J' units; for CPUE the
range was 0.72 t/trip.
""")

    st.markdown("### Model robustness")
    st.markdown("""
Spline variants (df = 3 and df = 5) and a quadratic alternative produced
qualitatively similar curves, confirming shape stability. Leave-one-year-out (LOYO)
validation returned R² values of 0.585–0.644 across all 23 years, indicating that no
single year drives the result. Leave-one-locality-out (LOLO) analysis revealed that
removing Areia Branca reduced the Pielou J' model R² from 0.605 to 0.339, reflecting
the high leverage of this locality given its atypically high CPUE and low diversity
at distances > 30 km from platforms. All remaining localities yielded stable R²
(0.59–0.66).
""")

    st.markdown("### Community structure")
    st.markdown("""
PERMANOVA confirmed significant compositional differences across platform-exposure
tertiles (Bray-Curtis: pseudo-F = 27.61, R² = 0.336, p = 0.001; Hellinger:
pseudo-F = 26.52, R² = 0.327, p = 0.001). PCoA Axis 1 (Bray-Curtis) captured 23.8%
of total variance; Axis 2 explained 16.4%. The primary ordination axis was strongly
correlated with the farthest platform distance (ρ = 0.897, Bray-Curtis), confirming
that platform proximity gradient explains a substantial fraction of inter-annual
community variation. Beta-dispersion was highest in the intermediate-distance tertile
(Q2; mean = 0.274) and lowest in the far tertile (Q3; 0.051), suggesting that
communities at intermediate distances are the most compositionally variable.

The three species most strongly loading on Axis 1 were Sardinha (ρ = −0.788),
Tainha (ρ = −0.763), and Lagosta (ρ = +0.627), reflecting a gradient from
coastal-dominated to offshore-dominated assemblages.
""")

    st.markdown("### Species turnover")
    st.markdown("""
The primary composition gradient (PCoA Bray-Curtis Axis 1 ~ mean platform distance;
ρ = 0.773) corresponded to a clear turnover in dominant species. Near-platform
localities (Q1) were dominated by coastal species — Sardinha (Δ = −0.154;
7,014 t historical total), Tainha (Δ = −0.150; 4,076 t), Caranguejo, and Buzios —
all showing significantly higher relative abundance close to platforms. Far-platform
localities (Q3) were characterised by oceanic or offshore species: Peixe Voador
(Δ = +0.226; 8,111 t), Dourado (Δ = +0.063; 2,900 t), and Lagosta
(Δ = +0.056; 3,090 t) (Table 4).
""")

    # Table 4
    st.markdown("**Table 4.** Key species in the composition gradient (Q3 − Q1 relative abundance difference).")
    tbl4 = pd.DataFrame({
        "Species":         ["Sardinha", "Tainha", "Caranguejo", "Buzios",
                            "Peixe Voador", "Dourado", "Lagosta"],
        "Δ (Q3 − Q1)":    [-0.154, -0.150, -0.052, -0.039, +0.226, +0.063, +0.056],
        "Dominant in":     ["Q1 (near)", "Q1 (near)", "Q1 (near)", "Q1 (near)",
                            "Q3 (far)", "Q3 (far)", "Q3 (far)"],
        "Total catch (t)": [7014, 4076, "—", "—", 8111, 2900, 3090],
    })
    st.dataframe(tbl4, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "All analyses were performed in Python (scikit-learn, scipy, statsmodels, skbio). "
        "PERMANOVA was run with 999 permutations. GAM models used OLS-based polynomial "
        "spline approximation. Data source: PMDP/IBAMA — Rio Grande do Norte, Brazil (2001–2023)."
    )
