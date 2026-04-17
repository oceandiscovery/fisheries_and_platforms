"""
analysis_tabs.py — Los 5 tabs de análisis del dashboard:
  tab_exposure     → Módulo 07: Exposición a plataformas
  tab_gam          → Módulo 08: Modelos GAM
  tab_robustness   → Módulo 09: Robustez GAM
  tab_ordination   → Módulo 10: Ordenación multivariante
  tab_gradient     → Módulo 11: Gradiente de composición
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.coords import PORT_COORDS

# ── Helpers ────────────────────────────────────────────────────────────────

EXPOSURE_LABELS = {
    "mean_nearest_platform_distance_km":     "Distancia media a plataforma (km)",
    "closest_nearest_platform_distance_km":  "Distancia mínima a plataforma (km)",
    "farthest_nearest_platform_distance_km": "Distancia máxima a plataforma (km)",
    "mean_n_platforms_within_20km":          "Nº plataformas en 20 km (media)",
    "mean_n_platforms_within_10km":          "Nº plataformas en 10 km (media)",
    "inv_distance_sum_mean":                 "Suma inversa de distancias (media)",
}

RESPONSE_LABELS = {
    "pielou_species":           "Índice de Pielou J'",
    "shannon_species":          "Índice de Shannon H'",
    "species_richness":         "Riqueza de especies (S)",
    "production_ton":           "Producción total (t)",
    "production_per_trip_ton":  "Producción por viaje (t)",
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
    st.markdown('<h3 class="section-title">Asociaciones entre exposición a plataformas y pesquerías</h3>',
                unsafe_allow_html=True)

    # ── Panel de métricas clave ──────────────────────────────────────────
    screening = ad["assoc_screening"].copy()
    top1 = screening.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Asociación más fuerte", f"ρ = {top1['spearman_corr']:.3f}",
              f"{_rlabel(top1['response_variable'])}")
    c2.metric("Variable respuesta", _rlabel(top1['response_variable']))
    c3.metric("Variable exposición", _elabel(top1['exposure_variable']))
    c4.metric("Candidatos formales",
              str(len(screening[screening["screening_note"] == "candidate_for_formal_followup"])))

    st.markdown("---")

    # ── Screening: ranking de correlaciones ──────────────────────────────
    col_l, col_r = st.columns([1.1, 1])

    with col_l:
        st.markdown("#### Ranking de asociaciones globales (Spearman |ρ|)")
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
            xaxis_title="Correlación de Spearman (ρ)",
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Tabla de screening priorizado")
        disp = screening[["priority_rank", "response_variable", "exposure_variable",
                           "spearman_corr", "pearson_corr", "n_complete", "screening_note"]].copy()
        disp["response_variable"] = disp["response_variable"].map(_rlabel)
        disp["exposure_variable"] = disp["exposure_variable"].map(_elabel)
        disp.columns = ["Rango", "Respuesta", "Exposición", "ρ Spearman", "r Pearson",
                        "N", "Nota"]
        disp["ρ Spearman"] = disp["ρ Spearman"].round(3)
        disp["r Pearson"]  = disp["r Pearson"].round(3)
        st.dataframe(
            disp.style.background_gradient(subset=["ρ Spearman"], cmap="RdYlGn"),
            use_container_width=True, hide_index=True, height=420,
        )

    st.markdown("---")

    # ── Scatter interactivo ───────────────────────────────────────────────
    st.markdown("#### Explorar relación exposición ↔ diversidad/producción")
    div = ad["div_table"].copy()
    div["port_name"] = div["local_canonical"].map(_port_name)

    col_sel1, col_sel2, col_sel3 = st.columns(3)
    x_col = col_sel1.selectbox("Variable X (exposición)",
        options=[c for c in EXPOSURE_LABELS if c in div.columns],
        format_func=_elabel)
    y_col = col_sel2.selectbox("Variable Y (respuesta)",
        options=[c for c in RESPONSE_LABELS if c in div.columns],
        format_func=_rlabel)
    color_by = col_sel3.selectbox("Color", ["Localidad", "Año"])

    color_col = "port_name" if color_by == "Localidad" else "year"
    fig2 = px.scatter(
        div.dropna(subset=[x_col, y_col]),
        x=x_col, y=y_col,
        color=color_col,
        trendline="ols",
        trendline_scope="overall",
        trendline_color_override="#2c3e50",
        height=420,
        labels={x_col: _elabel(x_col), y_col: _rlabel(y_col),
                "port_name": "Localidad", "year": "Año"},
        color_discrete_map=LOCALITY_COLORS if color_by == "Localidad" else None,
        hover_data=["port_name", "year"],
    )
    fig2.update_layout(margin=dict(t=20), legend_title_text=color_by)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Asociaciones within-locality ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Correlaciones intra-localidad (Spearman)")
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
            xaxis_title="Localidad",
            yaxis_title="",
            coloraxis_colorbar_title="ρ",
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Sin suficientes datos temporales para correlaciones intra-localidad.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODELOS GAM (módulo 08)
# ══════════════════════════════════════════════════════════════════════════════
def tab_gam(ad):
    st.markdown('<h3 class="section-title">Modelos GAM-spline y comparación con modelos lineales</h3>',
                unsafe_allow_html=True)

    best = ad["gam_best"].copy()
    comparison = ad["gam_comparison"].copy()
    smooth = ad["gam_smooth"].copy()
    fitted = ad["gam_fitted"].copy()
    coef = ad["gam_coef"].copy()

    # ── Selector de modelo ────────────────────────────────────────────────
    model_names = best["model_name"].tolist()
    sel_model = st.selectbox(
        "Seleccionar modelo",
        options=model_names,
        format_func=lambda x: f"{_rlabel(best[best['model_name']==x]['response_variable'].values[0])} "
                               f"↔ {_elabel(best[best['model_name']==x]['exposure_variable'].values[0])} "
                               f"[GAM spline]",
    )

    row = best[best["model_name"] == sel_model].iloc[0]

    # KPIs del modelo seleccionado
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R²", f"{row['r_squared']:.4f}")
    c2.metric("R² ajustado", f"{row['adj_r_squared']:.4f}")
    c3.metric("AIC", f"{row['aic']:.2f}")
    c4.metric("RMSE", f"{row['rmse']:.4f}")
    c5.metric("N parámetros", str(int(row['n_parameters'])))

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # ── Curva GAM con IC ──────────────────────────────────────────────────
    with col_l:
        st.markdown("#### Curva GAM con intervalo de confianza (95%)")
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
            name="GAM predicho",
        ))

        # Observaciones
        fit_sel = fitted[fitted["model_name"] == sel_model].copy()
        fit_sel["port_name"] = fit_sel["local_canonical"].map(_port_name)
        fig.add_trace(go.Scatter(
            x=fit_sel["observed_exposure"], y=fit_sel["observed_response"],
            mode="markers",
            marker=dict(color=[LOCALITY_COLORS.get(l, "#888") for l in fit_sel["local_canonical"]],
                        size=7, line=dict(width=1, color="white")),
            name="Observado",
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
        st.markdown("#### Observado vs Ajustado + residuos")
        fit_sel = fitted[fitted["model_name"] == sel_model].copy()
        fit_sel["port_name"] = fit_sel["local_canonical"].map(_port_name)

        fig2 = px.scatter(
            fit_sel, x="fitted", y="observed_response",
            color="port_name",
            color_discrete_map=LOCALITY_COLORS,
            height=250,
            labels={"fitted": "Ajustado", "observed_response": "Observado",
                    "port_name": "Localidad"},
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
            labels={"fitted": "Ajustado", "residual": "Residuo"},
        )
        fig3.add_hline(y=0, line_dash="dash", line_color="#aaa", line_width=1)
        fig3.update_layout(margin=dict(t=5, b=5), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Comparación GAM vs lineal ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Comparación de modelos: GAM spline vs lineal (AIC, R²)")
    resp_options = comparison["response_variable"].unique().tolist()
    sel_resp = st.selectbox("Respuesta para comparar",
                            options=resp_options, format_func=_rlabel, key="comp_resp")

    comp_sub = comparison[comparison["response_variable"] == sel_resp].copy()
    comp_sub["model_label"] = comp_sub["model_name"].str.replace("gam_", "GAM: ").str.replace("lin_", "LIN: ")
    comp_sub["model_type_label"] = comp_sub["model_type"].map(
        {"spline_gam_like": "GAM spline", "linear": "Lineal"})

    col_a, col_b = st.columns(2)
    with col_a:
        fig_r2 = px.bar(
            comp_sub.sort_values("adj_r_squared", ascending=False),
            x="adj_r_squared", y="model_label", orientation="h",
            color="model_type_label", height=350,
            color_discrete_map={"GAM spline": "#2980b9", "Lineal": "#e74c3c"},
            labels={"adj_r_squared": "R² ajustado", "model_label": "",
                    "model_type_label": "Tipo"},
        )
        fig_r2.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_b:
        fig_aic = px.bar(
            comp_sub.sort_values("aic"),
            x="aic", y="model_label", orientation="h",
            color="model_type_label", height=350,
            color_discrete_map={"GAM spline": "#2980b9", "Lineal": "#e74c3c"},
            labels={"aic": "AIC (menor = mejor)", "model_label": "",
                    "model_type_label": "Tipo"},
        )
        fig_aic.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_aic, use_container_width=True)

    # ── Coeficientes del modelo ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Coeficientes del modelo con intervalos de confianza 95%")
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
                               margin=dict(t=20), xaxis_title="Estimado (IC 95%)")
        st.plotly_chart(fig_coef, use_container_width=True)

        disp_coef = coef_sel[["term", "estimate", "std_error", "t_value",
                               "p_value", "sig"]].copy()
        disp_coef.columns = ["Término", "Estimado", "Error std.", "t", "p-valor", "Sig."]
        disp_coef = disp_coef.round({"Estimado": 5, "Error std.": 5, "t": 3, "p-valor": 4})
        st.dataframe(disp_coef, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROBUSTEZ DEL MODELO (módulo 09)
# ══════════════════════════════════════════════════════════════════════════════
def tab_robustness(ad):
    st.markdown('<h3 class="section-title">Robustez del modelo GAM — análisis de sensibilidad</h3>',
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
    sel_base = st.selectbox("Modelo base", all_base)

    st.markdown("---")

    # ── 1. Variantes de spline ────────────────────────────────────────────
    st.markdown("#### Estabilidad de la curva: variantes GAM")
    variants = rob_curves[rob_curves["model_name"].str.startswith(sel_base)].copy()

    # Añadir curva base del módulo 08
    base_name_exact = sel_base.replace("_alt_df3","").replace("_alt_df5","").replace("_quadratic","")
    base_key = [m for m in gam_smooth["model_name"].unique() if m.startswith(sel_base[:35])]
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
            showlegend=False, name="Observado",
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
            fill="toself", fillcolor=color.replace("#","rgba(") + ",0.08)",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=grp_s[exp_col], y=grp_s["predicted"],
            mode="lines", name=variant,
            line=dict(color=color, width=2 if variant == "base (df=4)" else 1.5,
                      dash="solid" if variant == "base (df=4)" else "dot"),
        ))

    row_resp = signature[signature["model_name"].str.startswith(sel_base[:35])].head(1)
    y_label = _rlabel(row_resp["response_variable"].values[0]) if not row_resp.empty else "Respuesta"
    x_label = _elabel(row_resp["exposure_variable"].values[0]) if not row_resp.empty else "Exposición"

    fig.update_layout(height=420, margin=dict(t=20),
                      xaxis_title=x_label, yaxis_title=y_label,
                      legend=dict(orientation="h", y=-0.22))
    st.plotly_chart(fig, use_container_width=True)

    # ── 2. Leave-one-locality-out ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Leave-one-locality-out (LOLO) — estabilidad por localidad")
    lolo_m = lolo[lolo["model_name"].str.startswith(sel_base[:35])].copy()

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
                mode="lines", name="Completo",
                line=dict(color="#2c3e50", width=2.5),
            ))
        fig2.update_layout(height=380, margin=dict(t=20),
                           xaxis_title=x_label, yaxis_title=y_label,
                           legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig2, use_container_width=True)

    # ── 3. Leave-one-year-out ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Leave-one-year-out (LOYO) — estabilidad temporal")
    loyo_m = loyo[loyo["model_name"].str.startswith(sel_base[:35])].copy()

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
                mode="lines", name="Completo",
                line=dict(color="#2c3e50", width=2.5),
            ))
        fig3.update_layout(height=380, margin=dict(t=20),
                           xaxis_title=x_label, yaxis_title=y_label,
                           legend=dict(orientation="h", y=-0.25, font_size=9))
        st.plotly_chart(fig3, use_container_width=True)

    # ── 4. Puntos de influencia ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Puntos de influencia — Cook's D y leverage")
    inf_m = influence[influence["model_name"].str.startswith(sel_base[:35])].copy()

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
                    "port_name": "Localidad", "flag": "Tipo"},
            hover_data=["year", "port_name", "student_resid"],
            size_max=12,
        )
        fig4.update_layout(margin=dict(t=20), legend_title_text="Diagnóstico")
        st.plotly_chart(fig4, use_container_width=True)

        n_flag = len(inf_m[inf_m["flag"] != "Normal"])
        if n_flag:
            st.warning(f"{n_flag} observaciones marcadas como influyentes.")
            with st.expander("Ver detalle de observaciones influyentes"):
                disp_inf = inf_m[inf_m["flag"] != "Normal"][
                    ["local_canonical", "year", "cooks_d", "hat_diag",
                     "student_resid", "flag"]].copy()
                disp_inf["local_canonical"] = disp_inf["local_canonical"].map(_port_name)
                disp_inf.columns = ["Localidad", "Año", "Cook's D", "Leverage", "Residuo estudentizado", "Diagnóstico"]
                st.dataframe(disp_inf.round(4), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ORDENACIÓN MULTIVARIANTE (módulo 10)
# ══════════════════════════════════════════════════════════════════════════════
def tab_ordination(ad):
    st.markdown('<h3 class="section-title">Ordenación multivariante de la comunidad pesquera</h3>',
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
    sel_ord = st.radio("Método de ordenación", list(ord_options.keys()), horizontal=True)
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
        st.markdown("#### Biplot de ordenación")
        exp_cols_avail = [c for c in EXPOSURE_LABELS if c in scores.columns]
        color_opt = st.selectbox("Color por",
                                 ["Localidad", "Año"] + [_elabel(c) for c in exp_cols_avail],
                                 key="ord_color")

        if color_opt == "Localidad":
            fig = px.scatter(
                scores, x="Axis1", y="Axis2",
                color="port_name",
                color_discrete_map={_port_name(k): v for k, v in LOCALITY_COLORS.items()},
                hover_data=["year"],
                height=450,
                labels={"Axis1": ax1_label, "Axis2": ax2_label, "port_name": "Localidad"},
            )
        elif color_opt == "Año":
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
        st.markdown("#### Correlaciones ejes ↔ exposición (|ρ|)")
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
        st.markdown("#### Dispersión β por tertil de exposición")
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
                                   barmode="group", yaxis_title="Dispersión media",
                                   xaxis_title="Tertil de exposición",
                                   legend=dict(font_size=9))
            st.plotly_chart(fig_disp, use_container_width=True)

    # ── Especies asociadas a los ejes ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Especies con mayor asociación a los ejes de ordenación (|ρ|)")
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
                title=f"Asociaciones de especies con {axis}",
                xaxis_title="ρ Spearman", yaxis_title="",
                font_size=11,
            )
            st.plotly_chart(fig_sp, use_container_width=True)

    # ── PERMANOVA completo ────────────────────────────────────────────────
    with st.expander("Tabla PERMANOVA completa"):
        perm_disp = permanova.copy()
        perm_disp["exposure_variable"] = perm_disp["exposure_variable"].map(_elabel)
        perm_disp.columns = [c.replace("_", " ").title() for c in perm_disp.columns]
        st.dataframe(perm_disp.round(4), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GRADIENTE DE COMPOSICIÓN (módulo 11)
# ══════════════════════════════════════════════════════════════════════════════
def tab_gradient(ad):
    st.markdown('<h3 class="section-title">Gradiente primario de composición de capturas</h3>',
                unsafe_allow_html=True)

    summary   = ad["grad_summary"].copy()
    scores    = ad["grad_scores"].copy()
    turnover  = ad["turnover_top"].copy()
    top_bin   = ad["top_by_bin"].copy()
    mean_ab   = ad["mean_abund_bin"].copy()

    # ── KPIs del gradiente ────────────────────────────────────────────────
    row = summary.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ordenación", f"{row['ordination'].replace('_',' ')}")
    c2.metric("PERMANOVA R²", f"{row['permanova_r2']:.4f}", "p=0.001")
    c3.metric("Correlación eje ↔ distancia", f"ρ={row['axis_spearman_corr']:.4f}")
    c4.metric("Variable exposición", _elabel(row["exposure_variable"]))

    st.markdown("---")
    col_l, col_r = st.columns([1.2, 1])

    # ── Scores del gradiente vs exposición ───────────────────────────────
    with col_l:
        st.markdown("#### Score del eje primario vs distancia a plataformas")
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
                "primary_axis_score": f"Score {row['primary_axis']}",
                "port_name": "Localidad",
            },
            hover_data=["year"],
        )
        fig.update_layout(margin=dict(t=20), legend_title_text="Localidad")
        st.plotly_chart(fig, use_container_width=True)

    # ── Evolución temporal del score ──────────────────────────────────────
    with col_r:
        st.markdown("#### Evolución temporal del gradiente por localidad")
        fig2 = px.line(
            scores, x="year", y="primary_axis_score",
            color="port_name",
            markers=True,
            color_discrete_map={_port_name(k): v for k, v in LOCALITY_COLORS.items()},
            height=400,
            labels={"primary_axis_score": f"Score {row['primary_axis']}",
                    "year": "Año", "port_name": "Localidad"},
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="#aaa", line_width=1)
        fig2.update_layout(margin=dict(t=20), legend_title_text="Localidad")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Turnover de especies ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Turnover de especies a lo largo del gradiente")
    st.caption("Cambio en abundancia relativa entre el tertil más cercano (Q1) "
               "y más lejano (Q3) de las plataformas petroleras.")

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
            "Cambio Q3−Q1: %{x:.4f}<br>"
            "Abundancia rel. Q1: %{customdata[0]:.4f}<br>"
            "Abundancia rel. Q3: %{customdata[1]:.4f}<br>"
            "Dominante en: %{customdata[2]}<br>"
            "Total histórico: %{customdata[3]:.0f} t<extra></extra>"
        ),
    ))
    fig3.add_vline(x=0, line_dash="dash", line_color="#aaa", line_width=1)
    fig3.update_layout(
        height=max(420, len(turn_plot) * 28),
        margin=dict(t=20, r=80),
        xaxis_title="Diferencia abundancia relativa (Q3 − Q1)",
        yaxis_title="",
        annotations=[
            dict(x=turn_plot["difference_Q3_minus_Q1"].max() * 0.7,
                 y=len(turn_plot) - 0.5,
                 text="↑ Más abundante lejos de plataformas",
                 showarrow=False, font=dict(size=10, color="#27ae60")),
            dict(x=turn_plot["difference_Q3_minus_Q1"].min() * 0.7,
                 y=len(turn_plot) - 0.5,
                 text="↑ Más abundante cerca de plataformas",
                 showarrow=False, font=dict(size=10, color="#e74c3c")),
        ],
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Abundancias por tertil ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Composición de capturas por tertil de exposición")
    top_bin["species_label"] = top_bin["species_name"]

    col_bins = st.columns(3)
    bin_colors = {"Q1": "#e74c3c", "Q2": "#f39c12", "Q3": "#27ae60"}
    bin_labels = {
        "Q1": "Q1 — Más cercano a plataformas",
        "Q2": "Q2 — Distancia intermedia",
        "Q3": "Q3 — Más lejano de plataformas",
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
                xaxis_title="Abundancia relativa",
                yaxis_title="",
                font_size=10,
            )
            st.plotly_chart(fig_b, use_container_width=True)

    # ── Heatmap abundancia completo por especie × tertil ─────────────────
    st.markdown("---")
    st.markdown("#### Heatmap abundancia relativa — todas las especies × tertil")
    mean_ab_pivot = mean_ab.pivot_table(
        index="species_name", columns="exposure_bin",
        values="mean_relative_abundance", aggfunc="mean"
    ).fillna(0)
    mean_ab_pivot = mean_ab_pivot.loc[
        mean_ab_pivot.max(axis=1).sort_values(ascending=False).index
    ]

    fig_heat = px.imshow(
        mean_ab_pivot.values,
        x=["Q1 (cerca)", "Q2 (medio)", "Q3 (lejos)"],
        y=mean_ab_pivot.index.tolist(),
        color_continuous_scale="YlOrRd",
        text_auto=".3f",
        labels={"color": "Ab. relativa"},
        height=max(500, len(mean_ab_pivot) * 22),
        aspect="auto",
    )
    fig_heat.update_layout(margin=dict(t=20))
    st.plotly_chart(fig_heat, use_container_width=True)
