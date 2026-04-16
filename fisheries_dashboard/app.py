"""
╔══════════════════════════════════════════════════════════════════╗
║  FISHERIES GIS DASHBOARD — Rio Grande do Norte (Brasil)          ║
║  Capturas, CPUE, Biodiversidad y Análisis Espacial               ║
║  Desarrollado para integración con QGIS / GeoPandas              ║
╚══════════════════════════════════════════════════════════════════╝

Ejecución:
    streamlit run app.py
"""

import os
import sys
import warnings
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import folium
from streamlit_folium import st_folium
from scipy import stats as scipy_stats

# Asegurar que el directorio de trabajo es correcto
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_pipeline import build_all
from utils.map_builder import species_distribution_map, cpue_map, biodiversity_hotspot_map
from utils.coords import PORT_COORDS, GEAR_LABELS

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fisheries GIS Dashboard · RN Brasil",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #0e6655 100%);
    padding: 1.2rem 1.5rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 1rem;
  }
  .metric-card {
    background: #f8f9fa;
    border-left: 4px solid #2980b9;
    padding: 0.8rem 1rem;
    border-radius: 6px;
    margin-bottom: 0.5rem;
  }
  .section-title {
    color: #1a3a5c;
    border-bottom: 2px solid #2980b9;
    padding-bottom: 4px;
    margin-bottom: 1rem;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    background: #eaf4fb;
    border-radius: 6px 6px 0 0;
    font-weight: 500;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA DE DATOS (CACHEADO)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando y procesando datos...")
def get_data():
    return build_all(export=True, output_dir="outputs/geojson")


@st.cache_resource(show_spinner=False)
def get_maps(year_min, year_max):
    artefacts = get_data()
    yr = (year_min, year_max)
    m_species  = species_distribution_map(artefacts["dfs"], year_range=yr)
    m_cpue     = cpue_map(artefacts["cpue_gear"], artefacts["master"], year_range=yr)
    m_bio      = biodiversity_hotspot_map(artefacts["biodiv"], artefacts["master"], year_range=yr)
    return m_species, m_cpue, m_bio


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def sidebar(artefacts):
    with st.sidebar:
        st.markdown("## 🎛️ Filtros globales")
        master = artefacts["master"]
        years = sorted(master["year"].dropna().unique().astype(int))
        yr_range = st.slider(
            "Rango de años",
            min_value=int(years[0]), max_value=int(years[-1]),
            value=(int(years[0]), int(years[-1])),
            step=1,
        )

        st.markdown("---")
        st.markdown("**Puertos**")
        ports = sorted(master["local_norm"].unique())
        selected_ports = st.multiselect(
            "Seleccionar puertos",
            options=ports,
            default=ports,
            format_func=lambda x: PORT_COORDS.get(x, {}).get("name", x),
        )

        st.markdown("---")
        species_list = sorted(artefacts["dfs"]["species"]["species"].unique())
        selected_species = st.multiselect(
            "Especies de interés",
            options=species_list,
            default=species_list[:6],
        )

        st.markdown("---")
        gear_list = sorted(artefacts["dfs"]["gear"]["gear_type"].unique())
        selected_gears = st.multiselect(
            "Artes de pesca",
            options=gear_list,
            default=gear_list,
        )

        st.markdown("---")
        st.markdown("### 📥 Exportar GeoJSON")
        geojson_path = "outputs/geojson/ports_indicators.geojson"
        if os.path.exists(geojson_path):
            with open(geojson_path, "rb") as f:
                st.download_button(
                    "⬇️ Puertos (QGIS)",
                    data=f,
                    file_name="ports_indicators.geojson",
                    mime="application/geo+json",
                )
        cpue_path = "outputs/geojson/cpue_by_gear.geojson"
        if os.path.exists(cpue_path):
            with open(cpue_path, "rb") as f:
                st.download_button(
                    "⬇️ CPUE por arte (QGIS)",
                    data=f,
                    file_name="cpue_by_gear.geojson",
                    mime="application/geo+json",
                )
        csv_path = "outputs/geojson/master_timeseries.csv"
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button(
                    "⬇️ Serie temporal (CSV)",
                    data=f,
                    file_name="master_timeseries.csv",
                    mime="text/csv",
                )

    return yr_range, selected_ports, selected_species, selected_gears


# ─────────────────────────────────────────────
# TAB 1: RESUMEN GENERAL
# ─────────────────────────────────────────────
def tab_overview(artefacts, yr_range, selected_ports):
    master = artefacts["master"]
    filt = master[
        (master["year"].between(*yr_range)) &
        (master["local_norm"].isin(selected_ports))
    ]

    st.markdown('<h3 class="section-title">Métricas globales del período seleccionado</h3>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Producción total (t)", f"{filt['production_ton'].sum():,.1f}")
    c2.metric("CPUE media", f"{filt['cpue'].mean():.4f} t/viaje")
    c3.metric("Pescadores estimados", f"{filt['estimated_fishermen'].sum():,.0f}")
    c4.metric("Viajes asistidos", f"{filt['assisted_trips'].sum():,.0f}")
    c5.metric("Riqueza media (spp)", f"{filt['species_richness'].mean():.1f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Producción anual por puerto (t)")
        fig = px.bar(
            filt.groupby(["year", "port_name"])["production_ton"].sum().reset_index(),
            x="year", y="production_ton", color="port_name",
            barmode="stack", height=350,
            labels={"production_ton": "Producción (t)", "year": "Año", "port_name": "Puerto"},
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(legend_title_text="Puerto", margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Evolución CPUE media anual")
        cpue_yr = artefacts["cpue_port"][
            (artefacts["cpue_port"]["year"].between(*yr_range)) &
            (artefacts["cpue_port"]["local_norm"].isin(selected_ports))
        ]
        cpue_yr["port_name"] = cpue_yr["local_norm"].map(
            lambda x: PORT_COORDS.get(x, {}).get("name", x))
        fig2 = px.line(
            cpue_yr, x="year", y="cpue", color="port_name",
            markers=True, height=350,
            labels={"cpue": "CPUE (t/viaje)", "year": "Año", "port_name": "Puerto"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig2.update_layout(legend_title_text="Puerto", margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("#### Pescadores por puerto (últimos datos)")
        socio = artefacts["dfs"]["socioeco"]
        latest = socio[socio["year"] == socio["year"].max()]
        latest = latest[latest["local_norm"].isin(selected_ports)].copy()
        latest["port_name"] = latest["local_norm"].map(
            lambda x: PORT_COORDS.get(x, {}).get("name", x))
        fig3 = px.bar(
            latest, x="port_name", y="estimated_fishermen",
            color="port_name", height=300,
            labels={"estimated_fishermen": "Pescadores", "port_name": "Puerto"},
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig3.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.markdown("#### Índice de Shannon por puerto")
        bio_filt = artefacts["biodiv"][
            (artefacts["biodiv"]["year"].between(*yr_range)) &
            (artefacts["biodiv"]["local_norm"].isin(selected_ports))
        ]
        bio_filt = bio_filt.copy()
        bio_filt["port_name"] = bio_filt["local_norm"].map(
            lambda x: PORT_COORDS.get(x, {}).get("name", x))
        fig4 = px.box(
            bio_filt, x="port_name", y="shannon_index",
            color="port_name", height=300,
            labels={"shannon_index": "Shannon H'", "port_name": "Puerto"},
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        fig4.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 2: MAPAS INTERACTIVOS
# ─────────────────────────────────────────────
def tab_maps(artefacts, yr_range):
    st.markdown('<h3 class="section-title">Mapas interactivos GIS</h3>',
                unsafe_allow_html=True)

    map_choice = st.radio(
        "Seleccionar mapa",
        ["🐟 Distribución de especies", "⚓ CPUE por arte de pesca", "🌿 Hotspots de biodiversidad"],
        horizontal=True,
    )

    st.info(
        "Los mapas son completamente interactivos: zoom, capas, popups con metadatos. "
        "Haz clic en los marcadores para ver información detallada.",
        icon="ℹ️",
    )

    if map_choice == "🐟 Distribución de especies":
        st.markdown("**Distribución espacial de capturas por especie y puerto.** "
                    "El tamaño de los círculos es proporcional a la captura total. "
                    "El heatmap muestra la intensidad de capturas en el territorio.")
        m = species_distribution_map(artefacts["dfs"], year_range=yr_range)

    elif map_choice == "⚓ CPUE por arte de pesca":
        st.markdown("**CPUE (t/viaje) por arte de pesca y puerto.** "
                    "Artes activos en rojo, pasivos en azul, mixtos en morado.")
        m = cpue_map(artefacts["cpue_gear"], artefacts["master"], year_range=yr_range)

    else:
        st.markdown("**Hotspots de biodiversidad** según índice de Shannon-Wiener (H'). "
                    "Círculos más grandes = mayor riqueza de especies. "
                    "El heatmap azul indica mayor diversidad.")
        m = biodiversity_hotspot_map(artefacts["biodiv"], artefacts["master"], year_range=yr_range)

    st_folium(m, width="100%", height=520, returned_objects=[])

    # Tabla resumen bajo el mapa
    with st.expander("📋 Tabla de indicadores por puerto"):
        master_filt = artefacts["master"][artefacts["master"]["year"].between(*yr_range)]
        summary = master_filt.groupby("port_name").agg(
            Producción_t=("production_ton", "sum"),
            CPUE_media=("cpue", "mean"),
            Shannon_H=("shannon_index", "mean"),
            Riqueza_spp=("species_richness", "mean"),
            Pescadores=("estimated_fishermen", "mean"),
            Embarcaciones=("total_vessels", "mean"),
        ).round(3).reset_index()
        summary.columns = ["Puerto", "Producción (t)", "CPUE (t/viaje)", "Shannon H'",
                           "Riqueza (spp)", "Pescadores", "Embarcaciones"]
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# TAB 3: ANÁLISIS DE ESPECIES
# ─────────────────────────────────────────────
def tab_species(artefacts, yr_range, selected_ports, selected_species):
    st.markdown('<h3 class="section-title">Análisis de desembarques por especie</h3>',
                unsafe_allow_html=True)

    sp = artefacts["dfs"]["species"]
    filt = sp[
        (sp["year"].between(*yr_range)) &
        (sp["local_norm"].isin(selected_ports)) &
        (sp["species"].isin(selected_species))
    ].copy()
    filt["port_name"] = filt["local_norm"].map(
        lambda x: PORT_COORDS.get(x, {}).get("name", x))

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Top 15 especies por captura total")
        top15 = (
            filt.groupby("species")["sp_production_ton"].sum()
            .sort_values(ascending=False).head(15).reset_index()
        )
        fig = px.bar(
            top15, x="sp_production_ton", y="species",
            orientation="h", height=420,
            labels={"sp_production_ton": "Captura (t)", "species": "Especie"},
            color="sp_production_ton",
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Composición de capturas por puerto (%)")
        sp_port = filt.groupby(["port_name", "species"])["sp_production_ton"].sum().reset_index()
        top5_sp = (
            filt.groupby("species")["sp_production_ton"].sum()
            .sort_values(ascending=False).head(8).index.tolist()
        )
        sp_port_top = sp_port[sp_port["species"].isin(top5_sp)]
        fig2 = px.bar(
            sp_port_top, x="port_name", y="sp_production_ton",
            color="species", barmode="stack", height=420,
            labels={"sp_production_ton": "Captura (t)", "port_name": "Puerto", "species": "Especie"},
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig2.update_layout(legend_title_text="Especie", margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Evolución temporal de capturas por especie")
    sp_yr = filt.groupby(["year", "species"])["sp_production_ton"].sum().reset_index()
    top_sp_yr = (
        sp_yr.groupby("species")["sp_production_ton"].sum()
        .sort_values(ascending=False).head(10).index.tolist()
    )
    sp_yr_top = sp_yr[sp_yr["species"].isin(top_sp_yr)]
    fig3 = px.line(
        sp_yr_top, x="year", y="sp_production_ton", color="species",
        markers=True, height=380,
        labels={"sp_production_ton": "Captura (t)", "year": "Año", "species": "Especie"},
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )
    fig3.update_layout(legend_title_text="Especie", margin=dict(t=20))
    st.plotly_chart(fig3, use_container_width=True)

    # Heatmap especie × puerto
    st.markdown("#### Heatmap: captura (t) — Especie × Puerto")
    pivot = filt.groupby(["species", "port_name"])["sp_production_ton"].sum().unstack(fill_value=0)
    top_sp_heat = pivot.sum(axis=1).sort_values(ascending=False).head(20).index
    pivot = pivot.loc[top_sp_heat]
    fig4 = px.imshow(
        pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        color_continuous_scale="YlOrRd",
        labels={"x": "Puerto", "y": "Especie", "color": "Captura (t)"},
        height=500,
        aspect="auto",
    )
    fig4.update_layout(margin=dict(t=20))
    st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 4: ANÁLISIS DE ARTES
# ─────────────────────────────────────────────
def tab_gear(artefacts, yr_range, selected_ports, selected_gears):
    st.markdown('<h3 class="section-title">Artes de pesca y CPUE</h3>',
                unsafe_allow_html=True)

    gear = artefacts["dfs"]["gear"]
    pmdp = artefacts["dfs"]["pmdp"]

    filt_g = gear[
        (gear["year"].between(*yr_range)) &
        (gear["local_norm"].isin(selected_ports)) &
        (gear["gear_type"].isin(selected_gears))
    ].copy()
    filt_g["port_name"] = filt_g["local_norm"].map(
        lambda x: PORT_COORDS.get(x, {}).get("name", x))

    # Merge CPUE
    cg = artefacts["cpue_gear"][
        (artefacts["cpue_gear"]["year"].between(*yr_range)) &
        (artefacts["cpue_gear"]["local_norm"].isin(selected_ports)) &
        (artefacts["cpue_gear"]["gear_type"].isin(selected_gears))
    ].copy()
    cg["port_name"] = cg["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("name", x))

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Producción por arte de pesca (t)")
        gear_sum = filt_g.groupby("gear_type")["gear_production_ton"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(
            gear_sum, x="gear_production_ton", y="gear_type",
            orientation="h", height=380,
            color="gear_production_ton", color_continuous_scale="Teal",
            labels={"gear_production_ton": "Producción (t)", "gear_type": "Arte"},
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### CPUE media por arte de pesca")
        cpue_gear_sum = cg.groupby("gear_type")["cpue"].mean().sort_values(ascending=False).reset_index()
        fig2 = px.bar(
            cpue_gear_sum, x="cpue", y="gear_type",
            orientation="h", height=380,
            color="cpue", color_continuous_scale="Oranges",
            labels={"cpue": "CPUE (t/viaje)", "gear_type": "Arte"},
        )
        fig2.update_layout(coloraxis_showscale=False, margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("#### Distribución de CPUE por arte (boxplot)")
        fig3 = px.box(
            cg, x="gear_type", y="cpue", color="gear_group",
            height=380,
            labels={"cpue": "CPUE (t/viaje)", "gear_type": "Arte", "gear_group": "Grupo"},
            color_discrete_map={"active": "#e74c3c", "passive": "#2980b9", "mixed": "#8e44ad"},
        )
        fig3.update_xaxes(tickangle=45)
        fig3.update_layout(legend_title_text="Grupo", margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.markdown("#### Evolución temporal de producción por grupo de arte")
        grp_yr = filt_g.groupby(["year", "gear_group"])["gear_production_ton"].sum().reset_index()
        fig4 = px.area(
            grp_yr, x="year", y="gear_production_ton", color="gear_group",
            height=380,
            labels={"gear_production_ton": "Producción (t)", "year": "Año", "gear_group": "Grupo"},
            color_discrete_map={"active": "#e74c3c", "passive": "#2980b9", "mixed": "#8e44ad"},
        )
        fig4.update_layout(legend_title_text="Grupo", margin=dict(t=20))
        st.plotly_chart(fig4, use_container_width=True)

    # CPUE heatmap arte × puerto
    st.markdown("#### Heatmap CPUE media — Arte × Puerto")
    pivot_cpue = cg.groupby(["gear_type", "port_name"])["cpue"].mean().unstack(fill_value=0)
    fig5 = px.imshow(
        pivot_cpue.values,
        x=pivot_cpue.columns.tolist(),
        y=pivot_cpue.index.tolist(),
        color_continuous_scale="RdYlGn",
        labels={"x": "Puerto", "y": "Arte de pesca", "color": "CPUE (t/viaje)"},
        height=420, aspect="auto",
    )
    st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 5: ESTADÍSTICA Y CORRELACIONES
# ─────────────────────────────────────────────
def tab_stats(artefacts, yr_range, selected_ports):
    st.markdown('<h3 class="section-title">Análisis estadístico de correlaciones</h3>',
                unsafe_allow_html=True)

    master = artefacts["master"]
    filt = master[
        (master["year"].between(*yr_range)) &
        (master["local_norm"].isin(selected_ports))
    ].copy()

    pearson  = artefacts["pearson"]
    spearman = artefacts["spearman"]
    top_pairs = artefacts["top_pairs"]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Correlación de Pearson (variables clave)")
        fig_p = px.imshow(
            pearson.round(3),
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=True,
            height=420, aspect="auto",
        )
        fig_p.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_p, use_container_width=True)

    with col_r:
        st.markdown("#### Correlación de Spearman (variables clave)")
        fig_s = px.imshow(
            spearman.round(3),
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=True,
            height=420, aspect="auto",
        )
        fig_s.update_layout(margin=dict(t=20))
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Pares de variables con mayor correlación (|r| > 0.3)")
    top_sig = top_pairs[top_pairs["pearson_r"].abs() > 0.3].copy()
    top_sig["significativo"] = top_sig["p_value"].apply(
        lambda p: "✅ Sí (p<0.05)" if p is not None and p < 0.05 else "⚠️ No")
    top_sig = top_sig[["var1", "var2", "pearson_r", "spearman_r", "p_value", "significativo"]].head(20)
    top_sig.columns = ["Variable 1", "Variable 2", "Pearson r", "Spearman ρ", "p-valor", "Significativo"]
    st.dataframe(top_sig, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Scatter: CPUE vs Riqueza de especies")
    scatter_df = filt.dropna(subset=["cpue", "species_richness"])
    scatter_df["port_name"] = scatter_df["local_norm"].map(
        lambda x: PORT_COORDS.get(x, {}).get("name", x))
    fig_sc = px.scatter(
        scatter_df, x="cpue", y="species_richness",
        color="port_name", size="production_ton",
        trendline="ols",
        height=380,
        labels={"cpue": "CPUE (t/viaje)", "species_richness": "Riqueza (spp)", "port_name": "Puerto"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_sc.update_layout(legend_title_text="Puerto", margin=dict(t=20))
    st.plotly_chart(fig_sc, use_container_width=True)

    # Regresión CPUE ~ Pescadores
    st.markdown("#### Regresión lineal: CPUE ~ Pescadores estimados")
    reg_df = filt.dropna(subset=["cpue", "estimated_fishermen"])
    slope, intercept, r_val, p_val, std_err = scipy_stats.linregress(
        reg_df["estimated_fishermen"], reg_df["cpue"])
    st.write(f"**R²** = {r_val**2:.4f} | **pendiente** = {slope:.6f} | **p-valor** = {p_val:.4f}")
    fig_reg = px.scatter(
        reg_df, x="estimated_fishermen", y="cpue",
        trendline="ols", trendline_color_override="#e74c3c",
        height=340,
        labels={"estimated_fishermen": "Pescadores estimados", "cpue": "CPUE (t/viaje)"},
        color_discrete_sequence=["#2980b9"],
    )
    fig_reg.update_layout(margin=dict(t=20))
    st.plotly_chart(fig_reg, use_container_width=True)

    # Análisis ANOVA: CPUE por arte
    st.markdown("#### ANOVA de un factor: CPUE por grupo de arte de pesca")
    cpue_g = artefacts["cpue_gear"][
        (artefacts["cpue_gear"]["year"].between(*yr_range)) &
        (artefacts["cpue_gear"]["local_norm"].isin(selected_ports))
    ]
    groups = [grp["cpue"].dropna().values for _, grp in cpue_g.groupby("gear_group")]
    if len(groups) >= 2:
        f_stat, p_anova = scipy_stats.f_oneway(*groups)
        st.write(f"**F-estadístico** = {f_stat:.4f} | **p-valor** = {p_anova:.4f}")
        if p_anova < 0.05:
            st.success("✅ Diferencias significativas en CPUE entre grupos de artes (p < 0.05)")
        else:
            st.warning("⚠️ Sin diferencias significativas entre grupos de artes (p ≥ 0.05)")
        fig_box = px.box(
            cpue_g, x="gear_group", y="cpue", color="gear_group",
            height=320,
            labels={"cpue": "CPUE (t/viaje)", "gear_group": "Grupo de arte"},
            color_discrete_map={"active": "#e74c3c", "passive": "#2980b9", "mixed": "#8e44ad"},
        )
        fig_box.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig_box, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 6: VALOR DE PRODUCCIÓN
# ─────────────────────────────────────────────
def tab_value(artefacts, yr_range, selected_ports):
    st.markdown('<h3 class="section-title">Valor económico de la producción pesquera</h3>',
                unsafe_allow_html=True)

    pv = artefacts["dfs"]["prod_value"].copy()
    pv["port_name"] = pv["municipality_canonical"]
    filt = pv[
        (pv["year"].between(*yr_range)) &
        (pv["local_norm"].isin(selected_ports))
    ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Valor total (BRL)", f"R$ {filt['production_value'].sum():,.0f}")
    c2.metric("Valor medio anual (BRL)", f"R$ {filt['production_value'].mean():,.0f}")
    c3.metric("Municipios con datos", str(filt["municipality_canonical"].nunique()))

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Valor de producción por municipio y año")
        pv_yr = filt.groupby(["year", "port_name"])["production_value"].sum().reset_index()
        fig = px.line(
            pv_yr, x="year", y="production_value", color="port_name",
            markers=True, height=380,
            labels={"production_value": "Valor (BRL)", "year": "Año", "port_name": "Municipio"},
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig.update_layout(legend_title_text="Municipio", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Distribución del valor total por municipio")
        pv_total = filt.groupby("port_name")["production_value"].sum().reset_index()
        fig2 = px.pie(
            pv_total, names="port_name", values="production_value",
            height=380,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig2.update_layout(margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    # Tabla completa
    with st.expander("📋 Datos completos de valor de producción"):
        disp = filt[["year", "port_name", "semester", "production_value"]].copy()
        disp.columns = ["Año", "Municipio", "Semestre", "Valor (BRL)"]
        disp["Valor (BRL)"] = disp["Valor (BRL)"].apply(
            lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "—")
        st.dataframe(disp.sort_values(["Año", "Municipio"]), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
      <h1 style='margin:0;font-size:1.8rem'>🐟 Fisheries GIS Dashboard</h1>
      <p style='margin:4px 0 0;opacity:0.85'>
        Análisis espacial de capturas, CPUE, biodiversidad y valor económico —
        Litoral do Rio Grande do Norte, Brasil
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Cargar datos
    artefacts = get_data()

    # Sidebar + filtros
    yr_range, selected_ports, selected_species, selected_gears = sidebar(artefacts)

    if not selected_ports:
        st.warning("Selecciona al menos un puerto en el panel lateral.")
        return

    # Tabs
    tabs = st.tabs([
        "📊 Resumen general",
        "🗺️ Mapas interactivos",
        "🐠 Análisis de especies",
        "⚓ Artes de pesca",
        "📈 Correlaciones",
        "💰 Valor económico",
    ])

    with tabs[0]:
        tab_overview(artefacts, yr_range, selected_ports)
    with tabs[1]:
        tab_maps(artefacts, yr_range)
    with tabs[2]:
        tab_species(artefacts, yr_range, selected_ports, selected_species)
    with tabs[3]:
        tab_gear(artefacts, yr_range, selected_ports, selected_gears)
    with tabs[4]:
        tab_stats(artefacts, yr_range, selected_ports)
    with tabs[5]:
        tab_value(artefacts, yr_range, selected_ports)

    # Footer
    st.markdown("---")
    st.caption(
        "Dashboard desarrollado con GeoPandas · Folium · Streamlit · Plotly · SciPy. "
        "Datos: PMDP/IBAMA — Rio Grande do Norte, Brasil. "
        "Capas GeoJSON disponibles para importar en QGIS desde el panel lateral."
    )


if __name__ == "__main__":
    main()
