"""
╔══════════════════════════════════════════════════════════════════╗
║  FISHERIES GIS DASHBOARD — Rio Grande do Norte (Brazil)          ║
║  Catches, CPUE, Biodiversity and Spatial Analysis                ║
║  GAM Models, Multivariate Ordination, Composition Gradient       ║
╚══════════════════════════════════════════════════════════════════╝

Run:
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

# Ensure working directory is correct
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_pipeline import build_all
from utils.map_builder import species_distribution_map, cpue_map, biodiversity_hotspot_map
from utils.coords import PORT_COORDS, GEAR_LABELS
from utils.analysis_loader import load_analysis
from utils.analysis_tabs import (
    tab_exposure, tab_assoc, tab_gam, tab_robustness, tab_ordination,
    tab_gradient, tab_protected_areas, tab_methods_results
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fisheries GIS Dashboard · RN Brazil",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
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
# DATA LOADING (CACHED)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and processing data...")
def get_data():
    return build_all(export=True, output_dir="outputs/geojson")


@st.cache_data(show_spinner="Loading analysis datasets...")
def get_analysis():
    return load_analysis()


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
        st.markdown("## 🎛️ Global filters")
        master = artefacts["master"]
        years = sorted(master["year"].dropna().unique().astype(int))
        yr_range = st.slider(
            "Year range",
            min_value=int(years[0]), max_value=int(years[-1]),
            value=(int(years[0]), int(years[-1])),
            step=1,
        )

        st.markdown("---")
        st.markdown("**Ports**")
        ports = sorted(master["local_norm"].unique())
        selected_ports = st.multiselect(
            "Select ports",
            options=ports,
            default=ports,
            format_func=lambda x: PORT_COORDS.get(x, {}).get("name", x),
        )

        st.markdown("---")
        species_list = sorted(artefacts["dfs"]["species"]["species"].unique())
        selected_species = st.multiselect(
            "Species of interest",
            options=species_list,
            default=species_list[:6],
        )

        st.markdown("---")
        gear_list = sorted(artefacts["dfs"]["gear"]["gear_type"].unique())
        selected_gears = st.multiselect(
            "Fishing gears",
            options=gear_list,
            default=gear_list,
        )

        st.markdown("---")
        st.markdown("### 📥 Export GeoJSON")
        geojson_path = "outputs/geojson/ports_indicators.geojson"
        if os.path.exists(geojson_path):
            with open(geojson_path, "rb") as f:
                st.download_button(
                    "⬇️ Ports (QGIS)",
                    data=f,
                    file_name="ports_indicators.geojson",
                    mime="application/geo+json",
                )
        cpue_path = "outputs/geojson/cpue_by_gear.geojson"
        if os.path.exists(cpue_path):
            with open(cpue_path, "rb") as f:
                st.download_button(
                    "⬇️ CPUE by gear (QGIS)",
                    data=f,
                    file_name="cpue_by_gear.geojson",
                    mime="application/geo+json",
                )
        csv_path = "outputs/geojson/master_timeseries.csv"
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button(
                    "⬇️ Time series (CSV)",
                    data=f,
                    file_name="master_timeseries.csv",
                    mime="text/csv",
                )

    return yr_range, selected_ports, selected_species, selected_gears


# ─────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────
def tab_overview(artefacts, yr_range, selected_ports):
    master = artefacts["master"]
    filt = master[
        (master["year"].between(*yr_range)) &
        (master["local_norm"].isin(selected_ports))
    ]

    # ── KPI cards ────────────────────────────────────────────────────────────
    st.markdown('<h3 class="section-title">Global metrics for the selected period</h3>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total production (t)", f"{filt['production_ton'].sum():,.1f}")
    c2.metric("Mean CPUE", f"{filt['cpue'].mean():.4f} t/trip")
    c3.metric("Estimated fishers", f"{filt['estimated_fishermen'].sum():,.0f}")
    c4.metric("Assisted trips", f"{filt['assisted_trips'].sum():,.0f}")
    c5.metric("Mean richness (spp)", f"{filt['species_richness'].mean():.1f}")

    st.markdown("---")

    # ── Metric catalogue ─────────────────────────────────────────────────────
    # Each entry: (label, source, column, aggregation, unit)
    # source: "master" | "biodiv" | "cpue_port" | "socioeco"
    METRICS = {
        "Production (t)":              ("master",   "production_ton",        "sum",  "t"),
        "CPUE (t/trip)":               ("master",   "cpue",                  "mean", "t/trip"),
        "Species richness (S)":        ("master",   "species_richness",      "mean", "spp"),
        "Shannon H'":                  ("master",   "shannon_index",         "mean", "H'"),
        "Pielou J'":                   ("master",   "pielou_index",          "mean", "J'"),
        "Estimated fishers":           ("master",   "estimated_fishermen",   "sum",  "fishers"),
        "Fishers per vessel":          ("master",   "fishermen_per_vessel",  "mean", "fisher/vessel"),
        "Total vessels":               ("master",   "total_vessels",         "sum",  "vessels"),
        "Assisted trips":              ("master",   "assisted_trips",        "sum",  "trips"),
        "Fleet production (t)":        ("master",   "fleet_production_ton",  "sum",  "t"),
        "Production per species (t)":  ("master",   "sp_production_ton",     "sum",  "t"),
    }
    METRIC_KEYS = list(METRICS.keys())

    # Chart type options
    CHART_TYPES = ["Line (by port)", "Stacked bar (by port)", "Area (by port)",
                   "Line (total)", "Bar (total)", "Box (by port)"]

    # Default selections per panel
    DEFAULTS = [
        ("Production (t)",      "Stacked bar (by port)"),
        ("CPUE (t/trip)",       "Line (by port)"),
        ("Estimated fishers",   "Line (by port)"),
        ("Shannon H'",          "Box (by port)"),
    ]

    PORT_COLOR_SEQ = px.colors.qualitative.Bold

    def build_chart(panel_id, default_metric, default_chart):
        """Renders one configurable chart panel."""
        sel_col1, sel_col2 = st.columns([3, 2])
        metric_key = sel_col1.selectbox(
            "Metric (Y axis)",
            options=METRIC_KEYS,
            index=METRIC_KEYS.index(default_metric),
            key=f"metric_{panel_id}",
        )
        chart_type = sel_col2.selectbox(
            "Chart type",
            options=CHART_TYPES,
            index=CHART_TYPES.index(default_chart),
            key=f"chart_{panel_id}",
        )

        src, col, agg, unit = METRICS[metric_key]

        # Build per-year data
        df_src = master[
            (master["year"].between(*yr_range)) &
            (master["local_norm"].isin(selected_ports))
        ].copy()
        df_src["port_name"] = df_src["local_norm"].map(
            lambda x: PORT_COORDS.get(x, {}).get("name", x))

        # Aggregate
        agg_fn = {"sum": "sum", "mean": "mean"}[agg]
        by_port = (df_src.groupby(["year", "port_name"])[col]
                   .agg(agg_fn).reset_index())
        by_total = (df_src.groupby("year")[col]
                    .agg(agg_fn).reset_index())

        ylab = f"{metric_key} ({unit})"
        h = 320

        if chart_type == "Line (by port)":
            fig = px.line(by_port, x="year", y=col, color="port_name",
                          markers=True, height=h,
                          labels={col: ylab, "year": "Year", "port_name": "Port"},
                          color_discrete_sequence=PORT_COLOR_SEQ)
            fig.update_layout(legend_title_text="Port")

        elif chart_type == "Stacked bar (by port)":
            fig = px.bar(by_port, x="year", y=col, color="port_name",
                         barmode="stack", height=h,
                         labels={col: ylab, "year": "Year", "port_name": "Port"},
                         color_discrete_sequence=PORT_COLOR_SEQ)
            fig.update_layout(legend_title_text="Port")

        elif chart_type == "Area (by port)":
            fig = px.area(by_port, x="year", y=col, color="port_name",
                          height=h,
                          labels={col: ylab, "year": "Year", "port_name": "Port"},
                          color_discrete_sequence=PORT_COLOR_SEQ)
            fig.update_layout(legend_title_text="Port")

        elif chart_type == "Line (total)":
            fig = px.line(by_total, x="year", y=col,
                          markers=True, height=h,
                          labels={col: ylab, "year": "Year"})
            fig.update_traces(line_color="#1a6eb5", line_width=2.5)

        elif chart_type == "Bar (total)":
            fig = px.bar(by_total, x="year", y=col, height=h,
                         labels={col: ylab, "year": "Year"},
                         color=col,
                         color_continuous_scale="Blues")
            fig.update_layout(coloraxis_showscale=False)

        else:  # Box (by port)
            fig = px.box(df_src, x="port_name", y=col, color="port_name",
                         height=h,
                         labels={col: ylab, "port_name": "Port"},
                         color_discrete_sequence=PORT_COLOR_SEQ)
            fig.update_layout(showlegend=False)

        # Add trend line for non-box charts
        if "by port" in chart_type and chart_type not in ("Box (by port)", "Stacked bar (by port)"):
            # per-port OLS trend as dashed line
            for port, grp in by_port.groupby("port_name"):
                if len(grp) >= 3:
                    z = np.polyfit(grp["year"], grp[col].fillna(0), 1)
                    p = np.poly1d(z)
                    yrs = np.linspace(grp["year"].min(), grp["year"].max(), 50)
                    fig.add_scatter(x=yrs, y=p(yrs),
                                    mode="lines",
                                    line=dict(dash="dot", width=1),
                                    showlegend=False,
                                    opacity=0.5)

        elif chart_type in ("Line (total)", "Bar (total)") and len(by_total) >= 3:
            z = np.polyfit(by_total["year"], by_total[col].fillna(0), 1)
            p = np.poly1d(z)
            yrs = np.linspace(by_total["year"].min(), by_total["year"].max(), 50)
            fig.add_scatter(x=yrs, y=p(yrs),
                            mode="lines",
                            line=dict(color="#e74c3c", dash="dash", width=1.8),
                            name="Trend",
                            showlegend=True)

        fig.update_layout(margin=dict(t=8, b=20, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True, key=f"overview_chart_{panel_id}")

    # ── 2×2 grid of panels ───────────────────────────────────────────────────
    row1_l, row1_r = st.columns(2)
    with row1_l:
        build_chart("A", *DEFAULTS[0])
    with row1_r:
        build_chart("B", *DEFAULTS[1])

    st.markdown("")
    row2_l, row2_r = st.columns(2)
    with row2_l:
        build_chart("C", *DEFAULTS[2])
    with row2_r:
        build_chart("D", *DEFAULTS[3])



# ─────────────────────────────────────────────
# TAB 2: INTERACTIVE MAPS
# ─────────────────────────────────────────────
def tab_maps(artefacts, yr_range):
    st.markdown('<h3 class="section-title">Interactive GIS maps</h3>',
                unsafe_allow_html=True)

    map_choice = st.radio(
        "Select map",
        ["🐟 Species distribution", "⚓ CPUE by fishing gear", "🌿 Biodiversity hotspots"],
        horizontal=True,
    )

    st.info(
        "Maps are fully interactive: zoom, layers, popups with metadata. "
        "Click on markers for detailed information.",
        icon="ℹ️",
    )

    if map_choice == "🐟 Species distribution":
        st.markdown("**Spatial distribution of catches by species and port.** "
                    "Circle size is proportional to total catch. "
                    "The heatmap shows catch intensity across the territory.")
        m = species_distribution_map(artefacts["dfs"], year_range=yr_range)

    elif map_choice == "⚓ CPUE by fishing gear":
        st.markdown("**CPUE (t/trip) by fishing gear and port.** "
                    "Active gears in red, passive in blue, mixed in purple.")
        m = cpue_map(artefacts["cpue_gear"], artefacts["master"], year_range=yr_range)

    else:
        st.markdown("**Biodiversity hotspots** based on the Shannon-Wiener index (H'). "
                    "Larger circles = higher species richness. "
                    "The blue heatmap indicates higher diversity.")
        m = biodiversity_hotspot_map(artefacts["biodiv"], artefacts["master"], year_range=yr_range)

    st_folium(m, width="100%", height=520, returned_objects=[])

    # Summary table below map
    with st.expander("📋 Port indicators table"):
        master_filt = artefacts["master"][artefacts["master"]["year"].between(*yr_range)]
        summary = master_filt.groupby("port_name").agg(
            Production_t=("production_ton", "sum"),
            Mean_CPUE=("cpue", "mean"),
            Shannon_H=("shannon_index", "mean"),
            Richness_spp=("species_richness", "mean"),
            Fishers=("estimated_fishermen", "mean"),
            Vessels=("total_vessels", "mean"),
        ).round(3).reset_index()
        summary.columns = ["Port", "Production (t)", "CPUE (t/trip)", "Shannon H'",
                           "Richness (spp)", "Fishers", "Vessels"]
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# TAB 3: SPECIES ANALYSIS
# ─────────────────────────────────────────────
def tab_species(artefacts, yr_range, selected_ports, selected_species):
    st.markdown('<h3 class="section-title">Landings analysis by species</h3>',
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
        st.markdown("#### Top 15 species by total catch")
        top15 = (
            filt.groupby("species")["sp_production_ton"].sum()
            .sort_values(ascending=False).head(15).reset_index()
        )
        fig = px.bar(
            top15, x="sp_production_ton", y="species",
            orientation="h", height=420,
            labels={"sp_production_ton": "Catch (t)", "species": "Species"},
            color="sp_production_ton",
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Catch composition by port (%)")
        sp_port = filt.groupby(["port_name", "species"])["sp_production_ton"].sum().reset_index()
        top5_sp = (
            filt.groupby("species")["sp_production_ton"].sum()
            .sort_values(ascending=False).head(8).index.tolist()
        )
        sp_port_top = sp_port[sp_port["species"].isin(top5_sp)]
        fig2 = px.bar(
            sp_port_top, x="port_name", y="sp_production_ton",
            color="species", barmode="stack", height=420,
            labels={"sp_production_ton": "Catch (t)", "port_name": "Port", "species": "Species"},
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig2.update_layout(legend_title_text="Species", margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Temporal trend of catches by species")
    sp_yr = filt.groupby(["year", "species"])["sp_production_ton"].sum().reset_index()
    top_sp_yr = (
        sp_yr.groupby("species")["sp_production_ton"].sum()
        .sort_values(ascending=False).head(10).index.tolist()
    )
    sp_yr_top = sp_yr[sp_yr["species"].isin(top_sp_yr)]
    fig3 = px.line(
        sp_yr_top, x="year", y="sp_production_ton", color="species",
        markers=True, height=380,
        labels={"sp_production_ton": "Catch (t)", "year": "Year", "species": "Species"},
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )
    fig3.update_layout(legend_title_text="Species", margin=dict(t=20))
    st.plotly_chart(fig3, use_container_width=True)

    # Species × Port heatmap
    st.markdown("#### Heatmap: catch (t) — Species × Port")
    pivot = filt.groupby(["species", "port_name"])["sp_production_ton"].sum().unstack(fill_value=0)
    top_sp_heat = pivot.sum(axis=1).sort_values(ascending=False).head(20).index
    pivot = pivot.loc[top_sp_heat]
    fig4 = px.imshow(
        pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        color_continuous_scale="YlOrRd",
        labels={"x": "Port", "y": "Species", "color": "Catch (t)"},
        height=500,
        aspect="auto",
    )
    fig4.update_layout(margin=dict(t=20))
    st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 4: GEAR ANALYSIS
# ─────────────────────────────────────────────
def tab_gear(artefacts, yr_range, selected_ports, selected_gears):
    st.markdown('<h3 class="section-title">Fishing gears and CPUE</h3>',
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
        st.markdown("#### Production by fishing gear (t)")
        gear_sum = filt_g.groupby("gear_type")["gear_production_ton"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(
            gear_sum, x="gear_production_ton", y="gear_type",
            orientation="h", height=380,
            color="gear_production_ton", color_continuous_scale="Teal",
            labels={"gear_production_ton": "Production (t)", "gear_type": "Gear"},
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Mean CPUE by fishing gear")
        cpue_gear_sum = cg.groupby("gear_type")["cpue"].mean().sort_values(ascending=False).reset_index()
        fig2 = px.bar(
            cpue_gear_sum, x="cpue", y="gear_type",
            orientation="h", height=380,
            color="cpue", color_continuous_scale="Oranges",
            labels={"cpue": "CPUE (t/trip)", "gear_type": "Gear"},
        )
        fig2.update_layout(coloraxis_showscale=False, margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("#### CPUE distribution by gear (boxplot)")
        fig3 = px.box(
            cg, x="gear_type", y="cpue", color="gear_group",
            height=380,
            labels={"cpue": "CPUE (t/trip)", "gear_type": "Gear", "gear_group": "Group"},
            color_discrete_map={"active": "#e74c3c", "passive": "#2980b9", "mixed": "#8e44ad"},
        )
        fig3.update_xaxes(tickangle=45)
        fig3.update_layout(legend_title_text="Group", margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.markdown("#### Temporal trend of production by gear group")
        grp_yr = filt_g.groupby(["year", "gear_group"])["gear_production_ton"].sum().reset_index()
        fig4 = px.area(
            grp_yr, x="year", y="gear_production_ton", color="gear_group",
            height=380,
            labels={"gear_production_ton": "Production (t)", "year": "Year", "gear_group": "Group"},
            color_discrete_map={"active": "#e74c3c", "passive": "#2980b9", "mixed": "#8e44ad"},
        )
        fig4.update_layout(legend_title_text="Group", margin=dict(t=20))
        st.plotly_chart(fig4, use_container_width=True)

    # CPUE heatmap gear × port
    st.markdown("#### CPUE heatmap — Gear × Port")
    pivot_cpue = cg.groupby(["gear_type", "port_name"])["cpue"].mean().unstack(fill_value=0)
    fig5 = px.imshow(
        pivot_cpue.values,
        x=pivot_cpue.columns.tolist(),
        y=pivot_cpue.index.tolist(),
        color_continuous_scale="RdYlGn",
        labels={"x": "Port", "y": "Fishing gear", "color": "CPUE (t/trip)"},
        height=420, aspect="auto",
    )
    st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 5: STATISTICS & CORRELATIONS
# ─────────────────────────────────────────────
def tab_stats(artefacts, yr_range, selected_ports):
    st.markdown('<h3 class="section-title">Statistical correlation analysis</h3>',
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
        st.markdown("#### Pearson correlation (key variables)")
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
        st.markdown("#### Spearman correlation (key variables)")
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
    st.markdown("#### Variable pairs with strongest correlation (|r| > 0.3)")
    top_sig = top_pairs[top_pairs["pearson_r"].abs() > 0.3].copy()
    top_sig["significant"] = top_sig["p_value"].apply(
        lambda p: "✅ Yes (p<0.05)" if p is not None and p < 0.05 else "⚠️ No")
    top_sig = top_sig[["var1", "var2", "pearson_r", "spearman_r", "p_value", "significant"]].head(20)
    top_sig.columns = ["Variable 1", "Variable 2", "Pearson r", "Spearman ρ", "p-value", "Significant"]
    st.dataframe(top_sig, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Scatter: CPUE vs Species richness")
    scatter_df = filt.dropna(subset=["cpue", "species_richness"])
    scatter_df["port_name"] = scatter_df["local_norm"].map(
        lambda x: PORT_COORDS.get(x, {}).get("name", x))
    fig_sc = px.scatter(
        scatter_df, x="cpue", y="species_richness",
        color="port_name", size="production_ton",
        trendline="ols",
        height=380,
        labels={"cpue": "CPUE (t/trip)", "species_richness": "Richness (spp)", "port_name": "Port"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_sc.update_layout(legend_title_text="Port", margin=dict(t=20))
    st.plotly_chart(fig_sc, use_container_width=True)

    # Regression CPUE ~ Fishers
    st.markdown("#### Linear regression: CPUE ~ Estimated fishers")
    reg_df = filt.dropna(subset=["cpue", "estimated_fishermen"])
    slope, intercept, r_val, p_val, std_err = scipy_stats.linregress(
        reg_df["estimated_fishermen"], reg_df["cpue"])
    st.write(f"**R²** = {r_val**2:.4f} | **slope** = {slope:.6f} | **p-value** = {p_val:.4f}")
    fig_reg = px.scatter(
        reg_df, x="estimated_fishermen", y="cpue",
        trendline="ols", trendline_color_override="#e74c3c",
        height=340,
        labels={"estimated_fishermen": "Estimated fishers", "cpue": "CPUE (t/trip)"},
        color_discrete_sequence=["#2980b9"],
    )
    fig_reg.update_layout(margin=dict(t=20))
    st.plotly_chart(fig_reg, use_container_width=True)

    # One-way ANOVA: CPUE by gear group
    st.markdown("#### One-way ANOVA: CPUE by fishing gear group")
    cpue_g = artefacts["cpue_gear"][
        (artefacts["cpue_gear"]["year"].between(*yr_range)) &
        (artefacts["cpue_gear"]["local_norm"].isin(selected_ports))
    ]
    groups = [grp["cpue"].dropna().values for _, grp in cpue_g.groupby("gear_group")]
    if len(groups) >= 2:
        f_stat, p_anova = scipy_stats.f_oneway(*groups)
        st.write(f"**F-statistic** = {f_stat:.4f} | **p-value** = {p_anova:.4f}")
        if p_anova < 0.05:
            st.success("✅ Significant differences in CPUE across gear groups (p < 0.05)")
        else:
            st.warning("⚠️ No significant differences across gear groups (p ≥ 0.05)")
        fig_box = px.box(
            cpue_g, x="gear_group", y="cpue", color="gear_group",
            height=320,
            labels={"cpue": "CPUE (t/trip)", "gear_group": "Gear group"},
            color_discrete_map={"active": "#e74c3c", "passive": "#2980b9", "mixed": "#8e44ad"},
        )
        fig_box.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig_box, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 6: PRODUCTION VALUE
# ─────────────────────────────────────────────
def tab_value(artefacts, yr_range, selected_ports):
    st.markdown('<h3 class="section-title">Economic value of fisheries production</h3>',
                unsafe_allow_html=True)

    pv = artefacts["dfs"]["prod_value"].copy()
    pv["port_name"] = pv["municipality_canonical"]
    filt = pv[
        (pv["year"].between(*yr_range)) &
        (pv["local_norm"].isin(selected_ports))
    ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total value (BRL)", f"R$ {filt['production_value'].sum():,.0f}")
    c2.metric("Mean annual value (BRL)", f"R$ {filt['production_value'].mean():,.0f}")
    c3.metric("Municipalities with data", str(filt["municipality_canonical"].nunique()))

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Production value by municipality and year")
        pv_yr = filt.groupby(["year", "port_name"])["production_value"].sum().reset_index()
        fig = px.line(
            pv_yr, x="year", y="production_value", color="port_name",
            markers=True, height=380,
            labels={"production_value": "Value (BRL)", "year": "Year", "port_name": "Municipality"},
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig.update_layout(legend_title_text="Municipality", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Total value distribution by municipality")
        pv_total = filt.groupby("port_name")["production_value"].sum().reset_index()
        fig2 = px.pie(
            pv_total, names="port_name", values="production_value",
            height=380,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig2.update_layout(margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    # Full table
    with st.expander("📋 Complete production value data"):
        disp = filt[["year", "port_name", "semester", "production_value"]].copy()
        disp.columns = ["Year", "Municipality", "Semester", "Value (BRL)"]
        disp["Value (BRL)"] = disp["Value (BRL)"].apply(
            lambda x: f"R$ {x:,.2f}" if pd.notna(x) else "—")
        st.dataframe(disp.sort_values(["Year", "Municipality"]), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
      <h1 style='margin:0;font-size:1.8rem'>🐟 Fisheries GIS Dashboard</h1>
      <p style='margin:4px 0 0;opacity:0.85'>
        Spatial analysis of catches, CPUE, biodiversity and economic value —
        Rio Grande do Norte coast, Brazil
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    artefacts = get_data()
    ad = get_analysis()

    # Sidebar + filters
    yr_range, selected_ports, selected_species, selected_gears = sidebar(artefacts)

    if not selected_ports:
        st.warning("Please select at least one port in the sidebar.")
        return

    # Tabs — data exploration
    st.markdown("### 📊 Data exploration")
    tabs_data = st.tabs([
        "📊 Overview",
        "🗺️ Interactive maps",
        "🐠 Species analysis",
        "⚓ Fishing gears",
        "📈 Correlations",
        "💰 Economic value",
    ])
    with tabs_data[0]:
        tab_overview(artefacts, yr_range, selected_ports)
    with tabs_data[1]:
        tab_maps(artefacts, yr_range)
    with tabs_data[2]:
        tab_species(artefacts, yr_range, selected_ports, selected_species)
    with tabs_data[3]:
        tab_gear(artefacts, yr_range, selected_ports, selected_gears)
    with tabs_data[4]:
        tab_stats(artefacts, yr_range, selected_ports)
    with tabs_data[5]:
        tab_value(artefacts, yr_range, selected_ports)

    # Tabs — analysis results
    st.markdown("---")
    st.markdown("### 🔬 Analysis results")
    tabs_analysis = st.tabs([
        "📡 Platform exposure",
        "🔗 Associations",
        "📈 GAM models",
        "🔬 Model robustness",
        "🦭 Multivariate ordination",
        "🌊 Composition gradient",
        "🛡️ Protected areas",
        "📄 Methods & Results",
    ])
    with tabs_analysis[0]:
        tab_exposure(ad)
    with tabs_analysis[1]:
        tab_assoc(ad)
    with tabs_analysis[2]:
        tab_gam(ad)
    with tabs_analysis[3]:
        tab_robustness(ad)
    with tabs_analysis[4]:
        tab_ordination(ad)
    with tabs_analysis[5]:
        tab_gradient(ad)
    with tabs_analysis[6]:
        tab_protected_areas(ad)
    with tabs_analysis[7]:
        tab_methods_results(ad)

    # Footer
    st.markdown("---")
    st.caption(
        "Dashboard built with Folium · Streamlit · Plotly · SciPy. "
        "Data: PMDP/IBAMA — Rio Grande do Norte, Brazil. "
        "GeoJSON layers available for import into QGIS from the sidebar."
    )


if __name__ == "__main__":
    main()
