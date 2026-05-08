"""
app_explorer.py
===============
Streamlit explorer for "Fish catches × Oil platforms Brazil".

Run:
    conda run -n spyder-env streamlit run app_explorer.py
"""

import os
import sys
import warnings
from pathlib import Path

# Fix PROJ_DATA before any geopandas/pyproj import
_proj = Path(sys.executable).parent.parent / "share" / "proj"
if _proj.exists():
    os.environ["PROJ_DATA"] = str(_proj)
warnings.filterwarnings("ignore")

import json
import tempfile
import zipfile

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

ROOT           = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INTERIM   = ROOT / "data" / "interim"
DATA_RAW       = ROOT / "data" / "raw"

# ─── Palettes & ordering (mirrors config_00.py) ───────────────────────────────
PALETTE_PLATFORM = {
    "0-20 km":    "#d73027",
    "20-50 km":   "#fc8d59",
    "50-100 km":  "#fee08b",
    "100-200 km": "#91bfdb",
    ">200 km":    "#4575b4",
}
PALETTE_MPA = {
    "inside":   "#1a9850",
    "0-10 km":  "#91cf60",
    "10-25 km": "#d9ef8b",
    "25-50 km": "#fee08b",
    ">50 km":   "#d73027",
}
PLATFORM_ORDER = ["0-20 km", "20-50 km", "50-100 km", "100-200 km", ">200 km"]
MPA_ORDER      = ["inside", "0-10 km", "10-25 km", "25-50 km", ">50 km"]
PERIOD_ORDER   = ["early", "middle", "recent"]
PERIOD_LABEL   = {"early": "2001–2008", "middle": "2009–2015", "recent": "2016–2024"}


# ─── Generic helpers ──────────────────────────────────────────────────────────
def _load(name: str, sub: str = "processed") -> pd.DataFrame | None:
    base = DATA_PROCESSED if sub == "processed" else DATA_INTERIM
    p = base / name
    return pd.read_csv(p) if p.exists() else None


def _cat(df: pd.DataFrame, col: str, order: list) -> pd.DataFrame:
    present = [c for c in order if c in df[col].dropna().unique()]
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=present, ordered=True)
    return df


def _period_bands(fig) -> None:
    fig.add_vrect(x0=2001, x1=2008, fillcolor="#e8f5e9", opacity=0.25,
                  annotation_text="early", annotation_position="top left")
    fig.add_vrect(x0=2009, x1=2015, fillcolor="#fff3e0", opacity=0.25,
                  annotation_text="middle", annotation_position="top left")
    fig.add_vrect(x0=2016, x1=2024, fillcolor="#e3f2fd", opacity=0.25,
                  annotation_text="recent", annotation_position="top left")


# ─── Cached data loaders ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all() -> dict:
    spec = {
        # key:                 (filename,                              subfolder)
        "regional":           ("timeseries_regional.csv",             "processed"),
        "ts_platform":        ("timeseries_platform.csv",             "processed"),
        "ts_mpa":             ("timeseries_mpa.csv",                  "processed"),
        "ts_gear":            ("timeseries_gear_year.csv",            "processed"),
        "ts_boat":            ("timeseries_boat_year.csv",            "processed"),
        "ts_species_top20":   ("timeseries_species_top20.csv",        "processed"),
        "period_cmp":         ("period_comparison.csv",               "processed"),
        "prod_local_year":    ("productivity_local_year.csv",         "processed"),
        "prod_platform":      ("productivity_platform_year.csv",      "processed"),
        "prod_mpa":           ("productivity_mpa_year.csv",           "processed"),
        "gear_plat":          ("gear_share_platform_year.csv",        "processed"),
        "gear_mpa":           ("gear_share_mpa_year.csv",             "processed"),
        "gear_year":          ("gear_year.csv",                       "processed"),
        "boat_year":          ("boat_year.csv",                       "processed"),
        "species_rank_plat":  ("species_rank_platform.csv",           "processed"),
        "species_rank_mpa":   ("species_rank_mpa.csv",                "processed"),
        "simper_plat":        ("species_SIMPER_platform.csv",         "processed"),
        "simper_mpa":         ("species_SIMPER_mpa.csv",              "processed"),
        "sp_share_local":     ("species_share_local_year.csv",        "processed"),
        "sp_year":            ("species_year.csv",                    "processed"),
        "turnover":           ("species_turnover_period.csv",         "processed"),
        "exposure":           ("local_exposure.csv",                  "interim"),
        "lp_exposure":        ("landing_points_exposure.csv",         "interim"),
        "socioeconomic":      ("socioeconomic_clean.csv",             "interim"),
        "reconciliation":     ("production_reconciliation.csv",       "processed"),
    }
    return {k: _load(f, s) for k, (f, s) in spec.items()}


@st.cache_data(show_spinner=False)
def load_spatial() -> dict:
    result: dict[str, gpd.GeoDataFrame] = {}
    try:
        p = DATA_INTERIM / "mpas_combined.gpkg"
        if p.exists():
            result["mpas"] = gpd.read_file(p).to_crs("EPSG:4326")
    except Exception:
        pass
    try:
        p = DATA_INTERIM / "platforms_projected.gpkg"
        if p.exists():
            result["platforms"] = gpd.read_file(p).to_crs("EPSG:4326")
    except Exception:
        pass
    try:
        z = DATA_RAW / "RN_Municipios_2025.zip"
        if z.exists():
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(td)
                shps = list(Path(td).rglob("*.shp"))
                if shps:
                    result["municipalities"] = gpd.read_file(shps[0]).to_crs("EPSG:4326")
    except Exception:
        pass
    return result


@st.cache_data(show_spinner=False)
def local_centroids(lp_hash: str) -> pd.DataFrame:
    lp = _load("landing_points_exposure.csv", "interim")
    if lp is None:
        return pd.DataFrame(columns=["local", "lat", "lon"])
    return (
        lp.dropna(subset=["latitude", "longitude"])
        .groupby("local", as_index=False)
        .agg(lat=("latitude", "mean"), lon=("longitude", "mean"))
    )


# ─── Tab: Summary ─────────────────────────────────────────────────────────────
def tab_resumen(d: dict) -> None:
    reg = d["regional"]
    if reg is None:
        st.warning("timeseries_regional.csv not found."); return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total production (t)",    f"{reg['production_ton_sum'].sum():,.0f}")
    c2.metric("Years covered",           f"{int(reg['year'].min())}–{int(reg['year'].max())}")
    c3.metric("Max. localities/year",    f"{int(reg['n_locals'].max())}")
    c4.metric("Total trips",             f"{reg['assisted_trips_sum'].sum():,.0f}")
    c5.metric("Mean CPUE (t/trip)",      f"{reg['cpue_regional'].mean():.4f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(reg, x="year", y="production_ton_sum",
                     labels={"production_ton_sum": "Production (t)", "year": "Year"},
                     title="Annual regional production",
                     color_discrete_sequence=["#2171b5"])
        _period_bands(fig)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, key="pc_181")

    with col2:
        fig2 = px.line(reg, x="year", y="cpue_regional", markers=True,
                       labels={"cpue_regional": "CPUE (t/trip)", "year": "Year"},
                       title="Regional CPUE",
                       color_discrete_sequence=["#e6550d"])
        _period_bands(fig2)
        fig2.update_layout(hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True, key="pc_190")

    # Fishermen & fleet
    soc = d["socioeconomic"]
    if soc is not None:
        col3, col4 = st.columns(2)
        with col3:
            tot_fish = soc.groupby("year", as_index=False)["estimated_fishermen"].sum()
            fig3 = px.bar(tot_fish, x="year", y="estimated_fishermen",
                          labels={"estimated_fishermen": "Estimated fishers", "year": "Year"},
                          title="Estimated fishers (regional total)",
                          color_discrete_sequence=["#756bb1"])
            _period_bands(fig3)
            st.plotly_chart(fig3, use_container_width=True, key="pc_203")
        with col4:
            fpv = soc.groupby("year", as_index=False)["fishermen_per_vessel"].mean()
            fig4 = px.line(fpv, x="year", y="fishermen_per_vessel", markers=True,
                           labels={"fishermen_per_vessel": "Fishers/vessel", "year": "Year"},
                           title="Fishers per vessel (regional mean)",
                           color_discrete_sequence=["#7b2d8b"])
            _period_bands(fig4)
            st.plotly_chart(fig4, use_container_width=True, key="pc_211")

    # Period comparison
    pc = d["period_cmp"]
    if pc is not None:
        st.subheader("Comparison by period and exposure class")
        pc = pc.copy()
        pc["Period"] = pc["period"].map(PERIOD_LABEL).fillna(pc["period"])
        st.dataframe(
            pc.rename(columns={
                "platform_exposure_class": "Platform class",
                "mpa_exposure_class":      "MPA class",
                "production_ton":          "Production (t)",
                "assisted_trips":          "Trips",
                "cpue_mean":               "Mean CPUE",
                "cpue_agg":                "Aggregated CPUE",
                "n_locals":                "N localities",
            })[["Period", "Platform class", "MPA class",
                "Production (t)", "Trips", "Mean CPUE", "Aggregated CPUE", "N localities"]],
            use_container_width=True, hide_index=True,
        )

    # Production reconciliation
    rec = d["reconciliation"]
    if rec is not None:
        with st.expander("Production reconciliation (master vs gear vs species)"):
            st.dataframe(rec.style.background_gradient(
                subset=["pct_diff_sp"], cmap="RdYlGn_r", vmin=0, vmax=0.1),
                use_container_width=True, hide_index=True)


# ─── Tab: CPUE & Productivity ─────────────────────────────────────────────────
def tab_cpue(d: dict) -> None:
    view = st.radio("Group by:", ["Distance to platform", "Distance to MPA"],
                    horizontal=True)

    if view == "Distance to platform":
        df, col, pal, order = d["ts_platform"], "platform_exposure_class", PALETTE_PLATFORM, PLATFORM_ORDER
        df_local = d["prod_platform"]
    else:
        df, col, pal, order = d["ts_mpa"], "mpa_exposure_class", PALETTE_MPA, MPA_ORDER
        df_local = d["prod_mpa"]

    if df is None:
        st.warning("Data not available."); return

    df   = _cat(df, col, order)
    cats = [c for c in order if c in df[col].cat.categories]

    # CPUE time series
    fig = go.Figure()
    for cls in cats:
        sub = df[df[col] == cls].sort_values("year")
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["cpue"],
            mode="lines+markers", name=cls,
            line=dict(color=pal.get(cls, "#888"), width=2),
            marker=dict(size=5),
        ))
    _period_bands(fig)
    fig.update_layout(
        title=f"CPUE by {col.replace('_', ' ')}",
        xaxis_title="Year", yaxis_title="CPUE (t/trip)",
        hovermode="x unified", legend_title="Class",
    )
    st.plotly_chart(fig, use_container_width=True, key="pc_276")

    col1, col2 = st.columns(2)

    with col1:
        # Production stacked area
        fig2 = px.area(
            df.sort_values(["year", col]),
            x="year", y="production_ton", color=col,
            color_discrete_map=pal,
            category_orders={col: cats},
            labels={"production_ton": "Production (t)", "year": "Year", col: "Class"},
            title="Total production by class",
        )
        st.plotly_chart(fig2, use_container_width=True, key="pc_290")

    with col2:
        # CPUE boxplot from local×year
        ly = d["prod_local_year"]
        if ly is not None and col in ly.columns:
            ly_f = _cat(ly.dropna(subset=[col, "cpue_ton_per_trip"]), col, order)
            fig3 = px.box(
                ly_f.sort_values(col),
                x=col, y="cpue_ton_per_trip",
                color=col, color_discrete_map=pal,
                category_orders={col: cats},
                points="outliers",
                labels={"cpue_ton_per_trip": "CPUE (t/trip)", col: "Class"},
                title="CPUE distribution by class (locality × year)",
            )
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True, key="pc_307")

    # Period × exposure heatmap
    pc = d["period_cmp"]
    if pc is not None:
        st.subheader("CPUE heatmap: period × platform class")
        pc_v = pc.dropna(subset=["platform_exposure_class", "period"])
        piv  = pc_v.pivot_table(index="platform_exposure_class",
                                 columns="period", values="cpue_agg", aggfunc="mean")
        piv  = piv.reindex([c for c in PLATFORM_ORDER if c in piv.index])
        piv  = piv[[p for p in PERIOD_ORDER if p in piv.columns]]
        piv.columns = [PERIOD_LABEL.get(c, c) for c in piv.columns]
        fig4 = px.imshow(piv, text_auto=".4f",
                         color_continuous_scale="RdYlGn",
                         labels={"color": "CPUE (t/trip)"},
                         title="Aggregated CPUE: platform × period",
                         aspect="auto")
        st.plotly_chart(fig4, use_container_width=True, key="pc_324")

    # n_locals per class per year
    with st.expander("No. of active localities by class and year"):
        if df_local is not None and col in df_local.columns:
            fig5 = px.line(
                _cat(df_local, col, order).sort_values(["year", col]),
                x="year", y="n_locals", color=col,
                color_discrete_map=pal,
                category_orders={col: cats},
                markers=True,
                labels={"n_locals": "N localities", "year": "Year", col: "Class"},
                title="Active localities by class and year",
            )
            st.plotly_chart(fig5, use_container_width=True, key="pc_338")


# ─── Tab: Effort ──────────────────────────────────────────────────────────────
def tab_esfuerzo(d: dict) -> None:
    sub_tab = st.radio("View:", ["Fishing gear", "Vessels"], horizontal=True)

    if sub_tab == "Fishing gear":
        gear = d["ts_gear"]
        if gear is None:
            st.warning("Data not available."); return

        top_g = (gear.groupby("gear_type")["gear_production_ton"].sum()
                 .nlargest(12).index.tolist())
        gear_f = gear[gear["gear_type"].isin(top_g)].sort_values("year")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.area(gear_f, x="year", y="share", color="gear_type",
                          labels={"share": "Production share",
                                  "year": "Year", "gear_type": "Gear"},
                          title="Production share by fishing gear (top 12)")
            _period_bands(fig)
            st.plotly_chart(fig, use_container_width=True, key="pc_361")
        with col2:
            fig2 = px.bar(gear_f, x="year", y="gear_production_ton", color="gear_type",
                          barmode="stack",
                          labels={"gear_production_ton": "Production (t)",
                                  "year": "Year", "gear_type": "Gear"},
                          title="Absolute production by gear")
            _period_bands(fig2)
            st.plotly_chart(fig2, use_container_width=True, key="pc_369")

        # By platform class (year slider)
        gp = d["gear_plat"]
        if gp is not None:
            yr = st.slider("Year for breakdown by platform class",
                           int(gp["year"].min()), int(gp["year"].max()),
                           int(gp["year"].max()), key="gear_yr")
            gp_y = gp[gp["year"] == yr].copy()
            top_8 = gp_y.groupby("gear_type")["gear_production_ton"].sum().nlargest(8).index
            gp_y  = _cat(gp_y[gp_y["gear_type"].isin(top_8)],
                         "platform_exposure_class", PLATFORM_ORDER)
            fig3 = px.bar(
                gp_y.sort_values("platform_exposure_class"),
                x="platform_exposure_class", y="gear_share",
                color="gear_type", barmode="stack",
                category_orders={"platform_exposure_class": PLATFORM_ORDER},
                labels={"gear_share": "Share", "platform_exposure_class": "Platform class",
                        "gear_type": "Gear"},
                title=f"Gear composition by platform class — {yr}",
            )
            st.plotly_chart(fig3, use_container_width=True, key="pc_390")

    else:  # Vessels
        boat = d["ts_boat"]
        if boat is None:
            st.warning("Data not available."); return

        col1, col2 = st.columns(2)
        with col1:
            fig = px.area(boat.sort_values("year"), x="year", y="share", color="propulsion",
                          labels={"share": "Fleet share",
                                  "year": "Year", "propulsion": "Propulsion"},
                          title="Fleet composition by propulsion type")
            _period_bands(fig)
            st.plotly_chart(fig, use_container_width=True, key="pc_404")
        with col2:
            fig2 = px.bar(boat.sort_values("year"), x="year", y="vessels_monitored",
                          color="propulsion", barmode="stack",
                          labels={"vessels_monitored": "Monitored vessels",
                                  "year": "Year", "propulsion": "Propulsion"},
                          title="Monitored fleet by propulsion")
            _period_bands(fig2)
            st.plotly_chart(fig2, use_container_width=True, key="pc_412")

        soc = d["socioeconomic"]
        if soc is not None:
            fpv_reg = soc.groupby("year", as_index=False)["fishermen_per_vessel"].mean()
            fig3 = px.line(fpv_reg, x="year", y="fishermen_per_vessel", markers=True,
                           labels={"fishermen_per_vessel": "Fishers/vessel", "year": "Year"},
                           title="Fishers per vessel (regional mean)",
                           color_discrete_sequence=["#7b2d8b"])
            _period_bands(fig3)
            st.plotly_chart(fig3, use_container_width=True, key="pc_422")


# ─── Tab: Species ─────────────────────────────────────────────────────────────
def tab_especies(d: dict) -> None:
    sub = st.radio("Analysis:",
                   ["Top species", "Ranking by exposure class",
                    "SIMPER", "Bray-Curtis turnover"],
                   horizontal=True)

    if sub == "Top species":
        sp20 = d["ts_species_top20"]
        if sp20 is None:
            st.warning("Data not available."); return
        n_sp = st.slider("Show top N species", 5, 20, 12, key="n_sp")
        top  = (sp20.groupby("species")["sp_production_ton"].sum()
                .nlargest(n_sp).index.tolist())
        sp_f = sp20[sp20["species"].isin(top)].sort_values("year")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.area(sp_f, x="year", y="species_share", color="species",
                          labels={"species_share": "Share", "year": "Year",
                                  "species": "Species"},
                          title=f"Share of top-{n_sp} species by year")
            _period_bands(fig)
            st.plotly_chart(fig, use_container_width=True, key="pc_447")
        with col2:
            fig2 = px.bar(sp_f, x="year", y="sp_production_ton", color="species",
                          barmode="stack",
                          labels={"sp_production_ton": "Production (t)",
                                  "year": "Year", "species": "Species"},
                          title=f"Production of top-{n_sp} species by year")
            _period_bands(fig2)
            st.plotly_chart(fig2, use_container_width=True, key="pc_455")

        # Year-total by species
        with st.expander("Species × year table"):
            sp_yr = d["sp_year"]
            if sp_yr is not None:
                pivot = sp_yr[sp_yr["species"].isin(top)].pivot_table(
                    index="species", columns="year", values="sp_production_ton",
                    aggfunc="sum", fill_value=0,
                )
                st.dataframe(pivot.style.background_gradient(cmap="Blues", axis=1),
                             use_container_width=True)

    elif sub == "Ranking by exposure class":
        view = st.radio("Exposure class:", ["Platform", "MPA"], horizontal=True, key="sp_rank_view")
        if view == "Platform":
            sr, col, pal, order = d["species_rank_plat"], "platform_exposure_class", PALETTE_PLATFORM, PLATFORM_ORDER
        else:
            sr, col, pal, order = d["species_rank_mpa"], "mpa_exposure_class", PALETTE_MPA, MPA_ORDER

        if sr is None:
            st.warning("Data not available."); return

        n_top = st.slider("Top N", 5, 20, 10, key="sp_rank_n")
        sr_f  = _cat(sr[sr["rank"] <= n_top].copy(), col, order)
        order_present = [o for o in order if o in sr_f[col].unique()]

        fig = px.bar(
            sr_f.sort_values([col, "rank"]),
            x="mean_species_share", y="species",
            color=col, color_discrete_map=pal,
            facet_col=col, facet_col_wrap=3,
            orientation="h",
            category_orders={col: order_present},
            labels={"mean_species_share": "Mean share", "species": "Species", col: "Class"},
            title=f"Top {n_top} species by {col.replace('_', ' ')}",
            height=500,
        )
        fig.update_layout(showlegend=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_yaxes(matches=None, showticklabels=True)
        st.plotly_chart(fig, use_container_width=True, key="pc_494")

    elif sub == "SIMPER":
        view = st.radio("SIMPER based on:", ["Platform", "MPA"], horizontal=True, key="simper_v")
        simper = d["simper_plat"] if view == "Platform" else d["simper_mpa"]
        if simper is None:
            st.warning("Data not available."); return

        pairs = (simper[["group_a", "group_b"]].drop_duplicates()
                 .assign(label=lambda x: x["group_a"] + "  vs  " + x["group_b"]))
        sel = st.selectbox("Group pair:", pairs["label"].tolist(), key="simper_sel")
        if sel:
            row  = pairs[pairs["label"] == sel].iloc[0]
            mask = (simper["group_a"] == row["group_a"]) & (simper["group_b"] == row["group_b"])
            sub_s = simper[mask].copy()
            if "cumulative_contribution" in sub_s.columns:
                sub_s = sub_s[sub_s["cumulative_contribution"] <= 0.75].head(15)
            else:
                sub_s = sub_s.nlargest(15, "contribution")

            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(
                    sub_s.sort_values("contribution"),
                    x="contribution", y="species", orientation="h",
                    labels={"contribution": "SIMPER contribution", "species": "Species"},
                    title=f"Differences between: {sel}",
                    color="contribution", color_continuous_scale="RdYlGn_r",
                )
                st.plotly_chart(fig, use_container_width=True, key="pc_523")
            with col2:
                st.dataframe(
                    sub_s[["species", "contribution",
                            "mean_abundance_a", "mean_abundance_b"]].reset_index(drop=True),
                    use_container_width=True, hide_index=True,
                )

    else:  # Bray-Curtis
        turn = d["turnover"]
        if turn is None:
            st.warning("Data not available."); return
        exp = d["exposure"]
        if exp is not None:
            turn = turn.merge(
                exp[["local", "platform_exposure_class", "mpa_exposure_class"]],
                on="local", how="left",
            )
        turn["Transition"] = turn["period_a"].astype(str) + " → " + turn["period_b"].astype(str)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                turn, x="Transition", y="bray_curtis",
                color="platform_exposure_class",
                color_discrete_map=PALETTE_PLATFORM,
                category_orders={"platform_exposure_class": PLATFORM_ORDER},
                points="all", labels={"bray_curtis": "Bray-Curtis",
                                       "platform_exposure_class": "Platform class"},
                title="Turnover by platform class",
            )
            st.plotly_chart(fig, use_container_width=True, key="pc_554")
        with col2:
            if "mpa_exposure_class" in turn.columns:
                fig2 = px.box(
                    turn, x="Transition", y="bray_curtis",
                    color="mpa_exposure_class",
                    color_discrete_map=PALETTE_MPA,
                    category_orders={"mpa_exposure_class": MPA_ORDER},
                    points="all", labels={"bray_curtis": "Bray-Curtis",
                                           "mpa_exposure_class": "MPA class"},
                    title="Turnover by MPA class",
                )
                st.plotly_chart(fig2, use_container_width=True, key="pc_566")


# ─── Tab: By Locality ─────────────────────────────────────────────────────────
def tab_localidades(d: dict) -> None:
    exp  = d["exposure"]
    ploy = d["prod_local_year"]
    ssl  = d["sp_share_local"]
    gsl  = d.get("gear_plat")      # fallback to None if absent

    if exp is None or ploy is None:
        st.warning("Data not available."); return

    # Load gear by local
    gear_loc = _load("gear_share_local_year.csv", "processed")

    sel = st.selectbox("Locality:", sorted(exp["local"].unique()), key="local_sel")

    exp_r = exp[exp["local"] == sel].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Platform class",      str(exp_r.get("platform_exposure_class", "–")))
    c2.metric("Platform dist. (km)", f"{exp_r.get('platform_dist_km_mean', float('nan')):.1f}")
    c3.metric("MPA class",           str(exp_r.get("mpa_exposure_class", "–")))
    c4.metric("MPA dist. (km)",      f"{exp_r.get('mpa_dist_km_mean', float('nan')):.1f}")
    c5.metric("Inside MPA",          "Yes" if exp_r.get("inside_any_mpa_any") else "No")

    if str(exp_r.get("coord_source", "exact")) == "municipality_centroid":
        st.info("ℹ Coordinates estimated from municipality centroid — no own landing points.")

    st.divider()

    ly_l = ploy[ploy["local"] == sel].sort_values("year")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(ly_l, x="year", y="production_ton",
                     labels={"production_ton": "Production (t)", "year": "Year"},
                     title=f"Production — {sel}",
                     color_discrete_sequence=["#2171b5"])
        _period_bands(fig)
        st.plotly_chart(fig, use_container_width=True, key="pc_607")
    with col2:
        fig2 = px.line(ly_l, x="year", y="cpue_ton_per_trip", markers=True,
                       labels={"cpue_ton_per_trip": "CPUE (t/trip)", "year": "Year"},
                       title=f"CPUE — {sel}",
                       color_discrete_sequence=["#e6550d"])
        _period_bands(fig2)
        st.plotly_chart(fig2, use_container_width=True, key="pc_614")

    # Top species for this local
    if ssl is not None:
        sp_l = ssl[ssl["local"] == sel].copy()
        if not sp_l.empty:
            top_sp = (sp_l.groupby("species")["sp_production_ton"].sum()
                      .nlargest(10).index.tolist())
            sp_top = sp_l[sp_l["species"].isin(top_sp)].sort_values("year")
            col3, col4 = st.columns(2)
            with col3:
                fig3 = px.area(sp_top, x="year", y="species_share", color="species",
                               labels={"species_share": "Share", "year": "Year",
                                       "species": "Species"},
                               title=f"Catch composition — {sel} (top 10)")
                st.plotly_chart(fig3, use_container_width=True, key="pc_629")
            with col4:
                sp_total = (sp_l.groupby("species")["sp_production_ton"]
                            .sum().nlargest(10).reset_index()
                            .rename(columns={"sp_production_ton": "Total production (t)"}))
                fig4 = px.bar(sp_total.sort_values("Total production (t)"),
                              x="Total production (t)", y="species", orientation="h",
                              title=f"Top 10 species (total) — {sel}",
                              color_discrete_sequence=["#238b45"])
                st.plotly_chart(fig4, use_container_width=True, key="pc_638")

    # Gear composition for this local
    if gear_loc is not None:
        gl = gear_loc[gear_loc["local"] == sel].copy()
        if not gl.empty:
            top_gear = (gl.groupby("gear_type")["gear_production_ton"].sum()
                        .nlargest(8).index.tolist())
            gl_top = gl[gl["gear_type"].isin(top_gear)].sort_values("year")
            fig5 = px.area(gl_top, x="year", y="gear_share", color="gear_type",
                           labels={"gear_share": "Share", "year": "Year", "gear_type": "Gear"},
                           title=f"Fishing gear — {sel}")
            _period_bands(fig5)
            st.plotly_chart(fig5, use_container_width=True, key="pc_651")

    with st.expander("Locality × year table"):
        st.dataframe(ly_l.reset_index(drop=True), use_container_width=True, hide_index=True)


# ─── Tab: Spatial Map ─────────────────────────────────────────────────────────
def tab_mapa(d: dict, spatial: dict) -> None:
    lp_exp = d["lp_exposure"]
    exp    = d["exposure"]
    ssl    = d["sp_share_local"]
    ploy   = d["prod_local_year"]

    if lp_exp is None:
        st.warning("landing_points_exposure.csv not found."); return

    # Build local-level coordinate table
    coords = (
        lp_exp.dropna(subset=["latitude", "longitude"])
        .groupby("local", as_index=False)
        .agg(lat=("latitude", "mean"), lon=("longitude", "mean"))
    )
    if exp is not None:
        coords = coords.merge(exp, on="local", how="left")
    if ploy is not None:
        totals = ploy.groupby("local", as_index=False).agg(
            total_prod=("production_ton", "sum"),
            mean_cpue=("cpue_ton_per_trip", "mean"),
        )
        coords = coords.merge(totals, on="local", how="left")

    # Top-3 species per local
    top3: dict[str, str] = {}
    if ssl is not None:
        t3 = (ssl.groupby(["local", "species"])["sp_production_ton"].sum()
              .reset_index().sort_values(["local", "sp_production_ton"],
                                         ascending=[True, False])
              .groupby("local").head(3))
        top3 = {loc: ", ".join(grp["species"].tolist())
                for loc, grp in t3.groupby("local")}

    # Controls
    col_ctrl1, col_ctrl2 = st.columns([2, 2])
    with col_ctrl1:
        color_by = st.radio("Color by:", ["MPA class", "Platform class"], horizontal=True)
    with col_ctrl2:
        show_platforms = st.checkbox("Show platforms", value=True)
        show_mpas      = st.checkbox("Show MPAs", value=True)

    pal     = PALETTE_MPA      if color_by == "MPA class" else PALETTE_PLATFORM
    cls_col = "mpa_exposure_class" if color_by == "MPA class" else "platform_exposure_class"
    max_p   = float(coords["total_prod"].max()) if "total_prod" in coords.columns else 1.0

    # Build folium map
    m = folium.Map(location=[-5.2, -36.5], zoom_start=7, tiles="CartoDB positron")

    # Municipality layer
    if "municipalities" in spatial:
        folium.GeoJson(
            spatial["municipalities"].__geo_interface__,
            name="Municipalities",
            style_function=lambda _: {
                "fillColor": "transparent", "color": "#999999",
                "weight": 0.6, "fillOpacity": 0,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NM_MUN"], aliases=["Municipality:"], sticky=False
            ),
        ).add_to(m)

    # MPA polygons
    if show_mpas and "mpas" in spatial:
        mpa_fill = {"RDS Ponta do Tubarão": "#1a9850", "APA Dunas do Rosado": "#66bd63"}
        for _, row in spatial["mpas"].iterrows():
            name  = row.get("mpa_name", "MPA")
            color = mpa_fill.get(name, "#2ca25f")
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda _, c=color: {
                    "fillColor": c, "color": c, "weight": 1.5, "fillOpacity": 0.3
                },
                tooltip=name,
                name=name,
            ).add_to(m)

    # Platform markers
    if show_platforms and "platforms" in spatial:
        plat_group = folium.FeatureGroup(name="Platforms").add_to(m)
        for _, row in spatial["platforms"].iterrows():
            lat  = row.geometry.y
            lon  = row.geometry.x
            name = next((row.get(c) for c in ["instalacao", "Nome", "name", "NAME", "NOME"]
                         if pd.notna(row.get(c, None))), "Platform")
            folium.RegularPolygonMarker(
                location=[lat, lon],
                number_of_sides=3, radius=8,
                color="#cc0000", fill=True, fill_color="#cc0000", fill_opacity=0.9,
                tooltip=f"🛢 {name}",
            ).add_to(plat_group)

    # Local markers (sized by production)
    local_group = folium.FeatureGroup(name="Localities").add_to(m)
    for _, row in coords.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue
        cls    = row.get(cls_col, None)
        color  = pal.get(str(cls), "#888888") if cls and str(cls) != "nan" else "#888888"
        prod   = float(row.get("total_prod", 0) or 0)
        radius = 6 + 18 * (prod / max_p) if max_p > 0 else 8
        cpue   = row.get("mean_cpue", float("nan"))
        src    = row.get("coord_source", "exact")
        top3s  = top3.get(row["local"], "–")

        popup_html = f"""
        <div style="font-family:sans-serif;font-size:13px;min-width:210px">
          <b style="font-size:15px">{row['local']}</b><br><hr style="margin:4px 0">
          <b>Platform class:</b> {row.get('platform_exposure_class','–')}<br>
          <b>MPA class:</b> {row.get('mpa_exposure_class','–')}<br>
          <b>Inside MPA:</b> {'✔' if row.get('inside_any_mpa_any') else '✗'}<br>
          <b>Total production:</b> {prod:,.0f} t<br>
          <b>Mean CPUE:</b> {cpue:.4f} t/trip<br>
          <b>Top 3 species:</b> {top3s}<br>
          <small style="color:#999">Coord: {src}</small>
        </div>
        """
        dash = "5,5" if src == "municipality_centroid" else None
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True, fill_color=color, fill_opacity=0.75,
            weight=2 if src == "municipality_centroid" else 1,
            dash_array=dash,
            tooltip=row["local"],
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(local_group)

    folium.LayerControl(collapsed=False).add_to(m)

    # HTML legend
    items = "".join(
        f'<div style="margin:2px 0"><span style="background:{c};display:inline-block;'
        f'width:11px;height:11px;border-radius:50%;margin-right:5px"></span>{k}</div>'
        for k, c in pal.items()
    )
    items += ('<div style="margin:2px 0"><span style="background:#888;display:inline-block;'
              'width:11px;height:11px;border-radius:50%;margin-right:5px"></span>no class</div>')
    legend = (
        '<div style="position:fixed;bottom:28px;left:28px;background:white;'
        'padding:10px 14px;border:1px solid #ccc;border-radius:7px;'
        'font-family:sans-serif;font-size:12px;z-index:9999">'
        f'<b>{"MPA class" if color_by=="MPA class" else "Platform class"}</b><br>'
        f'{items}</div>'
    )
    m.get_root().html.add_child(folium.Element(legend))

    st_folium(m, width="100%", height=650, returned_objects=[])

    st.caption(
        "Circles: size ∝ total production. "
        "Dashed border = coordinate estimated from municipality centroid. "
        "Click on any marker to see details."
    )

    # Summary table below map
    with st.expander("Localities table with spatial exposure"):
        cols_show = ["local", "platform_exposure_class", "platform_dist_km_mean",
                     "mpa_exposure_class", "mpa_dist_km_mean", "inside_any_mpa_any",
                     "coord_source"]
        cols_show = [c for c in cols_show if c in coords.columns]
        st.dataframe(
            coords[cols_show].sort_values("platform_dist_km_mean")
            .reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Fish × Platforms — RN Brazil",
        page_icon="🎣",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("🎣  Fish catches × Oil platforms — Rio Grande do Norte, Brazil")
    st.caption(
        "Small-scale fisheries | Offshore platforms | Marine Protected Areas "
        "| 22 localities | 2001–2024"
    )

    with st.spinner("Loading data and spatial layers…"):
        d       = load_all()
        spatial = load_spatial()

    tabs = st.tabs([
        "🏠 Summary",
        "📈 CPUE & Productivity",
        "⚓ Effort",
        "🐟 Species Composition",
        "📍 By Locality",
        "🗺️ Spatial Map",
    ])

    with tabs[0]: tab_resumen(d)
    with tabs[1]: tab_cpue(d)
    with tabs[2]: tab_esfuerzo(d)
    with tabs[3]: tab_especies(d)
    with tabs[4]: tab_localidades(d)
    with tabs[5]: tab_mapa(d, spatial)


if __name__ == "__main__":
    main()
