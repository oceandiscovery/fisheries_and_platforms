"""
map_builder.py — Interactive Folium map construction:
  1. Spatial species distribution map
  2. CPUE map by fishing gear and port
  3. Biodiversity hotspots map
"""

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap, Fullscreen
import branca.colormap as cm
from utils.coords import PORT_COORDS

# Paleta de colores por grupo de arte
GEAR_COLORS = {
    "active":  "#e74c3c",
    "passive": "#2980b9",
    "mixed":   "#8e44ad",
}


def _base_map(center=(-5.1, -36.6), zoom=9):
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Satellite",
        attr="Esri",
    ).add_to(m)
    Fullscreen(position="topright").add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    return m


# ─────────────────────────────────────────────
# 1. DISTRIBUCIÓN ESPACIAL DE ESPECIES
# ─────────────────────────────────────────────
def species_distribution_map(dfs, year_range=None):
    """
    Map with circles proportional to species catch per port.
    Uses MarkerCluster grouped by species.
    """
    sp = dfs["species"].copy()
    if year_range:
        sp = sp[sp["year"].between(*year_range)]

    sp_agg = sp.groupby(["local_norm", "species"])["sp_production_ton"].sum().reset_index()
    sp_agg["lat"] = sp_agg["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lat"))
    sp_agg["lon"] = sp_agg["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lon"))
    sp_agg = sp_agg.dropna(subset=["lat", "lon"])

    m = _base_map()

    # Heatmap de capturas totales
    heat_data = [
        [row["lat"] + np.random.uniform(-0.02, 0.02),
         row["lon"] + np.random.uniform(-0.02, 0.02),
         row["sp_production_ton"]]
        for _, row in sp_agg.iterrows()
    ]
    HeatMap(
        heat_data,
        name="Catch intensity",
        min_opacity=0.3,
        max_zoom=14,
        radius=25,
        blur=15,
        gradient={0.2: "#ffffcc", 0.5: "#fd8d3c", 0.8: "#e31a1c"},
    ).add_to(m)

    # Círculos por puerto con popup detallado
    top_species = sp_agg.groupby("local_norm")["sp_production_ton"].sum().reset_index()
    top_species.columns = ["local_norm", "total_ton"]
    max_ton = top_species["total_ton"].max()

    for _, row in top_species.iterrows():
        local = row["local_norm"]
        port_sp = sp_agg[sp_agg["local_norm"] == local].sort_values("sp_production_ton", ascending=False)
        top5 = port_sp.head(5)
        table_rows = "".join([
            f"<tr><td>{r['species']}</td><td>{r['sp_production_ton']:.1f} t</td></tr>"
            for _, r in top5.iterrows()
        ])
        popup_html = f"""
        <div style='font-family:Arial;min-width:220px'>
          <h4 style='color:#2c3e50;margin-bottom:4px'>{PORT_COORDS[local]['name']}</h4>
          <p><b>Total catch:</b> {row['total_ton']:.1f} t<br>
          <b>No. species:</b> {len(port_sp)}</p>
          <table style='width:100%;border-collapse:collapse;font-size:12px'>
            <tr style='background:#2980b9;color:white'>
              <th>Species</th><th>Catch</th>
            </tr>
            {table_rows}
          </table>
        </div>"""
        radius = 8 + 30 * (row["total_ton"] / max_ton)
        folium.CircleMarker(
            location=[PORT_COORDS[local]["lat"], PORT_COORDS[local]["lon"]],
            radius=radius,
            color="#2c3e50",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{PORT_COORDS[local]['name']}: {row['total_ton']:.0f} t",
        ).add_to(m)

    # Marcadores de especie top por puerto
    mc = MarkerCluster(name="Species by port", show=False)
    for _, row in sp_agg[sp_agg["sp_production_ton"] > 0].iterrows():
        folium.CircleMarker(
            location=[row["lat"] + np.random.uniform(-0.01, 0.01),
                      row["lon"] + np.random.uniform(-0.01, 0.01)],
            radius=4,
            color="#e74c3c",
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.6,
            tooltip=f"{row['species']}: {row['sp_production_ton']:.2f} t",
        ).add_to(mc)
    mc.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ─────────────────────────────────────────────
# 2. CPUE POR ARTE DE PESCA Y PUERTO
# ─────────────────────────────────────────────
def cpue_map(cpue_gear, master, year_range=None):
    """
    Choropleth map of CPUE by port with colored bubbles
    by dominant fishing gear and mean CPUE.
    """
    cg = cpue_gear.copy()
    if year_range:
        cg = cg[cg["year"].between(*year_range)]

    # CPUE media por puerto (todos los años seleccionados)
    port_cpue = cg.groupby("local_norm").agg(
        cpue_mean=("cpue", "mean"),
        production_sum=("production_ton", "sum"),
    ).reset_index()

    port_cpue["lat"] = port_cpue["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lat"))
    port_cpue["lon"] = port_cpue["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lon"))

    # Arte dominante por puerto
    dominant_gear = (
        cg.groupby(["local_norm", "gear_type"])["production_ton"].sum()
        .reset_index()
        .sort_values("production_ton", ascending=False)
        .groupby("local_norm").first()
        .reset_index()[["local_norm", "gear_type"]]
    )
    port_cpue = port_cpue.merge(dominant_gear, on="local_norm", how="left")

    m = _base_map()

    # Colormap CPUE
    max_cpue = port_cpue["cpue_mean"].max()
    colormap = cm.LinearColormap(
        ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
        vmin=0, vmax=max_cpue,
        caption="Mean CPUE (t/trip)",
    )
    colormap.add_to(m)

    # CPUE por arte — layer separado por grupo
    for group, grp_df in cg.groupby("gear_group"):
        fg = folium.FeatureGroup(name=f"Gear {group}", show=True)
        gear_port = grp_df.groupby(["local_norm", "gear_type"]).agg(
            cpue=("cpue", "mean"), prod=("production_ton", "sum")
        ).reset_index()
        gear_port["gear_group"] = group
        gear_port["lat"] = gear_port["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lat"))
        gear_port["lon"] = gear_port["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lon"))
        gear_port = gear_port.dropna(subset=["lat", "lon"])
        color = GEAR_COLORS.get(group, "#7f8c8d")

        for _, row in gear_port.iterrows():
            popup_html = f"""
            <div style='font-family:Arial'>
              <b>{PORT_COORDS[row['local_norm']]['name']}</b><br>
              Gear: <b>{row['gear_type']}</b><br>
              Group: {row['gear_group']}<br>
              CPUE: <b>{row['cpue']:.4f}</b> t/trip<br>
              Total production: {row['prod']:.1f} t
            </div>"""
            folium.CircleMarker(
                location=[row["lat"] + np.random.uniform(-0.015, 0.015),
                          row["lon"] + np.random.uniform(-0.015, 0.015)],
                radius=5 + 15 * (row["cpue"] / (max_cpue + 1e-9)),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{row['gear_type']}: CPUE={row['cpue']:.4f}",
            ).add_to(fg)
        fg.add_to(m)

    # Marcadores de puerto con CPUE total
    for _, row in port_cpue.dropna(subset=["lat", "lon"]).iterrows():
        color = colormap(row["cpue_mean"])
        popup_html = f"""
        <div style='font-family:Arial;min-width:200px'>
          <h4 style='color:#2c3e50'>{PORT_COORDS[row['local_norm']]['name']}</h4>
          <b>Mean CPUE:</b> {row['cpue_mean']:.4f} ton/viaje<br>
          <b>Dominant gear:</b> {row.get('gear_type','N/A')}<br>
          <b>Cumulative production:</b> {row['production_sum']:.1f} t
        </div>"""
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"CPUE={row['cpue_mean']:.4f}",
            icon=folium.DivIcon(
                html=f"""<div style='background:{color};border:2px solid #2c3e50;
                          border-radius:50%;width:22px;height:22px;
                          display:flex;align-items:center;justify-content:center;
                          font-size:9px;font-weight:bold;color:#2c3e50'>
                         {row['cpue_mean']:.2f}</div>""",
                icon_size=(22, 22),
                icon_anchor=(11, 11),
            ),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ─────────────────────────────────────────────
# 3. HOTSPOTS DE BIODIVERSIDAD
# ─────────────────────────────────────────────
def biodiversity_hotspot_map(biodiv, master, year_range=None):
    """
    Hotspot map with Shannon heatmap and richness bubbles.
    Includes full metadata in popups.
    """
    bio = biodiv.copy()
    if year_range:
        bio = bio[bio["year"].between(*year_range)]

    bio_port = bio.groupby("local_norm").agg(
        shannon_mean=("shannon_index", "mean"),
        richness_mean=("species_richness", "mean"),
        pielou_mean=("pielou_index", "mean"),
        sp_prod_sum=("sp_production_ton", "sum"),
    ).reset_index()

    bio_port["lat"] = bio_port["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lat"))
    bio_port["lon"] = bio_port["local_norm"].map(lambda x: PORT_COORDS.get(x, {}).get("lon"))

    m = _base_map()

    # Heatmap Shannon
    heat_data = [
        [row["lat"], row["lon"], row["shannon_mean"]]
        for _, row in bio_port.dropna(subset=["lat", "lon"]).iterrows()
    ]
    HeatMap(
        heat_data,
        name="Biodiversity heatmap (Shannon)",
        min_opacity=0.4,
        radius=60,
        blur=40,
        max_zoom=12,
        gradient={0.3: "#e0f3f8", 0.6: "#74add1", 0.85: "#313695"},
    ).add_to(m)

    # Colormap Shannon
    max_h = bio_port["shannon_mean"].max()
    colormap = cm.LinearColormap(
        ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"],
        vmin=0, vmax=max_h,
        caption="Shannon-Wiener index (H')",
    )
    colormap.add_to(m)

    # Marcadores con riqueza y metadatos
    max_richness = bio_port["richness_mean"].max()
    for _, row in bio_port.dropna(subset=["lat", "lon"]).iterrows():
        # Clasificación del hotspot
        if row["shannon_mean"] >= 0.75 * max_h:
            badge = "🔴 HIGH HOTSPOT"
            badge_color = "#c0392b"
        elif row["shannon_mean"] >= 0.5 * max_h:
            badge = "🟠 MID HOTSPOT"
            badge_color = "#e67e22"
        else:
            badge = "🟡 LOW BIODIVERSITY"
            badge_color = "#f39c12"

        # Serie temporal de Shannon para este puerto
        ts = biodiv[biodiv["local_norm"] == row["local_norm"]].sort_values("year")
        ts_rows = "".join([
            f"<tr><td>{int(r['year'])}</td><td>{r['shannon_index']:.3f}</td>"
            f"<td>{int(r['species_richness'])}</td></tr>"
            for _, r in ts.iterrows()
        ])
        popup_html = f"""
        <div style='font-family:Arial;min-width:260px;max-height:350px;overflow-y:auto'>
          <h4 style='color:#2c3e50;margin-bottom:4px'>{PORT_COORDS[row['local_norm']]['name']}</h4>
          <span style='background:{badge_color};color:white;padding:2px 6px;
                       border-radius:4px;font-size:11px'>{badge}</span>
          <hr style='margin:6px 0'>
          <b>Shannon H':</b> {row['shannon_mean']:.4f}<br>
          <b>Mean richness:</b> {row['richness_mean']:.1f} spp<br>
          <b>Pielou J':</b> {row['pielou_mean']:.4f}<br>
          <b>Total production:</b> {row['sp_prod_sum']:.1f} t<br>
          <hr style='margin:6px 0'>
          <b>Time series:</b>
          <table style='width:100%;border-collapse:collapse;font-size:11px'>
            <tr style='background:#2980b9;color:white'>
              <th>Year</th><th>H'</th><th>Richness</th>
            </tr>
            {ts_rows}
          </table>
        </div>"""

        radius = 12 + 28 * (row["richness_mean"] / (max_richness + 1e-9))
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=colormap(row["shannon_mean"]),
            weight=3,
            fill=True,
            fill_color=colormap(row["shannon_mean"]),
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=(
                f"{PORT_COORDS[row['local_norm']]['name']} | "
                f"H'={row['shannon_mean']:.3f} | S={row['richness_mean']:.0f} spp"
            ),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
