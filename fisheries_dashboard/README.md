# 🐟 Fisheries GIS Dashboard
**Análisis espacial de capturas, CPUE, biodiversidad y valor económico**  
*Litoral do Rio Grande do Norte, Brasil — PMDP/IBAMA*

---

## Descripción

Dashboard interactivo para integrar bases de datos de capturas pesqueras con capas GIS,
generando mapas de distribución espacial de especies, CPUE por arte de pesca y puerto,
y hotspots de biodiversidad con metadatos completos.

**Puertos analizados:** Areia Branca · Caiçara do Norte · Guamaré · Macau · Porto do Mangue  
**Período:** 2001–2022 (según disponibilidad)  
**Artes de pesca:** 29 artes canónicas (activas, pasivas, mixtas)  
**Especies:** 59 especies registradas

---

## Estructura del proyecto

```
fisheries_dashboard/
├── app.py                    ← Dashboard principal Streamlit
├── requirements.txt          ← Dependencias Python
├── run.sh                    ← Script de inicio
├── README.md
├── utils/
│   ├── coords.py             ← Coordenadas y metadatos de puertos
│   ├── data_pipeline.py      ← ETL, CPUE, biodiversidad, correlaciones
│   └── map_builder.py        ← Mapas Folium interactivos
└── outputs/
    ├── geojson/              ← Capas para QGIS (generadas al ejecutar)
    │   ├── ports_indicators.geojson
    │   ├── cpue_by_gear.geojson
    │   └── master_timeseries.csv
    └── maps/                 ← Mapas HTML standalone
        ├── species_distribution.html
        ├── cpue_by_gear.html
        └── biodiversity_hotspots.html
```

> ⚠️ Los archivos `.parquet` de entrada deben estar en el directorio padre (`../`).

---

## Instalación y ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Lanzar el dashboard
streamlit run app.py

# O usar el script de inicio:
bash run.sh
```

Abre en el navegador: **http://localhost:8501**

---

## Métricas calculadas

### CPUE (Captura Por Unidad de Esfuerzo)
```
CPUE = producción (ton) / viajes asistidos
```
Calculado por (puerto × año × arte de pesca).

### Índices de biodiversidad
| Índice | Fórmula | Descripción |
|--------|---------|-------------|
| **Shannon H'** | −Σ(pᵢ · ln pᵢ) | Diversidad general |
| **Riqueza S** | count(spp) | Número de especies |
| **Pielou J'** | H' / ln(S) | Equitatividad |

### Análisis estadísticos
- **Pearson r** y **Spearman ρ** entre todas las variables clave
- **Regresión lineal** CPUE ~ Pescadores estimados
- **ANOVA de un factor** CPUE entre grupos de artes de pesca

---

## Capas GIS para QGIS

Los archivos GeoJSON exportados son importables directamente en QGIS:

1. `ports_indicators.geojson` — Indicadores medios por puerto (puntos)
2. `cpue_by_gear.geojson` — CPUE por arte de pesca (puntos)
3. `master_timeseries.csv` — Serie temporal completa (tabla atributos)

**Importar en QGIS:**  
`Capa → Añadir capa → Añadir capa vectorial → seleccionar .geojson`

---

## Tabs del dashboard

| Tab | Contenido |
|-----|-----------|
| 📊 Resumen general | KPIs, producción anual, CPUE temporal, pescadores |
| 🗺️ Mapas interactivos | Distribución especies, CPUE, hotspots biodiversidad |
| 🐠 Análisis de especies | Top especies, heatmap especie×puerto, evolución temporal |
| ⚓ Artes de pesca | Producción y CPUE por arte, boxplot, heatmap |
| 📈 Correlaciones | Pearson, Spearman, scatter plots, regresión, ANOVA |
| 💰 Valor económico | Valor de producción (BRL) por municipio y semestre |

---

## Fuentes de datos

- **PMDP** (Programa de Monitoramento da Pesca Desembarcada)
- **IBAMA** — Instituto Brasileiro do Meio Ambiente e dos Recursos Naturais Renováveis
- Localidades: RN Brasil — Litoral Norte y Oeste

---

*Andres Ospina · CSIC · 2026*
