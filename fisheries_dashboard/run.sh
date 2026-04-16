#!/bin/bash
# ──────────────────────────────────────────────────
# run.sh — Inicia el dashboard Fisheries GIS
# ──────────────────────────────────────────────────
echo "Instalando dependencias..."
pip install -r requirements.txt -q

echo "Iniciando Streamlit dashboard..."
streamlit run app.py \
  --server.port 8501 \
  --server.headless true \
  --theme.base dark \
  --theme.primaryColor "#2980b9"
