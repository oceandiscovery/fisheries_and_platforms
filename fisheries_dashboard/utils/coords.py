"""
coords.py — Coordenadas geográficas conocidas para los 5 municipios costeros
del estado de Rio Grande do Norte (Brasil).
"""

# Coordenadas (lat, lon) verificadas para cada localidad canónica
PORT_COORDS = {
    "AREIA BRANCA":     {"lat": -4.9529, "lon": -37.1373, "name": "Areia Branca"},
    "CAICARA DO NORTE": {"lat": -5.0697, "lon": -36.0703, "name": "Caiçara do Norte"},
    "GUAMARE":          {"lat": -5.1131, "lon": -36.3266, "name": "Guamaré"},
    "MACAU":            {"lat": -5.1151, "lon": -36.6352, "name": "Macau"},
    "PORTO DO MANGUE":  {"lat": -5.1614, "lon": -36.8428, "name": "Porto do Mangue"},
}

# Metadata adicional para los mapas
PORT_META = {
    "AREIA BRANCA":     {"state": "RN", "country": "Brasil", "region": "Litoral Oeste"},
    "CAICARA DO NORTE": {"state": "RN", "country": "Brasil", "region": "Litoral Leste"},
    "GUAMARE":          {"state": "RN", "country": "Brasil", "region": "Litoral Norte"},
    "MACAU":            {"state": "RN", "country": "Brasil", "region": "Litoral Norte"},
    "PORTO DO MANGUE":  {"state": "RN", "country": "Brasil", "region": "Litoral Norte"},
}

GEAR_LABELS = {
    "APA": "Arrasto de praia",
    "APO": "Arrasto de porta",
    "CVL": "Covo Lagosta",
    "CPL": "Covo Peixe/Lagosta",
    "ESP": "Espinhel",
    "LIJ": "Linha e Jereré",
    "LIN": "Linha",
    "RAG": "Rede Agulha",
    "RAT": "Rede de Arrasto",
    "RCA": "Rede Camarão",
    "RCV": "Rede Corvina",
    "RED": "Rede Diversa",
    "REM": "Rede Emalhe",
    "RGT": "Rede Guaiamum/Tapagem",
    "RPV": "Rede Peroá/Viola",
    "RRO": "Rede Robalo",
    "RSS": "Rede Sertã/Sapeba",
    "RXA": "Rede Xaréu",
    "TAP": "Tapagem",
    "TBO": "Tarrafa Boa",
    "ZAP": "Zangaria/Pindaí",
}

BOAT_LABELS = {
    "BOV": "Bote a Vela",
    "BMG": "Barco Motor Grande",
    "BMM": "Barco Motor Médio",
    "BMP": "Barco Motor Pequeno",
    "CAV": "Canoa a Vela",
    "PQM": "Pequena Motor",
    "JAV": "Jangada a Vela",
    "CAM": "Canoa a Motor",
}
