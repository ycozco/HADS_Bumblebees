import pandas as pd
import numpy as np
import joblib

# =======================================
# 1. Cargar dataset y modelo entrenado
# =======================================
df = pd.read_csv("03_MASTER_DATASET_ANALISIS_IMPACTO.csv")
modelo = joblib.load("modelo_total_casos.pkl")

# =======================================
# 2. Distritos y meses de referencia
# =======================================
sitios = [
    ("CERRO COLORADO", 2),
    ("TIABAYA", 5),
    ("COCACHACRA", 8),
    ("ANDARAY", 10),
    ("PUNTA DE BOMBON", 9),
    ("LA JOYA", 1),
    ("CERRO COLORADO", 9),
]

ANIOS_FUTURO = [2024, 2025]

# =======================================
# 3. Construcción del set futuro
# =======================================
data_pred = []

for distrito, mes in sitios:
    prov = df[df["DISTRITO_NORM"] == distrito]["PROVINCIA_NORM"].iloc[0]
    poblacion = df[df["DISTRITO_NORM"] == distrito]["POBLACION"].iloc[-1]
    produccion = df[df["DISTRITO_NORM"] == distrito]["PRODUCCION"].mean()

    for anio in ANIOS_FUTURO:
        data_pred.append({
            "PROVINCIA_NORM": prov,
            "DISTRITO_NORM": distrito,
            "ANIO": anio,
            "MES": mes,
            "POBLACION": poblacion,
            "PRODUCCION": produccion
        })

df_futuro = pd.DataFrame(data_pred)

# Variables del modelo
features = ["PRODUCCION", "POBLACION", "ANIO", "MES",
            "PROVINCIA_NORM", "DISTRITO_NORM"]

# ==============================
# 4. Predicción
# ==============================
pred = modelo.predict(df_futuro[features])
pred = np.maximum(pred, 0)
df_futuro["PRED_TOTAL_CASOS"] = pred

# ==============================
# 5. Construir salida y guardar .txt
# ==============================
lines = []
lines.append("=== Predicciones TOTAL CASOS 2024 y 2025 ===\n")
lines.append(df_futuro.to_string(index=False))

output_text = "\n".join(lines)

print(output_text)

with open("ModelForecast_TotalCases_2024_2025.txt", "w", encoding="utf-8") as f:
    f.write(output_text + "\n")
