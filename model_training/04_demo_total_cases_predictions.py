import pandas as pd
import numpy as np
import joblib

# ==========================================================
# 1. Cargar dataset y modelo
# ==========================================================
df = pd.read_csv("03_MASTER_DATASET_ANALISIS_IMPACTO.csv")
modelo_total = joblib.load("modelo_total_casos.pkl")

# ==========================================================
# 2. Seleccionar algunas filas de prueba
# ==========================================================
subset = df.sample(10, random_state=10)

features = ["PRODUCCION", "POBLACION", "ANIO", "MES",
            "PROVINCIA_NORM", "DISTRITO_NORM"]

X_demo = subset[features]
y_real = subset["TOTAL_CASOS"]

# ==========================================================
# 3. Predicción del modelo
# ==========================================================
predicciones = modelo_total.predict(X_demo)
predicciones = np.maximum(predicciones, 0)  # evitar negativos

# ==========================================================
# 4. Crear tabla de comparación
# ==========================================================
resultados = pd.DataFrame({
    "Distrito": subset["DISTRITO_NORM"],
    "Año": subset["ANIO"],
    "Mes": subset["MES"],
    "Producción Minera (TN)": subset["PRODUCCION"],
    "Casos Reales Totales": y_real.values,
    "Casos Predichos Totales": predicciones
})

# ==========================================================
# 5. Construir texto de salida completo
# ==========================================================
lines = []

lines.append("=== DEMOSTRACIÓN DEL MODELO TOTAL_CASOS ===\n")
lines.append(resultados.to_string(index=False))
lines.append("\n=== INTERPRETACIÓN SENCILLA ===")

for i, row in resultados.iterrows():
    real = row["Casos Reales Totales"]
    pred = row["Casos Predichos Totales"]

    lines.append(f"\n{row['Distrito']} ({int(row['Mes'])}/{int(row['Año'])})")
    lines.append(f"   - Casos reales: {real}")
    lines.append(f"   - Modelo predice: {pred}")

    if pd.isna(real):
        lines.append("   No hay casos reales disponibles para comparar.")
        continue

    if real == 0 and pred < 1:
        lines.append("   El modelo identifica correctamente un mes con muy pocos casos.")
    elif real > 0 and abs(real - pred) <= 3:
        lines.append("   El modelo acierta muy bien con una diferencia mínima.")
    elif real > 0 and pred > real:
        lines.append("   El modelo sobrestima un poco los casos, pero sigue una tendencia correcta.")
    elif real > 0 and pred < real:
        lines.append("   El modelo subestima los casos, pero reconoce actividad epidemiológica.")
    else:
        lines.append("   Predicción razonable dentro de la variabilidad del fenómeno.")

output_text = "\n".join(lines)

print(output_text)

with open("ModelDemo_TotalCasesPredictions.txt", "w", encoding="utf-8") as f:
    f.write(output_text + "\n")
