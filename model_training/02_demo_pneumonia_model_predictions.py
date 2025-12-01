import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# 1. Cargar datos y modelo
# =========================
df = pd.read_csv("03_MASTER_DATASET_ANALISIS_IMPACTO.csv")
model_pneum = joblib.load("modelo_neumonias.pkl")

# =========================
# 2. Preparar X e y
# =========================
target_pneum = "neumonias_men5"
features = ["PRODUCCION", "POBLACION", "ANIO", "MES",
            "PROVINCIA_NORM", "DISTRITO_NORM"]

X = df[features]
y = df[target_pneum]

valid_idx = y.dropna().index
X_valid = X.loc[valid_idx]
y_valid = y.loc[valid_idx]

# Split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X_valid, y_valid, test_size=0.2, shuffle=True, random_state=42
)

# =========================
# 3. Predicciones
# =========================
predictions = model_pneum.predict(X_test)

# =========================
# 4. Construir salida y guardar .txt
# =========================
lines = []
lines.append("First 10 predictions for 'neumonias' model on X_test2:")
lines.append(str(predictions[:10]))
lines.append("")
lines.append("First 10 actual values from y_test2:")
lines.append(str(y_test.head(10).values))

output_text = "\n".join(lines)

print(output_text)

with open("ModelDemo_PneumoniaPredictions.txt", "w", encoding="utf-8") as f:
    f.write(output_text + "\n")
