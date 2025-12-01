import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ================================================
# 1. Cargar dataset unificado
# ================================================
df = pd.read_csv("03_MASTER_DATASET_ANALISIS_IMPACTO.csv")

# ================================================
# 2. Elegir variables del modelo
# ================================================
target = "TOTAL_CASOS"

features = [
    "PRODUCCION",
    "POBLACION",
    "ANIO",
    "MES",
    "PROVINCIA_NORM",
    "DISTRITO_NORM"
]

X = df[features]
y = df[target]

valid_idx = y.dropna().index
X_clean = X.loc[valid_idx]
y_clean = y.loc[valid_idx]

# ================================================
# 3. Preprocesamiento
# ================================================
categorical = ["PROVINCIA_NORM", "DISTRITO_NORM"]
numeric = ["PRODUCCION", "POBLACION", "ANIO", "MES"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ]
)

# ================================================
# 4. Crear modelo XGBoost
# ================================================
model_total = Pipeline(steps=[
    ("preprocessor", preprocess),
    ("model", XGBRegressor(
        n_estimators=350,
        learning_rate=0.07,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85
    ))
])

# ================================================
# 5. Entrenamiento
# ================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, shuffle=True
)

model_total.fit(X_train, y_train)

# ================================================
# 6. Evaluación
# ================================================
pred = model_total.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

# ================================================
# 7. Guardar modelo
# ================================================
joblib.dump(model_total, "modelo_total_casos.pkl")

# ================================================
# 8. Construir salida y guardar .txt
# ================================================
lines = []
lines.append(str(df.columns.tolist()))
lines.append("===== MODELO TOTAL CASOS =====")
lines.append(f"MAE: {mae}")
lines.append(f"R²: {r2}")
lines.append("Modelo guardado como modelo_total_casos.pkl")

output_text = "\n".join(lines)

print(output_text)

with open("ModelTrainingMetrics_TotalCases.txt", "w", encoding="utf-8") as f:
    f.write(output_text + "\n")
