import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# =============================
# 1. Cargar dataset unificado
# =============================
df = pd.read_csv("03_MASTER_DATASET_ANALISIS_IMPACTO.csv")

# =============================
# 2. Definir targets y features
# =============================
target_ira = "ira_no_neumonia"
target_pneum = "neumonias_men5"

features = [
    "PRODUCCION",
    "POBLACION",
    "ANIO",
    "MES",
    "PROVINCIA_NORM",
    "DISTRITO_NORM",
]

X = df[features]
y_ira = df[target_ira]
y_pneum = df[target_pneum]

# Eliminar filas con target faltante
valid_idx_ira = y_ira.dropna().index
X_ira = X.loc[valid_idx_ira]
y_ira = y_ira.loc[valid_idx_ira]

valid_idx_pneum = y_pneum.dropna().index
X_pneum = X.loc[valid_idx_pneum]
y_pneum = y_pneum.loc[valid_idx_pneum]

# =============================
# 3. Preprocesamiento
# =============================
categorical = ["PROVINCIA_NORM", "DISTRITO_NORM"]
numeric = ["PRODUCCION", "POBLACION", "ANIO", "MES"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ]
)

# =============================
# 4. Modelos XGBoost
# =============================
model_ira = Pipeline(
    steps=[
        ("preprocessor", preprocess),
        ("model", XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
        )),
    ]
)

model_pneum = Pipeline(
    steps=[
        ("preprocessor", preprocess),
        ("model", XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
        )),
    ]
)

# =============================
# 5. Entrenar ambos modelos
# =============================
X_train_ira, X_test_ira, y_train_ira, y_test_ira = train_test_split(
    X_ira, y_ira, test_size=0.2, shuffle=True
)

X_train_pneum, X_test_pneum, y_train_pneum, y_test_pneum = train_test_split(
    X_pneum, y_pneum, test_size=0.2, shuffle=True
)

model_ira.fit(X_train_ira, y_train_ira)
model_pneum.fit(X_train_pneum, y_train_pneum)

# =============================
# 6. Evaluación
# =============================
pred_ira = model_ira.predict(X_test_ira)
pred_pneum = model_pneum.predict(X_test_pneum)

mae_ira = mean_absolute_error(y_test_ira, pred_ira)
r2_ira = r2_score(y_test_ira, pred_ira)

mae_pneum = mean_absolute_error(y_test_pneum, pred_pneum)
r2_pneum = r2_score(y_test_pneum, pred_pneum)

# =============================
# 7. Guardar modelos
# =============================
joblib.dump(model_ira, "modelo_IRA_no_neumonia.pkl")
joblib.dump(model_pneum, "modelo_neumonias.pkl")

# =============================
# 8. Generar salida y guardar en .txt
# =============================
lines = []
lines.append(f"DataFrame Columns: {df.columns.tolist()}")
lines.append("----- IRA NO NEUMONÍA -----")
lines.append(f"MAE: {mae_ira}")
lines.append(f"R²: {r2_ira}")
lines.append("")
lines.append("----- NEUMONÍAS -----")
lines.append(f"MAE: {mae_pneum}")
lines.append(f"R²: {r2_pneum}")
lines.append(str(["modelo_neumonias.pkl"]))

output_text = "\n".join(lines)

print(output_text)

with open("ModelTrainingMetrics_IRA_Pneumonia.txt", "w", encoding="utf-8") as f:
    f.write(output_text + "\n")
