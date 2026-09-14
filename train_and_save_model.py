"""
train_and_save_model.py
=======================
Re-entrena la Regresión Logística con los mismos parámetros del cuaderno
SEMINARIO_2_corregido.ipynb y guarda el pipeline completo como modelo_rl.pkl.

Ejecutar una sola vez:
    python train_and_save_model.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

warnings.filterwarnings("ignore")
SEMILLA = 42

# ── 1. Ruta al CSV ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "application_train.csv")

print(f"Cargando datos desde: {CSV_PATH}")
raw = pd.read_csv(CSV_PATH)
print(f"  -> {raw.shape[0]:,} filas | {raw.shape[1]} columnas")

# ── 2. Selección de columnas (mismo subconjunto del notebook) ─────────────────
COLUMNAS = {
    "SK_ID_CURR":          "ID_CLIENTE",
    "TARGET":              "TARGET",
    "CNT_CHILDREN":        "NUM_HIJOS",
    "AMT_INCOME_TOTAL":    "INGRESO_ANUAL",
    "AMT_CREDIT":          "MONTO_PRESTAMO",
    "AMT_ANNUITY":         "CUOTA_PRESTAMO",
    "AMT_GOODS_PRICE":     "VALOR_BIEN",
    "REGION_POPULATION_RELATIVE": "DENSIDAD_POBLACIONAL",
    "DAYS_BIRTH":          "EDAD_DIAS",
    "DAYS_EMPLOYED":       "ANTIGUEDAD_LABORAL_DIAS",
    "CNT_FAM_MEMBERS":     "NUM_INTEGRANTES_HOGAR",
    "EXT_SOURCE_1":        "PUNTAJE_EXTERNO_1",
    "EXT_SOURCE_2":        "PUNTAJE_EXTERNO_2",
    "EXT_SOURCE_3":        "PUNTAJE_EXTERNO_3",
    "CODE_GENDER":         "SEXO",
    "FLAG_OWN_CAR":        "TIENE_AUTO",
    "FLAG_OWN_REALTY":     "TIENE_VIVIENDA",
    "NAME_INCOME_TYPE":    "TIPO_INGRESO",
    "NAME_EDUCATION_TYPE": "NIVEL_EDUCATIVO",
    "NAME_FAMILY_STATUS":  "ESTADO_CIVIL",
    "NAME_HOUSING_TYPE":   "TIPO_VIVIENDA",
    "OCCUPATION_TYPE":     "OCUPACION",
    "NAME_CONTRACT_TYPE":  "TIPO_CONTRATO",
}

df = raw[list(COLUMNAS.keys())].rename(columns=COLUMNAS).copy()

# ── 3. Ingeniería de variables (idéntica al notebook) ─────────────────────────
df["EDAD"] = (-df["EDAD_DIAS"] / 365).round(1)

# Código centinela 365243 → pensionistas/sin empleo
df["ANTIGUEDAD_DESCONOCIDA"] = (df["ANTIGUEDAD_LABORAL_DIAS"] == 365243).astype(int)
df["ANTIGUEDAD_LABORAL_DIAS"] = df["ANTIGUEDAD_LABORAL_DIAS"].replace(365243, np.nan)
df["ANTIGUEDAD_LABORAL_DIAS"] = (-df["ANTIGUEDAD_LABORAL_DIAS"]).where(
    df["ANTIGUEDAD_LABORAL_DIAS"].notna()
)

# Depurar XNA en SEXO
df = df[df["SEXO"] != "XNA"].copy()

# Descartar VALOR_BIEN (VIF extremo) y columnas auxiliares
df = df.drop(columns=["ID_CLIENTE", "EDAD_DIAS", "VALOR_BIEN"])

# ── 4. Separar X / y ─────────────────────────────────────────────────────────
y = df.pop("TARGET")
X = df

COLS_NUM = X.select_dtypes(include="number").columns.tolist()
COLS_CAT = X.select_dtypes(exclude="number").columns.tolist()
print(f"  -> Variables numericas: {len(COLS_NUM)} | categoricas: {len(COLS_CAT)}")

# ── 5. Preprocesador ──────────────────────────────────────────────────────────
num_transformer = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scl", StandardScaler()),
])

cat_transformer = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocesador = ColumnTransformer([
    ("num", num_transformer, COLS_NUM),
    ("cat", cat_transformer, COLS_CAT),
])

# ── 6. Pipeline completo ──────────────────────────────────────────────────────
pipeline = Pipeline([
    ("pre", preprocesador),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEMILLA, solver="lbfgs")),
])

# ── 7. Entrenamiento ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEMILLA, stratify=y
)
print("\nEntrenando Regresión Logística…")
pipeline.fit(X_train, y_train)

probs = pipeline.predict_proba(X_test)[:, 1]
preds = (probs >= 0.50).astype(int)

print("\n-- Metricas sobre conjunto de prueba (umbral 0.50) --")
print(f"  ROC-AUC   : {roc_auc_score(y_test, probs):.4f}")
print(f"  Recall    : {recall_score(y_test, preds):.4f}")
print(f"  Precision : {precision_score(y_test, preds):.4f}")
print(f"  F1        : {f1_score(y_test, preds):.4f}")

# ── 8. Guardar artefactos ─────────────────────────────────────────────────────
out_path = os.path.join(BASE_DIR, "modelo_rl.pkl")
meta = {
    "pipeline":  pipeline,
    "cols_num":  COLS_NUM,
    "cols_cat":  COLS_CAT,
    "X_test":    X_test,
    "y_test":    y_test,
    "probs_test": probs,
}
joblib.dump(meta, out_path)
print(f"\nOK  Modelo guardado en: {out_path}")
