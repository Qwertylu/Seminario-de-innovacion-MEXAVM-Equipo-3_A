"""
dashboard_credito.py
====================
Dashboard interactivo Streamlit – Análisis de Riesgo de Crédito
Equipo 3 | Seminario de Innovación | UNIR MAVM

Ejecución:
    streamlit run dashboard_credito.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, recall_score, precision_score, f1_score, accuracy_score,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Riesgo Crediticio",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_rl.pkl")
CSV_PATH    = os.path.join(BASE_DIR, "application_train.csv")
SAMPLE_PATH = os.path.join(BASE_DIR, "application_train_sample.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Carga de artefactos (con caché)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo…")
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner="Cargando datos…", max_entries=1)
def load_data():
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
    if os.path.exists(CSV_PATH):
        raw = pd.read_csv(CSV_PATH)
    elif os.path.exists(SAMPLE_PATH):
        raw = pd.read_csv(SAMPLE_PATH)
    else:
        st.error("No se encontró el archivo de datos.")
        return pd.DataFrame()
    cols_existentes = {k: v for k, v in COLUMNAS.items() if k in raw.columns}
    df = raw[list(cols_existentes.keys())].rename(columns=cols_existentes).copy()
    df["EDAD"] = (-df["EDAD_DIAS"] / 365).round(1)
    df["ANTIGUEDAD_DESCONOCIDA"] = (df["ANTIGUEDAD_LABORAL_DIAS"] == 365243).astype(int)
    df["ANTIGUEDAD_LABORAL_DIAS"] = df["ANTIGUEDAD_LABORAL_DIAS"].replace(365243, np.nan)
    df["ANTIGUEDAD_LABORAL_DIAS"] = (-df["ANTIGUEDAD_LABORAL_DIAS"]).where(
        df["ANTIGUEDAD_LABORAL_DIAS"].notna()
    )
    df = df[df["SEXO"] != "XNA"].copy()
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Barra lateral – Navegación
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.shields.io/badge/UNIR-Seminario%20Innovaci%C3%B3n-blue?style=for-the-badge",
    use_container_width=True,
)
st.sidebar.title("🏦 Riesgo Crediticio")
st.sidebar.caption("Equipo 3 · MAVM · 2024-2025")
st.sidebar.divider()

PAGINAS = {
    "🏠 Inicio":                   "inicio",
    "🔍 Análisis del Dataset":      "dataset",
    "🤖 Rendimiento del Modelo":    "modelo",
    "🎚️ Sensibilidad del Umbral":   "umbral",
    "👤 Simulador Individual":       "simulador",
}
seccion = st.sidebar.radio("Navegación", list(PAGINAS.keys()), label_visibility="collapsed")
pagina = PAGINAS[seccion]

st.sidebar.divider()
st.sidebar.caption(
    "**Repositorio:** [GitHub](https://github.com/Qwertylu/Seminario-de-innovacion-MEXAVM-Equipo-3_A)"
)

# ─────────────────────────────────────────────────────────────────────────────
# Carga de recursos
# ─────────────────────────────────────────────────────────────────────────────
meta = load_model()
pipeline  = meta["pipeline"]
cols_num  = meta["cols_num"]
cols_cat  = meta["cols_cat"]
X_test    = meta["X_test"]
y_test    = meta["y_test"]
probs_test = meta["probs_test"]

# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 1 – INICIO
# ─────────────────────────────────────────────────────────────────────────────
if pagina == "inicio":
    st.title("🏦 Análisis de Riesgo para Solicitudes de Crédito")
    st.subheader("Prototipo de Solución — Entregable 4 | Seminario de Innovación")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset", "307,511 solicitudes", "Home Credit Default Risk")
    with col2:
        st.metric("Tasa de Incumplimiento", "8.07 %", "Clase desbalanceada")
    with col3:
        st.metric("Modelo Seleccionado", "Reg. Logística", "Mayor Recall y ROC-AUC")
    with col4:
        st.metric("ROC-AUC Alcanzado", "0.744", "Conjunto de prueba (20%)")

    st.divider()
    st.markdown("""
    ### ¿Qué es este prototipo?

    Este dashboard es el **prototipo funcional** del proyecto de análisis de riesgo crediticio
    desarrollado como parte del Trabajo Final de Máster en Análisis y Visualización de Datos Masivos
    en la Universidad Internacional de La Rioja (UNIR).

    El prototipo integra **todos los componentes** del flujo de trabajo analítico:

    | Fase CRISP-DM | Componente | Sección del Dashboard |
    |---|---|---|
    | Business Understanding | Contexto del problema crediticio | Esta página |
    | Data Understanding | Estadísticas descriptivas y distribuciones | 🔍 Análisis del Dataset |
    | Data Preparation | Pipeline de limpieza e ingeniería de variables | 🤖 Rendimiento del Modelo |
    | Modeling | Regresión Logística entrenada y comparativa | 🤖 Rendimiento del Modelo |
    | Evaluation | Curvas ROC, matrices de confusión, sensibilidad | 🎚️ Sensibilidad del Umbral |
    | Deployment | Predicción interactiva por solicitud individual | 👤 Simulador Individual |

    ### Autores
    - Sariel Yovany López Maradiaga
    - José Alberto Ureño Esquivel
    - Rodrigo Miguel Berrocal Vera Gutiérrez

    **Directora:** Bárbaro Jorge Ferro
    """)

    st.info(
        "**Cómo usar este dashboard:** Utiliza el menú de la izquierda para navegar entre las secciones. "
        "Cada sección es interactiva: puedes ajustar parámetros con los controles y ver los resultados en tiempo real.",
        icon="ℹ️",
    )

# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 2 – ANÁLISIS DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "dataset":
    st.title("🔍 Análisis del Dataset")
    st.caption("Home Credit Default Risk — 307,511 solicitudes de crédito")

    df = load_data()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Registros", f"{len(df):,}")
    c2.metric("Incumplimiento", f"{df['TARGET'].mean()*100:.2f}%")
    c3.metric("Variables Numéricas", str(df.select_dtypes(include='number').shape[1]))
    c4.metric("Variables Categóricas", str(df.select_dtypes(exclude='number').shape[1]))

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📊 Distribuciones", "🗂️ Datos Faltantes", "📋 Estadísticas"])

    with tab1:
        st.subheader("Distribución de Variables Numéricas")
        var_num = st.selectbox(
            "Seleccionar variable:",
            ["INGRESO_ANUAL", "MONTO_PRESTAMO", "CUOTA_PRESTAMO", "EDAD",
             "PUNTAJE_EXTERNO_1", "PUNTAJE_EXTERNO_2", "PUNTAJE_EXTERNO_3"],
        )
        col_a, col_b = st.columns(2)
        with col_a:
            muestra = df[var_num].dropna().sample(min(50000, len(df)), random_state=42)
            fig = px.histogram(
                muestra, nbins=80, title=f"Distribución de {var_num}",
                color_discrete_sequence=["#4c72b0"],
                labels={"value": var_num, "count": "Frecuencia"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = px.box(
                df, y=var_num, color="TARGET",
                title=f"{var_num} por clase (0=Paga · 1=Incumple)",
                color_discrete_map={0: "#4c72b0", 1: "#c44e52"},
                labels={"TARGET": "Clase"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Distribución de Variables Categóricas")
        var_cat = st.selectbox(
            "Seleccionar variable:",
            ["SEXO", "TIPO_INGRESO", "NIVEL_EDUCATIVO", "ESTADO_CIVIL",
             "TIPO_VIVIENDA", "TIENE_AUTO", "TIENE_VIVIENDA"],
        )
        tasa = df.groupby(var_cat)["TARGET"].agg(["mean", "count"]).reset_index()
        tasa.columns = [var_cat, "Tasa_Incumplimiento", "N"]
        tasa["Tasa_Incumplimiento"] *= 100
        tasa = tasa.sort_values("Tasa_Incumplimiento", ascending=False)

        fig3 = px.bar(
            tasa, x=var_cat, y="Tasa_Incumplimiento",
            text=tasa["Tasa_Incumplimiento"].map("{:.1f}%".format),
            title=f"Tasa de incumplimiento por {var_cat}",
            color="Tasa_Incumplimiento",
            color_continuous_scale="RdYlGn_r",
            labels={"Tasa_Incumplimiento": "Tasa de Incumplimiento (%)"},
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.subheader("Valores Faltantes por Variable")
        nulos = df.isnull().mean().mul(100).sort_values(ascending=False)
        nulos = nulos[nulos > 0].reset_index()
        nulos.columns = ["Variable", "% Faltante"]
        fig_n = px.bar(
            nulos, x="Variable", y="% Faltante",
            title="Porcentaje de valores faltantes",
            color="% Faltante", color_continuous_scale="Oranges",
        )
        fig_n.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_n, use_container_width=True)
        st.dataframe(nulos.style.format({"% Faltante": "{:.2f}%"}), use_container_width=True)

    with tab3:
        st.subheader("Estadísticas Descriptivas")
        nums = ["INGRESO_ANUAL", "MONTO_PRESTAMO", "CUOTA_PRESTAMO", "EDAD",
                "NUM_HIJOS", "PUNTAJE_EXTERNO_1", "PUNTAJE_EXTERNO_2", "PUNTAJE_EXTERNO_3"]
        st.dataframe(df[nums].describe().T.style.format("{:.2f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 3 – RENDIMIENTO DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "modelo":
    st.title("🤖 Rendimiento del Modelo")

    tab1, tab2, tab3 = st.tabs(["📈 Curva ROC", "⚖️ Comparativa de Modelos", "🔲 Matriz de Confusión"])

    with tab1:
        st.subheader("Curva ROC – Regresión Logística")
        fpr, tpr, _ = roc_curve(y_test, probs_test)
        auc_val = roc_auc_score(y_test, probs_test)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"Regresión Logística (AUC = {auc_val:.4f})",
            line=dict(color="#4c72b0", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Clasificador Aleatorio",
            line=dict(color="gray", width=1.5, dash="dash"),
        ))
        fig.update_layout(
            xaxis_title="Tasa de Falsos Positivos",
            yaxis_title="Tasa de Verdaderos Positivos (Recall)",
            legend=dict(x=0.6, y=0.1),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"**ROC-AUC = {auc_val:.4f}** — El modelo discrimina correctamente entre solicitantes "
            "que pagarán y los que incumplirán en el 74.4% de los casos, superando ampliamente al "
            "clasificador aleatorio (AUC = 0.50).",
            icon="📊",
        )

    with tab2:
        st.subheader("Comparativa de Modelos (umbral = 0.50)")
        resultados = pd.DataFrame({
            "Modelo": ["Regresión Logística", "Árbol de Decisión", "Random Forest"],
            "ROC-AUC":   [0.744, 0.627, 0.718],
            "Recall":    [0.675, 0.608, 0.519],
            "Precision": [0.214, 0.177, 0.259],
            "F1":        [0.326, 0.272, 0.345],
            "Accuracy":  [0.693, 0.620, 0.866],
        })

        col1, col2 = st.columns(2)
        with col1:
            fig_auc = px.bar(
                resultados, x="Modelo", y="ROC-AUC",
                title="ROC-AUC por Modelo",
                color="Modelo",
                color_discrete_sequence=["#4c72b0", "#dd8452", "#55a868"],
                text=resultados["ROC-AUC"].map("{:.3f}".format),
            )
            fig_auc.update_traces(textposition="outside")
            fig_auc.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig_auc, use_container_width=True)

        with col2:
            fig_rec = px.bar(
                resultados, x="Modelo", y="Recall",
                title="Recall (Exhaustividad) por Modelo",
                color="Modelo",
                color_discrete_sequence=["#4c72b0", "#dd8452", "#55a868"],
                text=resultados["Recall"].map("{:.3f}".format),
            )
            fig_rec.update_traces(textposition="outside")
            fig_rec.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig_rec, use_container_width=True)

        st.dataframe(
            resultados.set_index("Modelo").style
                .highlight_max(axis=0, props="background-color:#d4edda;color:#155724;font-weight:bold")
                .format("{:.3f}"),
            use_container_width=True,
        )
        st.caption(
            "**Decisión:** La Regresión Logística se selecciona por su máximo Recall (0.675) "
            "y ROC-AUC (0.744), siendo también el modelo más interpretable y adecuado "
            "para datos desbalanceados."
        )

    with tab3:
        st.subheader("Matriz de Confusión – Regresión Logística")
        umbral_cm = st.slider("Umbral de clasificación:", 0.10, 0.80, 0.50, 0.05, key="cm_slider")
        preds_cm = (probs_test >= umbral_cm).astype(int)
        cm = confusion_matrix(y_test, preds_cm)

        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicho", y="Real", color="Conteo"),
            x=["Paga (0)", "Incumple (1)"],
            y=["Paga (0)", "Incumple (1)"],
            color_continuous_scale="Blues",
            text_auto=True,
            title=f"Matriz de Confusión (umbral = {umbral_cm})",
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall",    f"{recall_score(y_test, preds_cm):.3f}")
        c2.metric("Precision", f"{precision_score(y_test, preds_cm, zero_division=0):.3f}")
        c3.metric("F1",        f"{f1_score(y_test, preds_cm, zero_division=0):.3f}")
        c4.metric("Accuracy",  f"{accuracy_score(y_test, preds_cm):.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 4 – SENSIBILIDAD DEL UMBRAL
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "umbral":
    st.title("🎚️ Sensibilidad del Umbral de Clasificación")
    st.markdown(
        "El umbral de clasificación es el valor a partir del cual el modelo etiqueta a un "
        "solicitante como **potencial incumplidor**. Bajar el umbral detecta más impagos "
        "(mayor Recall) pero genera más falsas alarmas (menor Precision)."
    )

    umbrales = np.arange(0.10, 0.81, 0.05)
    filas = []
    for u in umbrales:
        p = (probs_test >= u).astype(int)
        filas.append({
            "Umbral":    round(u, 2),
            "Recall":    recall_score(y_test, p),
            "Precision": precision_score(y_test, p, zero_division=0),
            "F1":        f1_score(y_test, p, zero_division=0),
            "Accuracy":  accuracy_score(y_test, p),
            "Alertas_pct": p.mean() * 100,
        })
    df_u = pd.DataFrame(filas)

    fig_u = go.Figure()
    colores = {"Recall": "#c44e52", "Precision": "#4c72b0", "F1": "#55a868"}
    for metrica, color in colores.items():
        fig_u.add_trace(go.Scatter(
            x=df_u["Umbral"], y=df_u[metrica],
            mode="lines+markers", name=metrica,
            line=dict(color=color, width=2.5),
            marker=dict(size=7),
        ))

    # Umbral seleccionado por usuario
    umbral_sel = st.slider("Explorar umbral:", 0.10, 0.80, 0.30, 0.05, key="u_slider")
    fig_u.add_vline(
        x=umbral_sel, line_dash="dash", line_color="orange",
        annotation_text=f"Umbral = {umbral_sel}",
        annotation_position="top right",
    )
    fig_u.update_layout(
        title="Métricas vs. Umbral de Clasificación",
        xaxis_title="Umbral",
        yaxis_title="Valor de la Métrica",
        yaxis_range=[0, 1],
        height=480,
        legend=dict(x=0.75, y=0.95),
    )
    st.plotly_chart(fig_u, use_container_width=True)

    # Tabla de puntos operativos clave
    st.subheader("Puntos Operativos Clave")
    puntos = df_u[df_u["Umbral"].isin([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.70])]
    st.dataframe(
        puntos.set_index("Umbral").style
            .format({
                "Recall": "{:.3f}", "Precision": "{:.3f}",
                "F1": "{:.3f}", "Accuracy": "{:.3f}",
                "Alertas_pct": "{:.1f}%",
            })
            .highlight_between(subset=["Recall"], left=0.67, right=1.0,
                                props="background-color:#ffd7d7"),
        use_container_width=True,
    )

    # Impacto del umbral seleccionado
    st.subheader(f"Impacto del Umbral = {umbral_sel}")
    row = df_u[df_u["Umbral"] == umbral_sel].iloc[0]
    total_test = len(y_test)
    alertas = int(row["Alertas_pct"] / 100 * total_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall (Detección)", f"{row['Recall']:.3f}",
              help="Fracción de impagos reales que el modelo detecta")
    c2.metric("Precision", f"{row['Precision']:.3f}",
              help="De las alertas generadas, cuántas son impagos reales")
    c3.metric("F1-Score", f"{row['F1']:.3f}")
    c4.metric("Solicitudes alertadas", f"{row['Alertas_pct']:.1f}%",
              f"≈ {alertas:,} de {total_test:,} en el conjunto de prueba")

    st.info(
        f"Con umbral **{umbral_sel}**, el modelo detecta el **{row['Recall']*100:.1f}%** de los "
        f"incumplidores reales, generando alertas en el **{row['Alertas_pct']:.1f}%** de las solicitudes. "
        f"La Precision de **{row['Precision']:.3f}** indica que 1 de cada "
        f"{1/row['Precision']:.0f} alertas corresponde a un incumplidor real.",
        icon="💡",
    )

# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA 5 – SIMULADOR INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "simulador":
    st.title("👤 Simulador de Riesgo Individual")
    st.markdown(
        "Ingresa las características de un solicitante de crédito y el modelo "
        "calculará su **probabilidad estimada de incumplimiento** en tiempo real."
    )

    st.divider()
    col_form, col_res = st.columns([1.4, 1])

    with col_form:
        st.subheader("Datos del Solicitante")

        c1, c2 = st.columns(2)
        edad         = c1.number_input("Edad (años)", 18, 80, 35)
        num_hijos    = c2.number_input("Número de hijos", 0, 15, 0)

        c3, c4 = st.columns(2)
        ingreso      = c3.number_input("Ingreso anual (USD)", 10_000, 1_000_000, 135_000, step=5_000)
        monto        = c4.number_input("Monto del préstamo (USD)", 10_000, 4_000_000, 500_000, step=10_000)

        c5, c6 = st.columns(2)
        cuota        = c5.number_input("Cuota mensual (USD)", 1_000, 100_000, 25_000, step=500)
        densidad     = c6.number_input("Densidad poblacional (relativa)", 0.0, 0.1, 0.025, step=0.001, format="%.3f")

        c7, c8 = st.columns(2)
        antig_dias   = c7.number_input("Antigüedad laboral (días)", 0, 20_000, 2_000)
        num_int      = c8.number_input("Integrantes del hogar", 1, 20, 2)

        c9, c10 = st.columns(2)
        ext1 = c9.number_input("Puntaje Externo 1", 0.0, 1.0, 0.50, step=0.01)
        ext2 = c10.number_input("Puntaje Externo 2", 0.0, 1.0, 0.55, step=0.01)
        ext3 = st.number_input("Puntaje Externo 3", 0.0, 1.0, 0.50, step=0.01)

        st.markdown("---")
        st.subheader("Datos Cualitativos")

        c11, c12 = st.columns(2)
        sexo         = c11.selectbox("Sexo", ["M", "F"])
        tiene_auto   = c12.selectbox("¿Tiene automóvil?", ["Y", "N"])

        c13, c14 = st.columns(2)
        tiene_viv    = c13.selectbox("¿Tiene vivienda propia?", ["Y", "N"])
        tipo_ingreso = c14.selectbox("Tipo de ingreso", [
            "Working", "Commercial associate", "Pensioner", "State servant", "Student"
        ])

        c15, c16 = st.columns(2)
        nivel_educ   = c15.selectbox("Nivel educativo", [
            "Secondary / secondary special", "Higher education",
            "Incomplete higher", "Lower secondary", "Academic degree"
        ])
        estado_civil = c16.selectbox("Estado civil", [
            "Married", "Single / not married", "Civil marriage",
            "Separated", "Widow"
        ])

        c17, c18 = st.columns(2)
        tipo_viv     = c17.selectbox("Tipo de vivienda", [
            "House / apartment", "With parents", "Municipal apartment",
            "Rented apartment", "Office apartment", "Co-op apartment"
        ])
        ocupacion    = c18.selectbox("Ocupación", [
            "Laborers", "Sales staff", "Core staff", "Managers",
            "Drivers", "High skill tech staff", "Accountants",
            "Medicine staff", "Cleaning staff", "Cooking staff", None
        ])

        tipo_contrato = st.selectbox("Tipo de contrato", ["Cash loans", "Revolving loans"])

    with col_res:
        st.subheader("Resultado de la Evaluación")
        umbral_sim = st.slider("Umbral de decisión:", 0.10, 0.80, 0.30, 0.05, key="sim_slider")

        # Construir DataFrame con los valores del formulario
        solicitud = pd.DataFrame([{
            "NUM_HIJOS":              num_hijos,
            "INGRESO_ANUAL":          ingreso,
            "MONTO_PRESTAMO":         monto,
            "CUOTA_PRESTAMO":         cuota,
            "DENSIDAD_POBLACIONAL":   densidad,
            "ANTIGUEDAD_LABORAL_DIAS": antig_dias,
            "NUM_INTEGRANTES_HOGAR":  num_int,
            "PUNTAJE_EXTERNO_1":      ext1,
            "PUNTAJE_EXTERNO_2":      ext2,
            "PUNTAJE_EXTERNO_3":      ext3,
            "ANTIGUEDAD_DESCONOCIDA": 0,
            "EDAD":                   edad,
            "SEXO":                   sexo,
            "TIENE_AUTO":             tiene_auto,
            "TIENE_VIVIENDA":         tiene_viv,
            "TIPO_INGRESO":           tipo_ingreso,
            "NIVEL_EDUCATIVO":        nivel_educ,
            "ESTADO_CIVIL":           estado_civil,
            "TIPO_VIVIENDA":          tipo_viv,
            "OCUPACION":              ocupacion if ocupacion else "Laborers",
            "TIPO_CONTRATO":          tipo_contrato,
        }])

        prob = pipeline.predict_proba(solicitud)[0, 1]
        clasif = "⚠️ ALTO RIESGO" if prob >= umbral_sim else "✅ BAJO RIESGO"
        color  = "#c44e52" if prob >= umbral_sim else "#28a745"

        st.markdown(f"""
        <div style="text-align:center; padding:30px; border-radius:15px;
                    background-color:{color}22; border: 2px solid {color};">
            <h1 style="color:{color}; margin:0;">{clasif}</h1>
            <h2 style="color:{color}; margin:10px 0;">P(incumplimiento) = {prob:.3f}</h2>
            <p style="color:#666;">Umbral de decisión: {umbral_sim}</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Medidor tipo gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            title={"text": "Probabilidad de Incumplimiento (%)"},
            delta={"reference": umbral_sim * 100, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0, umbral_sim * 100], "color": "#d4edda"},
                    {"range": [umbral_sim * 100, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "orange", "width": 4},
                    "thickness": 0.75,
                    "value": umbral_sim * 100,
                },
            },
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        **Interpretación:**
        - Con un umbral de **{umbral_sim}**, se clasifica como **riesgo alto** cualquier 
          solicitud con probabilidad ≥ {umbral_sim:.0%}.
        - Este solicitante obtiene una probabilidad de **{prob:.1%}**, lo que indica {
            'mayor riesgo de incumplimiento que el umbral establecido.'
            if prob >= umbral_sim else
            'un perfil de crédito dentro del rango aceptable.'
          }
        """)
