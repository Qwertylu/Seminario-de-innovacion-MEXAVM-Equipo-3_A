# 🏦 Análisis de Riesgo para Solicitudes de Crédito
### Prototipo Funcional de Machine Learning & Dashboard Interactivo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://seminario-de-innovacion-mexavm-equipo-3a-6t8grp4s2drhwk8yuwndi.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositorio contiene el prototipo analítico y el cuadro de mando interactivo desarrollado para el proyecto de investigación **"Análisis de riesgo para solicitudes de crédito"**, correspondiente al Trabajo Final de Máster de la **Maestría en Análisis y Visualización de Datos Masivos (UNIR)**.

---

## 👥 Equipo de Trabajo (Equipo 3)
* **Sariel Yovany López Maradiaga**
* **José Alberto Ureño Esquivel**
* **Rodrigo Miguel Berrocal Vera Gutiérrez**
* **Directora:** Bárbaro Jorge Ferro

---

## 📌 Descripción del Proyecto
El objetivo del proyecto es estimar con rigor estadístico la probabilidad de incumplimiento (*default*) en solicitantes de crédito, basándose en el conjunto de datos abierto de **Home Credit Default Risk** (307.511 registros y 122 variables iniciales).

### Hallazgos Principales:
* **Fuerte desbalance:** Solo el **8,07%** de los clientes incurre en impago.
* **Modelo seleccionado:** **Regresión Logística ponderada** (`class_weight='balanced'`), logrando un **ROC-AUC de 0,744** y una exhaustividad (**Recall**) del **67,5%** sobre el conjunto de prueba, superando a los modelos basados en árboles y garantizando interpretabilidad regulatoria.
* **Sensibilidad operativa:** Se identificó que un umbral de decisión en **0,30** maximiza la detección de impagos (Recall = 78%), mientras que **0,50** ofrece equilibrio con los costes de falsas alarmas.

---

## 🚀 Estructura del Repositorio

```text
├── .streamlit/
│   └── config.toml               # Configuración del tema visual oscuro (Dark Mode)
├── dashboard_credito.py          # Aplicación web interactiva Streamlit (5 módulos)
├── train_and_save_model.py       # Script de re-entrenamiento y exportación del pipeline
├── modelo_rl.pkl                 # Pipeline serializado scikit-learn (preprocesador + modelo)
├── application_train_sample.csv  # Muestra estratificada representativa (25.000 registros)
├── SEMINARIO_2_corregido.ipynb   # Cuaderno Jupyter con las 5 fases analíticas completas
├── requirements.txt              # Dependencias del entorno Python
└── README.md                     # Documentación general
```

---

## 📊 Módulos del Dashboard (`dashboard_credito.py`)

1. **🏠 Inicio:** Visión general, métricas clave del proyecto y correspondencia con el estándar **CRISP-DM**.
2. **🔍 Análisis del Dataset:** Exploración interactiva de distribuciones, correlaciones, valores atípicos y valores faltantes.
3. **🤖 Rendimiento del Modelo:** Curva ROC interactiva, tabla comparativa de modelos (Regresión Logística vs. Árbol de Decisión vs. Random Forest) y matriz de confusión dinámica con umbral ajustable.
4. **🎚️ Sensibilidad del Umbral:** Curvas Precision-Recall-F1 vs. umbral de corte, evaluando el impacto de negocio.
5. **👤 Simulador de Riesgo Individual:** Formulario interactivo para ingresar las características de un solicitante y obtener en tiempo real la probabilidad de impago con un medidor gráfico (*gauge*).

---

## 💻 Ejecución Local

### 1. Clonar el repositorio
```bash
git clone https://github.com/Qwertylu/Seminario-de-inovacion-MEXAVM-Equipo-3_A.git
cd Seminario-de-inovacion-MEXAVM-Equipo-3_A
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Lanzar el dashboard
```bash
streamlit run dashboard_credito.py
```
El cuadro de mando se abrirá automáticamente en tu navegador en `http://localhost:8501`.
