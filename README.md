# 🩺 Clasificación de Supervivencia en Cáncer de Mama con KNN

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completado-success)

> Proyecto de Machine Learning que predice la supervivencia a 5 años de pacientes operadas de cáncer de mama mediante el algoritmo **K-Nearest Neighbors (KNN)**, con optimización bayesiana de hiperparámetros y técnicas de balanceo de clases.

---

## 📑 Tabla de Contenidos

- [Descripción](#-descripción-del-proyecto)
- [Dataset](#-dataset)
- [Tecnologías](#-tecnologías-utilizadas)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Instalación y Uso](#-instalación-y-uso)
- [Flujo de Trabajo](#-flujo-de-trabajo)
- [Resultados](#-resultados-esperados)
- [Documentación Teórica](#-documentación-teórica)
- [Autor](#-autor)

---

## 📋 Descripción del Proyecto

Este proyecto aborda un problema de **clasificación binaria** en el dominio médico: determinar si una paciente sobrevivirá **más de 5 años** tras una cirugía de cáncer de mama, basándose en variables clínicas básicas.

El estudio se enfoca en:

- 🔍 Realizar un **Análisis Exploratorio de Datos (EDA)** completo.
- ⚖️ Gestionar el **desbalanceo de clases** propio de los datasets médicos.
- 🤖 Entrenar y comparar modelos **KNN** con distintas configuraciones.
- 🎯 Optimizar hiperparámetros mediante **Optimización Bayesiana**.
- 📊 Evaluar el modelo con métricas adecuadas al contexto clínico.

---

## 📊 Dataset

Se utiliza el **Haberman's Survival Dataset** del [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival), proveniente de un estudio del Hospital Hamerman (Chicago) entre 1958 y 1970.

| Característica | Tipo | Descripción |
|----------------|------|-------------|
| `Edad` | Numérico | Edad de la paciente al momento de la operación |
| `Año_Operacion` | Numérico | Año de la cirugía (formato 19xx) |
| `Ganglios_Axilares` | Numérico | Número de ganglios axilares positivos detectados |
| `Estado_Supervivencia` | Categórico | **1** = Sobrevivió ≥ 5 años · **2** = Murió < 5 años |

- **Instancias totales:** 306
- **Valores faltantes:** Ninguno
- **Desbalanceo:** Aproximadamente 73% clase 1 vs 27% clase 2

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Herramientas |
|-----------|--------------|
| **Lenguaje** | Python 3.10+ |
| **Análisis de datos** | `pandas`, `numpy` |
| **Visualización** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` |
| **Balanceo de clases** | `imbalanced-learn` (SMOTE, ADASYN, SMOTETomek) |
| **Optimización** | `scikit-optimize` (BayesSearchCV) |
| **Entorno** | Jupyter Notebook |

---

## 📁 Estructura del Repositorio

```
CANCER_KNN/
├── CANCER_KNN.ipynb     # Notebook principal con el análisis completo
├── README.md            # Documentación del proyecto (este archivo)
└── TEORIA.md            # Fundamentos teóricos y matemáticos
```

---

## ⚙️ Instalación y Uso

### 1. Clona el repositorio

```bash
git clone https://github.com/<tu-usuario>/CANCER_KNN.git
cd CANCER_KNN
```

### 2. Instala las dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scikit-optimize tqdm
```

### 3. Ejecuta el notebook

```bash
jupyter notebook CANCER_KNN.ipynb
```

> 💡 También puedes abrirlo directamente en **Google Colab** o **VS Code**.

---

## 🚀 Flujo de Trabajo

```mermaid
flowchart LR
    A[📥 Carga UCI] --> B[🔍 EDA]
    B --> C[🧹 Outliers IQR]
    C --> D[📏 Escalado]
    D --> E[✂️ Split]
    E --> F[⚖️ Balanceo]
    F --> G[🤖 KNN Base]
    G --> H[🎯 BayesSearchCV]
    H --> I[📊 Evaluación]
```

1. **Carga de datos:** Descarga directa desde el repositorio UCI.
2. **EDA:** Histogramas, boxplots y matriz de correlación.
3. **Limpieza:** Eliminación de outliers en `Ganglios_Axilares` con el método IQR.
4. **Escalado:** Normalización Z-score con `StandardScaler`.
5. **División:** 80% entrenamiento / 20% prueba (estratificado).
6. **Balanceo:** Aplicación de `RandomOverSampler` (comparado con SMOTE, ADASYN, SMOTETomek).
7. **Modelado base:** KNN con $k = \sqrt{n}$ como punto de partida.
8. **Optimización:** Búsqueda bayesiana sobre `n_neighbors`, `weights`, `metric` y `p`.
9. **Evaluación:** Matriz de confusión, accuracy, precision, recall y F1-score.

---

## 📈 Resultados Esperados

El modelo optimizado se evalúa principalmente por su capacidad de **detectar correctamente la clase minoritaria** (pacientes con menor supervivencia), priorizando el **Recall** sobre la precisión global para reducir falsos negativos en un contexto clínico.

| Métrica | Importancia |
|---------|-------------|
| **Accuracy** | Visión general del rendimiento |
| **Recall (clase 2)** | ⭐ Crítica: detectar pacientes en riesgo |
| **F1-Score** | Equilibrio robusto frente al desbalanceo |
| **Matriz de Confusión** | Análisis visual de aciertos/errores |

---

## 📚 Documentación Teórica

Para profundizar en los fundamentos matemáticos y conceptuales utilizados (KNN, métricas de distancia, IQR, SMOTE, Optimización Bayesiana, etc.), consulta:

➡️ **[TEORIA.md](TEORIA.md)**

---

## 📄 Licencia

Distribuido bajo licencia **MIT**.

---

<p align="center">
  <i>⭐ Si te resultó útil, no olvides dejar una estrella al repositorio.</i>
</p>
