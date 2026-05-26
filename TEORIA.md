# 📚 Fundamentos Teóricos del Proyecto

> Documento complementario al [README.md](README.md). Explica en profundidad los algoritmos, fórmulas matemáticas y decisiones de diseño aplicadas en el notebook `CANCER_KNN.ipynb`.

---

## 📑 Tabla de Contenidos

1. [K-Nearest Neighbors (KNN)](#1-k-nearest-neighbors-knn)
2. [Preprocesamiento de Datos](#2-preprocesamiento-de-datos)
3. [Desbalanceo de Clases](#3-desbalanceo-de-clases)
4. [Optimización de Hiperparámetros](#4-optimización-de-hiperparámetros)
5. [Métricas de Evaluación](#5-métricas-de-evaluación)
6. [Referencias](#-referencias)

---

## 1. K-Nearest Neighbors (KNN)

El clasificador **K-Nearest Neighbors** es un algoritmo de aprendizaje supervisado **basado en instancias** (*lazy learning*). No construye un modelo explícito durante el entrenamiento; en su lugar, memoriza el dataset y realiza inferencia consultando los $k$ vecinos más cercanos al punto a clasificar.

### 🔹 Funcionamiento

1. Se calcula la distancia entre el nuevo punto y todos los puntos del entrenamiento.
2. Se seleccionan los $k$ puntos más cercanos.
3. Se asigna la clase por **votación mayoritaria** (clasificación) o **promedio** (regresión).

### 🔹 Elección de $k$

- **$k$ pequeño** → modelo flexible, riesgo de **overfitting** y sensibilidad al ruido.
- **$k$ grande** → modelo más suave, riesgo de **underfitting**.
- Regla heurística común: $k = \sqrt{n}$, donde $n$ es el número de muestras de entrenamiento.

### 🔹 Métricas de Distancia

| Métrica | Fórmula | Uso típico |
|---------|---------|------------|
| **Euclidiana** | $d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$ | Datos continuos, espacios geométricos |
| **Manhattan** | $d(x,y) = \sum_{i=1}^{n}\|x_i - y_i\|$ | Datos en cuadrículas, alta dimensionalidad |
| **Minkowski** | $d(x,y) = \left(\sum_{i=1}^{n}\|x_i - y_i\|^p\right)^{1/p}$ | Generalización: $p=1$ Manhattan, $p=2$ Euclidiana |

### 🔹 Pesos de los Vecinos (`weights`)

- **`uniform`**: todos los vecinos contribuyen por igual al voto.
- **`distance`**: el peso es inversamente proporcional a la distancia, dando más influencia a los vecinos cercanos:

$$w_i = \frac{1}{d(x, x_i)}$$

### 🔹 Ventajas y Desventajas

✅ Simple, intuitivo, sin entrenamiento explícito.
✅ Funciona bien con fronteras de decisión no lineales.
❌ Costoso computacionalmente en inferencia.
❌ Muy sensible a la escala de las variables y al ruido.
❌ Sufre la **maldición de la dimensionalidad**.

---

## 2. Preprocesamiento de Datos

### 🔹 Detección de Outliers (Método IQR)

El **Rango Intercuartílico (IQR)** identifica valores atípicos basándose en los cuartiles:

$$IQR = Q_3 - Q_1$$

Un valor $x$ se considera outlier si:

$$x < Q_1 - 1.5 \cdot IQR \quad \text{o} \quad x > Q_3 + 1.5 \cdot IQR$$

📌 **¿Por qué importa en KNN?** Como KNN se basa en distancias, los outliers distorsionan los cálculos y pueden generar fronteras de decisión incorrectas.

### 🔹 Escalado de Características (StandardScaler)

KNN es **sensible a la escala**. Una variable con rango [0, 1000] dominará el cálculo de distancia frente a otra con rango [0, 1]. `StandardScaler` aplica la **estandarización Z-score**:

$$z = \frac{x - \mu}{\sigma}$$

Donde $\mu$ es la media y $\sigma$ la desviación estándar. El resultado tiene media 0 y desviación estándar 1.

> ⚠️ **Importante:** el escalador se ajusta SOLO con los datos de entrenamiento (`fit_transform`) y se aplica al conjunto de prueba (`transform`) para evitar **data leakage**.

---

## 3. Desbalanceo de Clases

En problemas médicos es habitual que una clase esté subrepresentada. Si el 73% de las muestras son "Sobrevivió", un modelo trivial que prediga siempre esa clase tendría un 73% de accuracy, pero sería inútil clínicamente.

### 🔹 Técnicas de Sobremuestreo (Oversampling)

| Técnica | Descripción |
|---------|-------------|
| **RandomOverSampler** | Duplica aleatoriamente muestras de la clase minoritaria. Simple pero puede causar overfitting. |
| **SMOTE** | *Synthetic Minority Over-sampling Technique*: genera muestras sintéticas interpolando entre vecinos cercanos de la clase minoritaria. |
| **ADASYN** | *Adaptive Synthetic Sampling*: similar a SMOTE pero genera más muestras en zonas con frontera de decisión difícil. |
| **SMOTETomek** | Combina SMOTE con limpieza de **Tomek Links** (pares de puntos de clases distintas demasiado cercanos). |

### 🔹 Fórmula SMOTE

Dada una muestra $x_i$ de la clase minoritaria y uno de sus $k$ vecinos $x_{nn}$:

$$x_{new} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \in [0, 1]$$

---

## 4. Optimización de Hiperparámetros

### 🔹 Comparativa de Estrategias

| Método | Cómo explora | Eficiencia |
|--------|--------------|------------|
| **Grid Search** | Prueba TODAS las combinaciones de una malla. | ⛔ Costoso |
| **Random Search** | Muestrea aleatoriamente del espacio. | ⚠️ Mejor, pero ciego |
| **Bayesian Search** | Construye un modelo probabilístico y elige inteligentemente. | ✅ Óptimo |

### 🔹 Optimización Bayesiana

Usa un **modelo subrogado** (típicamente un *Gaussian Process*) que aproxima la función objetivo $f(\theta)$ (accuracy en función de los hiperparámetros). En cada iteración:

1. Ajusta el modelo subrogado con resultados previos.
2. Optimiza una **función de adquisición** (ej. *Expected Improvement*) para decidir el siguiente $\theta$ a evaluar.
3. Evalúa $f(\theta)$ real y actualiza el modelo.

Esto permite **converger más rápido** al óptimo con menos evaluaciones, especialmente útil cuando entrenar el modelo es costoso.

### 🔹 Espacio de Búsqueda Aplicado

```python
search_space = {
    'n_neighbors': Integer(3, 30),
    'weights':     Categorical(['uniform', 'distance']),
    'metric':      Categorical(['euclidean', 'manhattan', 'minkowski']),
    'p':           Integer(1, 2)
}
```

---

## 5. Métricas de Evaluación

En clasificación médica, la **accuracy global no es suficiente** debido al desbalanceo y al coste asimétrico de los errores.

### 🔹 Matriz de Confusión

|  | Predicho: Positivo | Predicho: Negativo |
|---|---|---|
| **Real: Positivo** | TP (Verdadero Positivo) | FN (Falso Negativo) |
| **Real: Negativo** | FP (Falso Positivo) | TN (Verdadero Negativo) |

### 🔹 Métricas Principales

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | % global de aciertos |
| **Precision** | $\frac{TP}{TP + FP}$ | De los predichos positivos, ¿cuántos lo son? |
| **Recall (Sensibilidad)** | $\frac{TP}{TP + FN}$ | De los positivos reales, ¿cuántos detectamos? |
| **F1-Score** | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | Media armónica equilibrada |
| **Specificity** | $\frac{TN}{TN + FP}$ | Capacidad de descartar negativos |

### 🔹 ¿Por qué Recall es Crítica Aquí?

Un **Falso Negativo** = predecir "Sobrevivirá" cuando en realidad la paciente está en alto riesgo. Este error puede impedir tratamientos agresivos preventivos. Por tanto, **maximizar el Recall de la clase minoritaria** es prioritario, incluso a costa de algunos Falsos Positivos.

---

## 📖 Referencias

- 📘 Cover, T., & Hart, P. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory.
- 📘 Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR.
- 📘 Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian Optimization of Machine Learning Algorithms*. NeurIPS.
- 🌐 [Documentación scikit-learn — KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- 🌐 [Documentación imbalanced-learn](https://imbalanced-learn.org/)
- 🌐 [Documentación scikit-optimize](https://scikit-optimize.github.io/)

---

<p align="center">
  <i>📌 Documento generado como apoyo teórico al notebook <code>CANCER_KNN.ipynb</code>.</i>
</p>
