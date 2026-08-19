# Detección de fraude: ¿Cómo cuido a mis clientes?

Trabajos Prácticos de la Mentoría de **DiploDatos 2026** — *"Detección de fraude: ¿Cómo cuido a mis clientes? Machine Learning aplicado a un caso de negocio financiero"*.

**Equipo:** Florencia Abagnale, Paula Karlen, Ana Maria Luna, Mauricio José Pintos.

## El problema

Un banco necesita detectar transacciones fraudulentas en tarjetas/cuentas de sus clientes. El dataset (`Muestra.del`, no incluido en este repo por su tamaño) contiene ~192.000 transacciones de 716 clientes durante el año 2025, con una variable binaria que indica si cada transacción fue fraude. La documentación de todas las variables y códigos está en [`Dr_Muestra.xlsx`](Dr_Muestra.xlsx).

El desafío central del problema: apenas el **0,62% de las transacciones son fraude** — cualquier análisis o modelo tiene que lidiar con ese desbalance extremo de clases.

## Los tres trabajos prácticos

### [TP1 — Análisis Exploratorio de Datos](TP1_Analisis_Exploratorio_EDA.ipynb)
Familiarización con el dataset y la documentación, identificación de variables, nulos, duplicados y outliers, y un análisis exhaustivo de qué distingue a una transacción fraudulenta: canal de la transacción (e-commerce vs. operatoria bancaria tradicional), moneda (pesos vs. dólares), uso de chip, horario, tiempo entre transacciones, rubro comercial y ciudad, entre otros.

### [TP2 — Curación de Datos](TP2_Curacion_Datos.ipynb)
Continúa sobre el TP1: tratamiento formal de duplicados/nulos/outliers/valores erróneos, creación de variables nuevas (*feature engineering* — en particular, el desvío del importe de cada transacción respecto al comportamiento habitual de cada cliente), codificación de variables categóricas evitando fuga de información hacia el conjunto de evaluación, y preparación final del dataset (dividido en train/test de forma estratificada) para el modelado.

### [TP3 — Modelos de Clasificación](TP3_Modelos_Clasificacion.ipynb)
Entrenamiento y comparación de modelos: un baseline ingenuo y uno informado por reglas de negocio, un modelo lineal (Regresión Logística), modelos basados en árboles (Árbol de Decisión y Random Forest), y ajuste de hiperparámetros con validación cruzada — incluyendo ajuste del umbral de decisión y comparación de técnicas de balanceo de clases (`class_weight` vs. SMOTE). El modelo final recomendado es un **Random Forest** (F1 ≈ 0,64, AUC-PR ≈ 0,70 en evaluación).

## Cómo correr los notebooks

Cada notebook está pensado para correr en Google Colab: al ejecutar las primeras celdas, van a pedir subir los archivos de datos que necesitan (`Muestra.del` y `Dr_Muestra.xlsx` para el TP1 y TP2; `tp2_train.csv`/`tp2_test.csv`, generados por el TP2, para el TP3).
