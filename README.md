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
Continúa sobre el TP1: tratamiento formal de duplicados/nulos/outliers/valores erróneos, creación de variables nuevas (*feature engineering* — en particular, el desvío del importe de cada transacción respecto al comportamiento habitual de cada cliente), codificación de variables categóricas evitando fuga de información hacia el conjunto de evaluación, y preparación final del dataset (dividido en train/test de forma estratificada) para el modelado. Incluye además un chequeo puntual (sección 4.1) que descarta `Franja_Horaria` del dataset final: su señal real ya está cubierta por `Hora_Dia`, y mantenerla como 5 categorías separadas empeora el resultado del modelo.

### [TP3 — Modelos de Clasificación](TP3_Modelos_Clasificacion.ipynb)
Entrenamiento y comparación de modelos: un baseline ingenuo y uno informado por reglas de negocio, un modelo lineal (Regresión Logística), modelos basados en árboles (Árbol de Decisión y Random Forest), y ajuste de hiperparámetros con validación cruzada — incluyendo ajuste del umbral de decisión y comparación de técnicas de balanceo de clases (`class_weight` vs. SMOTE). El modelo final recomendado es un **Random Forest** (F1 ≈ 0,64, AUC-PR ≈ 0,69 en evaluación).

## Trabajo adicional (no asignado por la mentoría)

### [Mejoras al modelo y consideraciones de producción](Mejoras_Modelo_y_Produccion.ipynb)
Resuelve puntualmente cuatro "próximos pasos" que quedaron señalados en las conclusiones del TP3:
1. **Corrige la fuga temporal** de las variables de comportamiento del cliente (se recalculan con ventana expansiva, usando solo transacciones anteriores) y reentrena el Random Forest -- su F1 real es 0,613, no el 0,638 original.
2. **Ajusta el umbral de decisión** del Random Forest (en el TP3 solo se había hecho para la Regresión Logística) -- con umbral 0,32 el F1 sube a **0,645**, el mejor resultado del proyecto hasta ese punto.
3. **Entrena y ajusta un modelo de boosting** (`HistGradientBoostingClassifier`) y lo compara contra el Random Forest.
4. **Desarrolla consideraciones de producción**: revisión humana en vez de bloqueo automático, y un chequeo real de *drift* sobre los propios datos que encontró algo importante -- la tasa de fraude se multiplica por ~40 (pesos) y ~10 (dólares) durante diciembre respecto del resto del año, un caso real de *concept drift* que quedó incorporado a la propuesta de monitoreo.

### [Extra: XGBoost + Isolation Forest](Extra_XGBoost_IsolationForest.ipynb)
Prueba dos técnicas más habituales en la industria de risk management/fraude que las usadas en el proyecto original: **XGBoost con `scale_pos_weight`** (mejor equipado que `HistGradientBoostingClassifier` para un desbalance tan extremo) y **Isolation Forest** como señal complementaria no supervisada. XGBoost con hiperparámetros ajustados por validación cruzada y umbral ajustado (0,91) se convierte en el **mejor modelo del proyecto** (F1 = 0,651, superando al Random Forest de 0,645). Isolation Forest, usado solo, es mucho más débil (F1 = 0,101) y unir sus alertas binariamente con las de XGBoost empeora la precision -- pero usado como *ranking* de prioridad secundaria sobre lo que XGBoost no marcó, encuentra fraude a una tasa hasta 12 veces mayor que el azar, confirmando su rol de complemento y no de reemplazo.

## Cómo correr los notebooks

Cada notebook está pensado para correr en Google Colab: al ejecutar las primeras celdas, van a pedir subir los archivos de datos que necesitan (`Muestra.del` y `Dr_Muestra.xlsx` para el TP1, TP2, el notebook de mejoras y el de XGBoost + Isolation Forest; `tp2_train.csv`/`tp2_test.csv`, generados por el TP2, para el TP3). El notebook de XGBoost + Isolation Forest necesita además instalar `xgboost` (`!pip install xgboost`), aunque suele venir preinstalado en Colab.
