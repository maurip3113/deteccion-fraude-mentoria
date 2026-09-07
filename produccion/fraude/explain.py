"""Explicacion de cada alerta con valores SHAP.

Un modelo de fraude que solo devuelve una probabilidad le deja al analista el
trabajo entero: le pedimos que revise sin decirle que mirar. Este modulo agrega
la razon -- que variables empujaron la transaccion hacia la alerta y cuanto.

Usa el TreeSHAP que **ya trae XGBoost** (`pred_contribs=True`) en vez del paquete
`shap`. Es el mismo algoritmo y el mismo resultado exacto, pero evita sumarle al
contenedor de produccion una dependencia pesada (numba, llvmlite) solo para
calcular algo que el modelo sabe hacer solo. El paquete `shap` queda para el
analisis global con graficos, donde el peso no importa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

# Nombres legibles para quien revisa la alerta, que no tiene por que conocer el
# esquema del dataset.
ETIQUETAS = {
    "Trx_Importe": "importe de la transaccion",
    "Es_Outlier_Importe": "importe atipico para el conjunto",
    "Hora_Dia": "hora del dia",
    "Es_Fin_de_Semana": "es fin de semana",
    "Tiempo_Entre_Trx_Horas": "horas desde la transaccion anterior",
    "Es_Moneda_Dolar": "operacion en dolares",
    "Cliente_Edad": "edad del cliente",
    "Cliente_Trx_Count": "cantidad de transacciones previas del cliente",
    "Desvio_Importe_Cliente_Abs": "cuanto se aparta el importe de lo habitual del cliente",
    "Rubro_Categoria_TasaFraude": "riesgo historico del rubro del comercio",
    "Presencia_Cliente_Presencial": "transaccion presencial",
}


def etiqueta(columna: str) -> str:
    if columna in ETIQUETAS:
        return ETIQUETAS[columna]
    if columna.startswith("Cliente_Segmento_"):
        return f"segmento {columna.split('_')[-1]} del cliente"
    if columna.startswith("Cliente_EstadoCivil_"):
        return f"estado civil {columna.split('_')[-1].lower()}"
    if columna.startswith("Cliente_Sexo_"):
        return f"sexo {columna.split('_')[-1].lower()}"
    return columna


def contribuciones(modelo, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Devuelve las contribuciones SHAP por fila y el valor base del modelo.

    Las contribuciones son aditivas en el espacio de *log-odds*, no de
    probabilidad: suman exactamente el margen que produce el modelo, y
    `sigmoide(base + suma) == predict_proba`. Por eso no se pueden leer como
    "aporta 12% de probabilidad" -- se leen como cuanto empuja cada variable, en
    la misma escala en la que el modelo decide.
    """
    matriz = xgb.DMatrix(X, feature_names=list(X.columns))
    crudo = modelo.get_booster().predict(matriz, pred_contribs=True)
    return crudo[:, :-1], float(crudo[0, -1])


def explicar(modelo, X: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Las `top_n` variables que mas pesaron en la decision, para una fila."""
    contribs, _ = contribuciones(modelo, X.iloc[[0]])
    fila = contribs[0]

    orden = np.argsort(-np.abs(fila))[:top_n]
    return [
        {
            "variable": etiqueta(X.columns[i]),
            "valor": round(float(X.iloc[0, i]), 4),
            "contribucion": round(float(fila[i]), 4),
            "empuja": "hacia fraude" if fila[i] > 0 else "hacia legitima",
        }
        for i in orden
        if abs(fila[i]) > 1e-6
    ]


def importancia_global(modelo, X: pd.DataFrame) -> pd.DataFrame:
    """Contribucion media absoluta de cada variable sobre un conjunto.

    Es la version agregada de lo mismo: en vez de explicar una alerta, muestra en
    que se apoya el modelo en general. Sirve para auditarlo -- si una sola
    variable concentra casi todo el peso, conviene entender por que antes de
    confiar en el resultado.
    """
    contribs, _ = contribuciones(modelo, X)
    medias = np.abs(contribs).mean(axis=0)
    tabla = pd.DataFrame({
        "variable": X.columns,
        "shap_medio_abs": medias,
        "share": medias / medias.sum(),
    })
    return tabla.sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)
