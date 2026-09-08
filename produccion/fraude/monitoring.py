"""Monitoreo de un modelo ya desplegado: drift de datos y degradacion de performance.

    python -m fraude.monitoring --demo

Un modelo de fraude no se rompe con un error, se rompe en silencio: sigue
devolviendo probabilidades razonables mientras el mundo que modelaba cambia
debajo. Este modulo mide ese cambio de tres formas complementarias:

1. **Data drift** -- las variables de entrada se corrieron respecto de train
   (PSI y test de Kolmogorov-Smirnov por variable).
2. **Prediction drift** -- la distribucion de scores y la tasa de alertas se
   movieron, aunque las entradas parezcan iguales.
3. **Degradacion de performance** -- solo se puede medir cuando llegan las
   etiquetas reales, que en fraude tardan semanas (contracargos, reclamos).
   Hasta entonces, 1 y 2 son la unica senal disponible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from fraude.api import cargar_bundle
from fraude.features import VARIABLES_MONOTONAS, cargar_historico
from fraude.train import evaluar

UMBRAL_PSI_MODERADO = 0.10
UMBRAL_PSI_SEVERO = 0.25

# Estados que no cuentan como drift: la variable esta quieta, o se movio por una
# razon que ya conocemos y que no dice nada sobre la salud del modelo.
ESTADOS_SIN_DRIFT = ("estable", "estructural")


def psi(proporciones_ref: np.ndarray, proporciones_nuevas: np.ndarray) -> float:
    """Population Stability Index entre la referencia de train y un batch nuevo."""
    # Piso para no dividir por cero cuando un bin queda vacio.
    ref = np.clip(proporciones_ref, 1e-6, None)
    nuevo = np.clip(proporciones_nuevas, 1e-6, None)
    return float(np.sum((nuevo - ref) * np.log(nuevo / ref)))


def proporciones_batch(valores: np.ndarray, ref: dict) -> np.ndarray:
    if ref["tipo"] == "categorica":
        return np.array([float((valores == v).mean()) for v in ref["categorias"]])
    conteos, _ = np.histogram(valores, bins=np.array(ref["bordes"]))
    return conteos / max(len(valores), 1)


def clasificar(valor_psi: float) -> str:
    if valor_psi >= UMBRAL_PSI_SEVERO:
        return "SEVERO"
    if valor_psi >= UMBRAL_PSI_MODERADO:
        return "MODERADO"
    return "estable"


def hay_alarma(tabla: pd.DataFrame, factor_alertas: float,
               factor_limite: float = 2.0) -> bool:
    """Cuando el monitoreo pide intervencion.

    Tres disparadores independientes. El primero existe porque el corte 0,25 del
    PSI significa exactamente eso: una sola variable ahi ya amerita mirar, y
    exigir tres a la vez desactivaba en la practica el umbral mas importante.
    """
    severas = (tabla["estado"] == "SEVERO").sum()
    con_drift = (~tabla["estado"].isin(ESTADOS_SIN_DRIFT)).sum()
    return bool(severas >= 1 or con_drift >= 3 or factor_alertas >= factor_limite)


def drift_de_datos(X_nuevo: pd.DataFrame, referencia: dict, X_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    filas = []
    for col, ref in referencia.items():
        if col not in X_nuevo.columns:
            continue
        valores = X_nuevo[col].to_numpy()
        valor_psi = psi(np.array(ref["proporciones"]), proporciones_batch(valores, ref))
        fila = {
            "variable": col,
            "tipo": ref["tipo"],
            "psi": round(valor_psi, 4),
            # El PSI se calcula igual --una caida repentina de un contador si
            # significaria algo-- pero no cuenta como drift ni dispara alarma.
            "estado": "estructural" if col in VARIABLES_MONOTONAS else clasificar(valor_psi),
            "media_train": round(ref["media"], 4),
            "media_batch": round(float(valores.mean()), 4),
        }
        if X_ref is not None:
            fila["ks_pvalue"] = round(float(ks_2samp(X_ref[col].to_numpy(), valores).pvalue), 6)
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)


def drift_de_predicciones(proba_ref: np.ndarray, proba_nuevo: np.ndarray, umbral: float) -> dict:
    return {
        "tasa_alertas_referencia": round(float((proba_ref >= umbral).mean()), 5),
        "tasa_alertas_batch": round(float((proba_nuevo >= umbral).mean()), 5),
        "score_medio_referencia": round(float(proba_ref.mean()), 5),
        "score_medio_batch": round(float(proba_nuevo.mean()), 5),
        "ks_pvalue": round(float(ks_2samp(proba_ref, proba_nuevo).pvalue), 8),
    }


def reporte(X_ref, proba_ref, X_nuevo, proba_nuevo, bundle, y_nuevo=None, etiqueta="batch") -> dict:
    tabla = drift_de_datos(X_nuevo, bundle["referencia_drift"], X_ref)
    predicciones = drift_de_predicciones(proba_ref, proba_nuevo, bundle["umbral"])

    print(f"\n{'='*74}\nREPORTE DE MONITOREO - {etiqueta}  (modelo v{bundle['version']}, "
          f"{len(X_nuevo):,} transacciones)\n{'='*74}")
    print("\n-- Drift de datos (PSI por variable, top 8) --")
    print(tabla.head(8).to_string(index=False))

    con_drift = tabla[~tabla["estado"].isin(ESTADOS_SIN_DRIFT)]
    estructurales = tabla[tabla["estado"] == "estructural"]
    print(f"\n   {len(con_drift)} de {len(tabla)} variables con drift detectable "
          f"(PSI >= {UMBRAL_PSI_MODERADO}).")
    if len(estructurales):
        print(f"   {len(estructurales)} excluida(s) por crecer con el calendario: "
              f"{', '.join(estructurales['variable'])}.")

    print("\n-- Drift de predicciones --")
    for k, v in predicciones.items():
        print(f"   {k:32} {v}")
    factor = predicciones["tasa_alertas_batch"] / max(predicciones["tasa_alertas_referencia"], 1e-9)
    print(f"   {'factor de alertas vs referencia':32} x{factor:.2f}")

    resultado = {
        "etiqueta": etiqueta,
        "modelo_version": bundle["version"],
        "n_transacciones": len(X_nuevo),
        "drift_datos": tabla.to_dict("records"),
        "variables_con_drift": int((~tabla["estado"].isin(ESTADOS_SIN_DRIFT)).sum()),
        "variables_estructurales": int((tabla["estado"] == "estructural").sum()),
        "drift_predicciones": predicciones,
    }

    if y_nuevo is not None:
        performance = evaluar(y_nuevo, proba_nuevo, bundle["umbral"])
        print("\n-- Performance con etiquetas reales --")
        for k, v in performance.items():
            print(f"   {k:32} {v:.4f}")
        print(f"   {'tasa de fraude observada':32} {y_nuevo.mean():.4%}")
        resultado["performance"] = performance
        resultado["tasa_fraude_observada"] = float(y_nuevo.mean())

    alarma = hay_alarma(tabla, factor)
    resultado["requiere_atencion"] = bool(alarma)
    print(f"\n>> {'ALARMA: amerita revision y posible reentrenamiento.' if alarma else 'Sin alarmas.'}")
    return resultado


def demo(path_datos: str, path_doc: str) -> dict:
    """Compara diciembre contra el resto del anio.

    La tasa de fraude se multiplica por ~40 en diciembre, asi que sirve como
    prueba de que el monitoreo levanta un cambio grande. Si ese salto es un
    episodio real de fraude o una consecuencia de como se armo la muestra es algo
    que el dataset no permite decidir (ver seccion 5.3 del notebook de mejoras) --
    para el monitoreo da lo mismo: en produccion tampoco se sabe la causa al
    momento de la alerta, y la respuesta operativa es la misma.
    """
    bundle = cargar_bundle()
    prep = bundle["preprocesador"]

    df = cargar_historico(path_datos, path_doc)
    df["Cliente_Edad"] = (pd.Timestamp("2025-01-01") - df["Cliente_FechaNacimiento"]).dt.days // 365

    es_diciembre = df["Trx_Fecha"].dt.month == 12
    df_ref, df_dic = df[~es_diciembre], df[es_diciembre]

    X_ref, X_dic = prep.transform(df_ref), prep.transform(df_dic)
    proba_ref = bundle["modelo"].predict_proba(X_ref)[:, 1]
    proba_dic = bundle["modelo"].predict_proba(X_dic)[:, 1]

    return reporte(X_ref, proba_ref, X_dic, proba_dic, bundle,
                   y_nuevo=df_dic["Es_Fraude"], etiqueta="diciembre 2025 vs. resto del año")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitorea drift del modelo en produccion.")
    parser.add_argument("--demo", action="store_true", help="Compara diciembre contra el resto del anio")
    parser.add_argument("--datos", default="../Muestra/Muestra.del")
    parser.add_argument("--doc", default="../Dr_Muestra.xlsx")
    parser.add_argument("--salida", default="artifacts/reporte_monitoreo.json")
    args = parser.parse_args()

    resultado = demo(args.datos, args.doc)
    Path(args.salida).write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte guardado en {args.salida}")
