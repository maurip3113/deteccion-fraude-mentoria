"""Cuanto vale el numero, y donde deja de valer.

    python -m fraude.evaluacion

Dos analisis que un F1 puntual no puede dar:

**El intervalo de confianza.** Un F1 de 0,599 sobre 239 fraudes no es un numero
exacto. Remuestreando el conjunto de evaluacion se ve cuanto se mueve, y eso
define el minimo de diferencia que dos modelos necesitan para ser realmente
distinguibles. Sin esa banda, cualquier ranking de modelos es superstición: se
elige el mejor de una lista donde las diferencias son ruido.

**La evaluacion por segmento.** Un F1 global es un promedio, y los promedios
esconden. El mismo modelo puede ser util en un canal e inservible en otro, y eso
decide donde conviene confiar en el y donde hace falta otra cosa.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from fraude.api import cargar_bundle
from fraude.features import cargar_historico
from fraude.train import SEMILLA

DIR_ARTEFACTOS = Path(__file__).resolve().parent.parent / "artifacts"


def intervalo_f1(y_true: np.ndarray, y_pred: np.ndarray,
                 n_remuestreos: int = 2000, semilla: int = 0) -> dict:
    """Banda de incertidumbre del F1 por bootstrap sobre el conjunto de evaluacion."""
    rng = np.random.default_rng(semilla)
    n = len(y_true)
    valores = []
    for _ in range(n_remuestreos):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0:
            continue  # remuestreo sin fraudes: F1 indefinido
        valores.append(f1_score(y_true[idx], y_pred[idx]))

    v = np.array(valores)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {
        "f1_puntual": round(float(f1_score(y_true, y_pred)), 4),
        "ic95_bajo": round(float(lo), 4),
        "ic95_alto": round(float(hi), 4),
        "desvio": round(float(v.std()), 4),
        # Dos modelos mas cerca que esto no se distinguen con estos datos.
        "diferencia_minima_distinguible": round(float(2 * v.std()), 4),
        "n_fraudes": int(y_true.sum()),
        "n_transacciones": int(n),
    }


def definir_segmentos(df: pd.DataFrame) -> pd.DataFrame:
    """Cortes por los que tiene sentido preguntarse si el modelo sirve."""
    seg = pd.DataFrame(index=df.index)
    seg["canal"] = df["Presencia_Cliente"]
    seg["moneda"] = np.where(df["Es_Moneda_Dolar"] == 1, "dolares", "pesos")
    seg["importe"] = pd.qcut(df["Trx_Importe"], 4,
                             labels=["bajo", "medio-bajo", "medio-alto", "alto"])
    seg["historial"] = pd.cut(df["Cliente_Trx_Count"], [-1, 50, 200, np.inf],
                              labels=["<50 trx", "50-200", ">200"])
    seg["franja"] = pd.cut(df["Hora_Dia"], [-1, 5, 11, 17, 23],
                           labels=["madrugada", "manana", "tarde", "noche"])
    return seg


def por_segmento(df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    filas = []
    segmentos = definir_segmentos(df)
    for corte in segmentos.columns:
        for valor, indices in segmentos.groupby(corte, observed=True).groups.items():
            yt = y_true.loc[indices].to_numpy()
            yp = y_pred[df.index.get_indexer(indices)]
            if yt.sum() == 0:
                continue
            filas.append({
                "corte": corte,
                "segmento": str(valor),
                "trx": len(yt),
                "fraudes": int(yt.sum()),
                "precision": round(float(precision_score(yt, yp, zero_division=0)), 3),
                "recall": round(float(recall_score(yt, yp)), 3),
                "f1": round(float(f1_score(yt, yp)), 3),
            })
    return pd.DataFrame(filas).sort_values("f1", ascending=False).reset_index(drop=True)


def analizar(path_datos: str, path_doc: str) -> dict:
    bundle = cargar_bundle()
    prep, modelo, umbral = bundle["preprocesador"], bundle["modelo"], bundle["umbral"]

    df = cargar_historico(path_datos, path_doc)
    df["Cliente_Edad"] = (pd.Timestamp("2025-01-01") - df["Cliente_FechaNacimiento"]).dt.days // 365
    y = df["Es_Fraude"]
    _, idx_test = train_test_split(df.index, test_size=0.2, random_state=SEMILLA, stratify=y)

    test = df.loc[idx_test]
    proba = modelo.predict_proba(prep.transform(test))[:, 1]
    pred = (proba >= umbral).astype(int)
    y_test = test["Es_Fraude"]

    ic = intervalo_f1(y_test.to_numpy(), pred)
    print(f"{'='*72}\nCUANTO VALE EL NUMERO\n{'='*72}")
    print(f"  F1 puntual .................... {ic['f1_puntual']}")
    print(f"  IC 95% (bootstrap) ............ [{ic['ic95_bajo']}, {ic['ic95_alto']}]")
    print(f"  Desvio estandar ............... {ic['desvio']}")
    print(f"  Sobre {ic['n_fraudes']} fraudes en {ic['n_transacciones']:,} transacciones")
    print(f"\n  => Dos modelos que difieran menos de {ic['diferencia_minima_distinguible']} "
          f"en F1 son indistinguibles con estos datos.")

    tabla = por_segmento(test, y_test, pred)
    print(f"\n{'='*72}\nDONDE FUNCIONA Y DONDE NO\n{'='*72}")
    for corte, grupo in tabla.groupby("corte", sort=False):
        print(f"\n-- por {corte} --")
        print(grupo[["segmento", "trx", "fraudes", "precision", "recall", "f1"]]
              .to_string(index=False))

    mejor, peor = tabla.iloc[0], tabla.iloc[-1]
    print(f"\n  Mejor segmento: {mejor['segmento']} (F1 {mejor['f1']}, {mejor['fraudes']} fraudes)")
    print(f"  Peor segmento : {peor['segmento']} (F1 {peor['f1']}, {peor['fraudes']} fraudes)")
    print("\n  Los segmentos con pocos fraudes son aun mas ruidosos que el total:")
    print("  leerlos como direccion, no como medicion.")

    resultado = {"intervalo_f1": ic, "segmentos": tabla.to_dict("records")}
    salida = DIR_ARTEFACTOS / "evaluacion_detallada.json"
    salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida.name}")
    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intervalo de confianza del F1 y evaluacion por segmento.")
    parser.add_argument("--datos", default="../Muestra/Muestra.del")
    parser.add_argument("--doc", default="../Dr_Muestra.xlsx")
    args = parser.parse_args()
    analizar(args.datos, args.doc)
