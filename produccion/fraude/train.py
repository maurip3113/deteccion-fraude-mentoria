"""Entrena, valida y serializa el modelo de fraude como un artefacto versionado.

    python -m fraude.train --datos ../Muestra/Muestra.del --doc ../Dr_Muestra.xlsx

Diferencia importante con el notebook: aca el umbral de decision se elige sobre
un conjunto de *validacion* separado, no sobre el de evaluacion. En el notebook
se elegia mirando test, lo que infla la metrica reportada porque el umbral ya
vio los datos con los que despues se lo mide. En produccion eso no se puede
hacer: el umbral es un hiperparametro mas y tiene que salir de datos que el
reporte final no use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from fraude.features import EstadoClientes, PreprocesadorFraude, cargar_historico

DIR_ARTEFACTOS = Path(__file__).resolve().parent.parent / "artifacts"
SEMILLA = 42


def entorno() -> dict:
    """Versiones con las que se genero el artefacto.

    joblib serializa referencias a las clases de la libreria que creo el modelo:
    cargarlo con otra version de scikit-learn o xgboost falla, o corre con un
    comportamiento distinto al que se valido. Guardarlas permite comparar al
    levantar el servicio en vez de descubrirlo en produccion.
    """
    import sklearn
    import xgboost
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def evaluar(y_true, y_proba, umbral: float) -> dict:
    y_pred = (y_proba >= umbral).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "tasa_alertas": float(y_pred.mean()),
    }


def umbral_optimo_f1(y_true, y_proba) -> float:
    precisions, recalls, umbrales = precision_recall_curve(y_true, y_proba)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_vals = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
    return float(umbrales[np.argmax(np.nan_to_num(f1_vals))])


def referencia_drift(X_train: pd.DataFrame, n_bins: int = 10) -> dict:
    """Congela la distribucion de entrenamiento para poder medir drift despues.

    Las variables continuas se resumen en bins por cuantiles; las binarias y
    categoricas, por la proporcion de cada valor. Sin esa distincion las dummies
    (16 de las 23 features) quedarian sin monitorear, porque los cuantiles de una
    variable 0/1 colapsan en un solo borde.
    """
    referencia = {}
    for col in X_train.columns:
        valores = X_train[col].to_numpy()
        distintos = np.unique(valores)

        if len(distintos) <= n_bins:
            referencia[col] = {
                "tipo": "categorica",
                "categorias": distintos.tolist(),
                "proporciones": [float((valores == v).mean()) for v in distintos],
                "media": float(valores.mean()),
            }
            continue

        bordes = np.unique(np.quantile(valores, np.linspace(0, 1, n_bins + 1)))
        if len(bordes) < 3:
            continue  # variable casi constante: PSI no aporta nada
        bordes[0], bordes[-1] = -np.inf, np.inf
        proporciones, _ = np.histogram(valores, bins=bordes)
        referencia[col] = {
            "tipo": "continua",
            "bordes": bordes.tolist(),
            "proporciones": (proporciones / len(valores)).tolist(),
            "media": float(valores.mean()),
        }
    return referencia


def ajustar(df_fit: pd.DataFrame, y_fit: pd.Series, df_val: pd.DataFrame, y_val: pd.Series,
            mapa_rubro_codigo: dict) -> tuple[XGBClassifier, PreprocesadorFraude, float, dict]:
    """Ajusta preprocesador + modelo + umbral. Reutilizado por el reentrenamiento."""
    prep = PreprocesadorFraude().fit(df_fit, y_fit)
    prep.mapa_rubro_codigo = mapa_rubro_codigo

    hiperparametros = {
        "scale_pos_weight": float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1)),
        "random_state": SEMILLA,
        "eval_metric": "aucpr",
        "n_jobs": -1,
    }
    modelo = XGBClassifier(**hiperparametros)
    modelo.fit(prep.transform(df_fit), y_fit)

    umbral = umbral_optimo_f1(y_val, modelo.predict_proba(prep.transform(df_val))[:, 1])
    return modelo, prep, umbral, hiperparametros


def leer_mapa_rubro(path_doc: str) -> dict[int, str]:
    df_rubro = pd.read_excel(path_doc, sheet_name="RubroRed")
    col_cat = [c for c in df_rubro.columns if c.startswith("Categor")][0]
    return {int(k): str(v) for k, v in zip(df_rubro["Codigo"], df_rubro[col_cat]) if pd.notna(k)}


def proxima_version() -> int:
    existentes = [int(p.stem.split("_v")[-1]) for p in DIR_ARTEFACTOS.glob("modelo_v*.joblib")]
    return max(existentes, default=0) + 1


def entrenar(path_datos: str, path_doc: str, version: int | None = None) -> dict:
    print("Reconstruyendo el historico (TP1 + TP2)...")
    df = cargar_historico(path_datos, path_doc)
    df["Cliente_Edad"] = (
        pd.Timestamp("2025-01-01") - df["Cliente_FechaNacimiento"]
    ).dt.days // 365
    y = df["Es_Fraude"]
    print(f"  {len(df):,} transacciones | tasa de fraude {y.mean()*100:.2f}%")

    # Evaluacion apartada de entrada: no participa de nada mas que del reporte final.
    idx_train_full, idx_test = train_test_split(
        df.index, test_size=0.2, random_state=SEMILLA, stratify=y
    )
    # Validacion, para elegir el umbral sin contaminar la evaluacion.
    idx_train, idx_val = train_test_split(
        idx_train_full, test_size=0.2, random_state=SEMILLA, stratify=y.loc[idx_train_full]
    )

    df_train, df_val, df_test = df.loc[idx_train], df.loc[idx_val], df.loc[idx_test]
    y_train, y_val, y_test = y.loc[idx_train], y.loc[idx_val], y.loc[idx_test]
    print(f"  train {len(df_train):,} | validacion {len(df_val):,} | evaluacion {len(df_test):,}")

    print("Entrenando XGBoost...")
    modelo, prep, umbral, hiperparametros = ajustar(
        df_train, y_train, df_val, y_val, leer_mapa_rubro(path_doc)
    )
    X_train, X_val, X_test = prep.transform(df_train), prep.transform(df_val), prep.transform(df_test)
    proba_test = modelo.predict_proba(X_test)[:, 1]
    metricas = {
        "validacion": evaluar(y_val, modelo.predict_proba(X_val)[:, 1], umbral),
        "evaluacion": evaluar(y_test, proba_test, umbral),
        "evaluacion_umbral_05": evaluar(y_test, proba_test, 0.5),
    }
    print(f"  umbral elegido en validacion: {umbral:.4f}")
    print(f"  F1 en evaluacion: {metricas['evaluacion']['f1']:.4f} | "
          f"AUC-PR: {metricas['evaluacion']['auc_pr']:.4f}")

    version = version or proxima_version()
    DIR_ARTEFACTOS.mkdir(exist_ok=True)

    bundle = {
        "modelo": modelo,
        "preprocesador": prep,
        "umbral": umbral,
        "version": version,
        "entorno": entorno(),
        "estado_clientes": EstadoClientes.desde_historico(df),
        "referencia_drift": referencia_drift(X_train),
    }
    ruta_modelo = DIR_ARTEFACTOS / f"modelo_v{version}.joblib"
    joblib.dump(bundle, ruta_modelo, compress=3)

    metadata = {
        "version": version,
        "algoritmo": "XGBClassifier",
        "entrenado_en": datetime.now(timezone.utc).isoformat(),
        "hiperparametros": hiperparametros,
        "umbral_decision": umbral,
        "umbral_elegido_en": "validacion",
        "n_features": len(prep.columnas_modelo),
        "features": prep.columnas_modelo,
        "filas": {"train": len(df_train), "validacion": len(df_val), "evaluacion": len(df_test)},
        "tasa_fraude_train": float(y_train.mean()),
        "metricas": metricas,
        "entorno": entorno(),
        "hash_datos": hashlib.sha256(Path(path_datos).read_bytes()).hexdigest()[:16],
    }
    (DIR_ARTEFACTOS / f"modelo_v{version}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (DIR_ARTEFACTOS / "modelo_actual.txt").write_text(str(version), encoding="utf-8")

    print(f"Artefacto guardado: {ruta_modelo.name} "
          f"({ruta_modelo.stat().st_size / 1e6:.1f} MB)")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo de deteccion de fraude.")
    parser.add_argument("--datos", default="../Muestra/Muestra.del")
    parser.add_argument("--doc", default="../Dr_Muestra.xlsx")
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args()
    entrenar(args.datos, args.doc, args.version)
