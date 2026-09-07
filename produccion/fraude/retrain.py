"""Reentrenamiento periodico con decision champion/challenger.

    python -m fraude.retrain

Un modelo de riesgo no se reemplaza porque se reentreno: se reemplaza si el
candidato le gana al que esta en produccion, medido sobre datos que ninguno de
los dos vio. Este script implementa ese contrato:

  champion    = el modelo desplegado hoy
  challenger  = reentrenado incluyendo el lote nuevo de datos etiquetados
  holdout     = periodo posterior a ambos, comun a los dos

La promocion exige una mejora minima (`MARGEN_PROMOCION`) y no solo un numero
mas alto: reemplazar el modelo tiene un costo operativo (revalidacion, aviso al
equipo de fraude, recalibracion de umbrales) que una diferencia de ruido no
justifica.

La particion es **temporal**, no aleatoria. Con una particion aleatoria el
reentrenamiento siempre parece funcionar, porque train y test comparten el mismo
periodo; el punto de reentrenar es justamente responder a que el mundo cambio
con el tiempo, y eso solo se ve evaluando hacia adelante.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from fraude.features import EstadoClientes
from fraude.train import (
    DIR_ARTEFACTOS, ajustar, cargar_historico, evaluar, leer_mapa_rubro,
    proxima_version, referencia_drift, umbral_optimo_f1,
)

MARGEN_PROMOCION = 0.01  # el challenger tiene que ganar por mas que esto en F1


def reentrenar(path_datos: str, path_doc: str, mes_corte_champion: int = 8,
               mes_corte_lote: int = 10, promover: bool = False) -> dict:
    df = cargar_historico(path_datos, path_doc)
    df["Cliente_Edad"] = (pd.Timestamp("2025-01-01") - df["Cliente_FechaNacimiento"]).dt.days // 365
    mes = df["Trx_Fecha"].dt.month

    df_champion = df[mes <= mes_corte_champion]
    df_lote_nuevo = df[(mes > mes_corte_champion) & (mes <= mes_corte_lote)]
    df_holdout = df[mes > mes_corte_lote]
    df_challenger = df[mes <= mes_corte_lote]

    print(f"champion  : meses 1-{mes_corte_champion}   {len(df_champion):,} trx "
          f"({df_champion['Es_Fraude'].mean()*100:.2f}% fraude)")
    print(f"lote nuevo: meses {mes_corte_champion+1}-{mes_corte_lote}  {len(df_lote_nuevo):,} trx "
          f"({df_lote_nuevo['Es_Fraude'].mean()*100:.2f}% fraude)")
    print(f"holdout   : meses {mes_corte_lote+1}-12  {len(df_holdout):,} trx "
          f"({df_holdout['Es_Fraude'].mean()*100:.2f}% fraude)")

    mapa_rubro = leer_mapa_rubro(path_doc)
    resultados = {}
    modelos = {}

    for nombre, df_entrenamiento in [("champion", df_champion), ("challenger", df_challenger)]:
        # Ultimo 20% cronologico como validacion, para el umbral.
        corte = int(len(df_entrenamiento) * 0.8)
        df_ent = df_entrenamiento.sort_values("Trx_Timestamp")
        df_fit, df_val = df_ent.iloc[:corte], df_ent.iloc[corte:]

        modelo, prep, umbral, hiper = ajustar(
            df_fit, df_fit["Es_Fraude"], df_val, df_val["Es_Fraude"], mapa_rubro
        )
        proba = modelo.predict_proba(prep.transform(df_holdout))[:, 1]
        resultados[nombre] = evaluar(df_holdout["Es_Fraude"], proba, umbral)
        resultados[nombre]["umbral"] = umbral
        resultados[nombre]["n_entrenamiento"] = len(df_fit)
        # Techo de referencia: el mejor umbral posible sobre el holdout. No es
        # alcanzable en produccion (usa las etiquetas que todavia no llegaron),
        # pero separa cuanto del error es del ranking del modelo y cuanto es
        # del umbral que quedo desactualizado.
        umbral_oraculo = umbral_optimo_f1(df_holdout["Es_Fraude"], proba)
        resultados[nombre]["f1_con_umbral_oraculo"] = evaluar(
            df_holdout["Es_Fraude"], proba, umbral_oraculo
        )["f1"]
        modelos[nombre] = (modelo, prep, umbral, hiper, df_entrenamiento, df_fit)

    tabla = pd.DataFrame(resultados).T
    print(f"\n{'='*70}\nEvaluacion sobre el mismo holdout (meses {mes_corte_lote+1}-12)\n{'='*70}")
    print(tabla[["precision", "recall", "f1", "auc_pr", "tasa_alertas", "umbral",
                 "f1_con_umbral_oraculo"]].round(4).to_string())

    print("\n-- Cuanto del error es umbral y cuanto es modelo --")
    for nombre in ("champion", "challenger"):
        r = resultados[nombre]
        print(f"   {nombre:11} F1 real {r['f1']:.4f}  ->  con umbral optimo {r['f1_con_umbral_oraculo']:.4f}"
              f"   (pierde {r['f1_con_umbral_oraculo'] - r['f1']:.4f} solo por calibracion)")

    delta_f1 = resultados["challenger"]["f1"] - resultados["champion"]["f1"]
    delta_pr = resultados["challenger"]["auc_pr"] - resultados["champion"]["auc_pr"]
    gana = delta_f1 > MARGEN_PROMOCION

    print(f"\nDelta F1     : {delta_f1:+.4f}  (margen exigido: {MARGEN_PROMOCION:+.4f})")
    print(f"Delta AUC-PR : {delta_pr:+.4f}")
    print(f"\nDECISION: {'PROMOVER el challenger' if gana else 'MANTENER el champion'}")

    decision = {
        "fecha": datetime.now(timezone.utc).isoformat(),
        "particion": "temporal",
        "corte_champion": f"meses 1-{mes_corte_champion}",
        "lote_nuevo": f"meses {mes_corte_champion+1}-{mes_corte_lote}",
        "holdout": f"meses {mes_corte_lote+1}-12",
        "metricas": resultados,
        "delta_f1": round(delta_f1, 5),
        "delta_auc_pr": round(delta_pr, 5),
        "margen_exigido": MARGEN_PROMOCION,
        "promovido": bool(gana and promover),
        "recomendacion": "promover" if gana else "mantener",
    }

    if gana and promover:
        modelo, prep, umbral, hiper, df_completo, df_fit = modelos["challenger"]
        version = proxima_version()
        joblib.dump({
            "modelo": modelo, "preprocesador": prep, "umbral": umbral, "version": version,
            "estado_clientes": EstadoClientes.desde_historico(df_completo),
            "referencia_drift": referencia_drift(prep.transform(df_fit)),
        }, DIR_ARTEFACTOS / f"modelo_v{version}.joblib", compress=3)
        (DIR_ARTEFACTOS / "modelo_actual.txt").write_text(str(version), encoding="utf-8")
        decision["version_promovida"] = version
        print(f"Promovido como modelo_v{version}.joblib")
    elif gana:
        print("(corre con --promover para que reemplace al modelo en produccion)")

    log = DIR_ARTEFACTOS / "log_reentrenamientos.jsonl"
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision, ensure_ascii=False) + "\n")
    print(f"Decision registrada en {log.name}")
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reentrena y decide si reemplazar el modelo.")
    parser.add_argument("--datos", default="../Muestra/Muestra.del")
    parser.add_argument("--doc", default="../Dr_Muestra.xlsx")
    parser.add_argument("--promover", action="store_true",
                        help="Si el challenger gana, lo deja como modelo en produccion")
    args = parser.parse_args()
    reentrenar(args.datos, args.doc, promover=args.promover)
