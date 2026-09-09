"""Corre el lazo completo sobre el tiempo simulado y verifica el monitoreo.

    python -m simulacion.orquestador

Mes a mes: puntua el periodo, mide drift contra la referencia del modelo vigente,
y si salta la alarma intenta reentrenar. El challenger solo puede usar datos cuya
etiqueta ya habria llegado --el retraso esta modelado explicitamente-- porque el
punto entero del ejercicio es respetar esa restriccion.

Al final compara lo detectado contra `escenarios.py`, que es la verdad declarada:
cuantos escenarios encontro, con cuanto retraso, y cuantas veces se alarmo en
meses donde no pasaba nada.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraude.api import cargar_bundle
from fraude.features import cargar_historico
from fraude.monitoring import (
    ESTADOS_SIN_DRIFT, disparadores, drift_de_datos, drift_de_predicciones,
)
from fraude.train import ajustar, evaluar, leer_mapa_rubro, referencia_drift, umbral_optimo_f1
from simulacion import escenarios as esc

DIR_ARTEFACTOS = Path(__file__).resolve().parent.parent / "artifacts"

LAG_ETIQUETAS_MESES = 2      # cuanto tarda en confirmarse un fraude
MARGEN_PROMOCION = 0.01      # cuanto tiene que ganar el challenger para reemplazar
FACTOR_ALERTAS_ALARMA = 2.0  # multiplicador de alertas que dispara la alarma


def meses_antes(periodo: str, n: int) -> str:
    ts = pd.Timestamp(f"{periodo}-01") - pd.DateOffset(months=n)
    return ts.strftime("%Y-%m")


class Modelo:
    """El modelo vigente, con lo que necesita para monitorearse a si mismo."""

    def __init__(self, bundle: dict, version: int, entrenado_hasta: str):
        self.modelo = bundle["modelo"]
        self.prep = bundle["preprocesador"]
        self.umbral = bundle["umbral"]
        self.referencia = bundle["referencia_drift"]
        self.version = version
        self.entrenado_hasta = entrenado_hasta
        self.X_ref: pd.DataFrame | None = None
        self.proba_ref: np.ndarray | None = None

    def fijar_referencia(self, df_ref: pd.DataFrame) -> None:
        self.X_ref = self.prep.transform(df_ref)
        self.proba_ref = self.modelo.predict_proba(self.X_ref)[:, 1]

    def puntuar(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        X = self.prep.transform(df)
        return X, self.modelo.predict_proba(X)[:, 1]


def intentar_reentrenar(campeon: Modelo, historia: pd.DataFrame, periodo: str,
                        mapa_rubro: dict) -> dict:
    """Champion vs challenger con los datos cuya etiqueta ya llego."""
    corte = meses_antes(periodo, LAG_ETIQUETAS_MESES)
    disponible = historia[historia["periodo"] <= corte]
    if disponible["Es_Fraude"].sum() < 50:
        return {"resultado": "sin datos suficientes", "corte_etiquetas": corte}

    # El ultimo mes etiquetado queda como holdout: ninguno de los dos lo entreno.
    ultimo = disponible["periodo"].max()
    entrenamiento = disponible[disponible["periodo"] < ultimo]
    holdout = disponible[disponible["periodo"] == ultimo]
    if holdout["Es_Fraude"].sum() < 5 or entrenamiento["Es_Fraude"].sum() < 50:
        return {"resultado": "sin datos suficientes", "corte_etiquetas": corte}

    corte_val = int(len(entrenamiento) * 0.8)
    ordenado = entrenamiento.sort_values("Trx_Timestamp")
    df_fit, df_val = ordenado.iloc[:corte_val], ordenado.iloc[corte_val:]

    modelo_c, prep_c, umbral_c, _ = ajustar(
        df_fit, df_fit["Es_Fraude"], df_val, df_val["Es_Fraude"], mapa_rubro
    )

    y_hold = holdout["Es_Fraude"]
    m_challenger = evaluar(y_hold, modelo_c.predict_proba(prep_c.transform(holdout))[:, 1], umbral_c)
    m_campeon = evaluar(y_hold, campeon.puntuar(holdout)[1], campeon.umbral)
    delta = m_challenger["f1"] - m_campeon["f1"]

    return {
        "resultado": "promover" if delta > MARGEN_PROMOCION else "mantener",
        "corte_etiquetas": corte,
        "holdout": ultimo,
        "f1_campeon": round(m_campeon["f1"], 4),
        "f1_challenger": round(m_challenger["f1"], 4),
        "delta_f1": round(delta, 4),
        "_bundle": {
            "modelo": modelo_c, "preprocesador": prep_c, "umbral": umbral_c,
            "referencia_drift": referencia_drift(prep_c.transform(df_fit)),
        } if delta > MARGEN_PROMOCION else None,
    }


def correr(path_datos: str, path_doc: str, reentrenar: bool = True) -> dict:
    print("Preparando la simulacion...")
    real = cargar_historico(path_datos, path_doc)
    real["Cliente_Edad"] = (pd.Timestamp("2025-01-01") - real["Cliente_FechaNacimiento"]).dt.days // 365
    real["periodo"] = real["Trx_Fecha"].dt.strftime("%Y-%m")

    sim = pd.read_parquet(DIR_ARTEFACTOS / "transacciones_simuladas.parquet")
    sim["Cliente_Edad"] = (pd.Timestamp("2025-01-01") - sim["Cliente_FechaNacimiento"]).dt.days // 365
    mapa_rubro = leer_mapa_rubro(path_doc)

    campeon = Modelo(cargar_bundle(), version=1, entrenado_hasta="2025-12")
    campeon.fijar_referencia(real)
    print(f"Modelo vigente v{campeon.version} | umbral {campeon.umbral:.4f}")
    print(f"Retraso de etiquetas: {LAG_ETIQUETAS_MESES} meses\n")

    tasa_ref = float((campeon.proba_ref >= campeon.umbral).mean())
    historia = real.copy()
    linea, promociones = [], 0

    print(f"{'periodo':>9} {'trx':>7} {'alertas':>8} {'factor':>7} {'PSI max':>8} "
          f"{'drift':>6} {'alarma':>7}  escenario")
    print("-" * 92)

    for periodo in esc.PERIODOS_SIMULADOS:
        mes = sim[sim["periodo"] == periodo]
        if mes.empty:
            continue

        X_mes, proba = campeon.puntuar(mes)
        tabla = drift_de_datos(X_mes, campeon.referencia, campeon.X_ref)
        pred = drift_de_predicciones(campeon.proba_ref, proba, campeon.umbral)
        con_drift = int((~tabla["estado"].isin(ESTADOS_SIN_DRIFT)).sum())
        factor = pred["tasa_alertas_batch"] / max(tasa_ref, 1e-9)
        gatillos = disparadores(tabla, factor, FACTOR_ALERTAS_ALARMA)
        alarma = bool(gatillos)

        peor = tabla[~tabla["estado"].isin(ESTADOS_SIN_DRIFT)]
        psi_max = float(peor["psi"].max()) if len(peor) else 0.0
        var_peor = peor.iloc[0]["variable"] if len(peor) else None

        activos = [e.nombre for e in esc.escenarios_activos(periodo)]
        registro = {
            "periodo": periodo,
            "n_transacciones": int(len(mes)),
            "tasa_alertas": pred["tasa_alertas_batch"],
            "factor_alertas": round(factor, 2),
            "psi_max": round(psi_max, 4),
            "variable_psi_max": var_peor,
            "variables_con_drift": con_drift,
            "alarma": bool(alarma),
            "disparadores": gatillos,
            "variables_severas": tabla.loc[tabla["estado"] == "SEVERO", "variable"].tolist(),
            "variables_en_drift": tabla.loc[
                ~tabla["estado"].isin(ESTADOS_SIN_DRIFT), "variable"].tolist(),
            "modelo_version": campeon.version,
            "escenarios_activos": activos,
            "tasa_fraude_real": round(float(mes["Es_Fraude"].mean()), 5),
        }

        # La performance solo se puede medir cuando la etiqueta ya llego.
        if pd.Timestamp(f"{periodo}-01") <= pd.Timestamp(f"{meses_antes(esc.PERIODOS_SIMULADOS[-1], LAG_ETIQUETAS_MESES)}-01"):
            registro["performance"] = {
                k: round(v, 4) for k, v in
                evaluar(mes["Es_Fraude"], proba, campeon.umbral).items()
            }

        historia = pd.concat([historia, mes], ignore_index=True)

        if alarma and reentrenar:
            decision = intentar_reentrenar(campeon, historia, periodo, mapa_rubro)
            nuevo = decision.pop("_bundle", None)
            registro["reentrenamiento"] = decision
            if nuevo:
                promociones += 1
                campeon = Modelo(nuevo, campeon.version + 1, periodo)
                campeon.fijar_referencia(historia[historia["periodo"] <= decision["corte_etiquetas"]])
                tasa_ref = float((campeon.proba_ref >= campeon.umbral).mean())
                registro["promovido_a"] = campeon.version

        marca = "SI" if alarma else "-"
        extra = ""
        if "reentrenamiento" in registro:
            extra = f"  -> {registro['reentrenamiento']['resultado']}"
            if "promovido_a" in registro:
                extra += f" (v{registro['promovido_a']})"
        print(f"{periodo:>9} {len(mes):>7,} {pred['tasa_alertas_batch']:>8.4f} "
              f"{factor:>7.1f} {psi_max:>8.4f} {con_drift:>6} {marca:>7}  "
              f"{', '.join(activos) or '-'}{extra}")
        linea.append(registro)

    verificacion = verificar(linea)
    resultado = {
        "lag_etiquetas_meses": LAG_ETIQUETAS_MESES,
        "factor_alarma": FACTOR_ALERTAS_ALARMA,
        "promociones": promociones,
        "escenarios_declarados": [
            {"nombre": e.nombre, "desde": e.desde, "hasta": e.hasta,
             "descripcion": e.descripcion} for e in esc.ESCENARIOS
        ],
        "periodos": linea,
        "verificacion": verificacion,
    }
    salida = DIR_ARTEFACTOS / "simulacion_linea_tiempo.json"
    salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    imprimir_verificacion(verificacion)
    print(f"\nLinea de tiempo guardada en {salida.name}")
    return resultado


CAIDA_F1_RELATIVA = 0.30  # cuanto tiene que caer el F1 para llamarlo degradacion


def atribuible(registro: dict, e: esc.Escenario) -> bool:
    """¿La alarma de este mes se explica por *este* escenario?

    Que una alarma caiga dentro de la ventana de un escenario no significa que la
    haya causado: con dos cambios superpuestos, el gatillo puede pertenecer al
    otro. Se exige que el disparador corresponda a la firma declarada.
    """
    afectadas = set(e.variables_afectadas)

    if "psi_severo" in registro["disparadores"]:
        if afectadas & set(registro.get("variables_severas", [])):
            return True
    if "multiples_moderadas" in registro["disparadores"]:
        if afectadas & set(registro.get("variables_en_drift", [])):
            return True
    if "factor_alertas" in registro["disparadores"] and e.espera_factor_alertas:
        return True
    return False


def degradacion(meses: list[str], por_periodo: dict) -> dict | None:
    """Caida de F1 dentro de la ventana respecto de los tres meses previos.

    Es la senal que sí ve el concept drift, y la que solo aparece cuando llegan
    las etiquetas: semanas o meses despues de que el problema empezo.
    """
    idx = esc.PERIODOS_SIMULADOS.index(meses[0])
    previos = [p for p in esc.PERIODOS_SIMULADOS[max(0, idx - 3):idx]
               if por_periodo.get(p, {}).get("performance")]
    dentro = [p for p in meses if por_periodo.get(p, {}).get("performance")]
    if not previos or not dentro:
        return None

    base = sorted(por_periodo[p]["performance"]["f1"] for p in previos)[len(previos) // 2]
    peor = min(por_periodo[p]["performance"]["f1"] for p in dentro)
    if base <= 0:
        return None
    caida = (base - peor) / base
    return {
        "f1_previo": round(base, 4),
        "f1_minimo_en_ventana": round(peor, 4),
        "caida_relativa": round(caida, 4),
        "degradado": bool(caida >= CAIDA_F1_RELATIVA),
    }


def verificar(linea: list[dict]) -> dict:
    """Contrasta lo detectado contra lo declarado, atribuyendo causa.

    La version anterior contaba como acierto cualquier alarma dentro de la
    ventana de un escenario. Con la campania presencial y la migracion a
    e-commerce solapadas en agosto de 2027, eso le daba a la campania el credito
    de una alarma que en realidad disparo el PSI de la migracion.
    """
    por_periodo = {r["periodo"]: r for r in linea}
    detalle = []
    for e in esc.ESCENARIOS:
        meses = [p for p in esc.PERIODOS_SIMULADOS if e.activo_en(p) and p in por_periodo]
        con_alarma = [p for p in meses if por_periodo[p]["alarma"]]
        propias = [p for p in con_alarma if atribuible(por_periodo[p], e)]
        deg = degradacion(meses, por_periodo) if meses else None

        detalle.append({
            "escenario": e.nombre,
            "detectable_sin_etiquetas": e.detectable_sin_etiquetas,
            "meses": len(meses),
            "meses_con_alarma": len(con_alarma),
            "meses_atribuibles": len(propias),
            "detectado": bool(propias),
            "alarma_ajena": bool(con_alarma) and not propias,
            "primer_mes": meses[0] if meses else None,
            "primera_alarma_propia": propias[0] if propias else None,
            "retraso_meses": (
                esc.PERIODOS_SIMULADOS.index(propias[0]) - esc.PERIODOS_SIMULADOS.index(meses[0])
                if propias else None
            ),
            "degradacion": deg,
        })

    tranquilos = [r for r in linea if esc.es_periodo_tranquilo(r["periodo"])]
    falsos = [r["periodo"] for r in tranquilos if r["alarma"]]
    esperables = [d for d in detalle if d["detectable_sin_etiquetas"]]
    return {
        "escenarios": detalle,
        "detectados": sum(d["detectado"] for d in esperables),
        "total_esperables": len(esperables),
        "total_escenarios": len(detalle),
        "invisibles_por_diseno": len(detalle) - len(esperables),
        "meses_tranquilos": len(tranquilos),
        "falsos_positivos": len(falsos),
        "meses_falso_positivo": falsos,
    }


def imprimir_verificacion(v: dict) -> None:
    print(f"\n{'='*92}\nVERIFICACION: lo detectado contra lo declarado\n{'='*92}")
    for d in v["escenarios"]:
        if d["detectado"]:
            estado = f"detectado (retraso {d['retraso_meses']} mes)"
        elif not d["detectable_sin_etiquetas"]:
            estado = "invisible por diseno"
        else:
            estado = "NO DETECTADO"
        print(f"  {d['escenario']:<34} {estado:<30} "
              f"{d['meses_atribuibles']}/{d['meses']} meses atribuibles")
        if d["alarma_ajena"]:
            print(f"    {'':<32} hubo alarma en la ventana, pero la disparo otro escenario")
        deg = d["degradacion"]
        if deg:
            variacion = -deg["caida_relativa"] * 100  # positivo = el F1 mejoro
            marca = "SI" if deg["degradado"] else "no"
            print(f"    {'':<32} degradacion de F1: {marca} "
                  f"({deg['f1_previo']} -> {deg['f1_minimo_en_ventana']}, "
                  f"{variacion:+.0f}%)")

    print(f"\n  Detectados sin etiquetas : {v['detectados']} de {v['total_esperables']} esperables"
          f"  ({v['invisibles_por_diseno']} invisible(s) por diseno)")
    print(f"  Falsos positivos         : {v['falsos_positivos']} en {v['meses_tranquilos']} meses tranquilos"
          + (f" ({', '.join(v['meses_falso_positivo'])})" if v["meses_falso_positivo"] else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corre el lazo completo sobre el tiempo simulado.")
    parser.add_argument("--datos", default="../Muestra/Muestra.del")
    parser.add_argument("--doc", default="../Dr_Muestra.xlsx")
    parser.add_argument("--sin-reentrenar", action="store_true",
                        help="Solo monitorear, sin disparar champion/challenger")
    args = parser.parse_args()
    correr(args.datos, args.doc, reentrenar=not args.sin_reentrenar)
