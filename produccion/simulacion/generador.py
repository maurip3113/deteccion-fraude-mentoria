"""Genera anios sinteticos de transacciones para los mismos clientes.

    python -m simulacion.generador

**Los timestamps no se inventan: se desplaza el anio real hacia adelante.** Cada
cliente repite su secuencia de 2025 corrida 364 dias (52 semanas exactas, asi que
hasta el dia de la semana se conserva). Sobre eso se aplican los escenarios.

La razon es metodologica, no de comodidad. Intentar generar los tiempos entre
transacciones --sorteando huecos y acumulandolos-- produce un proceso de
renovacion cuya fase deriva: los huecos salen mas largos que los reales y la hora
del dia queda casi uniforme, cuando en la realidad se concentra en horario de
vigilia. Ese sesgo hacia que `Tiempo_Entre_Trx_Horas` marcara drift **todos** los
meses, tapando los escenarios y disparando reentrenamientos por un artefacto del
generador.

Desplazando el historico real, los meses tranquilos son identicos a la
distribucion de entrenamiento por construccion. Eso es exactamente lo que un
banco de pruebas necesita: si el monitoreo se alarma en un mes tranquilo, es un
falso positivo suyo, no ruido de los datos.

La etiqueta de fraude si se regenera, desde la regla declarada en `escenarios.py`
--construida con las tasas realmente observadas en 2025-- porque es lo que los
escenarios necesitan poder mover.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraude.features import cargar_historico, features_de_historial
from simulacion import escenarios as esc

DIR_ARTEFACTOS = Path(__file__).resolve().parent.parent / "artifacts"
SEMILLA = 7
DIAS_POR_ANIO = 364  # 52 semanas exactas: conserva el dia de la semana

TERMINAL_NO_PRESENCIAL = 3


def franja(hora: int) -> str:
    if hora < 6:
        return "madrugada"
    if hora < 12:
        return "manana"
    if hora < 18:
        return "tarde"
    return "noche"


def riesgo_relativo(df: pd.DataFrame, invertir_presencia: bool) -> np.ndarray:
    """Riesgo de cada fila segun la regla declarada, en unidades relativas.

    Multiplicativo, asi que el valor absoluto no significa nada: se reescala
    despues al nivel que pide el escenario. Lo que importa es el orden relativo,
    que es donde vive la relacion entre variables y fraude.
    """
    presencial = (df["Presencia_Cliente"] == "Presencial").to_numpy()
    if invertir_presencia:
        presencial = ~presencial

    riesgo = np.where(presencial, esc.MULT_PRESENCIAL, esc.MULT_NO_PRESENCIAL)
    riesgo = riesgo * np.where(df["Es_Moneda_Dolar"].to_numpy() == 1,
                               esc.MULT_DOLAR, esc.MULT_PESOS)
    riesgo = riesgo * np.where(df["Es_Fin_de_Semana"].to_numpy() == 1,
                               esc.MULT_FIN_DE_SEMANA, 1.0)
    riesgo = riesgo * df["Hora_Dia"].map(lambda h: esc.MULT_POR_FRANJA[franja(h)]).to_numpy()
    riesgo = riesgo * df["Rubro_Categoria"].map(esc.MULT_POR_RUBRO).fillna(
        esc.MULT_RUBRO_DESCONOCIDO).to_numpy()
    return riesgo


def aplicar_escenarios(mes: pd.DataFrame, periodo: str,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Deforma un mes ya desplazado segun los escenarios activos."""
    mes = mes.copy()
    factor_fraude, invertir = 1.0, False

    for e in esc.escenarios_activos(periodo):
        factor_fraude *= e.factor_fraude
        invertir = invertir or e.invierte_presencia
        if e.factor_importe != 1.0:
            mes["Trx_Importe"] = mes["Trx_Importe"] * e.factor_importe
        if e.desplaza_ecommerce > 0:
            presenciales = mes.index[mes["Presencia_Cliente"] == "Presencial"]
            n_mover = min(int(len(mes) * e.desplaza_ecommerce), len(presenciales))
            if n_mover:
                mover = rng.choice(presenciales, size=n_mover, replace=False)
                mes.loc[mover, "Presencia_Cliente"] = "No presencial"
                mes.loc[mover, "Trx_TipoTerminal"] = TERMINAL_NO_PRESENCIAL

    riesgo = riesgo_relativo(mes, invertir)
    p = np.clip(riesgo * (esc.TASA_BASE / riesgo.mean()), 0, 0.9)
    mes["Es_Fraude"] = (rng.random(len(mes)) < p).astype(int)

    # Una campania genera operaciones nuevas, no reetiquetea las que ya estaban.
    # Subir la tasa sobre las filas existentes cambiaria solo la etiqueta: las
    # transacciones se seguirian viendo igual y el modelo no podria notar nada.
    if factor_fraude > 1.0:
        n_extra = int(mes["Es_Fraude"].sum() * (factor_fraude - 1.0))
        if n_extra > 0:
            elegidas = rng.choice(len(mes), size=n_extra, replace=True,
                                  p=riesgo / riesgo.sum())
            campania = mes.iloc[elegidas].copy()
            campania["Es_Fraude"] = 1
            campania["Trx_Importe"] = campania["Trx_Importe"] * rng.uniform(1.5, 4.0, n_extra)
            # El defraudador opera de madrugada, cuando nadie mira.
            horas = rng.choice([0, 1, 2, 3, 4, 5], size=n_extra)
            campania["Trx_Timestamp"] = (
                campania["Trx_Timestamp"].dt.normalize() + pd.to_timedelta(horas, unit="h")
            )
            campania["Hora_Dia"] = horas
            mes = pd.concat([mes, campania], ignore_index=True)

    return mes


def generar(path_datos: str, path_doc: str, salida: Path | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(SEMILLA)
    print("Cargando el historico real de 2025...")
    real = cargar_historico(path_datos, path_doc)
    real["periodo"] = real["Trx_Fecha"].dt.strftime("%Y-%m")
    print(f"  {len(real):,} transacciones, {real['Cliente_Id'].nunique()} clientes\n")
    print(esc.resumen())
    print()

    anios = sorted({int(p[:4]) for p in esc.PERIODOS_SIMULADOS})
    piezas = []
    for i, anio in enumerate(anios, start=1):
        corrido = real.copy()
        desplazamiento = pd.Timedelta(days=DIAS_POR_ANIO * i)
        corrido["Trx_Timestamp"] = corrido["Trx_Timestamp"] + desplazamiento
        corrido["Trx_Fecha"] = corrido["Trx_Fecha"] + desplazamiento
        corrido["periodo"] = corrido["Trx_Fecha"].dt.strftime("%Y-%m")
        corrido = corrido[corrido["periodo"].isin(esc.PERIODOS_SIMULADOS)]

        for periodo, mes in corrido.groupby("periodo"):
            piezas.append(aplicar_escenarios(mes, periodo, rng))

    sintetico = pd.concat(piezas, ignore_index=True)
    for periodo in esc.PERIODOS_SIMULADOS:
        mes = sintetico[sintetico["periodo"] == periodo]
        if mes.empty:
            continue
        activos = ", ".join(e.nombre for e in esc.escenarios_activos(periodo)) or "-"
        print(f"  {periodo}  {len(mes):>6,} trx  fraude {mes['Es_Fraude'].mean()*100:>5.2f}%   {activos}")

    # Las variables de historial se recalculan sobre real + sintetico: el contador
    # de cada cliente tiene que seguir desde donde quedo, no reiniciarse.
    completo = features_de_historial(pd.concat([real, sintetico], ignore_index=True))
    resultado = completo[completo["Trx_Fecha"].dt.year >= anios[0]].reset_index(drop=True)

    print(f"\nGenerado: {len(resultado):,} transacciones sinteticas, "
          f"{int(resultado['Es_Fraude'].sum()):,} fraudes ({resultado['Es_Fraude'].mean()*100:.2f}%)")

    salida = salida or (DIR_ARTEFACTOS / "transacciones_simuladas.parquet")
    columnas = [c for c in resultado.columns if c != "Trx_Hora_fmt"]
    resultado[columnas].to_parquet(salida, index=False)
    print(f"Guardado en {salida.name} ({salida.stat().st_size / 1e6:.1f} MB)")
    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera anios sinteticos de transacciones.")
    parser.add_argument("--datos", default="../Muestra/Muestra.del")
    parser.add_argument("--doc", default="../Dr_Muestra.xlsx")
    args = parser.parse_args()
    generar(args.datos, args.doc)
