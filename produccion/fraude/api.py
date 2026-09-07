"""Servicio de scoring de fraude en tiempo real.

    uvicorn fraude.api:app --reload

Expone el modelo entrenado detras de un endpoint HTTP: recibe una transaccion
tal como la emitiria el sistema transaccional, arma las mismas features que vio
el modelo entrenando y devuelve una probabilidad de fraude junto con la decision
de alertar o no.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraude.features import MAPA_PRESENCIA, MONEDA_DOLAR

DIR_ARTEFACTOS = Path(__file__).resolve().parent.parent / "artifacts"
LOG_PREDICCIONES = DIR_ARTEFACTOS / "predicciones.jsonl"


# ---------------------------------------------------------------------------
# Contratos de entrada y salida
# ---------------------------------------------------------------------------

class Transaccion(BaseModel):
    cliente_id: int = Field(..., description="Identificador del cliente")
    trx_timestamp: datetime = Field(..., description="Fecha y hora de la transaccion")
    trx_importe: float = Field(..., gt=0)
    trx_moneda: int = Field(..., description="1 = pesos, 2 = dolares")
    trx_rubro_red: int = Field(..., description="Codigo de rubro del comercio")
    trx_tipo_terminal: int = Field(..., ge=0, le=12)
    cliente_fecha_nacimiento: date
    cliente_sexo: Literal["MASCULINO", "FEMENINO"]
    cliente_estado_civil: str
    cliente_segmento: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "cliente_id": 101,
                "trx_timestamp": "2025-12-24T03:47:00",
                "trx_importe": 185000.0,
                "trx_moneda": 1,
                "trx_rubro_red": 5311,
                "trx_tipo_terminal": 3,
                "cliente_fecha_nacimiento": "1985-04-12",
                "cliente_sexo": "MASCULINO",
                "cliente_estado_civil": "SOLTERO/A",
                "cliente_segmento": "S3",
            }]
        }
    }


class Prediccion(BaseModel):
    cliente_id: int
    probabilidad_fraude: float
    alerta: bool
    accion: str
    umbral: float
    modelo_version: int
    cliente_conocido: bool
    latencia_ms: float


# ---------------------------------------------------------------------------
# Carga del artefacto
# ---------------------------------------------------------------------------

def cargar_bundle(version: int | None = None) -> dict:
    if version is None:
        puntero = DIR_ARTEFACTOS / "modelo_actual.txt"
        if not puntero.exists():
            raise FileNotFoundError(
                "No hay modelo entrenado. Corre primero: python -m fraude.train"
            )
        version = int(puntero.read_text().strip())
    return joblib.load(DIR_ARTEFACTOS / f"modelo_v{version}.joblib")


app = FastAPI(
    title="API de deteccion de fraude",
    description="Scoring de transacciones en tiempo real - Mentoria DiploDatos 2026",
    version="1.0.0",
)

BUNDLE = cargar_bundle()
CONTADORES = {"predicciones": 0, "alertas": 0}
PROBABILIDADES_RECIENTES: deque[float] = deque(maxlen=10_000)


# ---------------------------------------------------------------------------
# Armado de features desde el payload crudo
# ---------------------------------------------------------------------------

def construir_fila(trx: Transaccion) -> tuple[pd.DataFrame, bool]:
    prep = BUNDLE["preprocesador"]
    historicas = BUNDLE["estado_clientes"].features_historicas(
        trx.cliente_id, trx.trx_importe, trx.trx_timestamp
    )
    fila = {
        "Trx_Importe": trx.trx_importe,
        "Es_Outlier_Importe": 0,  # lo recalcula el preprocesador con los limites de train
        "Hora_Dia": trx.trx_timestamp.hour,
        "Es_Fin_de_Semana": int(trx.trx_timestamp.weekday() in (5, 6)),
        "Tiempo_Entre_Trx_Horas": historicas["Tiempo_Entre_Trx_Horas"],
        "Es_Moneda_Dolar": int(trx.trx_moneda == MONEDA_DOLAR),
        "Cliente_Edad": prep.edad(trx.cliente_fecha_nacimiento),
        "Cliente_Trx_Count": historicas["Cliente_Trx_Count"],
        "Desvio_Importe_Cliente_Abs": historicas["Desvio_Importe_Cliente_Abs"],
        "Cliente_Sexo": trx.cliente_sexo,
        "Cliente_EstadoCivil": trx.cliente_estado_civil,
        "Cliente_Segmento": trx.cliente_segmento,
        "Presencia_Cliente": MAPA_PRESENCIA.get(trx.trx_tipo_terminal, "No presencial"),
        "Rubro_Categoria": prep.categoria_rubro(trx.trx_rubro_red),
    }
    return prep.transform(pd.DataFrame([fila])), historicas["cliente_conocido"]


@app.post("/predict", response_model=Prediccion, summary="Score de una transaccion")
def predict(trx: Transaccion, actualizar_estado: bool = True) -> Prediccion:
    inicio = time.perf_counter()
    try:
        X, conocido = construir_fila(trx)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"No se pudo procesar la transaccion: {exc}")

    proba = float(BUNDLE["modelo"].predict_proba(X)[0, 1])
    umbral = BUNDLE["umbral"]
    alerta = proba >= umbral

    # El estado del cliente solo deberia avanzar con transacciones efectivamente
    # liquidadas. Un reintento o una simulacion no tienen que ensuciar el historico.
    if actualizar_estado:
        BUNDLE["estado_clientes"].actualizar(trx.cliente_id, trx.trx_importe, trx.trx_timestamp)

    CONTADORES["predicciones"] += 1
    CONTADORES["alertas"] += int(alerta)
    PROBABILIDADES_RECIENTES.append(proba)

    latencia = (time.perf_counter() - inicio) * 1000
    with LOG_PREDICCIONES.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(),
            "cliente_id": trx.cliente_id,
            "proba": round(proba, 6),
            "alerta": alerta,
            "version": BUNDLE["version"],
        }) + "\n")

    return Prediccion(
        cliente_id=trx.cliente_id,
        probabilidad_fraude=round(proba, 6),
        alerta=alerta,
        # El TP3 y el notebook de mejoras concluyen lo mismo: con esta precision,
        # bloquear automaticamente genera mas dano que el fraude que evita.
        accion="revision_humana" if alerta else "aprobar",
        umbral=umbral,
        modelo_version=BUNDLE["version"],
        cliente_conocido=conocido,
        latencia_ms=round(latencia, 2),
    )


@app.get("/health", summary="Estado del servicio")
def health() -> dict:
    return {
        "estado": "ok",
        "modelo_version": BUNDLE["version"],
        "umbral": BUNDLE["umbral"],
        "n_features": len(BUNDLE["preprocesador"].columnas_modelo),
        "clientes_en_estado": len(BUNDLE["estado_clientes"].estados),
    }


@app.get("/metrics", summary="Metricas operativas para monitoreo")
def metrics() -> dict:
    n = CONTADORES["predicciones"]
    probas = pd.Series(PROBABILIDADES_RECIENTES)
    return {
        "predicciones_totales": n,
        "alertas_totales": CONTADORES["alertas"],
        "tasa_alertas": round(CONTADORES["alertas"] / n, 6) if n else 0.0,
        "probabilidad_media": round(float(probas.mean()), 6) if n else 0.0,
        "probabilidad_p95": round(float(probas.quantile(0.95)), 6) if n else 0.0,
        "modelo_version": BUNDLE["version"],
    }
