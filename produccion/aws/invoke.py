"""Prueba el endpoint desplegado con una transaccion sospechosa y una normal.

    python -m aws.invoke --region us-east-1

Sirve de verificacion post-despliegue: que el endpoint responda no alcanza, hay
que ver que discrimine. Si las dos transacciones dan probabilidades parecidas,
algo se rompio entre el artefacto y el contenedor aunque el health check pase.
"""

from __future__ import annotations

import argparse
import json

import boto3

NOMBRE = "deteccion-fraude"

BASE = {
    "cliente_id": 101,
    "trx_moneda": 1,
    "cliente_fecha_nacimiento": "1985-04-12",
    "cliente_sexo": "MASCULINO",
    "cliente_estado_civil": "SOLTERO/A",
    "cliente_segmento": "S3",
}

CASOS = [
    ("normal", {**BASE, "trx_timestamp": "2025-06-10T14:00:00", "trx_importe": 1200.0,
                "trx_rubro_red": 5411, "trx_tipo_terminal": 1}),
    ("sospechosa", {**BASE, "trx_timestamp": "2025-12-24T03:47:00", "trx_importe": 185000.0,
                    "trx_rubro_red": 5311, "trx_tipo_terminal": 3}),
]


def invocar(region: str, endpoint: str) -> dict[str, float]:
    cliente = boto3.client("sagemaker-runtime", region_name=region)
    probabilidades = {}

    for etiqueta, transaccion in CASOS:
        respuesta = cliente.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=json.dumps(transaccion),
        )
        cuerpo = json.loads(respuesta["Body"].read())
        probabilidades[etiqueta] = cuerpo["probabilidad_fraude"]
        print(f"{etiqueta:12} proba={cuerpo['probabilidad_fraude']:.6f}  "
              f"accion={cuerpo['accion']}  modelo=v{cuerpo['modelo_version']}")

    factor = probabilidades["sospechosa"] / max(probabilidades["normal"], 1e-9)
    print(f"\nLa sospechosa puntua {factor:.0f} veces mas alto que la normal.")
    if factor < 2:
        print("ATENCION: el modelo casi no las distingue. Revisar el artefacto del contenedor.")
    return probabilidades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoca el endpoint desplegado.")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--endpoint", default=NOMBRE)
    args = parser.parse_args()
    invocar(args.region, args.endpoint)
