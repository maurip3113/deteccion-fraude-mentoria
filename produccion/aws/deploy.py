"""Despliega la imagen del servicio como endpoint serverless de SageMaker.

    python -m aws.deploy --rol arn:aws:iam::<cuenta>:role/<rol> --imagen <uri-ecr>

Requiere que la imagen ya este en ECR (ver README) y credenciales de AWS con
permisos de SageMaker. Crea tres objetos encadenados: un Model que apunta a la
imagen, un EndpointConfig que define la capacidad, y el Endpoint en si.

**Esto crea recursos que se facturan.** Serverless cobra por tiempo de computo
de cada invocacion y por el almacenamiento del modelo, sin costo fijo por hora
mientras no se invoca -- por eso es la opcion razonable para un proyecto que
recibe trafico esporadico. Consultar la calculadora de precios de AWS antes de
correrlo, y borrar el endpoint al terminar con `--borrar`.
"""

from __future__ import annotations

import argparse
import sys
import time

import boto3
from botocore.exceptions import ClientError

NOMBRE = "deteccion-fraude"


def esperar_en_servicio(cliente, endpoint: str, timeout: int = 900) -> str:
    inicio = time.time()
    while time.time() - inicio < timeout:
        estado = cliente.describe_endpoint(EndpointName=endpoint)["EndpointStatus"]
        if estado in ("InService", "Failed"):
            return estado
        print(f"  {estado}... ({time.time() - inicio:.0f}s)", flush=True)
        time.sleep(15)
    return "Timeout"


def desplegar(rol: str, imagen: str, region: str, memoria: int, concurrencia: int) -> str:
    cliente = boto3.client("sagemaker", region_name=region)
    sufijo = time.strftime("%Y%m%d-%H%M%S")
    nombre_modelo = f"{NOMBRE}-modelo-{sufijo}"
    nombre_config = f"{NOMBRE}-config-{sufijo}"

    print(f"Creando modelo {nombre_modelo}...")
    cliente.create_model(
        ModelName=nombre_modelo,
        ExecutionRoleArn=rol,
        PrimaryContainer={"Image": imagen, "Mode": "SingleModel"},
    )

    print(f"Creando configuracion {nombre_config} ({memoria} MB, concurrencia {concurrencia})...")
    cliente.create_endpoint_config(
        EndpointConfigName=nombre_config,
        ProductionVariants=[{
            "VariantName": "principal",
            "ModelName": nombre_modelo,
            "ServerlessConfig": {
                "MemorySizeInMB": memoria,
                "MaxConcurrency": concurrencia,
            },
        }],
    )

    try:
        cliente.describe_endpoint(EndpointName=NOMBRE)
        print(f"El endpoint {NOMBRE} ya existe: se actualiza en vez de recrearlo.")
        cliente.update_endpoint(EndpointName=NOMBRE, EndpointConfigName=nombre_config)
    except ClientError:
        print(f"Creando endpoint {NOMBRE}...")
        cliente.create_endpoint(EndpointName=NOMBRE, EndpointConfigName=nombre_config)

    estado = esperar_en_servicio(cliente, NOMBRE)
    print(f"\nEstado final: {estado}")
    if estado != "InService":
        detalle = cliente.describe_endpoint(EndpointName=NOMBRE).get("FailureReason", "sin detalle")
        print(f"Fallo: {detalle}")
        sys.exit(1)

    print(f"\nEndpoint listo. Probalo con:\n  python -m aws.invoke --region {region}")
    return NOMBRE


def borrar(region: str) -> None:
    cliente = boto3.client("sagemaker", region_name=region)
    cliente.delete_endpoint(EndpointName=NOMBRE)
    print(f"Endpoint {NOMBRE} borrado. Los Model y EndpointConfig quedan (no se facturan).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Despliega el servicio en SageMaker.")
    parser.add_argument("--rol", help="ARN del rol de ejecucion de SageMaker")
    parser.add_argument("--imagen", help="URI de la imagen en ECR")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--memoria", type=int, default=2048,
                        help="MB de memoria (el modelo y sus dependencias entran en 2 GB)")
    parser.add_argument("--concurrencia", type=int, default=5)
    parser.add_argument("--borrar", action="store_true", help="Borra el endpoint y sale")
    args = parser.parse_args()

    if args.borrar:
        borrar(args.region)
    elif not args.rol or not args.imagen:
        parser.error("--rol y --imagen son obligatorios para desplegar")
    else:
        desplegar(args.rol, args.imagen, args.region, args.memoria, args.concurrencia)
