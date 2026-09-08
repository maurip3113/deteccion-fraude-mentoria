"""Reproduce las transacciones simuladas contra el servicio, en orden cronologico.

    uvicorn fraude.api:app --port 8000        (en otra terminal)
    python -m simulacion.reproductor --periodo 2026-12 --n 500

Es la parte "streaming" del ejercicio: en vez de puntuar un DataFrame entero de
una, manda las transacciones **de a una, en el orden en que habrian llegado**, y
deja que el servicio actualice el estado de cada cliente entre llamada y llamada.

Sirve para comprobar algo que el modo batch no puede: que la ruta de serving
--contrato de entrada, feature store, preprocesador, modelo, politica-- produzca
lo mismo que el pipeline offline. Si las dos rutas se separan, es training/serving
skew y ninguna metrica lo avisa.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
import pandas as pd

DIR_ARTEFACTOS = Path(__file__).resolve().parent.parent / "artifacts"


def a_payload(fila) -> dict:
    return {
        "cliente_id": int(fila.Cliente_Id),
        "trx_timestamp": fila.Trx_Timestamp.isoformat(),
        "trx_importe": round(float(fila.Trx_Importe), 2),
        "trx_moneda": int(fila.Trx_Moneda),
        "trx_rubro_red": int(fila.Trx_RubroRed),
        "trx_tipo_terminal": int(fila.Trx_TipoTerminal),
        "cliente_fecha_nacimiento": fila.Cliente_FechaNacimiento.date().isoformat(),
        "cliente_sexo": fila.Cliente_Sexo,
        "cliente_estado_civil": fila.Cliente_EstadoCivil,
        "cliente_segmento": fila.Cliente_Segmento,
    }


def reproducir(url: str, periodo: str | None, n: int | None,
               pausa: float, explicar: bool) -> pd.DataFrame:
    df = pd.read_parquet(DIR_ARTEFACTOS / "transacciones_simuladas.parquet")
    if periodo:
        df = df[df["periodo"] == periodo]
    # Orden de llegada real: por reloj, no por cliente.
    df = df.sort_values("Trx_Timestamp")
    if n:
        df = df.head(n)
    if df.empty:
        raise SystemExit("No hay transacciones para reproducir con ese filtro.")

    print(f"Reproduciendo {len(df):,} transacciones contra {url}")
    print(f"Periodo: {periodo or 'todos'} | fraude real en el lote: {df['Es_Fraude'].mean()*100:.2f}%\n")

    registros, fallidas = [], 0
    inicio = time.perf_counter()
    with httpx.Client(base_url=url, timeout=20.0) as cliente:
        for i, fila in enumerate(df.itertuples(), 1):
            try:
                r = cliente.post("/predict", json=a_payload(fila),
                                 params={"explicar": explicar})
                r.raise_for_status()
                cuerpo = r.json()
            except Exception as exc:
                fallidas += 1
                if fallidas <= 3:
                    print(f"  fallo en la transaccion {i}: {exc}")
                continue

            registros.append({
                "trx_timestamp": fila.Trx_Timestamp,
                "cliente_id": int(fila.Cliente_Id),
                "proba": cuerpo["probabilidad_fraude"],
                "alerta": cuerpo["alerta"],
                "es_fraude_real": int(fila.Es_Fraude),
                "latencia_ms": cuerpo["latencia_ms"],
            })
            if i % 250 == 0:
                print(f"  {i:>6,} enviadas | {sum(r['alerta'] for r in registros):>4} alertas")
            if pausa:
                time.sleep(pausa)

    transcurrido = time.perf_counter() - inicio
    res = pd.DataFrame(registros)
    salida = DIR_ARTEFACTOS / "reproduccion.jsonl"
    with salida.open("w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps({**r, "trx_timestamp": r["trx_timestamp"].isoformat()}) + "\n")

    print(f"\n{'='*62}")
    print(f"Enviadas      : {len(res):,} en {transcurrido:.1f}s "
          f"({len(res)/max(transcurrido,1e-9):.0f}/s)")
    if fallidas:
        print(f"Fallidas      : {fallidas}")
    print(f"Latencia media: {res['latencia_ms'].mean():.1f} ms "
          f"(p95 {res['latencia_ms'].quantile(0.95):.1f} ms)")
    print(f"Alertas       : {res['alerta'].sum():,} ({res['alerta'].mean()*100:.2f}%)")
    if res["es_fraude_real"].sum():
        atrapados = int((res["alerta"] & res["es_fraude_real"].astype(bool)).sum())
        print(f"Fraude real   : {int(res['es_fraude_real'].sum())} | detectado: {atrapados}")
    print(f"Registrado en {salida.name}")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce transacciones simuladas contra la API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--periodo", default=None, help='Un mes, por ejemplo "2026-12"')
    parser.add_argument("--n", type=int, default=500, help="Cuantas mandar (0 = todas)")
    parser.add_argument("--pausa", type=float, default=0.0,
                        help="Segundos entre transacciones, para ver el flujo despacio")
    parser.add_argument("--sin-explicacion", action="store_true",
                        help="No pedir SHAP, para medir la latencia del scoring solo")
    args = parser.parse_args()
    reproducir(args.url, args.periodo, args.n or None, args.pausa, not args.sin_explicacion)
