"""Pruebas del servicio de scoring.

Cubren lo que romperia el servicio en silencio: que el contrato de entrada
valide, que las features historicas se calculen igual que en entrenamiento, y
que el modelo distinga una transaccion sospechosa de una normal.
"""

import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import sklearn
from fastapi.testclient import TestClient

from fraude.api import Transaccion, app, verificar_entorno
from fraude.features import EstadoClientes

cliente = TestClient(app)

TRX_BASE = {
    "cliente_id": 101,
    "trx_timestamp": "2025-06-10T14:00:00",
    "trx_importe": 1200.0,
    "trx_moneda": 1,
    "trx_rubro_red": 5411,
    "trx_tipo_terminal": 1,
    "cliente_fecha_nacimiento": "1985-04-12",
    "cliente_sexo": "MASCULINO",
    "cliente_estado_civil": "SOLTERO/A",
    "cliente_segmento": "S3",
}


def test_health_expone_la_version_del_modelo():
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"
    assert r.json()["modelo_version"] >= 1


def test_predict_devuelve_probabilidad_valida():
    r = cliente.post("/predict", json=TRX_BASE, params={"actualizar_estado": False})
    assert r.status_code == 200
    cuerpo = r.json()
    assert 0.0 <= cuerpo["probabilidad_fraude"] <= 1.0
    assert cuerpo["alerta"] == (cuerpo["probabilidad_fraude"] >= cuerpo["umbral"])
    assert cuerpo["accion"] in ("aprobar", "revision_humana")


def test_transaccion_sospechosa_puntua_mas_alto_que_una_normal():
    sospechosa = dict(TRX_BASE, trx_importe=185000.0,
                      trx_timestamp="2025-12-24T03:47:00", trx_tipo_terminal=3)
    p_normal = cliente.post("/predict", json=TRX_BASE,
                            params={"actualizar_estado": False}).json()["probabilidad_fraude"]
    p_sospechosa = cliente.post("/predict", json=sospechosa,
                                params={"actualizar_estado": False}).json()["probabilidad_fraude"]
    assert p_sospechosa > p_normal


def test_cliente_desconocido_no_rompe_el_servicio():
    r = cliente.post("/predict", json=dict(TRX_BASE, cliente_id=999_999),
                     params={"actualizar_estado": False})
    assert r.status_code == 200
    assert r.json()["cliente_conocido"] is False


@pytest.mark.parametrize("campo,valor", [
    ("trx_importe", -5.0),          # importe negativo
    ("trx_tipo_terminal", 99),      # fuera del catalogo
    ("cliente_sexo", "OTRO"),       # categoria no vista en entrenamiento
])
def test_entrada_invalida_es_rechazada(campo, valor):
    r = cliente.post("/predict", json=dict(TRX_BASE, **{campo: valor}))
    assert r.status_code == 422


def test_estado_del_cliente_replica_la_ventana_expansiva():
    """La media y el desvio online tienen que coincidir con los de pandas.

    Es la prueba que evita el training/serving skew: si Welford y la ventana
    expansiva de entrenamiento se separan, el modelo recibe en produccion un
    z-score con otra escala y nadie se entera.
    """
    importes = [100.0, 250.0, 90.0, 400.0, 120.0, 310.0]
    ts = [datetime(2025, 1, d + 1, 12) for d in range(len(importes))]

    esperado_promedio = pd.Series(importes).expanding().mean().shift(1)
    esperado_std = pd.Series(importes).expanding().std().shift(1)

    estado = EstadoClientes()
    for i, (importe, momento) in enumerate(zip(importes, ts)):
        features = estado.features_historicas(1, importe, momento)
        assert features["Cliente_Trx_Count"] == i

        if i > 1:
            promedio, std = esperado_promedio.iloc[i], esperado_std.iloc[i]
            z = np.clip((importe - promedio) / std, -10, 10)
            assert features["Desvio_Importe_Cliente_Abs"] == pytest.approx(abs(z), rel=1e-9)

        estado.actualizar(1, importe, momento)


def test_entorno_desalineado_avisa():
    """El artefacto declara con que versiones se entreno; cargarlo con otras avisa."""
    assert verificar_entorno({"scikit-learn": sklearn.__version__}) == []

    with pytest.warns(RuntimeWarning):
        diferencias = verificar_entorno({"scikit-learn": "0.0.1"})
    assert len(diferencias) == 1

    # Un patch distinto de Python no rompe la carga y la imagen base avanza sola:
    # avisar por eso convertiria la alerta en ruido permanente.
    mayor_menor = ".".join(map(str, sys.version_info[:2]))
    assert verificar_entorno({"python": f"{mayor_menor}.0"}) == []
    with pytest.warns(RuntimeWarning):
        assert verificar_entorno({"python": "3.0.0"}) != []


def test_contrato_de_sagemaker():
    """Las dos rutas que exige un endpoint con contenedor propio."""
    assert cliente.get("/ping").status_code == 200

    unica = cliente.post("/invocations", json=TRX_BASE)
    assert unica.status_code == 200
    assert 0.0 <= unica.json()["probabilidad_fraude"] <= 1.0

    lote = cliente.post("/invocations", json=[TRX_BASE, TRX_BASE])
    assert lote.status_code == 200
    assert len(lote.json()) == 2


def test_invocations_no_modifica_el_estado_del_cliente():
    """El endpoint es de solo lectura: varias replicas no comparten memoria.

    Si acumulara historia, dos replicas darian respuestas distintas para la misma
    transaccion y todo se perderia en cada arranque en frio.
    """
    respuestas = cliente.post("/invocations", json=[TRX_BASE, TRX_BASE, TRX_BASE]).json()
    probabilidades = {r["probabilidad_fraude"] for r in respuestas}
    assert len(probabilidades) == 1


def test_acepta_importe_cero():
    """El dataset real tiene 78 transacciones de importe cero y el modelo las vio.

    Rechazarlas en serving crearía una inconsistencia con el entrenamiento. Un
    importe negativo, en cambio, sí es imposible y se sigue rechazando.
    """
    r = cliente.post("/predict", json=dict(TRX_BASE, trx_importe=0.0),
                     params={"actualizar_estado": False})
    assert r.status_code == 200
    assert cliente.post("/predict", json=dict(TRX_BASE, trx_importe=-1.0)).status_code == 422


def test_la_prediccion_viene_explicada():
    """Una alerta sin razones le deja al analista todo el trabajo."""
    cuerpo = cliente.post("/predict", json=TRX_BASE,
                          params={"actualizar_estado": False}).json()
    explicacion = cuerpo["explicacion"]
    assert 1 <= len(explicacion) <= 5
    assert all(e["empuja"] in ("hacia fraude", "hacia legitima") for e in explicacion)
    # Ordenadas por peso, para que lo primero que se lee sea lo que mas pesó.
    pesos = [abs(e["contribucion"]) for e in explicacion]
    assert pesos == sorted(pesos, reverse=True)

    sin = cliente.post("/predict", json=TRX_BASE,
                       params={"actualizar_estado": False, "explicar": False}).json()
    assert sin["explicacion"] is None


def test_las_contribuciones_suman_la_prediccion():
    """SHAP es exacto: base + contribuciones reproduce el margen del modelo.

    Si esto se rompe, la explicación dejó de corresponderse con la decisión y
    estaríamos mostrándole al analista un motivo que no es el real.
    """
    from fraude.api import BUNDLE, construir_fila
    from fraude.explain import contribuciones

    X, _ = construir_fila(Transaccion(**TRX_BASE))
    contribs, base = contribuciones(BUNDLE["modelo"], X)

    margen = base + contribs[0].sum()
    proba_shap = 1 / (1 + np.exp(-margen))
    proba_modelo = BUNDLE["modelo"].predict_proba(X)[0, 1]
    assert proba_shap == pytest.approx(proba_modelo, abs=1e-5)


def test_metrics_cuenta_las_predicciones():
    antes = cliente.get("/metrics").json()["predicciones_totales"]
    cliente.post("/predict", json=TRX_BASE, params={"actualizar_estado": False})
    assert cliente.get("/metrics").json()["predicciones_totales"] == antes + 1
