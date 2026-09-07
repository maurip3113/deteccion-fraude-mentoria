"""Pruebas del servicio de scoring.

Cubren lo que romperia el servicio en silencio: que el contrato de entrada
valide, que las features historicas se calculen igual que en entrenamiento, y
que el modelo distinga una transaccion sospechosa de una normal.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import sklearn
from fastapi.testclient import TestClient

from fraude.api import app, verificar_entorno
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


def test_metrics_cuenta_las_predicciones():
    antes = cliente.get("/metrics").json()["predicciones_totales"]
    cliente.post("/predict", json=TRX_BASE, params={"actualizar_estado": False})
    assert cliente.get("/metrics").json()["predicciones_totales"] == antes + 1
