"""Pruebas del monitoreo de drift."""

import numpy as np
import pandas as pd

from fraude.monitoring import ESTADOS_SIN_DRIFT, clasificar, drift_de_datos, psi


def referencia_de(valores: np.ndarray, n_bins: int = 10) -> dict:
    bordes = np.unique(np.quantile(valores, np.linspace(0, 1, n_bins + 1)))
    bordes[0], bordes[-1] = -np.inf, np.inf
    conteos, _ = np.histogram(valores, bins=bordes)
    return {"tipo": "continua", "bordes": bordes.tolist(),
            "proporciones": (conteos / len(valores)).tolist(),
            "media": float(valores.mean())}


def test_psi_es_cero_cuando_nada_se_movio():
    proporciones = np.array([0.2, 0.3, 0.5])
    assert psi(proporciones, proporciones) == 0.0


def test_psi_crece_con_el_desplazamiento():
    ref = np.array([0.33, 0.34, 0.33])
    poco = psi(ref, np.array([0.30, 0.34, 0.36]))
    mucho = psi(ref, np.array([0.05, 0.15, 0.80]))
    assert 0 < poco < mucho


def test_los_cortes_convencionales():
    assert clasificar(0.05) == "estable"
    assert clasificar(0.15) == "MODERADO"
    assert clasificar(0.40) == "SEVERO"


def test_un_contador_monotono_no_cuenta_como_drift():
    """`Cliente_Trx_Count` se aparta del entrenamiento por construccion.

    Es un acumulado: cuanto mas tiempo pasa, mas alto esta. Medirlo contra una
    referencia congelada da una alarma permanente, y una alarma que suena siempre
    deja de mirarse. Se sigue calculando su PSI, pero no dispara nada.
    """
    generador = np.random.default_rng(0)
    base = generador.normal(100, 20, 5000)
    desplazado = base + 200  # el mismo corrimiento para las dos variables

    referencia = {"Cliente_Trx_Count": referencia_de(base),
                  "Trx_Importe": referencia_de(base)}
    nuevo = pd.DataFrame({"Cliente_Trx_Count": desplazado, "Trx_Importe": desplazado})

    tabla = drift_de_datos(nuevo, referencia).set_index("variable")

    # Mismo corrimiento, mismo PSI: lo que cambia es como se lo interpreta.
    assert tabla.loc["Cliente_Trx_Count", "psi"] == tabla.loc["Trx_Importe", "psi"]
    assert tabla.loc["Cliente_Trx_Count", "estado"] == "estructural"
    assert tabla.loc["Trx_Importe", "estado"] == "SEVERO"

    assert tabla.loc["Cliente_Trx_Count", "estado"] in ESTADOS_SIN_DRIFT
    assert tabla.loc["Trx_Importe", "estado"] not in ESTADOS_SIN_DRIFT
