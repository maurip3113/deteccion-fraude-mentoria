"""Ingenieria de variables compartida entre entrenamiento e inferencia.

Este modulo es el unico lugar donde se define como se construye una fila de
features. `train.py` lo usa sobre el historico completo y `api.py` lo usa sobre
una transaccion suelta: si las dos rutas no comparten el mismo codigo, el modelo
ve en produccion una distribucion distinta a la que vio entrenando
(*training/serving skew*), que es la falla mas comun al desplegar un modelo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes del dominio (documentadas en Dr_Muestra.xlsx y derivadas en TP1/TP2)
# ---------------------------------------------------------------------------

MAPA_PRESENCIA = {
    0: "Presencial", 1: "Presencial", 2: "Presencial", 3: "No presencial",
    4: "Presencial", 5: "No presencial", 6: "No presencial", 7: "No presencial",
    8: "No presencial", 9: "No presencial", 10: "No presencial",
    11: "No presencial", 12: "Presencial",
}

MONEDA_DOLAR = 2
RUBRO_SIN_DATO = "SIN RUBRO / NO APLICA"
LIMITE_ZSCORE = 10.0

COLS_ONEHOT = ["Cliente_Sexo", "Cliente_EstadoCivil", "Cliente_Segmento", "Presencia_Cliente"]

VARIABLES_CRUDAS = [
    "Trx_Importe", "Es_Outlier_Importe", "Hora_Dia", "Es_Fin_de_Semana",
    "Tiempo_Entre_Trx_Horas", "Es_Moneda_Dolar", "Cliente_Edad",
    "Cliente_Trx_Count", "Desvio_Importe_Cliente_Abs",
    "Cliente_Sexo", "Cliente_EstadoCivil", "Cliente_Segmento",
    "Presencia_Cliente", "Rubro_Categoria",
]


# ---------------------------------------------------------------------------
# Reconstruccion del historico (offline, solo entrenamiento)
# ---------------------------------------------------------------------------

def cargar_historico(path_del: str, path_xlsx: str) -> pd.DataFrame:
    """Replica el pipeline TP1 + TP2 sobre el archivo crudo de transacciones."""
    df = pd.read_csv(path_del, sep="|", encoding="latin-1")

    df["Trx_Importe"] = df["Trx_Importe"].str.replace(",", ".").astype(float)
    df["Trx_Fecha"] = pd.to_datetime(df["Trx_Fecha"], format="%y%m%d")
    df["Cliente_FechaNacimiento"] = pd.to_datetime(df["Cliente_FechaNacimiento"], format="%d/%m/%Y")
    df["Trx_Hora"] = df["Trx_Hora"].astype(str).str.zfill(8)
    df["Trx_Hora_fmt"] = pd.to_datetime(df["Trx_Hora"].str[:6], format="%H%M%S", errors="coerce").dt.time
    df["Es_Fraude"] = df["Es_Fraude"].map({"NO": 0, "SI": 1})

    df = df.drop_duplicates().reset_index(drop=True)
    for col in ["Cliente_Sexo", "Cliente_EstadoCivil", "Cliente_Segmento"]:
        df[col] = df.groupby("Cliente_Id")[col].transform(lambda x: x.ffill().bfill())
    df["Trx_CiudadTerminal"] = df["Trx_CiudadTerminal"].fillna("SIN_TERMINAL")
    df["Trx_EsChip"] = df["Trx_EsChip"].replace("0", "N")

    df_rubro = pd.read_excel(path_xlsx, sheet_name="RubroRed")
    col_categoria = [c for c in df_rubro.columns if c.startswith("Categor")][0]
    df = df.merge(
        df_rubro[["Codigo", col_categoria]].rename(
            columns={"Codigo": "Rubro_Codigo", col_categoria: "Rubro_Categoria"}),
        left_on="Trx_RubroRed", right_on="Rubro_Codigo", how="left",
    ).drop(columns=["Rubro_Codigo"])
    df["Rubro_Categoria"] = df["Rubro_Categoria"].fillna(RUBRO_SIN_DATO)

    df["Trx_Timestamp"] = pd.to_datetime(
        df["Trx_Fecha"].dt.strftime("%Y-%m-%d") + " " + df["Trx_Hora_fmt"].astype(str),
        errors="coerce",
    )
    df = df.sort_values(["Cliente_Id", "Trx_Timestamp"]).reset_index(drop=True)

    df["Tiempo_Entre_Trx_Horas"] = df.groupby("Cliente_Id")["Trx_Timestamp"].diff().dt.total_seconds() / 3600
    df["Hora_Dia"] = df["Trx_Hora_fmt"].apply(lambda x: x.hour)
    df["Es_Fin_de_Semana"] = df["Trx_Fecha"].dt.dayofweek.isin([5, 6]).astype(int)
    df["Presencia_Cliente"] = df["Trx_TipoTerminal"].astype(int).map(MAPA_PRESENCIA)
    df["Es_Moneda_Dolar"] = (df["Trx_Moneda"] == MONEDA_DOLAR).astype(int)

    # Ventana expansiva: cada transaccion se compara solo contra el pasado del cliente.
    grp = df.groupby("Cliente_Id")["Trx_Importe"]
    df["Cliente_Trx_Count"] = df.groupby("Cliente_Id").cumcount()
    promedio = grp.transform(lambda s: s.expanding().mean().shift(1)).fillna(df["Trx_Importe"])
    desvio_std = grp.transform(lambda s: s.expanding().std().shift(1)).fillna(0)
    z = np.where(desvio_std > 0, (df["Trx_Importe"] - promedio) / desvio_std, 0)
    df["Desvio_Importe_Cliente_Abs"] = np.abs(np.clip(z, -LIMITE_ZSCORE, LIMITE_ZSCORE))

    return df


# ---------------------------------------------------------------------------
# Preprocesador ajustable (se serializa junto al modelo)
# ---------------------------------------------------------------------------

@dataclass
class PreprocesadorFraude:
    """Guarda todo parametro estimado con datos de entrenamiento.

    Cada atributo de aca es una decision que se congela al entrenar: si se
    recalculara en inferencia con los datos que llegan, el modelo cambiaria de
    escala silenciosamente entre una version y la siguiente.
    """

    lim_outlier_inf: float = 0.0
    lim_outlier_sup: float = 0.0
    mediana_tiempo: float = 0.0
    tasa_fraude_global: float = 0.0
    mapa_rubro: dict[str, float] = field(default_factory=dict)
    mapa_rubro_codigo: dict[int, str] = field(default_factory=dict)
    fecha_referencia_edad: str = "2025-01-01"
    columnas_modelo: list[str] = field(default_factory=list)

    def fit(self, df_train: pd.DataFrame, y_train: pd.Series) -> "PreprocesadorFraude":
        q1, q3 = df_train["Trx_Importe"].quantile(0.25), df_train["Trx_Importe"].quantile(0.75)
        iqr = q3 - q1
        self.lim_outlier_inf = float(q1 - 1.5 * iqr)
        self.lim_outlier_sup = float(q3 + 1.5 * iqr)

        self.mediana_tiempo = float(df_train["Tiempo_Entre_Trx_Horas"].median())
        self.tasa_fraude_global = float(y_train.mean())
        self.mapa_rubro = {
            str(k): float(v) for k, v in y_train.groupby(df_train["Rubro_Categoria"]).mean().items()
        }

        X = self.transform(df_train, alinear=False)
        self.columnas_modelo = list(X.columns)
        return self

    def transform(self, df: pd.DataFrame, alinear: bool = True) -> pd.DataFrame:
        X = df.reindex(columns=VARIABLES_CRUDAS).copy()

        X["Es_Outlier_Importe"] = (
            (X["Trx_Importe"] < self.lim_outlier_inf) | (X["Trx_Importe"] > self.lim_outlier_sup)
        ).astype(int)
        # to_numeric primero: si la fila viene de la API con None, la columna llega
        # como object y el fillna quedaria con un dtype distinto al de entrenamiento.
        X["Tiempo_Entre_Trx_Horas"] = pd.to_numeric(
            X["Tiempo_Entre_Trx_Horas"], errors="coerce"
        ).fillna(self.mediana_tiempo)
        X["Rubro_Categoria_TasaFraude"] = (
            X["Rubro_Categoria"].astype(str).map(self.mapa_rubro).fillna(self.tasa_fraude_global)
        )
        X = X.drop(columns=["Rubro_Categoria"])
        X = pd.get_dummies(X, columns=COLS_ONEHOT, drop_first=True)

        if alinear:
            X = X.reindex(columns=self.columnas_modelo, fill_value=0)
        return X.astype(float)

    def edad(self, fecha_nacimiento: date) -> int:
        ref = pd.Timestamp(self.fecha_referencia_edad)
        return int((ref - pd.Timestamp(fecha_nacimiento)).days // 365)

    def categoria_rubro(self, codigo_rubro: int) -> str:
        return self.mapa_rubro_codigo.get(int(codigo_rubro), RUBRO_SIN_DATO)


# ---------------------------------------------------------------------------
# Estado por cliente (el "feature store" minimo que necesita la inferencia online)
# ---------------------------------------------------------------------------

@dataclass
class EstadoCliente:
    n: int = 0
    media: float = 0.0
    m2: float = 0.0  # suma de cuadrados de las diferencias (Welford)
    ultimo_timestamp: str | None = None

    @property
    def desvio_std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0


class EstadoClientes:
    """Historico acumulado por cliente, actualizado transaccion a transaccion.

    Tres de las features del modelo (`Cliente_Trx_Count`, `Tiempo_Entre_Trx_Horas`
    y `Desvio_Importe_Cliente_Abs`) no se pueden calcular mirando solo la
    transaccion que llega: dependen de todo lo que el cliente hizo antes. En un
    banco esto vive en un *feature store*; aca se resuelve con esta clase, que
    mantiene media y varianza corrientes (algoritmo de Welford) para reproducir
    exactamente la ventana expansiva del entrenamiento sin guardar el historico
    entero en memoria.
    """

    def __init__(self, estados: dict[int, EstadoCliente] | None = None):
        self.estados: dict[int, EstadoCliente] = estados or {}

    @classmethod
    def desde_historico(cls, df: pd.DataFrame) -> "EstadoClientes":
        estados: dict[int, EstadoCliente] = {}
        for cliente_id, grupo in df.sort_values(["Cliente_Id", "Trx_Timestamp"]).groupby("Cliente_Id"):
            importes = grupo["Trx_Importe"].to_numpy()
            n = len(importes)
            media = float(importes.mean())
            m2 = float(((importes - media) ** 2).sum())
            ultimo = grupo["Trx_Timestamp"].iloc[-1]
            estados[int(cliente_id)] = EstadoCliente(
                n=n, media=media, m2=m2,
                ultimo_timestamp=None if pd.isna(ultimo) else ultimo.isoformat(),
            )
        return cls(estados)

    def features_historicas(self, cliente_id: int, importe: float, timestamp: datetime) -> dict:
        """Features derivadas del pasado del cliente, sin incluir la transaccion actual."""
        estado = self.estados.get(int(cliente_id), EstadoCliente())

        promedio = estado.media if estado.n > 0 else importe
        std = estado.desvio_std
        z = (importe - promedio) / std if std > 0 else 0.0
        z = max(-LIMITE_ZSCORE, min(LIMITE_ZSCORE, z))

        if estado.ultimo_timestamp is None:
            horas = None  # primera transaccion del cliente: la imputa el preprocesador
        else:
            delta = timestamp - datetime.fromisoformat(estado.ultimo_timestamp)
            horas = delta.total_seconds() / 3600

        return {
            "Cliente_Trx_Count": estado.n,
            "Tiempo_Entre_Trx_Horas": horas,
            "Desvio_Importe_Cliente_Abs": abs(z),
            "cliente_conocido": int(cliente_id) in self.estados,
        }

    def actualizar(self, cliente_id: int, importe: float, timestamp: datetime) -> None:
        estado = self.estados.setdefault(int(cliente_id), EstadoCliente())
        estado.n += 1
        delta = importe - estado.media
        estado.media += delta / estado.n
        estado.m2 += delta * (importe - estado.media)
        estado.ultimo_timestamp = timestamp.isoformat()
