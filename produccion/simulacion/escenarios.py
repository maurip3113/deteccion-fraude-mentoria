"""Verdad declarada: que cambia, cuando, y cuanto.

Este archivo es el centro del ejercicio. Los datos sinteticos no sirven para
demostrar que el modelo funciona --si uno inyecta un patron, el modelo lo
encuentra, porque uno lo puso ahi--. Sirven para lo contrario: **poner a prueba
el monitoreo**.

Como aca esta escrito de antemano que cambia y cuando, despues se puede verificar
dos cosas que si valen:

  1. Que el monitoreo detecte lo que se inyecto, y con cuanto retraso.
  2. Que **no** marque nada en los periodos tranquilos, que es la mitad que casi
     nadie prueba y la que decide si una alarma se mira o se ignora.

La regla generativa no usa el modelo. Sale de las tasas de fraude realmente
observadas en 2025, medidas sobre el dataset original, para que las relaciones
entre variables y fraude sean las del mundo real y no las que el modelo cree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Regla generativa: multiplicadores medidos sobre el dataset real de 2025 ---
# p(fraude) = TASA_BASE * producto de los multiplicadores que apliquen
TASA_BASE = 0.00621

MULT_NO_PRESENCIAL = 2.495
MULT_PRESENCIAL = 0.220
MULT_DOLAR = 15.216
MULT_PESOS = 0.544
MULT_FIN_DE_SEMANA = 1.160
MULT_POR_FRANJA = {"madrugada": 4.191, "manana": 0.674, "tarde": 0.877, "noche": 0.993}

# El rubro es la senal mas fuerte que tiene el modelo (36,5% del peso segun SHAP),
# asi que omitirlo hacia que el fraude sintetico cayera en lugares donde el modelo
# aprendio que es imposible --sobre todo en SIN RUBRO, que son 103.499 filas reales
# sin un solo fraude-- y su recall se iba a cero. Los valores salen de las tasas
# observadas por categoria en 2025.
MULT_POR_RUBRO = {
    "SIN RUBRO / NO APLICA": 0.000,
    "Comercio Mayorista y Minorista": 0.463,
    "Servicios Varios": 3.570,
    "Servicios Profesionales y Educativos": 5.054,
    "Transporte": 0.937,
    "Servicios Públicos (Utilities)": 4.996,
    "Servicios Gubernamentales": 2.013,
    "Tiendas de Ropa y Calzado": 3.461,
    "Agrícola / Agronomía": 0.000,
    "Contratistas / Constructores": 0.000,
    "Aerolíneas y Viajes": 0.000,
    "Hoteles y Alojamiento": 29.267,
    "Alquiler de Vehículos": 0.000,
}
MULT_RUBRO_DESCONOCIDO = 1.0


@dataclass
class Escenario:
    """Un cambio deliberado sobre un tramo del tiempo simulado.

    Los factores multiplican; 1.0 significa "no toca nada". `desplaza_ecommerce`
    es la unica excepcion: es un delta sobre la proporcion de transacciones no
    presenciales, en puntos de proporcion.
    """

    nombre: str
    desde: str                      # "AAAA-MM" inclusive
    hasta: str                      # "AAAA-MM" inclusive
    descripcion: str
    factor_fraude: float = 1.0      # multiplica la probabilidad de fraude
    factor_importe: float = 1.0     # escala los importes
    desplaza_ecommerce: float = 0.0 # +0.15 = 15 puntos mas de no presencial
    # Cambia *donde* ocurre el fraude, no cuanto. Es el caso dificil: la tasa
    # global no se mueve, pero la relacion entre variables y fraude si.
    invierte_presencia: bool = False

    # --- Firma esperada: como deberia delatarse este escenario ---
    # Sin esto la verificacion solo puede mirar si hubo *alguna* alarma dentro de
    # la ventana, y cuando dos escenarios se superponen le atribuye a uno el
    # merito del otro. Declarando la firma se puede exigir que el disparador de
    # la alarma corresponda al escenario, y no solo que coincida en el tiempo.
    variables_afectadas: tuple[str, ...] = ()   # features que deberia mover
    espera_factor_alertas: bool = False         # deberia disparar por volumen
    detectable_sin_etiquetas: bool = True       # False = solo visible con etiquetas

    def activo_en(self, periodo: str) -> bool:
        return self.desde <= periodo <= self.hasta


ESCENARIOS: list[Escenario] = [
    Escenario(
        nombre="temporada alta",
        desde="2026-12", hasta="2026-12",
        descripcion="Repeticion del pico de fin de anio: mas fraude y mas consumo.",
        factor_fraude=6.0,
        factor_importe=1.25,
        variables_afectadas=("Trx_Importe", "Es_Outlier_Importe"),
        espera_factor_alertas=True,
    ),
    Escenario(
        nombre="migracion a e-commerce",
        desde="2027-04", hasta="2027-12",
        descripcion="La cartera se vuelve mas digital de forma permanente. Es drift "
                    "de entrada puro: cambia la mezcla de canales, no la relacion "
                    "entre canal y fraude.",
        desplaza_ecommerce=0.18,
        variables_afectadas=("Presencia_Cliente_Presencial",),
    ),
    Escenario(
        nombre="campania de fraude presencial",
        desde="2027-08", hasta="2027-09",
        descripcion="El fraude se corre a transacciones presenciales, donde el modelo "
                    "aprendio que casi no hay. La tasa global apenas se mueve: lo que "
                    "cambia es la relacion. El PSI de entrada no deberia verlo.",
        invierte_presencia=True,
        factor_fraude=1.8,
        # No deja firma en las entradas: reetiqueta que transacciones son fraude
        # sin cambiar como se ven. Solo se puede ver cuando llegan las etiquetas.
        detectable_sin_etiquetas=False,
    ),
    Escenario(
        nombre="temporada alta 2027",
        desde="2027-12", hasta="2027-12",
        descripcion="Segunda repeticion del pico estacional.",
        factor_fraude=6.0,
        factor_importe=1.25,
        variables_afectadas=("Trx_Importe", "Es_Outlier_Importe"),
        espera_factor_alertas=True,
    ),
]

# Todo lo que no cae en un escenario es un periodo tranquilo, y ahi el monitoreo
# no deberia levantar nada. Los falsos positivos cuentan tanto como los aciertos.
PERIODOS_SIMULADOS = [f"{anio}-{mes:02d}" for anio in (2026, 2027) for mes in range(1, 13)]


def escenarios_activos(periodo: str) -> list[Escenario]:
    return [e for e in ESCENARIOS if e.activo_en(periodo)]


def es_periodo_tranquilo(periodo: str) -> bool:
    return not escenarios_activos(periodo)


def resumen() -> str:
    lineas = [f"{len(ESCENARIOS)} escenarios declarados sobre {len(PERIODOS_SIMULADOS)} meses:"]
    for e in ESCENARIOS:
        lineas.append(f"  {e.desde} a {e.hasta}  {e.nombre}")
    tranquilos = [p for p in PERIODOS_SIMULADOS if es_periodo_tranquilo(p)]
    lineas.append(f"  {len(tranquilos)} meses tranquilos (no deberian disparar nada)")
    return "\n".join(lineas)
