"""Genera el tablero de monitoreo a partir del reporte de `monitoring.py`.

    python -m fraude.dashboard

Lee `artifacts/reporte_monitoreo.json` y escribe un HTML autocontenido. El
reporte en JSON sirve para que otro proceso lo consuma; esto es para que una
persona vea en cinco segundos si algo se movio y que.

El tablero esta ordenado por lo que decide una accion, no por lo que es facil de
graficar: primero el volumen de alertas --que es lo que satura al equipo de
fraude-- y recien despues el detalle por variable.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

DIR_ARTEFACTOS = Path(__file__).resolve().parent.parent / "artifacts"

ESTADOS = {
    "SEVERO": ("critical", "Severo"),
    "MODERADO": ("warning", "Moderado"),
    "estable": ("good", "Estable"),
}


def barra_psi(fila: dict, maximo: float) -> str:
    tono, etiqueta = ESTADOS[fila["estado"]]
    ancho = max(fila["psi"] / maximo * 100, 0.6) if maximo else 0.6
    return f"""
      <div class="fila">
        <div class="var">{fila['variable']}</div>
        <div class="pista">
          <div class="barra t-{tono}" style="width:{ancho:.2f}%"></div>
        </div>
        <div class="psi">{fila['psi']:.4f}</div>
        <div class="chip c-{tono}">{etiqueta}</div>
      </div>"""


def par_comparado(titulo: str, antes: float, despues: float, formato: str) -> str:
    escala = max(antes, despues) or 1
    return f"""
      <div class="par">
        <div class="par-titulo">{titulo}</div>
        <div class="par-linea">
          <span class="par-etq">referencia</span>
          <div class="pista pista-sm"><div class="barra t-neutro" style="width:{antes / escala * 100:.1f}%"></div></div>
          <span class="par-val">{formato.format(antes)}</span>
        </div>
        <div class="par-linea">
          <span class="par-etq">diciembre</span>
          <div class="pista pista-sm"><div class="barra t-critical" style="width:{despues / escala * 100:.1f}%"></div></div>
          <span class="par-val">{formato.format(despues)}</span>
        </div>
      </div>"""


def metrica(nombre: str, valor: float) -> str:
    return f"""
      <div class="met">
        <div class="met-cab"><span>{nombre}</span><span class="met-val">{valor:.3f}</span></div>
        <div class="pista pista-sm"><div class="barra t-accent" style="width:{valor * 100:.1f}%"></div></div>
      </div>"""


def construir(reporte: dict) -> str:
    pred = reporte["drift_predicciones"]
    perf = reporte.get("performance")
    factor = pred["tasa_alertas_batch"] / max(pred["tasa_alertas_referencia"], 1e-9)
    drift = reporte["drift_datos"]
    maximo = max(f["psi"] for f in drift)
    con_drift = reporte["variables_con_drift"]

    estado_tono = "critical" if reporte["requiere_atencion"] else "good"
    estado_txt = "Requiere revisión" if reporte["requiere_atencion"] else "Sin alarmas"

    filas = "".join(barra_psi(f, maximo) for f in drift)
    metricas = "".join(metrica(n, perf[k]) for n, k in
                       [("Precisión", "precision"), ("Recall", "recall"),
                        ("F1", "f1"), ("AUC-PR", "auc_pr")]) if perf else ""

    return f"""<title>Monitoreo del Modelo de Fraude</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  --ground:#f7f8fa; --surface:#ffffff; --ink:#0d1117; --ink-soft:#55606e;
  --ink-mute:#8a94a1; --line:#e3e7ec; --line-fuerte:#cbd2db;
  --accent:#2a78d6; --neutro:#8a94a1;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --wash-critical:rgba(208,59,59,.10); --wash-good:rgba(12,163,12,.10);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#101317; --surface:#191d23; --ink:#e8eaed; --ink-soft:#9aa4b2;
    --ink-mute:#6c7684; --line:#262c34; --line-fuerte:#39414b;
    --accent:#3987e5; --neutro:#6c7684;
    --wash-critical:rgba(208,59,59,.16); --wash-good:rgba(12,163,12,.16);
  }}
}}
:root[data-theme="dark"] {{
  --ground:#101317; --surface:#191d23; --ink:#e8eaed; --ink-soft:#9aa4b2;
  --ink-mute:#6c7684; --line:#262c34; --line-fuerte:#39414b;
  --accent:#3987e5; --neutro:#6c7684;
  --wash-critical:rgba(208,59,59,.16); --wash-good:rgba(12,163,12,.16);
}}
* {{ box-sizing:border-box; }}
body {{
  background:var(--ground); color:var(--ink); margin:0;
  font-family:"IBM Plex Sans",system-ui,sans-serif; line-height:1.5;
  font-size:15px; padding:32px 24px 56px;
}}
.wrap {{ max-width:1080px; margin:0 auto; display:flex; flex-direction:column; gap:24px; }}
h1 {{ font-size:26px; font-weight:600; margin:0; letter-spacing:-.01em; }}
h2 {{ font-size:13px; font-weight:600; margin:0 0 16px; text-transform:uppercase;
     letter-spacing:.08em; color:var(--ink-mute); }}
.mono {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums; }}

header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:24px; flex-wrap:wrap; }}
.meta {{ color:var(--ink-soft); font-size:13px; margin-top:6px; }}
.meta span {{ color:var(--ink); }}
.estado {{ display:flex; align-items:center; gap:9px; padding:9px 15px; border-radius:999px;
          font-size:13px; font-weight:600; white-space:nowrap; }}
.estado.critical {{ background:var(--wash-critical); color:var(--critical); }}
.estado.good {{ background:var(--wash-good); color:var(--good); }}
.punto {{ width:8px; height:8px; border-radius:50%; background:currentColor; }}

.tarjeta {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:22px; }}

.destacado {{ display:grid; grid-template-columns:minmax(230px,1fr) 2fr; gap:28px; align-items:center; }}
.cifra {{ font-size:60px; font-weight:600; line-height:1; letter-spacing:-.03em; color:var(--critical); }}
.cifra-pie {{ color:var(--ink-soft); font-size:13px; margin-top:8px; max-width:30ch; }}
.pares {{ display:flex; flex-direction:column; gap:18px; }}
.par-titulo {{ font-size:12px; color:var(--ink-mute); text-transform:uppercase;
              letter-spacing:.07em; margin-bottom:8px; }}
.par-linea {{ display:grid; grid-template-columns:74px 1fr 76px; align-items:center; gap:12px; }}
.par-etq {{ font-size:12px; color:var(--ink-soft); }}
.par-val {{ font-size:13px; text-align:right; font-family:"IBM Plex Mono",monospace;
           font-variant-numeric:tabular-nums; }}

.pista {{ background:var(--line); border-radius:3px; height:9px; overflow:hidden; }}
.pista-sm {{ height:7px; }}
.barra {{ height:100%; border-radius:3px; }}
.t-critical {{ background:var(--critical); }}
.t-warning {{ background:var(--warning); }}
.t-good {{ background:var(--good); }}
.t-accent {{ background:var(--accent); }}
.t-neutro {{ background:var(--neutro); }}

.fila {{ display:grid; grid-template-columns:1fr 200px 68px 84px; gap:14px; align-items:center;
        padding:7px 0; border-bottom:1px solid var(--line); }}
.fila:last-child {{ border-bottom:none; }}
.var {{ font-family:"IBM Plex Mono",monospace; font-size:12.5px; color:var(--ink-soft);
       overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.psi {{ font-family:"IBM Plex Mono",monospace; font-size:12.5px; text-align:right;
       font-variant-numeric:tabular-nums; }}
.chip {{ font-size:11px; font-weight:600; padding:3px 9px; border-radius:5px; text-align:center; }}
.c-critical {{ background:var(--wash-critical); color:var(--critical); }}
.c-warning {{ background:rgba(250,178,25,.16); color:#8a6100; }}
.c-good {{ background:var(--wash-good); color:var(--good); }}
:root[data-theme="dark"] .c-warning, :root:not([data-theme="light"]) .c-warning {{ color:var(--warning); }}

.duo {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
.met {{ margin-bottom:15px; }}
.met:last-child {{ margin-bottom:0; }}
.met-cab {{ display:flex; justify-content:space-between; font-size:13px; margin-bottom:6px;
           color:var(--ink-soft); }}
.met-val {{ color:var(--ink); font-family:"IBM Plex Mono",monospace;
           font-variant-numeric:tabular-nums; }}
.nota {{ color:var(--ink-soft); font-size:13px; line-height:1.65; }}
.nota strong {{ color:var(--ink); font-weight:600; }}
.leyenda {{ display:flex; gap:18px; flex-wrap:wrap; font-size:12px; color:var(--ink-mute);
           margin-top:14px; padding-top:14px; border-top:1px solid var(--line); }}
.leyenda span {{ display:flex; align-items:center; gap:6px; }}
.pip {{ width:9px; height:9px; border-radius:2px; }}
footer {{ color:var(--ink-mute); font-size:12px; text-align:center; }}
@media (max-width:820px) {{
  .destacado, .duo {{ grid-template-columns:1fr; }}
  .fila {{ grid-template-columns:1fr 88px 74px; }}
  .fila .chip {{ display:none; }}
}}
</style>

<div class="wrap">
  <header>
    <div>
      <h1>Monitoreo del modelo de fraude</h1>
      <div class="meta">
        Modelo <span class="mono">v{reporte['modelo_version']}</span> ·
        Lote: <span>{reporte['etiqueta']}</span> ·
        <span class="mono">{reporte['n_transacciones']:,}</span> transacciones
      </div>
    </div>
    <div class="estado {estado_tono}"><span class="punto"></span>{estado_txt}</div>
  </header>

  <div class="tarjeta destacado">
    <div>
      <div class="cifra mono">×{factor:.1f}</div>
      <div class="cifra-pie">Las alertas se multiplicaron respecto del período de
      referencia. Es la restricción operativa: define cuántos casos por día tiene
      que revisar el equipo.</div>
    </div>
    <div class="pares">
      {par_comparado("Tasa de alertas", pred["tasa_alertas_referencia"], pred["tasa_alertas_batch"], "{:.3%}")}
      {par_comparado("Score medio del modelo", pred["score_medio_referencia"], pred["score_medio_batch"], "{:.4f}")}
    </div>
  </div>

  <div class="tarjeta">
    <h2>Estabilidad de las variables de entrada · PSI</h2>
    {filas}
    <div class="leyenda">
      <span><i class="pip t-good"></i>Estable · PSI &lt; 0,10</span>
      <span><i class="pip t-warning"></i>Moderado · 0,10 – 0,25</span>
      <span><i class="pip t-critical"></i>Severo · PSI ≥ 0,25</span>
      <span>{con_drift} de {len(drift)} variables con desplazamiento detectable</span>
    </div>
  </div>

  <div class="duo">
    <div class="tarjeta">
      <h2>Performance con etiquetas reales</h2>
      {metricas}
      <p class="nota" style="margin:16px 0 0">Tasa de fraude observada en el lote:
      <strong>{reporte.get('tasa_fraude_observada', 0):.2%}</strong>.</p>
    </div>
    <div class="tarjeta">
      <h2>Cómo leer este tablero</h2>
      <p class="nota">La performance <strong>mejoró</strong>, y aun así hay alarma. No es
      una contradicción: en diciembre la tasa de fraude es diez veces mayor, así que
      con el mismo modelo la precisión sube sola.</p>
      <p class="nota" style="margin-bottom:0">Lo que se rompe es el <strong>volumen</strong>.
      Un umbral calibrado contra una tasa base veinte veces menor produce ×{factor:.0f} alertas,
      y ninguna capacidad de revisión absorbe eso. La acción no es cambiar el modelo:
      es recalibrar el punto de operación contra lo que el equipo puede revisar.</p>
    </div>
  </div>

  <div class="tarjeta">
    <h2>Advertencia sobre estos números</h2>
    <p class="nota">Las métricas de performance sólo existen porque este es un análisis
    retrospectivo sobre datos ya etiquetados. <strong>En producción no estarían
    disponibles:</strong> la etiqueta real de fraude llega semanas después, vía contracargo
    o reclamo. Hasta entonces las únicas señales son el PSI de las variables de entrada y
    el desplazamiento de los scores — las dos secciones de arriba.</p>
  </div>

  <footer>Generado desde <span class="mono">artifacts/reporte_monitoreo.json</span> ·
  {datetime.now().strftime('%Y-%m-%d %H:%M')}</footer>
</div>
"""


def main() -> Path:
    reporte = json.loads((DIR_ARTEFACTOS / "reporte_monitoreo.json").read_text(encoding="utf-8"))
    salida = DIR_ARTEFACTOS / "dashboard.html"
    salida.write_text(construir(reporte), encoding="utf-8")
    print(f"Tablero escrito en {salida}")
    return salida


if __name__ == "__main__":
    main()
