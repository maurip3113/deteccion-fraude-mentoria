"""Tablero temporal de la simulacion, a partir de `simulacion_linea_tiempo.json`.

    python -m simulacion.tablero

El tablero de `fraude.dashboard` muestra una foto: un periodo contra la
referencia. Este muestra la pelicula, y sobre todo **superpone lo declarado con
lo detectado**: los escenarios inyectados van dibujados como barras arriba, y
justo debajo, en el mismo eje, las alarmas que levanto el monitoreo. Si las dos
filas se alinean, el monitoreo funciono; donde no, hay un retraso o una omision
que se lee de un vistazo.

Todos los graficos comparten geometria horizontal para que se pueda trazar una
vertical imaginaria desde un escenario hasta la metrica que lo delato.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

DIR_ARTEFACTOS = Path(__file__).resolve().parent.parent / "artifacts"

W, ML, MR = 940, 120, 16
PW = W - ML - MR

NOMBRES_CORTOS = {
    "temporada alta": "Temporada alta '26",
    "migracion a e-commerce": "Migración e-commerce",
    "campania de fraude presencial": "Campaña presencial",
    "temporada alta 2027": "Temporada alta '27",
}


def cx(i: int, n: int) -> float:
    return ML + (i + 0.5) * (PW / n)


def etiqueta_mes(periodo: str) -> str:
    anio, mes = periodo.split("-")
    return f"{mes}"


def eje_x(periodos: list[str], y: float) -> str:
    """Meses abajo, con el anio marcado en enero."""
    n = len(periodos)
    partes = [f'<line x1="{ML}" y1="{y - 6}" x2="{ML + PW}" y2="{y - 6}" class="eje"/>']
    for i, p in enumerate(periodos):
        x = cx(i, n)
        partes.append(f'<text x="{x:.1f}" y="{y + 8}" class="tick" text-anchor="middle">'
                      f'{etiqueta_mes(p)}</text>')
        if p.endswith("-01"):
            partes.append(f'<text x="{x:.1f}" y="{y + 22}" class="anio" text-anchor="middle">'
                          f'{p[:4]}</text>')
    return "".join(partes)


def bandas_escenarios(periodos: list[str], declarados: list[dict], alto: float) -> str:
    """Sombreado vertical de fondo, para poder trazar de una fila a otra."""
    n = len(periodos)
    paso = PW / n
    partes = []
    for e in declarados:
        idx = [i for i, p in enumerate(periodos) if e["desde"] <= p <= e["hasta"]]
        if not idx:
            continue
        x0 = ML + idx[0] * paso
        partes.append(f'<rect x="{x0:.1f}" y="0" width="{len(idx) * paso:.1f}" '
                      f'height="{alto}" class="banda"/>')
    return "".join(partes)


def serie(valores: list[float], maximo: float, alto: float, base: float,
          clase: str, n: int) -> str:
    puntos = " ".join(
        f"{cx(i, n):.1f},{base - (v / maximo) * alto:.1f}" for i, v in enumerate(valores)
    )
    return f'<polyline points="{puntos}" class="{clase}"/>'


def puntos(valores: list[float], maximo: float, alto: float, base: float,
           clase: str, n: int, r: float = 3.0) -> str:
    return "".join(
        f'<circle cx="{cx(i, n):.1f}" cy="{base - (v / maximo) * alto:.1f}" r="{r}" class="{clase}"/>'
        for i, v in enumerate(valores)
    )


def grafico(titulo: str, periodos: list[str], series: list[tuple[str, list[float], str]],
            maximo: float, formato, declarados: list[dict],
            lineas_ref: list[tuple[float, str]] | None = None,
            alto_plot: float = 132) -> str:
    """Un grafico de lineas con el eje x compartido y las bandas de escenario."""
    n = len(periodos)
    base = alto_plot + 14
    H = base + 40
    cuerpo = [bandas_escenarios(periodos, declarados, base)]

    # Grilla horizontal y etiquetas del eje y.
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = base - frac * alto_plot
        cuerpo.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" class="grilla"/>')
        cuerpo.append(f'<text x="{ML - 10}" y="{y + 4:.1f}" class="tick" text-anchor="end">'
                      f'{formato(frac * maximo)}</text>')

    for valor, etq in (lineas_ref or []):
        y = base - (valor / maximo) * alto_plot
        cuerpo.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" class="umbral"/>')
        cuerpo.append(f'<text x="{ML + PW - 4}" y="{y - 5:.1f}" class="umbral-t" '
                      f'text-anchor="end">{etq}</text>')

    for _, valores, clase in series:
        cuerpo.append(serie(valores, maximo, alto_plot, base, f"linea {clase}", n))
        cuerpo.append(puntos(valores, maximo, alto_plot, base, f"punto {clase}", n))

    cuerpo.append(eje_x(periodos, base + 14))

    leyenda = ""
    if len(series) > 1:
        piezas = []
        for nombre, _, clase in series:
            piezas.append(f'<span><i class="pip {clase}"></i>{nombre}</span>')
        leyenda = f'<div class="leyenda-g">{"".join(piezas)}</div>'

    return f"""
  <figure>
    <figcaption class="tit-g">{titulo}</figcaption>
    {leyenda}
    <svg viewBox="0 0 {W} {H:.0f}" role="img" aria-label="{titulo}">{''.join(cuerpo)}</svg>
  </figure>"""


def ribbon(periodos: list[str], declarados: list[dict], per: list[dict]) -> str:
    """Escenarios declarados arriba, alarmas y promociones abajo, mismo eje."""
    n = len(periodos)
    paso = PW / n
    filas, y = [], 10
    for e in declarados:
        idx = [i for i, p in enumerate(periodos) if e["desde"] <= p <= e["hasta"]]
        nombre = NOMBRES_CORTOS.get(e["nombre"], e["nombre"])
        filas.append(f'<text x="{ML - 10}" y="{y + 12}" class="tick" text-anchor="end">{nombre}</text>')
        if idx:
            filas.append(f'<rect x="{ML + idx[0] * paso:.1f}" y="{y}" '
                         f'width="{len(idx) * paso:.1f}" height="16" rx="3" class="esc"/>')
        y += 24

    y += 8
    filas.append(f'<text x="{ML - 10}" y="{y + 12}" class="tick-f" text-anchor="end">Alarma</text>')
    for i, p in enumerate(per):
        if p["alarma"]:
            filas.append(f'<rect x="{ML + i * paso + 3:.1f}" y="{y}" width="{paso - 6:.1f}" '
                         f'height="16" rx="3" class="alarma"/>')
    y += 26
    filas.append(f'<text x="{ML - 10}" y="{y + 11}" class="tick-f" text-anchor="end">Modelo</text>')
    for i, p in enumerate(per):
        filas.append(f'<text x="{cx(i, n):.1f}" y="{y + 11}" class="ver" text-anchor="middle">'
                     f'v{p["modelo_version"]}</text>')
        if "promovido_a" in p:
            filas.append(f'<line x1="{ML + i * paso:.1f}" y1="{y - 4}" '
                         f'x2="{ML + i * paso:.1f}" y2="{y + 15}" class="promo"/>')
    H = y + 46
    filas.append(eje_x(periodos, y + 30))

    return f"""
  <figure>
    <figcaption class="tit-g">Lo declarado, y lo que el monitoreo encontró</figcaption>
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Escenarios inyectados y meses en que el monitoreo levanto alarma, sobre el mismo eje temporal.">{''.join(filas)}</svg>
  </figure>"""


def construir(d: dict) -> str:
    per = d["periodos"]
    periodos = [p["periodo"] for p in per]
    v = d["verificacion"]
    decl = d["escenarios_declarados"]

    alertas = [p["tasa_alertas"] * 100 for p in per]
    fraude = [p["tasa_fraude_real"] * 100 for p in per]
    psi = [p["psi_max"] for p in per]
    recall = [p.get("performance", {}).get("recall", 0.0) for p in per]

    def fila_ver(e: dict) -> str:
        if e["detectado"]:
            chip, texto = "ok", "detectado"
        elif not e["detectable_sin_etiquetas"]:
            chip, texto = "neutro", "invisible por diseño"
        else:
            chip, texto = "mal", "no detectado"
        retraso = ("—" if e["retraso_meses"] is None
                   else f"{e['retraso_meses']} mes" + ("es" if e["retraso_meses"] != 1 else ""))
        deg = e.get("degradacion")
        if deg and deg["degradado"]:
            caida = f"−{deg['caida_relativa']*100:.0f}% de F1"
        elif deg:
            caida = "sin caída"
        else:
            caida = "—"
        nota = ('<div class="nota-f">hubo alarma en la ventana, pero la disparó otro '
                'escenario</div>' if e.get("alarma_ajena") else "")
        return (f'<tr><td>{NOMBRES_CORTOS.get(e["escenario"], e["escenario"])}{nota}</td>'
                f'<td class="n">{e["meses_atribuibles"]}/{e["meses"]}</td>'
                f'<td class="n">{retraso}</td><td class="n">{caida}</td>'
                f'<td><span class="chip {chip}">{texto}</span></td></tr>')

    filas_ver = "".join(fila_ver(e) for e in v["escenarios"])

    return f"""<title>Simulación del Monitoreo</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{
  --ground:#f7f8fa; --surface:#ffffff; --ink:#0d1117; --ink-2:#55606e; --ink-3:#8a94a1;
  --line:#e3e7ec; --line-2:#cbd2db;
  --s1:#2a78d6; --s2:#eb6834; --good:#0ca30c; --critical:#d03b3b; --warning:#fab219;
  --banda:rgba(42,120,214,.055); --wash-ok:rgba(12,163,12,.10); --wash-mal:rgba(208,59,59,.10);
}}
@media (prefers-color-scheme:dark){{
  :root:not([data-theme="light"]){{
    --ground:#101317; --surface:#191d23; --ink:#e8eaed; --ink-2:#9aa4b2; --ink-3:#6c7684;
    --line:#262c34; --line-2:#39414b;
    --s1:#3987e5; --s2:#d95926;
    --banda:rgba(57,135,229,.10); --wash-ok:rgba(12,163,12,.16); --wash-mal:rgba(208,59,59,.16);
  }}
}}
:root[data-theme="dark"]{{
  --ground:#101317; --surface:#191d23; --ink:#e8eaed; --ink-2:#9aa4b2; --ink-3:#6c7684;
  --line:#262c34; --line-2:#39414b;
  --s1:#3987e5; --s2:#d95926;
  --banda:rgba(57,135,229,.10); --wash-ok:rgba(12,163,12,.16); --wash-mal:rgba(208,59,59,.16);
}}
*{{box-sizing:border-box}}
body{{background:var(--ground);color:var(--ink);margin:0;
  font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.6;
  padding:34px 22px 56px}}
.wrap{{max-width:1000px;margin:0 auto;display:flex;flex-direction:column;gap:22px}}
h1{{font-size:27px;font-weight:600;margin:0;letter-spacing:-.015em}}
.meta{{color:var(--ink-2);font-size:13.5px;margin-top:7px;max-width:70ch}}
h2{{font-size:12.5px;font-weight:600;margin:0 0 14px;text-transform:uppercase;
   letter-spacing:.09em;color:var(--ink-3)}}
section{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:22px 24px}}
p{{margin:0 0 11px;max-width:72ch}}p:last-child{{margin-bottom:0}}
strong{{font-weight:600}}
.mono{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}

.tiras{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px}}
.tira{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px}}
.tira .v{{font-size:30px;font-weight:600;letter-spacing:-.02em;
  font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
.tira .e{{color:var(--ink-3);font-size:12.5px;margin-top:3px}}
.v.ok{{color:var(--good)}}.v.al{{color:var(--critical)}}

figure{{margin:0 0 26px}}figure:last-child{{margin-bottom:0}}
figure svg{{display:block;width:100%;height:auto}}
.tit-g{{font-size:13.5px;font-weight:600;color:var(--ink);margin-bottom:8px}}
.leyenda-g{{display:flex;gap:16px;font-size:12px;color:var(--ink-2);margin-bottom:6px}}
.leyenda-g span{{display:flex;align-items:center;gap:6px}}
.pip{{width:18px;height:3px;border-radius:2px;display:inline-block}}
.pip.c1{{background:var(--s1)}}.pip.c2{{background:var(--s2)}}

.grilla{{stroke:var(--line);stroke-width:1}}
.eje{{stroke:var(--line-2);stroke-width:1}}
.umbral{{stroke:var(--warning);stroke-width:1.2;stroke-dasharray:4 3}}
.umbral-t{{fill:var(--ink-3);font-family:"IBM Plex Mono",monospace;font-size:9.5px}}
.tick{{fill:var(--ink-3);font-family:"IBM Plex Sans",sans-serif;font-size:10px}}
.tick-f{{fill:var(--ink-2);font-family:"IBM Plex Sans",sans-serif;font-size:11px;font-weight:500}}
.anio{{fill:var(--ink-2);font-family:"IBM Plex Sans",sans-serif;font-size:10.5px;font-weight:600}}
.ver{{fill:var(--ink-3);font-family:"IBM Plex Mono",monospace;font-size:9.5px}}
.banda{{fill:var(--banda)}}
.esc{{fill:var(--s1);opacity:.75}}
.alarma{{fill:var(--critical)}}
.promo{{stroke:var(--good);stroke-width:2}}
.linea{{fill:none;stroke-width:2}}
.punto{{stroke:var(--surface);stroke-width:1.5}}
.c1.linea{{stroke:var(--s1)}}.c1.punto{{fill:var(--s1)}}
.c2.linea{{stroke:var(--s2)}}.c2.punto{{fill:var(--s2)}}

table{{width:100%;border-collapse:collapse;font-size:14px}}
th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
   color:var(--ink-3);font-weight:600;padding:0 10px 9px 0;border-bottom:1px solid var(--line-2)}}
td{{padding:10px 10px 10px 0;border-bottom:1px solid var(--line);vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
td.n{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
.chip{{font-size:11.5px;font-weight:600;padding:3px 9px;border-radius:5px}}
.chip.ok{{background:var(--wash-ok);color:var(--good)}}
.chip.mal{{background:var(--wash-mal);color:var(--critical)}}
.chip.neutro{{background:var(--line);color:var(--ink-2)}}
.nota-f{{color:var(--ink-3);font-size:11.5px;margin-top:3px;max-width:34ch}}
footer{{color:var(--ink-3);font-size:12px;text-align:center}}
@media(max-width:640px){{body{{padding:22px 12px 40px}}h1{{font-size:22px}}}}
</style>

<div class="wrap">
  <header>
    <h1>Simulación del monitoreo</h1>
    <p class="meta">Dos años sintéticos con cambios <strong>declarados de antemano</strong>.
    Como se sabe qué se inyectó y cuándo, se puede medir si el monitoreo lo encontró — y,
    lo que casi nunca se prueba, si se calla cuando no pasa nada.</p>
  </header>

  <div class="tiras">
    <div class="tira"><div class="v ok">{v['detectados']}/{v['total_esperables']}</div>
      <div class="e">detectados de los {v['total_esperables']} visibles sin etiquetas
      ({v['invisibles_por_diseno']} sólo se ve con etiquetas)</div></div>
    <div class="tira"><div class="v {'ok' if v['falsos_positivos'] == 0 else 'al'}">{v['falsos_positivos']}</div>
      <div class="e">falsos positivos en {v['meses_tranquilos']} meses tranquilos</div></div>
    <div class="tira"><div class="v">{d['promociones']}</div>
      <div class="e">modelos promovidos</div></div>
    <div class="tira"><div class="v">{d['lag_etiquetas_meses']}</div>
      <div class="e">meses de retraso de las etiquetas</div></div>
  </div>

  <section>
    <h2>Declarado contra detectado</h2>
    {ribbon(periodos, decl, per)}
    <p style="color:var(--ink-2);font-size:13.5px">Las barras de arriba son los escenarios
    que se inyectaron; los bloques rojos, los meses en que el monitoreo pidió intervención.
    Las marcas verdes señalan dónde se promovió un modelo nuevo.</p>
  </section>

  <section>
    <h2>Qué vio el monitoreo</h2>
    {grafico("Tasa de alertas y tasa de fraude real (% de transacciones)", periodos,
             [("Alertas del modelo", alertas, "c1"), ("Fraude real (no observable en el momento)", fraude, "c2")],
             6.0, lambda x: f"{x:.0f}%", decl)}
    {grafico("PSI máximo entre las variables de entrada", periodos,
             [("PSI", psi, "c1")], 0.5, lambda x: f"{x:.2f}", decl,
             lineas_ref=[(0.10, "moderado 0,10"), (0.25, "severo 0,25")])}
    {grafico("Recall: qué proporción del fraude atrapó el modelo", periodos,
             [("Recall", recall, "c1")], 0.7, lambda x: f"{x:.1f}", decl)}
  </section>

  <section>
    <h2>Verificación</h2>
    <table>
      <tr><th>Escenario</th><th>Meses atribuibles</th><th>Retraso</th><th>Performance</th><th>Resultado</th></tr>
      {filas_ver}
    </table>
    <p style="margin-top:16px;color:var(--ink-2);font-size:13.5px">Un escenario cuenta como
    detectado sólo si el <strong>disparador de la alarma corresponde a su firma declarada</strong>.
    Que una alarma caiga dentro de su ventana no alcanza: en agosto de 2027 se superponen dos
    escenarios, y la alarma la disparó el PSI de la migración, no la campaña. La campaña es
    concept drift puro — no deja rastro en las entradas y sólo aparece cuando llegan las
    etiquetas, con una caída del 73% en F1.</p>
    <p style="color:var(--ink-2);font-size:13.5px">El otro dato que importa es el
    retraso. Los cambios abruptos se detectan el mismo mes; la migración a e-commerce es un
    drift <strong>gradual</strong> y el PSI tardó cuatro meses en cruzar 0,25. Eso no se
    puede saber sin un banco de pruebas con respuesta conocida.</p>
  </section>

  <footer>Generado desde <span class="mono">artifacts/simulacion_linea_tiempo.json</span> ·
  {datetime.now().strftime('%Y-%m-%d %H:%M')}</footer>
</div>
"""


def main() -> Path:
    d = json.loads((DIR_ARTEFACTOS / "simulacion_linea_tiempo.json").read_text(encoding="utf-8"))
    salida = DIR_ARTEFACTOS / "tablero_simulacion.html"
    salida.write_text(construir(d), encoding="utf-8")
    print(f"Tablero temporal escrito en {salida}")
    return salida


if __name__ == "__main__":
    main()
