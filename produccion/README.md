# Puesta en producción del modelo de detección de fraude

Este directorio toma el modelo que quedó entrenado en los notebooks del proyecto y lo convierte en un servicio: un artefacto versionado, una API que lo consulta en tiempo real, un chequeo de *drift* y un ciclo de reentrenamiento que decide si conviene reemplazarlo.

La razón de que exista es que un modelo de riesgo no se entrega una vez. La parte difícil no es alcanzar un F1 aceptable en un notebook — es sostenerlo cuando los datos que llegan dejan de parecerse a los del entrenamiento. Los notebooks del proyecto llegan hasta el modelo entrenado; acá empieza lo que viene después.

## El ciclo de vida

```mermaid
flowchart LR
    A[Muestra.del<br/>transacciones crudas] --> B[features.py<br/>ingeniería de variables]
    B --> C[train.py<br/>ajuste + validación]
    C --> D[(artifacts/<br/>modelo_vN.joblib)]
    D --> E[api.py<br/>POST /predict]
    E --> F[revisión humana<br/>equipo de fraude]
    E --> G[predicciones.jsonl<br/>+ GET /metrics]
    G --> H[monitoring.py<br/>PSI · KS · performance]
    F -->|etiquetas reales<br/>semanas después| I[retrain.py<br/>champion vs challenger]
    H -->|alarma de drift| I
    I -->|solo si gana<br/>por margen| D
```

El lazo que importa es el de abajo: las etiquetas reales de fraude no existen en el momento de la transacción, llegan semanas después vía contracargo o reclamo. Todo el diseño está condicionado por eso.

## Cómo correrlo

```bash
cd produccion
pip install -r requirements.txt
```

```bash
python -m fraude.train
```
Reconstruye el histórico, entrena y deja `artifacts/modelo_v1.joblib` con su metadata.

```bash
uvicorn fraude.api:app --reload
```
Levanta la API en `http://localhost:8000` — documentación interactiva en `/docs`.

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"cliente_id":101,"trx_timestamp":"2025-12-24T03:47:00","trx_importe":185000.0,"trx_moneda":1,"trx_rubro_red":5311,"trx_tipo_terminal":3,"cliente_fecha_nacimiento":"1985-04-12","cliente_sexo":"MASCULINO","cliente_estado_civil":"SOLTERO/A","cliente_segmento":"S3"}'
```

```bash
python -m fraude.monitoring --demo
```
Compara diciembre contra el resto del año y reporta drift.

```bash
python -m fraude.retrain
```
Entrena un challenger con datos más recientes y decide si merece reemplazar al modelo en producción.

```bash
python -m pytest tests/ -q
```

```bash
docker build -t fraude-api . && docker run -p 8000:8000 fraude-api
```

## Decisiones de diseño

**Un solo lugar define las variables.** `features.py` lo usan tanto el entrenamiento como la API. Si cada camino armara las features por su cuenta, el modelo vería en producción una distribución distinta a la que aprendió — *training/serving skew*, la falla más común y más silenciosa al desplegar. El test `test_estado_del_cliente_replica_la_ventana_expansiva` compara el cálculo online contra el de pandas justamente para que esa equivalencia no se rompa sin que nadie lo note.

**La API necesita memoria del cliente.** Tres de las 23 variables (`Cliente_Trx_Count`, `Tiempo_Entre_Trx_Horas`, `Desvio_Importe_Cliente_Abs`) no se pueden calcular mirando sólo la transacción que llega: dependen de todo lo que el cliente hizo antes. En un banco eso vive en un *feature store*; acá lo resuelve `EstadoClientes`, que mantiene media y varianza corrientes con el algoritmo de Welford y reproduce exactamente la ventana expansiva del TP2 sin guardar el histórico completo en memoria.

**El umbral se elige en validación, no en evaluación.** Los notebooks eligen el umbral que maximiza F1 mirando el conjunto de test, y después reportan el F1 sobre ese mismo conjunto. Eso infla el número: el umbral ya vio los datos con los que se lo mide. Acá el histórico se parte en tres — entrenamiento, validación y evaluación — y el umbral sale de validación.

**Alertar no es bloquear.** La respuesta de `/predict` incluye una acción (`revision_humana` / `aprobar`), no un bloqueo. Con una precisión del 66%, bloquear automáticamente significa frenar una transacción legítima de cada tres alertas.

**Reemplazar el modelo exige ganar por un margen.** `retrain.py` promueve el challenger sólo si mejora el F1 en más de 0,01. Cambiar el modelo tiene un costo operativo — revalidación, aviso al equipo de fraude, recalibración de umbrales — que una diferencia del tamaño del ruido no justifica.

**La partición del reentrenamiento es temporal, no aleatoria.** Con partición aleatoria el reentrenamiento siempre parece funcionar, porque entrenamiento y evaluación comparten el mismo período. El punto de reentrenar es responder a que el mundo cambió con el tiempo, y eso sólo se ve evaluando hacia adelante.

## Lo que apareció al hacer esto

**El F1 del proyecto estaba optimista, por dos motivos independientes.** Con el umbral elegido en validación en vez de en test, el F1 de evaluación queda en **0,601** (precisión 0,66 / recall 0,55 / AUC-PR 0,620), contra el 0,653 que reportan los notebooks. Parte de esa diferencia es el umbral; la otra parte es más seria y aparece abajo.

**El fraude de este dataset está concentrado al final del año.**

| Mes | ene–ago | sep | oct | nov | dic |
|---|---|---|---|---|---|
| Tasa de fraude | 0,05 % | 0,10 % | 0,25 % | 1,28 % | 6,22 % |

De los 1.196 fraudes del año, 1.076 caen en noviembre y diciembre. Una partición aleatoria le permite al modelo entrenar con fraude de diciembre y después evaluarse contra fraude de diciembre — que es exactamente lo que nunca va a poder hacer en producción.

**Evaluado como se lo usaría de verdad, el modelo rinde mucho menos.** Entrenando con enero–agosto y evaluando sobre noviembre–diciembre, el F1 cae de 0,601 a **0,145**. Y agregarle septiembre–octubre al entrenamiento lo empeora todavía más en F1 (0,032) aunque le mejore el AUC-PR (+0,027).

**Esa contradicción es el hallazgo central.** El AUC-PR sube — el modelo ordena *mejor* las transacciones por riesgo — mientras el F1 se desploma. La diferencia entera está en el umbral: con el umbral óptimo del propio período, el mismo challenger daría 0,274 en vez de 0,032. Es decir, **pierde 0,242 de F1 sólo por tener un umbral calibrado contra una tasa base veinte veces menor a la que enfrenta**.

La conclusión operativa no es "reentrenar más seguido". Es que el umbral no puede ser una constante estimada una vez: tiene que re-derivarse periódicamente, y en un caso como diciembre — donde `monitoring.py` detecta que el volumen de alertas se multiplica por 24,65 — la restricción real que manda no es el F1 histórico sino cuántos casos por día puede revisar el equipo de fraude.

## Lo que falta para que esto sea producción de verdad

Vale la pena ser explícito sobre el límite de este ejercicio:

- **El estado de clientes vive en memoria del proceso.** Se reinicia con el servicio y no se comparte entre réplicas. En producción va a Redis, DynamoDB o un feature store gestionado.
- **No hay autenticación, rate limiting ni trazas distribuidas.**
- **El despliegue está containerizado pero no desplegado.** El paso natural es un endpoint de SageMaker (serverless para este volumen) o el contenedor en ECS/Fargate detrás de API Gateway.
- **El reentrenamiento se dispara a mano.** Automatizarlo es un scheduler (EventBridge → SageMaker Pipeline o Step Functions), no un cambio de lógica.
- **No hay registro formal de modelos.** `modelo_actual.txt` alcanza para un proyecto; en producción es SageMaker Model Registry o MLflow, con aprobación explícita para promover.
- **El monitoreo escribe reportes, no dispara alertas.** Falta publicar las métricas a CloudWatch y conectar los umbrales de PSI a una notificación real.

## Estructura

```
produccion/
├── fraude/
│   ├── features.py     # ingeniería de variables compartida + estado por cliente
│   ├── train.py        # entrenamiento, validación y serialización versionada
│   ├── api.py          # servicio FastAPI de scoring
│   ├── monitoring.py   # PSI, KS y performance con etiquetas reales
│   └── retrain.py      # champion/challenger con partición temporal
├── artifacts/          # modelos versionados, metadata y logs
├── tests/
├── Dockerfile
└── requirements.txt
```
