# Misma version menor de Python con la que se entreno y serializo el modelo.
FROM python:3.13-slim

WORKDIR /app

# Las dependencias en una capa aparte: cambian mucho menos que el codigo,
# asi que reconstruir la imagen tras editar un .py no vuelve a instalarlas.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fraude/ ./fraude/
COPY artifacts/ ./artifacts/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Usuario sin privilegios: si alguien escapa del proceso, no es root.
RUN useradd --create-home --shell /bin/bash servicio && chown -R servicio:servicio /app
USER servicio

# 8080 es el puerto que exige un endpoint de SageMaker con contenedor propio.
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f\"http://localhost:{os.environ['PORT']}/ping\")"

ENTRYPOINT ["/app/entrypoint.sh"]
