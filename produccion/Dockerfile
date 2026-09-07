# Misma version menor de Python con la que se entreno y serializo el modelo.
FROM python:3.13-slim

WORKDIR /app

# Las dependencias en una capa aparte: cambian mucho menos que el codigo,
# asi que reconstruir la imagen tras editar un .py no vuelve a instalarlas.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fraude/ ./fraude/
COPY artifacts/ ./artifacts/

# Usuario sin privilegios: si alguien escapa del proceso, no es root.
RUN useradd --create-home --shell /bin/bash servicio && chown -R servicio:servicio /app
USER servicio

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "fraude.api:app", "--host", "0.0.0.0", "--port", "8000"]
