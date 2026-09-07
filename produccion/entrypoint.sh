#!/bin/sh
# SageMaker arranca la imagen como `docker run <imagen> serve`, asi que el
# argumento llega igual y se ignora: la app se sirve siempre del mismo modo.
exec uvicorn fraude.api:app --host 0.0.0.0 --port "${PORT:-8080}"
