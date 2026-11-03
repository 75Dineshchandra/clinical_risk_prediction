# The image that Render builds (render.yaml points here). Build context: the repository root.
# Unlike serving/Dockerfile (used by docker-compose, with the model mounted as a folder), this image has the
# model INSIDE it (deploy/bundle), because a hosted service has no mounted folders.
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BUNDLE_DIR=/app/bundle \
    PREDICTION_LOG=/app/logs/predictions.jsonl

WORKDIR /app

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# requirements first, so Docker's layer cache skips the slow install when only the code changes
COPY serving/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY serving/app.py ./
COPY serving/ui ./ui
COPY deploy/bundle ./bundle

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser /app
USER appuser

# Render tells the service which port to use in $PORT (default 10000). One worker: the Prometheus counters live in the process.
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
