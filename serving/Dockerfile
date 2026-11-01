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

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app.py ./
COPY ui ./ui

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/bundle /app/logs \
    && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# one worker: the Prometheus counters live inside the process
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]