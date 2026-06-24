FROM python:3.14.4-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .  
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.14.4-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /home/appuser

COPY --from=builder /install /usr/local

COPY app/      ./app/
COPY src/      ./src/
COPY artifacts/ ./artifacts/
COPY configs/  ./configs/

USER appuser

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info"]