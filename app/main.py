

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

##import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.artifacts import get_artifacts
from app.inference import run_batch, run_single
from app.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)

# ── Structured logging ────────────────────────────────────────────────────────
"""
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()
logging.basicConfig(level=logging.INFO)
"""
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total predictions made",
    ["prediction_label"],          # "fraud" or "legitimate"
)
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
BATCH_SIZE = Histogram(
    "fraud_batch_size",
    "Batch prediction size",
    buckets=[1, 5, 10, 50, 100, 500, 1000],
)


# ── Lifespan — runs on startup and shutdown ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
   
    log.info("startup.begin")
    try:
        get_artifacts()                       # loads model, preprocessor, feature_engineer
        log.info("startup.complete — ready")
        log.info("startup.begin")
    except Exception as e:
        log.error(f"startup.failed: {e}")
        raise
    yield
    log.info("shutdown")                      # This runs when: pressing CTRL+C


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost fraud scoring service — credit card transactions",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc UI
)
  
app.add_middleware(                       #This enables CORS = Cross-Origin Resource Sharing
    CORSMiddleware,                       #allows the frontend to call the API from another origin
    allow_origins=["*"],                  # Restrict in production via env var, Allow requests from any website
    allow_methods=["GET", "POST"],
    allow_headers=["*"],                  # Allows all request headers: Authorization, Content-Type, X-API-Key

)


# ── Request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)          # means Continue processing the request and call the appropriate endpoint
    latency = (time.perf_counter() - t0) * 1000   #request duration
    #log.info(f"{request.method} {request.url.path} {response.status_code} {round(latency, 2)}ms")  # produces exmple : POST /predict 200 18.7ms 
   
    log.info(
        "http.request",
        #"http.request | method=%s path=%s status_code=%s latency_ms=%s",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency, 2),
        },   
        
    
        #method=request.method,
        #path=request.url.path,
        #status=response.status_code,
        #latency_ms=round(latency, 2),
    
    )
    
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    
   
    try:
        arts = get_artifacts()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=arts["model_version"],
            threshold=arts["threshold"],
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {e}")


@app.get("/info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info():
   
    arts = get_artifacts()
    return ModelInfoResponse(
        model_version=arts["model_version"],
        threshold=arts["threshold"],
        metrics=arts["metrics"],
        feature_count=len(arts["feature_names"]),
        features=arts["feature_names"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: TransactionRequest):
 
    try:
        arts = get_artifacts()
        t0 = time.perf_counter()

        result = run_single(request.model_dump(), arts)

        latency = time.perf_counter() - t0
        PREDICTION_LATENCY.observe(latency)
        label = "fraud" if result["prediction"] == 1 else "legitimate"
        PREDICTIONS_TOTAL.labels(prediction_label=label).inc()

        log.info(f"predict.single | pred={result['prediction']} | prob={result['probability']}")
        return PredictionResponse(**result)

    except Exception as e:
        log.error(f"predict.error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
async def predict_batch(body: BatchTransactionRequest):
   
    if len(body.transactions) == 0:
        raise HTTPException(status_code=422, detail="Empty batch")
    if len(body.transactions) > 1000:
        raise HTTPException(status_code=422, detail="Max batch size is 1000")

    try:
        arts = get_artifacts()
        t0 = time.perf_counter()

        requests_dicts = [t.model_dump() for t in body.transactions]
        result = run_batch(requests_dicts, arts)

        latency = time.perf_counter() - t0
        PREDICTION_LATENCY.observe(latency)
        BATCH_SIZE.observe(len(body.transactions))
        PREDICTIONS_TOTAL.labels(prediction_label="fraud").inc(result["fraud_count"])
        PREDICTIONS_TOTAL.labels(prediction_label="legitimate").inc(
            result["total"] - result["fraud_count"]
        )

        log.info(f"predict.batch | total={result['total']} | fraud={result['fraud_count']}")
        
        return BatchPredictionResponse(**result)

    except Exception as e:
        log.error(f"predict.batch.error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["ops"])
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
