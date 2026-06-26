# 🛡️ Credit Card Fraud Detection System

An end-to-end **production-grade MLOps pipeline** for real-time credit card fraud detection. This project covers the full ML lifecycle: from exploratory analysis and feature engineering, through model training with automated hyperparameter tuning, to containerized deployment on Kubernetes with live monitoring.

> **Note on infrastructure:** AWS EKS is the production-target deployment (CI/CD pipeline included). Due to not having an AWS account yet, the deployment was validated locally using **Minikube** — which runs the same Kubernetes manifests, proving real cloud-deployment readiness without the cost. The architecture, configs, and k8s files are fully AWS-compatible.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Step-by-Step: What Was Built](#step-by-step-what-was-built)
  - [Step 0 — Notebooks: Exploratory Prototyping](#step-0--notebooks-exploratory-prototyping)
  - [Step 1 — Exploratory Data Analysis](#step-1--exploratory-data-analysis)
  - [Step 2 — Data Preprocessing](#step-2--data-preprocessing)
  - [Step 3 — Feature Engineering](#step-3--feature-engineering)
  - [Step 4 — Class Balancing with SMOTE (Inside Cross-Validation)](#step-4--class-balancing-with-smote-inside-cross-validation)
  - [Step 5 — Model Training with Optuna](#step-5--model-training-with-optuna)
  - [Step 6 — Evaluation & Threshold Tuning](#step-6--evaluation--threshold-tuning)
  - [Step 7 — Data Drift Detection](#step-7--data-drift-detection)
  - [Step 8 — MLflow Experiment Tracking & Model Registry](#step-8--mlflow-experiment-tracking--model-registry)
  - [Step 9 — FastAPI Serving Layer](#step-9--fastapi-serving-layer)
  - [Step 10 — Streamlit Dashboard](#step-10--streamlit-dashboard)
  - [Step 11 — Containerization with Docker](#step-11--containerization-with-docker)
  - [Step 12 — Kubernetes Deployment (Minikube / AWS-ready)](#step-12--kubernetes-deployment-minikube--aws-ready)
  - [Step 13 — Monitoring with Prometheus & Grafana](#step-13--monitoring-with-prometheus--grafana)
  - [Step 14 — CI/CD with GitHub Actions](#step-14--cicd-with-github-actions)
- [Model Results](#model-results)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Why This Project Matters](#why-this-project-matters)

---

## Project Overview

| Property | Value |
|---|---|
| **Domain** | Financial Fraud Detection |
| **Task** | Binary Classification (Anomaly Detection) |
| **Dataset** | Kaggle Credit Card Fraud (284,807 transactions) |
| **Class Imbalance** | 99.83% legitimate / 0.17% fraud (1:577 ratio) |
| **Model** | XGBoost with Optuna-tuned hyperparameters |
| **Serving** | FastAPI REST API (single + batch prediction) |
| **Deployment** | Docker → Kubernetes (Minikube locally, AWS EKS-ready) |
| **Monitoring** | Prometheus metrics + Grafana dashboards |
| **Experiment Tracking** | MLflow |

---

## Architecture

```
notebooks/                         scripts/
01_eda.ipynb                       run_pipeline.py
02_feature_engineering.ipynb       train.py
(prototype & explore)              predict.py
         │                              │
         └──────────────┬───────────────┘
                        ▼
              data/raw/creditcard.csv
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│               DATA PIPELINE                     │
│  DataIngestion → DataPreprocessor → FeatureEngineer → FeatureSelector │
│              data/interim/   data/processed/    │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│            TRAINING PIPELINE                    │
│  Optuna HPO → XGBoost → Threshold Tuning        │
│  Cross-Validation → PSI Drift Check → MLflow    │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
           ┌───────────────────────────┐
           │   Saved Artifacts         │
           │  model.pkl │ preprocessor │
           │  feature_engineer.pkl     │
           │  threshold_analysis.json  │
           └──────────────┬────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│              SERVING LAYER                      │
│   FastAPI  (/predict │ /predict/batch │ /health) │
│   Pydantic validation + Prometheus metrics      │
└──────────────────────────┬──────────────────────┘
                           │
          ┌────────────────┼──────────────────┐
          ▼                ▼                  ▼
   Docker Image      Streamlit UI      Prometheus
   (multi-stage)     (Dashboard)       + Grafana
          │
          ▼
  Kubernetes (Minikube / AWS EKS)
  ┌──────────────────────────────────┐
  │  Deployment + HPA (auto-scale)   │
  │  NodePort Service                │
  │  Liveness / Readiness Probes     │
  │  Rolling Update Strategy         │
  └──────────────────────────────────┘
          │
          ▼
  GitHub Actions CI/CD
  (test → build → push to Docker Hub)
```

---

## Project Structure

```
fraud-detection-system/
│
├── .github/workflows/
│   ├── deploy.yml                # GitHub Actions: test → build → push Docker Hub
│   └── AWS/deploy.yml            # AWS variant: push to ECR + kubectl rollout
│
├── app/                          # FastAPI application
│   ├── main.py                   # Routes, middleware, Prometheus metrics
│   ├── inference.py              # Single and batch scoring logic
│   ├── artifacts.py              # LRU-cached artifact loader
│   └── schemas.py                # Pydantic request/response models
│
├── artifacts/
│   ├── models/                   # Trained model + version file + metadata
│   ├── preprocessing/            # Fitted preprocessor + scaler
│   ├── feature_engineering/      # Fitted FeatureEngineer + feature names
│   ├── evaluation/               # Metrics, confusion matrix, threshold analysis
│   └── figures/                  # 13 EDA plots generated during exploration
│
├── configs/
│   ├── config.yaml               # Training, MLflow, artifact paths
│   ├── data_config.yaml          # Preprocessing, FE, selection params
│   └── model_config.yaml         # XGBoost defaults
│
├── k8s/
│   ├── minikube/                 # Local Kubernetes manifests (tested)
│   │   ├── deployment.yaml       # Deployment + probes + security context
│   │   ├── service.yaml          # NodePort service
│   │   ├── hpa.yaml              # HorizontalPodAutoscaler
│   │   └── monitoring.yaml       # ConfigMap
│   └── (AWS-ready manifests)     # Same structure, LoadBalancer service type
│
├── monitoring/
│   ├── prometheus.yml            # Scrape configs
│   └── grafana/                  # Dashboard JSON + datasource provisioning
│
├── notebooks/
│   ├── 01_eda.ipynb              # Data exploration, imbalance analysis, EDA figures
│   └── 02_feature_engineering.ipynb  # Feature prototyping before modularization
│
├── scripts/
│   ├── run_pipeline.py           # Entry point: runs data pipeline + training end-to-end
│   ├── train.py                  # Standalone training script
│   ├── predict.py                # Offline batch inference from command line
│   └── aws_setup.sh              # AWS ECR + EKS setup commands (future deployment)
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py     # CSV/Parquet loader + overview report
│   │   └── data_preprocessing.py # Dedup, dtype fix, imputation, outlier treatment
│   ├── features/
│   │   ├── build_features.py     # FeatureEngineer (time, amount, scaling)
│   │   └── feature_selection.py  # Variance, correlation, RFE (reserved for linear models)
│   ├── models/
│   │   ├── train_model.py        # Optuna HPO objective
│   │   ├── evaluate_model.py     # Threshold tuning, CV, feature importance
│   │   ├── model_registry.py     # Save, version, MLflow registration
│   │   └── predict_model.py      # Standalone prediction utilities
│   ├── pipelines/
│   │   ├── data_pipeline.py      # End-to-end data preparation
│   │   ├── training_pipeline.py  # Full training + MLflow run
│   │   └── inference_pipeline.py # Offline batch inference
│   ├── utils/
│   │   ├── config.py             # YAML config loader + model versioning
│   │   ├── helpers.py            # PSI drift calculator
│   │   └── logger.py             # Structured logging setup
│   └── visualization/
│       ├── eda_plots.py          # EDA charts (13 figures)
│       ├── dashboard.py          # Dashboard data preparation
│       └── save_dashboard_artifacts.py  # Serializes data to data/dashboard/
│
├── streamlit/
│   └── app.py                    # Interactive prediction dashboard
│
├── tests/
│   ├── test_api.py               # FastAPI endpoint tests (mocked artifacts)
│   ├── test_data.py              # Data pipeline unit tests
│   ├── test_features.py          # Feature engineering tests
│   └── test_model.py             # Model inference tests
│
├── .dockerignore
├── .gitignore
├── Dockerfile                    # Multi-stage build
├── Makefile                      # Convenience commands
├── README.md
├── docker-compose.yml            # API + Streamlit + Prometheus + Grafana
└── requirements.txt
```

---

## Step-by-Step: What Was Built

### Step 0 — Notebooks: Exploratory Prototyping

**What:** Before writing any production code, two Jupyter notebooks were used to explore the data and prototype ideas interactively.

- **`notebooks/01_eda.ipynb`** — loaded the raw `creditcard.csv`, profiled the 284,807-row dataset, visualized the class imbalance, explored feature distributions per class, checked correlations, and generated the 13 EDA figures now saved in `artifacts/figures/`. This notebook was the decision-making space for everything that followed.

- **`notebooks/02_feature_engineering.ipynb`** — experimented with time decomposition (hour, day, cyclical encoding), amount transformations (log, zscore, sqrt, quantile binning), and interaction features. Confirmed which engineered features were meaningful before committing them to the production `FeatureEngineer` class in `src/features/build_features.py`.

**Why notebooks first:** Notebooks are the right tool for open-ended exploration — fast iteration, inline visualization, and no need for a full class structure yet. Once the ideas were validated, they were refactored into reusable, testable, sklearn-compatible classes in `src/`. The notebooks remain in the repo as a transparent record of the thinking process.

**Scripts — the other entry points:** The `scripts/` folder contains the command-line entry points that wire the pipeline together for reproducibility:
- `scripts/run_pipeline.py` — runs the full data pipeline (ingestion → preprocessing → feature engineering → splits) followed by model training, with an optional `--smoke-test` flag that runs inference on the validation set afterward
- `scripts/train.py` — standalone training script, calls `training_pipeline.py` directly
- `scripts/predict.py` — offline batch inference from the command line, useful for scoring new CSV files without spinning up the API
- `scripts/aws_setup.sh` — shell commands to provision ECR repository and configure EKS credentials, ready for when an AWS account is available

---

### Step 1 — Exploratory Data Analysis

**What:** Ran a thorough EDA on the 284,807-row credit card dataset to understand the data before touching any model.

**What was found and done:**
- The dataset has a **577:1 class imbalance** (legitimate vs fraud) — a naive model that predicts everything as legitimate would achieve 99.83% accuracy, making accuracy useless as a metric. This shaped every decision from here on.
- Generated **13 EDA figures** covering class distribution, feature histograms per class, boxplots, correlation heatmap, amount distribution, time analysis, and skewness profiles.
- Found that fraud transactions are slightly correlated with lower transaction amounts and specific time windows.
- Detected 1,081 duplicate rows, zero missing values.
- Wrote a full `overview_report()` in `DataIngestion` capturing shape, dtypes, missing values, duplicates, class counts, and imbalance ratio — all serialized to JSON for the Streamlit dashboard.

**Why it matters:** EDA findings directly drove decisions in preprocessing (outlier flagging strategy), feature engineering (time decomposition, amount transformations), and model evaluation (AUPRC over accuracy).

---

### Step 2 — Data Preprocessing

**What:** Built a stateful `DataPreprocessor` class (sklearn-compatible: `fit_transform` / `transform`) that applies the same learned statistics to train, val, and test sets without leakage.

**Steps implemented:**

- **Duplicate removal** — configurable strategy: `drop | keep_first | keep_last`
- **Dtype fixing** — auto-coerces object columns that look numeric; safely downcasts `int64` to `int32`
- **Missing value handling** — supports `drop`, `mean`, `median`, `mode` (SimpleImputer), and `knn` (KNNImputer) strategies, all configured via YAML
- **Outlier treatment** — supports `flag`, `clip`, `remove`, and `none` strategies with either z-score or IQR detection. The chosen strategy was **flagging**: the top 5 outlier columns (by count, learned from train only) get binary `_is_outlier` indicator columns added. This preserves the data while giving the model a signal about anomalous values.

**Why this design:** Fitting statistics only on training data and applying them to val/test is the correct way to prevent data leakage. The preprocessor is serialized to `artifacts/preprocessing/preprocessor.pkl` and loaded at API inference time — so the exact same transformations happen in production.

---

### Step 3 — Feature Engineering

**What:** Built a `FeatureEngineer` class (inherits `BaseEstimator, TransformerMixin`) that creates domain-meaningful features and fits scalers on training data only.

**Features created:**

From `Time` (seconds elapsed since first transaction):
- `hour_of_day`, `day_of_week`, `is_night`, `is_weekend`, `is_business_hr`
- `hour_sin`, `hour_cos` — cyclical sine/cosine encoding so that hour 23 and hour 0 are treated as adjacent (a circle), which raw integers cannot express
- `Time` is then dropped since all information is extracted

From `Amount`:
- `log_amount` — compresses the heavy right-skew of transaction amounts
- `amount_zscore` — how far this transaction deviates from the mean
- `amount_sqrt` — another variance-stabilizing transform
- `amount_bin` — quantile-based discretization into 5 bins (bin edges learned from train only)
- `is_high_value` — binary flag for transactions above the 95th percentile

**Scaling:** `Amount`, `log_amount`, and `amount_zscore` are scaled using `StandardScaler` fitted on training data and applied consistently to val and test. All scaler state is saved in the `FeatureEngineer` artifact.

**Why separate amounts needed multiple transforms:** XGBoost handles raw amounts fine, but log/zscore forms help with feature importance interpretability and give the model more ways to find patterns in value distributions.

---

### Step 4 — Class Balancing with SMOTE (Inside Cross-Validation)

**What:** Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the 577:1 class ratio — but only inside each cross-validation fold's training portion, never before the split.

**Why feature selection was skipped:** A `FeatureSelector` class (variance threshold, correlation filter, RFE) was built and is available in `src/features/feature_selection.py` for future use with linear models. For this project, it was intentionally not applied — tree-based models like XGBoost handle feature selection internally through split gain and feature importance. Adding an external selection step on top would discard features that XGBoost's own splitting logic could have used meaningfully, and adds unnecessary complexity without benefit for this model type.

**SMOTE placement — why it matters:** SMOTE was applied strictly inside each cross-validation fold, only on that fold's training portion. This is the correct approach. A common mistake is applying SMOTE before the train/val/test split, which causes data leakage — synthetic fraud points are generated from the full dataset, so information from the validation and test sets bleeds into training, producing inflated CV metrics that don't reflect real performance.

**What SMOTE does:** For each real fraud sample, it picks k nearest neighbors (also fraud) and generates synthetic samples along the line segments between them. This creates a more balanced training distribution without simply duplicating existing fraud rows, giving the model more decision boundary information around the fraud class.

**Result:** The training fold after SMOTE had a balanced class distribution. Validation and test sets were never touched by SMOTE — they remained at the real 577:1 ratio, so evaluation metrics reflect true production conditions.

---

### Step 5 — Model Training with Optuna

**What:** Trained an XGBoost classifier with Bayesian hyperparameter optimization using **Optuna** (50 trials, TPE sampler with Median pruner).

**Optimization target:** Area Under the Precision-Recall Curve (AUPRC), evaluated via 5-fold StratifiedKFold cross-validation inside each Optuna trial. AUPRC was chosen over ROC-AUC because it is more informative on heavily imbalanced data — it penalizes false positives among the rare positive class more severely.

**Pruning:** Optuna's `MedianPruner` stopped unpromising trials early (after 5 startup trials, with 2 warmup steps), saving compute time.

**Best hyperparameters found:**

| Parameter | Value |
|---|---|
| `n_estimators` | 450 |
| `max_depth` | 10 |
| `learning_rate` | 0.0650 |
| `subsample` | 0.799 |
| `colsample_bytree` | 0.578 |
| `gamma` | 0.290 |
| `reg_alpha` | 2.142 |
| `reg_lambda` | 0.101 |
| `min_child_weight` | 4 |
| `scale_pos_weight` | 577 (inverse class ratio) |

`scale_pos_weight=577` tells XGBoost to weight each fraud sample as heavily as 577 legitimate samples — this directly addresses the imbalance at the loss function level, complementing SMOTE.

**Best Optuna CV AUPRC:** 0.8482

---

### Step 6 — Evaluation & Threshold Tuning

**What:** After training, the default 0.5 decision threshold is almost never optimal on imbalanced data. A threshold sweep was performed over 200 values from 0.01 to 0.99, evaluated using **F-beta score with beta=0.5** (which weights precision twice as heavily as recall — because in fraud detection, a false positive means a legitimate customer is declined, which has a real cost).

**Best threshold found:** `0.6896`

**Final test set results:**

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.9741** |
| **AUPRC** | **0.8417** |
| **Precision (fraud class)** | **0.9655** |
| **Recall (fraud class)** | **0.7568** |
| **F1 (fraud class)** | **0.8485** |
| **Accuracy** | **99.95%** |

**Confusion matrix on 42,722 test transactions:**

|  | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| **Actual Legitimate** | 42,646 | 2 |
| **Actual Fraud** | 18 | 56 |

Only **2 false positives** (legitimate transactions wrongly flagged) and **18 false negatives** (missed frauds) on the held-out test set. The model caught **75.68% of all fraud cases** with **96.55% precision** — meaning that when it flags fraud, it is right 96.55% of the time.

**Why AUPRC over accuracy:** With 42,648 legitimate and 74 fraud samples in the test set, a model predicting "all legitimate" would score 99.83% accuracy. AUPRC = 0.8417 proves the model genuinely discriminates fraud, not just class frequency.

**Cross-validation AUPRC:** mean = 0.8531 ± (std from 5 folds) — confirming the test score is not a lucky split.

---

### Step 7 — Data Drift Detection

**What:** Implemented Population Stability Index (PSI) as a drift detector to compare the feature distributions of train vs. validation and train vs. test.

**Why PSI:** PSI measures how much a feature's distribution has shifted between two datasets. PSI < 0.1 = no change, PSI 0.1–0.2 = slight change, PSI > 0.2 = significant drift that may degrade model performance.

**Implementation:** The `psi_report()` helper in `src/utils/helpers.py` bins each feature into 10 buckets based on training quantiles and computes PSI against val and test. Features exceeding the 0.20 warning threshold are logged and tracked separately (high_drift_val vs high_drift_test) and logged to MLflow as params for each run.

**Why this matters for production:** PSI checks at training time reveal whether the test/val distributions meaningfully differ from train. The same check can be run in production to trigger model retraining alerts when real incoming data starts drifting.

---

### Step 8 — MLflow Experiment Tracking & Model Registry

**What:** Every training run is tracked with MLflow, logging:
- All Optuna best hyperparameters as params
- All test metrics (ROC-AUC, AUPRC, precision, recall, F1, CV stats)
- Decision threshold, drift-flagged features, model version tag
- Artifacts: config snapshot, feature importance CSV, drift JSON files
- The model itself, registered in the MLflow Model Registry as `FraudXGB v1`

**Model versioning:** `model_registry.py` auto-generates a `vMAJOR.MINOR.PATCH` version tag based on git history, saves both a canonical `xgboost_fraud_model.pkl` and a timestamped versioned snapshot, and writes a `.meta.json` sidecar with all metadata.

**Why this matters:** Reproducibility. Any future engineer can look at the MLflow UI and know exactly which hyperparameters, data version, and threshold produced the deployed model.

---

### Step 9 — FastAPI Serving Layer

**What:** Built a production-quality REST API in FastAPI (`app/`) with:

- **`POST /predict`** — scores a single transaction, returns prediction, fraud probability, threshold used, and model version
- **`POST /predict/batch`** — scores up to 1,000 transactions in one call, returns all predictions with a fraud count summary
- **`GET /health`** — liveness check that verifies the model is loaded; used by Kubernetes probes
- **`GET /info`** — returns model version, threshold, feature count, and all test metrics
- **`GET /metrics`** — Prometheus scrape endpoint exposing `fraud_predictions_total`, `fraud_prediction_latency_seconds`, and `fraud_batch_size` histograms

**Inference pipeline:** Each request goes through the same `DataPreprocessor.transform()` → `FeatureEngineer.transform()` → feature alignment → `model.predict_proba()` → threshold apply steps that training did. The preprocessor and feature_engineer artifacts are loaded once at startup via `@lru_cache` and kept in memory.

**Input validation:** Pydantic schemas enforce all 30 input fields (V1–V28, Amount, Time), with `Amount >= 0` constraint. Invalid inputs return `422 Unprocessable Entity` automatically.

**Middleware:** An HTTP middleware logs every request with method, path, status code, and latency in milliseconds. CORS is enabled with a note to restrict origins in production.

---

### Step 10 — Streamlit Dashboard

**What:** A `streamlit/app.py` dashboard that connects to the FastAPI backend and provides:
- Live model status sidebar (version, threshold, feature count, all 4 metrics)
- **Single transaction scoring** — form with all 30 input fields, pre-filled with example values, submits to `/predict` and shows fraud probability with color-coded result
- **Batch scoring** — CSV upload mode, submits to `/predict/batch`, shows results table with fraud flags
- **API health check** view

The Streamlit app and FastAPI API communicate over the internal Docker network (`http://api:8000`) when running via docker-compose.

---

### Step 11 — Containerization with Docker

**What:** A **multi-stage Dockerfile** separates build and runtime concerns:

```
Stage 1 (builder): installs all Python dependencies into /install
Stage 2 (runtime): copies only /install + application code
                   runs as non-root user (uid 1000)
                   exposes port 8000
                   built-in HEALTHCHECK calling /health
                   uvicorn with 4 workers
```

The multi-stage build keeps the final image small by not including the build toolchain.

**docker-compose.yml** orchestrates four services:
- `fraud-api` — the FastAPI container
- `fraud-dashboard` — the Streamlit dashboard, waits for API health check
- `fraud-prometheus` — scrapes `/metrics` every 15 seconds (started with `--profile monitoring`)
- `fraud-grafana` — pre-provisioned with Prometheus datasource and custom dashboard JSON

---

### Step 12 — Kubernetes Deployment (Minikube / AWS-ready)

**What:** Full Kubernetes manifests in `k8s/minikube/` were tested on Minikube and are structured identically to what would run on AWS EKS (only the service type changes from `NodePort` to `LoadBalancer` for AWS).

**Deployment (`deployment.yaml`):**
- `rollingUpdate` strategy with `maxUnavailable: 0` ensures zero-downtime updates
- **Startup probe** — waits up to 60s for the container to initialize before starting liveness checks
- **Liveness probe** — restarts the container if `/health` fails 3 times
- **Readiness probe** — removes the pod from service if not ready, preventing traffic to unhealthy instances
- **Security context** — runs as non-root user (uid 1000), `runAsNonRoot: true`
- Resource requests (512Mi / 250m CPU) and limits (1Gi / 1000m CPU) set for scheduler stability

**Horizontal Pod Autoscaler (`hpa.yaml`):**
- Scales between 1 and 3 replicas
- Triggers on CPU utilization > 70% or memory > 80%
- Scale-up stabilization: 60s (prevents thrashing); Scale-down stabilization: 300s (conservative)

**Why Minikube instead of AWS:** Minikube runs the same Kubernetes API and accepts the same YAML manifests as EKS. Everything validated locally — deployment, probes, HPA, service, rolling updates — maps directly to AWS by changing `imagePullPolicy: Never` to pull from Docker Hub and the service type to `LoadBalancer`. This was a deliberate cost-free way to prove Kubernetes proficiency.

---

### Step 13 — Monitoring with Prometheus & Grafana

**What:** The FastAPI app exposes a `/metrics` endpoint using `prometheus_client`. Three custom metrics are tracked:

- **`fraud_predictions_total`** (Counter, labeled `fraud` / `legitimate`) — total prediction volume by outcome
- **`fraud_prediction_latency_seconds`** (Histogram with buckets 5ms → 1s) — API response time distribution
- **`fraud_batch_size`** (Histogram) — distribution of batch sizes

**Prometheus** is configured to scrape the API every 15 seconds. Kubernetes pods are annotated with `prometheus.io/scrape: "true"` so a cluster-level Prometheus picks them up automatically.

**Grafana** is provisioned with a pre-built `fraud_detection.json` dashboard that visualizes these metrics in real time, with the Prometheus datasource auto-configured.

---

### Step 14 — CI/CD with GitHub Actions

**What:** `.github/workflows/deploy.yml` defines a two-job pipeline triggered on pushes to `main`:

**Job 1 — Test:**
- Sets up Python 3.11
- Installs all dependencies + pytest
- Runs the full test suite (`tests/`) covering API endpoints, data pipeline, feature engineering, and model inference

**Job 2 — Build & Push (only if tests pass, only on main):**
- Logs into Docker Hub using repository secrets (`DOCKER_USERNAME`, `DOCKER_TOKEN`)
- Builds the multi-stage Docker image
- Pushes two tags: `latest` and the exact git commit SHA (for traceability)

**AWS variant** (`workflows/AWS/deploy.yml`) replaces the Docker Hub push step with an ECR push + `kubectl rollout` trigger against an EKS cluster — ready to activate once an AWS account is available.

**Why commit SHA tags:** Tagging images with the git SHA means every deployed image is traceable to the exact code that produced it. Combined with MLflow tracking, this creates a full audit trail from commit → image → model → metrics.

---

## Model Results

```
Dataset: 284,807 credit card transactions (2-day European cardholder data)
Train/Val/Test split: 70% / 15% / 15% (stratified)
After SMOTE: training set balanced for model fitting

Test Set: 42,722 transactions (74 fraud, 42,648 legitimate)
```

| Metric | Score |
|---|---|
| ROC-AUC | **0.9741** |
| AUPRC | **0.8417** |
| Precision (fraud) | **96.55%** |
| Recall (fraud) | **75.68%** |
| F1 (fraud) | **84.85%** |
| Accuracy | **99.95%** |
| False Positives | **2** |
| False Negatives | **18** |
| Decision Threshold | **0.6896** (tuned, not default 0.5) |
| CV AUPRC (5-fold) | **0.8531** |
| Optuna Best CV AUPRC | **0.8482** (across 50 trials) |

---

## Tech Stack

| Category | Tools |
|---|---|
| **ML / Data** | Python 3.11, XGBoost, scikit-learn, imbalanced-learn (SMOTE), Optuna, pandas, NumPy |
| **Feature Engineering** | Custom sklearn-compatible transformers, RobustScaler, PowerTransformer |
| **Experiment Tracking** | MLflow (tracking + model registry) |
| **API** | FastAPI, Uvicorn (4 workers), Pydantic v2 |
| **Monitoring** | Prometheus client, Grafana |
| **Dashboard** | Streamlit |
| **Containerization** | Docker (multi-stage), docker-compose |
| **Orchestration** | Kubernetes (Minikube tested; AWS EKS-ready) |
| **CI/CD** | GitHub Actions → Docker Hub |
| **Configuration** | YAML config files (data, model, training) |
| **Testing** | pytest, httpx, unittest.mock |

---

## Quick Start

### Run with Docker Compose

```bash
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system

# Start API + Streamlit dashboard
docker-compose up --build

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up --build
```

| Service | URL |
|---|---|
| API (Swagger UI) | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

### Run on Minikube

```bash
# Start cluster
minikube start
minikube addons enable metrics-server

# Build image into Minikube's Docker daemon
eval $(minikube docker-env)
docker build -t fraud-detection:1.0.0 .

# Deploy
kubectl create namespace fraud-detection
kubectl apply -f k8s/minikube/

# Access the API
minikube service fraud-detection -n fraud-detection
```

### Re-train the model

```bash
pip install -r requirements.txt

# Run full data pipeline (ingestion → preprocessing → feature engineering → splits)
python scripts/run_pipeline.py

# Train model only (with Optuna HPO, assumes data pipeline already ran)
python scripts/train.py

# Run everything end-to-end with a smoke test inference check at the end
python scripts/run_pipeline.py --smoke-test

# Run offline batch inference on a CSV from the command line
python scripts/predict.py
```

### Explore the notebooks

```bash
pip install jupyter
jupyter notebook notebooks/

# 01_eda.ipynb           — data exploration, imbalance analysis, EDA figures
# 02_feature_engineering.ipynb — feature prototyping before modularization
```

---

## API Reference

### `POST /predict`

Score a single transaction.

```json
{
  "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
  "V4": 1.3782, "V5": -0.3383, "V6": 0.4624,
  "V7": 0.2396, "V8": 0.0987, "V9": 0.3638,
  "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
  "V13": -0.9914, "V14": -0.3112, "V15": 1.4682,
  "V16": -0.4704, "V17": 0.2080, "V18": 0.0258,
  "V19": 0.4040, "V20": 0.2514, "V21": -0.0183,
  "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
  "V25": 0.1285, "V26": -0.1891, "V27": 0.1336,
  "V28": -0.0211, "Amount": 149.62, "Time": 0.0
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.002341,
  "threshold": 0.6896,
  "model_version": "v1.0.0",
  "is_fraud": false
}
```

### `POST /predict/batch`

Score up to 1,000 transactions. Body: `{ "transactions": [ ... ] }`.

### `GET /health`

Returns model load status, version, and active threshold.

### `GET /info`

Returns model version, all test metrics, feature count and names.

### `GET /metrics`

Prometheus-format metrics for scraping.

---

## Why This Project Matters

This project was built to demonstrate real production ML engineering skills, not just model accuracy:

**End-to-end ownership** — the same engineer built the data pipeline, model, API, container, Kubernetes manifests, monitoring, and CI/CD. No "hand-off" gaps.

**Correct ML practices** — no data leakage (all preprocessing fitted on train only), no SMOTE applied before splitting, threshold tuned on validation not test set, AUPRC used instead of accuracy.

**Production-safe design** — the inference pipeline loads the same serialized preprocessor and feature engineer that training used, so the model sees identical transformations in production. This is how real systems work.

**Observability** — the system exposes Prometheus metrics, structured logs with latency, and health probes that Kubernetes can act on. A model nobody can observe is a model nobody can trust.

**Kubernetes proficiency without AWS** — Minikube is not a workaround; it is the standard local Kubernetes environment. The same YAML, the same probes, the same HPA logic runs on EKS. The only difference is infrastructure provisioning, not engineering.

**Reproducibility** — every training run is logged in MLflow with hyperparameters, metrics, artifacts, git commit SHA, and a versioned model artifact. Any result can be reproduced.
