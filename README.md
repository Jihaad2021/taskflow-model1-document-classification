# TaskFlow Model 1 — Document Classification (AG News)

TaskFlow Model 1 is a production-grade document classification system built with a modular ML pipeline architecture. The project covers the full lifecycle from data ingestion to a Dockerized ONNX inference API.

This repository is designed as a portfolio-quality ML engineering project with an emphasis on clean architecture, reproducibility, test coverage, and deployment readiness.

---

## High-Level Architecture

### Pipeline layers

1. **Data Pipeline**

   * Ingestion (HuggingFace datasets)
   * Validation (structural + semantic checks)
   * Preprocessing (Transformer-friendly)
   * Feature engineering (tokenization)

2. **Model & Training**

   * HuggingFace classification model builder
   * Training pipeline
   * Optional Optuna hyperparameter tuning

3. **Inference & Deployment**

   * ONNX export
   * Graph optimization
   * INT8 quantization
   * FastAPI inference service
   * Docker containerization

---

## Repository Structure

```
.
├── src/                 # Core ML library
│   ├── data/            # Ingestion, validation, preprocessing
│   ├── models/          # Model builder + ONNX utilities
│   ├── training/        # Trainer + tuning logic
│   ├── evaluation/      # Metrics
│   └── pipeline/        # Orchestrated pipelines
│
├── api/                 # FastAPI inference service
│   ├── app.py
│   └── schemas.py
│
├── docker/              # Docker build configuration
│   ├── Dockerfile
│   └── .dockerignore
│
├── scripts/             # Benchmarking & tooling
├── configs/             # YAML configs (Pydantic v2)
├── tests/               # Unit + integration tests
└── README.md
```

---

## Environment Setup (Local)

### Conda environment

Recommended environment name:

* `taskflow-m1`

Activate:

```bash
conda activate taskflow-m1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all tests:

```bash
pytest
```

---

## Data Pipeline (Phase 1)

The data pipeline is fully config-driven and tested.

Example usage:

```python
from src.config import load_app_config
from src.data.pipeline import build_tokenized_datasets

app_cfg = load_app_config("configs/local.yaml")
tokenized_datasets = build_tokenized_datasets(app_cfg)
```

Output:

* HuggingFace `DatasetDict`
* Fields:

  * `input_ids`
  * `attention_mask`
  * `labels`

---

## Training & Hyperparameter Tuning

### Training only

```bash
python -m src.pipeline.training_pipeline --train-only
```

### Hyperparameter tuning (Optuna)

Supported modes:

* `--tune-only`
* `--tune-and-train`
* `--n-trials`

Example:

```bash
python -m src.pipeline.training_pipeline --tune-only --n-trials 10
```

---

## ONNX Inference Artifacts

This project generates three ONNX variants.

| Variant          | Description                      | Use case                |
| ---------------- | -------------------------------- | ----------------------- |
| `baseline.onnx`  | Direct export from HF model      | Reference               |
| `optimized.onnx` | Graph-optimized via ONNX Runtime | Default deployment      |
| `int8.onnx`      | Dynamic INT8 quantization        | CPU-optimized inference |

Benchmark artifacts are stored locally (not committed):

```
artifacts/onnx_bench/
```

---

## Inference API (FastAPI)

### Run locally (no Docker)

Environment variables:

```bash
export MODEL1_ONNX_PATH="artifacts/onnx_bench/optimized.onnx"
export MODEL1_TOKENIZER_NAME_OR_PATH="hf-internal-testing/tiny-random-distilbert"
export MODEL1_MAX_LENGTH="16"
```

Run API:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts":["breaking news market rallies","sports update local team wins"]}'
```

---

## Inference API (Docker)

### Build image

```bash
docker build -t taskflow-model1-inference -f docker/Dockerfile .
```

### Run container

Mount ONNX artifacts at runtime:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts/onnx_bench:/models" \
  -e MODEL1_ONNX_PATH=/models/optimized.onnx \
  taskflow-model1-inference
```

Switch to INT8 variant:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts/onnx_bench:/models" \
  -e MODEL1_ONNX_PATH=/models/int8.onnx \
  taskflow-model1-inference
```

---

## Design Principles

* Separation of concerns

  * `src/` = ML core library
  * `api/` = serving layer
* Config-driven behavior
* Test-first development
* Production-oriented inference
* CPU-first deployment strategy

---

## Notes

* Runtime artifacts are excluded from version control:

  * `artifacts/`
  * `data/hf_cache/`
* Docker image does not bundle model files (mounted at runtime)
* Default deployment uses `optimized.onnx`

---

## Status

* Data pipeline: complete
* Training & tuning pipeline: complete
* ONNX export & optimization: complete
* FastAPI inference service: complete
* Dockerized inference: complete

Model 1 is production-ready.
