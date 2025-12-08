# TaskFlow – Model 1: Document Classification

This repository contains **Model 1 – Document Classification** for the TaskFlow project.

- Base model: `distilbert-base-uncased`
- Task: multi-class document / news classification (AG News)
- Features:
  - End-to-end training pipeline (local & Kaggle modes)
  - Evaluation (accuracy, F1)
  - ONNX export for optimized inference
  - FastAPI inference service (port 8001)

## Local setup

```bash
python3 -m venv taskflow-m1
source taskflow-m1/bin/activate

pip install -r requirements.txt
