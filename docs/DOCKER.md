# Docker – Step 14 (Dockerization)

This document explains how to build and run the **serving** and **training** images, and the contracts they follow. No single "all-in-one" container: serving and training are separate images and responsibilities.

---

## Why two images?

- **Serving image**: Runs the FastAPI app only. It loads a model from a path (or later from MLflow) and serves predictions. It is **stateless** and can be scaled by adding more containers. No training libraries (e.g. SHAP, tuning) are required at runtime.
- **Training image**: Runs the retraining pipeline as a **batch job** (e.g. `python scripts/retrain_model.py`). It uses data and writes models/artifacts, then exits. It is not a long-running service.

This separation matches production MLOps: train in one place, serve in another; same codebase, different runtime contracts.

---

## Prerequisites

- Docker (and optionally Docker Compose)
- For **serving**: the training pipeline creates a fixed alias `models/production.pkl` (symlink to the best model). Run training first so that alias exists, or ensure `production.pkl` and `production_metadata.json` point to a valid model.

---

## Serving image

### Build

```bash
docker build -f Dockerfile.serve -t credit-risk-api:latest .
```

### Run (model from host directory)

Serving loads a **fixed alias** (`production.pkl`), not a model-specific file. The training pipeline sets that alias (symlink) to the best model (e.g. XGBoost when it wins). No code or config change is needed when the champion changes.

Mount your `models/` directory and set `MODEL_PATH` to the alias path inside the container:

```bash
docker run -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  -e MODEL_PATH=/app/models/production.pkl \
  credit-risk-api:latest
```

- **MODEL_PATH**: Path **inside the container** to the champion alias (default: `production.pkl`). The training pipeline creates this symlink to the best model; serving does not know or care which algorithm it is.
- **MODEL_METADATA_PATH** (optional): If not set, the app derives metadata from `MODEL_PATH` (e.g. `production.pkl` → `production_metadata.json`; training also symlinks that to the best model’s metadata).

Then open: http://localhost:8000/docs and http://localhost:8000/metrics (Prometheus).

---

## Training image

### Build

```bash
docker build -f Dockerfile.train -t credit-risk-train:latest .
```

### Run (batch job)

Mount data, models, and artifacts so the pipeline can read raw/processed data and write models and registry:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  credit-risk-train:latest
```

The default command is `python scripts/retrain_model.py`. It will:

1. Read from `data/` (raw and processed)
2. Run clean → feature engineering → train → tune → evaluate
3. Write models to `models/` and registry metadata to `artifacts/registry/`
4. Select the best model (e.g. by ROC-AUC) and set the champion alias: `production.pkl` and `production_metadata.json` are symlinks to that model so the API serves it without any config change.

To run a single step (e.g. only cleaning):

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  credit-risk-train:latest python scripts/clean_data.py
```

---

## Docker Compose (optional)

Use Compose to run the API and, on demand, the training job.

### Start the API only

```bash
docker compose up api
```

Requires that `./models` contains the champion alias: `production.pkl` (and `production_metadata.json`), which the training pipeline creates as symlinks to the best model. Run training at least once, then start the API; it loads `MODEL_PATH=/app/models/production.pkl` and thus serves whichever model won the last run.

### Run training once (batch job)

```bash
docker compose run train
```

This runs the training image once with `data`, `models`, and `artifacts` mounted. After it finishes, you can start or restart the API so it uses the new model.

---

## Image contracts (summary)

| Image   | Responsibility        | Model / data in image? | How to provide model / data        |
|--------|------------------------|-------------------------|------------------------------------|
| **Serve** | FastAPI + /metrics     | No                      | Mount dir + `MODEL_PATH` (and optional `MODEL_METADATA_PATH`) |
| **Train** | Retrain + evaluate + registry | No                | Mount `data/`, `models/`, `artifacts/` at run time |

Both images are **reproducible**: same Dockerfile + same `requirements.txt` (and `pyproject.toml`) yield the same environment everywhere (local, CI, production).

---

## Troubleshooting

- **API fails at startup with "Model file not found"**  
  Ensure the volume mount and `MODEL_PATH` point to the champion alias inside the container (e.g. `-v $(pwd)/models:/app/models` and `MODEL_PATH=/app/models/production.pkl`). Run the training pipeline first so it creates `production.pkl` (and `production_metadata.json`) as symlinks to the best model.

- **Training container exits with "No such file or directory" for data**  
  Ensure `data/raw` (and any paths used by the scripts) exist on the host and are mounted into the container (e.g. `-v $(pwd)/data:/app/data`).

- **Import errors when running training**  
  The training image sets `PYTHONPATH=/app:/app/src` so both `scripts.*` and `registry`/`config`/etc. resolve. If you override `CMD`, keep the same working directory and env.
