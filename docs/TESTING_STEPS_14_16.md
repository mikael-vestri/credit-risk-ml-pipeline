# Testing Steps 14–16 (Docker, MLflow, Rollback)

Use this guide to verify Docker, MLflow, champion symlink, promote, and rollback. Run in order when you have time; (1) and (2) already cover most of the new pipeline.

---

## 1. Path-based serving (no Docker, no MLflow)

**Goal:** API and champion symlink work as before.

```bash
pip install -r requirements.txt
python scripts/retrain_model.py   # needs data in data/raw; creates models/ + production.pkl symlink
python scripts/run_api.py
```

- **Check:** http://localhost:8000/docs → `/health` and `/predict`. API loads `production.pkl` (symlink to best model).

---

## 2. MLflow only (local)

**Goal:** Retrain logs to MLflow; promote and rollback in the registry.

```bash
python scripts/retrain_model.py
```

- **Check:** Folder `./mlruns` exists. Start UI: `mlflow ui` (from project root) → http://localhost:5000 → Experiments → `credit-risk-retrain` → last run (params, metrics, 3 model artifacts). Model Registry: one model (e.g. `credit-risk-model`) with versions in Staging.

Promote a version:

```bash
python scripts/promote_model.py --version <N> --archive-current
```

- **Check:** In MLflow UI, that version is Production; previous is Archived.

Rollback:

```bash
python scripts/rollback_model.py --version <previous_N>
```

- **Check:** In UI, Production is the version you rolled back to.

---

## 3. Serving from MLflow (optional)

**Goal:** API loads the Production model from MLflow instead of from disk.

With MLflow UI still using default `./mlruns`:

```bash
set MLFLOW_TRACKING_URI=./mlruns
set MLFLOW_MODEL_NAME=credit-risk-model
python scripts/run_api.py
```

(PowerShell: `$env:MLFLOW_TRACKING_URI="./mlruns"`; `$env:MLFLOW_MODEL_NAME="credit-risk-model"`.)

- **Check:** `/health` or `/stats` and a `/predict` call. Change Production in MLflow (promote/rollback), restart API, and confirm the served model changes.

---

## 4. Docker: training image

**Goal:** Retrain runs in the training container; outputs in mounted dirs.

From project root (with `data/` and, if needed, data in `data/raw`):

```bash
docker build -f Dockerfile.train -t credit-risk-train:latest .
docker run --rm -v "%cd%\data:/app/data" -v "%cd%\models:/app/models" -v "%cd%\artifacts:/app/artifacts" -v "%cd%\mlruns:/app/mlruns" credit-risk-train:latest
```

PowerShell:

```powershell
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/models:/app/models" -v "${PWD}/artifacts:/app/artifacts" -v "${PWD}/mlruns:/app/mlruns" credit-risk-train:latest
```

- **Check:** `models/` has new `.pkl` and `production.pkl` symlink; `mlruns/` has a new run; `artifacts/` updated if used.

---

## 5. Docker: serving image

**Goal:** API runs in a container and loads the champion from mounted `models/`.

```bash
docker build -f Dockerfile.serve -t credit-risk-api:latest .
docker run -p 8000:8000 -v "%cd%\models:/app/models:ro" -e MODEL_PATH=/app/models/production.pkl credit-risk-api:latest
```

PowerShell:

```powershell
docker run -p 8000:8000 -v "${PWD}/models:/app/models:ro" -e MODEL_PATH=/app/models/production.pkl credit-risk-api:latest
```

- **Check:** http://localhost:8000/docs, `/health`, `/predict`. Same as (1) but in Docker.

---

## 6. Docker Compose (optional)

**Goal:** Start API with one command; run training on demand.

```bash
docker compose up api
```

(Requires `./models/production.pkl` from a previous run; if missing, run training first.)

Run training once:

```bash
docker compose run train
```

- **Check:** API responds; after `run train`, new models and symlink in `./models`.

---

## 7. Full flow (retrain → promote → rollback)

1. Run retrain (local or Docker train image).
2. In MLflow UI, pick a version; run `python scripts/promote_model.py --version N --archive-current`.
3. If serving from MLflow, restart API and confirm it serves that version.
4. Run `python scripts/rollback_model.py --version M` (M = previous Production).
5. In UI, confirm Production is M; if serving from MLflow, restart API and confirm again.

---

## Quick reference

| What to test        | Command / action |
|---------------------|------------------|
| Path-based API      | `retrain_model.py` → `run_api.py` |
| MLflow tracking     | `retrain_model.py` → `mlflow ui` |
| Promote             | `promote_model.py --version N --archive-current` |
| Rollback            | `rollback_model.py --version M` |
| Docker train        | `Dockerfile.train` + `docker run` with mounts |
| Docker serve        | `Dockerfile.serve` + `docker run` with `MODEL_PATH` and volume |
| Compose             | `docker compose up api` / `docker compose run train` |

See also: [DOCKER.md](DOCKER.md), [MLFLOW.md](MLFLOW.md), [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md).
