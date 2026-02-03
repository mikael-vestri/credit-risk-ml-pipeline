# Implementation Plan: Steps 14–17 (Docker, MLflow, Rollback, Airflow)

This document outlines how we will extend the credit-risk-ml-pipeline from Step 13 to a production-like MLOps setup **without refactoring existing logic**. We add Docker, MLflow, rollback/governance, and Airflow on top of what you already have.

---

## Current State (What We Keep)

- **Training**: `scripts/train_models.py`, `tune_models.py`, `evaluate_models.py`, `retrain_model.py`
- **Registry**: `src/registry/artifact_registry.py` (local JSON); promotion via `scripts/promote_model.py`
- **Serving**: FastAPI in `src/api/app.py`; model loaded from `models/` directory
- **Flow**: Train → Evaluate → Register (staging) → **Manual** promote to production → API loads “production” model

We keep this flow. We will:
- **Wrap** it in Docker and MLflow.
- **Replace** only the registry implementation with MLflow (same concepts: stages, promotion).
- **Add** explicit rollback and Airflow orchestration.

---

## Step 14 – Dockerization

### Goal
Two separate images: one for **serving** (API only), one for **training** (batch job). No all-in-one container.

### Concepts (learning)

- **Serving container**: Runs one process (FastAPI). Loads model from a **mount or from a registry URL**. Stateless; scale by adding more containers.
- **Training container**: Runs a **batch job** (e.g. `python scripts/retrain_model.py`). Exits when done. No long-running server.
- **Reproducibility**: Same `Dockerfile` + same base image + same dependency file ⇒ same environment everywhere (laptop, CI, production).

### Deliverables

| Item | Description |
|------|-------------|
| **Dockerfile.serve** | Multi-stage build: install deps from `requirements.txt` (or pyproject), copy `src/`, run FastAPI. No `scripts/retrain_*`, no heavy train deps. |
| **Dockerfile.train** | Install full deps (train + evaluate + current registry). Copy `src/`, `scripts/`. Default CMD: run retrain (or override for single steps). |
| **.dockerignore** | Ignore `data/`, `models/`, `artifacts/`, `.git`, `__pycache__`, docs, tests (unless we add test stage later). |
| **Serve image contract** | Expect model to be available at a path (e.g. env `MODEL_PATH`) or document that you mount `models/`; no training inside. |
| **Train image contract** | Expect `data/` (or raw data path) and registry output path (or MLflow later); no serving. |
| **Optional: docker-compose.yml** | One service for API (image serve, port 8000), one “job” service (image train, run once). Helps run locally; not required for CI/production. |

### Out of scope for Step 14
- MLflow (Step 15); registry stays file-based for now.
- Kubernetes/cloud; focus on images and contracts only.

### Order of work
1. Add `.dockerignore`.
2. Add `Dockerfile.serve` and document `MODEL_PATH` (or mount).
3. Add `Dockerfile.train` and document inputs/outputs.
4. Optionally add `docker-compose.yml` and a short `docs/DOCKER.md` (how to build, run, and use the two images).

---

## Step 15 – MLflow Integration

### Goal
Replace the **local artifact registry** with MLflow Tracking + Model Registry. Same concepts: “log run”, “register model”, “promote to production”.

### Concepts (learning)

- **MLflow Tracking**: Logs **runs** (each train/eval = one run) with params, metrics, and artifacts (e.g. model pickle, plots). Stored in a **backend** (local folder, SQLite, or server).
- **MLflow Model Registry**: Stores **registered models** (name + versions). Each version can be in a **stage**: `Staging`, `Production`, `Archived`. “Promotion” = transition version to `Production`; optional “demotion” of previous production to `Archived`.
- **No auto-promotion**: We keep promotion as an explicit action (script or API call), not triggered by a metric threshold.

### Deliverables

| Item | Description |
|------|-------------|
| **Dependency** | Add `mlflow` to `requirements.txt` (or a `requirements-mlflow.txt`). |
| **MLflow backend** | Default: local directory (e.g. `./mlruns`) so it works without a server. Document how to point to a remote tracking server later. |
| **Training integration** | In `retrain_model.py` (and/or `train_models.py` / `tune_models.py`): start an MLflow run; log hyperparameters, metrics (ROC-AUC, etc.), and the model artifact; register the model under a name (e.g. `credit-risk-model`). Do **not** put the run in `Production` by default (keep it unstated or `Staging`). |
| **Promotion script** | Adapt `scripts/promote_model.py`: call MLflow Model Registry API to transition a given **model version** to `Production` (and optionally transition current production to `Archived`). Inputs: model name, version or run id. |
| **Serving integration** | Adapt API (or a small loader in `serving.py`) to load the **production** model from MLflow (by model name + stage `Production`). If MLflow is remote, use MLflow client; if local, can resolve to a path. Keep existing “load from path” as fallback (e.g. env `MODEL_PATH`). |
| **Backward compatibility** | Keep `src/registry/artifact_registry.py` in the codebase but **don’t use** it in the main flow once MLflow is wired; or call it only when MLflow is disabled (feature flag). No big delete yet. |

### Order of work
1. Add MLflow dependency and minimal “hello MLflow” script or test (log a run, register a model) to confirm setup.
2. Wire training/retrain to log runs and register models; keep promotion manual.
3. Wire `promote_model.py` to MLflow stage transition.
4. Wire serving to load from MLflow Production; keep path fallback.
5. Document: where runs are stored, how to promote, how to point to a remote server.

---

## Step 16 – Rollback & Governance

### Goal
Explicit **rollback**: revert production to a **previous** model version. Auditable and manual.

### Concepts (learning)

- **Rollback** = “Set Production to a previously deployed version.” So: transition that version to `Production` and (optionally) the current one to `Archived`.
- **Governance** = Who can promote/rollback, and how it’s recorded. For this project we keep it simple: one script (or Airflow task) that performs the transition; logging and approvals can be added later.

### Deliverables

| Item | Description |
|------|-------------|
| **Rollback script** | e.g. `scripts/rollback_model.py`: takes model name and **target version** (the one to restore to). Calls MLflow to transition that version to `Production` and current production to `Archived`. Same idea as promote, but semantics are “restore previous”. |
| **Documentation** | In `OPERATIONS_RUNBOOK.md`: when to rollback, how to run `rollback_model.py`, how to verify after rollback. |
| **Optional** | List “production history” (which versions were ever in Production) from MLflow; document in runbook. |

### Order of work
1. Implement `rollback_model.py` using MLflow Model Registry.
2. Update runbook and, if present, `PRODUCTION_CHECKLIST.md`.

---

## Step 17 – Airflow Orchestration

### Goal
A **scheduled DAG** that runs retraining, logs to MLflow, and **stops before promotion**. Promotion stays manual.

### Concepts (learning)

- **DAG** = Directed Acyclic Graph of **tasks**. Example: task “retrain” → task “evaluate” → task “register_model”; no “promote” in the DAG.
- **Airflow** runs tasks on a schedule (e.g. weekly) or trigger. It runs your **existing** scripts (e.g. in the training Docker image or a venv).
- **Manual promotion**: After the DAG runs, a human (or a separate “approval” task) runs the promote script when satisfied.

### Deliverables

| Item | Description |
|------|-------------|
| **DAG file** | One DAG, e.g. `dags/credit_risk_retrain.py`: tasks for “run retrain script”, “optional: run evaluation report”. Uses DockerOperator or BashOperator to run the training image or `python scripts/retrain_model.py`. |
| **Schedule** | e.g. weekly; configurable via DAG args. |
| **No auto-promote** | DAG does **not** call promote; it only trains and registers. Documentation states: “After DAG succeeds, promote manually if desired.” |
| **Airflow setup** | Document how to run Airflow (e.g. Docker Compose with `airflow` image, or local install). Minimal: scheduler + webserver; optional worker if using Celery. |
| **Integration** | Retrain script already logs to MLflow (Step 15); DAG just invokes it. |

### Order of work
1. Add a minimal Airflow setup (e.g. `docker-compose.airflow.yml` or docs to run Airflow locally).
2. Add the DAG that runs retrain (and optionally evaluate); document schedule and “no promote”.
3. Update runbook: “Weekly retrain runs via Airflow; promote via `promote_model.py` when ready.”

---

## Dependency Order

```
Step 14 (Docker)     →  Optional: use these images in Step 17 (Airflow).
Step 15 (MLflow)     →  Step 16 (rollback) and Step 17 (DAG) depend on MLflow.
Step 16 (Rollback)   →  Depends on Step 15 (MLflow registry).
Step 17 (Airflow)    →  Depends on Step 15 (retrain logs to MLflow); can use Step 14 images.
```

Recommended implementation order: **14 → 15 → 16 → 17**.

---

## What We Do *Not* Do

- No refactor of core training/evaluation logic.
- No single “do everything” container.
- No automatic promotion by default.
- No removal of the existing artifact_registry code until MLflow is the single source of truth and we’re comfortable; then we can deprecate it.
- No Kubernetes/Helm in this plan (can be a later step).

---

## Next Step

Once you confirm this plan (or suggest changes), we proceed with **Step 14 – Dockerization**: add `.dockerignore`, `Dockerfile.serve`, `Dockerfile.train`, and optional `docker-compose.yml` + `docs/DOCKER.md`, without changing any application code.
