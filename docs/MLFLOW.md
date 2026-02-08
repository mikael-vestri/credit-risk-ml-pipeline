# MLflow – Step 15 (Tracking & Model Registry)

This project uses **MLflow** for experiment tracking and model versioning. Training runs are logged to MLflow; promotion to production is a manual step via the promote script.

---

## Where runs are stored

- **Default (local):** `./mlruns` in the project root. No server required.
- **Remote:** Set `MLFLOW_TRACKING_URI` to your tracking server (e.g. `http://mlflow-server:5000`). Same variable is used by retrain, promote, and (when loading from MLflow) the API.

The `mlruns/` directory is in `.gitignore`; do not commit it. For shared tracking, use a remote server or shared storage.

---

## What gets logged (retrain)

When you run `python scripts/retrain_model.py`:

1. **One MLflow run** per retrain (experiment: `credit-risk-retrain`).
2. **Params:** e.g. mode, models list, tuning, cv_folds.
3. **Metrics:** per-model ROC-AUC, precision, recall, etc. (prefixed by model name).
4. **Artifacts:** each trained model (sklearn or xgboost flavor) logged and **registered** as a new version of the model name `credit-risk-model`. New versions start in **Staging**; promotion to Production is manual.

---

## Promoting a model to Production

After a retrain, you have new versions (e.g. v4, v5, v6). To put a specific version in Production:

```bash
# Promote version 5 to Production (and optionally archive current Production)
python scripts/promote_model.py --version 5 --archive-current

# Custom model name or tracking URI
python scripts/promote_model.py --version 5 --model-name credit-risk-model --tracking-uri http://mlflow:5000
```

- **`--version`** (required): Model version number from MLflow (e.g. `5`).
- **`--model-name`**: Registered model name (default: `credit-risk-model`).
- **`--archive-current`**: Move the current Production version to Archived before promoting.
- **`--tracking-uri`**: Override tracking server (otherwise uses `MLFLOW_TRACKING_URI` or `./mlruns`).

Promotion is **manual** by design: no auto-promote on metric threshold.

---

## Serving from MLflow vs path

- **Path-based (default):** API loads from `MODEL_PATH` (e.g. `production.pkl`). The retrain script updates the `production.pkl` symlink to the best model, so no MLflow is needed for serving.
- **MLflow-based:** If `MLFLOW_TRACKING_URI` (and optionally `MLFLOW_MODEL_NAME`) are set when the API starts, it loads the **Production** model from the MLflow Model Registry. Otherwise it falls back to `MODEL_PATH`.

Use path-based when running locally or with Docker and the champion symlink. Use MLflow when you want the API to always resolve “Production” from the registry (e.g. remote tracking server).

---

## Pointing to a remote tracking server

1. Start or use an MLflow tracking server (e.g. `mlflow server --host 0.0.0.0 --port 5000`).
2. Set the URI when running retrain, promote, or API:
   - **Env:** `export MLFLOW_TRACKING_URI=http://your-server:5000`
   - **Promote script:** `--tracking-uri http://your-server:5000`
3. Retrain and promote as above; runs and versions are stored on the server. The API can load from MLflow Production when started with the same `MLFLOW_TRACKING_URI`.
