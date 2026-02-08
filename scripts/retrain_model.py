"""
Script to retrain models with new data (Step 13 + Step 15 MLflow).

Orchestrates the complete retraining pipeline:
1. Data ingestion and cleaning
2. Feature engineering
3. Model training and tuning
4. Model evaluation
5. MLflow: log run, register models (Staging), champion symlink for path-based serving.
Promotion to Production is manual (scripts/promote_model.py).
"""

import logging
import os
import pickle
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow import xgboost as mlflow_xgboost

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MLflow: one registered model name; versions per run (one per algorithm).
MLFLOW_MODEL_NAME = "credit-risk-model"


def main():
    """Run complete retraining pipeline with MLflow tracking and registry."""

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # MLflow: default local backend (./mlruns); override with MLFLOW_TRACKING_URI
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or str(project_root / "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit-risk-retrain")

    print("=" * 80)
    print("MODEL RETRAINING PIPELINE (MLflow)")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"MLflow tracking: {tracking_uri}")
    print("=" * 80)

    with mlflow.start_run(run_name=run_id) as run:
        mlflow.log_params({
            "mode": "full",
            "models": "logistic_regression,random_forest,xgboost",
            "tuning": True,
            "cv_folds": 5,
        })

        try:
            # Step 1: Data cleaning
            print("\n" + "=" * 80)
            print("STEP 1: DATA CLEANING")
            print("=" * 80)
            from scripts.clean_data import main as clean_main
            clean_main()

            # Step 2: Feature engineering
            print("\n" + "=" * 80)
            print("STEP 2: FEATURE ENGINEERING")
            print("=" * 80)
            from scripts.engineer_features import main as engineer_main
            engineer_main()

            # Step 3: Model training
            print("\n" + "=" * 80)
            print("STEP 3: MODEL TRAINING")
            print("=" * 80)
            from scripts.train_models import main as train_main
            train_main()

            # Step 4: Hyperparameter tuning
            print("\n" + "=" * 80)
            print("STEP 4: HYPERPARAMETER TUNING")
            print("=" * 80)
            from scripts.tune_models import main as tune_main
            tune_main()

            # Step 5: Model evaluation
            print("\n" + "=" * 80)
            print("STEP 5: MODEL EVALUATION")
            print("=" * 80)
            from scripts.evaluate_models import main as eval_main
            evaluation_results = eval_main()

            # Step 6: Log to MLflow and register each model (Staging; no auto-promote)
            print("\n" + "=" * 80)
            print("STEP 6: MLFLOW LOGGING & REGISTRY")
            print("=" * 80)

            models_dir = project_root / "models"
            registered_models = {}

            for model_name in ["logistic_regression_tuned", "random_forest_tuned", "xgboost_tuned"]:
                model_path = models_dir / f"{model_name}.pkl"
                metadata_path = models_dir / f"{model_name}_metadata.json"

                if not model_path.exists():
                    logger.warning(f"Model not found: {model_path}. Skipping.")
                    continue

                base_name = model_name.replace("_tuned", "")
                metrics = evaluation_results.get(base_name, {}).get("metrics", {})

                # Log metrics with prefix for this model
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{base_name}_{k}", float(v))

                # Log model artifact (sklearn or xgboost flavor) and register as new version (Staging)
                artifact_path = base_name
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                if "xgboost" in model_name:
                    mlflow_xgboost.log_model(model, artifact_path)
                else:
                    mlflow_sklearn.log_model(model, artifact_path)
                model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
                mv = mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)
                registered_models[model_name] = mv.version
                print(f"  Registered: {model_name} -> {MLFLOW_MODEL_NAME} v{mv.version} (Staging)")

            # Step 7: Best model and champion symlink (path-based serving)
            print("\n" + "=" * 80)
            print("STEP 7: CHAMPION SYMLINK")
            print("=" * 80)

            best_model = max(
                evaluation_results.items(),
                key=lambda x: x[1]["metrics"].get("roc_auc", 0),
            )
            best_model_name = best_model[0]
            best_metrics = best_model[1]["metrics"]

            print(f"Best model: {best_model_name}")
            print(f"  ROC-AUC: {best_metrics.get('roc_auc', 0):.4f}")

            # Champion alias: production.pkl -> best model (no MLflow stage change here; promote manually)
            best_pkl = models_dir / f"{best_model_name}_tuned.pkl"
            best_metadata = models_dir / f"{best_model_name}_tuned_metadata.json"
            production_pkl = models_dir / "production.pkl"
            production_metadata = models_dir / "production_metadata.json"
            if best_pkl.exists() and best_metadata.exists():
                for link_path, target_name in [
                    (production_pkl, best_pkl.name),
                    (production_metadata, best_metadata.name),
                ]:
                    if link_path.exists():
                        os.remove(link_path)
                    os.symlink(target_name, str(link_path))
                print(f"  Champion alias: production.pkl -> {best_pkl.name}")
            else:
                logger.warning("Best model artifact missing; skipping production symlink")

            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_roc_auc", best_metrics.get("roc_auc", 0))

            print("\n" + "=" * 80)
            print("RETRAINING COMPLETE")
            print("=" * 80)
            print(f"MLflow run: {run.info.run_id}")
            print(f"To promote a version to Production: python scripts/promote_model.py --version <N>")
            print("=" * 80)

            return {
                "run_id": run_id,
                "mlflow_run_id": run.info.run_id,
                "registered_models": registered_models,
                "best_model": best_model_name,
            }

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            raise


if __name__ == "__main__":
    main()
