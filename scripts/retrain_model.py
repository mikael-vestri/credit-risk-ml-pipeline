"""
Script to retrain models with new data (Step 13 - Production Hardening).

This script orchestrates the complete retraining pipeline:
1. Data ingestion and cleaning
2. Feature engineering
3. Model training and tuning
4. Model evaluation
5. Model registration and promotion
"""

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from registry.artifact_registry import ArtifactRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run complete retraining pipeline."""

    # Generate unique run ID
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    print("=" * 80)
    print("MODEL RETRAINING PIPELINE - STEP 13")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 80)

    # Initialize registry
    registry_path = project_root / "artifacts" / "registry"
    registry = ArtifactRegistry(registry_path)

    # Register training run
    training_config = {
        "mode": "full",  # Use FULL mode for production retraining
        "models": ["logistic_regression", "random_forest", "xgboost"],
        "tuning": True,
        "cv_folds": 5,
    }

    registry.register_training_run(
        run_id=run_id, config=training_config, dataset_version=None  # TODO: Track dataset version
    )

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

        train_df, test_df, feature_metadata = engineer_main()

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

        # Step 6: Register models
        print("\n" + "=" * 80)
        print("STEP 6: REGISTERING MODELS")
        print("=" * 80)

        models_dir = project_root / "models"
        registered_models = {}

        for model_name in ["logistic_regression_tuned", "random_forest_tuned", "xgboost_tuned"]:
            model_path = models_dir / f"{model_name}.pkl"
            metadata_path = models_dir / f"{model_name}_metadata.json"

            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}. Skipping registration.")
                continue

            # Get metrics from evaluation results
            base_name = model_name.replace("_tuned", "")
            metrics = evaluation_results.get(base_name, {}).get("metrics", {})

            version_id = registry.register_model(
                model_name=base_name,
                model_path=model_path,
                metadata_path=metadata_path,
                metrics=metrics,
                training_run_id=run_id,
                stage="staging",  # Start in staging
            )

            registered_models[model_name] = version_id
            print(f"  Registered: {model_name} -> {version_id}")

        # Step 7: Identify best model and promote to production
        print("\n" + "=" * 80)
        print("STEP 7: MODEL PROMOTION")
        print("=" * 80)

        # Find best model based on ROC-AUC
        best_model = max(
            evaluation_results.items(), key=lambda x: x[1]["metrics"].get("roc_auc", 0)
        )

        best_model_name = best_model[0]
        best_metrics = best_model[1]["metrics"]

        print(f"Best model: {best_model_name}")
        print(f"  ROC-AUC: {best_metrics.get('roc_auc', 0):.4f}")
        print(f"  Precision: {best_metrics.get('precision', 0):.4f}")
        print(f"  Recall: {best_metrics.get('recall', 0):.4f}")

        # Promote best model to production
        if best_model_name in registered_models:
            version_id = registered_models[f"{best_model_name}_tuned"]
            registry.promote_model(
                model_name=best_model_name, version_id=version_id, target_stage="production"
            )
            print(f"  Promoted {best_model_name} to production")
        else:
            logger.warning(f"Best model {best_model_name} not found in registered models")

        # Update training run with results
        registry.update_training_run(
            run_id=run_id,
            results={
                "evaluation_results": evaluation_results,
                "registered_models": registered_models,
                "best_model": best_model_name,
                "best_metrics": best_metrics,
            },
            completed=True,
        )

        # Summary
        print("\n" + "=" * 80)
        print("RETRAINING COMPLETE")
        print("=" * 80)
        summary = registry.get_registry_summary()
        print(f"Total models in registry: {summary['total_models']}")
        print(f"Production models: {summary['production_models']}")
        print(f"Registry location: {registry_path}")
        print("=" * 80)

        return {
            "run_id": run_id,
            "registered_models": registered_models,
            "best_model": best_model_name,
            "summary": summary,
        }

    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        registry.update_training_run(run_id=run_id, results={"error": str(e)}, completed=False)
        raise


if __name__ == "__main__":
    result = main()
