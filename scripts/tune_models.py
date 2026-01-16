"""
Script to tune model hyperparameters (Step 8).

This script performs hyperparameter tuning using GridSearch or RandomizedSearch
to find optimal parameters for all models.
"""

import sys
from pathlib import Path
import logging
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import dataset_config
from models.tuning import tune_all_models
from models.trainers import prepare_features_and_target, save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Tune hyperparameters for all models."""
    
    # Set up paths
    train_data_path = project_root / dataset_config.PROCESSED_DATA_PATH / "engineered_features_train.parquet"
    models_dir = project_root / "models"
    tuning_results_dir = project_root / "docs"
    
    # Tuning configuration
    scoring = "roc_auc"  # Primary metric (can be changed)
    cv = 5  # Cross-validation folds
    method = "random"  # "grid" or "random"
    n_iter = 20  # For RandomizedSearch (ignored for GridSearch)
    
    print("="*80)
    print("HYPERPARAMETER TUNING - STEP 8")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Scoring metric: {scoring}")
    print(f"  CV folds: {cv}")
    print(f"  Method: {method.upper()} search")
    if method == "random":
        print(f"  Iterations: {n_iter}")
    print("\nThis will tune:")
    print("  1. Logistic Regression")
    print("  2. Random Forest")
    print("  3. XGBoost")
    print("\nNote: This may take several minutes...")
    print("="*80)
    
    # Load training data
    print(f"\nLoading training data from: {train_data_path}")
    df_train = pd.read_parquet(train_data_path)
    print(f"Loaded {len(df_train):,} rows and {len(df_train.columns)} columns")
    
    # Prepare features and target
    print("\nPreparing features and target...")
    X_train, y_train = prepare_features_and_target(df_train, target_column="target")
    
    # Tune all models
    print("\n" + "="*80)
    print("STARTING HYPERPARAMETER TUNING")
    print("="*80)
    
    tuning_results = tune_all_models(
        X_train,
        y_train,
        scoring=scoring,
        cv=cv,
        method=method,
        n_iter=n_iter
    )
    
    # Save tuned models
    print("\n" + "="*80)
    print("SAVING TUNED MODELS")
    print("="*80)
    
    saved_models = {}
    for model_name, (best_model, results) in tuning_results.items():
        print(f"\nSaving tuned {model_name}...")
        save_paths = save_model(
            best_model,
            f"{model_name}_tuned",
            models_dir,
            feature_names=X_train.columns.tolist()
        )
        saved_models[model_name] = save_paths
        
        # Save tuning results
        results_path = tuning_results_dir / f"{model_name}_tuning_results.json"
        tuning_results_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Tuning results saved to: {results_path}")
    
    # Summary
    print("\n" + "="*80)
    print("TUNING COMPLETE")
    print("="*80)
    print(f"\nTuned {len(tuning_results)} models:")
    for model_name, (best_model, results) in tuning_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Best CV score ({scoring}): {results['best_score']:.4f}")
        print(f"  Best parameters:")
        for param, value in results['best_params'].items():
            print(f"    {param}: {value}")
    
    print(f"\nTuned models saved to: {models_dir}")
    print(f"Tuning results saved to: {tuning_results_dir}")
    print("="*80)
    
    return tuning_results

if __name__ == "__main__":
    tuning_results = main()

