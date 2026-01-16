"""
Script to train models (Step 7).

This script trains baseline and advanced models for credit risk prediction.
"""

import sys
from pathlib import Path
import logging
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import dataset_config
from models.trainers import (
    prepare_features_and_target,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    save_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Train all models."""
    
    # Set up paths
    train_data_path = project_root / dataset_config.PROCESSED_DATA_PATH / "engineered_features_train.parquet"
    models_dir = project_root / "models"
    
    print("="*80)
    print("MODEL TRAINING - STEP 7")
    print("="*80)
    print("\nThis script will train:")
    print("  1. Logistic Regression (baseline)")
    print("  2. Random Forest (baseline)")
    print("  3. XGBoost (advanced)")
    print("="*80)
    
    # Load training data
    print(f"\nLoading training data from: {train_data_path}")
    df_train = pd.read_parquet(train_data_path)
    print(f"Loaded {len(df_train):,} rows and {len(df_train.columns)} columns")
    
    # Prepare features and target
    print("\nPreparing features and target...")
    X_train, y_train = prepare_features_and_target(df_train, target_column="target")
    
    # Train models
    trained_models = {}
    
    # 1. Logistic Regression
    print("\n" + "="*80)
    print("TRAINING: Logistic Regression")
    print("="*80)
    lr_model = train_logistic_regression(X_train, y_train)
    trained_models["logistic_regression"] = lr_model
    
    # Save Logistic Regression
    save_paths_lr = save_model(
        lr_model,
        "logistic_regression",
        models_dir,
        feature_names=X_train.columns.tolist()
    )
    
    # 2. Random Forest
    print("\n" + "="*80)
    print("TRAINING: Random Forest")
    print("="*80)
    rf_model = train_random_forest(X_train, y_train)
    trained_models["random_forest"] = rf_model
    
    # Save Random Forest
    save_paths_rf = save_model(
        rf_model,
        "random_forest",
        models_dir,
        feature_names=X_train.columns.tolist()
    )
    
    # 3. XGBoost
    print("\n" + "="*80)
    print("TRAINING: XGBoost")
    print("="*80)
    xgb_model = train_xgboost(X_train, y_train)
    trained_models["xgboost"] = xgb_model
    
    # Save XGBoost
    save_paths_xgb = save_model(
        xgb_model,
        "xgboost",
        models_dir,
        feature_names=X_train.columns.tolist()
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Trained {len(trained_models)} models:")
    for model_name in trained_models.keys():
        print(f"  - {model_name}")
    print(f"\nModels saved to: {models_dir}")
    print("="*80)
    
    return trained_models

if __name__ == "__main__":
    trained_models = main()

