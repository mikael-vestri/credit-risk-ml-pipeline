"""
Model training module.

This module provides functions to train baseline and advanced models
for credit risk prediction.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pickle
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str = "target"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for training.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features and target
    target_column : str
        Name of target column
        
    Returns
    -------
    Tuple of (features DataFrame, target Series)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Remove any remaining non-numeric columns (should be minimal after feature engineering)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    logger.info(f"Prepared features: {X.shape[1]} features, {len(X)} samples")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs
) -> Pipeline:
    """
    Train a Logistic Regression model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    **kwargs
        Additional arguments for LogisticRegression
        
    Returns
    -------
    Pipeline with StandardScaler and LogisticRegression
    """
    logger.info("Training Logistic Regression model...")
    
    # Default parameters
    # C: Inverse of regularization strength (smaller = stronger regularization)
    # Default is C=1.0, but we set it explicitly for clarity
    # Note: penalty="l2" is default for lbfgs, but sklearn 1.8+ deprecated explicit penalty
    # We use l1_ratio=0 for L2 (equivalent to penalty="l2")
    default_params = {
        "C": 1.0,  # Regularization: smaller = stronger (prevents overfitting)
        "max_iter": 1000,
        "random_state": 42,
        "solver": "lbfgs",
        "l1_ratio": 0,  # L2 regularization (0 = L2, 1 = L1) - avoids deprecation warning
        "class_weight": "balanced"  # Handle class imbalance
    }
    default_params.update(kwargs)
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**default_params))
    ])
    
    pipeline.fit(X_train, y_train)
    
    logger.info("Logistic Regression training complete")
    
    return pipeline


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs
) -> Pipeline:
    """
    Train a Random Forest model.
    
    Note: Tree-based models don't require scaling, but we wrap in Pipeline
    for consistency and easier deployment.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    **kwargs
        Additional arguments for RandomForestClassifier
        
    Returns
    -------
    Pipeline with RandomForestClassifier
    """
    logger.info("Training Random Forest model...")
    
    # Default parameters
    default_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "class_weight": "balanced",  # Handle class imbalance
        "n_jobs": -1
    }
    default_params.update(kwargs)
    
    # Wrap in Pipeline for consistency (tree models don't need scaling)
    pipeline = Pipeline([
        ("classifier", RandomForestClassifier(**default_params))
    ])
    
    pipeline.fit(X_train, y_train)
    
    logger.info("Random Forest training complete")
    
    return pipeline


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs
) -> Pipeline:
    """
    Train an XGBoost model.
    
    Note: Tree-based models don't require scaling, but we wrap in Pipeline
    for consistency and easier deployment.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    **kwargs
        Additional arguments for XGBClassifier
        
    Returns
    -------
    Pipeline with XGBClassifier
    """
    logger.info("Training XGBoost model...")
    
    # Default parameters
    default_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "eval_metric": "logloss"
        # Note: use_label_encoder removed (deprecated in newer XGBoost versions)
    }
    default_params.update(kwargs)
    
    # Handle class imbalance with scale_pos_weight
    if "scale_pos_weight" not in default_params:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        if pos_count > 0:
            default_params["scale_pos_weight"] = neg_count / pos_count
    
    # Wrap in Pipeline for consistency (tree models don't need scaling)
    pipeline = Pipeline([
        ("classifier", XGBClassifier(**default_params))
    ])
    
    pipeline.fit(X_train, y_train)
    
    logger.info("XGBoost training complete")
    
    return pipeline


def save_model(
    model: Any,
    model_name: str,
    save_dir: Path,
    feature_names: Optional[list] = None
) -> Dict[str, str]:
    """
    Save trained model to disk.
    
    Parameters
    ----------
    model : Any
        Trained model object
    model_name : str
        Name of the model (e.g., 'logistic_regression', 'xgboost')
    save_dir : Path
        Directory to save the model
    feature_names : Optional[list]
        List of feature names (for reference)
        
    Returns
    -------
    Dict with paths to saved files
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = save_dir / f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved model to: {model_path}")
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "model_path": str(model_path),
        "timestamp": datetime.now().isoformat(),
        "feature_count": len(feature_names) if feature_names else None,
        "feature_names": feature_names if feature_names else None
    }
    
    metadata_path = save_dir / f"{model_name}_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved metadata to: {metadata_path}")
    
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path)
    }

