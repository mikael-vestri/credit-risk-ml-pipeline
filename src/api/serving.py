"""
Model serving module.

This module provides functions to load models and make predictions.
Supports loading from path (MODEL_PATH) or from MLflow Model Registry (Production stage).
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_model_from_mlflow(
    model_name: str = "credit-risk-model",
    stage: str = "Production",
    tracking_uri: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Load the production model from MLflow Model Registry.

    Returns (model, metadata) where metadata has model_name, feature_count, feature_names
    for compatibility with get_model_info and the API.
    """
    import mlflow
    from mlflow import pyfunc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    uri = f"models:/{model_name}/{stage}"
    logger.info(f"Loading model from MLflow: {uri}")
    model = pyfunc.load_model(uri)
    # Build metadata from MLflow signature if available
    feature_names = []
    if hasattr(model, "metadata") and model.metadata and hasattr(model.metadata, "signature"):
        sig = model.metadata.signature
        if sig and sig.inputs:
            feature_names = [inp.name for inp in sig.inputs]
    metadata = {
        "model_name": model_name,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "timestamp": "mlflow",
    }
    logger.info(f"Model loaded from MLflow: {model_name} (features: {len(feature_names)})")
    return model, metadata


def load_model_from_disk(model_path: Path) -> Any:
    """
    Load a trained model from disk.

    Parameters
    ----------
    model_path : Path
        Path to the saved model (.pkl file)

    Returns
    -------
    Loaded model (Pipeline or XGBClassifier)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Model loaded successfully: {type(model).__name__}")
    return model


def load_model_metadata(metadata_path: Path) -> dict[str, Any]:
    """
    Load model metadata.

    Parameters
    ----------
    metadata_path : Path
        Path to the metadata JSON file

    Returns
    -------
    Dictionary with model metadata
    """
    import json

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    return metadata


def prepare_input_features(input_data: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    """
    Prepare input features from a dictionary to a DataFrame with correct feature order.

    Parameters
    ----------
    input_data : Dict[str, Any]
        Dictionary with feature names as keys and values
    feature_names : List[str]
        List of expected feature names in the correct order

    Returns
    -------
    pd.DataFrame with features in the correct order
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_data])

    # Check for missing features
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}. "
            f"Expected {len(feature_names)} features, got {len(df.columns)}"
        )

    # Check for extra features (warn but don't fail)
    extra_features = set(df.columns) - set(feature_names)
    if extra_features:
        logger.warning(f"Extra features provided (will be ignored): {extra_features}")

    # Select and order features correctly
    df_features = df[feature_names].copy()

    # Ensure numeric types
    for col in df_features.columns:
        df_features[col] = pd.to_numeric(df_features[col], errors="coerce")

    # Check for NaN values
    nan_cols = df_features.columns[df_features.isna().any()].tolist()
    if nan_cols:
        raise ValueError(f"NaN values found in features: {nan_cols}")

    return df_features


def predict_default_probability(
    model: Any, features: pd.DataFrame, return_proba: bool = True
) -> dict[str, Any]:
    """
    Make a prediction using the loaded model.

    Parameters
    ----------
    model : Any
        Loaded model (Pipeline or XGBClassifier)
    features : pd.DataFrame
        Input features DataFrame
    return_proba : bool
        If True, return probability; if False, return binary prediction

    Returns
    -------
    Dictionary with prediction results
    """
    try:
        # Make prediction
        if return_proba:
            # Get probability of default (class 1)
            proba = model.predict_proba(features)[0]
            default_probability = float(proba[1])  # Probability of class 1 (default)
            prediction = int(default_probability >= 0.5)  # Binary prediction
        else:
            prediction = int(model.predict(features)[0])
            default_probability = None

        result = {
            "prediction": prediction,
            "default_probability": default_probability,
            "status": "success",
        }

        return result

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise RuntimeError(f"Failed to make prediction: {str(e)}") from e


def get_model_info(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Extract model information from metadata.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Model metadata dictionary

    Returns
    -------
    Dictionary with model information
    """
    return {
        "model_name": metadata.get("model_name", "unknown"),
        "feature_count": metadata.get("feature_count", 0),
        "timestamp": metadata.get("timestamp", "unknown"),
        "feature_names": metadata.get("feature_names", []),
    }
