"""
Model interpretability module (Step 10).

This module provides SHAP-based interpretability analysis for trained models,
including global feature importance and local explanations for individual predictions.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pickle
import json
from datetime import datetime

import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .trainers import prepare_features_and_target
from .evaluation import load_model

logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Note: shap.initjs() is not needed for static plots (only for interactive JS visualizations)


def extract_model_from_pipeline(pipeline: Pipeline, model_name: str) -> Tuple[Any, Optional[Any]]:
    """
    Extract the underlying model from a scikit-learn Pipeline.
    
    Parameters
    ----------
    pipeline : Pipeline
        Trained scikit-learn Pipeline
    model_name : str
        Name of the model (for logging)
        
    Returns
    -------
    Tuple of (model, preprocessor)
    """
    # Get the classifier (last step in pipeline)
    classifier = pipeline.named_steps.get('classifier')
    
    # Get preprocessor if it exists (e.g., StandardScaler)
    preprocessor = pipeline.named_steps.get('scaler')
    
    if classifier is None:
        raise ValueError(f"Could not find 'classifier' step in pipeline for {model_name}")
    
    logger.info(f"Extracted {type(classifier).__name__} from pipeline")
    if preprocessor:
        logger.info(f"Found preprocessor: {type(preprocessor).__name__}")
    
    return classifier, preprocessor


def create_shap_explainer(
    model: Any,
    X_background: pd.DataFrame,
    model_name: str,
    preprocessor: Optional[Any] = None
) -> Any:
    """
    Create an appropriate SHAP explainer for the model.
    
    Parameters
    ----------
    model : Any
        Trained model
    X_background : pd.DataFrame
        Background dataset for SHAP (typically a sample of training data)
    model_name : str
        Name of the model
    preprocessor : Optional[Any]
        Preprocessor (e.g., StandardScaler) if model is in a pipeline
        
    Returns
    -------
    SHAP explainer object
    """
    model_type = type(model).__name__
    
    # Preprocess data if needed
    if preprocessor is not None:
        X_background_processed = preprocessor.transform(X_background)
        X_background_processed = pd.DataFrame(
            X_background_processed,
            columns=X_background.columns,
            index=X_background.index
        )
    else:
        X_background_processed = X_background
    
    # Create appropriate explainer based on model type
    if model_type in ['RandomForestClassifier', 'XGBClassifier']:
        logger.info(f"Using TreeExplainer for {model_type}")
        explainer = shap.TreeExplainer(model)
    elif model_type == 'LogisticRegression':
        logger.info(f"Using LinearExplainer for {model_type}")
        explainer = shap.LinearExplainer(model, X_background_processed)
    else:
        # Fallback to KernelExplainer (slower but works for any model)
        logger.warning(f"Using KernelExplainer (slower) for {model_type}")
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1],
            X_background_processed.iloc[:100]  # Use smaller sample for KernelExplainer
        )
    
    return explainer, X_background_processed


def compute_shap_values(
    explainer: Any,
    X_explain: pd.DataFrame,
    model: Any,
    model_name: str,
    max_evals: int = 100
) -> np.ndarray:
    """
    Compute SHAP values for given data.
    
    Parameters
    ----------
    explainer : Any
        SHAP explainer
    X_explain : pd.DataFrame
        Data to explain
    model : Any
        Trained model (for fallback explainers)
    model_name : str
        Name of the model
    max_evals : int
        Maximum evaluations for KernelExplainer
        
    Returns
    -------
    SHAP values array (2D numpy array: n_samples x n_features)
    """
    explainer_type = type(explainer).__name__
    
    if explainer_type == 'KernelExplainer':
        logger.info(f"Computing SHAP values using KernelExplainer (this may take a while)...")
        shap_values = explainer.shap_values(
            X_explain,
            nsamples=max_evals
        )
        # KernelExplainer returns list for binary classification, take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        logger.info(f"Computing SHAP values...")
        shap_values = explainer.shap_values(X_explain)
        
        # Handle different return formats
        if isinstance(shap_values, list):
            # List of arrays (one per class) - take positive class (index 1)
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray):
            # Check if it's 3D (n_samples, n_features, n_classes)
            if shap_values.ndim == 3:
                # Take positive class (index 1)
                shap_values = shap_values[:, :, 1]
            # Check if shape suggests both classes concatenated
            elif shap_values.ndim == 2 and shap_values.shape[1] == X_explain.shape[1] * 2:
                # Split into two classes and take positive class
                n_features = X_explain.shape[1]
                shap_values = shap_values[:, n_features:]
    
    # Ensure it's a numpy array and 2D
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        # If 1D, reshape to (1, n_features) - single sample
        shap_values = shap_values.reshape(1, -1)
    elif shap_values.ndim > 2:
        # If more than 2D, take the positive class dimension
        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]
        else:
            # Flatten to 2D
            shap_values = shap_values.reshape(shap_values.shape[0], -1)
    
    # Final check: ensure we have the right number of features
    if shap_values.shape[1] != X_explain.shape[1]:
        logger.warning(
            f"SHAP values shape {shap_values.shape} doesn't match features {X_explain.shape[1]}. "
            f"Attempting to fix..."
        )
        # If it's exactly double, take the second half (positive class)
        if shap_values.shape[1] == X_explain.shape[1] * 2:
            shap_values = shap_values[:, X_explain.shape[1]:]
        else:
            raise ValueError(
                f"Cannot reconcile SHAP values shape {shap_values.shape} with "
                f"features {X_explain.shape[1]}"
            )
    
    return shap_values


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    model_name: str,
    save_path: Optional[Path] = None,
    max_display: int = 20
) -> None:
    """
    Plot SHAP summary plot (beeswarm plot).
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    X_explain : pd.DataFrame
        Data that was explained
    model_name : str
        Name of the model
    save_path : Optional[Path]
        Path to save the plot
    max_display : int
        Maximum number of features to display
    """
    plt.figure(figsize=(12, 8))
    
    shap.summary_plot(
        shap_values,
        X_explain,
        plot_type="dot",
        max_display=max_display,
        show=False
    )
    
    plt.title(f'SHAP Summary Plot - {model_name.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to: {save_path}")
    
    plt.close()


def plot_shap_bar(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    shap_explainer: Any,
    model_name: str,
    save_path: Optional[Path] = None,
    max_display: int = 20
) -> None:
    """
    Plot SHAP bar plot (mean absolute SHAP values).
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    X_explain : pd.DataFrame
        Data that was explained
    shap_explainer : Any
        SHAP explainer object
    model_name : str
        Name of the model
    save_path : Optional[Path]
        Path to save the plot
    max_display : int
        Maximum number of features to display
    """
    try:
        # Get expected value
        if hasattr(shap_explainer, 'expected_value'):
            expected_value = shap_explainer.expected_value
            # Handle case where expected_value is an array
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            expected_value = 0.0
        
        # Create SHAP Explanation object
        shap_explanation = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=X_explain.values,
            feature_names=X_explain.columns.tolist()
        )
        
        plt.figure(figsize=(10, 8))
        
        shap.plots.bar(
            shap_explanation,
            max_display=max_display,
            show=False
        )
        
        plt.title(f'SHAP Feature Importance - {model_name.replace("_", " ").title()}', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP bar plot to: {save_path}")
        
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create bar plot: {e}")
        logger.warning("Skipping bar plot...")


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    shap_explainer: Any,
    instance_idx: int,
    model_name: str,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot SHAP waterfall plot for a single instance.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    X_explain : pd.DataFrame
        Data that was explained
    shap_explainer : Any
        SHAP explainer object
    instance_idx : int
        Index of instance to explain
    model_name : str
        Name of the model
    save_path : Optional[Path]
        Path to save the plot
    """
    try:
        # Get expected value
        if hasattr(shap_explainer, 'expected_value'):
            expected_value = shap_explainer.expected_value
            # Handle case where expected_value is an array
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            expected_value = 0.0
        
        # Create SHAP Explanation object for waterfall plot
        shap_explanation = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=expected_value,
            data=X_explain.iloc[instance_idx].values,
            feature_names=X_explain.columns.tolist()
        )
        
        plt.figure(figsize=(10, 8))
        
        shap.waterfall_plot(
            shap_explanation,
            max_display=15,
            show=False
        )
        
        plt.title(f'SHAP Waterfall Plot - {model_name.replace("_", " ").title()}\nInstance {instance_idx}', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP waterfall plot to: {save_path}")
        
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create waterfall plot for instance {instance_idx}: {e}")
        logger.warning("Skipping waterfall plot...")


def get_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Calculate feature importance from SHAP values.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values (n_samples, n_features)
    feature_names : List[str]
        List of feature names
        
    Returns
    -------
    DataFrame with feature importance (sorted by mean absolute SHAP value)
    """
    # Ensure shap_values is a 2D numpy array
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        # If 1D, reshape to (1, n_features)
        shap_values = shap_values.reshape(1, -1)
    elif shap_values.ndim > 2:
        # If more than 2D, flatten to 2D
        shap_values = shap_values.reshape(shap_values.shape[0], -1)
    
    # Ensure we have the right number of features
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"Shape mismatch: SHAP values have {shap_values.shape[1]} features, "
            f"but {len(feature_names)} feature names provided"
        )
    
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    std_shap = shap_values.std(axis=0)
    
    # Ensure all are 1D arrays
    mean_abs_shap = np.atleast_1d(mean_abs_shap).flatten()
    mean_shap = np.atleast_1d(mean_shap).flatten()
    std_shap = np.atleast_1d(std_shap).flatten()
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': mean_shap,
        'std_shap': std_shap
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
    
    return importance_df


def explain_model(
    model: Any,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    model_name: str,
    preprocessor: Optional[Any] = None,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform SHAP analysis on a model.
    
    Parameters
    ----------
    model : Any
        Trained model (Pipeline or raw model)
    X_background : pd.DataFrame
        Background dataset for SHAP (training data sample)
    X_explain : pd.DataFrame
        Data to explain (test data sample)
    model_name : str
        Name of the model
    preprocessor : Optional[Any]
        Preprocessor if model is not a Pipeline
    sample_size : Optional[int]
        Sample size for explanation (None = use all)
        
    Returns
    -------
    Dict with SHAP values and analysis results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"SHAP Analysis: {model_name}")
    logger.info(f"{'='*80}")
    
    # Extract model from pipeline if needed
    if isinstance(model, Pipeline):
        model, preprocessor = extract_model_from_pipeline(model, model_name)
    
    # Sample data if needed (for faster computation)
    if sample_size and len(X_background) > sample_size:
        logger.info(f"Sampling {sample_size} rows from background data ({len(X_background)} total)")
        X_background = X_background.sample(n=sample_size, random_state=42)
    
    if sample_size and len(X_explain) > sample_size:
        logger.info(f"Sampling {sample_size} rows from explanation data ({len(X_explain)} total)")
        X_explain = X_explain.sample(n=sample_size, random_state=42)
    
    # Create explainer
    explainer, X_background_processed = create_shap_explainer(
        model, X_background, model_name, preprocessor
    )
    
    # Preprocess explanation data if needed
    if preprocessor is not None:
        X_explain_processed = preprocessor.transform(X_explain)
        X_explain_processed = pd.DataFrame(
            X_explain_processed,
            columns=X_explain.columns,
            index=X_explain.index
        )
    else:
        X_explain_processed = X_explain
    
    # Compute SHAP values
    shap_values = compute_shap_values(explainer, X_explain_processed, model, model_name)
    
    # Get feature importance
    feature_importance = get_feature_importance(shap_values, X_explain.columns.tolist())
    
    logger.info(f"\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    return {
        "model_name": model_name,
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "explainer": explainer,
        "X_explain": X_explain,
        "X_explain_processed": X_explain_processed,
        "preprocessor": preprocessor
    }


def explain_all_models(
    models_dir: Path,
    train_data_path: Path,
    test_data_path: Path,
    output_dir: Optional[Path] = None,
    background_sample_size: int = 500,
    explanation_sample_size: int = 200
) -> Dict[str, Dict[str, Any]]:
    """
    Perform SHAP analysis on all tuned models.
    
    Parameters
    ----------
    models_dir : Path
        Directory containing saved models
    train_data_path : Path
        Path to training data (for background)
    test_data_path : Path
        Path to test data (for explanation)
    output_dir : Optional[Path]
        Directory to save SHAP visualizations and results
    background_sample_size : int
        Sample size for background data (for faster computation)
    explanation_sample_size : int
        Sample size for explanation data
        
    Returns
    -------
    Dict mapping model names to SHAP analysis results
    """
    logger.info("="*80)
    logger.info("MODEL INTERPRETABILITY - STEP 10")
    logger.info("="*80)
    
    # Load data
    logger.info(f"\nLoading training data from: {train_data_path}")
    df_train = pd.read_parquet(train_data_path)
    logger.info(f"Loaded {len(df_train):,} rows")
    
    logger.info(f"\nLoading test data from: {test_data_path}")
    df_test = pd.read_parquet(test_data_path)
    logger.info(f"Loaded {len(df_test):,} rows")
    
    # Prepare features and target
    logger.info("\nPreparing features...")
    X_train, y_train = prepare_features_and_target(df_train, target_column="target")
    X_test, y_test = prepare_features_and_target(df_test, target_column="target")
    
    # Find all tuned models
    model_files = {
        "logistic_regression": models_dir / "logistic_regression_tuned.pkl",
        "random_forest": models_dir / "random_forest_tuned.pkl",
        "xgboost": models_dir / "xgboost_tuned.pkl"
    }
    
    # Explain each model
    shap_results = {}
    
    for model_name, model_path in model_files.items():
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}. Skipping...")
            continue
        
        # Load model
        model = load_model(model_path)
        
        # Perform SHAP analysis
        result = explain_model(
            model=model,
            X_background=X_train,
            X_explain=X_test,
            model_name=model_name,
            sample_size=explanation_sample_size
        )
        
        shap_results[model_name] = result
        
        # Generate visualizations
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"\nGenerating visualizations for {model_name}...")
            
            # Summary plot
            plot_shap_summary(
                result["shap_values"],
                result["X_explain_processed"],
                model_name,
                model_output_dir / "shap_summary.png"
            )
            
            # Bar plot
            plot_shap_bar(
                result["shap_values"],
                result["X_explain_processed"],
                result["explainer"],
                model_name,
                model_output_dir / "shap_bar.png"
            )
            
            # Waterfall plots for a few instances
            for idx in [0, 1, 2]:
                plot_shap_waterfall(
                    result["shap_values"],
                    result["X_explain_processed"],
                    result["explainer"],
                    idx,
                    model_name,
                    model_output_dir / f"shap_waterfall_instance_{idx}.png"
                )
            
            # Save feature importance
            importance_path = model_output_dir / "feature_importance.csv"
            result["feature_importance"].to_csv(importance_path, index=False)
            logger.info(f"Saved feature importance to: {importance_path}")
    
    if not shap_results:
        raise ValueError("No models found to explain!")
    
    # Save summary
    if output_dir:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "background_sample_size": background_sample_size,
            "explanation_sample_size": explanation_sample_size,
            "models_analyzed": list(shap_results.keys()),
            "top_features_by_model": {
                name: result["feature_importance"].head(10)[["feature", "mean_abs_shap"]].to_dict("records")
                for name, result in shap_results.items()
            }
        }
        
        summary_path = output_dir / "shap_analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved SHAP analysis summary to: {summary_path}")
    
    logger.info("\n" + "="*80)
    logger.info("SHAP ANALYSIS COMPLETE")
    logger.info("="*80)
    
    return shap_results
