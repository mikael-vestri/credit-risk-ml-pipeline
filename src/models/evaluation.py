"""
Model evaluation module (Step 9).

This module provides functions to evaluate trained models on the test set,
calculate metrics, generate visualizations, and compare model performance.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pickle
import json
from datetime import datetime

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from .trainers import prepare_features_and_target

logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_model(model_path: Path) -> Any:
    """
    Load a trained model from disk.
    
    Parameters
    ----------
    model_path : Path
        Path to the saved model (.pkl file)
        
    Returns
    -------
    Loaded model object
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, Any]:
    """
    Evaluate a model on the test set and calculate metrics.
    
    Parameters
    ----------
    model : Any
        Trained model (Pipeline)
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    model_name : str
        Name of the model (for logging)
        
    Returns
    -------
    Dict with evaluation metrics and predictions
    """
    logger.info(f"\nEvaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
    
    # Precision-Recall curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    results = {
        "model_name": model_name,
        "metrics": {
            "roc_auc": float(roc_auc),
            "average_precision": float(avg_precision),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_positive_rate": float(true_positive_rate),
            "false_positive_rate": float(false_positive_rate),
            "true_negative_rate": float(true_negative_rate),
            "false_negative_rate": float(false_negative_rate)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "classification_report": class_report,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist()
        },
        "pr_curve": {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
            "thresholds": pr_thresholds.tolist()
        },
        "predictions": {
            "y_pred": y_pred.tolist(),
            "y_pred_proba": y_pred_proba.tolist()
        }
    }
    
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  Average Precision: {avg_precision:.4f}")
    
    return results


def plot_roc_curves(
    evaluation_results: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None
) -> None:
    """
    Plot ROC curves for all models.
    
    Parameters
    ----------
    evaluation_results : Dict[str, Dict[str, Any]]
        Dictionary mapping model names to evaluation results
    save_path : Optional[Path]
        Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)', linewidth=2)
    
    # Plot ROC curve for each model
    for model_name, results in evaluation_results.items():
        roc_data = results["roc_curve"]
        fpr = np.array(roc_data["fpr"])
        tpr = np.array(roc_data["tpr"])
        auc = results["metrics"]["roc_auc"]
        
        plt.plot(
            fpr, tpr,
            label=f'{model_name.replace("_", " ").title()} (AUC = {auc:.3f})',
            linewidth=2
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curves plot to: {save_path}")
    
    plt.close()


def plot_pr_curves(
    evaluation_results: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None
) -> None:
    """
    Plot Precision-Recall curves for all models.
    
    Parameters
    ----------
    evaluation_results : Dict[str, Dict[str, Any]]
        Dictionary mapping model names to evaluation results
    save_path : Optional[Path]
        Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot PR curve for each model
    for model_name, results in evaluation_results.items():
        pr_data = results["pr_curve"]
        precision = np.array(pr_data["precision"])
        recall = np.array(pr_data["recall"])
        avg_precision = results["metrics"]["average_precision"]
        
        plt.plot(
            recall, precision,
            label=f'{model_name.replace("_", " ").title()} (AP = {avg_precision:.3f})',
            linewidth=2
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PR curves plot to: {save_path}")
    
    plt.close()


def plot_confusion_matrices(
    evaluation_results: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None
) -> None:
    """
    Plot confusion matrices for all models.
    
    Parameters
    ----------
    evaluation_results : Dict[str, Dict[str, Any]]
        Dictionary mapping model names to evaluation results
    save_path : Optional[Path]
        Path to save the plot
    """
    n_models = len(evaluation_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(evaluation_results.items()):
        cm_data = results["confusion_matrix"]
        cm = np.array([
            [cm_data["true_negatives"], cm_data["false_positives"]],
            [cm_data["false_negatives"], cm_data["true_positives"]]
        ])
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'],
            ax=axes[idx],
            cbar_kws={'label': 'Count'}
        )
        
        axes[idx].set_title(
            f'{model_name.replace("_", " ").title()}\n'
            f'F1: {results["metrics"]["f1_score"]:.3f}',
            fontsize=12,
            fontweight='bold'
        )
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrices plot to: {save_path}")
    
    plt.close()


def plot_metrics_comparison(
    evaluation_results: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None
) -> None:
    """
    Plot bar chart comparing key metrics across models.
    
    Parameters
    ----------
    evaluation_results : Dict[str, Dict[str, Any]]
        Dictionary mapping model names to evaluation results
    save_path : Optional[Path]
        Path to save the plot
    """
    metrics_to_plot = ['roc_auc', 'precision', 'recall', 'f1_score']
    model_names = [name.replace('_', ' ').title() for name in evaluation_results.keys()]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        values = [results["metrics"][metric] for results in evaluation_results.values()]
        
        bars = axes[idx].bar(model_names, values, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].set_ylim([0, 1.05])
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9
            )
        
        # Rotate x-axis labels if needed
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Model Performance Metrics Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics comparison plot to: {save_path}")
    
    plt.close()


def compare_models(
    evaluation_results: Dict[str, Dict[str, Any]],
    primary_metric: str = "roc_auc"
) -> Tuple[str, Dict[str, Any]]:
    """
    Compare models and select the best one based on primary metric.
    
    Parameters
    ----------
    evaluation_results : Dict[str, Dict[str, Any]]
        Dictionary mapping model names to evaluation results
    primary_metric : str
        Primary metric to use for comparison (default: "roc_auc")
        
    Returns
    -------
    Tuple of (best_model_name, best_model_results)
    """
    logger.info(f"\nComparing models using {primary_metric}...")
    
    best_model = None
    best_score = -np.inf
    best_results = None
    
    for model_name, results in evaluation_results.items():
        score = results["metrics"][primary_metric]
        logger.info(f"  {model_name}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model_name
            best_results = results
    
    logger.info(f"\nBest model: {best_model} ({primary_metric} = {best_score:.4f})")
    
    return best_model, best_results


def evaluate_all_models(
    models_dir: Path,
    test_data_path: Path,
    output_dir: Optional[Path] = None,
    primary_metric: str = "roc_auc"
) -> Dict[str, Any]:
    """
    Evaluate all tuned models on the test set.
    
    Parameters
    ----------
    models_dir : Path
        Directory containing saved models
    test_data_path : Path
        Path to test data (parquet file)
    output_dir : Optional[Path]
        Directory to save evaluation results and plots
    primary_metric : str
        Primary metric for model comparison
        
    Returns
    -------
    Dict with evaluation results for all models
    """
    # Load test data
    logger.info(f"\nLoading test data from: {test_data_path}")
    df_test = pd.read_parquet(test_data_path)
    logger.info(f"Loaded {len(df_test):,} rows and {len(df_test.columns)} columns")
    
    # Prepare features and target
    logger.info("\nPreparing features and target...")
    X_test, y_test = prepare_features_and_target(df_test, target_column="target")
    
    # Find all tuned models
    model_files = {
        "logistic_regression": models_dir / "logistic_regression_tuned.pkl",
        "random_forest": models_dir / "random_forest_tuned.pkl",
        "xgboost": models_dir / "xgboost_tuned.pkl"
    }
    
    # Evaluate each model
    evaluation_results = {}
    
    for model_name, model_path in model_files.items():
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}. Skipping...")
            continue
        
        # Load model
        model = load_model(model_path)
        
        # Evaluate
        results = evaluate_model(model, X_test, y_test, model_name)
        evaluation_results[model_name] = results
    
    if not evaluation_results:
        raise ValueError("No models found to evaluate!")
    
    # Compare models
    best_model_name, best_model_results = compare_models(evaluation_results, primary_metric)
    
    # Generate visualizations
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\nGenerating visualizations...")
        plot_roc_curves(evaluation_results, output_dir / "roc_curves.png")
        plot_pr_curves(evaluation_results, output_dir / "pr_curves.png")
        plot_confusion_matrices(evaluation_results, output_dir / "confusion_matrices.png")
        plot_metrics_comparison(evaluation_results, output_dir / "metrics_comparison.png")
        
        # Save evaluation results
        summary = {
            "timestamp": datetime.now().isoformat(),
            "test_set_size": len(X_test),
            "primary_metric": primary_metric,
            "best_model": best_model_name,
            "best_score": best_model_results["metrics"][primary_metric],
            "model_results": {
                name: {
                    "metrics": results["metrics"],
                    "confusion_matrix": results["confusion_matrix"]
                }
                for name, results in evaluation_results.items()
            }
        }
        
        summary_path = output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved evaluation summary to: {summary_path}")
        
        # Save detailed results
        detailed_path = output_dir / "evaluation_results.json"
        with open(detailed_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        logger.info(f"Saved detailed results to: {detailed_path}")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    
    return evaluation_results
