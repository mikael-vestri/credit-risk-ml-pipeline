"""
Hyperparameter tuning module (Step 8).

This module provides functions to tune hyperparameters using GridSearch
or RandomizedSearch for all models.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
from datetime import datetime

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

from .trainers import prepare_features_and_target
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def get_parameter_grids() -> Dict[str, Dict[str, list]]:
    """
    Define parameter grids for hyperparameter tuning.
    
    Returns
    -------
    Dict with parameter grids for each model
    """
    return {
        "logistic_regression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__l1_ratio": [0],  # L2 regularization only (lbfgs solver supports only L2)
        },
        "random_forest": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [5, 10, 15, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2", None]
        },
        "xgboost": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 4, 5, 6, 7],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__gamma": [0, 0.1, 0.5, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__subsample": [0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.8, 0.9, 1.0]
        }
    }


def get_scoring_metrics() -> Dict[str, Any]:
    """
    Define scoring metrics for model evaluation.
    
    Returns
    -------
    Dict with scoring metrics
    """
    return {
        "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score)
    }


def tune_model(
    model: Pipeline,
    param_grid: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "roc_auc",
    cv: int = 5,
    method: str = "grid",
    n_iter: int = 20,
    n_jobs: int = -1,
    verbose: int = 1
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Tune hyperparameters for a model.
    
    Parameters
    ----------
    model : Pipeline
        Base model pipeline to tune
    param_grid : Dict[str, list]
        Parameter grid for tuning
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    scoring : str
        Primary scoring metric
    cv : int
        Number of cross-validation folds
    method : str
        "grid" for GridSearchCV, "random" for RandomizedSearchCV
    n_iter : int
        Number of iterations for RandomizedSearchCV
    n_jobs : int
        Number of parallel jobs
    verbose : int
        Verbosity level
        
    Returns
    -------
    Tuple of (best_model, tuning_results)
    """
    logger.info(f"Starting hyperparameter tuning ({method} search)...")
    logger.info(f"  Scoring metric: {scoring}")
    logger.info(f"  CV folds: {cv}")
    logger.info(f"  Parameter grid size: {sum(len(v) for v in param_grid.values())} combinations")
    
    if method == "grid":
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
    elif method == "random":
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=42,
            return_train_score=True
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'grid' or 'random'")
    
    # Perform search
    logger.info("Running hyperparameter search (this may take a while)...")
    search.fit(X_train, y_train)
    
    # Extract results
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV score ({scoring}): {best_score:.4f}")
    
    # Compile results
    tuning_results = {
        "best_params": best_params,
        "best_score": float(best_score),
        "scoring_metric": scoring,
        "cv_folds": cv,
        "method": method,
        "n_iterations": n_iter if method == "random" else len(search.cv_results_["params"]),
        "cv_results": {
            "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": search.cv_results_["std_test_score"].tolist(),
            "mean_train_score": search.cv_results_["mean_train_score"].tolist() if "mean_train_score" in search.cv_results_ else None,
            "params": [str(p) for p in search.cv_results_["params"]]
        }
    }
    
    return best_model, tuning_results


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "roc_auc",
    cv: int = 5,
    method: str = "random",
    n_iter: int = 20,
    tune_logistic: bool = True,
    tune_random_forest: bool = True,
    tune_xgboost: bool = True
) -> Dict[str, Tuple[Pipeline, Dict[str, Any]]]:
    """
    Tune hyperparameters for all models.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    scoring : str
        Primary scoring metric
    cv : int
        Number of cross-validation folds
    method : str
        "grid" for GridSearchCV, "random" for RandomizedSearchCV
    n_iter : int
        Number of iterations for RandomizedSearchCV
    tune_logistic : bool
        Whether to tune Logistic Regression
    tune_random_forest : bool
        Whether to tune Random Forest
    tune_xgboost : bool
        Whether to tune XGBoost
        
    Returns
    -------
    Dict mapping model names to (best_model, tuning_results) tuples
    """
    param_grids = get_parameter_grids()
    results = {}
    
    if tune_logistic:
        logger.info("\n" + "="*80)
        logger.info("TUNING: Logistic Regression")
        logger.info("="*80)
        # Create base pipeline (not trained yet - GridSearch will train it)
        # Using lbfgs solver which supports only L2 regularization
        base_model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
                l1_ratio=0  # L2 regularization (required for lbfgs)
            ))
        ])
        best_model, tuning_results = tune_model(
            base_model,
            param_grids["logistic_regression"],
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            method=method,
            n_iter=n_iter
        )
        results["logistic_regression"] = (best_model, tuning_results)
    
    if tune_random_forest:
        logger.info("\n" + "="*80)
        logger.info("TUNING: Random Forest")
        logger.info("="*80)
        # Create base pipeline (not trained yet)
        base_model = Pipeline([
            ("classifier", RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])
        best_model, tuning_results = tune_model(
            base_model,
            param_grids["random_forest"],
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            method=method,
            n_iter=n_iter
        )
        results["random_forest"] = (best_model, tuning_results)
    
    if tune_xgboost:
        logger.info("\n" + "="*80)
        logger.info("TUNING: XGBoost")
        logger.info("="*80)
        # Calculate scale_pos_weight for class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        # Create base pipeline (not trained yet)
        base_model = Pipeline([
            ("classifier", XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight
            ))
        ])
        best_model, tuning_results = tune_model(
            base_model,
            param_grids["xgboost"],
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            method=method,
            n_iter=n_iter
        )
        results["xgboost"] = (best_model, tuning_results)
    
    return results

