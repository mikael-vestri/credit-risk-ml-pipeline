"""
Model training and evaluation module.
"""

from .trainers import (
    prepare_features_and_target,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    save_model
)
from .tuning import (
    get_parameter_grids,
    get_scoring_metrics,
    tune_model,
    tune_all_models
)
from .evaluation import (
    load_model,
    evaluate_model,
    evaluate_all_models,
    compare_models,
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrices,
    plot_metrics_comparison
)
from .interpretability import (
    explain_model,
    explain_all_models,
    get_feature_importance,
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_waterfall
)

__all__ = [
    'prepare_features_and_target',
    'train_logistic_regression',
    'train_random_forest',
    'train_xgboost',
    'save_model',
    'get_parameter_grids',
    'get_scoring_metrics',
    'tune_model',
    'tune_all_models',
    'load_model',
    'evaluate_model',
    'evaluate_all_models',
    'compare_models',
    'plot_roc_curves',
    'plot_pr_curves',
    'plot_confusion_matrices',
    'plot_metrics_comparison',
    'explain_model',
    'explain_all_models',
    'get_feature_importance',
    'plot_shap_summary',
    'plot_shap_bar',
    'plot_shap_waterfall',
]
