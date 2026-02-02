"""
Test that all modules can be imported without errors.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_api_imports():
    """Test that API modules can be imported."""
    from api import app, serving

    assert app is not None
    assert serving is not None


def test_data_imports():
    """Test that data modules can be imported."""
    from data import (
        cleaning,
        ingestion,
        pipeline,
    )

    assert ingestion is not None
    assert cleaning is not None
    assert pipeline is not None


def test_models_imports():
    """Test that model modules can be imported (load by path to avoid 'models' name shadowing)."""
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    models_init = src / "models" / "__init__.py"
    assert models_init.exists(), f"Expected {models_init} to exist"

    # Load models package from path and register in sys.modules so it's findable
    spec = importlib.util.spec_from_file_location(
        "models",
        models_init,
        submodule_search_locations=[str(src / "models")],
    )
    models_pkg = importlib.util.module_from_spec(spec)
    sys.modules["models"] = models_pkg
    spec.loader.exec_module(models_pkg)

    # __init__.py imports from .trainers, .tuning, .evaluation; verify they're loaded
    assert hasattr(models_pkg, "train_logistic_regression")
    assert hasattr(models_pkg, "tune_all_models")
    assert hasattr(models_pkg, "evaluate_all_models")


def test_features_imports():
    """Test that feature modules can be imported."""
    from features import builders, pipeline

    assert builders is not None
    assert pipeline is not None


def test_config_imports():
    """Test that config modules can be imported."""
    from config import dataset_config, feature_config

    assert dataset_config is not None
    assert feature_config is not None
