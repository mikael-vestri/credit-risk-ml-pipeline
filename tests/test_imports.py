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
    """Test that model modules can be imported (restrict path so only src provides 'models')."""
    root = Path(__file__).resolve().parent.parent
    src = str((root / "src").resolve())
    # Ensure only our src can provide 'models' (avoid cwd or other 'models' shadowing)
    old_path = sys.path.copy()
    sys.path = [src] + [p for p in old_path if p not in ("", ".")]
    try:
        from models import evaluation, trainers, tuning
        assert trainers is not None
        assert tuning is not None
        assert evaluation is not None
    finally:
        sys.path = old_path


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
