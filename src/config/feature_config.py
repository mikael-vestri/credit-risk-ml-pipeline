"""
Feature engineering configuration - simplified.

This module contains minimal configuration for feature engineering.
Most feature logic is now explicit in src/features/builders.py
"""

# Features to exclude from modeling
EXCLUDED_FEATURES = [
    "id",  # Unique identifier, not useful for prediction
    "url",  # Not useful
    "zip_code",  # Too granular, use addr_state instead
]
