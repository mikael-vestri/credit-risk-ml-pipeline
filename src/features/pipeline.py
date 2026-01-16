"""
Feature engineering pipeline.

This module orchestrates the complete feature engineering process:
1. Load cleaned data
2. Perform temporal split (BEFORE feature engineering)
3. Apply feature engineering to train/test separately
4. Handle target variable
5. Save engineered features

Critical: Split happens BEFORE feature engineering to avoid leakage.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def prepare_target(df: pd.DataFrame, target_column: str, target_values: Dict) -> pd.DataFrame:
    """
    Prepare target variable by converting to binary and filtering ongoing loans.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str
        Name of target column
    target_values : Dict
        Mapping of target values to binary labels
        
    Returns
    -------
    pd.DataFrame with target prepared and ongoing loans removed
    """
    df_new = df.copy()
    
    if target_column not in df.columns:
        logger.warning(f"Target column '{target_column}' not found!")
        return df_new
    
    # Convert target to binary
    df_new["target"] = df_new[target_column].map(target_values)
    
    # Remove ongoing loans (None values)
    ongoing_mask = df_new["target"].isna()
    ongoing_count = ongoing_mask.sum()
    
    if ongoing_count > 0:
        logger.info(f"Removing {ongoing_count:,} ongoing loans (target=None)")
        df_new = df_new[~ongoing_mask].copy()
    
    # Convert to int
    df_new["target"] = df_new["target"].astype(int)
    
    logger.info(f"Target prepared: {df_new['target'].value_counts().to_dict()}")
    
    return df_new


def select_features(
    df: pd.DataFrame,
    excluded_features: List[str],
    target_column: str = "target"
) -> pd.DataFrame:
    """
    Select features to use (exclude specified columns).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    excluded_features : List[str]
        List of feature names to exclude
    target_column : str
        Name of target column (always keep)
        
    Returns
    -------
    pd.DataFrame with selected features
    """
    # Always keep target
    features_to_keep = [col for col in df.columns 
                       if col not in excluded_features or col == target_column]
    
    df_selected = df[features_to_keep].copy()
    
    logger.info(f"Feature selection: {len(df.columns)} -> {len(df_selected.columns)} columns")
    if excluded_features:
        logger.info(f"Excluded: {excluded_features}")
    
    return df_selected


def engineer_features_pipeline(
    cleaned_data_path: Path,
    date_column: str,
    output_path: Optional[Path] = None,
    excluded_features: Optional[List[str]] = None,
    mode: str = "dev",
    split_date: Optional[str] = None,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Complete feature engineering pipeline.
    
    Steps:
    1. Load cleaned data
    2. Prepare target variable
    3. Compute or validate split date
    4. Perform temporal split (BEFORE feature engineering)
    5. Apply feature engineering to train/test separately
    6. Select features
    7. Save engineered features
    
    Parameters
    ----------
    cleaned_data_path : Path
        Path to cleaned data (parquet)
    date_column : str
        Name of date column for temporal split
    output_path : Optional[Path]
        Path to save engineered features
    excluded_features : Optional[List[str]]
        List of features to exclude (e.g., ['id', 'url'])
        If None, uses default: ['id', 'url', 'zip_code']
    mode : str
        Execution mode: 'dev' or 'full'
    split_date : Optional[str]
        Split date (format: 'YYYY-MM-DD'). If None, computed from data.
    train_ratio : float
        Train ratio for computing split date (default 0.8 = 80/20)
        
    Returns
    -------
    Tuple of (train_df, test_df, metadata)
    """
    logger.info("="*80)
    logger.info(f"FEATURE ENGINEERING PIPELINE - STEP 6 [{mode.upper()} MODE]")
    logger.info("="*80)
    
    # Import here to avoid circular imports
    from config import dataset_config
    from data.splitting import perform_temporal_split
    from data.validation import validate_temporal_split_correctness
    from data.split_utils import compute_split_date, validate_split_date
    from features.builders import engineer_features
    
    # Validate mode
    if mode not in ["dev", "full"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'dev' or 'full'")
    
    # Default excluded features
    if excluded_features is None:
        excluded_features = ['id', 'url', 'zip_code']
    
    # Step 1: Load cleaned data
    logger.info(f"\nStep 1: Loading cleaned data from {cleaned_data_path}")
    logger.info(f"[MODE: {mode.upper()}] Loading cleaned parquet data...")
    
    if str(cleaned_data_path).endswith('.parquet'):
        df_cleaned = pd.read_parquet(cleaned_data_path)
    else:
        df_cleaned = pd.read_csv(cleaned_data_path)
    
    logger.info(f"Loaded {len(df_cleaned):,} rows and {len(df_cleaned.columns)} columns")
    
    # Check date range
    dates = pd.to_datetime(df_cleaned[date_column], errors='coerce')
    date_min = dates.min()
    date_max = dates.max()
    logger.info(f"Date range: {date_min} to {date_max}")
    
    # Step 2: Prepare target variable
    logger.info("\nStep 2: Preparing target variable...")
    df_cleaned = prepare_target(
        df_cleaned,
        dataset_config.TARGET_COLUMN,
        dataset_config.TARGET_VALUES
    )
    logger.info(f"After target preparation: {len(df_cleaned):,} rows")
    
    # Step 3: Compute or validate split date
    logger.info("\nStep 3: Determining temporal split date...")
    if split_date is None:
        logger.info(f"Computing split date from data (train_ratio={train_ratio:.1%})...")
        split_date_dt = compute_split_date(df_cleaned, date_column, train_ratio=train_ratio)
        split_date = split_date_dt.strftime('%Y-%m-%d')
        logger.info(f"Computed split date: {split_date}")
    else:
        logger.info(f"Validating provided split date: {split_date}")
        is_valid, validation_report = validate_split_date(
            df_cleaned, date_column, split_date, mode=mode
        )
        
        if not is_valid:
            if mode == "full":
                logger.error("Split date validation failed in FULL mode. Cannot proceed.")
                raise ValueError(f"Invalid split date: {validation_report['errors']}")
            else:
                logger.warning("Split date validation failed in DEV mode.")
                for error in validation_report.get("errors", []):
                    logger.warning(f"  ERROR: {error}")
                for warning in validation_report.get("warnings", []):
                    logger.warning(f"  WARNING: {warning}")
                logger.warning("Proceeding with split despite warnings (DEV mode)")
        
        logger.info(f"Using split date: {split_date}")
    
    # Step 4: Temporal split (BEFORE feature engineering)
    logger.info(f"\nStep 4: Performing temporal split at {split_date}...")
    logger.info("CRITICAL: Split happens BEFORE feature engineering to avoid leakage")
    
    train_df, test_df = perform_temporal_split(
        df_cleaned,
        date_column=date_column,
        split_date=split_date
    )
    
    logger.info(f"Train: {len(train_df):,} rows")
    logger.info(f"Test: {len(test_df):,} rows")
    
    # Validate split correctness
    split_validation = validate_temporal_split_correctness(
        train_df, test_df, date_column, split_date
    )
    
    if not split_validation["is_valid"]:
        if mode == "full":
            logger.error("Temporal split validation failed in FULL mode!")
            logger.error(split_validation["errors"])
            raise ValueError("Invalid temporal split")
        else:
            logger.warning("Temporal split validation failed in DEV mode.")
            for error in split_validation["errors"]:
                logger.warning(f"  ERROR: {error}")
            for warning in split_validation["warnings"]:
                logger.warning(f"  WARNING: {warning}")
            logger.warning("Proceeding despite validation issues (DEV mode)")
    else:
        logger.info("Temporal split validated successfully")
    
    # Store validation results for metadata
    split_validation_result = split_validation
    
    # Step 5: Feature engineering (applied separately to train/test)
    logger.info("\nStep 5: Applying feature engineering...")
    logger.info("  - Training data: Creating features")
    train_engineered = engineer_features(train_df)
    
    logger.info("  - Test data: Creating features")
    test_engineered = engineer_features(test_df)
    
    logger.info(f"Train features: {len(train_engineered.columns)}")
    logger.info(f"Test features: {len(test_engineered.columns)}")
    
    # Step 6: Feature selection
    logger.info("\nStep 6: Selecting features...")
    train_final = select_features(train_engineered, excluded_features)
    test_final = select_features(test_engineered, excluded_features)
    
    logger.info(f"Final train shape: {train_final.shape}")
    logger.info(f"Final test shape: {test_final.shape}")
    
    # Metadata
    # Calculate features created BEFORE feature selection (to get actual count)
    features_created = len(train_engineered.columns) - len(train_df.columns)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "split_date": split_date,
        "date_range": {
            "min": str(date_min) if not pd.isna(date_min) else None,
            "max": str(date_max) if not pd.isna(date_max) else None
        },
        "train_shape": {"rows": len(train_final), "columns": len(train_final.columns)},
        "test_shape": {"rows": len(test_final), "columns": len(test_final.columns)},
        "features_created": features_created,
        "excluded_features": excluded_features,
        "split_validation": split_validation_result
    }
    
    # Step 6: Save engineered features
    if output_path:
        logger.info(f"\nStep 6: Saving engineered features to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save train and test separately
        train_path = output_path.parent / f"{output_path.stem}_train.parquet"
        test_path = output_path.parent / f"{output_path.stem}_test.parquet"
        
        train_final.to_parquet(train_path, index=False)
        test_final.to_parquet(test_path, index=False)
        
        logger.info(f"  - Train: {train_path}")
        logger.info(f"  - Test: {test_path}")
        
        # Save metadata
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"  - Metadata: {metadata_path}")
    
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("="*80)
    
    return train_final, test_final, metadata
