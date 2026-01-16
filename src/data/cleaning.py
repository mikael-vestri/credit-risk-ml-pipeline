"""
Data cleaning and transformation module (Step 5).

This module handles:
- Dropping leakage and high-missing columns
- Handling missing values explicitly
- Detecting outliers
- Ensuring type consistency
- Logging all transformations

All transformations are logged for reproducibility and auditability.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def load_columns_to_drop(validation_report_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Load columns to drop from validation report.
    
    Parameters
    ----------
    validation_report_path : Optional[Path]
        Path to validation report JSON
        If None, uses default location
        
    Returns
    -------
    Dict with 'leakage_columns' and 'high_missing_columns' lists
    """
    if validation_report_path is None:
        # Default location
        validation_report_path = Path(__file__).parent.parent.parent / "docs" / "validation_report.json"
    
    if not validation_report_path.exists():
        logger.warning(f"Validation report not found at {validation_report_path}")
        return {"leakage_columns": [], "high_missing_columns": []}
    
    with open(validation_report_path, 'r') as f:
        report = json.load(f)
    
    dropped = report.get("dropped_columns", {})
    return {
        "leakage_columns": dropped.get("leakage_columns", []),
        "high_missing_columns": dropped.get("high_missing_columns", [])
    }


def drop_columns(
    df: pd.DataFrame,
    columns_to_drop: List[str],
    reason: str = "unspecified"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Drop specified columns and log the operation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns_to_drop : List[str]
        List of column names to drop
    reason : str
        Reason for dropping (for logging)
        
    Returns
    -------
    Tuple of (cleaned DataFrame, transformation log)
    """
    # Find columns that actually exist
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    missing_columns = [col for col in columns_to_drop if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Columns not found (will be skipped): {missing_columns}")
    
    if not existing_columns:
        logger.info(f"No columns to drop (reason: {reason})")
        return df.copy(), {
            "operation": "drop_columns",
            "reason": reason,
            "columns_dropped": [],
            "columns_not_found": missing_columns,
            "columns_before": len(df.columns),
            "columns_after": len(df.columns)
        }
    
    df_cleaned = df.drop(columns=existing_columns)
    
    log_entry = {
        "operation": "drop_columns",
        "reason": reason,
        "columns_dropped": existing_columns,
        "columns_not_found": missing_columns,
        "columns_before": len(df.columns),
        "columns_after": len(df_cleaned.columns)
    }
    
    logger.info(f"Dropped {len(existing_columns)} columns (reason: {reason})")
    logger.info(f"  Columns before: {len(df.columns)}, after: {len(df_cleaned.columns)}")
    
    return df_cleaned, log_entry


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "explicit",
    numeric_fill: str = "median",
    categorical_fill: str = "mode"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle missing values explicitly with logging.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    strategy : str
        Strategy: "explicit" (fill with strategy), "drop" (drop rows/cols), "indicator" (add indicator)
    numeric_fill : str
        How to fill numeric missing: "mean", "median", "zero", "drop"
    categorical_fill : str
        How to fill categorical missing: "mode", "unknown", "drop"
        
    Returns
    -------
    Tuple of (cleaned DataFrame, transformation log)
    """
    log_entry = {
        "operation": "handle_missing_values",
        "strategy": strategy,
        "missing_before": {},
        "missing_after": {},
        "transformations_applied": []
    }
    
    # Count missing before
    missing_before = df.isnull().sum()
    missing_before_pct = (missing_before / len(df) * 100).round(2)
    log_entry["missing_before"] = {
        col: {
            "count": int(count),
            "percentage": float(pct)
        }
        for col, count, pct in zip(missing_before.index, missing_before.values, missing_before_pct.values)
        if count > 0
    }
    
    df_cleaned = df.copy()
    
    if strategy == "explicit":
        # Handle numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_cleaned[col].isnull().sum() > 0:
                if numeric_fill == "median":
                    fill_value = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                    log_entry["transformations_applied"].append({
                        "column": col,
                        "type": "numeric",
                        "method": "median",
                        "fill_value": float(fill_value) if not pd.isna(fill_value) else None
                    })
                elif numeric_fill == "mean":
                    fill_value = df_cleaned[col].mean()
                    df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                    log_entry["transformations_applied"].append({
                        "column": col,
                        "type": "numeric",
                        "method": "mean",
                        "fill_value": float(fill_value) if not pd.isna(fill_value) else None
                    })
                elif numeric_fill == "zero":
                    df_cleaned[col] = df_cleaned[col].fillna(0)
                    log_entry["transformations_applied"].append({
                        "column": col,
                        "type": "numeric",
                        "method": "zero"
                    })
        
        # Handle categorical columns
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                if categorical_fill == "mode":
                    mode_value = df_cleaned[col].mode()
                    if len(mode_value) > 0:
                        fill_value = mode_value[0]
                        df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                        log_entry["transformations_applied"].append({
                            "column": col,
                            "type": "categorical",
                            "method": "mode",
                            "fill_value": str(fill_value)
                        })
                elif categorical_fill == "unknown":
                    df_cleaned[col] = df_cleaned[col].fillna("Unknown")
                    log_entry["transformations_applied"].append({
                        "column": col,
                        "type": "categorical",
                        "method": "unknown"
                    })
    
    # Count missing after
    missing_after = df_cleaned.isnull().sum()
    missing_after_pct = (missing_after / len(df_cleaned) * 100).round(2)
    log_entry["missing_after"] = {
        col: {
            "count": int(count),
            "percentage": float(pct)
        }
        for col, count, pct in zip(missing_after.index, missing_after.values, missing_after_pct.values)
        if count > 0
    }
    
    total_missing_before = missing_before.sum()
    total_missing_after = missing_after.sum()
    logger.info(f"Missing values handled: {total_missing_before:,} -> {total_missing_after:,}")
    
    # Log method summary
    if log_entry["transformations_applied"]:
        numeric_fills = [t for t in log_entry["transformations_applied"] if t["type"] == "numeric"]
        categorical_fills = [t for t in log_entry["transformations_applied"] if t["type"] == "categorical"]
        
        if numeric_fills:
            methods_used = {}
            for t in numeric_fills:
                method = t["method"]
                methods_used[method] = methods_used.get(method, 0) + 1
            logger.info(f"  Numeric columns: {', '.join(f'{k}({v} cols)' for k, v in methods_used.items())}")
        
        if categorical_fills:
            methods_used = {}
            for t in categorical_fills:
                method = t["method"]
                methods_used[method] = methods_used.get(method, 0) + 1
            logger.info(f"  Categorical columns: {', '.join(f'{k}({v} cols)' for k, v in methods_used.items())}")
    
    return df_cleaned, log_entry


def detect_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Detect outliers in numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    method : str
        Method: "iqr" (Interquartile Range) or "zscore"
    columns : Optional[List[str]]
        Specific columns to check. If None, checks all numeric columns.
        
    Returns
    -------
    Dict with outlier detection results
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in [np.number, 'int64', 'float64']]
    
    outlier_report = {
        "method": method,
        "columns_checked": numeric_cols,
        "outliers": {}
    }
    
    for col in numeric_cols:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            outlier_report["outliers"][col] = {
                "count": int(outlier_count),
                "percentage": float(outlier_count / len(df) * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "min_value": float(df[col].min()),
                "max_value": float(df[col].max())
            }
        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > 3]
            outlier_count = len(outliers)
            
            outlier_report["outliers"][col] = {
                "count": int(outlier_count),
                "percentage": float(outlier_count / len(df) * 100),
                "threshold": 3.0
            }
    
    # Summary
    total_outlier_rows = len(df)
    for col_info in outlier_report["outliers"].values():
        # This is approximate - rows can have outliers in multiple columns
        pass
    
    logger.info(f"Outlier detection complete ({method} method)")
    logger.info(f"  Columns checked: {len(numeric_cols)}")
    
    return outlier_report


def ensure_type_consistency(
    df: pd.DataFrame,
    date_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ensure data type consistency across the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_columns : Optional[List[str]]
        List of columns that should be datetime
        
    Returns
    -------
    Tuple of (cleaned DataFrame, transformation log)
    """
    log_entry = {
        "operation": "ensure_type_consistency",
        "type_changes": []
    }
    
    df_cleaned = df.copy()
    
    # Convert date columns
    if date_columns:
        for col in date_columns:
            if col in df_cleaned.columns:
                original_type = str(df_cleaned[col].dtype)
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                    log_entry["type_changes"].append({
                        "column": col,
                        "from": original_type,
                        "to": "datetime64[ns]"
                    })
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {e}")
    
    # Ensure ID columns and string-like numeric columns are kept as strings (for Parquet compatibility)
    # PyArrow has issues converting object columns with numeric strings to int64
    id_like_cols = [col for col in df_cleaned.columns if 'id' in col.lower()]
    
    # Convert object columns that should stay as strings
    object_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in object_cols:
        # If it's an ID-like column, ensure it's string type
        if col in id_like_cols:
            try:
                # Convert to string type (pandas nullable string, available in pandas >= 1.0)
                # If 'string' dtype not available, keep as object but ensure it's treated as string
                if hasattr(pd, 'StringDtype'):
                    df_cleaned[col] = df_cleaned[col].astype('string')
                    log_entry["type_changes"].append({
                        "column": col,
                        "from": "object",
                        "to": "string",
                        "reason": "ID column - keep as string for Parquet compatibility"
                    })
                else:
                    # For older pandas, ensure it's treated as string by converting to str
                    df_cleaned[col] = df_cleaned[col].astype(str)
                    log_entry["type_changes"].append({
                        "column": col,
                        "from": "object",
                        "to": "str",
                        "reason": "ID column - keep as string for Parquet compatibility"
                    })
            except Exception as e:
                logger.warning(f"Could not convert {col} to string: {e}")
    
    # Ensure numeric columns are properly typed
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Convert to float if has NaN, otherwise keep as int
        if df_cleaned[col].isnull().any():
            if df_cleaned[col].dtype != 'float64':
                original_type = str(df_cleaned[col].dtype)
                df_cleaned[col] = df_cleaned[col].astype('float64')
                log_entry["type_changes"].append({
                    "column": col,
                    "from": original_type,
                    "to": "float64",
                    "reason": "contains NaN values"
                })
    
    logger.info(f"Type consistency ensured: {len(log_entry['type_changes'])} changes made")
    
    # Log what changes were made
    if log_entry["type_changes"]:
        for change in log_entry["type_changes"]:
            reason = change.get("reason", "")
            logger.info(f"  {change['column']}: {change['from']} -> {change['to']}" + 
                       (f" ({reason})" if reason else ""))
    
    return df_cleaned, log_entry


def clean_and_transform(
    df: pd.DataFrame,
    validation_report_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
    mode: str = "dev"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete data cleaning and transformation pipeline.
    
    This is the main entry point for Step 5. It:
    1. Drops leakage and high-missing columns
    2. Handles missing values
    3. Detects outliers
    4. Ensures type consistency
    5. Logs all transformations
    6. Saves cleaned data
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame
    validation_report_path : Optional[Path]
        Path to validation report (for columns to drop)
    save_path : Optional[Path]
        Path to save cleaned dataset
    mode : str
        Execution mode: 'dev' or 'full' (for logging purposes)
        
    Returns
    -------
    Tuple of (cleaned DataFrame, transformation log)
    """
    logger.info("="*80)
    logger.info(f"DATA CLEANING & TRANSFORMATION - STEP 5 [{mode.upper()} MODE]")
    logger.info("="*80)
    
    transformation_log = {
        "timestamp": datetime.now().isoformat(),
        "initial_shape": {"rows": len(df), "columns": len(df.columns)},
        "transformations": []
    }
    
    df_cleaned = df.copy()
    
    # Step 1: Drop leakage and high-missing columns
    logger.info("\nStep 1: Dropping leakage and high-missing columns...")
    columns_to_drop = load_columns_to_drop(validation_report_path)
    
    all_columns_to_drop = (
        columns_to_drop["leakage_columns"] + 
        columns_to_drop["high_missing_columns"]
    )
    
    df_cleaned, drop_log = drop_columns(
        df_cleaned,
        all_columns_to_drop,
        reason="leakage and high missing values"
    )
    transformation_log["transformations"].append(drop_log)
    
    # Step 2: Handle missing values
    logger.info("\nStep 2: Handling missing values...")
    df_cleaned, missing_log = handle_missing_values(
        df_cleaned,
        strategy="explicit",
        numeric_fill="median",
        categorical_fill="mode"
    )
    transformation_log["transformations"].append(missing_log)
    
    # Step 3: Detect outliers (logging only, not removing)
    logger.info("\nStep 3: Detecting outliers...")
    outlier_report = detect_outliers(df_cleaned, method="iqr")
    transformation_log["outlier_detection"] = outlier_report
    
    # Log outlier summary
    if outlier_report["outliers"]:
        total_outlier_values = sum(info["count"] for info in outlier_report["outliers"].values())
        logger.info(f"  Total outlier values detected: {total_outlier_values:,}")
        logger.info(f"  Top 5 columns with most outliers:")
        sorted_outliers = sorted(
            outlier_report["outliers"].items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:5]
        for col, info in sorted_outliers:
            logger.info(f"    {col}: {info['count']:,} outliers ({info['percentage']:.2f}%)")
    else:
        logger.info("  No outliers detected")
    
    # Step 4: Ensure type consistency
    logger.info("\nStep 4: Ensuring type consistency...")
    from config import dataset_config
    date_cols = [dataset_config.DATE_COLUMN] if dataset_config.DATE_COLUMN in df_cleaned.columns else []
    df_cleaned, type_log = ensure_type_consistency(df_cleaned, date_columns=date_cols)
    transformation_log["transformations"].append(type_log)
    
    # Final summary
    transformation_log["final_shape"] = {"rows": len(df_cleaned), "columns": len(df_cleaned.columns)}
    
    logger.info("\n" + "="*80)
    logger.info("CLEANING COMPLETE")
    logger.info("="*80)
    logger.info(f"Initial shape: {transformation_log['initial_shape']}")
    logger.info(f"Final shape: {transformation_log['final_shape']}")
    logger.info("="*80)
    
    # Save cleaned data
    if save_path:
        logger.info(f"\nSaving cleaned data to: {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet for efficiency, CSV as backup
        if str(save_path).endswith('.parquet'):
            df_cleaned.to_parquet(save_path, index=False)
        elif str(save_path).endswith('.csv'):
            df_cleaned.to_csv(save_path, index=False)
        else:
            # Default to parquet
            parquet_path = save_path.with_suffix('.parquet')
            df_cleaned.to_parquet(parquet_path, index=False)
            logger.info(f"Saved as: {parquet_path}")
        
        # Save transformation log
        log_path = save_path.parent / f"{save_path.stem}_transformation_log.json"
        with open(log_path, 'w') as f:
            json.dump(transformation_log, f, indent=2, default=str)
        logger.info(f"Transformation log saved to: {log_path}")
    
    return df_cleaned, transformation_log

