"""
Feature engineering - explicit implementation.

This module contains a single function to create engineered features.
All features are implemented explicitly for clarity and maintainability.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features explicitly.

    This function creates the following features:
    1. funded_to_loan_ratio: Ratio of funded amount to loan amount
    2. installment_to_income_ratio: Monthly installment relative to annual income
    3. fico_midpoint: Midpoint of FICO score range
    4. fico_range: Width of FICO score range
    5. credit_history_years: Years of credit history at loan issue
    6. issue_year: Year when loan was issued
    7. issue_month: Month when loan was issued

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with cleaned data

    Returns
    -------
    pd.DataFrame with engineered features added
    """
    df_new = df.copy()

    logger.info("Creating engineered features...")

    # 1. funded_to_loan_ratio
    # Ratio of funded amount to requested loan amount
    # Indicates if full amount was funded
    if "funded_amnt" in df.columns and "loan_amnt" in df.columns:
        df_new["funded_to_loan_ratio"] = np.where(
            df["loan_amnt"] != 0, df["funded_amnt"] / df["loan_amnt"], np.nan
        )
        logger.debug("Created: funded_to_loan_ratio")
    else:
        logger.warning("Cannot create funded_to_loan_ratio: required columns missing")

    # 2. installment_to_income_ratio
    # Monthly installment payment relative to annual income
    # Higher ratio indicates higher debt burden
    if "installment" in df.columns and "annual_inc" in df.columns:
        df_new["installment_to_income_ratio"] = np.where(
            df["annual_inc"] != 0, df["installment"] / df["annual_inc"], np.nan
        )
        logger.debug("Created: installment_to_income_ratio")
    else:
        logger.warning("Cannot create installment_to_income_ratio: required columns missing")

    # 3. fico_midpoint
    # Midpoint of FICO score range (average of low and high)
    # Single numeric representation of credit score
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df_new["fico_midpoint"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        logger.debug("Created: fico_midpoint")
    else:
        logger.warning("Cannot create fico_midpoint: required columns missing")

    # 4. fico_range
    # Width of FICO score range (high - low)
    # Indicates uncertainty/variability in credit score
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df_new["fico_range"] = df["fico_range_high"] - df["fico_range_low"]
        logger.debug("Created: fico_range")
    else:
        logger.warning("Cannot create fico_range: required columns missing")

    # 5. credit_history_years
    # Years of credit history at time of loan issue
    # Longer history may indicate more stable borrowers
    if "earliest_cr_line" in df.columns and "issue_d" in df.columns:
        try:
            earliest_cr = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
            issue_date = pd.to_datetime(df["issue_d"], errors="coerce")
            # Calculate difference in years
            df_new["credit_history_years"] = (issue_date - earliest_cr).dt.days / 365.25
            logger.debug("Created: credit_history_years")
        except Exception as e:
            logger.warning(f"Cannot create credit_history_years: {e}")
    else:
        logger.warning("Cannot create credit_history_years: required columns missing")

    # 6. issue_year
    # Year when loan was issued
    # Captures temporal trends and economic conditions
    if "issue_d" in df.columns:
        try:
            issue_date = pd.to_datetime(df["issue_d"], errors="coerce")
            df_new["issue_year"] = issue_date.dt.year
            logger.debug("Created: issue_year")
        except Exception as e:
            logger.warning(f"Cannot create issue_year: {e}")
    else:
        logger.warning("Cannot create issue_year: required column missing")

    # 7. issue_month
    # Month when loan was issued (1-12)
    # Captures seasonal patterns
    if "issue_d" in df.columns:
        try:
            issue_date = pd.to_datetime(df["issue_d"], errors="coerce")
            df_new["issue_month"] = issue_date.dt.month
            logger.debug("Created: issue_month")
        except Exception as e:
            logger.warning(f"Cannot create issue_month: {e}")
    else:
        logger.warning("Cannot create issue_month: required column missing")

    logger.info(f"Feature engineering complete. Shape: {df.shape} -> {df_new.shape}")

    return df_new
