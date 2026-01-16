"""
Dataset configuration and constants.

This module contains configuration settings for the Lending Club dataset,
including column mappings, validation rules, and data paths.
"""

# Dataset paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

# Target variable
TARGET_COLUMN = "loan_status"  # Will be converted to binary default indicator

# Expected target values (Lending Club specific)
TARGET_VALUES = {
    "Fully Paid": 0,  # No default
    "Charged Off": 1,  # Default
    "Current": None,  # Ongoing loan, exclude from training
    "Late (31-120 days)": None,  # Ongoing, exclude
    "Late (16-30 days)": None,  # Ongoing, exclude
    "In Grace Period": None,  # Ongoing, exclude
    "Default": 1,  # Default
}

# Known leakage columns (to be validated and confirmed)
KNOWN_LEAKAGE_COLUMNS = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "last_credit_pull_d",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
]

# Columns to review for potential leakage
REVIEW_COLUMNS = [
    "issue_d",
    "loan_status",
    "pymnt_plan",
]

# Date column for temporal split
DATE_COLUMN = "issue_d"  # Loan issue date

# Temporal split configuration
# Will be set based on data analysis
TRAIN_SPLIT_DATE = None  # To be determined
TEST_SPLIT_DATE = None   # To be determined

# Rejected loans indicator
# Column that indicates if loan was rejected (if available)
REJECTED_LOAN_INDICATOR = None  # To be identified from dataset


