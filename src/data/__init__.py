"""
Data ingestion and processing module.
"""

from .ingestion import (
    load_raw_data,
    ingest_raw_data,
    get_dataset_schema,
    get_basic_stats,
    get_file_metadata,
)
from .validation import (
    validate_schema_consistency,
    identify_target_leakage,
    validate_temporal_split_correctness,
    generate_validation_report,
)
from .splitting import perform_temporal_split
from .cleaning import (
    clean_and_transform,
    drop_columns,
    handle_missing_values,
    detect_outliers,
    ensure_type_consistency,
)
from .sampling import sample_temporally_diverse
from .chunked_processing import process_chunked_cleaning
from .split_utils import compute_split_date, validate_split_date
from .pipeline import process_data_pipeline

__all__ = [
    # Ingestion
    "load_raw_data",
    "ingest_raw_data",
    "get_dataset_schema",
    "get_basic_stats",
    "get_file_metadata",
    # Validation
    "validate_schema_consistency",
    "identify_target_leakage",
    "validate_temporal_split_correctness",
    "generate_validation_report",
    # Splitting
    "perform_temporal_split",
    # Cleaning
    "clean_and_transform",
    "drop_columns",
    "handle_missing_values",
    "detect_outliers",
    "ensure_type_consistency",
    # Sampling
    "sample_temporally_diverse",
    # Chunked processing
    "process_chunked_cleaning",
    # Split utilities
    "compute_split_date",
    "validate_split_date",
    # Pipeline
    "process_data_pipeline",
]
