"""Data loading and validation module for the pipeline.

This module provides functionality to:
- Load raw data from CSV files based on configuration
- Validate data using Pandera schemas dynamically generated from config
"""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

from cmlops.config import (
    PipelineConfig,
    DatasetConfig,
    InputField,
    OutputField,
    load_config,
)


class DataValidationError(Exception):
    """Raised when data validation fails."""

    def __init__(self, message: str, failure_cases: pd.DataFrame | None = None):
        super().__init__(message)
        self.failure_cases = failure_cases


def _build_column_schema(
    field: InputField | OutputField,
) -> Column:
    """Build a Pandera Column schema from a field specification.

    Args:
        field: Input or output field specification from config.

    Returns:
        Pandera Column with appropriate checks.
    """
    # Determine pandas dtype based on config data_type
    if field.data_type == "float":
        dtype = pa.Float64
    else:  # int
        dtype = pa.Int64

    # Build checks list
    checks = []

    # Range check
    min_val, max_val = field.range
    checks.append(pa.Check.in_range(min_val, max_val))

    return Column(
        dtype=dtype,
        nullable=field.missing_values_allowed,
        checks=checks,
        coerce=True,  # Attempt type coercion during validation
    )


def build_schema_from_config(dataset_config: DatasetConfig) -> DataFrameSchema:
    """Build a Pandera DataFrameSchema from dataset configuration.

    Args:
        dataset_config: Dataset configuration containing field specifications.

    Returns:
        DataFrameSchema for validating the dataset.
    """
    columns: dict[str, Column] = {}

    # Add input columns
    for field in dataset_config.inputs:
        columns[field.name] = _build_column_schema(field)

    # Add output columns
    for field in dataset_config.outputs:
        columns[field.name] = _build_column_schema(field)

    # Build schema with strict=True to catch unexpected columns
    # Pandera skips the uniqueness check when `unique` is empty/None, so to
    # reject duplicate rows all columns must be listed as jointly unique.
    return DataFrameSchema(
        columns=columns,
        strict=True,  # Fail if there are columns not in schema
        unique=None if dataset_config.duplicate_rows else list(columns),
        coerce=True,
    )


def load_raw_data(
    config: PipelineConfig | None = None,
    data_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load raw data from CSV file based on configuration.

    Args:
        config: Pipeline configuration. If None, loads from default path.
        data_path: Override path for data file. If None, uses path from config.

    Returns:
        DataFrame containing the raw data.

    Raises:
        FileNotFoundError: If the data file doesn't exist.
    """
    if config is None:
        config = load_config()

    dataset_config = config.dataset
    format_meta = dataset_config.format_metadata

    # Determine file path
    if data_path is not None:
        file_path = Path(data_path)
    else:
        file_path = Path(dataset_config.path)

    # Handle relative paths (relative to config directory)
    if not file_path.is_absolute():
        # Resolve relative to project root
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "config"
        file_path = (config_dir / file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load data with format metadata
    df = pd.read_csv(
        file_path,
        delimiter=format_meta.delimiter,
        decimal=format_meta.decimal_separator,
        header=0 if format_meta.header_lines > 0 else None,
        encoding=format_meta.encoding,
    )

    return df


def validate_data(
    df: pd.DataFrame,
    config: PipelineConfig | None = None,
    schema: DataFrameSchema | None = None,
) -> pd.DataFrame:
    """Validate a DataFrame against the schema derived from configuration.

    Args:
        df: DataFrame to validate.
        config: Pipeline configuration. If None, loads from default path.
        schema: Pre-built schema. If None, builds from config.

    Returns:
        Validated DataFrame (potentially with coerced types).

    Raises:
        DataValidationError: If validation fails with details about failures.
    """
    if schema is None:
        if config is None:
            config = load_config()
        schema = build_schema_from_config(config.dataset)

    try:
        validated_df = schema.validate(df, lazy=True)
        return validated_df
    except pa.errors.SchemaErrors as e:
        raise DataValidationError(
            message=f"Data validation failed with {len(e.failure_cases)} errors:\n{e.message}",
            failure_cases=e.failure_cases,
        ) from e


def load_and_validate_data(
    config: PipelineConfig | None = None,
    data_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and validate raw data in a single step.

    This is the main entry point for loading data in the pipeline.

    Args:
        config: Pipeline configuration. If None, loads from default path.
        data_path: Override path for data file. If None, uses path from config.

    Returns:
        Validated DataFrame with correct types.

    Raises:
        FileNotFoundError: If the data file doesn't exist.
        DataValidationError: If validation fails.

    Example:
        >>> from cmlops.data_loading import load_and_validate_data
        >>> df = load_and_validate_data()
        >>> df.shape
        (1000000, 97)
    """
    if config is None:
        config = load_config()

    df = load_raw_data(config=config, data_path=data_path)
    validated_df = validate_data(df, config=config)

    return validated_df


def get_input_columns(config: PipelineConfig | None = None) -> list[str]:
    """Get list of input column names from configuration.

    Args:
        config: Pipeline configuration. If None, loads from default path.

    Returns:
        List of input column names.
    """
    if config is None:
        config = load_config()
    return [field.name for field in config.dataset.inputs]


def get_output_columns(config: PipelineConfig | None = None) -> list[str]:
    """Get list of output column names from configuration.

    Args:
        config: Pipeline configuration. If None, loads from default path.

    Returns:
        List of output column names.
    """
    if config is None:
        config = load_config()
    return [field.name for field in config.dataset.outputs]
