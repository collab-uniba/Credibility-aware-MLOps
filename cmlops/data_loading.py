"""Data loading and validation module for the pipeline.

This module provides functionality to:
- Load raw data from CSV files based on configuration
- Validate data using Pandera schemas dynamically generated from config
- Identify the version of a dataset file and check it against the one
  recorded by DVC
"""

import hashlib
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import yaml
from pandera.pandas import Column, DataFrameSchema

from cmlops.config import (
    PROJECT_ROOT,
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


class DataVersionError(Exception):
    """Raised when a dataset file differs from the version recorded by DVC."""


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


def resolve_dataset_path(
    config: PipelineConfig,
    data_path: str | Path | None = None,
) -> Path:
    """Absolute path of the dataset file a run uses.

    Args:
        config: Pipeline configuration.
        data_path: Override path for data file. If None, uses path from config.
            Relative paths are resolved from the config directory.
    """
    file_path = Path(data_path) if data_path is not None else Path(config.dataset.path)
    if not file_path.is_absolute():
        file_path = (PROJECT_ROOT / "config" / file_path).resolve()
    return file_path


def file_md5(path: str | Path) -> str:
    """MD5 of a file's content, the hash DVC identifies a data version by."""
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_dvc_pointer(path: str | Path, md5: str) -> None:
    """Check a dataset file against the data version recorded by DVC.

    A file tracked with ``dvc add`` has a pointer file ``<path>.dvc`` next to
    it, committed to git, that holds the MD5 of the tracked version. A file
    that differs from it corresponds to no recorded data version, so evidence
    produced from it could not be traced back to its data. Files without a
    pointer are not tracked by DVC and are not checked.

    Raises:
        DataVersionError: If the file differs from the version in the pointer.
    """
    path = Path(path)
    pointer = path.with_name(path.name + ".dvc")
    if not pointer.exists():
        return

    with pointer.open() as f:
        recorded_md5 = yaml.safe_load(f)["outs"][0]["md5"]
    if recorded_md5 != md5:
        raise DataVersionError(
            f"{path} (md5 {md5}) differs from the data version recorded in "
            f"{pointer.name} (md5 {recorded_md5}). Record it as a new version "
            f"with `dvc add {path}` and commit the pointer, or restore the "
            f"recorded version with `dvc checkout {pointer}`."
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

    format_meta = config.dataset.format_metadata

    file_path = resolve_dataset_path(config, data_path)

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
