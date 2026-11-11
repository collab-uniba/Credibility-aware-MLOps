"""Configuration models for the pipeline.

This module defines Pydantic models for validating the YAML configuration file.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# Project Configuration Models
# =============================================================================


class ProjectConfig(BaseModel):
    """Identity of the project the pipeline is applied to.

    The name is the default MLflow experiment name and the prefix of the
    registered model names.
    """

    name: str


# =============================================================================
# Dataset Configuration Models
# =============================================================================


class FormatMetadata(BaseModel):
    """Metadata about the file format."""

    delimiter: str = ";"
    decimal_separator: str = "."
    header_lines: int = 1
    encoding: str = "UTF-8"


class InputField(BaseModel):
    """Input field specification."""

    name: str
    data_type: Literal["float", "int"]
    range: tuple[float, float]
    units: str
    missing_values_allowed: bool = False

    @field_validator("range", mode="before")
    @classmethod
    def convert_range(cls, v: list[float] | tuple[float, float]) -> tuple[float, float]:
        """Convert list to tuple for range."""
        if isinstance(v, list) and len(v) == 2:
            return (v[0], v[1])
        return v  # type: ignore[return-value]


class OutputField(BaseModel):
    """Output field specification."""

    name: str
    data_type: Literal["float", "int"]
    range: tuple[float, float]
    units: str
    missing_values_allowed: bool = False

    @field_validator("range", mode="before")
    @classmethod
    def convert_range(cls, v: list[float] | tuple[float, float]) -> tuple[float, float]:
        """Convert list to tuple for range."""
        if isinstance(v, list) and len(v) == 2:
            return (v[0], v[1])
        return v  # type: ignore[return-value]


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str
    path: Path
    source: str
    provenance: str
    version: str
    file_format: str
    format_metadata: FormatMetadata
    inputs: list[InputField]
    outputs: list[OutputField]
    duplicate_rows: bool = False


# =============================================================================
# Gate Thresholds Configuration Models
# =============================================================================

MetricType = Literal[
    "MaximumAbsoluteError",
    "MeanRelativeError",
    "MaximumRelativeErrorRatio",
    "StandardDeviation",
    "RMSE",
    "PercentageRMSE",
    "MaxStandardDeviation"
]


class ThresholdCriterion(BaseModel):
    """A single threshold criterion for validation/verification."""

    field: str
    metric: MetricType
    value: float


class GateThresholds(BaseModel):
    """Gate thresholds for validation."""

    validation: list[ThresholdCriterion]


# =============================================================================
# Verification Configuration Models
# =============================================================================


class VerificationTestEntry(BaseModel):
    """A single verification test, self-contained with thresholds and parameters.

    Mirrors the shape of ``ModelConfig``: each entry's ``type`` is resolved
    against the verification check registry, and ``parameters`` is an
    open-ended dict so future tests can add new fields without schema changes.
    """

    type: str
    thresholds: list[ThresholdCriterion] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class VerificationConfig(BaseModel):
    """Verification configuration: the list of tests to run after training."""

    test_list: list[VerificationTestEntry] = Field(default_factory=list)


# =============================================================================
# Model Building Configuration Models
# =============================================================================


class TrainTestSplitConfig(BaseModel):
    """Train/test split configuration."""

    train_fraction: float = Field(gt=0, lt=1)
    method: Literal["random", "grouped"] = "random"
    group_by: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_group_by_when_grouped(self) -> "TrainTestSplitConfig":
        """Require the grouping columns when the split method is "grouped".

        ``group_by`` lists input field names or glob patterns (e.g. ``shape_*``);
        rows sharing the same values in the matching columns form a group.
        """
        if self.method == "grouped" and not self.group_by:
            raise ValueError('group_by is required when method is "grouped"')
        return self


class OptimizationConfig(BaseModel):
    """Optimization configuration for hyperparameter tuning."""

    optimization_method: Literal["GridSearch", "RandomizedSearchCV"] = "GridSearch"
    n_jobs: int = -1
    cv: int = 5


class ModelConfig(BaseModel):
    """Individual model configuration."""

    type: Literal[
        "RandomForestRegressor",
        "XGBoostRegressor",
        "SupportVectorRegressor",
        "LinearRegressor",
        "MLPRegressor",
        "PLSRegression",
        "CatBoostRegressor",
        "GaussianProcessRegressor",
    ]
    label: str | None = None
    hyperparameters: dict
    optimization: OptimizationConfig | None = None

    @model_validator(mode="after")
    def set_default_label(self) -> "ModelConfig":
        """Set default label to model type if not provided."""
        if self.label is None:
            self.label = self.type
        return self


class ModelBuildingConfig(BaseModel):
    """Model building configuration."""

    train_test_split: TrainTestSplitConfig
    optimization: OptimizationConfig
    model_list: list[ModelConfig]


# =============================================================================
# Root Configuration Model
# =============================================================================


class PipelineConfig(BaseModel):
    """Root configuration for the pipeline."""

    project: ProjectConfig
    dataset: DatasetConfig
    gate_thresholds: GateThresholds
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    model_building: ModelBuildingConfig

    def registered_model_name(self, output_name: str, model_label: str) -> str:
        """Name under which a model is registered in the MLflow Model Registry."""
        return f"{self.project.name}_{output_name}_{model_label}"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated PipelineConfig instance.

        Raises:
            yaml.YAMLError: If the YAML is invalid.
            pydantic.ValidationError: If the configuration doesn't match the schema.
        """
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


# Project root directory (parent of the cmlops package)
PROJECT_ROOT = Path(__file__).parent.parent


# Configuration loaded when none is given: the case study of the paper this
# repository accompanies (the BoneStrengthML surrogate model).
DEFAULT_CONFIG_PATH = Path("config/BoneStrengthML.yml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    """Load and validate the pipeline configuration.

    Args:
        path: Path to the configuration file. Defaults to DEFAULT_CONFIG_PATH.
              Relative paths are resolved from the project root directory.

    Returns:
        Validated configuration object.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return PipelineConfig.from_yaml(path)
