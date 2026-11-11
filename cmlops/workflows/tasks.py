"""Prefect tasks for the training pipeline."""

from pathlib import Path

import pandas as pd
from prefect import task
from prefect.runtime import flow_run, task_run

from cmlops.config import PipelineConfig, ModelConfig, load_config
from cmlops.data_loading import (
    get_input_columns,
    get_output_columns,
    load_and_validate_data,
)
from cmlops.training.acceptance import assess_accuracy
from cmlops.training.mlflow_utils import (
    RunLineage,
    capture_lineage,
    log_experiment_summary,
    log_training_result,
    log_verification_results,
    register_run_model,
    setup_mlflow,
)
from cmlops.training.splitter import TrainTestData, prepare_train_test_split
from cmlops.training.trainer import TrainingResult, train_single_model
from cmlops.verification.base import VerificationReport
from cmlops.verification.verify import run_full_verification


@task(name="load_config", retries=0)
def task_load_config(config_path: str | Path | None = None) -> PipelineConfig:
    """Load and validate configuration."""
    if config_path:
        return load_config(config_path)
    return load_config()


@task(name="load_data", retries=1, retry_delay_seconds=5)
def task_load_data(config: PipelineConfig) -> pd.DataFrame:
    """Load and validate dataset."""
    return load_and_validate_data(config=config)


@task(name="capture_lineage")
def task_capture_lineage(
    df: pd.DataFrame,
    config: PipelineConfig,
    config_path: str | Path | None = None,
    random_state: int = 42,
) -> RunLineage:
    """Identify the data version, configuration and code version of this
    pipeline execution, to be logged on each of its MLflow runs.

    Fails if the dataset file is tracked by DVC and differs from the recorded
    version.
    """
    return capture_lineage(
        df=df,
        config=config,
        config_path=config_path,
        random_state=random_state,
        flow_run_id=str(flow_run.id) if flow_run.id else None,
    )


@task(name="prepare_and_split_data")
def task_prepare_and_split_data(
    df: pd.DataFrame,
    config: PipelineConfig,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, TrainTestData]]:
    """Project the dataframe into features (X) / targets (y) and build the
    per-output train/test splits.

    Returns ``(X, y, splits)``: the full feature/target frames (used as-is by
    downstream verification) plus the per-output ``TrainTestData`` splits used
    for training.

    When the split method is "grouped", group labels are derived from the
    input columns matched by the configured ``group_by`` patterns, so that all
    rows of a group land exclusively in either train or test.
    """
    X = df[get_input_columns(config)]
    y = df[get_output_columns(config)]
    splits = prepare_train_test_split(X, y, config, random_state)
    return X, y, splits


@task(name="setup_mlflow")
def task_setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """Initialize MLflow tracking."""
    return setup_mlflow(tracking_uri, experiment_name)


def _train_task_run_name() -> str:
    """Generate descriptive task run name for training tasks."""
    params = task_run.parameters
    model_config = params["model_config"]
    data = params["data"]
    model_name = model_config.label or model_config.type
    return f"train_{model_name}_{data.output_name}"


@task(
    name="train_model",
    task_run_name=_train_task_run_name,
    retries=2,
    retry_delay_seconds=10,
    tags=["training"],
)
def task_train_model(
    model_config: ModelConfig,
    data: TrainTestData,
    config: PipelineConfig,
    lineage: RunLineage,
    random_state: int = 42,
    n_jobs_override: int | None = None,
) -> tuple[TrainingResult, str]:
    """Train a single model and log it to MLflow immediately.

    Logging happens here (rather than in a later, sequential step) so each
    model's run appears in MLflow as soon as that model finishes training,
    in parallel with the others. Returns ``(result, run_id)``.

    ``n_jobs_override`` lets the flow size CV-search parallelism against a
    global CPU budget instead of the per-model config value (see
    ``workflows.flows``).
    """
    result = train_single_model(
        model_config=model_config,
        data=data,
        global_optimization=config.model_building.optimization,
        random_state=random_state,
        n_jobs_override=n_jobs_override,
    )
    run_id = log_training_result(
        result=result,
        config=config,
        X_sample=data.X_train.iloc[:100],
        lineage=lineage,
    )
    return result, run_id


@task(name="log_experiment_summary")
def task_log_summary(
    results: list[TrainingResult],
    config: PipelineConfig,
    lineage: RunLineage,
) -> str:
    """Log experiment summary to MLflow."""
    return log_experiment_summary(results, config, lineage)


@task(name="verify_model", tags=["verification"])
def task_verify_model(
    result: TrainingResult,
    X: pd.DataFrame,
    y: pd.DataFrame,
    config: PipelineConfig,
    run_id: str,
    random_state: int = 42,
) -> VerificationReport:
    """Run every enabled verification check on a winner and log to its MLflow run."""
    report = run_full_verification(
        result=result,
        X=X,
        y=y,
        config=config,
        random_state=random_state,
    )
    log_verification_results(run_id=run_id, report=report)
    return report


@task(name="register_winner")
def task_register_winner(
    result: TrainingResult,
    run_id: str,
    report: VerificationReport,
    config: PipelineConfig,
) -> str | None:
    """Register the model under the winner's run iff it meets the accuracy
    thresholds and all verification tests passed.

    Returns the registered model URI, or ``None`` if registration was skipped.
    """
    accuracy = assess_accuracy(result.test_metrics, result.output_name, config)
    if not (accuracy.passed and report.all_passed):
        return None
    model_name = config.registered_model_name(result.output_name, result.model_label)
    return register_run_model(run_id=run_id, model_name=model_name)
