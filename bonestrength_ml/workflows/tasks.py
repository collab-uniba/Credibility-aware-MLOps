"""Prefect tasks for BoneStrengthML training pipeline."""

from pathlib import Path

import pandas as pd
from prefect import task
from prefect.runtime import task_run

from bonestrength_ml.config import BoneStrengthMLConfig, ModelConfig, load_config
from bonestrength_ml.data_loading import (
    get_input_columns,
    get_output_columns,
    load_and_validate_data,
)
from bonestrength_ml.training.mlflow_utils import (
    log_experiment_summary,
    log_training_result,
    log_verification_results,
    register_run_model,
    setup_mlflow,
)
from bonestrength_ml.training.splitter import TrainTestData, prepare_train_test_split
from bonestrength_ml.training.trainer import TrainingResult, train_single_model
from bonestrength_ml.verification.base import VerificationReport
from bonestrength_ml.verification.verify import run_full_verification


@task(name="load_config", retries=0)
def task_load_config(config_path: str | Path | None = None) -> BoneStrengthMLConfig:
    """Load and validate configuration."""
    if config_path:
        return load_config(config_path)
    return load_config()


@task(name="load_data", retries=1, retry_delay_seconds=5)
def task_load_data(config: BoneStrengthMLConfig) -> pd.DataFrame:
    """Load and validate dataset."""
    return load_and_validate_data(config=config)


@task(name="prepare_and_split_data")
def task_prepare_and_split_data(
    df: pd.DataFrame,
    config: BoneStrengthMLConfig,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, TrainTestData]]:
    """Project the dataframe into features (X) / targets (y) and build the
    per-output train/test splits.

    Returns ``(X, y, splits)``: the full feature/target frames (used as-is by
    downstream verification) plus the per-output ``TrainTestData`` splits used
    for training.

    When the split method is "grouped", patient group labels are derived from
    the principal component columns (PC_*) so that all rows for a patient land
    exclusively in either train or test.
    """
    X = df[get_input_columns(config)]
    y = df[get_output_columns(config)]
    splits = prepare_train_test_split(X, y, config, random_state)
    return X, y, splits


@task(name="setup_mlflow")
def task_setup_mlflow(
    tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str = "BoneStrengthML",
) -> str:
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
    config: BoneStrengthMLConfig,
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
    )
    return result, run_id


@task(name="log_experiment_summary")
def task_log_summary(
    results: list[TrainingResult],
    config: BoneStrengthMLConfig,
) -> str:
    """Log experiment summary to MLflow."""
    return log_experiment_summary(results, config)


@task(name="verify_model", tags=["verification"])
def task_verify_model(
    result: TrainingResult,
    X: pd.DataFrame,
    y: pd.DataFrame,
    config: BoneStrengthMLConfig,
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
) -> str | None:
    """Register the model under the winner's run iff all verification tests passed.

    Returns the registered model URI, or ``None`` if registration was skipped.
    """
    if not report.all_passed:
        return None
    model_name = f"BoneStrengthML_{result.output_name}_{result.model_label}"
    return register_run_model(run_id=run_id, model_name=model_name)
