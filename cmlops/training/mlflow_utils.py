"""MLflow integration utilities for experiment tracking."""

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator

from cmlops.config import PipelineConfig
from cmlops.training.trainer import TrainingResult
from cmlops.verification.base import VerificationReport


@dataclass
class BestRunInfo:
    """Information about the best MLflow run for a model/output combination."""

    model: BaseEstimator
    best_params: dict[str, Any]
    model_type: str
    run_id: str


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """Configure MLflow tracking and select the experiment.

    Args:
        tracking_uri: MLflow tracking URI.
        experiment_name: Name of the MLflow experiment.

    Returns:
        Experiment ID.
    """
    mlflow.set_tracking_uri(tracking_uri)

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_training_result(
    result: TrainingResult,
    config: PipelineConfig,
    X_sample: pd.DataFrame,
    run_name: str | None = None,
) -> str:
    """Log a training result to MLflow as a new run.

    Returns the new run's id. Registration is a separate step (see
    ``register_run_model``) gated on verification outcomes.
    """
    run_name = run_name or f"{result.model_label}_{result.output_name}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        opt_config = config.model_building.optimization
        params = {
            "model_type": result.model_type,
            "model_label": result.model_label,
            "output_name": result.output_name,
            "train_fraction": config.model_building.train_test_split.train_fraction,
            "split_method": config.model_building.train_test_split.method,
            "cv_folds": opt_config.cv,
            "optimization_method": opt_config.optimization_method,
        }
        # Add best hyperparameters with prefix
        for k, v in result.best_params.items():
            params[f"best_{k}"] = v

        mlflow.log_params(params)

        # Log test metrics
        for metric_name, value in result.test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Log train metrics
        for metric_name, value in result.train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)

        # Log model with signature
        signature = infer_signature(X_sample, result.best_estimator.predict(X_sample))

        mlflow.sklearn.log_model(
            result.best_estimator,
            name="model",
            signature=signature,
            input_example=X_sample.iloc[:5],
        )

        # Log CV results if available
        if result.cv_results:
            cv_df = pd.DataFrame(result.cv_results)
            with tempfile.TemporaryDirectory() as tmpdir:
                cv_path = Path(tmpdir) / "cv_results.csv"
                cv_df.to_csv(cv_path, index=False)
                mlflow.log_artifact(str(cv_path), artifact_path="cv_results")

        return run.info.run_id


def register_run_model(run_id: str, model_name: str) -> str:
    """Register the ``model`` artifact of an existing run in the MLflow Model Registry.

    Returns the registered model URI for downstream logging.
    """
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, model_name)
    return model_uri


def log_verification_results(run_id: str, report: VerificationReport) -> None:
    """Attach a verification report to an existing training run.

    Resumes the run by ``run_id`` and writes:
      - metrics ``verification.{test}.{key}`` (one per test metric)
      - params  ``verification.{test}.{key}`` (one per test param)
      - tags    ``verification.{test}`` = "pass"/"fail" and ``verification.status``
      - artifact ``verification/report.json`` (full structured report)

    The function iterates ``report.tests`` generically — adding new check types
    requires no changes here.
    """
    with mlflow.start_run(run_id=run_id):
        for test in report.tests:
            for k, v in test.metrics.items():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    mlflow.log_metric(f"verification.{test.name}.{k}", float(v))
            for k, v in test.params.items():
                mlflow.log_param(f"verification.{test.name}.{k}", v)
            mlflow.set_tag(
                f"verification.{test.name}", "pass" if test.passed else "fail"
            )

        mlflow.set_tag(
            "verification.status", "passed" if report.all_passed else "failed"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            with report_path.open("w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            mlflow.log_artifact(str(report_path), artifact_path="verification")


def log_experiment_summary(
    results: list[TrainingResult],
    config: PipelineConfig,
) -> str:
    """Log a summary run comparing all models.

    Args:
        results: List of all training results.
        config: Configuration used.

    Returns:
        MLflow run ID for summary.
    """
    with mlflow.start_run(run_name="experiment_summary") as run:
        # Create comparison table
        summary_data = []
        for r in results:
            summary_data.append({
                "output": r.output_name,
                "model": r.model_label,
                "test_rmse": r.test_metrics["rmse"],
                "test_r2": r.test_metrics["r2"],
                "test_mae": r.test_metrics["mae"],
                "test_max_ae": r.test_metrics["max_ae"],
            })

        summary_df = pd.DataFrame(summary_data)
        mlflow.log_table(summary_df, artifact_file="model_comparison.json")

        # Log best model per output
        for output_name in summary_df["output"].unique():
            output_results = summary_df[summary_df["output"] == output_name]
            best_idx = output_results["test_rmse"].idxmin()
            best_model = output_results.loc[best_idx, "model"]
            best_rmse = output_results.loc[best_idx, "test_rmse"]

            mlflow.log_param(f"best_model_{output_name}", best_model)
            mlflow.log_metric(f"best_rmse_{output_name}", best_rmse)

        return run.info.run_id


def load_best_run(
    output_name: str,
    experiment_name: str,
    tracking_uri: str = "sqlite:///mlruns.db",
    model_type: str | None = None,
    run_id: str | None = None,
) -> BestRunInfo:
    """Load the best MLflow run for a given output (and optionally model type).

    If ``run_id`` is provided, loads that specific run directly.
    Otherwise, searches for the run with the lowest test_rmse matching
    the given ``output_name`` (and ``model_type`` if specified).

    Args:
        output_name: Target output name.
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking URI.
        model_type: Optional model type filter (e.g. "RandomForestRegressor").
        run_id: Optional specific run ID to load.

    Returns:
        BestRunInfo with loaded model, best hyperparameters, model type, and run ID.

    Raises:
        ValueError: If no matching runs are found or experiment doesn't exist.
    """
    mlflow.set_tracking_uri(tracking_uri)

    if run_id is not None:
        # Load a specific run
        run = mlflow.get_run(run_id)
        loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        params = run.data.params
        best_params = {
            k.removeprefix("best_"): v
            for k, v in params.items()
            if k.startswith("best_")
        }
        return BestRunInfo(
            model=loaded_model,
            best_params=best_params,
            model_type=params.get("model_type", "unknown"),
            run_id=run_id,
        )

    # Search for best run
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. "
            "Run training first with: cmlops train"
        )

    filter_parts = [f"params.output_name = '{output_name}'"]
    if model_type is not None:
        filter_parts.append(f"params.model_type = '{model_type}'")
    filter_string = " and ".join(filter_parts)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["metrics.test_rmse ASC"],
        max_results=1,
    )

    if runs.empty:
        filter_desc = f"output_name='{output_name}'"
        if model_type:
            filter_desc += f", model_type='{model_type}'"
        raise ValueError(
            f"No runs found matching {filter_desc} in experiment '{experiment_name}'. "
            "Run training first with: cmlops train"
        )

    best_run_id = runs.iloc[0]["run_id"]
    loaded_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

    # Extract best_* params
    run = mlflow.get_run(best_run_id)
    params = run.data.params
    best_params = {
        k.removeprefix("best_"): v
        for k, v in params.items()
        if k.startswith("best_")
    }

    return BestRunInfo(
        model=loaded_model,
        best_params=best_params,
        model_type=params.get("model_type", "unknown"),
        run_id=best_run_id,
    )
