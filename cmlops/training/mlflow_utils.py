"""MLflow integration utilities for experiment tracking."""

import json
import math
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from cmlops.config import PipelineConfig
from cmlops.training.trainer import TrainingResult
from cmlops.verification.base import VerificationReport


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

