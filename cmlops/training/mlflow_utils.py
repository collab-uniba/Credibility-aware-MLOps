"""MLflow integration utilities for experiment tracking."""

import json
import math
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.data.dataset import Dataset
from mlflow.models import infer_signature

from cmlops.config import PROJECT_ROOT, PipelineConfig, resolve_config_path
from cmlops.data_loading import check_dvc_pointer, file_md5, resolve_dataset_path
from cmlops.training.acceptance import AccuracyAssessment, assess_accuracy
from cmlops.training.trainer import TrainingResult
from cmlops.verification.base import VerificationReport


@dataclass(frozen=True)
class RunLineage:
    """What the runs of one pipeline execution were produced from.

    Captured once per execution and logged on each of its MLflow runs, so the
    evidence a run holds stays traceable to the data version, configuration,
    code version and seed behind it.
    """

    dataset: Dataset
    dataset_version: str
    config_path: Path
    random_state: int
    git_commit: str | None = None
    git_dirty: bool | None = None
    flow_run_id: str | None = None


def _relative_to_project(path: Path) -> str:
    """Path relative to the project root when inside it, absolute otherwise."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _git_state() -> tuple[str | None, bool | None]:
    """Commit of the project checkout and whether tracked files are modified.

    MLflow's own lookup starts from the launched script; for the ``cmlops``
    command that is a file inside the git-ignored virtual environment, for
    which it finds no commit. Returns ``(None, None)`` when the code does not
    run from a git checkout.
    """
    try:
        from git import Repo

        repo = Repo(PROJECT_ROOT)
        return repo.head.commit.hexsha, repo.is_dirty()
    except Exception:
        return None, None


def capture_lineage(
    df: pd.DataFrame,
    config: PipelineConfig,
    config_path: str | Path | None,
    random_state: int,
    flow_run_id: str | None = None,
) -> RunLineage:
    """Identify the data, configuration and code a pipeline execution uses.

    The dataset is identified by the MD5 of its file, the same hash DVC records
    in the file's ``.dvc`` pointer: the digest logged on a run therefore names
    a data version committed to git.

    Raises:
        DataVersionError: If the dataset file is tracked by DVC and differs
            from the recorded version.
    """
    dataset_path = resolve_dataset_path(config)
    md5 = file_md5(dataset_path)
    check_dvc_pointer(dataset_path, md5)

    with warnings.catch_warnings():
        # MLflow tries every source type it knows on the path, and warns about
        # those that cannot handle it and about its local-file source being
        # registered twice
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"mlflow\.data\.dataset_source_registry",
        )
        dataset = mlflow.data.from_pandas(
            df,
            source=_relative_to_project(dataset_path),
            name=config.dataset.name,
            digest=md5,
        )

    git_commit, git_dirty = _git_state()
    return RunLineage(
        dataset=dataset,
        dataset_version=config.dataset.version,
        config_path=resolve_config_path(config_path),
        random_state=random_state,
        git_commit=git_commit,
        git_dirty=git_dirty,
        flow_run_id=flow_run_id,
    )


def log_lineage(lineage: RunLineage) -> None:
    """Record a pipeline execution's lineage on the active MLflow run.

    Writes:
      - dataset input (name, MD5 digest, source, schema), context ``training``
      - param ``random_state``
      - tags ``dataset.version``, ``mlflow.source.git.commit``,
        ``source.git.dirty`` and ``prefect.flow_run_id`` (when available)
      - artifact ``config/<file>``: the configuration file
    """
    mlflow.log_input(lineage.dataset, context="training")
    mlflow.log_param("random_state", lineage.random_state)

    tags = {"dataset.version": lineage.dataset_version}
    if lineage.git_commit is not None:
        tags["mlflow.source.git.commit"] = lineage.git_commit
        tags["source.git.dirty"] = str(lineage.git_dirty).lower()
    if lineage.flow_run_id is not None:
        tags["prefect.flow_run_id"] = lineage.flow_run_id
    mlflow.set_tags(tags)

    mlflow.log_artifact(str(lineage.config_path), artifact_path="config")


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
    lineage: RunLineage,
    run_name: str | None = None,
) -> str:
    """Log a training result to MLflow as a new run.

    Returns the new run's id. Registration is a separate step (see
    ``register_run_model``) gated on the accuracy and verification outcomes.
    """
    run_name = run_name or f"{result.model_label}_{result.output_name}"

    with mlflow.start_run(run_name=run_name) as run:
        log_lineage(lineage)

        # Log parameters
        params = {
            "model_type": result.model_type,
            "model_label": result.model_label,
            "output_name": result.output_name,
            "train_fraction": config.model_building.train_test_split.train_fraction,
            "split_method": config.model_building.train_test_split.method,
            # The search settings the trainer actually used, not the global config
            "optimization_method": result.optimization_method or "none",
        }
        if result.cv_folds is not None:
            params["cv_folds"] = result.cv_folds
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

        log_accuracy_assessment(
            assess_accuracy(result.test_metrics, result.output_name, config)
        )

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


def log_accuracy_assessment(assessment: AccuracyAssessment) -> None:
    """Record on the active run how its model fares against the acceptance thresholds.

    Writes, next to the ``test_{metric}`` metrics they refer to:
      - params ``accuracy.{metric}.threshold`` (one per criterion)
      - tags   ``accuracy.{metric}`` = "pass"/"fail" and ``accuracy.status``
        ("passed", "failed", or "not_assessed" when the output has no criterion)
    """
    for outcome in assessment.outcomes:
        mlflow.log_param(f"accuracy.{outcome.metric}.threshold", outcome.threshold)
        mlflow.set_tag(
            f"accuracy.{outcome.metric}", "pass" if outcome.passed else "fail"
        )
    mlflow.set_tag("accuracy.status", assessment.status)


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
    lineage: RunLineage,
) -> str:
    """Log a summary run comparing all models.

    Args:
        results: List of all training results.
        config: Configuration used.
        lineage: Lineage of the pipeline execution that produced the results.

    Returns:
        MLflow run ID for summary.
    """
    with mlflow.start_run(run_name="experiment_summary") as run:
        log_lineage(lineage)

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
                "accuracy_status": assess_accuracy(
                    r.test_metrics, r.output_name, config
                ).status,
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

