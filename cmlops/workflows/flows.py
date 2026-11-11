"""Prefect flows for the training + verification pipeline."""

from pathlib import Path

import pandas as pd
from joblib import cpu_count
from prefect import flow
from prefect.futures import PrefectFuture
from prefect.task_runners import ThreadPoolTaskRunner

from cmlops.config import PipelineConfig
from cmlops.training.trainer import TrainingResult
from cmlops.workflows.tasks import (task_capture_lineage,
                                             task_load_config, task_load_data,
                                             task_log_summary,
                                             task_prepare_and_split_data,
                                             task_register_winner,
                                             task_setup_mlflow,
                                             task_train_model,
                                             task_verify_model)

# Default number of training tasks to run concurrently. Kept small because the
# runtime is dominated by a few heavy CV searches that each saturate many cores;
# overridable per run via the CLI ``--max-workers`` flag.
DEFAULT_MAX_WORKERS = 2

# Fraction of available CPUs the concurrent hyperparameter searches are allowed
# to use together. Leaves headroom so the machine stays responsive for other
# work. Verification is not covered by this budget: it retrains models through
# vv4ml, with each library's default threading.
DEFAULT_CPU_FRACTION = 0.9


def _cpu_budget(cpu_fraction: float = DEFAULT_CPU_FRACTION) -> int:
    """Worker-process budget for the training searches (e.g. 96 CPUs * 0.9 = 86).

    ``joblib.cpu_count`` reports the CPUs this process may actually use (CPU
    affinity and container CPU quotas included). Recomputed each run, so the
    budget adapts automatically when the workload moves to another server.
    """
    if not 0 < cpu_fraction <= 1:
        raise ValueError(f"cpu_fraction must be in (0, 1], got {cpu_fraction}")
    return max(1, int(cpu_count() * cpu_fraction))


def _inner_n_jobs(
    max_workers: int, cpu_fraction: float = DEFAULT_CPU_FRACTION
) -> int:
    """Per-search ``n_jobs`` so that ``max_workers * n_jobs`` stays within budget.

    This keeps the two nested levels of parallelism (concurrent training tasks x
    parallel CV search) from oversubscribing the CPU.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be at least 1, got {max_workers}")
    return max(1, _cpu_budget(cpu_fraction) // max_workers)


def _verify_and_register_winners(
    winners: dict[str, tuple[TrainingResult, str]],
    X: pd.DataFrame,
    y: pd.DataFrame,
    config: PipelineConfig,
    random_state: int,
) -> None:
    """Verify each winner in parallel, then auto-register passing ones.

    ``winners`` maps output_name -> (TrainingResult, run_id).
    """
    verify_futures = {
        output_name: task_verify_model.submit(
            result=result,
            X=X,
            y=y,
            config=config,
            run_id=run_id,
            random_state=random_state,
        )
        for output_name, (result, run_id) in winners.items()
    }

    for output_name, (result, run_id) in winners.items():
        report = verify_futures[output_name].result()
        status = "passed" if report.all_passed else "failed"
        per_test = ", ".join(
            f"{t.name}={'pass' if t.passed else 'fail'}" for t in report.tests
        )
        print(f"  Verification for {output_name}: {status} ({per_test})")

        registered = task_register_winner(
            result=result, run_id=run_id, report=report, config=config
        )
        if registered:
            model_name = config.registered_model_name(output_name, result.model_label)
            print(f"  Registered: {model_name}")
        else:
            print(f"  Registration skipped for {output_name} (verification failed)")


@flow(
    name="train_single_output",
    task_runner=ThreadPoolTaskRunner(max_workers=DEFAULT_MAX_WORKERS),
)
def train_single_output_flow(
    output_name: str,
    config_path: str | Path | None = None,
    mlflow_tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str | None = None,
    random_state: int = 42,
    skip_verification: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    cpu_fraction: float = DEFAULT_CPU_FRACTION,
) -> list[TrainingResult]:
    """Train all configured models for a single output, then verify the winner.

    ``max_workers`` controls how many models train concurrently; per-search
    ``n_jobs`` is sized so ``max_workers * n_jobs`` stays within ``cpu_fraction``
    of the available CPUs. To actually run ``max_workers`` tasks at once, call
    via ``.with_options(task_runner=ThreadPoolTaskRunner(max_workers=...))`` (the
    CLI does this); the decorator default keeps direct calls consistent.
    """
    config = task_load_config(config_path)
    task_setup_mlflow(mlflow_tracking_uri, experiment_name or config.project.name)

    df = task_load_data(config)
    lineage = task_capture_lineage(df, config, config_path, random_state)
    X, y, split_data = task_prepare_and_split_data(df, config, random_state)

    data = split_data[output_name]

    n_jobs_override = _inner_n_jobs(max_workers, cpu_fraction)
    futures = [
        task_train_model.submit(
            model_config=model_config,
            data=data,
            config=config,
            lineage=lineage,
            random_state=random_state,
            n_jobs_override=n_jobs_override,
        )
        for model_config in config.model_building.model_list
    ]

    # Each (result, run_id) is already logged to MLflow by the training task.
    results_with_runs = [future.result() for future in futures]
    results = [result for result, _ in results_with_runs]

    if not skip_verification:
        best_result, best_run_id = min(
            results_with_runs, key=lambda rr: rr[0].test_metrics["rmse"]
        )
        print(f"\n  Best model for {output_name}: {best_result.model_label}")
        winners = {output_name: (best_result, best_run_id)}
        _verify_and_register_winners(winners, X, y, config, random_state)

    return results


@flow(
    name="train_all_outputs",
    log_prints=True,
    task_runner=ThreadPoolTaskRunner(max_workers=DEFAULT_MAX_WORKERS),
)
def train_all_outputs_flow(
    config_path: str | Path | None = None,
    mlflow_tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str | None = None,
    random_state: int = 42,
    skip_verification: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    cpu_fraction: float = DEFAULT_CPU_FRACTION,
) -> dict[str, list[TrainingResult]]:
    """Train every model × output, then verify the per-output winners.

    ``max_workers`` controls how many models train concurrently; per-search
    ``n_jobs`` is sized so ``max_workers * n_jobs`` stays within ``cpu_fraction``
    of the available CPUs. To actually run ``max_workers`` tasks at once, call
    via ``.with_options(task_runner=ThreadPoolTaskRunner(max_workers=...))`` (the
    CLI does this); the decorator default keeps direct calls consistent.
    """
    config = task_load_config(config_path)
    task_setup_mlflow(mlflow_tracking_uri, experiment_name or config.project.name)

    df = task_load_data(config)
    lineage = task_capture_lineage(df, config, config_path, random_state)
    X, y, all_split_data = task_prepare_and_split_data(df, config, random_state)

    n_jobs_override = _inner_n_jobs(max_workers, cpu_fraction)
    print(
        f"\nSubmitting {len(config.model_building.model_list)} models x "
        f"{len(all_split_data)} outputs for parallel training "
        f"(max_workers={max_workers}, n_jobs/search={n_jobs_override}, "
        f"cpu_budget={_cpu_budget(cpu_fraction)} of {cpu_count()})..."
    )
    futures: list[tuple[str, str, PrefectFuture]] = []
    for output_name, data in all_split_data.items():
        for model_config in config.model_building.model_list:
            future = task_train_model.submit(
                model_config=model_config,
                data=data,
                config=config,
                lineage=lineage,
                random_state=random_state,
                n_jobs_override=n_jobs_override,
            )
            futures.append(
                (output_name, model_config.label or model_config.type, future)
            )

    # Each training task logs its own (result, run_id) to MLflow as soon as it
    # finishes; here we just collect the results for the summary and verification.
    results_with_runs: dict[str, list[tuple[TrainingResult, str]]] = {
        name: [] for name in all_split_data.keys()
    }
    for output_name, model_label, future in futures:
        result, run_id = future.result()
        results_with_runs[output_name].append((result, run_id))
        print(
            f"  {model_label} -> {output_name}: "
            f"RMSE={result.test_metrics['rmse']:.6f}, "
            f"R2={result.test_metrics['r2']:.4f}"
        )

    flat_results = [
        result for pairs in results_with_runs.values() for result, _ in pairs
    ]
    task_log_summary(flat_results, config, lineage)

    if not skip_verification:
        winners: dict[str, tuple[TrainingResult, str]] = {}
        for output_name, pairs in results_with_runs.items():
            best_result, best_run_id = min(
                pairs, key=lambda rr: rr[0].test_metrics["rmse"]
            )
            print(f"\n  Best model for {output_name}: {best_result.model_label}")
            winners[output_name] = (best_result, best_run_id)
        _verify_and_register_winners(winners, X, y, config, random_state)

    return {
        name: [result for result, _ in pairs]
        for name, pairs in results_with_runs.items()
    }


@flow(name="train_specific_model")
def train_specific_model_flow(
    model_type: str,
    output_name: str,
    config_path: str | Path | None = None,
    mlflow_tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str | None = None,
    random_state: int = 42,
) -> TrainingResult:
    """Train a specific model for a specific output.

    Debug/retrain shortcut: no verification, no registration. Run a full
    or single-output flow to produce a registered model.
    """
    config = task_load_config(config_path)
    task_setup_mlflow(mlflow_tracking_uri, experiment_name or config.project.name)

    model_config = None
    for mc in config.model_building.model_list:
        if mc.type == model_type or mc.label == model_type:
            model_config = mc
            break

    if model_config is None:
        raise ValueError(f"Model type '{model_type}' not found in config")

    df = task_load_data(config)
    lineage = task_capture_lineage(df, config, config_path, random_state)
    _X, _y, split_data = task_prepare_and_split_data(df, config, random_state)
    data = split_data[output_name]

    result, _run_id = task_train_model(
        model_config=model_config,
        data=data,
        config=config,
        lineage=lineage,
        random_state=random_state,
    )

    return result
