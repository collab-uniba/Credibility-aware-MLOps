"""CLI for BoneStrengthML training + verification pipeline."""

import subprocess
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="bsml",
    help="BoneStrengthML training pipeline with MLflow tracking",
)
console = Console()


def _validate_cpu_fraction(value: float) -> float:
    """Reject CPU fractions outside (0, 1]."""
    if not 0 < value <= 1:
        raise typer.BadParameter("must be greater than 0 and at most 1")
    return value


@app.command()
def train(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Specific output to train (maxStrain_11, maxStrain_33). Trains all if not specified.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Specific model type to train. Trains all configured models if not specified.",
    ),
    mlflow_uri: str = typer.Option(
        "sqlite:///mlruns.db",
        "--mlflow-uri",
        help="MLflow tracking URI",
    ),
    experiment: str = typer.Option(
        "BoneStrengthML",
        "--experiment",
        "-e",
        help="MLflow experiment name",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    skip_verification: bool = typer.Option(
        False,
        "--skip-verification",
        help="Skip verification (and therefore registration) of winning models. "
        "No effect when --model is also specified (that path never verifies).",
    ),
    max_workers: int = typer.Option(
        2,  # keep in sync with flows.DEFAULT_MAX_WORKERS
        "--max-workers",
        min=1,
        help="How many models to train concurrently. Per-search n_jobs is sized "
        "so max_workers x n_jobs stays within --cpu-fraction of the CPUs, "
        "avoiding oversubscription. Ignored with --model (single inline run).",
    ),
    cpu_fraction: float = typer.Option(
        0.9,  # keep in sync with flows.DEFAULT_CPU_FRACTION
        "--cpu-fraction",
        callback=_validate_cpu_fraction,
        help="Fraction of available CPUs the hyperparameter searches may use "
        "together (default 0.9 leaves headroom for other activity). Verification "
        "is not covered: it retrains models with each library's default threading. "
        "Ignored with --model.",
    ),
) -> None:
    """Run the training + verification pipeline.

    Behavior:

      bsml train                          -> trains all models x both outputs,
                                              verifies the 2 winners, registers
                                              passing winners to MLflow Registry.

      bsml train --output X               -> trains all models for one output,
                                              verifies + registers the winner.

      bsml train --output X --model Y     -> trains one specific model. Debug/
                                              retrain shortcut: NO verification,
                                              NO registration.
    """
    from prefect.task_runners import ThreadPoolTaskRunner

    from bonestrength_ml.workflows.flows import (
        train_all_outputs_flow,
        train_single_output_flow,
        train_specific_model_flow,
    )

    # Run max_workers training tasks concurrently. The flows size per-search
    # n_jobs against the CPU budget so the two levels don't oversubscribe.
    task_runner = ThreadPoolTaskRunner(max_workers=max_workers)

    console.print("[bold blue]BoneStrengthML Training Pipeline[/bold blue]")
    console.print(f"MLflow URI: {mlflow_uri}")
    console.print(f"Experiment: {experiment}")
    console.print(f"Random seed: {seed}")
    console.print()

    if model and output:
        console.print(f"Training {model} for {output} (no verification)...")
        result = train_specific_model_flow(
            model_type=model,
            output_name=output,
            config_path=config,
            mlflow_tracking_uri=mlflow_uri,
            experiment_name=experiment,
            random_state=seed,
        )
        _print_result(result)

    elif output:
        console.print(f"Training all models for {output}...")
        results = train_single_output_flow.with_options(task_runner=task_runner)(
            output_name=output,
            config_path=config,
            mlflow_tracking_uri=mlflow_uri,
            experiment_name=experiment,
            random_state=seed,
            skip_verification=skip_verification,
            max_workers=max_workers,
            cpu_fraction=cpu_fraction,
        )
        _print_results_table(results)

    else:
        console.print("Training all models for all outputs...")
        all_results = train_all_outputs_flow.with_options(task_runner=task_runner)(
            config_path=config,
            mlflow_tracking_uri=mlflow_uri,
            experiment_name=experiment,
            random_state=seed,
            skip_verification=skip_verification,
            max_workers=max_workers,
            cpu_fraction=cpu_fraction,
        )
        for output_name, results in all_results.items():
            console.print(f"\n[bold]{output_name}[/bold]")
            _print_results_table(results)

    console.print("\n[green]Training complete![/green]")
    console.print(f"View results: mlflow ui --backend-store-uri {mlflow_uri}")


@app.command()
def mlflow_ui(
    mlflow_uri: str = typer.Option(
        "sqlite:///mlruns.db",
        "--mlflow-uri",
        help="MLflow tracking URI",
    ),
    port: int = typer.Option(
        5000,
        "--port",
        "-p",
        help="Port for MLflow UI",
    ),
) -> None:
    """Start MLflow UI to view experiment results."""
    console.print(f"Starting MLflow UI at http://localhost:{port}")
    subprocess.run(
        ["mlflow", "ui", "--backend-store-uri", mlflow_uri, "--port", str(port)],
        check=False,
    )


@app.command()
def list_models(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """List configured models from config file."""
    from bonestrength_ml.config import load_config

    cfg = load_config(config) if config else load_config()

    table = Table(title="Configured Models")
    table.add_column("Label", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Optimization", style="yellow")
    table.add_column("Hyperparameters (search)", style="dim")

    for model_cfg in cfg.model_building.model_list:
        opt = model_cfg.optimization or cfg.model_building.optimization
        search_params = [
            k for k, v in model_cfg.hyperparameters.items() if isinstance(v, list)
        ]

        table.add_row(
            model_cfg.label or model_cfg.type,
            model_cfg.type,
            opt.optimization_method,
            ", ".join(search_params) if search_params else "none",
        )

    console.print(table)


def _print_result(result) -> None:  # noqa: ANN001
    """Print single training result."""
    console.print(f"\n[bold]{result.model_label}[/bold] -> {result.output_name}")
    console.print(f"  Test RMSE:   {result.test_metrics['rmse']:.6f}")
    console.print(f"  Test R2:     {result.test_metrics['r2']:.4f}")
    console.print(f"  Test MAE:    {result.test_metrics['mae']:.6f}")
    console.print(f"  Test Max AE: {result.test_metrics['max_ae']:.6f}")


def _print_results_table(results: list) -> None:
    """Print results as rich table."""
    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Test RMSE", style="green")
    table.add_column("Test R2", style="yellow")
    table.add_column("Test MAE", style="dim")

    for r in sorted(results, key=lambda x: x.test_metrics["rmse"]):
        table.add_row(
            r.model_label,
            f"{r.test_metrics['rmse']:.6f}",
            f"{r.test_metrics['r2']:.4f}",
            f"{r.test_metrics['mae']:.6f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
