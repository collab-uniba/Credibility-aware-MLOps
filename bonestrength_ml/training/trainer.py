"""Core training logic with hyperparameter optimization."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV

from bonestrength_ml.config import ModelConfig, OptimizationConfig
from bonestrength_ml.training.metrics import evaluate_predictions, rmse_scorer
from bonestrength_ml.training.model_factory import (
    create_model,
    get_param_grid,
    normalize_gpr_best_params,
)
from bonestrength_ml.training.splitter import TrainTestData


@dataclass
class TrainingResult:
    """Result of training a single model."""

    model_label: str
    model_type: str
    output_name: str
    best_estimator: BaseEstimator
    best_params: dict[str, Any]
    cv_results: dict[str, Any]
    test_metrics: dict[str, float]
    train_metrics: dict[str, float]


def train_single_model(
    model_config: ModelConfig,
    data: TrainTestData,
    global_optimization: OptimizationConfig,
    random_state: int = 42,
    n_jobs_override: int | None = None,
) -> TrainingResult:
    """Train a single model with hyperparameter optimization.

    Args:
        model_config: Model configuration from YAML.
        data: TrainTestData with X_train, y_train, X_test, y_test.
        global_optimization: Global optimization config (used if model-specific is None).
        random_state: Random seed.
        n_jobs_override: When set, overrides the config's ``n_jobs`` for the CV
            search. The orchestrating flow uses this to size search parallelism
            against a global CPU budget (see ``workflows.flows``). When ``None``,
            the config value is used.

    Returns:
        TrainingResult with trained model and metrics.
    """
    # Use model-specific optimization config, fallback to global
    opt_config = model_config.optimization or global_optimization

    # Create base model and parameter grid
    base_model = create_model(model_config, random_state=random_state)
    param_grid = get_param_grid(model_config)

    # Skip search if no hyperparameters to tune
    if not param_grid:
        base_model.fit(data.X_train, data.y_train)
        best_estimator = base_model
        best_params: dict[str, Any] = {}
        cv_results: dict[str, Any] = {}
    else:
        # Select search strategy
        search = _create_search(
            base_model=base_model,
            param_grid=param_grid,
            opt_config=opt_config,
            random_state=random_state,
            groups_train=data.groups_train,
            n_jobs_override=n_jobs_override,
        )
        search.fit(data.X_train, data.y_train, groups=data.groups_train)
        best_estimator = search.best_estimator_
        best_params = search.best_params_
        if model_config.type == "GaussianProcessRegressor":
            best_params = normalize_gpr_best_params(best_params)
        cv_results = search.cv_results_

    # Evaluate on train and test
    # ravel() ensures 1D output (PLSRegression returns 2D arrays)
    y_pred_train = best_estimator.predict(data.X_train).ravel()
    y_pred_test = best_estimator.predict(data.X_test).ravel()

    train_metrics = evaluate_predictions(data.y_train.values, y_pred_train)
    test_metrics = evaluate_predictions(data.y_test.values, y_pred_test)

    return TrainingResult(
        model_label=model_config.label or model_config.type,
        model_type=model_config.type,
        output_name=data.output_name,
        best_estimator=best_estimator,
        best_params=best_params,
        cv_results=cv_results,
        test_metrics=test_metrics,
        train_metrics=train_metrics,
    )


def _create_search(
    base_model: BaseEstimator,
    param_grid: dict[str, list[Any]],
    opt_config: OptimizationConfig,
    random_state: int,
    groups_train: np.ndarray | None = None,
    n_jobs_override: int | None = None,
) -> GridSearchCV | RandomizedSearchCV:
    """Create the appropriate search object based on config."""
    cv = GroupKFold(n_splits=opt_config.cv) if groups_train is not None else opt_config.cv

    n_jobs = n_jobs_override if n_jobs_override is not None else opt_config.n_jobs

    common_kwargs = {
        "estimator": base_model,
        "scoring": rmse_scorer,
        "cv": cv,
        "n_jobs": n_jobs,
        "return_train_score": True,
        "verbose": 1,
    }

    if opt_config.optimization_method == "RandomizedSearchCV":
        # Calculate reasonable n_iter based on grid size
        grid_size = 1
        for v in param_grid.values():
            grid_size *= len(v)
        n_iter = min(grid_size, 50)

        return RandomizedSearchCV(
            param_distributions=param_grid,
            n_iter=n_iter,
            random_state=random_state,
            **common_kwargs,
        )
    else:
        return GridSearchCV(
            param_grid=param_grid,
            **common_kwargs,
        )
