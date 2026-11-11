"""Training module for the pipeline."""

from cmlops.training.metrics import evaluate_predictions, rmse, rmse_scorer
from cmlops.training.model_factory import (
    KERNEL_REGISTRY,
    MODEL_REGISTRY,
    create_model,
    get_param_grid,
    normalize_gpr_best_params,
)
from cmlops.training.splitter import TrainTestData, prepare_train_test_split, split_data_for_all_outputs
from cmlops.training.trainer import TrainingResult, train_single_model

__all__ = [
    "KERNEL_REGISTRY",
    "MODEL_REGISTRY",
    "TrainTestData",
    "TrainingResult",
    "create_model",
    "evaluate_predictions",
    "get_param_grid",
    "normalize_gpr_best_params",
    "prepare_train_test_split",
    "rmse",
    "rmse_scorer",
    "split_data_for_all_outputs",
    "train_single_model",
]
