"""Custom scorers and evaluation metrics for model training."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate Root Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def max_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate Maximum Absolute Error (infinity norm).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Maximum absolute error value.
    """
    return float(np.max(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# Scorer for GridSearchCV (negative because sklearn maximizes)
rmse_scorer = make_scorer(rmse, greater_is_better=False)


def evaluate_predictions(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Compute all evaluation metrics for predictions.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with mse, rmse, mae, max_ae, and r2.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    return {
        "mse": float(mean_squared_error(y_true_arr, y_pred_arr)),
        "rmse": rmse(y_true_arr, y_pred_arr),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "max_ae": max_absolute_error(y_true_arr, y_pred_arr),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }
