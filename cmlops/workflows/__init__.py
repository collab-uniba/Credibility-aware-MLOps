"""Prefect workflows for the pipeline."""

from cmlops.workflows.flows import (
    train_all_outputs_flow,
    train_single_output_flow,
    train_specific_model_flow,
)

__all__ = [
    "train_all_outputs_flow",
    "train_single_output_flow",
    "train_specific_model_flow",
]
