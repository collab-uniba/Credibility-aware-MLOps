"""Core abstractions for the pluggable verification suite."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from bonestrength_ml.config import BoneStrengthMLConfig, VerificationTestEntry
from bonestrength_ml.training.trainer import TrainingResult


@dataclass
class VerificationContext:
    """Inputs every verification check needs.

    Built per check for a winning model. ``prior`` holds the results of the
    checks already run on the same model, keyed by test name, for checks that
    build on an earlier one.
    """

    result: TrainingResult
    X: pd.DataFrame
    y: pd.DataFrame
    config: BoneStrengthMLConfig
    entry: VerificationTestEntry
    random_state: int = 42
    prior: "dict[str, TestResult]" = field(default_factory=dict)


@dataclass
class TestResult:
    """Common shape every verification check returns."""

    name: str
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": self.metrics,
            "params": self.params,
            "details": self.details,
        }


class VerificationCheck(Protocol):
    """Callable interface every verification test must implement."""

    def __call__(self, ctx: VerificationContext) -> TestResult: ...


@dataclass
class VerificationReport:
    """Aggregated outcome of all checks run against a winning model."""

    output_name: str
    tests: list[TestResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return bool(self.tests) and all(t.passed for t in self.tests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_name": self.output_name,
            "all_passed": self.all_passed,
            "tests": [t.to_dict() for t in self.tests],
        }
