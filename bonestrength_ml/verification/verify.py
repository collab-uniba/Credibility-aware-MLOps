"""Orchestrator: runs every enabled verification check against a winning model."""

import pandas as pd

from bonestrength_ml.config import BoneStrengthMLConfig
from bonestrength_ml.training.trainer import TrainingResult
from bonestrength_ml.verification.base import (
    VerificationContext,
    VerificationReport,
)
from bonestrength_ml.verification.registry import iter_enabled_checks


def run_full_verification(
    result: TrainingResult,
    X: pd.DataFrame,
    y: pd.DataFrame,
    config: BoneStrengthMLConfig,
    random_state: int = 42,
) -> VerificationReport:
    """Run every verification check declared in ``config.verification.test_list``.

    Pure function: builds a fresh ``VerificationContext`` per check (each carries
    its own YAML entry and the results of the checks run before it) and assembles a ``VerificationReport``. Does not touch
    MLflow — that happens in ``log_verification_results``.
    """
    report = VerificationReport(output_name=result.output_name)
    for entry, check in iter_enabled_checks(config):
        ctx = VerificationContext(
            result=result,
            X=X,
            y=y,
            config=config,
            entry=entry,
            random_state=random_state,
            prior={t.name: t for t in report.tests},
        )
        report.tests.append(check(ctx))
    return report
