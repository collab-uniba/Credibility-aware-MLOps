"""Verification tests for BoneStrengthML models.

The tests themselves are implemented by the ``vv4ml`` library; each module
here adapts one of them to the pipeline (configuration, evaluation points,
``TestResult``).

Importing this package eagerly imports each test module so its check
self-registers with the registry. Adding a new V&V 40 test = drop a new
module here and add its import below.
"""

from bonestrength_ml.verification.base import (
    TestResult,
    VerificationCheck,
    VerificationContext,
    VerificationReport,
)
from bonestrength_ml.verification.convergence import convergence_check
from bonestrength_ml.verification.registry import (
    get_check,
    iter_enabled_checks,
    register,
)
from bonestrength_ml.verification.smoothness import smoothness_check
from bonestrength_ml.verification.verify import run_full_verification

__all__ = [
    "TestResult",
    "VerificationCheck",
    "VerificationContext",
    "VerificationReport",
    "convergence_check",
    "get_check",
    "iter_enabled_checks",
    "register",
    "run_full_verification",
    "smoothness_check",
]
