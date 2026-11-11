"""Registry of verification check implementations.

Tests self-register at import time; the orchestrator iterates checks selected
by ``config.verification.test_list`` and resolves each entry's ``type`` against
this registry. Adding a new V&V 40 test means dropping a module that calls
``register("<name>", <check_fn>)`` — no changes to the flow, MLflow logging,
or CLI required.
"""

from collections.abc import Iterator

from cmlops.config import PipelineConfig, VerificationTestEntry
from cmlops.verification.base import VerificationCheck

_CHECKS: dict[str, VerificationCheck] = {}


def register(name: str, check: VerificationCheck) -> None:
    """Register a verification check under a YAML ``type`` name."""
    _CHECKS[name] = check


def get_check(name: str) -> VerificationCheck:
    """Look up a registered check; raise if unknown."""
    if name not in _CHECKS:
        raise ValueError(
            f"Unknown verification test type '{name}'. "
            f"Registered types: {sorted(_CHECKS)}"
        )
    return _CHECKS[name]


def iter_enabled_checks(
    config: PipelineConfig,
) -> Iterator[tuple[VerificationTestEntry, VerificationCheck]]:
    """Yield (entry, check) pairs in the order declared in YAML."""
    for entry in config.verification.test_list:
        yield entry, get_check(entry.type)
