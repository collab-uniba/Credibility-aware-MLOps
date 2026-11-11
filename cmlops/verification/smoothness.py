"""Smoothness test for verifying model output regularity under input perturbations.

The test itself is implemented by ``vv4ml``; this module adapts it to the
pipeline's ``VerificationContext`` / ``TestResult`` interface.
"""

from vv4ml import generate_lhs_samples, run_smoothness_test

from cmlops.verification.base import TestResult, VerificationContext
from cmlops.verification.registry import register


def smoothness_check(ctx: VerificationContext) -> TestResult:
    """Run the ``vv4ml`` smoothness test from a ``VerificationContext``."""
    output_name = ctx.result.output_name

    threshold = next(
        (t.value for t in ctx.entry.thresholds if t.field == output_name),
        None,
    )
    if threshold is None:
        raise ValueError(
            f"No smoothness threshold configured for output '{output_name}'"
        )

    params = ctx.entry.parameters
    sample_fraction = params.get("sample_fraction")
    perturbation_scaled_magnitude = params.get("perturbation_scaled_magnitude")
    if sample_fraction is None or perturbation_scaled_magnitude is None:
        raise ValueError(
            "smoothness test requires 'sample_fraction' and "
            "'perturbation_scaled_magnitude' in parameters"
        )

    lhs_samples = params.get("lhs_samples", 1000)
    X_eval = generate_lhs_samples(
        inputs=ctx.config.dataset.inputs,
        sample_size=lhs_samples,
        random_state=ctx.random_state,
    )
    input_ranges = [(inp.range[0], inp.range[1]) for inp in ctx.config.dataset.inputs]

    result = run_smoothness_test(
        model=ctx.result.best_estimator,
        x_eval=X_eval,
        input_ranges=input_ranges,
        sample_fraction=float(sample_fraction),
        perturbation_scaled_magnitude=float(perturbation_scaled_magnitude),
        threshold=float(threshold),
        random_state=ctx.random_state,
    )

    return TestResult(
        name="smoothness",
        passed=bool(result.passed),
        metrics={"max_error_ratio": float(result.max_error_ratio)},
        params={
            "threshold": float(threshold),
            "sample_fraction": float(sample_fraction),
            "perturbation_scaled_magnitude": float(perturbation_scaled_magnitude),
            "lhs_samples": lhs_samples,
        },
        details={},
    )


register("smoothness", smoothness_check)
