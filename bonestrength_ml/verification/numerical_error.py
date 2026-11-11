"""Numerical error test for verifying prediction stability across training subsets.

The test itself is implemented by ``vv4ml``; this module adapts it to the
pipeline's ``VerificationContext`` / ``TestResult`` interface.
"""

from vv4ml import generate_lhs_samples, run_numerical_error_test

from bonestrength_ml.verification.base import TestResult, VerificationContext
from bonestrength_ml.verification.registry import register


def numerical_error_check(ctx: VerificationContext) -> TestResult:
    """Run the ``vv4ml`` numerical error test from a ``VerificationContext``.

    The model is retrained ``sets`` times on independent random subsets of the
    training data, each as large as the minimum converging size found by the
    convergence test, and the spread of the predictions on LHS-sampled points
    is compared with the threshold. The convergence test must therefore be
    listed before this one in ``verification.test_list``; if it did not
    converge there is no subset size to use and this test fails without
    running.
    """
    from bonestrength_ml.training.splitter import prepare_train_test_split

    output_name = ctx.result.output_name

    threshold = next(
        (t.value for t in ctx.entry.thresholds if t.field == output_name),
        None,
    )
    if threshold is None:
        raise ValueError(
            f"No numerical error threshold configured for output '{output_name}'"
        )

    sets = ctx.entry.parameters.get("sets")
    if not sets:
        raise ValueError("numerical error test requires 'sets' in parameters")

    convergence = ctx.prior.get("convergence")
    if convergence is None:
        raise ValueError(
            "numerical error test requires the convergence test to be listed "
            "before it in verification.test_list"
        )

    lhs_samples = ctx.entry.parameters.get("lhs_samples", 1000)
    params = {
        "threshold": float(threshold),
        "sets": int(sets),
        "lhs_samples": lhs_samples,
    }

    min_k = convergence.details.get("min_k")
    if min_k is None:
        return TestResult(
            name="numerical_error",
            passed=False,
            params=params,
            details={"skipped": "the convergence test did not converge"},
        )

    split = prepare_train_test_split(ctx.X, ctx.y, ctx.config, ctx.random_state)
    data = split[output_name]

    X_eval = generate_lhs_samples(
        inputs=ctx.config.dataset.inputs,
        sample_size=lhs_samples,
        random_state=ctx.random_state,
    )

    result = run_numerical_error_test(
        x_train=data.X_train,
        y_train=data.y_train,
        x_eval=X_eval,
        model_type=ctx.result.model_type,
        best_params=ctx.result.best_params,
        n_repetitions=int(sets),
        convergence_rows=int(min_k),
        threshold=float(threshold),
        random_state=ctx.random_state,
    )

    return TestResult(
        name="numerical_error",
        passed=bool(result.passed),
        # vv4ml names the field max_variance, but it holds a standard deviation
        metrics={"max_std_dev": float(result.max_variance)},
        params={**params, "convergence_rows": int(min_k)},
        details={},
    )


register("numerical_error", numerical_error_check)
