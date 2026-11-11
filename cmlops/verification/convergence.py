"""Convergence test for verifying model stability across training set sizes.

The test itself is implemented by ``vv4ml``; this module adapts it to the
pipeline's ``VerificationContext`` / ``TestResult`` interface.
"""

from vv4ml import generate_lhs_samples, run_convergence_test

from cmlops.verification.base import TestResult, VerificationContext
from cmlops.verification.registry import register


def convergence_check(ctx: VerificationContext) -> TestResult:
    """Run the ``vv4ml`` convergence test from a ``VerificationContext``.

    Uses LHS-sampled evaluation points; the winning model's predictions on
    those points are the reference predictions to which subset-trained models
    are compared, using the metric configured for the output's threshold.
    """
    from cmlops.training.splitter import prepare_train_test_split

    output_name = ctx.result.output_name

    criterion = next(
        (t for t in ctx.entry.thresholds if t.field == output_name),
        None,
    )
    if criterion is None:
        raise ValueError(
            f"No convergence threshold configured for output '{output_name}'"
        )
    threshold = float(criterion.value)

    n_rows = ctx.entry.parameters.get("n_rows")
    if not n_rows:
        raise ValueError("convergence test requires 'n_rows' in parameters")

    split = prepare_train_test_split(ctx.X, ctx.y, ctx.config, ctx.random_state)
    data = split[output_name]

    lhs_samples = ctx.entry.parameters.get("lhs_samples", 1000)
    X_eval = generate_lhs_samples(
        inputs=ctx.config.dataset.inputs,
        sample_size=lhs_samples,
        random_state=ctx.random_state,
    )
    ref_predictions = ctx.result.best_estimator.predict(X_eval)

    result = run_convergence_test(
        x_train=data.X_train,
        y_train=data.y_train,
        x_eval=X_eval,
        ref_predictions=ref_predictions,
        model_type=ctx.result.model_type,
        best_params=ctx.result.best_params,
        n_rows=list(n_rows),
        threshold=threshold,
        metric=criterion.metric,
        random_state=ctx.random_state,
    )

    metrics: dict[str, float] = {
        "max_error": float(max(result.errors)) if result.errors else float("nan"),
        "final_error": float(result.errors[-1]) if result.errors else float("nan"),
    }
    if result.min_k is not None:
        metrics["min_k"] = float(result.min_k)

    return TestResult(
        name="convergence",
        passed=bool(result.passed),
        metrics=metrics,
        params={
            "threshold": threshold,
            "n_rows": ",".join(str(k) for k in n_rows),
            "lhs_samples": lhs_samples,
        },
        details={
            "errors": [float(e) for e in result.errors],
            "n_rows_tested": list(result.n_rows_tested),
            "min_k": result.min_k,
        },
    )


register("convergence", convergence_check)
