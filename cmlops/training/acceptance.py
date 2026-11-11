"""Assessment of model accuracy against the acceptance thresholds.

The thresholds are those declared under ``gate_thresholds`` in the
configuration, derived from the context of use of the predictor.
"""

from dataclasses import dataclass, field

from cmlops.config import PipelineConfig

# Test metric (as named by ``evaluate_predictions``) behind each gate metric
GATE_METRICS = {
    "MaximumAbsoluteError": "max_ae",
    "RMSE": "rmse",
}


@dataclass
class AccuracyOutcome:
    """Outcome of one acceptance criterion: a test metric against its threshold."""

    metric: str
    threshold: float
    value: float
    passed: bool


@dataclass
class AccuracyAssessment:
    """Outcomes of all the acceptance criteria declared for one output."""

    output_name: str
    outcomes: list[AccuracyOutcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True unless a criterion failed: an output with no criterion has no gate."""
        return all(o.passed for o in self.outcomes)

    @property
    def status(self) -> str:
        if not self.outcomes:
            return "not_assessed"
        return "passed" if self.passed else "failed"


def assess_accuracy(
    test_metrics: dict[str, float],
    output_name: str,
    config: PipelineConfig,
) -> AccuracyAssessment:
    """Compare a model's test metrics with the acceptance thresholds of its output.

    Args:
        test_metrics: Test metrics of the model, as returned by
            ``evaluate_predictions``.
        output_name: Output the model predicts.
        config: Pipeline configuration holding the thresholds.

    Returns:
        One outcome per criterion declared for the output; a criterion passes
        when the metric does not exceed its threshold.
    """
    assessment = AccuracyAssessment(output_name=output_name)
    for criterion in config.gate_thresholds.validation:
        if criterion.field != output_name:
            continue
        metric = GATE_METRICS[criterion.metric]
        value = test_metrics[metric]
        assessment.outcomes.append(
            AccuracyOutcome(
                metric=metric,
                threshold=criterion.value,
                value=value,
                passed=value <= criterion.value,
            )
        )
    return assessment
