"""Combining the eight metric families into one composite score.

Why not an arithmetic mean
--------------------------
Because it launders a fatal defect into a passing grade. With the configured
weights, a photograph that is perfectly exposed, perfectly contrasted, noise
free, high resolution and completely out of focus scores **78** under a plain
mean. That image is worthless for face recognition.

Two mechanisms fix it:

**Weighted power mean, exponent 0.5.** The generalised mean with an exponent
below 1 is pulled towards its smallest component. The same image scores 61
instead of 78 - visibly degraded, in proportion to how bad the worst dimension
is, and still continuous so the fraud engine gets a gradient rather than a
cliff.

**Critical floor.** A power mean still cannot express "this single dimension
is disqualifying whatever the others say". Any component in
``critical_components`` at or below ``critical_floor`` marks the image
``usable=False`` outright. Only blur, sharpness and resolution are critical by
default: exposure and contrast degrade recognition but do not destroy the
signal, and a dim-but-sharp photograph is still workable.

Unmeasured metrics are dropped and the remaining weights renormalised, rather
than being scored zero or one. Scoring zero would punish an image for a
measurement we chose not to take; scoring one would flatter it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hamqadam_ai.core.config import QualityAggregationConfig
from hamqadam_ai.quality.base import MetricResult
from hamqadam_ai.quality.scoring import (
    limiting_component,
    weighted_mean,
    weighted_power_mean,
)


@dataclass(slots=True)
class QualityAssessment:
    """The complete quality verdict for one image.

    Attributes:
        overall_score: Composite quality on the 0-100 scale.
        usable: Whether the image clears both the role threshold and every
            critical floor.
        metrics: Every metric family's result, keyed by name.
        component_scores: Each measured family's score in ``[0, 1]``.
        effective_weights: Weights actually applied, after renormalising
            around any unmeasured family.
        limiting_factor: The family contributing the largest weighted deficit.
        critical_failures: Families that fell at or below the critical floor.
        unmeasured: Families that could not be computed.
        min_required: The role threshold this image was judged against.
        arithmetic_score: The plain weighted mean, on the 0-100 scale.
            Reported alongside the composite so the effect of the power mean is
            auditable rather than mysterious.
        notes: Human-readable observations gathered from the analysers.
    """

    overall_score: float
    usable: bool
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    component_scores: dict[str, float] = field(default_factory=dict)
    effective_weights: dict[str, float] = field(default_factory=dict)
    limiting_factor: str | None = None
    critical_failures: list[str] = field(default_factory=list)
    unmeasured: list[str] = field(default_factory=list)
    min_required: float = 0.0
    arithmetic_score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def score_for(self, family: str) -> float:
        """Return one family's score on the 0-100 scale, or 0.0 if absent."""
        result = self.metrics.get(family)
        return round(result.score * 100.0, 2) if result is not None else 0.0

    def measurement(self, family: str, key: str) -> float | None:
        """Return one raw measurement, or ``None`` when it was not taken."""
        result = self.metrics.get(family)
        if result is None:
            return None
        return result.measurements.get(key)

    def summary(self) -> dict[str, Any]:
        """Compact, PII-free summary for logging."""
        return {
            "overall": round(self.overall_score, 2),
            "usable": self.usable,
            "limiting": self.limiting_factor,
            "min_required": self.min_required,
            "critical_failures": list(self.critical_failures),
            "unmeasured": list(self.unmeasured),
        }


class QualityAggregator:
    """Combines metric results into a composite assessment.

    Args:
        config: The aggregation section of the quality configuration.
    """

    __slots__ = ("_config",)

    def __init__(self, config: QualityAggregationConfig) -> None:
        self._config = config

    def aggregate(
        self,
        results: list[MetricResult],
        *,
        min_required: float,
        critical_components: Sequence[str] | None = None,
    ) -> QualityAssessment:
        """Combine metric results into a verdict.

        Args:
            results: One result per metric family.
            min_required: The role-specific acceptance threshold, 0-100.
            critical_components: Components whose critical floor is enforced
                for this role. Defaults to the global list. Passed in rather
                than looked up here because the floor is genuinely role
                dependent: a print's softness is the medium, not a fault.

        Returns:
            The composite assessment.
        """
        enforced = (
            list(critical_components)
            if critical_components is not None
            else list(self._config.critical_components)
        )
        metrics = {result.name: result for result in results}

        measured = {
            result.name: result.score for result in results if result.measured
        }
        unmeasured = [result.name for result in results if not result.measured]

        if not measured:
            return QualityAssessment(
                overall_score=0.0,
                usable=False,
                metrics=metrics,
                unmeasured=unmeasured,
                min_required=min_required,
                notes=["No quality metric could be computed for this image."],
            )

        weights = self._effective_weights(set(measured))

        composite = weighted_power_mean(measured, weights, self._config.power)
        arithmetic = weighted_mean(measured, weights)

        critical_failures = [
            name
            for name in enforced
            if name in measured and measured[name] <= self._config.critical_floor
        ]

        overall = round(composite * 100.0, 2)
        usable = overall >= min_required and not critical_failures

        notes = [
            f"{result.name}: {result.note}"
            for result in results
            if result.note is not None
        ]
        if critical_failures:
            readable = ", ".join(critical_failures)
            notes.insert(
                0,
                f"Rejected outright: {readable} fell at or below the critical "
                f"floor of {self._config.critical_floor:.2f}, which makes the "
                f"image unusable regardless of its composite score.",
            )

        return QualityAssessment(
            overall_score=overall,
            usable=usable,
            metrics=metrics,
            component_scores=measured,
            effective_weights=weights,
            limiting_factor=limiting_component(measured, weights),
            critical_failures=critical_failures,
            unmeasured=unmeasured,
            min_required=min_required,
            arithmetic_score=round(arithmetic * 100.0, 2),
            notes=notes,
        )

    def _effective_weights(self, measured: set[str]) -> dict[str, float]:
        """Renormalise the configured weights over the measured families only.

        Dropping an unmeasured family and rescaling is the only honest option:
        scoring it zero would punish the image for a measurement we chose not
        to take, and scoring it one would flatter it.
        """
        subset = {
            name: weight
            for name, weight in self._config.weights.items()
            if name in measured
        }
        total = sum(subset.values())
        if total <= 0.0:
            share = 1.0 / max(len(measured), 1)
            return dict.fromkeys(measured, share)
        return {name: weight / total for name, weight in subset.items()}


__all__ = ["QualityAggregator", "QualityAssessment"]
