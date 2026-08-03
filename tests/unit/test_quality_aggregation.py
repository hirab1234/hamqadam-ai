"""Composite aggregation: power mean, critical floors and weight handling.

This is where eight independent measurements become the single number the
Backend's rules engine switches on, so its behaviour under partial information
and under a single catastrophic dimension is worth pinning exactly.
"""

from __future__ import annotations

import pytest

from hamqadam_ai.core.config import QualityAggregationConfig, get_settings
from hamqadam_ai.quality.aggregator import QualityAggregator, QualityAssessment
from hamqadam_ai.quality.base import MetricResult

ALL_FAMILIES = (
    "blur",
    "sharpness",
    "brightness",
    "contrast",
    "noise",
    "resolution",
    "pixelation",
    "distortion",
)


def results(**scores: float) -> list[MetricResult]:
    """Build a full set of metric results, defaulting unlisted ones to 1.0."""
    return [
        MetricResult(name=name, score=scores.get(name, 1.0))
        for name in ALL_FAMILIES
    ]


@pytest.fixture
def aggregator() -> QualityAggregator:
    """Aggregator using the real production configuration."""
    return QualityAggregator(get_settings().quality.aggregation)


# --------------------------------------------------------------------------- #
# Basic behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_perfect_image_scores_one_hundred(aggregator: QualityAggregator) -> None:
    assessment = aggregator.aggregate(results(), min_required=50.0)
    assert assessment.overall_score == pytest.approx(100.0)
    assert assessment.usable is True
    assert assessment.critical_failures == []


@pytest.mark.unit
def test_a_worthless_image_scores_zero(aggregator: QualityAggregator) -> None:
    assessment = aggregator.aggregate(
        results(**dict.fromkeys(ALL_FAMILIES, 0.0)), min_required=50.0
    )
    assert assessment.overall_score == pytest.approx(0.0)
    assert assessment.usable is False


@pytest.mark.unit
def test_the_composite_sits_below_the_arithmetic_mean(
    aggregator: QualityAggregator,
) -> None:
    """The power mean's whole purpose: a single bad dimension must not be
    averaged away by seven healthy ones."""
    assessment = aggregator.aggregate(results(blur=0.10), min_required=50.0)
    assert assessment.overall_score < assessment.arithmetic_score
    assert assessment.arithmetic_score - assessment.overall_score > 2.0


@pytest.mark.unit
def test_both_scores_are_reported_for_auditability(
    aggregator: QualityAggregator,
) -> None:
    """Publishing the plain mean alongside the composite makes the power-mean
    adjustment inspectable rather than mysterious."""
    assessment = aggregator.aggregate(results(noise=0.4), min_required=50.0)
    assert assessment.arithmetic_score > 0.0
    assert assessment.overall_score > 0.0
    assert assessment.arithmetic_score != assessment.overall_score


# --------------------------------------------------------------------------- #
# Critical floors
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("family", ["blur", "sharpness", "resolution"])
def test_a_critical_dimension_below_the_floor_rejects_outright(
    aggregator: QualityAggregator, family: str
) -> None:
    """A perfectly exposed, perfectly contrasted, completely out-of-focus
    photograph is not a passing grade whatever the composite says."""
    assessment = aggregator.aggregate(results(**{family: 0.05}), min_required=50.0)

    assert assessment.overall_score > 50.0, "composite alone would have passed"
    assert assessment.usable is False
    assert family in assessment.critical_failures


@pytest.mark.unit
@pytest.mark.parametrize("family", ["brightness", "contrast", "noise", "distortion"])
def test_a_non_critical_dimension_does_not_reject_outright(
    aggregator: QualityAggregator, family: str
) -> None:
    """Exposure and contrast degrade recognition but do not destroy the signal.
    A dim-but-sharp photograph is still workable."""
    assessment = aggregator.aggregate(results(**{family: 0.02}), min_required=50.0)
    assert assessment.critical_failures == []
    assert assessment.usable is (assessment.overall_score >= 50.0)


@pytest.mark.unit
def test_the_critical_failure_is_explained_first(
    aggregator: QualityAggregator,
) -> None:
    assessment = aggregator.aggregate(results(blur=0.01), min_required=50.0)
    assert assessment.notes
    assert "critical floor" in assessment.notes[0]
    assert "blur" in assessment.notes[0]


@pytest.mark.unit
def test_a_dimension_exactly_at_the_floor_still_fails() -> None:
    """The floor is inclusive; a boundary value must not slip through."""
    config = QualityAggregationConfig(critical_floor=0.12)
    assessment = QualityAggregator(config).aggregate(
        results(blur=0.12), min_required=10.0
    )
    assert "blur" in assessment.critical_failures


# --------------------------------------------------------------------------- #
# Role thresholds
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_role_threshold_decides_usability(
    aggregator: QualityAggregator,
) -> None:
    """A CNIC portrait is a low-DPI print behind a laminate and is held to a
    laxer bar than a live selfie - the same pixels can pass as one and fail as
    the other."""
    scores = dict.fromkeys(ALL_FAMILIES, 0.42)

    lax = aggregator.aggregate(results(**scores), min_required=25.0)
    strict = aggregator.aggregate(results(**scores), min_required=55.0)

    assert lax.overall_score == strict.overall_score
    assert lax.usable is True
    assert strict.usable is False


@pytest.mark.unit
def test_role_thresholds_are_ordered_as_configured() -> None:
    """Selfie strictest, CNIC portrait laxest."""
    config = get_settings().quality
    assert (
        config.min_overall_for("live_selfie")
        > config.min_overall_for("profile_image")
        > config.min_overall_for("secondary_image")
        > config.min_overall_for("cnic_image")
        > config.min_overall_for("cnic_portrait")
    )


@pytest.mark.unit
def test_an_unknown_role_falls_back_to_the_default() -> None:
    config = get_settings().quality
    assert config.min_overall_for("not_a_real_role") == config.default_min_overall


# --------------------------------------------------------------------------- #
# Unmeasured dimensions
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_unmeasured_dimensions_are_excluded_not_scored_zero(
    aggregator: QualityAggregator,
) -> None:
    """Scoring an unmeasured dimension zero would punish an image for a
    measurement nobody took."""
    partial = [
        MetricResult(name=name, score=1.0)
        for name in ALL_FAMILIES
        if name != "sharpness"
    ]
    partial.append(MetricResult.unmeasured("sharpness", "no face detected"))

    assessment = aggregator.aggregate(partial, min_required=50.0)

    assert assessment.overall_score == pytest.approx(100.0)
    assert assessment.unmeasured == ["sharpness"]
    assert "sharpness" not in assessment.component_scores


@pytest.mark.unit
def test_remaining_weights_renormalise_to_one(
    aggregator: QualityAggregator,
) -> None:
    partial = [
        MetricResult(name=name, score=0.6)
        for name in ALL_FAMILIES
        if name not in {"sharpness", "distortion"}
    ]
    partial.append(MetricResult.unmeasured("sharpness", "no face"))
    partial.append(MetricResult.unmeasured("distortion", "no landmarks"))

    assessment = aggregator.aggregate(partial, min_required=50.0)

    assert sum(assessment.effective_weights.values()) == pytest.approx(1.0)
    assert set(assessment.effective_weights) == set(assessment.component_scores)


@pytest.mark.unit
def test_an_unmeasured_critical_dimension_cannot_trigger_a_failure(
    aggregator: QualityAggregator,
) -> None:
    """It was not measured, so it cannot have failed."""
    partial = [
        MetricResult(name=name, score=1.0) for name in ALL_FAMILIES if name != "blur"
    ]
    partial.append(MetricResult.unmeasured("blur", "image has no pixels"))

    assessment = aggregator.aggregate(partial, min_required=50.0)
    assert assessment.critical_failures == []


@pytest.mark.unit
def test_nothing_measurable_is_reported_honestly(
    aggregator: QualityAggregator,
) -> None:
    nothing = [MetricResult.unmeasured(name, "no pixels") for name in ALL_FAMILIES]
    assessment = aggregator.aggregate(nothing, min_required=0.0)

    assert assessment.overall_score == 0.0
    assert assessment.usable is False
    assert assessment.notes


@pytest.mark.unit
def test_a_warning_is_raised_for_every_unmeasured_dimension(
    aggregator: QualityAggregator,
) -> None:
    partial = [
        MetricResult(name=name, score=0.9)
        for name in ALL_FAMILIES
        if name != "sharpness"
    ]
    partial.append(MetricResult.unmeasured("sharpness", "no face"))
    assessment = aggregator.aggregate(partial, min_required=50.0)
    assert "sharpness" in assessment.unmeasured


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_limiting_factor_names_the_costliest_dimension(
    aggregator: QualityAggregator,
) -> None:
    assessment = aggregator.aggregate(results(brightness=0.3), min_required=50.0)
    assert assessment.limiting_factor == "brightness"


@pytest.mark.unit
def test_limiting_factor_weighs_importance_not_just_lowness(
    aggregator: QualityAggregator,
) -> None:
    """`distortion` carries 4% of the weight and `blur` 22%, so a slightly
    weak blur costs the composite more than a badly weak distortion."""
    assessment = aggregator.aggregate(
        results(distortion=0.55, blur=0.60), min_required=50.0
    )
    assert assessment.limiting_factor == "blur"


@pytest.mark.unit
def test_score_for_returns_the_percentage_scale(
    aggregator: QualityAggregator,
) -> None:
    assessment = aggregator.aggregate(results(noise=0.5), min_required=50.0)
    assert assessment.score_for("noise") == pytest.approx(50.0)
    assert assessment.score_for("not_a_family") == 0.0


@pytest.mark.unit
def test_measurement_lookup_survives_a_missing_family(
    aggregator: QualityAggregator,
) -> None:
    assessment = aggregator.aggregate(results(), min_required=50.0)
    assert assessment.measurement("nope", "whatever") is None


@pytest.mark.unit
def test_analyser_notes_are_carried_into_the_assessment(
    aggregator: QualityAggregator,
) -> None:
    annotated = results()
    annotated[0] = MetricResult(
        name="blur", score=0.9, note="slightly soft but acceptable"
    )
    assessment = aggregator.aggregate(annotated, min_required=50.0)
    assert any("slightly soft" in note for note in assessment.notes)


@pytest.mark.unit
def test_summary_is_pii_free_and_serialisable(
    aggregator: QualityAggregator,
) -> None:
    assessment = aggregator.aggregate(results(blur=0.3), min_required=50.0)
    summary = assessment.summary()

    assert set(summary) == {
        "overall",
        "usable",
        "limiting",
        "min_required",
        "critical_failures",
        "unmeasured",
    }
    import json

    json.dumps(summary)


# --------------------------------------------------------------------------- #
# Configuration validation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_aggregation_weights_are_normalised_on_load() -> None:
    """An operator raising one weight must not have to rebalance the rest."""
    config = QualityAggregationConfig(
        weights={name: 2.0 for name in ALL_FAMILIES}
    )
    assert sum(config.weights.values()) == pytest.approx(1.0)


@pytest.mark.unit
def test_a_missing_weight_is_rejected() -> None:
    partial = {name: 0.125 for name in ALL_FAMILIES if name != "noise"}
    with pytest.raises(ValueError, match="missing keys"):
        QualityAggregationConfig(weights=partial)


@pytest.mark.unit
def test_an_unknown_weight_is_rejected() -> None:
    extra = {name: 0.1 for name in ALL_FAMILIES}
    extra["not_a_dimension"] = 0.2
    with pytest.raises(ValueError, match="unknown keys"):
        QualityAggregationConfig(weights=extra)


@pytest.mark.unit
def test_an_unknown_critical_component_is_rejected() -> None:
    with pytest.raises(ValueError, match="critical_components"):
        QualityAggregationConfig(critical_components=["blur", "nonsense"])


@pytest.mark.unit
def test_the_power_exponent_is_bounded() -> None:
    with pytest.raises(ValueError):
        QualityAggregationConfig(power=0.0)
    with pytest.raises(ValueError):
        QualityAggregationConfig(power=5.0)


@pytest.mark.unit
def test_the_configured_exponent_is_below_one() -> None:
    """If this ever becomes 1.0 the composite silently reverts to an arithmetic
    mean and a single fatal dimension stops being visible."""
    assert get_settings().quality.aggregation.power < 1.0


@pytest.mark.unit
def test_assessment_is_a_plain_dataclass() -> None:
    """No hidden state - the aggregator returns data, not a live object."""
    assessment = QualityAssessment(overall_score=42.0, usable=False)
    assert assessment.overall_score == 42.0
    assert assessment.metrics == {}
