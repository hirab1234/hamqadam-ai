"""Score mapping and aggregation arithmetic.

These are the functions every quality number passes through, so their edge
cases are worth pinning precisely: an off-by-one at an anchor, or a power mean
that silently behaves like an arithmetic one, changes every verification
decision the service makes.
"""

from __future__ import annotations

import math

import pytest

from hamqadam_ai.core.config import BandMapping, RampMapping
from hamqadam_ai.quality.scoring import (
    band_score,
    limiting_component,
    ramp_score,
    smoothstep,
    weighted_mean,
    weighted_power_mean,
)

# --------------------------------------------------------------------------- #
# smoothstep
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0, 0.0), (0.0, 0.0), (0.25, 0.15625), (0.5, 0.5), (0.75, 0.84375), (1.0, 1.0), (2.0, 1.0)],
)
def test_smoothstep_values(value: float, expected: float) -> None:
    assert smoothstep(value) == pytest.approx(expected)


@pytest.mark.unit
def test_smoothstep_is_monotone() -> None:
    values = [smoothstep(t / 50.0) for t in range(51)]
    assert values == sorted(values)


@pytest.mark.unit
def test_smoothstep_has_zero_gradient_at_the_anchors() -> None:
    """Flat ends are the point: they stop the composite jittering on sensor
    noise when a measurement happens to sit near an anchor."""
    delta = 1e-4
    assert smoothstep(delta) / delta < 0.01
    assert (1.0 - smoothstep(1.0 - delta)) / delta < 0.01


# --------------------------------------------------------------------------- #
# ramp_score - linear
# --------------------------------------------------------------------------- #


@pytest.fixture
def higher_better() -> RampMapping:
    return RampMapping(floor=10.0, good=110.0)


@pytest.fixture
def lower_better() -> RampMapping:
    return RampMapping(floor=15.0, good=1.5)


@pytest.mark.unit
def test_ramp_anchors_map_to_the_extremes(higher_better: RampMapping) -> None:
    assert ramp_score(10.0, higher_better) == 0.0
    assert ramp_score(110.0, higher_better) == 1.0


@pytest.mark.unit
def test_ramp_midpoint_is_a_half(higher_better: RampMapping) -> None:
    assert ramp_score(60.0, higher_better) == pytest.approx(0.5)


@pytest.mark.unit
def test_ramp_clamps_outside_the_anchors(higher_better: RampMapping) -> None:
    assert ramp_score(-500.0, higher_better) == 0.0
    assert ramp_score(1e9, higher_better) == 1.0


@pytest.mark.unit
def test_lower_is_better_inverts_automatically(lower_better: RampMapping) -> None:
    assert lower_better.lower_is_better is True
    assert ramp_score(1.5, lower_better) == 1.0
    assert ramp_score(15.0, lower_better) == 0.0
    assert ramp_score(0.1, lower_better) == 1.0
    assert ramp_score(100.0, lower_better) == 0.0


@pytest.mark.unit
def test_lower_is_better_is_monotone_decreasing(lower_better: RampMapping) -> None:
    values = [ramp_score(v, lower_better) for v in (1.0, 3.0, 6.0, 9.0, 12.0, 16.0)]
    assert values == sorted(values, reverse=True)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_measurements_score_zero(
    bad: float, higher_better: RampMapping
) -> None:
    """The safe direction: an unmeasurable image can be rejected but never
    approved on the strength of a measurement that did not happen."""
    assert ramp_score(bad, higher_better) == 0.0


# --------------------------------------------------------------------------- #
# ramp_score - logarithmic
# --------------------------------------------------------------------------- #


@pytest.fixture
def log_mapping() -> RampMapping:
    return RampMapping(floor=40.0, good=4000.0, log_scale=True)


@pytest.mark.unit
def test_log_ramp_anchors_map_to_the_extremes(log_mapping: RampMapping) -> None:
    assert ramp_score(40.0, log_mapping) == 0.0
    assert ramp_score(4000.0, log_mapping) == 1.0


@pytest.mark.unit
def test_log_ramp_midpoint_is_the_geometric_mean(log_mapping: RampMapping) -> None:
    """Half-way in log space is sqrt(floor * good), not (floor + good) / 2."""
    geometric = math.sqrt(40.0 * 4000.0)
    assert ramp_score(geometric, log_mapping) == pytest.approx(0.5)


@pytest.mark.unit
def test_log_ramp_spreads_a_wide_dynamic_range(log_mapping: RampMapping) -> None:
    """The reason log scale exists.

    Measured Laplacian variances on the reference portrait span 37 (unusable)
    to 10014 (pristine). A linear ramp over that range assigns everything below
    ~1000 a score under 0.25 and cannot rank the degradations at all.
    """
    measured = {"gauss21": 37.0, "gauss9": 275.0, "gauss3": 2150.0, "pristine": 10014.0}
    scores = {name: ramp_score(v, log_mapping) for name, v in measured.items()}

    assert scores["gauss21"] == 0.0
    assert 0.2 < scores["gauss9"] < 0.6
    assert 0.85 < scores["gauss3"] < 1.0
    assert scores["pristine"] == 1.0

    linear = RampMapping(floor=40.0, good=4000.0)
    linear_scores = {name: ramp_score(v, linear) for name, v in measured.items()}
    # Under a linear ramp the two worst cases are both crushed against zero and
    # cannot be told apart, which is exactly the failure log scale removes.
    assert linear_scores["gauss21"] < 0.02
    assert linear_scores["gauss9"] < 0.02
    assert scores["gauss9"] - scores["gauss21"] > 0.2


@pytest.mark.unit
def test_log_ramp_handles_zero_for_higher_is_better(log_mapping: RampMapping) -> None:
    """Zero is off the bottom of a log scale; for higher-is-better that is the
    worst possible reading."""
    assert ramp_score(0.0, log_mapping) == 0.0
    assert ramp_score(-5.0, log_mapping) == 0.0


@pytest.mark.unit
def test_log_ramp_handles_zero_for_lower_is_better() -> None:
    """...and for lower-is-better it is the best possible reading."""
    mapping = RampMapping(floor=1.30, good=0.20, log_scale=True)
    assert mapping.lower_is_better is True
    assert ramp_score(0.0, mapping) == 1.0


@pytest.mark.unit
def test_log_ramp_rejects_non_positive_anchors() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        RampMapping(floor=0.0, good=10.0, log_scale=True)


@pytest.mark.unit
def test_ramp_rejects_identical_anchors() -> None:
    with pytest.raises(ValueError, match="must differ"):
        RampMapping(floor=5.0, good=5.0)


# --------------------------------------------------------------------------- #
# band_score
# --------------------------------------------------------------------------- #


@pytest.fixture
def band() -> BandMapping:
    return BandMapping(
        min_acceptable=30.0, ideal_low=95.0, ideal_high=180.0, max_acceptable=230.0
    )


@pytest.mark.unit
@pytest.mark.parametrize("value", [95.0, 120.0, 137.5, 180.0])
def test_the_ideal_plateau_scores_one(band: BandMapping, value: float) -> None:
    """A photograph at mean luminance 120 and one at 160 are both simply well
    exposed; ramping to a single ideal point would penalise one for nothing."""
    assert band_score(value, band) == 1.0


@pytest.mark.unit
@pytest.mark.parametrize("value", [0.0, 30.0, 230.0, 255.0])
def test_outside_the_band_scores_zero(band: BandMapping, value: float) -> None:
    assert band_score(value, band) == 0.0


@pytest.mark.unit
def test_shoulders_interpolate(band: BandMapping) -> None:
    low_mid = band_score((30.0 + 95.0) / 2, band)
    high_mid = band_score((180.0 + 230.0) / 2, band)
    assert low_mid == pytest.approx(0.5)
    assert high_mid == pytest.approx(0.5)


@pytest.mark.unit
def test_band_is_monotone_on_each_shoulder(band: BandMapping) -> None:
    rising = [band_score(v, band) for v in (30, 45, 60, 75, 90, 95)]
    falling = [band_score(v, band) for v in (180, 195, 205, 215, 225, 230)]
    assert rising == sorted(rising)
    assert falling == sorted(falling, reverse=True)


@pytest.mark.unit
def test_band_rejects_unordered_bounds() -> None:
    with pytest.raises(ValueError, match="min_acceptable <= ideal_low"):
        BandMapping(
            min_acceptable=100.0, ideal_low=50.0, ideal_high=180.0, max_acceptable=230.0
        )


@pytest.mark.unit
def test_band_handles_a_degenerate_plateau() -> None:
    """A zero-width shoulder must not divide by zero."""
    mapping = BandMapping(
        min_acceptable=50.0, ideal_low=50.0, ideal_high=200.0, max_acceptable=200.0
    )
    assert band_score(120.0, mapping) == 1.0
    assert band_score(20.0, mapping) == 0.0


@pytest.mark.unit
def test_non_finite_band_measurement_scores_zero(band: BandMapping) -> None:
    assert band_score(float("nan"), band) == 0.0


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_weighted_mean_respects_weights() -> None:
    scores = {"a": 1.0, "b": 0.0}
    assert weighted_mean(scores, {"a": 0.75, "b": 0.25}) == pytest.approx(0.75)


@pytest.mark.unit
def test_weighted_mean_renormalises_over_present_components() -> None:
    """A component that was not measured shrinks the denominator rather than
    silently contributing zero."""
    scores = {"a": 0.8}
    assert weighted_mean(scores, {"a": 0.2, "b": 0.8}) == pytest.approx(0.8)


@pytest.mark.unit
def test_weighted_mean_of_nothing_is_zero() -> None:
    assert weighted_mean({}, {"a": 1.0}) == 0.0


@pytest.mark.unit
def test_power_mean_with_exponent_one_is_the_arithmetic_mean() -> None:
    scores = {"a": 0.9, "b": 0.4, "c": 0.7}
    weights = {"a": 0.5, "b": 0.2, "c": 0.3}
    assert weighted_power_mean(scores, weights, 1.0) == pytest.approx(
        weighted_mean(scores, weights)
    )


@pytest.mark.unit
def test_power_mean_below_one_penalises_the_worst_component() -> None:
    """The whole reason the composite does not use an arithmetic mean.

    Five perfect dimensions and one catastrophic one must not average out to a
    passing grade.
    """
    scores = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 0.0}
    weights = dict.fromkeys(scores, 0.2)

    arithmetic = weighted_mean(scores, weights)
    power = weighted_power_mean(scores, weights, 0.5)

    assert arithmetic == pytest.approx(0.8)
    assert power == pytest.approx(0.64)
    assert power < arithmetic


@pytest.mark.unit
def test_power_mean_equals_the_mean_when_components_are_equal() -> None:
    scores = dict.fromkeys("abcd", 0.6)
    weights = dict.fromkeys("abcd", 0.25)
    assert weighted_power_mean(scores, weights, 0.5) == pytest.approx(0.6)


@pytest.mark.unit
def test_power_mean_is_bounded_by_its_components() -> None:
    scores = {"a": 0.3, "b": 0.55, "c": 0.9}
    weights = {"a": 0.3, "b": 0.3, "c": 0.4}
    result = weighted_power_mean(scores, weights, 0.5)
    assert 0.3 <= result <= 0.9


@pytest.mark.unit
def test_power_mean_penalty_deepens_as_the_exponent_falls() -> None:
    scores = {"a": 1.0, "b": 1.0, "c": 0.1}
    weights = dict.fromkeys(scores, 1 / 3)
    results = [weighted_power_mean(scores, weights, p) for p in (1.0, 0.7, 0.5, 0.3)]
    assert results == sorted(results, reverse=True)


@pytest.mark.unit
def test_power_mean_clamps_out_of_range_components() -> None:
    scores = {"a": 1.5, "b": -0.4}
    weights = {"a": 0.5, "b": 0.5}
    assert 0.0 <= weighted_power_mean(scores, weights, 0.5) <= 1.0


@pytest.mark.unit
def test_power_mean_rejects_a_non_positive_exponent() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        weighted_power_mean({"a": 0.5}, {"a": 1.0}, 0.0)


@pytest.mark.unit
def test_limiting_component_finds_the_largest_weighted_deficit() -> None:
    """Not the lowest score - the one costing the composite the most."""
    scores = {"tiny_weight": 0.0, "big_weight": 0.5}
    weights = {"tiny_weight": 0.02, "big_weight": 0.98}
    assert limiting_component(scores, weights) == "big_weight"


@pytest.mark.unit
def test_limiting_component_of_nothing_is_none() -> None:
    assert limiting_component({}, {"a": 1.0}) is None
