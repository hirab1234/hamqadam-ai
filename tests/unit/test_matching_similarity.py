"""Score calibration and the decision boundaries.

Every reported match score passes through :func:`calibrate_score`, so its
behaviour at the anchors, its monotonicity and its round-trip are worth pinning
exactly. A drift here silently shifts every verification decision the service
makes.
"""

from __future__ import annotations

import pytest

from hamqadam_ai.core.config import MatchThresholds, ScoreCalibration, get_settings
from hamqadam_ai.core.constants import MatchDecision
from hamqadam_ai.matching.similarity import (
    calibrate_score,
    decide,
    score_for_decision,
    uncalibrate_score,
)


@pytest.fixture
def thresholds() -> MatchThresholds:
    """The configured selfie-vs-profile operating points."""
    return MatchThresholds(strong_match=0.62, review=0.45)


@pytest.fixture
def calibration() -> ScoreCalibration:
    """The default anchor scores."""
    return ScoreCalibration()


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("similarity", "expected"),
    [
        (1.0, MatchDecision.STRONG_MATCH),
        (0.90, MatchDecision.STRONG_MATCH),
        (0.62, MatchDecision.STRONG_MATCH),
        (0.6199, MatchDecision.REVIEW),
        (0.50, MatchDecision.REVIEW),
        (0.45, MatchDecision.REVIEW),
        (0.4499, MatchDecision.FAILED),
        (0.0, MatchDecision.FAILED),
        (-0.5, MatchDecision.FAILED),
    ],
)
def test_decision_bands(
    similarity: float, expected: MatchDecision, thresholds: MatchThresholds
) -> None:
    assert decide(similarity, thresholds) is expected


@pytest.mark.unit
def test_the_thresholds_are_inclusive(thresholds: MatchThresholds) -> None:
    """A score exactly at the boundary passes it, matching the confusion
    matrix in the evaluation harness."""
    assert decide(0.62, thresholds) is MatchDecision.STRONG_MATCH
    assert decide(0.45, thresholds) is MatchDecision.REVIEW


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_anchors_map_exactly(
    thresholds: MatchThresholds, calibration: ScoreCalibration
) -> None:
    assert calibrate_score(0.0, thresholds, calibration) == pytest.approx(0.0)
    assert calibrate_score(0.45, thresholds, calibration) == pytest.approx(50.0)
    assert calibrate_score(0.62, thresholds, calibration) == pytest.approx(75.0)
    assert calibrate_score(1.0, thresholds, calibration) == pytest.approx(100.0)


@pytest.mark.unit
def test_an_impostor_does_not_report_fifty(
    thresholds: MatchThresholds, calibration: ScoreCalibration
) -> None:
    """The bug the calibration exists to prevent.

    A naive (cos + 1) / 2 * 100 rescaling reports 50 for a confident non-match,
    which reads as "borderline" when it means "certainly a different person".
    The measured impostor pair in Module 3 scored 0.011.
    """
    naive = (0.011 + 1.0) / 2.0 * 100.0
    assert naive == pytest.approx(50.55, abs=0.1)

    calibrated = calibrate_score(0.011, thresholds, calibration)
    assert calibrated < 2.0


@pytest.mark.unit
def test_a_measured_genuine_pair_reports_a_high_score(
    thresholds: MatchThresholds, calibration: ScoreCalibration
) -> None:
    """Module 3 measured genuine self-similarity at 0.898-0.977."""
    assert calibrate_score(0.898, thresholds, calibration) > 90.0
    assert calibrate_score(0.977, thresholds, calibration) > 97.0


@pytest.mark.unit
def test_calibration_is_monotonic(
    thresholds: MatchThresholds, calibration: ScoreCalibration
) -> None:
    values = [
        calibrate_score(similarity / 100.0, thresholds, calibration)
        for similarity in range(-50, 101)
    ]
    assert values == sorted(values)


@pytest.mark.unit
def test_calibration_stays_in_range(
    thresholds: MatchThresholds, calibration: ScoreCalibration
) -> None:
    for similarity in (-1.0, -0.3, 0.0, 0.2, 0.45, 0.62, 0.8, 1.0, 1.5):
        score = calibrate_score(similarity, thresholds, calibration)
        assert 0.0 <= score <= 100.0


@pytest.mark.unit
def test_below_the_floor_scores_zero(
    thresholds: MatchThresholds, calibration: ScoreCalibration
) -> None:
    """Negative cosine between face embeddings carries no more information
    than orthogonality, so the scale does not extend below zero."""
    assert calibrate_score(-0.4, thresholds, calibration) == 0.0
    assert calibrate_score(-1.0, thresholds, calibration) == 0.0


@pytest.mark.unit
def test_the_segment_midpoints_interpolate(
    thresholds: MatchThresholds, calibration: ScoreCalibration
) -> None:
    lower_mid = calibrate_score((0.0 + 0.45) / 2, thresholds, calibration)
    middle_mid = calibrate_score((0.45 + 0.62) / 2, thresholds, calibration)
    upper_mid = calibrate_score((0.62 + 1.0) / 2, thresholds, calibration)

    assert lower_mid == pytest.approx(25.0)
    assert middle_mid == pytest.approx(62.5)
    assert upper_mid == pytest.approx(87.5)


# --------------------------------------------------------------------------- #
# Commensurability across comparison types
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_different_comparison_types_produce_comparable_scores(
    calibration: ScoreCalibration,
) -> None:
    """The property that makes the weighted identity fusion defensible.

    A CNIC score of 75 and a profile score of 75 must mean the same thing about
    the strength of the evidence, even though they came from very different
    cosine values (0.42 against 0.62).
    """
    profile = MatchThresholds(strong_match=0.62, review=0.45)
    cnic = MatchThresholds(strong_match=0.42, review=0.30)

    assert calibrate_score(0.62, profile, calibration) == pytest.approx(
        calibrate_score(0.42, cnic, calibration)
    )
    assert calibrate_score(0.45, profile, calibration) == pytest.approx(
        calibrate_score(0.30, cnic, calibration)
    )


@pytest.mark.unit
def test_a_score_is_self_describing(calibration: ScoreCalibration) -> None:
    """75+ strong, 50-75 review, below 50 failed - whatever produced it."""
    for thresholds in (
        MatchThresholds(strong_match=0.62, review=0.45),
        MatchThresholds(strong_match=0.42, review=0.30),
        MatchThresholds(strong_match=0.80, review=0.70),
    ):
        for similarity in [i / 100.0 for i in range(0, 101)]:
            score = calibrate_score(similarity, thresholds, calibration)
            decision = decide(similarity, thresholds)

            if decision is MatchDecision.STRONG_MATCH:
                assert score >= calibration.strong_match_score - 1e-9
            elif decision is MatchDecision.REVIEW:
                assert (
                    calibration.review_score - 1e-9
                    <= score
                    < calibration.strong_match_score + 1e-9
                )
            else:
                assert score < calibration.review_score + 1e-9


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "similarity", [0.05, 0.2, 0.35, 0.45, 0.5, 0.62, 0.7, 0.85, 0.95]
)
def test_calibration_round_trips(
    similarity: float, thresholds: MatchThresholds, calibration: ScoreCalibration
) -> None:
    """The evaluation harness moves back into cosine space to compute an ROC,
    so the inverse must be exact."""
    score = calibrate_score(similarity, thresholds, calibration)
    recovered = uncalibrate_score(score, thresholds, calibration)
    assert recovered == pytest.approx(similarity, abs=1e-6)


@pytest.mark.unit
def test_the_inverse_handles_the_extremes(
    thresholds: MatchThresholds, calibration: ScoreCalibration
) -> None:
    assert uncalibrate_score(0.0, thresholds, calibration) == pytest.approx(0.0)
    assert uncalibrate_score(100.0, thresholds, calibration) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Configuration invariants
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_review_must_sit_below_strong_match() -> None:
    with pytest.raises(ValueError, match="strictly below"):
        MatchThresholds(strong_match=0.4, review=0.6)


@pytest.mark.unit
def test_the_review_anchor_must_sit_below_the_strong_anchor() -> None:
    with pytest.raises(ValueError, match="below strong_match_score"):
        ScoreCalibration(strong_match_score=50.0, review_score=75.0)


@pytest.mark.unit
def test_the_cnic_operating_point_is_the_laxest() -> None:
    """A CNIC portrait is a sub-300-dpi print behind a laminate. Holding it to
    the selfie bar would reject a large share of genuine documents."""
    matching = get_settings().matching
    assert (
        matching.selfie_vs_cnic.strong_match
        < matching.selfie_vs_secondary.strong_match
        <= matching.selfie_vs_profile.strong_match
    )


@pytest.mark.unit
def test_score_for_decision_returns_the_band_boundaries(
    calibration: ScoreCalibration,
) -> None:
    assert score_for_decision(MatchDecision.STRONG_MATCH, calibration) == 75.0
    assert score_for_decision(MatchDecision.REVIEW, calibration) == 50.0
    assert score_for_decision(MatchDecision.FAILED, calibration) == 0.0
    assert score_for_decision(MatchDecision.NOT_COMPARED, calibration) == 0.0
