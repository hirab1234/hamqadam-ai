"""Unit tests for the composite visibility scorer.

Two properties matter most and are tested directly:

* **Weight redistribution.** When pose or occlusion cannot be measured, their
  weight must be spread over the measured components - not silently treated as
  a perfect 1.0, which would flatter a face nobody examined.
* **Limiting factor.** The reported bottleneck must be the component with the
  largest *weighted* deficit, since that is what the user is told to fix.
"""

from __future__ import annotations

import pytest

from hamqadam_ai.core.config import VisibilityConfig
from hamqadam_ai.detectors.visibility import VisibilityScorer


@pytest.fixture
def scorer() -> VisibilityScorer:
    """Scorer with the default configured weights."""
    return VisibilityScorer(VisibilityConfig())


@pytest.mark.unit
def test_perfect_face_scores_one(scorer: VisibilityScorer) -> None:
    """Every component at its best must produce a score of exactly 1.0."""
    result = scorer.score(
        detector_confidence=1.0,
        occlusion_score=0.0,
        pose_deviation=0.0,
        face_area_ratio=0.20,
        truncation_ratio=0.0,
    )

    assert result.score == pytest.approx(1.0)
    assert result.acceptable
    assert result.missing == ()


@pytest.mark.unit
def test_worst_face_scores_zero(scorer: VisibilityScorer) -> None:
    """Every component at its worst must produce a score of exactly 0.0."""
    result = scorer.score(
        detector_confidence=0.0,
        occlusion_score=1.0,
        pose_deviation=1.0,
        face_area_ratio=0.001,
        truncation_ratio=1.0,
    )

    assert result.score == pytest.approx(0.0)
    assert not result.acceptable


@pytest.mark.unit
def test_weights_are_applied(scorer: VisibilityScorer) -> None:
    """The composite must equal the configured weighted sum."""
    weights = VisibilityConfig().weights
    result = scorer.score(
        detector_confidence=0.80,
        occlusion_score=0.25,  # -> 0.75
        pose_deviation=0.40,  # -> 0.60
        face_area_ratio=0.20,  # -> 1.00 (inside the ideal band)
        truncation_ratio=0.10,  # -> 0.90
    )

    expected = (
        weights["detector_confidence"] * 0.80
        + weights["occlusion"] * 0.75
        + weights["pose"] * 0.60
        + weights["face_size"] * 1.00
        + weights["framing"] * 0.90
    )
    assert result.score == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------- #
# Face-size plateau
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("ratio", [0.06, 0.10, 0.25, 0.45])
def test_ideal_band_scores_full_marks(scorer: VisibilityScorer, ratio: float) -> None:
    """Anywhere inside the ideal band is equally good."""
    result = scorer.score(
        detector_confidence=1.0,
        occlusion_score=0.0,
        pose_deviation=0.0,
        face_area_ratio=ratio,
        truncation_ratio=0.0,
    )

    assert result.components["face_size"] == pytest.approx(1.0)


@pytest.mark.unit
def test_tiny_face_scores_zero_size(scorer: VisibilityScorer) -> None:
    """Below the floor the size component collapses."""
    result = scorer.score(
        detector_confidence=1.0,
        occlusion_score=0.0,
        pose_deviation=0.0,
        face_area_ratio=0.002,
        truncation_ratio=0.0,
    )

    assert result.components["face_size"] == pytest.approx(0.0)
    assert result.limiting_factor == "face_size"


@pytest.mark.unit
def test_face_filling_the_frame_is_penalised(scorer: VisibilityScorer) -> None:
    """An over-large face usually means a clipped head or a photo of a photo."""
    result = scorer.score(
        detector_confidence=1.0,
        occlusion_score=0.0,
        pose_deviation=0.0,
        face_area_ratio=0.97,
        truncation_ratio=0.0,
    )

    assert result.components["face_size"] == pytest.approx(0.0)


@pytest.mark.unit
def test_size_score_ramps_monotonically(scorer: VisibilityScorer) -> None:
    """Between the floor and the band the score must rise monotonically."""
    previous = -1.0
    for ratio in (0.004, 0.01, 0.02, 0.04, 0.06):
        current = scorer._size_score(ratio)  # noqa: SLF001
        assert current >= previous
        previous = current


# --------------------------------------------------------------------------- #
# Weight redistribution
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_missing_components_are_reported(scorer: VisibilityScorer) -> None:
    """A landmark-free detector produces neither pose nor occlusion."""
    result = scorer.score(
        detector_confidence=0.9,
        occlusion_score=None,
        pose_deviation=None,
        face_area_ratio=0.20,
        truncation_ratio=0.0,
    )

    assert set(result.missing) == {"occlusion", "pose"}
    assert "occlusion" not in result.components
    assert "pose" not in result.components


@pytest.mark.unit
def test_redistributed_weights_still_sum_to_one(scorer: VisibilityScorer) -> None:
    """Redistribution must preserve the total weight, or the scale shifts."""
    result = scorer.score(
        detector_confidence=0.9,
        occlusion_score=None,
        pose_deviation=None,
        face_area_ratio=0.20,
        truncation_ratio=0.0,
    )

    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(result.weights) == set(result.components)


@pytest.mark.unit
def test_redistribution_preserves_relative_importance(
    scorer: VisibilityScorer,
) -> None:
    """Spreading forfeited weight must not reorder the surviving components."""
    base = VisibilityConfig().weights
    result = scorer.score(
        detector_confidence=0.9,
        occlusion_score=None,
        pose_deviation=None,
        face_area_ratio=0.20,
        truncation_ratio=0.0,
    )

    expected_ratio = base["detector_confidence"] / base["face_size"]
    actual_ratio = result.weights["detector_confidence"] / result.weights["face_size"]
    assert actual_ratio == pytest.approx(expected_ratio)


@pytest.mark.unit
def test_missing_components_do_not_inflate_the_score(
    scorer: VisibilityScorer,
) -> None:
    """A face with unmeasured pose must not outscore one measured as perfect.

    Treating a missing component as 1.0 would make a Haar detection look better
    than an SCRFD detection of the same face, which is backwards.
    """
    measured = scorer.score(
        detector_confidence=0.7,
        occlusion_score=0.0,
        pose_deviation=0.0,
        face_area_ratio=0.20,
        truncation_ratio=0.0,
    )
    unmeasured = scorer.score(
        detector_confidence=0.7,
        occlusion_score=None,
        pose_deviation=None,
        face_area_ratio=0.20,
        truncation_ratio=0.0,
    )

    assert unmeasured.score <= measured.score + 1e-9


# --------------------------------------------------------------------------- #
# Limiting factor
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "detector_confidence": 1.0,
                "occlusion_score": 0.9,
                "pose_deviation": 0.0,
                "face_area_ratio": 0.2,
                "truncation_ratio": 0.0,
            },
            "occlusion",
        ),
        (
            {
                "detector_confidence": 1.0,
                "occlusion_score": 0.0,
                "pose_deviation": 0.95,
                "face_area_ratio": 0.2,
                "truncation_ratio": 0.0,
            },
            "pose",
        ),
        (
            {
                "detector_confidence": 1.0,
                "occlusion_score": 0.0,
                "pose_deviation": 0.0,
                "face_area_ratio": 0.2,
                "truncation_ratio": 0.9,
            },
            "framing",
        ),
        (
            {
                "detector_confidence": 0.1,
                "occlusion_score": 0.0,
                "pose_deviation": 0.0,
                "face_area_ratio": 0.2,
                "truncation_ratio": 0.0,
            },
            "detector_confidence",
        ),
    ],
)
def test_limiting_factor_identifies_the_bottleneck(
    scorer: VisibilityScorer, kwargs: dict[str, float], expected: str
) -> None:
    """The reported bottleneck is what the user is told to fix; it must be right."""
    assert scorer.score(**kwargs).limiting_factor == expected


@pytest.mark.unit
def test_acceptable_tracks_the_configured_minimum() -> None:
    """The acceptability flag must follow the configured floor, not a constant."""
    strict = VisibilityScorer(VisibilityConfig(min_acceptable=0.95))
    lenient = VisibilityScorer(VisibilityConfig(min_acceptable=0.10))

    kwargs = {
        "detector_confidence": 0.7,
        "occlusion_score": 0.2,
        "pose_deviation": 0.2,
        "face_area_ratio": 0.15,
        "truncation_ratio": 0.05,
    }

    assert not strict.score(**kwargs).acceptable
    assert lenient.score(**kwargs).acceptable
