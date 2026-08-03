"""Turning detector signals into one verdict.

Pure logic - no images, no detectors, no models. The signals are built by
hand, which is the point: the aggregation rules can then be pinned against
combinations no fixture happens to produce, including the ones that only occur
when something has gone wrong.
"""

from __future__ import annotations

import json

import pytest

from hamqadam_ai.authenticity.aggregate import (
    FINDING_CODES,
    FINDING_MESSAGES,
    aggregate,
)
from hamqadam_ai.authenticity.base import AuthenticitySignal


def signal(
    name: str,
    *,
    triggered: bool = False,
    confidence: float = 0.0,
    note: str | None = None,
) -> AuthenticitySignal:
    """One detector's reading."""
    return AuthenticitySignal(
        name=name,
        triggered=triggered,
        confidence=confidence,
        measurements={"probe": 1.0},
        note=note,
    )


SILENT = [
    signal("screenshot"),
    signal("screen_recapture"),
    signal("print_recapture"),
    signal("synthetic_image"),
]


# --------------------------------------------------------------------------- #
# Nothing found
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_no_findings_scores_full_marks() -> None:
    assessment = aggregate(SILENT)

    assert assessment.score == pytest.approx(100.0)
    assert assessment.clean is True
    assert assessment.strongest is None


@pytest.mark.unit
def test_a_clean_score_is_named_clean_not_genuine() -> None:
    """Deliberate. Nothing in this package can establish that an image is a
    genuine capture - only that four specific tests found nothing. A property
    called ``genuine`` would invite exactly the inference the module spends
    its docstrings warning against."""
    assessment = aggregate(SILENT)

    assert hasattr(assessment, "clean")
    assert not hasattr(assessment, "genuine")


@pytest.mark.unit
def test_no_signals_at_all_still_scores_full_marks() -> None:
    """Degenerate but reachable: every detector could be unavailable."""
    assert aggregate([]).score == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# One finding
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_one_confident_finding_drives_the_score_to_zero() -> None:
    signals = [signal("screenshot", triggered=True, confidence=1.0), *SILENT[1:]]
    assessment = aggregate(signals)

    assert assessment.score == pytest.approx(0.0)
    assert assessment.clean is False


@pytest.mark.unit
def test_a_marginal_finding_scores_proportionally() -> None:
    signals = [signal("screenshot", triggered=True, confidence=0.6), *SILENT[1:]]
    assert aggregate(signals).score == pytest.approx(40.0)


@pytest.mark.unit
def test_the_strongest_finding_wins_not_the_average() -> None:
    """Each detector looks for a *different* way an image can fail, and the
    failures are alternatives rather than contributions. Averaging one strong
    positive against three correct silences would dilute the detector that
    found something into a shrug - and the silent three were not even looking
    for what it found.
    """
    signals = [signal("screen_recapture", triggered=True, confidence=0.9), *SILENT[:1],
               *SILENT[2:]]
    assessment = aggregate(signals)

    assert assessment.score == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Several findings
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_findings_are_ordered_by_evidence() -> None:
    signals = [
        signal("screenshot", triggered=True, confidence=0.6),
        signal("synthetic_image", triggered=True, confidence=0.95),
        signal("print_recapture", triggered=True, confidence=0.7),
    ]
    codes = [finding.detector for finding in aggregate(signals).findings]

    assert codes == ["synthetic_image", "print_recapture", "screenshot"]


@pytest.mark.unit
def test_a_second_finding_lowers_the_score_but_by_less() -> None:
    """Two detectors firing on one image usually means one artefact tripped
    both, not that two independent things are wrong."""
    one = aggregate([signal("screenshot", triggered=True, confidence=0.6)])
    two = aggregate(
        [
            signal("screenshot", triggered=True, confidence=0.6),
            signal("synthetic_image", triggered=True, confidence=0.6),
        ]
    )

    assert two.score < one.score
    assert (one.score - two.score) < 60.0 * 0.6


@pytest.mark.unit
def test_the_score_never_goes_below_zero() -> None:
    signals = [
        signal(name, triggered=True, confidence=1.0)
        for name in FINDING_CODES
    ]
    assert aggregate(signals).score == pytest.approx(0.0)


@pytest.mark.unit
def test_the_strongest_finding_is_exposed() -> None:
    signals = [
        signal("screenshot", triggered=True, confidence=0.6),
        signal("screen_recapture", triggered=True, confidence=0.85),
    ]
    strongest = aggregate(signals).strongest

    assert strongest is not None
    assert strongest.detector == "screen_recapture"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("detector", sorted(FINDING_CODES))
def test_every_detector_has_a_code_and_a_message(detector: str) -> None:
    """A finding with no guidance leaves the user with an accusation and no
    way to act on it."""
    assert FINDING_CODES[detector]
    assert FINDING_MESSAGES[detector]
    assert len(FINDING_MESSAGES[detector]) > 40


@pytest.mark.unit
@pytest.mark.parametrize("detector", sorted(FINDING_MESSAGES))
def test_every_message_tells_the_user_what_to_do(detector: str) -> None:
    """Each of these accuses somebody of uploading something dishonest. The
    least it can do is say what would fix it."""
    assert "Upload" in FINDING_MESSAGES[detector]


@pytest.mark.unit
def test_an_unknown_detector_still_produces_a_finding() -> None:
    """A detector added without registering a message must not silently vanish
    from the report."""
    assessment = aggregate([signal("something_new", triggered=True, confidence=0.8)])

    assert len(assessment.findings) == 1
    assert assessment.findings[0].code == "SOMETHING_NEW"
    assert assessment.findings[0].message


@pytest.mark.unit
def test_unmeasured_detectors_are_reported() -> None:
    """Silence because a detector could not run is different from silence
    because it looked and found nothing, and the caller has to be able to tell
    them apart."""
    signals = [*SILENT[:3], signal("synthetic_image", note="image too small")]
    assessment = aggregate(signals)

    assert assessment.unmeasured == ["synthetic_image: image too small"]
    assert assessment.score == pytest.approx(100.0)


@pytest.mark.unit
def test_every_signal_is_kept_including_the_silent_ones() -> None:
    """An engineer debugging a false negative needs the readings from the
    detectors that did *not* fire."""
    assessment = aggregate(SILENT)
    assert len(assessment.signals) == 4


@pytest.mark.unit
def test_the_summary_serialises_and_carries_no_pixels() -> None:
    signals = [signal("screenshot", triggered=True, confidence=0.8), *SILENT[1:]]
    summary = aggregate(signals).describe()

    json.dumps(summary)
    assert summary["strongest"] == "PROFILE_IMAGE_IS_SCREENSHOT"
    assert summary["clean"] is False


@pytest.mark.unit
def test_a_finding_serialises() -> None:
    finding = aggregate(
        [signal("screenshot", triggered=True, confidence=0.8)]
    ).findings[0]

    json.dumps(finding.as_dict())
    assert finding.as_dict()["confidence"] == pytest.approx(0.8)


@pytest.mark.unit
def test_a_triggered_signal_with_zero_confidence_is_still_a_finding() -> None:
    """``triggered`` is the detector's verdict; confidence is how strongly.
    A detector that sets one without the other is buggy, but the aggregator
    must not silently drop its finding on the floor."""
    assessment = aggregate([signal("screenshot", triggered=True, confidence=0.0)])

    assert len(assessment.findings) == 1
    assert assessment.score == pytest.approx(100.0)
