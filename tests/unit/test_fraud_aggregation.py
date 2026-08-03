"""Combining fraud signals: the arithmetic, and why it is this arithmetic.

Pure logic. Every signal is built by hand, which is the point - the properties
worth pinning are the ones that only show up in combinations no fixture would
naturally produce.
"""

from __future__ import annotations

import pytest

from hamqadam_ai.core.constants import RiskLevel
from hamqadam_ai.fraud_detection.aggregator import aggregate
from hamqadam_ai.fraud_detection.signals import (
    FraudSignal,
    SignalDefinition,
    SignalFamily,
)

LOW_MAX = 30.0
MEDIUM_MAX = 65.0


def signal(
    family: SignalFamily,
    weight: float,
    *,
    code: str | None = None,
    confidence: float = 1.0,
    decisive: bool = False,
) -> FraudSignal:
    """One piece of evidence."""
    return FraudSignal(
        definition=SignalDefinition(
            code=code or f"{family.upper()}_{int(weight * 100)}",
            family=family,
            weight=weight,
            message="test signal",
            decisive=decisive,
        ),
        confidence=confidence,
    )


def score(signals, **kwargs) -> float:
    """Aggregate and return just the score."""
    return aggregate(signals, low_max=LOW_MAX, medium_max=MEDIUM_MAX, **kwargs).score


# --------------------------------------------------------------------------- #
# The rule that makes this design worth having
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_one_fact_reported_four_ways_counts_once() -> None:
    """The whole reason signals carry a family.

    An out-of-focus photograph produces four quality sub-findings. Added up
    they reach 65 - the boundary of HIGH risk - for what is a bad photo, not
    fraud.
    """
    quality = [
        signal(SignalFamily.CAPTURE_QUALITY, 0.20, code="A"),
        signal(SignalFamily.CAPTURE_QUALITY, 0.15, code="B"),
        signal(SignalFamily.CAPTURE_QUALITY, 0.20, code="C"),
        signal(SignalFamily.CAPTURE_QUALITY, 0.10, code="D"),
    ]

    assert score(quality) == pytest.approx(20.0)
    assert sum(s.contribution for s in quality) * 100 == pytest.approx(65.0)


@pytest.mark.unit
def test_independent_facts_do_compound() -> None:
    """The other half of the rule. Suppressing duplicates must not suppress
    genuine accumulation."""
    independent = [
        signal(SignalFamily.DOCUMENT_INTEGRITY, 0.70),
        signal(SignalFamily.DUPLICATION, 0.60),
        signal(SignalFamily.IMAGE_AUTHENTICITY, 0.55),
    ]
    assert score(independent) == pytest.approx(94.6, abs=0.1)


@pytest.mark.unit
def test_two_moderate_facts_exceed_either_alone() -> None:
    one = score([signal(SignalFamily.DUPLICATION, 0.50)])
    two = score(
        [
            signal(SignalFamily.DUPLICATION, 0.50),
            signal(SignalFamily.DOCUMENT_INTEGRITY, 0.50),
        ]
    )
    assert two > one
    assert two == pytest.approx(75.0)


@pytest.mark.unit
def test_the_family_reports_which_finding_drove_it() -> None:
    """A reviewer needs to know which of four quality findings mattered."""
    assessment = aggregate(
        [
            signal(SignalFamily.CAPTURE_QUALITY, 0.10, code="WEAK"),
            signal(SignalFamily.CAPTURE_QUALITY, 0.22, code="STRONGEST"),
        ],
        low_max=LOW_MAX,
        medium_max=MEDIUM_MAX,
    )

    assert assessment.families[0].driver == "STRONGEST"
    assert assessment.families[0].signal_count == 2


@pytest.mark.unit
def test_adding_innocent_noise_cannot_lower_the_score() -> None:
    """A weighted mean would let an attacker dilute a serious finding by
    piling on trivia. Noisy-OR is monotonic: more evidence never helps."""
    serious = [signal(SignalFamily.IDENTITY_CONSISTENCY, 0.80)]
    with_noise = [
        *serious,
        signal(SignalFamily.CAPTURE_QUALITY, 0.02),
        signal(SignalFamily.PRESENTATION, 0.01),
    ]

    assert score(with_noise) >= score(serious)


@pytest.mark.unit
def test_the_score_never_exceeds_one_hundred() -> None:
    many = [
        signal(family, 0.95)
        for family in SignalFamily
        if family is not SignalFamily.UNAVAILABLE
    ]
    assert score(many) <= 100.0


@pytest.mark.unit
def test_no_signals_is_no_risk() -> None:
    assessment = aggregate([], low_max=LOW_MAX, medium_max=MEDIUM_MAX)

    assert assessment.score == pytest.approx(0.0)
    assert assessment.level is RiskLevel.LOW
    assert assessment.top_factors == []


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_an_uncertain_finding_weighs_less() -> None:
    certain = score([signal(SignalFamily.IMAGE_AUTHENTICITY, 0.60)])
    unsure = score([signal(SignalFamily.IMAGE_AUTHENTICITY, 0.60, confidence=0.5)])

    assert unsure == pytest.approx(30.0)
    assert unsure < certain


@pytest.mark.unit
def test_a_zero_confidence_finding_contributes_nothing() -> None:
    assert score([signal(SignalFamily.DUPLICATION, 0.9, confidence=0.0)]) == (
        pytest.approx(0.0)
    )


# --------------------------------------------------------------------------- #
# Caps
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_cap_bounds_what_one_family_can_do_alone() -> None:
    """Capture quality is capped because most bad photographs are just bad
    photographs."""
    uncapped = score([signal(SignalFamily.CAPTURE_QUALITY, 0.80)])
    capped = score(
        [signal(SignalFamily.CAPTURE_QUALITY, 0.80)],
        family_caps={"capture_quality": 0.25},
    )

    assert uncapped == pytest.approx(80.0)
    assert capped == pytest.approx(25.0)


@pytest.mark.unit
def test_a_cap_is_reported_when_it_bites() -> None:
    """A reviewer seeing 25 needs to know the raw evidence was 80."""
    assessment = aggregate(
        [signal(SignalFamily.CAPTURE_QUALITY, 0.80)],
        low_max=LOW_MAX,
        medium_max=MEDIUM_MAX,
        family_caps={"capture_quality": 0.25},
    )
    family = assessment.families[0]

    assert family.capped is True
    assert family.raw_contribution == pytest.approx(0.80)
    assert family.contribution == pytest.approx(0.25)


@pytest.mark.unit
def test_an_uncapped_family_is_not_reported_as_capped() -> None:
    assessment = aggregate(
        [signal(SignalFamily.DUPLICATION, 0.40)],
        low_max=LOW_MAX,
        medium_max=MEDIUM_MAX,
        family_caps={"capture_quality": 0.25},
    )
    assert assessment.families[0].capped is False


@pytest.mark.unit
def test_a_cap_does_not_stop_a_family_combining_with_others() -> None:
    """Capped evidence is still evidence."""
    combined = score(
        [
            signal(SignalFamily.CAPTURE_QUALITY, 0.80),
            signal(SignalFamily.DUPLICATION, 0.60),
        ],
        family_caps={"capture_quality": 0.25},
    )
    assert combined == pytest.approx(70.0)


# --------------------------------------------------------------------------- #
# Banding
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, RiskLevel.LOW),
        (30.0, RiskLevel.LOW),
        (30.01, RiskLevel.MEDIUM),
        (65.0, RiskLevel.MEDIUM),
        (65.01, RiskLevel.HIGH),
        (100.0, RiskLevel.HIGH),
    ],
)
def test_band_boundaries_are_inclusive_at_the_top(
    value: float, expected: RiskLevel
) -> None:
    """``low_max`` is the highest score that still counts as LOW - matching
    what the configuration key is called."""
    weight = value / 100.0
    assessment = aggregate(
        [signal(SignalFamily.DUPLICATION, weight)] if weight else [],
        low_max=LOW_MAX,
        medium_max=MEDIUM_MAX,
    )
    assert assessment.level is expected


# --------------------------------------------------------------------------- #
# Decisive findings
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_decisive_finding_raises_the_band_without_faking_the_score() -> None:
    """The arithmetic keeps saying what the evidence weighed; the band says
    what to do about it. Overwriting the score would destroy the only number a
    reviewer can audit."""
    assessment = aggregate(
        [signal(SignalFamily.DUPLICATION, 0.30, code="DECISIVE", decisive=True)],
        low_max=LOW_MAX,
        medium_max=MEDIUM_MAX,
    )

    assert assessment.score == pytest.approx(30.0)
    assert assessment.level is RiskLevel.HIGH
    assert assessment.floored_by == "DECISIVE"


@pytest.mark.unit
def test_nothing_is_decisive_by_default() -> None:
    """A confirmed duplicate and a face mismatch are the obvious candidates,
    and they are exactly the findings whose limits are documented most heavily
    - identical twins, an uncalibrated 1:N threshold. Hard-coding them to HIGH
    would quietly undo that care."""
    from hamqadam_ai.fraud_detection.signals import CATALOGUE

    assert not [d for d in CATALOGUE.values() if d.decisive]


@pytest.mark.unit
def test_a_decisive_finding_at_zero_confidence_does_not_fire() -> None:
    assessment = aggregate(
        [
            signal(
                SignalFamily.DUPLICATION, 0.9, decisive=True, confidence=0.0
            )
        ],
        low_max=LOW_MAX,
        medium_max=MEDIUM_MAX,
    )
    assert assessment.floored_by is None


# --------------------------------------------------------------------------- #
# Missing evidence
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_missing_check_does_not_change_the_score() -> None:
    """An absent check is not evidence of innocence and not evidence of guilt.
    It is an absence."""
    with_all = aggregate([], low_max=LOW_MAX, medium_max=MEDIUM_MAX)
    with_gap = aggregate(
        [], low_max=LOW_MAX, medium_max=MEDIUM_MAX,
        unavailable=["duplicate"], expected_checks=6,
    )

    assert with_gap.score == pytest.approx(with_all.score)


@pytest.mark.unit
def test_a_missing_check_lowers_assessment_confidence() -> None:
    assessment = aggregate(
        [], low_max=LOW_MAX, medium_max=MEDIUM_MAX,
        unavailable=["duplicate", "ocr"], expected_checks=6,
    )

    assert assessment.assessment_confidence == pytest.approx(4 / 6, abs=1e-4)
    assert assessment.unavailable == ["duplicate", "ocr"]


@pytest.mark.unit
def test_confidence_never_goes_negative() -> None:
    assessment = aggregate(
        [], low_max=LOW_MAX, medium_max=MEDIUM_MAX,
        unavailable=[f"check-{i}" for i in range(20)], expected_checks=6,
    )
    assert assessment.assessment_confidence == pytest.approx(0.0)


@pytest.mark.unit
def test_confidence_is_one_when_nothing_is_missing() -> None:
    assessment = aggregate(
        [], low_max=LOW_MAX, medium_max=MEDIUM_MAX, expected_checks=6
    )
    assert assessment.assessment_confidence == pytest.approx(1.0)


@pytest.mark.unit
def test_the_unavailable_family_never_contributes() -> None:
    assessment = aggregate(
        [signal(SignalFamily.UNAVAILABLE, 0.9)],
        low_max=LOW_MAX,
        medium_max=MEDIUM_MAX,
    )
    assert assessment.score == pytest.approx(0.0)
    assert assessment.families == []


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_families_are_ordered_by_contribution() -> None:
    assessment = aggregate(
        [
            signal(SignalFamily.CAPTURE_QUALITY, 0.10),
            signal(SignalFamily.DUPLICATION, 0.70),
            signal(SignalFamily.PRESENTATION, 0.40),
        ],
        low_max=LOW_MAX,
        medium_max=MEDIUM_MAX,
    )
    contributions = [f.contribution for f in assessment.families]

    assert contributions == sorted(contributions, reverse=True)
    assert assessment.top_factors[0].startswith("DUPLICATION")


@pytest.mark.unit
def test_every_signal_is_retained_even_the_suppressed_ones() -> None:
    """A reviewer asking why a family scored what it did needs the members it
    did not choose."""
    assessment = aggregate(
        [
            signal(SignalFamily.CAPTURE_QUALITY, 0.10),
            signal(SignalFamily.CAPTURE_QUALITY, 0.22),
        ],
        low_max=LOW_MAX,
        medium_max=MEDIUM_MAX,
    )
    assert len(assessment.signals) == 2
    assert len(assessment.families) == 1


@pytest.mark.unit
def test_the_summary_serialises() -> None:
    import json

    assessment = aggregate(
        [signal(SignalFamily.DUPLICATION, 0.7)],
        low_max=LOW_MAX,
        medium_max=MEDIUM_MAX,
        unrecognised=["SOMETHING_NEW"],
    )
    json.dumps(assessment.describe())
    assert assessment.describe()["unrecognised"] == ["SOMETHING_NEW"]
