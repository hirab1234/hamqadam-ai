"""A stranger among the secondary images must not be averaged away.

Reported: three images of the account holder plus one secondary photograph of a
different person, and the service returned APPROVE.

The cause was the smallest of the three weights doing the work of a rule. A
FAILED secondary contributed 0 at weight 0.20 and nothing capped the fused
score, so whether the submission was blocked depended on how the other two
comparisons happened to land:

    profile 100, CNIC 87.34, secondary FAILED  ->  74.30   blocked by 0.70
    profile 100, CNIC 90.00, secondary FAILED  ->  75.50   APPROVED

`test_the_old_arithmetic_approved_a_stranger` pins that second line, so the
defect stays reproducible rather than becoming a story about a fix. The others
pin the cap and, just as importantly, the cases it must *not* fire on - a cap
that also catches legitimate submissions has only moved the error.
"""

from __future__ import annotations

import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import MatchDecision
from hamqadam_ai.matching.aggregator import IdentityAggregator
from hamqadam_ai.matching.comparator import ComparisonType, MatchOutcome


@pytest.fixture
def aggregator() -> IdentityAggregator:
    """An aggregator on the real configuration."""
    return IdentityAggregator(get_settings().matching)


def outcome(
    comparison: ComparisonType,
    score: float,
    decision: MatchDecision,
    *,
    confidence: float = 0.95,
    low: bool = False,
) -> MatchOutcome:
    """A synthetic outcome, for testing the aggregator in isolation."""
    return MatchOutcome(
        comparison=comparison,
        compared=True,
        similarity=0.7,
        score=score,
        decision=decision,
        confidence=confidence,
        low_confidence=low,
    )


def _stranger_submission() -> list[MatchOutcome]:
    """Everything agrees except one secondary image, which is someone else."""
    return [
        outcome(ComparisonType.PROFILE, 100.0, MatchDecision.STRONG_MATCH),
        outcome(ComparisonType.CNIC, 90.0, MatchDecision.STRONG_MATCH),
        outcome(ComparisonType.SECONDARY, 0.0, MatchDecision.FAILED),
    ]


class TestTheReportedBug:
    """The submission the user sent."""

    def test_the_old_arithmetic_approved_a_stranger(self) -> None:
        """Without the cap the fused score clears the 75 approval threshold.

        Raising the cap to 100 disables it without deleting it, which is what
        makes this a regression test for the defect and not merely a second
        assertion about the fix.
        """
        settings = get_settings().matching.model_copy(deep=True)
        settings.identity.secondary_failure_cap = 100.0
        uncapped = IdentityAggregator(settings)

        assessment = uncapped.aggregate(_stranger_submission())

        assert assessment.identity_confidence is not None
        approve_floor = get_settings().decision.approve.min_identity_confidence
        assert assessment.identity_confidence >= approve_floor, (
            "the defect no longer reproduces; this test is not measuring it"
        )
        assert assessment.capped_by is None

    def test_it_is_now_capped(self, aggregator: IdentityAggregator) -> None:
        assessment = aggregator.aggregate(_stranger_submission())
        cap = get_settings().matching.identity.secondary_failure_cap

        assert assessment.identity_confidence == pytest.approx(cap)
        assert assessment.capped_by == "secondary_failure"
        assert any("secondary" in reason.lower() for reason in assessment.reasons)

    def test_it_lands_below_the_approval_floor(
        self, aggregator: IdentityAggregator
    ) -> None:
        """The point of the cap: a margin held by rule, not by arithmetic."""
        assessment = aggregator.aggregate(_stranger_submission())
        approve_floor = get_settings().decision.approve.min_identity_confidence

        assert assessment.identity_confidence is not None
        assert assessment.identity_confidence < approve_floor

    def test_even_a_perfect_document_cannot_outvote_it(
        self, aggregator: IdentityAggregator
    ) -> None:
        """A cap, not a heavier weight - 100 on both others must not rescue it."""
        assessment = aggregator.aggregate(
            [
                outcome(ComparisonType.PROFILE, 100.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.CNIC, 100.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.SECONDARY, 0.0, MatchDecision.FAILED),
            ]
        )
        cap = get_settings().matching.identity.secondary_failure_cap
        assert assessment.identity_confidence == pytest.approx(cap)

    def test_one_bad_image_among_several_good_ones_still_caps(
        self, aggregator: IdentityAggregator
    ) -> None:
        """Averaging the secondaries must not hide a single stranger."""
        assessment = aggregator.aggregate(
            [
                outcome(ComparisonType.PROFILE, 98.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.CNIC, 92.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.SECONDARY, 97.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.SECONDARY, 96.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.SECONDARY, 2.0, MatchDecision.FAILED),
            ]
        )
        assert assessment.capped_by == "secondary_failure"


class TestItDoesNotFireOnLegitimateSubmissions:
    """A cap that also catches honest users has just moved the error."""

    def test_matching_secondaries_are_not_capped(
        self, aggregator: IdentityAggregator
    ) -> None:
        assessment = aggregator.aggregate(
            [
                outcome(ComparisonType.PROFILE, 100.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.CNIC, 90.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.SECONDARY, 95.0, MatchDecision.STRONG_MATCH),
            ]
        )
        assert assessment.capped_by is None
        assert assessment.identity_confidence is not None
        assert assessment.identity_confidence > 90.0

    def test_a_secondary_needing_review_is_not_capped(
        self, aggregator: IdentityAggregator
    ) -> None:
        """REVIEW is inconclusive, not contradictory - an old or odd photograph
        of yourself must not be treated as somebody else."""
        assessment = aggregator.aggregate(
            [
                outcome(ComparisonType.PROFILE, 95.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.CNIC, 90.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.SECONDARY, 55.0, MatchDecision.REVIEW),
            ]
        )
        assert assessment.capped_by is None

    def test_a_low_confidence_secondary_is_not_capped(
        self, aggregator: IdentityAggregator
    ) -> None:
        """Poor image quality is down-weighted, never treated as a mismatch."""
        assessment = aggregator.aggregate(
            [
                outcome(ComparisonType.PROFILE, 95.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.CNIC, 90.0, MatchDecision.STRONG_MATCH),
                outcome(
                    ComparisonType.SECONDARY,
                    70.0,
                    MatchDecision.REVIEW,
                    confidence=0.30,
                    low=True,
                ),
            ]
        )
        assert assessment.capped_by is None

    def test_no_secondary_images_at_all_is_unaffected(
        self, aggregator: IdentityAggregator
    ) -> None:
        """Secondary images are optional; their absence is not a failure."""
        assessment = aggregator.aggregate(
            [
                outcome(ComparisonType.PROFILE, 100.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.CNIC, 90.0, MatchDecision.STRONG_MATCH),
            ]
        )
        assert assessment.capped_by is None
        assert assessment.identity_confidence is not None
        assert assessment.identity_confidence > 90.0

    def test_a_secondary_that_could_not_be_compared_is_not_capped(
        self, aggregator: IdentityAggregator
    ) -> None:
        """No face found is missing evidence, not contradictory evidence."""
        assessment = aggregator.aggregate(
            [
                outcome(ComparisonType.PROFILE, 100.0, MatchDecision.STRONG_MATCH),
                outcome(ComparisonType.CNIC, 90.0, MatchDecision.STRONG_MATCH),
                MatchOutcome.not_compared(
                    ComparisonType.SECONDARY, reason="no face detected"
                ),
            ]
        )
        assert assessment.capped_by is None


class TestTheThreeCapsAreSymmetric:
    """Each was added after the previous one showed the same pattern. A future
    change that lowers one and forgets the others reopens the gap."""

    def test_all_three_caps_exist_and_agree(self) -> None:
        identity = get_settings().matching.identity
        assert identity.cnic_failure_cap == identity.profile_failure_cap
        assert identity.profile_failure_cap == identity.secondary_failure_cap

    def test_every_cap_sits_below_the_approval_floor(self) -> None:
        """A cap above the floor would not block anything."""
        identity = get_settings().matching.identity
        floor = get_settings().decision.approve.min_identity_confidence
        for name in (
            "cnic_failure_cap",
            "profile_failure_cap",
            "secondary_failure_cap",
        ):
            assert getattr(identity, name) < floor, f"{name} cannot block an approval"

    def test_the_yaml_declares_all_three(self) -> None:
        """README promises every threshold lives in thresholds.yaml, and only
        the CNIC cap was actually there."""
        from pathlib import Path

        text = Path("configs/thresholds.yaml").read_text(encoding="utf-8")
        for name in (
            "cnic_failure_cap",
            "profile_failure_cap",
            "secondary_failure_cap",
        ):
            assert f"{name}:" in text, f"{name} is not declared in thresholds.yaml"
