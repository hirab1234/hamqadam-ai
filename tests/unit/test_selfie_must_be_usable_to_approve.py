"""The live selfie is the reference; a defect in it cannot support an approval.

Observed in a real submission:

    selfie_detection.passed      false
    error_code                   FACE_POSE_OUT_OF_RANGE
    pose.pitch                   -71 degrees
    visibility_breakdown.pose    0        limiting_factor "pose"

    cnic  similarity 0.489  threshold 0.42  ->  STRONG_MATCH  77.98  ->  APPROVE

Every comparison is measured against the selfie, so a defect there does not stay
in its own stage - it lowers each similarity score, and those scores are exactly
what the approval thresholds read. The same person photographed frontally scores
above 0.70 against the same card. The margin was the pose, not the identity.

Why this is an approval blocker and not a blocking condition
-----------------------------------------------------------
A blocking condition means the evidence is absent, so the answer is
MANUAL_REVIEW regardless of anything else - it is evaluated *before* the
rejection rules. Routing a degraded selfie through it would mean an impostor who
also submits a badly posed selfie is queued for a human instead of rejected,
which rewards submitting a worse photograph. `TestItNeverRescuesARejection`
pins that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import Recommendation, RiskLevel
from hamqadam_ai.decision.engine import APPROVAL_BLOCKERS, DecisionEngine
from hamqadam_ai.pipelines.verification import VerificationPipeline


@dataclass
class _Detection:
    passed: bool = True


@dataclass
class _Quality:
    usable: bool = True


def _selfie(*, detection_passed: bool = True, quality_usable: bool = True) -> Any:
    return {
        "result": _Detection(passed=detection_passed),
        "quality": _Quality(usable=quality_usable),
        "embedding": object(),
    }


@pytest.fixture
def engine() -> DecisionEngine:
    return DecisionEngine(get_settings().decision)


@pytest.fixture
def pipeline() -> VerificationPipeline:
    return VerificationPipeline(services={}, settings=get_settings())


def _approvable(**kwargs: Any) -> dict[str, Any]:
    """Arguments that would otherwise be a clean APPROVE."""
    return {
        "identity_confidence": 93.0,
        "fraud_risk": 10.0,
        "fraud_level": RiskLevel.LOW,
        "assessment_confidence": 1.0,
        **kwargs,
    }


class TestTheGateItself:
    """Which selfies disqualify an approval."""

    def test_a_selfie_that_failed_detection_is_flagged(
        self, pipeline: VerificationPipeline
    ) -> None:
        assert pipeline._approval_blockers(_selfie(detection_passed=False)) == [
            "SELFIE_NOT_USABLE"
        ]

    def test_a_selfie_below_the_quality_floor_is_flagged(
        self, pipeline: VerificationPipeline
    ) -> None:
        assert pipeline._approval_blockers(_selfie(quality_usable=False)) == [
            "SELFIE_NOT_USABLE"
        ]

    def test_a_good_selfie_is_not_flagged(
        self, pipeline: VerificationPipeline
    ) -> None:
        assert pipeline._approval_blockers(_selfie()) == []

    def test_a_missing_selfie_is_left_to_the_blocking_rule(
        self, pipeline: VerificationPipeline
    ) -> None:
        """Absence is already NO_LIVE_SELFIE. Reporting it twice would put one
        submission in two categories and name the wrong cause first."""
        assert pipeline._approval_blockers(None) == []

    def test_the_gate_can_be_turned_off(self) -> None:
        settings = get_settings().model_copy(deep=True)
        settings.decision.approve.require_usable_selfie = False
        pipeline = VerificationPipeline(services={}, settings=settings)

        assert pipeline._approval_blockers(_selfie(detection_passed=False)) == []


class TestItStopsTheApproval:
    """The reported outcome must no longer be reachable."""

    def test_an_otherwise_perfect_submission_is_held(
        self, engine: DecisionEngine
    ) -> None:
        outcome = engine.decide(
            **_approvable(), approval_blockers=["SELFIE_NOT_USABLE"]
        )
        assert outcome.recommendation is Recommendation.MANUAL_REVIEW
        assert outcome.blocked_by == "SELFIE_NOT_USABLE"

    def test_the_same_submission_approves_with_a_good_selfie(
        self, engine: DecisionEngine
    ) -> None:
        """Guards the test above: if this did not approve, the gate would not be
        what is holding the other one."""
        outcome = engine.decide(**_approvable(), approval_blockers=[])
        assert outcome.recommendation is Recommendation.APPROVE


class TestItNeverRescuesARejection:
    """The reason this is not a blocking condition."""

    def test_high_fraud_still_rejects(self, engine: DecisionEngine) -> None:
        outcome = engine.decide(
            **_approvable(fraud_risk=90.0, fraud_level=RiskLevel.HIGH),
            approval_blockers=["SELFIE_NOT_USABLE"],
        )
        assert outcome.recommendation is Recommendation.REJECT

    def test_low_identity_still_rejects(self, engine: DecisionEngine) -> None:
        outcome = engine.decide(
            **_approvable(identity_confidence=20.0),
            approval_blockers=["SELFIE_NOT_USABLE"],
        )
        assert outcome.recommendation is Recommendation.REJECT

    def test_a_categorical_refusal_still_rejects(
        self, engine: DecisionEngine
    ) -> None:
        outcome = engine.decide(
            **_approvable(),
            rejecting=["DUPLICATE_FACE_CONFIRMED"],
            approval_blockers=["SELFIE_NOT_USABLE"],
        )
        assert outcome.recommendation is Recommendation.REJECT

    def test_a_genuine_blocking_condition_still_wins(
        self, engine: DecisionEngine
    ) -> None:
        """Absent evidence outranks degraded evidence, and should be reported
        as the cause."""
        outcome = engine.decide(
            **_approvable(),
            blocking=["NO_LIVE_SELFIE"],
            approval_blockers=["SELFIE_NOT_USABLE"],
        )
        assert outcome.recommendation is Recommendation.MANUAL_REVIEW
        assert outcome.blocked_by == "NO_LIVE_SELFIE"


class TestTheReasonIsExplained:
    def test_the_code_has_a_message(self) -> None:
        assert "SELFIE_NOT_USABLE" in APPROVAL_BLOCKERS
        assert len(APPROVAL_BLOCKERS["SELFIE_NOT_USABLE"]) > 40
