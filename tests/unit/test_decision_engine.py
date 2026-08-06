"""The decision engine's rules, including the ones that cost me a defect.

The engine is small and entirely policy, which makes it exactly the sort of
component where a test is worth more than a careful reading: every branch here
corresponds to a real accept-or-refuse consequence for an applicant.
"""

from __future__ import annotations

import pytest

from hamqadam_ai.core.config import ApproveRule, DecisionConfig, RejectRule
from hamqadam_ai.core.constants import Recommendation, RiskLevel
from hamqadam_ai.decision.engine import DecisionEngine


@pytest.fixture
def engine() -> DecisionEngine:
    """An engine on the shipped thresholds."""
    return DecisionEngine(
        DecisionConfig(
            approve=ApproveRule(
                min_identity_confidence=75.0,
                max_fraud_risk=30.0,
                min_assessment_confidence=0.70,
            ),
            reject=RejectRule(max_identity_confidence=40.0, min_fraud_risk=65.0),
        )
    )


def _decide(engine: DecisionEngine, **kwargs: object) -> object:
    """Decide with sensible defaults for whatever the test is not varying."""
    params: dict[str, object] = {
        "identity_confidence": 90.0,
        "fraud_risk": 5.0,
        "fraud_level": RiskLevel.LOW,
        "assessment_confidence": 1.0,
    }
    params.update(kwargs)
    return engine.decide(**params)  # type: ignore[arg-type]


class TestApproval:
    """Approval needs every condition, not a good average."""

    def test_strong_evidence_approves(self, engine: DecisionEngine) -> None:
        outcome = _decide(engine)
        assert outcome.recommendation is Recommendation.APPROVE  # type: ignore[attr-defined]
        assert outcome.automated is True  # type: ignore[attr-defined]

    def test_high_confidence_cannot_outvote_high_fraud_risk(
        self, engine: DecisionEngine
    ) -> None:
        """A near-perfect face match with a fraud risk of 50 is not an approval.

        The conditions are conjunctive, not additive. Were they averaged, a
        confident match would buy an attacker past a signal specifically
        raised about their submission - which is the whole point of raising it.
        """
        outcome = _decide(engine, identity_confidence=99.9, fraud_risk=50.0)
        assert outcome.recommendation is Recommendation.MANUAL_REVIEW  # type: ignore[attr-defined]

    def test_thin_evidence_blocks_approval(self, engine: DecisionEngine) -> None:
        """The defect this rule was added for.

        A request supplying only a selfie scored identity 100 and fraud 0, and
        was approved: nothing contradicted the applicant because almost nothing
        had been checked. "No evidence against" is not "evidence for".
        """
        outcome = _decide(engine, assessment_confidence=0.17)
        assert outcome.recommendation is Recommendation.MANUAL_REVIEW  # type: ignore[attr-defined]
        codes = {reason.code for reason in outcome.reasons}  # type: ignore[attr-defined]
        assert "EVIDENCE_SUFFICIENT" in codes
        assert not next(
            reason.satisfied  # type: ignore[attr-defined]
            for reason in outcome.reasons  # type: ignore[attr-defined]
            if reason.code == "EVIDENCE_SUFFICIENT"
        )

    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [
            (75.0, Recommendation.APPROVE),  # the boundary is inclusive
            (74.99, Recommendation.MANUAL_REVIEW),
        ],
    )
    def test_confidence_boundary_is_inclusive(
        self, engine: DecisionEngine, confidence: float, expected: Recommendation
    ) -> None:
        """`min_identity_confidence` means at-or-above, not strictly above.

        Worth pinning: an off-by-one here silently moves the accept boundary
        for every applicant who lands exactly on a configured threshold, and
        thresholds are round numbers precisely because people aim at them.
        """
        assert _decide(engine, identity_confidence=confidence).recommendation is expected  # type: ignore[attr-defined]


class TestRejection:
    """Rejection needs any one condition, not all of them."""

    def test_low_confidence_rejects(self, engine: DecisionEngine) -> None:
        outcome = _decide(engine, identity_confidence=20.0)
        assert outcome.recommendation is Recommendation.REJECT  # type: ignore[attr-defined]

    def test_high_fraud_risk_rejects_despite_a_good_match(
        self, engine: DecisionEngine
    ) -> None:
        """A genuine face on a fraudulent submission is still a rejection.

        This is the impostor-with-a-real-document case: the biometric agrees
        because the document is genuine; it simply is not theirs.
        """
        outcome = _decide(
            engine,
            identity_confidence=95.0,
            fraud_risk=80.0,
            fraud_level=RiskLevel.HIGH,
        )
        assert outcome.recommendation is Recommendation.REJECT  # type: ignore[attr-defined]

    def test_rejection_wins_over_approval_when_both_could_fire(
        self, engine: DecisionEngine
    ) -> None:
        """Configured so both rule sets match, to pin the precedence.

        Approval wants confidence >= 75 and risk <= 30; rejection wants risk
        >= 20 here. At confidence 90 / risk 25 both are satisfied, and the
        answer must be the refusal. Ambiguity resolved towards admitting
        someone is the expensive direction.
        """
        overlapping = DecisionEngine(
            DecisionConfig(
                approve=ApproveRule(
                    min_identity_confidence=75.0,
                    max_fraud_risk=30.0,
                    min_assessment_confidence=0.7,
                ),
                reject=RejectRule(
                    max_identity_confidence=40.0, min_fraud_risk=20.0
                ),
            )
        )
        outcome = _decide(overlapping, identity_confidence=90.0, fraud_risk=25.0)
        assert outcome.recommendation is Recommendation.REJECT  # type: ignore[attr-defined]


class TestBlocking:
    """A blocking finding overrides the scores entirely."""

    def test_blocking_finding_forces_review(self, engine: DecisionEngine) -> None:
        outcome = _decide(engine, blocking=["NO_FACE_IN_LIVE_SELFIE"])
        assert outcome.recommendation is not Recommendation.APPROVE  # type: ignore[attr-defined]
        assert outcome.blocked_by  # type: ignore[attr-defined]

    def test_missing_identity_confidence_is_not_zero(
        self, engine: DecisionEngine
    ) -> None:
        """An absent score must not be read as a score of nought.

        A verification with no usable face has *no* identity confidence.
        Substituting 0.0 would reject the applicant for impersonation when the
        actual finding is that their photograph was unusable - a different
        answer, and one they can fix.
        """
        outcome = _decide(engine, identity_confidence=None)
        assert outcome.recommendation is not Recommendation.APPROVE  # type: ignore[attr-defined]


class TestReasons:
    """Every outcome must explain itself."""

    def test_reasons_are_always_present(self, engine: DecisionEngine) -> None:
        for kwargs in (
            {},
            {"identity_confidence": 20.0},
            {"fraud_risk": 90.0, "fraud_level": RiskLevel.HIGH},
            {"assessment_confidence": 0.1},
        ):
            outcome = _decide(engine, **kwargs)
            assert outcome.reasons, f"no reasons for {kwargs}"  # type: ignore[attr-defined]
            for reason in outcome.reasons:  # type: ignore[attr-defined]
                assert reason.code
                assert reason.message
                # Every message must name its number and its threshold, so a
                # reviewer can act on the response without also holding the
                # configuration file open.
                assert any(ch.isdigit() for ch in reason.message)


class TestRejectingConditions:
    """Categorical adverse findings must REJECT, not go to review.

    `blocking` and `rejecting` mean opposite things and were briefly conflated:
    a confirmed duplicate was routed through `blocking`, which yields
    MANUAL_REVIEW - inverting a deliberate `on_duplicate=reject` policy into a
    review. Blocking is "the evidence to decide was never produced";
    rejecting is "the evidence is in and it is adverse".
    """

    def test_a_rejecting_condition_yields_reject(self, engine: DecisionEngine) -> None:
        outcome = _decide(engine, rejecting=["DUPLICATE_FACE_CONFIRMED"])
        assert outcome.recommendation is Recommendation.REJECT  # type: ignore[attr-defined]
        assert outcome.blocked_by == "DUPLICATE_FACE_CONFIRMED"  # type: ignore[attr-defined]

    def test_it_wins_over_otherwise_perfect_evidence(
        self, engine: DecisionEngine
    ) -> None:
        """A duplicate must not be outvoted by a strong biometric match.

        The whole point is that the face genuinely matches - it matches an
        account that already exists.
        """
        outcome = _decide(
            engine,
            identity_confidence=99.0,
            fraud_risk=0.0,
            rejecting=["DUPLICATE_FACE_CONFIRMED"],
        )
        assert outcome.recommendation is Recommendation.REJECT  # type: ignore[attr-defined]

    def test_it_does_not_depend_on_the_fraud_score(
        self, engine: DecisionEngine
    ) -> None:
        """The reason this exists.

        A duplicate does reach REJECT through fraud - weight 0.70 aggregates to
        about 70, over the 65 threshold - but only by coincidence of tuning.
        With fraud pinned at zero the rejection must still hold.
        """
        outcome = _decide(
            engine, fraud_risk=0.0, rejecting=["DUPLICATE_FACE_CONFIRMED"]
        )
        assert outcome.recommendation is Recommendation.REJECT  # type: ignore[attr-defined]

    def test_the_reason_is_stated_as_satisfied(self, engine: DecisionEngine) -> None:
        """A rejecting finding is evidence that IS present, so satisfied=True.

        The opposite of a blocking condition, which reports satisfied=False
        because something was missing.
        """
        outcome = _decide(engine, rejecting=["DUPLICATE_FACE_CONFIRMED"])
        reason = outcome.reasons[0]  # type: ignore[attr-defined]
        assert reason.satisfied is True
        assert "already enrolled" in reason.message

    def test_review_policy_uses_blocking_instead(
        self, engine: DecisionEngine
    ) -> None:
        """`on_duplicate=manual_review` routes through blocking, not rejecting."""
        outcome = _decide(engine, blocking=["DUPLICATE_FACE_NEEDS_REVIEW"])
        assert outcome.recommendation is Recommendation.MANUAL_REVIEW  # type: ignore[attr-defined]
