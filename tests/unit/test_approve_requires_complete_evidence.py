"""APPROVE must be impossible unless every mandatory stage completed.

The bug this pins
-----------------
Observed live: a submission carrying a profile photograph of a **different
person** returned APPROVE while the network was unstable.

Reproduced exactly. The profile stage failed mid-request, so no profile
comparison was produced. Module 4 renormalises identity weights over the
comparisons that actually produced a result - correct when evidence was never
*requested*, since a submission without secondary images should not be penalised
- and the absent profile comparison was simply dropped. The weights collapsed
to ``{cnic: 1.0}`` and identity confidence became the CNIC score alone:

    profile compared, FAILED (score 3) + cnic 95  ->  54.75  MANUAL_REVIEW
    profile NEVER COMPARED           + cnic 95  ->  95.00  APPROVE   <-- bug

The wrong photograph was never compared, so it never counted against the
applicant.

``assessment_confidence`` did not catch it. Losing one of six expected checks
leaves 0.83, over the 0.70 approval floor. And it is a *ratio*: losing the
profile comparison and losing the benign moiré observation score identically, so
it cannot express "the evidence that mattered is missing".

Two independent fixes, both tested here:

1. **A named mandatory-stage gate.** Any mandatory stage that did not run, or
   ran and failed, makes APPROVE impossible. Reads the pipeline's stage record,
   because a score cannot distinguish "never requested" from "requested and did
   not arrive" and those must lead to opposite outcomes.
2. **A symmetric ``profile_failure_cap``.** Only the CNIC had one; a mismatching
   profile was averaged in at weight 0.35 and could be outvoted by a strong
   document match.
"""

from __future__ import annotations

import pytest

from hamqadam_ai.core.config import Settings
from hamqadam_ai.core.constants import (
    ImageRole,
    Recommendation,
    RiskLevel,
)
from hamqadam_ai.decision.engine import DecisionEngine
from hamqadam_ai.matching.aggregator import IdentityAggregator
from hamqadam_ai.matching.comparator import (
    ComparisonType,
    MatchDecision,
    MatchOutcome,
)
from hamqadam_ai.pipelines.verification import VerificationPipeline, _Stage

pytestmark = pytest.mark.unit

MANDATORY = ("selfie", "profile", "cnic_ocr", "cnic_portrait", "matching", "duplicate")


def _pipeline(settings: Settings | None = None) -> VerificationPipeline:
    """A pipeline with no services; only the pure gates are exercised."""
    return VerificationPipeline(services={}, settings=settings or Settings())


def _all_good() -> dict[str, _Stage]:
    """A stage record where every mandatory stage ran and succeeded."""
    return {
        name: _Stage(name=name, ran=True, succeeded=True)
        for name in (*MANDATORY, "cnic_authenticity")
    }


def _outcome(
    kind: ComparisonType,
    *,
    compared: bool,
    score: float = 0.0,
    decision: MatchDecision = MatchDecision.NOT_COMPARED,
    role: ImageRole = ImageRole.PROFILE_IMAGE,
    reason: str | None = None,
) -> MatchOutcome:
    return MatchOutcome(
        comparison=kind,
        compared=compared,
        similarity=score / 100.0,
        score=score,
        decision=decision,
        confidence=0.9,
        low_confidence=False,
        target_role=role,
        target_label=str(kind),
        reason=reason,
    )


class TestTheReportedBug:
    """The exact submission that was wrongly approved."""

    def test_a_dropped_profile_comparison_no_longer_approves(self) -> None:
        """The headline regression.

        Wrong profile image, profile stage failed, CNIC matches strongly. The
        fused identity confidence is still 95 - Module 4 has no way to know the
        missing comparison mattered - so the gate must refuse on the stage
        record instead.
        """
        settings = Settings()
        stages = _all_good()
        stages["profile"] = _Stage(
            name="profile", ran=True, succeeded=False, error="network reset"
        )

        blocking = _pipeline(settings)._blocking_conditions(  # noqa: SLF001
            selfie={"embedding": object()},
            matching=type("M", (), {"identity_available": True})(),
            duplicate=None,
            stages=stages,
        )
        assert "MANDATORY_STAGE_INCOMPLETE" in blocking

        outcome = DecisionEngine(settings.decision).decide(
            identity_confidence=95.0,
            fraud_risk=10.0,
            fraud_level=RiskLevel.LOW,
            assessment_confidence=5 / 6,
            blocking=blocking,
        )
        assert outcome.recommendation is not Recommendation.APPROVE
        assert outcome.recommendation is Recommendation.MANUAL_REVIEW

    def test_assessment_confidence_alone_would_have_let_it_through(self) -> None:
        """Why the ratio was not enough, stated as a test.

        5/6 = 0.83 clears the 0.70 floor. Without the named gate every approve
        condition is satisfied and the answer is APPROVE - which is precisely
        what happened.
        """
        outcome = DecisionEngine(Settings().decision).decide(
            identity_confidence=95.0,
            fraud_risk=10.0,
            fraud_level=RiskLevel.LOW,
            assessment_confidence=5 / 6,
        )
        assert outcome.recommendation is Recommendation.APPROVE


class TestMandatoryStageGate:
    """Each mandatory stage, failed and skipped, blocks approval."""

    @pytest.mark.parametrize("stage_name", MANDATORY)
    @pytest.mark.parametrize("mode", ["failed", "skipped", "absent"])
    def test_every_mandatory_stage_blocks_when_incomplete(
        self, stage_name: str, mode: str
    ) -> None:
        stages = _all_good()
        if mode == "failed":
            stages[stage_name] = _Stage(
                name=stage_name, ran=True, succeeded=False, error="boom"
            )
        elif mode == "skipped":
            stages[stage_name] = _Stage(name=stage_name, ran=False)
        else:
            del stages[stage_name]

        incomplete = _pipeline()._incomplete_mandatory_stages(stages)  # noqa: SLF001
        assert stage_name in incomplete

    def test_a_complete_run_does_not_block(self) -> None:
        """The gate must not break the working path."""
        assert _pipeline()._incomplete_mandatory_stages(_all_good()) == []  # noqa: SLF001

    def test_an_optional_stage_failing_does_not_block(self) -> None:
        """`cnic_authenticity` is not mandatory.

        It contributes one benign moiré observation, and Module 10 measured its
        other detectors as not transferring to documents. Blocking on it would
        turn a cosmetic finding into a review queue.
        """
        stages = _all_good()
        stages["cnic_authenticity"] = _Stage(
            name="cnic_authenticity", ran=True, succeeded=False, error="boom"
        )
        assert _pipeline()._incomplete_mandatory_stages(stages) == []  # noqa: SLF001

    def test_a_secondary_image_failing_does_not_block(self) -> None:
        """Secondary images are optional by contract, so they cannot be gates."""
        stages = _all_good()
        stages["secondary[0]"] = _Stage(
            name="secondary[0]", ran=True, succeeded=False, error="boom"
        )
        assert _pipeline()._incomplete_mandatory_stages(stages) == []  # noqa: SLF001


class TestFailureModesFromTheBriefing:
    """The named failure modes, each reduced to a stage outcome.

    Network interruption, timeout, inference failure, OCR failure, detection
    failure, embedding failure, comparison failure and duplicate-search failure
    all arrive at the pipeline as the same thing: a stage that ran and did not
    succeed, or one that never ran. Testing the shape rather than each cause is
    deliberate - it covers causes nobody has thought of yet.
    """

    @pytest.mark.parametrize(
        ("stage_name", "cause"),
        [
            ("selfie", "face detection failure"),
            ("selfie", "embedding generation failure"),
            ("profile", "network interruption"),
            ("cnic_ocr", "OCR failure"),
            ("cnic_portrait", "portrait extraction failure"),
            ("matching", "face comparison failure"),
            ("duplicate", "duplicate search failure"),
            ("matching", "inference timeout"),
        ],
    )
    def test_no_cause_can_yield_approve(self, stage_name: str, cause: str) -> None:
        settings = Settings()
        stages = _all_good()
        stages[stage_name] = _Stage(
            name=stage_name, ran=True, succeeded=False, error=cause
        )

        blocking = _pipeline(settings)._blocking_conditions(  # noqa: SLF001
            selfie={"embedding": object()},
            matching=type("M", (), {"identity_available": True})(),
            duplicate=None,
            stages=stages,
        )
        outcome = DecisionEngine(settings.decision).decide(
            identity_confidence=99.0,
            fraud_risk=0.0,
            fraud_level=RiskLevel.LOW,
            assessment_confidence=1.0,
            blocking=blocking,
        )
        assert outcome.recommendation is not Recommendation.APPROVE, (
            f"{cause} in the {stage_name} stage still produced an approval"
        )

    def test_a_deadline_skip_cannot_approve(self) -> None:
        """A stage skipped because the time budget ran out is still missing."""
        stages = _all_good()
        stages["matching"] = _Stage(name="matching", ran=False)
        assert "matching" in _pipeline()._incomplete_mandatory_stages(stages)  # noqa: SLF001


class TestProfileFailureCap:
    """A mismatching profile caps identity confidence, as a CNIC one does."""

    @staticmethod
    def _fuse(profile_score: float, cnic_score: float) -> float | None:
        settings = Settings()
        return (
            IdentityAggregator(settings.matching)
            .aggregate(
                [
                    _outcome(
                        ComparisonType.PROFILE,
                        compared=True,
                        score=profile_score,
                        decision=(
                            MatchDecision.FAILED
                            if profile_score < 20
                            else MatchDecision.STRONG_MATCH
                        ),
                    ),
                    _outcome(
                        ComparisonType.CNIC,
                        compared=True,
                        score=cnic_score,
                        decision=MatchDecision.STRONG_MATCH,
                        role=ImageRole.CNIC_PORTRAIT,
                    ),
                ]
            )
            .identity_confidence
        )

    def test_a_wrong_profile_is_capped_not_averaged(self) -> None:
        """Measured before the cap: 54.75. A strong document outvoted it."""
        assert self._fuse(3.0, 95.0) == pytest.approx(45.0)

    def test_a_matching_profile_is_untouched(self) -> None:
        assert (self._fuse(95.0, 95.0) or 0.0) > 90.0

    def test_the_cap_is_symmetric_with_the_cnic_one(self) -> None:
        settings = Settings().matching.identity
        assert settings.profile_failure_cap == settings.cnic_failure_cap
