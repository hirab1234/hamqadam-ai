"""MODULE 10 - turning evidence into a recommendation.

What this is, precisely
-----------------------
A **recommendation**, not a decision. Section 25 of the requirements document
fixes the boundary: the AI service returns ``APPROVE``, ``REJECT`` or
``MANUAL_REVIEW``, and the Backend's rules engine turns that into an account
outcome using things this service will never see - a manual allow-list, a
regulatory hold, an account's history, a support ticket from last week.

That is not a disclaimer, it is a design constraint, and it shows up in the
code: nothing here reaches for a database, and the response says
``recommendation`` rather than ``decision`` in every field name.

Why the default is MANUAL_REVIEW
--------------------------------
The rules are deliberately asymmetric and the middle is deliberately wide.

``APPROVE`` requires **every** condition to hold. ``REJECT`` requires **any**
one to hold. Everything else - which is most of the interesting space - is
``MANUAL_REVIEW``. A verification system that automates the easy cases and
hands a human the rest is doing its job; one that forces every case into a
binary is making somebody else's mistake for them.

The specific reason the middle is wide here is that this project cannot
validate its own thresholds. Modules 4, 8 and 9 all report their operating
points as unvalidated, because the datasets that would calibrate them do not
exist in this repository. Automating a rejection on an uncalibrated threshold
is how an honest applicant gets locked out with no recourse, so the rules only
automate where the evidence is unambiguous and route the rest to a person.

Blocking conditions
-------------------
Some outcomes are not scores. If no face was found in the live selfie there is
nothing to verify, and averaging that into a confidence would produce a number
implying a comparison that never happened. Those cases short-circuit to
``MANUAL_REVIEW`` with the reason named, rather than being scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hamqadam_ai.core.config import DecisionConfig
from hamqadam_ai.core.constants import Recommendation, RiskLevel


@dataclass(frozen=True, slots=True)
class DecisionReason:
    """One condition that shaped the recommendation.

    Attributes:
        code: Stable machine-readable identifier.
        message: What it means, for a human reviewer.
        satisfied: Whether the condition held.
        detail: The numbers behind it, so a reviewer can check the arithmetic
            rather than take it on trust.
    """

    code: str
    message: str
    satisfied: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "code": self.code,
            "message": self.message,
            "satisfied": self.satisfied,
            "detail": dict(self.detail),
        }


@dataclass(slots=True)
class DecisionOutcome:
    """The recommendation and everything behind it.

    Attributes:
        recommendation: APPROVE, REJECT or MANUAL_REVIEW.
        reasons: Every condition evaluated, in the order they were considered.
        blocked_by: The blocking condition that forced review, if any.
        automated: Whether a rule fired, as opposed to falling through to the
            default. A reviewer seeing MANUAL_REVIEW needs to know whether the
            system decided that or simply could not decide.
    """

    recommendation: Recommendation
    reasons: list[DecisionReason] = field(default_factory=list)
    blocked_by: str | None = None
    automated: bool = False

    @property
    def failed_conditions(self) -> list[str]:
        """Codes of the conditions that did not hold."""
        return [reason.code for reason in self.reasons if not reason.satisfied]

    def describe(self) -> dict[str, Any]:
        """PII-free summary for logging."""
        return {
            "recommendation": str(self.recommendation),
            "automated": self.automated,
            "blocked_by": self.blocked_by,
            "failed": self.failed_conditions,
        }


#: Conditions that make a recommendation impossible rather than negative.
#:
#: Each means "the evidence needed to decide was never produced". Scoring them
#: would express an absence as a low number, which reads as a judgement about
#: the applicant rather than about the request.
BLOCKING_CONDITIONS: dict[str, str] = {
    "NO_LIVE_SELFIE": (
        "No usable live selfie was supplied, so there is nothing to verify the "
        "other images against."
    ),
    "NO_IDENTITY_COMPARISON": (
        "No face comparison could be completed, so there is no identity "
        "confidence to judge."
    ),
    "PIPELINE_INCOMPLETE": (
        "Not enough of the verification ran to support a recommendation."
    ),
    "DUPLICATE_FACE_NEEDS_REVIEW": (
        "This face is already enrolled in the gallery under a different "
        "account reference, so a human must confirm whether the two are the "
        "same person."
    ),
}

#: Categorical adverse findings. Unlike a blocking condition these are *not* an
#: absence of evidence - they are a definite negative, and they yield REJECT.
REJECTING_CONDITIONS: dict[str, str] = {
    "DUPLICATE_FACE_CONFIRMED": (
        "This face is already enrolled in the gallery under a different account "
        "reference, and policy is to refuse duplicates outright."
    ),
}


class DecisionEngine:
    """Applies the configured rules to one verification's evidence.

    Args:
        config: The ``decision`` section of the settings.
    """

    __slots__ = ("_config",)

    def __init__(self, config: DecisionConfig) -> None:
        self._config = config

    def decide(
        self,
        *,
        identity_confidence: float | None,
        fraud_risk: float,
        fraud_level: RiskLevel,
        assessment_confidence: float = 1.0,
        blocking: list[str] | None = None,
        rejecting: list[str] | None = None,
    ) -> DecisionOutcome:
        """Produce a recommendation.

        Args:
            identity_confidence: Module 4's fused confidence, 0-100. ``None``
                means no comparison was possible, which is a blocking
                condition rather than a score of zero.
            fraud_risk: Module 9's score, 0-100.
            fraud_level: Module 9's band.
            assessment_confidence: Share of the intended checks that ran.
                Gates **approval only**. A rejection on strong evidence stays
                valid however many other checks were unavailable - finding a
                gender mismatch does not become less true because the duplicate
                gallery was down.
            blocking: Conditions that make a recommendation impossible.
                These yield MANUAL_REVIEW - "the evidence to decide was never
                produced" is not a judgement about the applicant.
            rejecting: Categorical adverse findings, which yield REJECT
                outright.

                Separate from ``blocking`` because the two mean opposite things
                and were briefly conflated here. A confirmed duplicate is not an
                absence of evidence; it is evidence. Routing it through
                ``blocking`` would have turned a deliberate
                ``on_duplicate=reject`` policy into a manual review.

                These also bypass the fraud arithmetic on purpose. A duplicate
                does reach REJECT through the score - weight 0.70 aggregates to
                about 70, over the 65 reject threshold - but only by
                coincidence of tuning. Retune the weight, raise the threshold,
                or let a family cap bite, and the same face on two accounts
                starts being approved with no test failing.

        Returns:
            The outcome, with every condition evaluated and reported.
        """
        refusals = list(rejecting or [])
        if refusals:
            return DecisionOutcome(
                recommendation=Recommendation.REJECT,
                reasons=[
                    DecisionReason(
                        code=code,
                        message=REJECTING_CONDITIONS.get(
                            code, "A categorical adverse finding was recorded."
                        ),
                        satisfied=True,
                    )
                    for code in refusals
                ],
                blocked_by=refusals[0],
                automated=True,
            )

        blockers = list(blocking or [])
        if identity_confidence is None and "NO_IDENTITY_COMPARISON" not in blockers:
            blockers.append("NO_IDENTITY_COMPARISON")

        if blockers:
            return DecisionOutcome(
                recommendation=Recommendation.MANUAL_REVIEW,
                reasons=[
                    DecisionReason(
                        code=code,
                        message=BLOCKING_CONDITIONS.get(
                            code, "A required input was missing."
                        ),
                        satisfied=False,
                    )
                    for code in blockers
                ],
                blocked_by=blockers[0],
                automated=True,
            )

        confidence = float(identity_confidence or 0.0)
        approve = self._config.approve
        reject = self._config.reject

        # Rejection is evaluated first. Any one condition is enough, and a
        # request that trips one should not also be reported as having "failed
        # to qualify for approval" - that reads as a near miss.
        reject_reasons = [
            DecisionReason(
                code="IDENTITY_CONFIDENCE_TOO_LOW",
                message=(
                    f"Identity confidence of {confidence:.1f} is at or below "
                    f"the automatic-rejection threshold of "
                    f"{reject.max_identity_confidence:.1f}."
                ),
                satisfied=confidence <= reject.max_identity_confidence,
                detail={
                    "identity_confidence": round(confidence, 2),
                    "threshold": reject.max_identity_confidence,
                },
            ),
            DecisionReason(
                code="FRAUD_RISK_TOO_HIGH",
                message=(
                    f"Fraud risk of {fraud_risk:.1f} is at or above the "
                    f"automatic-rejection threshold of {reject.min_fraud_risk:.1f}."
                ),
                satisfied=fraud_risk >= reject.min_fraud_risk,
                detail={
                    "fraud_risk": round(fraud_risk, 2),
                    "fraud_level": str(fraud_level),
                    "threshold": reject.min_fraud_risk,
                },
            ),
        ]

        if any(reason.satisfied for reason in reject_reasons):
            return DecisionOutcome(
                recommendation=Recommendation.REJECT,
                reasons=[r for r in reject_reasons if r.satisfied],
                automated=True,
            )

        approve_reasons = [
            DecisionReason(
                code="EVIDENCE_SUFFICIENT",
                message=(
                    f"{assessment_confidence:.0%} of the intended checks ran, "
                    f"meeting the {approve.min_assessment_confidence:.0%} "
                    f"required before a verification can be approved "
                    f"automatically."
                ),
                satisfied=(
                    assessment_confidence >= approve.min_assessment_confidence
                ),
                detail={
                    "assessment_confidence": round(assessment_confidence, 3),
                    "threshold": approve.min_assessment_confidence,
                },
            ),
            DecisionReason(
                code="IDENTITY_CONFIDENCE_SUFFICIENT",
                message=(
                    f"Identity confidence of {confidence:.1f} meets the "
                    f"automatic-approval threshold of "
                    f"{approve.min_identity_confidence:.1f}."
                ),
                satisfied=confidence >= approve.min_identity_confidence,
                detail={
                    "identity_confidence": round(confidence, 2),
                    "threshold": approve.min_identity_confidence,
                },
            ),
            DecisionReason(
                code="FRAUD_RISK_ACCEPTABLE",
                message=(
                    f"Fraud risk of {fraud_risk:.1f} is within the "
                    f"automatic-approval ceiling of {approve.max_fraud_risk:.1f}."
                ),
                satisfied=fraud_risk <= approve.max_fraud_risk,
                detail={
                    "fraud_risk": round(fraud_risk, 2),
                    "fraud_level": str(fraud_level),
                    "threshold": approve.max_fraud_risk,
                },
            ),
        ]

        if all(reason.satisfied for reason in approve_reasons):
            return DecisionOutcome(
                recommendation=Recommendation.APPROVE,
                reasons=approve_reasons,
                automated=True,
            )

        # The wide middle. Reported with the approval conditions attached, so a
        # reviewer sees which one fell short rather than an unexplained
        # "review".
        return DecisionOutcome(
            recommendation=Recommendation.MANUAL_REVIEW,
            reasons=approve_reasons,
            automated=False,
        )

    def describe(self) -> dict[str, Any]:
        """Serialisable summary of the active rules, for ``/health``."""
        return {
            "approve": {
                "min_identity_confidence": (
                    self._config.approve.min_identity_confidence
                ),
                "max_fraud_risk": self._config.approve.max_fraud_risk,
                "min_assessment_confidence": (
                    self._config.approve.min_assessment_confidence
                ),
                "requires": "all conditions",
            },
            "reject": {
                "max_identity_confidence": self._config.reject.max_identity_confidence,
                "min_fraud_risk": self._config.reject.min_fraud_risk,
                "requires": "any condition",
            },
            "default": str(Recommendation.MANUAL_REVIEW),
            "authority": (
                "recommendation only; the Backend's rules engine owns the "
                "account outcome"
            ),
            "thresholds_validated": False,
        }


__all__ = ["BLOCKING_CONDITIONS", "DecisionEngine", "DecisionOutcome", "DecisionReason"]
