"""Fusing individual comparisons into one identity confidence.

The specification asks for a single ``identity_confidence_score``. Producing
one from up to six comparisons of three different kinds is a fusion problem
with several defensible answers, so the choices here are spelled out.

Why the CNIC carries the most weight
------------------------------------
It is the only state-issued anchor in the request. The profile photographs are
whatever the user chose to upload; two of them agreeing with the selfie proves
that the same person appears in all three, which is a claim about internal
consistency rather than about identity. Only the document ties that face to a
name.

Why a failed CNIC comparison caps the result
--------------------------------------------
Without the cap, a user could upload three selfies of themselves alongside
somebody else's identity document and score highly on internal consistency
while the one comparison that matters had failed. The cap makes the document
comparison a gate rather than merely a heavy vote.

Why secondary images are averaged, not maximised
------------------------------------------------
A secondary image that does not match is a fraud signal - somebody else's
photograph on the profile - not noise to be discarded, and taking the maximum
would silently ignore it. Taking the minimum is wrong in the other direction,
because people legitimately upload old or unflattering photographs of
themselves. The mean is used for the score and the worst individual result is
reported separately, so Module 9 can weigh it as a risk signal without it
dominating the identity decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hamqadam_ai.core.config import MatchingConfig
from hamqadam_ai.core.constants import MatchDecision
from hamqadam_ai.matching.comparator import ComparisonType, MatchOutcome


@dataclass(slots=True)
class IdentityAssessment:
    """The fused view of who this person is.

    Attributes:
        identity_confidence: Composite 0-100 score, or ``None`` when too few
            comparisons were possible for the number to mean anything.
        available: Whether enough comparisons succeeded.
        face_match_score: The headline score the specification names. Equal to
            the profile comparison where one exists, otherwise the best
            available selfie-to-photograph comparison.
        profile_score: Selfie against the main profile image.
        secondary_scores: Selfie against each secondary image, in order.
        secondary_aggregate: The combined secondary score.
        secondary_worst: The weakest individual secondary score. Reported
            separately as a fraud signal.
        cnic_score: Selfie against the CNIC portrait.
        contributions: What each comparison type contributed, after weighting.
        capped_by: Set when a rule limited the score below its computed value.
        outcomes: Every individual comparison.
        reasons: Human-readable notes about the aggregation.
    """

    identity_confidence: float | None
    available: bool
    face_match_score: float = 0.0
    profile_score: float | None = None
    secondary_scores: list[float] = field(default_factory=list)
    secondary_aggregate: float | None = None
    secondary_worst: float | None = None
    cnic_score: float | None = None
    contributions: dict[str, float] = field(default_factory=dict)
    effective_weights: dict[str, float] = field(default_factory=dict)
    capped_by: str | None = None
    outcomes: list[MatchOutcome] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def any_failed(self) -> bool:
        """Whether any comparison actively contradicted the identity claim."""
        return any(outcome.failed for outcome in self.outcomes)

    @property
    def compared_count(self) -> int:
        """How many comparisons actually ran."""
        return sum(1 for outcome in self.outcomes if outcome.compared)

    def summary(self) -> dict[str, Any]:
        """Compact, PII-free summary for logging."""
        return {
            "identity_confidence": (
                round(self.identity_confidence, 2)
                if self.identity_confidence is not None
                else None
            ),
            "available": self.available,
            "face_match": round(self.face_match_score, 2),
            "profile": round(self.profile_score, 2) if self.profile_score else None,
            "cnic": round(self.cnic_score, 2) if self.cnic_score else None,
            "secondary_n": len(self.secondary_scores),
            "compared": self.compared_count,
            "any_failed": self.any_failed,
            "capped_by": self.capped_by,
        }


class IdentityAggregator:
    """Combines comparison outcomes into one identity confidence.

    Args:
        config: The matching section of the settings.
    """

    __slots__ = ("_config",)

    def __init__(self, config: MatchingConfig) -> None:
        self._config = config

    def aggregate(self, outcomes: list[MatchOutcome]) -> IdentityAssessment:
        """Fuse a set of comparison outcomes.

        Args:
            outcomes: Every comparison made for this verification.

        Returns:
            The fused assessment. ``identity_confidence`` is ``None`` when
            fewer than ``identity.min_comparisons`` comparisons succeeded -
            reporting a number derived from nothing would be worse than
            reporting its absence.
        """
        identity = self._config.identity

        profile = self._first(outcomes, ComparisonType.PROFILE)
        cnic = self._first(outcomes, ComparisonType.CNIC)
        secondaries = [
            outcome
            for outcome in outcomes
            if outcome.comparison is ComparisonType.SECONDARY and outcome.compared
        ]

        reasons: list[str] = []
        compared = [outcome for outcome in outcomes if outcome.compared]

        if len(compared) < identity.min_comparisons:
            return IdentityAssessment(
                identity_confidence=None,
                available=False,
                outcomes=outcomes,
                reasons=[
                    f"Only {len(compared)} comparison(s) could be made; at least "
                    f"{identity.min_comparisons} is required for an identity "
                    f"confidence to mean anything."
                ],
            )

        secondary_scores = [outcome.score for outcome in secondaries]
        secondary_aggregate = self._combine_secondary(secondary_scores)
        secondary_worst = min(secondary_scores) if secondary_scores else None

        # Weights are renormalised over the comparison types that actually
        # produced a result, so a request without secondary images is not
        # penalised for the absence.
        available: dict[str, tuple[float, float]] = {}
        if profile is not None and profile.compared:
            available["profile"] = (profile.score, self._weight_for(profile))
        if cnic is not None and cnic.compared:
            available["cnic"] = (cnic.score, self._weight_for(cnic))
        if secondary_aggregate is not None:
            worst_confidence = min(
                (outcome.confidence for outcome in secondaries), default=1.0
            )
            low = any(outcome.low_confidence for outcome in secondaries)
            weight = (
                self._config.confidence.low_confidence_weight if low else 1.0
            )
            available["secondary"] = (secondary_aggregate, weight)
            if low:
                reasons.append(
                    f"At least one secondary image produced a low-confidence "
                    f"embedding (weakest {worst_confidence:.2f}); its "
                    f"contribution was down-weighted."
                )

        weighted_total = 0.0
        weight_total = 0.0
        contributions: dict[str, float] = {}
        effective: dict[str, float] = {}

        for key, (score, modifier) in available.items():
            weight = identity.weights[key] * modifier
            weighted_total += weight * score
            weight_total += weight
            contributions[key] = round(weight * score, 3)
            effective[key] = round(weight, 4)

        if weight_total <= 0.0:
            return IdentityAssessment(
                identity_confidence=None,
                available=False,
                outcomes=outcomes,
                reasons=[
                    "Every available comparison carried zero weight, so no "
                    "identity confidence could be derived."
                ],
            )

        confidence = weighted_total / weight_total
        effective = {key: value / weight_total for key, value in effective.items()}

        capped_by: str | None = None

        # A failing profile comparison caps the score too, not just a failing
        # CNIC one. Only the CNIC cap existed; a mismatching profile was
        # averaged in at weight 0.35 and could be outvoted by a strong document
        # match. Measured: profile 3 with CNIC 95 fused to 54.75. That stayed
        # under 75 by arithmetic rather than by rule, and a reweighting would
        # have quietly removed the margin.
        if (
            profile is not None
            and profile.compared
            and profile.decision is MatchDecision.FAILED
        ):
            cap = identity.profile_failure_cap
            if confidence > cap:
                reasons.append(
                    f"The profile photograph does not match the live selfie, so "
                    f"identity confidence was capped at {cap:.0f}. A profile "
                    f"image of a different person contradicts the submission; "
                    f"it is not a low score to be averaged against the "
                    f"document match."
                )
                confidence = cap
                capped_by = "profile_failure"

        if cnic is not None and cnic.compared and cnic.decision is MatchDecision.FAILED:
            cap = identity.cnic_failure_cap
            if confidence > cap:
                reasons.append(
                    f"The CNIC portrait does not match the live selfie, so "
                    f"identity confidence was capped at {cap:.0f} despite the "
                    f"profile images agreeing with each other. Two photographs "
                    f"of the same person are not evidence of identity when the "
                    f"document names somebody else."
                )
                confidence = cap
                capped_by = "cnic_failure"

        for outcome in outcomes:
            if outcome.compared and outcome.low_confidence:
                if outcome.comparison is not ComparisonType.SECONDARY:
                    reasons.append(
                        f"The {outcome.comparison} comparison was low-confidence "
                        f"({outcome.confidence:.2f}); its contribution was "
                        f"down-weighted."
                    )
            elif not outcome.compared and outcome.reason:
                reasons.append(
                    f"No {outcome.comparison} comparison: {outcome.reason}."
                )

        return IdentityAssessment(
            identity_confidence=round(confidence, 2),
            available=True,
            face_match_score=self._headline(profile, secondaries, cnic),
            profile_score=profile.score if profile and profile.compared else None,
            secondary_scores=[round(score, 2) for score in secondary_scores],
            secondary_aggregate=(
                round(secondary_aggregate, 2)
                if secondary_aggregate is not None
                else None
            ),
            secondary_worst=(
                round(secondary_worst, 2) if secondary_worst is not None else None
            ),
            cnic_score=cnic.score if cnic and cnic.compared else None,
            contributions=contributions,
            effective_weights=effective,
            capped_by=capped_by,
            outcomes=outcomes,
            reasons=reasons,
        )

    # -- Internals ----------------------------------------------------------- #

    @staticmethod
    def _first(
        outcomes: list[MatchOutcome], comparison: ComparisonType
    ) -> MatchOutcome | None:
        """Return the first outcome of a given type, if any."""
        for outcome in outcomes:
            if outcome.comparison is comparison:
                return outcome
        return None

    def _weight_for(self, outcome: MatchOutcome) -> float:
        """Modifier applied to a single outcome's configured weight."""
        if outcome.low_confidence:
            return self._config.confidence.low_confidence_weight
        return 1.0

    def _combine_secondary(self, scores: list[float]) -> float | None:
        """Collapse the secondary scores per the configured strategy."""
        if not scores:
            return None
        strategy = self._config.identity.secondary_aggregation
        if strategy == "max":
            return max(scores)
        if strategy == "min":
            return min(scores)
        return sum(scores) / len(scores)

    @staticmethod
    def _headline(
        profile: MatchOutcome | None,
        secondaries: list[MatchOutcome],
        cnic: MatchOutcome | None,
    ) -> float:
        """The single ``face_match_score`` the specification names.

        Defined as the selfie-to-profile comparison, which is what a reader
        will assume it means. Falls back to the best available
        selfie-to-photograph comparison, and finally to the CNIC, so the field
        is populated whenever *any* comparison succeeded rather than reading
        zero - which would be indistinguishable from a total mismatch.
        """
        if profile is not None and profile.compared:
            return profile.score
        if secondaries:
            return max(outcome.score for outcome in secondaries)
        if cnic is not None and cnic.compared:
            return cnic.score
        return 0.0


__all__ = ["IdentityAggregator", "IdentityAssessment"]
