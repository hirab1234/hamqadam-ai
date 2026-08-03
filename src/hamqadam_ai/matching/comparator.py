"""Comparing one pair of face embeddings.

The live selfie is the reference for every comparison in the request. That is
deliberate: it is the only image captured under the app's control, so it is
the closest thing to a trusted sample available. Everything else - the profile
photographs the user chose to upload, the portrait printed on the CNIC - is
compared *against* it.

Confidence propagation
----------------------
A comparison is only as trustworthy as its weaker embedding, so the pair takes
the **minimum** of the two confidences rather than their mean. An excellent
selfie compared against a box-aligned profile photograph is not a moderately
confident comparison; it is limited by the bad one, and Module 3's measurement
quantifies how badly - a box-aligned face scores 0.786 against a properly
aligned embedding of the same person, and 0.573 once the head is tilted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hamqadam_ai.core.config import MatchingConfig
from hamqadam_ai.core.constants import ImageRole, MatchDecision
from hamqadam_ai.embeddings.base import FaceEmbedding
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.matching.similarity import calibrate_score, decide

log = get_logger(__name__)


class ComparisonType(StrEnum):
    """Which pair of images a comparison relates.

    Each has its own operating point. A CNIC portrait is a sub-300-dpi print
    photographed through a laminate and cannot be held to the same bar as two
    phone selfies.
    """

    PROFILE = "profile"
    SECONDARY = "secondary"
    CNIC = "cnic"


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    """The result of comparing the live selfie against one other image.

    Attributes:
        comparison: Which pair this relates.
        compared: False when one side was missing or unusable, in which case
            every score is zero and the decision is ``NOT_COMPARED``.
        similarity: Raw cosine similarity in ``[-1, 1]``. The internal
            quantity; thresholds are applied to this.
        score: The calibrated 0-100 score. The reporting quantity.
        decision: ``STRONG_MATCH``, ``REVIEW``, ``FAILED`` or ``NOT_COMPARED``.
        confidence: How much this comparison can be relied on, in ``[0, 1]``.
            The minimum of the two embeddings' confidences.
        low_confidence: Whether ``confidence`` fell below the configured floor.
        target_role: Which image was compared against the selfie.
        target_label: Free-form identifier, so a caller can reassociate a
            secondary-image outcome with the upload it came from.
        reason: Why the comparison could not be made, when it could not.
    """

    comparison: ComparisonType
    compared: bool
    similarity: float = 0.0
    score: float = 0.0
    decision: MatchDecision = MatchDecision.NOT_COMPARED
    confidence: float = 0.0
    low_confidence: bool = False
    target_role: ImageRole | None = None
    target_label: str | None = None
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def matched(self) -> bool:
        """Whether this comparison passed outright."""
        return self.decision is MatchDecision.STRONG_MATCH

    @property
    def failed(self) -> bool:
        """Whether this comparison actively contradicts the identity claim.

        Distinct from ``not matched``: a REVIEW outcome is inconclusive, while
        FAILED is positive evidence of a different person.
        """
        return self.decision is MatchDecision.FAILED

    @property
    def effective_weight(self) -> float:
        """Multiplier this outcome carries in aggregation.

        Zero when nothing was compared, reduced when the pair was
        low-confidence, one otherwise.
        """
        if not self.compared:
            return 0.0
        return 1.0

    def describe(self) -> dict[str, Any]:
        """PII-free summary for logging."""
        return {
            "comparison": str(self.comparison),
            "label": self.target_label,
            "compared": self.compared,
            "similarity": round(self.similarity, 4),
            "score": round(self.score, 2),
            "decision": str(self.decision),
            "confidence": round(self.confidence, 3),
            "low_confidence": self.low_confidence,
        }

    @classmethod
    def not_compared(
        cls,
        comparison: ComparisonType,
        *,
        reason: str,
        target_role: ImageRole | None = None,
        target_label: str | None = None,
    ) -> MatchOutcome:
        """Build the outcome for a comparison that could not be made.

        Distinguished from a failed comparison throughout. "There was no CNIC
        portrait to compare against" and "the CNIC portrait is a different
        person" are opposite findings, and collapsing them would let a missing
        document read as a passing one, or a failing one as an absent one.
        """
        return cls(
            comparison=comparison,
            compared=False,
            decision=MatchDecision.NOT_COMPARED,
            target_role=target_role,
            target_label=target_label,
            reason=reason,
        )


class FaceComparator:
    """Compares embeddings against the live-selfie reference.

    Args:
        config: The matching section of the settings.
    """

    __slots__ = ("_config",)

    def __init__(self, config: MatchingConfig) -> None:
        self._config = config

    def compare(
        self,
        reference: FaceEmbedding | None,
        candidate: FaceEmbedding | None,
        comparison: ComparisonType,
        *,
        target_role: ImageRole | None = None,
        target_label: str | None = None,
    ) -> MatchOutcome:
        """Compare one candidate against the reference selfie.

        Args:
            reference: The live-selfie embedding. ``None`` when the selfie
                could not be embedded, which makes every comparison impossible.
            candidate: The embedding to compare against it.
            comparison: Which comparison type this is, selecting the operating
                point.
            target_role: Which image the candidate came from.
            target_label: Free-form identifier echoed onto the outcome.

        Returns:
            The outcome. A missing or unusable side yields ``NOT_COMPARED``
            rather than a zero score, because those mean different things.
        """
        if reference is None:
            return MatchOutcome.not_compared(
                comparison,
                reason="the live selfie could not be embedded",
                target_role=target_role,
                target_label=target_label,
            )
        if candidate is None:
            return MatchOutcome.not_compared(
                comparison,
                reason="the comparison image could not be embedded",
                target_role=target_role,
                target_label=target_label,
            )

        if reference.model_version != candidate.model_version:
            # Vectors from different networks occupy unrelated spaces. Refusing
            # is the only correct behaviour: a cosine between them is a number,
            # but it is not a similarity.
            log.error(
                "matching.model_version_mismatch",
                comparison=str(comparison),
                reference_version=reference.model_version,
                candidate_version=candidate.model_version,
            )
            return MatchOutcome.not_compared(
                comparison,
                reason=(
                    f"embeddings come from different model versions "
                    f"({reference.model_version} vs {candidate.model_version}) "
                    f"and are not comparable"
                ),
                target_role=target_role,
                target_label=target_label,
            )

        if reference.is_degenerate or candidate.is_degenerate:
            return MatchOutcome.not_compared(
                comparison,
                reason="one of the embeddings carries no usable direction",
                target_role=target_role,
                target_label=target_label,
            )

        similarity = reference.similarity_to(candidate)
        thresholds = self._config.thresholds_for(str(comparison))
        score = calibrate_score(similarity, thresholds, self._config.calibration)
        decision = decide(similarity, thresholds)

        # The weaker embedding governs. See the module docstring.
        confidence = min(reference.confidence, candidate.confidence)
        low_confidence = confidence < self._config.confidence.min_pair_confidence

        return MatchOutcome(
            comparison=comparison,
            compared=True,
            similarity=similarity,
            score=score,
            decision=decision,
            confidence=confidence,
            low_confidence=low_confidence,
            target_role=target_role,
            target_label=target_label,
            detail={
                "strong_match_threshold": thresholds.strong_match,
                "review_threshold": thresholds.review,
                "reference_confidence": round(reference.confidence, 4),
                "candidate_confidence": round(candidate.confidence, 4),
                "reference_aligned": reference.aligned,
                "candidate_aligned": candidate.aligned,
            },
        )

    def compare_many(
        self,
        reference: FaceEmbedding | None,
        candidates: list[tuple[FaceEmbedding | None, ImageRole | None, str | None]],
        comparison: ComparisonType,
    ) -> list[MatchOutcome]:
        """Compare several candidates of the same type against the reference.

        Args:
            reference: The live-selfie embedding.
            candidates: ``(embedding, role, label)`` triples.
            comparison: The comparison type shared by all of them.

        Returns:
            One outcome per candidate, in order.
        """
        return [
            self.compare(
                reference,
                candidate,
                comparison,
                target_role=role,
                target_label=label,
            )
            for candidate, role, label in candidates
        ]


__all__ = ["ComparisonType", "FaceComparator", "MatchOutcome"]
