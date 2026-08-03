"""MODULE 4 service - orchestrates comparison and identity fusion.

Pipeline for one verification::

    selfie embedding (reference)
        |
        +-- vs profile embedding      -> MatchOutcome
        +-- vs each secondary         -> MatchOutcome
        +-- vs CNIC portrait          -> MatchOutcome
                                          |
                                          v
                                   IdentityAggregator -> MatchingResult

Needs no model weights of its own: it consumes the embeddings Module 3
produced. That makes it fully unit-testable against synthetic vectors, which is
what lets the fusion rules be pinned exactly rather than sampled.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from hamqadam_ai.core.config import Settings, get_settings
from hamqadam_ai.core.constants import ImageRole, MatchDecision
from hamqadam_ai.embeddings.base import FaceEmbedding
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.matching.aggregator import IdentityAggregator, IdentityAssessment
from hamqadam_ai.matching.comparator import (
    ComparisonType,
    FaceComparator,
    MatchOutcome,
)
from hamqadam_ai.schemas.common import AnalysisWarning
from hamqadam_ai.schemas.matching import ComparisonResult, MatchingResult

log = get_logger(__name__)


class MatchingService:
    """Compares a verification's embeddings and fuses an identity confidence.

    Args:
        comparator: Pairwise comparison.
        aggregator: Identity fusion.
        settings: Service configuration.
    """

    __slots__ = ("_aggregator", "_comparator", "_config", "_settings")

    def __init__(
        self,
        *,
        comparator: FaceComparator,
        aggregator: IdentityAggregator,
        settings: Settings,
    ) -> None:
        self._comparator = comparator
        self._aggregator = aggregator
        self._settings = settings
        self._config = settings.matching

    # -- Public surface ----------------------------------------------------- #

    def match(
        self,
        *,
        selfie: FaceEmbedding | None,
        profile: FaceEmbedding | None = None,
        secondaries: Sequence[FaceEmbedding | None] | None = None,
        cnic: FaceEmbedding | None = None,
        secondary_labels: Sequence[str] | None = None,
    ) -> MatchingResult:
        """Run every comparison and fuse the result.

        Args:
            selfie: The live-selfie embedding, the reference for every
                comparison. Without it nothing can be compared.
            profile: The main profile image's embedding.
            secondaries: Embeddings for the optional secondary images. A
                ``None`` entry means that image could not be embedded and is
                reported as ``NOT_COMPARED`` rather than skipped, so the caller
                can see which upload failed.
            cnic: The CNIC portrait's embedding.
            secondary_labels: Identifiers echoed back onto each secondary
                outcome, so a caller can reassociate them with its uploads.

        Returns:
            The full matching result. A missing selfie is reported through
            ``identity_available`` rather than raised - the pipeline still
            needs the quality and OCR findings for the same request.
        """
        started = time.perf_counter()
        outcomes: list[MatchOutcome] = []

        outcomes.append(
            self._comparator.compare(
                selfie,
                profile,
                ComparisonType.PROFILE,
                target_role=ImageRole.PROFILE_IMAGE,
                target_label="profile",
            )
        )

        entries = list(secondaries or [])
        labels = list(secondary_labels or [])
        for index, candidate in enumerate(entries):
            label = labels[index] if index < len(labels) else f"secondary_{index + 1}"
            outcomes.append(
                self._comparator.compare(
                    selfie,
                    candidate,
                    ComparisonType.SECONDARY,
                    target_role=ImageRole.SECONDARY_IMAGE,
                    target_label=label,
                )
            )

        outcomes.append(
            self._comparator.compare(
                selfie,
                cnic,
                ComparisonType.CNIC,
                target_role=ImageRole.CNIC_PORTRAIT,
                target_label="cnic",
            )
        )

        assessment = self._aggregator.aggregate(outcomes)
        result = self._to_schema(
            assessment,
            selfie=selfie,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

        log.info("matching.completed", **result.summary())
        return result

    async def match_async(self, **kwargs: Any) -> MatchingResult:
        """Fuse without blocking the event loop.

        The work is pure arithmetic over 512-element vectors and completes in
        well under a millisecond, so this exists for interface symmetry with
        the other services rather than for throughput.
        """
        return await asyncio.to_thread(lambda: self.match(**kwargs))

    def compare_pair(
        self,
        reference: FaceEmbedding | None,
        candidate: FaceEmbedding | None,
        comparison: ComparisonType = ComparisonType.PROFILE,
    ) -> MatchOutcome:
        """Compare two embeddings directly.

        Used by Module 6 for the CNIC path and by Module 8 for duplicate
        adjudication, both of which need one comparison rather than a full
        verification fusion.
        """
        return self._comparator.compare(reference, candidate, comparison)

    def describe(self) -> dict[str, Any]:
        """Serialisable summary of the active configuration, for ``/health``."""
        config = self._config
        return {
            "thresholds": {
                "profile": {
                    "strong_match": config.selfie_vs_profile.strong_match,
                    "review": config.selfie_vs_profile.review,
                },
                "secondary": {
                    "strong_match": config.selfie_vs_secondary.strong_match,
                    "review": config.selfie_vs_secondary.review,
                },
                "cnic": {
                    "strong_match": config.selfie_vs_cnic.strong_match,
                    "review": config.selfie_vs_cnic.review,
                },
            },
            "calibration": {
                "strong_match_score": config.calibration.strong_match_score,
                "review_score": config.calibration.review_score,
                "floor_similarity": config.calibration.floor_similarity,
            },
            "identity": {
                "weights": config.identity.weights,
                "secondary_aggregation": config.identity.secondary_aggregation,
                "cnic_failure_cap": config.identity.cnic_failure_cap,
            },
            "thresholds_validated": False,
        }

    # -- Internals ----------------------------------------------------------- #

    def _to_schema(
        self,
        assessment: IdentityAssessment,
        *,
        selfie: FaceEmbedding | None,
        duration_ms: float,
    ) -> MatchingResult:
        """Render the assessment as the API response model."""
        comparisons = [
            ComparisonResult(
                comparison=str(outcome.comparison),
                compared=outcome.compared,
                decision=outcome.decision,
                score=round(outcome.score, 2),
                similarity=round(outcome.similarity, 6),
                confidence=round(outcome.confidence, 4),
                low_confidence=outcome.low_confidence,
                target_role=outcome.target_role,
                target_label=outcome.target_label,
                strong_match_threshold=outcome.detail.get("strong_match_threshold"),
                review_threshold=outcome.detail.get("review_threshold"),
                reason=outcome.reason,
            )
            for outcome in assessment.outcomes
        ]

        warnings: list[AnalysisWarning] = []
        if selfie is None:
            warnings.append(
                AnalysisWarning(
                    code="MATCHING_NO_REFERENCE",
                    message=(
                        "The live selfie could not be embedded, so nothing could "
                        "be compared against it. Every comparison in this "
                        "request is reported as NOT_COMPARED."
                    ),
                    stage="matching",
                )
            )
        for reason in assessment.reasons:
            warnings.append(
                AnalysisWarning(
                    code="MATCHING_OBSERVATION", message=reason, stage="matching"
                )
            )

        # The operating points have not been derived from a labelled corpus of
        # this deployment's own traffic. Saying so in every response is more
        # useful than a note in a document nobody reads at 3am.
        warnings.append(
            AnalysisWarning(
                code="MATCHING_THRESHOLDS_UNVALIDATED",
                message=(
                    "Match thresholds are engineering defaults from the ArcFace "
                    "literature, not values derived from this deployment's own "
                    "labelled data. Run scripts/evaluate_matching.py against a "
                    "labelled corpus and re-pin them before production use."
                ),
                stage="matching",
            )
        )

        cnic_outcome = next(
            (
                outcome
                for outcome in assessment.outcomes
                if outcome.comparison is ComparisonType.CNIC
            ),
            None,
        )
        cnic_match: bool | None = None
        if cnic_outcome is not None and cnic_outcome.compared:
            cnic_match = cnic_outcome.decision is MatchDecision.STRONG_MATCH

        return MatchingResult(
            face_match_score=round(assessment.face_match_score, 2),
            profile_face_match_score=(
                round(assessment.profile_score, 2)
                if assessment.profile_score is not None
                else None
            ),
            secondary_face_match_scores=list(assessment.secondary_scores),
            cnic_face_match_score=(
                round(assessment.cnic_score, 2)
                if assessment.cnic_score is not None
                else None
            ),
            identity_confidence_score=assessment.identity_confidence,
            identity_available=assessment.available,
            cnic_identity_match=cnic_match,
            any_comparison_failed=assessment.any_failed,
            capped_by=assessment.capped_by,
            comparisons=comparisons,
            secondary_worst_score=assessment.secondary_worst,
            contributions=assessment.contributions,
            effective_weights=assessment.effective_weights,
            reasons=assessment.reasons,
            warnings=warnings,
            model_version=selfie.model_version if selfie else "",
            thresholds_validated=False,
            duration_ms=duration_ms,
        )


def build_matching_service(settings: Settings | None = None) -> MatchingService:
    """Wire up a :class:`MatchingService` from configuration.

    Args:
        settings: Service configuration. Loaded from the environment when
            omitted.

    Returns:
        A ready service. Needs no model weights, so it cannot fail to start.
    """
    settings = settings or get_settings()
    config = settings.matching

    log.info(
        "matching.service_ready",
        profile_strong=config.selfie_vs_profile.strong_match,
        cnic_strong=config.selfie_vs_cnic.strong_match,
        identity_weights=config.identity.weights,
    )

    return MatchingService(
        comparator=FaceComparator(config),
        aggregator=IdentityAggregator(config),
        settings=settings,
    )


__all__ = ["MatchingService", "build_matching_service"]
