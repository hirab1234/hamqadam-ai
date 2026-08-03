"""MODULE 4 response contract - face matching.

Carries every field the specification names for this stage:
``face_match_score``, ``cnic_face_match_score``, ``profile_face_match_score``,
``secondary_face_match_scores[]`` and ``identity_confidence_score``.

Both the calibrated score and the raw cosine are reported for every
comparison. The calibrated score is what a human or a rules engine should read;
the raw similarity is what a threshold is actually applied to, and exposing it
is what lets an engineer recalibrate against production traffic without having
to re-run the whole pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, computed_field

from hamqadam_ai.core.constants import ImageRole, MatchDecision
from hamqadam_ai.schemas.common import (
    AnalysisWarning,
    OutputModel,
    PercentScore,
    Similarity,
    UnitScore,
)


class ComparisonResult(OutputModel):
    """One comparison of the live selfie against another image."""

    comparison: str = Field(
        description="Which pair this relates: profile, secondary or cnic."
    )
    compared: bool = Field(
        description=(
            "False when one side was missing or unusable. Distinct from a "
            "failed match: 'there was no CNIC portrait' and 'the CNIC portrait "
            "is a different person' are opposite findings."
        )
    )
    decision: MatchDecision = Field(
        description="STRONG_MATCH, REVIEW, FAILED or NOT_COMPARED."
    )
    score: PercentScore = Field(
        default=0.0,
        description=(
            "Calibrated 0-100 score. Interpolated through the decision "
            "boundaries, so 75+ is a strong match and below 50 failed - "
            "regardless of which comparison type produced it."
        ),
    )
    similarity: Similarity = Field(
        default=0.0,
        description=(
            "Raw cosine similarity in [-1, 1]. The quantity thresholds are "
            "applied to. Note that this is NOT a linear function of `score`."
        ),
    )
    confidence: UnitScore = Field(
        default=0.0,
        description=(
            "How far this comparison can be relied on. The minimum of the two "
            "embeddings' confidences - a comparison is only as good as its "
            "weaker side."
        ),
    )
    low_confidence: bool = Field(
        default=False,
        description="Whether confidence fell below the configured floor.",
    )
    target_role: ImageRole | None = Field(
        default=None, description="Which image was compared against the selfie."
    )
    target_label: str | None = Field(
        default=None,
        description="Caller-supplied identifier, echoed back for reassociation.",
    )
    strong_match_threshold: float | None = Field(
        default=None,
        description="The cosine threshold this comparison was judged against.",
    )
    review_threshold: float | None = Field(
        default=None, description="The cosine review boundary for this type."
    )
    reason: str | None = Field(
        default=None, description="Why the comparison could not be made."
    )

    def summary(self) -> dict[str, Any]:
        """Compact summary for logging."""
        return {
            "comparison": self.comparison,
            "decision": str(self.decision),
            "score": self.score,
            "similarity": round(self.similarity, 4),
        }


class MatchingResult(OutputModel):
    """MODULE 4 output for one verification request."""

    # -- Specification-mandated scalars --------------------------------- #
    face_match_score: PercentScore = Field(
        description=(
            "Headline match score. The selfie-to-profile comparison where one "
            "exists, otherwise the best available selfie-to-photograph "
            "comparison."
        )
    )
    profile_face_match_score: PercentScore | None = Field(
        default=None, description="Live selfie against the main profile image."
    )
    secondary_face_match_scores: list[float] = Field(
        default_factory=list,
        description="Live selfie against each secondary image, in submission order.",
    )
    cnic_face_match_score: PercentScore | None = Field(
        default=None, description="Live selfie against the portrait on the CNIC."
    )
    identity_confidence_score: PercentScore | None = Field(
        default=None,
        description=(
            "Fused confidence that the person in the selfie is who the request "
            "claims. Null when too few comparisons succeeded for the number to "
            "mean anything - which is reported honestly rather than guessed."
        ),
    )

    # -- Verdict ------------------------------------------------------------ #
    identity_available: bool = Field(
        description="Whether enough comparisons succeeded to fuse a confidence."
    )
    cnic_identity_match: bool | None = Field(
        default=None,
        description=(
            "Whether the CNIC portrait matched outright. Null when no CNIC "
            "comparison was possible."
        ),
    )
    any_comparison_failed: bool = Field(
        default=False,
        description=(
            "Whether any comparison actively contradicted the identity claim. "
            "Distinct from 'not every comparison matched' - a REVIEW outcome is "
            "inconclusive, FAILED is positive evidence of a different person."
        ),
    )
    capped_by: str | None = Field(
        default=None,
        description=(
            "Set when a rule limited identity confidence below its computed "
            "value; `cnic_failure` when the document comparison failed."
        ),
    )

    # -- Evidence ------------------------------------------------------------ #
    comparisons: list[ComparisonResult] = Field(
        default_factory=list, description="Every comparison attempted."
    )
    secondary_worst_score: float | None = Field(
        default=None,
        description=(
            "The weakest individual secondary score. Reported separately from "
            "the aggregate because a single mismatching secondary image is a "
            "fraud signal that the mean would dilute."
        ),
    )
    contributions: dict[str, float] = Field(
        default_factory=dict,
        description="Each comparison type's weighted contribution to the fusion.",
    )
    effective_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Weights actually applied, after renormalising around unavailable "
            "comparisons and down-weighting low-confidence ones."
        ),
    )
    reasons: list[str] = Field(
        default_factory=list, description="Human-readable notes on the fusion."
    )
    warnings: list[AnalysisWarning] = Field(
        default_factory=list, description="Non-fatal observations."
    )

    model_version: str = Field(
        default="",
        description=(
            "Recogniser version every comparison used. Comparisons across "
            "versions are refused, not attempted."
        ),
    )
    thresholds_validated: bool = Field(
        default=False,
        description=(
            "Whether the operating points were derived from a labelled dataset. "
            "False means they are engineering defaults from the literature and "
            "have not been validated against this deployment's own traffic."
        ),
    )
    duration_ms: float = Field(
        default=0.0, description="Time spent matching, in milliseconds."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def comparisons_made(self) -> int:
        """How many comparisons actually ran."""
        return sum(1 for entry in self.comparisons if entry.compared)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def strong_matches(self) -> int:
        """How many comparisons passed outright."""
        return sum(
            1
            for entry in self.comparisons
            if entry.decision is MatchDecision.STRONG_MATCH
        )

    def summary(self) -> dict[str, Any]:
        """Compact, PII-free summary for logging."""
        return {
            "identity_confidence": self.identity_confidence_score,
            "face_match": self.face_match_score,
            "profile": self.profile_face_match_score,
            "cnic": self.cnic_face_match_score,
            "secondary_n": len(self.secondary_face_match_scores),
            "compared": self.comparisons_made,
            "strong": self.strong_matches,
            "any_failed": self.any_comparison_failed,
            "capped_by": self.capped_by,
            "duration_ms": round(self.duration_ms, 1),
        }


__all__ = ["ComparisonResult", "MatchingResult"]
