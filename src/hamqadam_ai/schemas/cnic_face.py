"""MODULE 6 response contract - CNIC portrait extraction and matching.

Carries ``cnic_face_match_score`` and ``cnic_identity_match``, which Module 4's
contract also names, plus the evidence behind them: where the portrait was
found, how plausible its placement was, whether the card also carried a ghost
reproduction, and - the field that matters most for fraud - how many faces were
present that are **not** plausibly printed on the card.

That last count is the one to read when a match score looks too good. A live
face held behind the card produces a near-perfect comparison and a non-zero
``foreign_face_count``; the score alone cannot distinguish it from a genuine
verification, and the count can.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.schemas.common import (
    AnalysisWarning,
    OutputModel,
    PercentScore,
    Similarity,
    UnitScore,
)
from hamqadam_ai.schemas.detection import BoundingBoxModel


class PortraitCandidateModel(OutputModel):
    """One face considered as the card's printed portrait."""

    box: BoundingBoxModel = Field(
        description="Face bounds, in the coordinates of the image searched."
    )
    detector_confidence: UnitScore = Field(description="Detector objectness.")
    area_ratio: float = Field(
        description=(
            "Face area as a fraction of the card image. Bounded by physical "
            "proportion: an ID-1 photo box is a fixed size, so a printed "
            "portrait's face occupies a bounded share of the card at any "
            "capture resolution."
        )
    )
    band_score: UnitScore = Field(
        description=(
            "How squarely the face sits in a template photo-box band; 1.0 "
            "inside, falling to 0.0 outside the tolerance."
        )
    )
    plausibility: UnitScore = Field(
        description="Combined rank score used to choose between candidates."
    )
    rejections: list[str] = Field(
        default_factory=list,
        description="Why this face cannot be the portrait. Empty means it can.",
    )


class CnicPortraitResult(OutputModel):
    """Everything the portrait search found on one card image."""

    found: bool = Field(description="Whether a usable portrait was located.")
    usable: bool = Field(
        default=False,
        description=(
            "Whether the portrait was also good enough to embed. A portrait "
            "can be found and still be unusable - an illegible print yields a "
            "vector that is not wrong so much as meaningless."
        ),
    )

    portrait: PortraitCandidateModel | None = Field(
        default=None, description="The chosen face."
    )
    quality_score: PercentScore | None = Field(
        default=None,
        description=(
            "Portrait quality on the CNIC scale, which is far laxer than the "
            "live-selfie scale. A sub-300-dpi print behind laminate cannot "
            "meet a selfie's bar and is not asked to."
        ),
    )

    has_ghost: bool = Field(
        default=False,
        description=(
            "A second face was found on the card. Expected: modern cards "
            "print a faded reproduction of the portrait as a security "
            "feature. Not a fraud signal."
        ),
    )
    foreign_face_count: int = Field(
        default=0,
        description=(
            "Faces present that are not plausibly printed on the card - too "
            "large, or outside the photo-box region. A card held in front of "
            "a face lands here. Not proof of fraud on its own; a poster on "
            "the wall behind would also do it."
        ),
    )
    candidates: list[PortraitCandidateModel] = Field(
        default_factory=list, description="Every face considered."
    )

    rectified: bool = Field(
        default=False,
        description=(
            "Whether the card was isolated to its own quadrilateral before "
            "searching. When true, nothing outside the card was even visible "
            "to the detector, which is the strongest form of the containment "
            "guarantee."
        ),
    )
    card_coverage: UnitScore = Field(
        default=1.0, description="Share of the frame the card occupied."
    )
    searched_width: int = Field(default=0, description="Width searched, in pixels.")
    searched_height: int = Field(default=0, description="Height searched, in pixels.")

    error_code: ErrorCode | None = Field(default=None, description="Failure code.")
    error_message: str | None = Field(
        default=None, description="Actionable guidance for the user."
    )

    def summary(self) -> dict[str, Any]:
        """PII-free summary for logging. Geometry only, never pixels."""
        return {
            "found": self.found,
            "usable": self.usable,
            "quality": self.quality_score,
            "has_ghost": self.has_ghost,
            "foreign_faces": self.foreign_face_count,
            "rectified": self.rectified,
            "candidates": len(self.candidates),
            "error_code": str(self.error_code) if self.error_code else None,
        }


class CnicFaceMatchResult(OutputModel):
    """The selfie compared against the portrait printed on the CNIC."""

    success: bool = Field(
        description=(
            "Whether a comparison was actually performed. False means no "
            "score exists - which is different from a score that failed."
        )
    )
    cnic_identity_match: bool | None = Field(
        default=None,
        description=(
            "Whether the selfie and the CNIC portrait are the same person. "
            "``null`` when no comparison could be made; the Backend must not "
            "read that as False."
        ),
    )
    cnic_face_match_score: PercentScore | None = Field(
        default=None,
        description=(
            "Calibrated 0-100 score for the comparison. Interpolated through "
            "the CNIC decision boundaries, which sit lower than the "
            "selfie-to-selfie ones because print-versus-live is intrinsically "
            "a harder comparison."
        ),
    )
    similarity: Similarity | None = Field(
        default=None, description="Raw cosine, before calibration."
    )
    decision: str | None = Field(
        default=None, description="MATCHED, REVIEW or FAILED."
    )
    match_confidence: UnitScore | None = Field(
        default=None,
        description=(
            "How much weight to put on the score, given the quality of both "
            "sides. A confident score from two poor images does not exist."
        ),
    )

    portrait: CnicPortraitResult = Field(
        description="What the portrait search found."
    )

    strong_match_threshold: float | None = Field(
        default=None, description="Cosine above which this counts as matched."
    )
    review_threshold: float | None = Field(
        default=None, description="Cosine below which this counts as failed."
    )
    thresholds_validated: bool = Field(
        default=False,
        description=(
            "Whether the operating point has been validated on Pakistani "
            "print-versus-live pairs. Always false here: no such dataset has "
            "been used, and reporting otherwise would be a claim nobody made."
        ),
    )

    error_code: ErrorCode | None = Field(default=None, description="Failure code.")
    error_message: str | None = Field(
        default=None, description="Actionable guidance for the user."
    )
    warnings: list[AnalysisWarning] = Field(
        default_factory=list, description="Non-fatal findings."
    )

    model_version: str = Field(
        default="", description="Recognition model that produced the vectors."
    )
    duration_ms: float = Field(default=0.0, description="Wall-clock time.")

    def summary(self) -> dict[str, Any]:
        """PII-free summary for logging."""
        return {
            "success": self.success,
            "match": self.cnic_identity_match,
            "score": self.cnic_face_match_score,
            "decision": self.decision,
            "confidence": self.match_confidence,
            "foreign_faces": self.portrait.foreign_face_count,
            "portrait_found": self.portrait.found,
            "error_code": str(self.error_code) if self.error_code else None,
            "duration_ms": round(self.duration_ms, 1),
        }


__all__ = [
    "CnicFaceMatchResult",
    "CnicPortraitResult",
    "PortraitCandidateModel",
]
