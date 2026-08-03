"""MODULE 7 response contract - profile image analysis.

Answers a question the identity modules do not: **is this a genuine camera
photograph, taken by the person uploading it?** A profile photo that is a
screenshot of somebody else's social media, a picture of a laptop screen, or a
cartoon avatar is a different failure from a photo of the wrong person, and
the Backend needs to tell a user something different about each.

Reading ``authenticity_score`` correctly
----------------------------------------
It is an **upper bound on suspicion, not a measure of trust**. 100 means four
specific tests found nothing; it does not mean the image is genuine. A cropped
screenshot, a screen capture taken far enough away to lose its moire, or a
competent composite all score 100. Treat a low score as evidence and a high
score as the absence of evidence, because that is all it is.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, computed_field

from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.schemas.common import (
    AnalysisWarning,
    OutputModel,
    PercentScore,
    UnitScore,
)


class AuthenticitySignalModel(OutputModel):
    """One detector's reading, with the evidence behind it."""

    name: str = Field(description="Detector identifier.")
    triggered: bool = Field(description="Whether it found what it looks for.")
    confidence: UnitScore = Field(
        description=(
            "Strength of the evidence for this finding. A low value means "
            "weak evidence *for* the finding, not evidence against it."
        )
    )
    measurements: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Raw values behind the verdict. Reported verbatim because these "
            "are what an engineer recalibrates against, and because a "
            "rejection nobody can explain cannot be appealed."
        ),
    )
    note: str | None = Field(
        default=None, description="Why the detector could not measure."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def measured(self) -> bool:
        """Whether the detector produced a usable reading.

        Exposed rather than left to the caller to derive from ``note``,
        because the distinction matters and is easy to miss: a detector that
        stayed silent because it looked and found nothing is telling you
        something, and one that could not run at all is not.
        """
        return self.note is None


class AuthenticityFindingModel(OutputModel):
    """Something found wrong with the image, and what to do about it."""

    code: str = Field(description="Stable machine-readable identifier.")
    detector: str = Field(description="Which detector produced it.")
    confidence: UnitScore = Field(description="Strength of the evidence.")
    message: str = Field(description="Actionable guidance for the user.")


class ProfileAnalysisResult(OutputModel):
    """MODULE 7 output for one user-supplied photograph."""

    # -- Headline ------------------------------------------------------- #
    usable_as_profile: bool = Field(
        description=(
            "Whether this photograph can serve as a profile picture: a "
            "genuine-looking capture, of one person, of adequate quality. A "
            "recommendation to the Backend's rules engine, not a decision."
        )
    )
    authenticity_score: PercentScore = Field(
        description=(
            "0-100. An upper bound on suspicion, NOT a measure of trust. 100 "
            "means four specific tests found nothing - a cropped screenshot "
            "or a distant screen capture also scores 100."
        )
    )
    is_probably_genuine_capture: bool = Field(
        description=(
            "Whether the authenticity score cleared its threshold. Same "
            "caveat: absence of evidence, not evidence of absence."
        )
    )

    # -- What was found ------------------------------------------------- #
    findings: list[AuthenticityFindingModel] = Field(
        default_factory=list,
        description="Detectors that triggered, strongest evidence first.",
    )
    signals: list[AuthenticitySignalModel] = Field(
        default_factory=list,
        description="Every detector's reading, including the silent ones.",
    )

    # -- Subject -------------------------------------------------------- #
    face_detected: bool = Field(
        default=False, description="Whether a qualifying face was found."
    )
    face_count: int = Field(
        default=0, ge=0, description="How many qualifying faces were found."
    )
    is_group_photo: bool = Field(
        default=False,
        description=(
            "More than one qualifying face. Not dishonest - but ambiguous "
            "about which person the account belongs to."
        ),
    )
    face_visibility_score: PercentScore = Field(
        default=0.0, description="Visibility of the primary face, 0-100."
    )

    # -- Quality -------------------------------------------------------- #
    image_quality_score: PercentScore = Field(
        default=0.0, description="Module 2 composite, on this role's scale."
    )
    quality_usable: bool = Field(
        default=False, description="Whether quality cleared this role's bar."
    )

    # -- Provenance ----------------------------------------------------- #
    role: ImageRole = Field(description="Which image in the request this is.")
    image_width: int = Field(default=0, description="Source width in pixels.")
    image_height: int = Field(default=0, description="Source height in pixels.")

    error_code: ErrorCode | None = Field(default=None, description="Failure code.")
    error_message: str | None = Field(
        default=None, description="Actionable guidance for the user."
    )
    warnings: list[AnalysisWarning] = Field(
        default_factory=list, description="Non-fatal findings."
    )
    duration_ms: float = Field(default=0.0, description="Wall-clock time.")

    def summary(self) -> dict[str, Any]:
        """PII-free summary for logging."""
        return {
            "usable": self.usable_as_profile,
            "authenticity": self.authenticity_score,
            "findings": [finding.code for finding in self.findings],
            "faces": self.face_count,
            "quality": self.image_quality_score,
            "role": str(self.role),
            "error_code": str(self.error_code) if self.error_code else None,
            "duration_ms": round(self.duration_ms, 1),
        }


__all__ = [
    "AuthenticityFindingModel",
    "AuthenticitySignalModel",
    "ProfileAnalysisResult",
]
