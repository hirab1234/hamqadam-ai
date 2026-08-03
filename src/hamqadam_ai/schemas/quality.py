"""MODULE 2 response contract - image and face quality.

Carries the four fields the specification names explicitly
(``image_quality_score``, ``blur_score``, ``brightness_score``,
``noise_score``) plus the five further dimensions it asks to be *detected*
(contrast, resolution, face sharpness, distortion, pixelation), each with the
raw measurements behind it.

The raw measurements are part of the contract, not debug output. They are what
lets an engineer recalibrate a threshold against real traffic, and what lets a
human reviewer see that a rejected image failed on a single borderline
dimension rather than being uniformly bad.
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
)


class MetricDetail(OutputModel):
    """One quality dimension, with the evidence behind its score."""

    score: PercentScore = Field(description="Dimension score on the 0-100 scale.")
    measured: bool = Field(
        default=True,
        description=(
            "False when this dimension could not be computed - no face for "
            "sharpness, no landmarks for geometric distortion. An unmeasured "
            "dimension is excluded from the composite, not scored zero."
        ),
    )
    measurements: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Raw values behind the score, in their natural units. These are the "
            "numbers thresholds are calibrated against."
        ),
    )
    sub_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-sub-metric normalised scores in [0, 1], before weighting.",
    )
    limiting_factor: str | None = Field(
        default=None, description="Sub-metric contributing the largest deficit."
    )
    note: str | None = Field(
        default=None, description="Human-readable explanation, when notable."
    )


class QualityResult(OutputModel):
    """MODULE 2 output for one image."""

    # -- Specification-mandated scalars --------------------------------- #
    image_quality_score: PercentScore = Field(
        description="Composite quality on the 0-100 scale."
    )
    blur_score: PercentScore = Field(
        description="Whole-image focus, 0-100. Higher is sharper."
    )
    brightness_score: PercentScore = Field(
        description="Exposure quality, 0-100. Penalises both extremes."
    )
    noise_score: PercentScore = Field(
        description="Noise quality, 0-100. Higher means cleaner."
    )

    # -- The remaining required dimensions ------------------------------- #
    contrast_score: PercentScore = Field(description="Tonal separation, 0-100.")
    resolution_score: PercentScore = Field(
        description="Genuine as-captured resolution of the face, 0-100."
    )
    sharpness_score: PercentScore = Field(
        description="Face-region focus, 0-100. Distinct from whole-image blur."
    )
    distortion_score: PercentScore = Field(
        description="Freedom from posterisation, fringing and stretching, 0-100."
    )
    pixelation_score: PercentScore = Field(
        description="Freedom from block artefacts and upscaling, 0-100."
    )

    # -- Verdict ---------------------------------------------------------- #
    role: ImageRole = Field(description="Which image in the request this describes.")
    usable: bool = Field(
        description=(
            "True when the image clears its role threshold and every critical "
            "floor, and may proceed to the next pipeline stage."
        )
    )
    min_required: PercentScore = Field(
        description="Role-specific acceptance threshold this was judged against."
    )
    limiting_factor: str | None = Field(
        default=None,
        description="The dimension contributing the largest weighted deficit.",
    )
    critical_failures: list[str] = Field(
        default_factory=list,
        description=(
            "Dimensions that fell at or below the critical floor. Any entry "
            "here makes the image unusable regardless of the composite score."
        ),
    )
    unmeasured: list[str] = Field(
        default_factory=list,
        description="Dimensions that could not be computed for this image.",
    )

    # -- Evidence ---------------------------------------------------------- #
    metrics: dict[str, MetricDetail] = Field(
        default_factory=dict,
        description="Every dimension with its raw measurements and sub-scores.",
    )
    effective_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Weights actually applied, after renormalising around any "
            "unmeasured dimension."
        ),
    )
    arithmetic_score: PercentScore = Field(
        default=0.0,
        description=(
            "The plain weighted mean, for comparison. The composite uses a "
            "power mean with exponent < 1, which penalises a single "
            "catastrophic dimension instead of averaging it away; publishing "
            "both makes that adjustment auditable."
        ),
    )

    error_code: ErrorCode | None = Field(
        default=None, description="Set to LOW_IMAGE_QUALITY when unusable."
    )
    error_message: str | None = Field(
        default=None, description="Human-readable explanation of the rejection."
    )
    warnings: list[AnalysisWarning] = Field(
        default_factory=list, description="Non-fatal observations."
    )

    image_width: int = Field(gt=0, description="Source image width in pixels.")
    image_height: int = Field(gt=0, description="Source image height in pixels.")
    face_analysed: bool = Field(
        description=(
            "Whether a face region was available. Without one, exposure, "
            "contrast and noise fall back to whole-image measurement and "
            "sharpness is not computed at all."
        )
    )
    duration_ms: float = Field(
        default=0.0, description="Time spent assessing this image, in milliseconds."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def degraded_dimensions(self) -> list[str]:
        """Dimensions scoring below 50, for a quick triage view."""
        return sorted(
            name
            for name, detail in self.metrics.items()
            if detail.measured and detail.score < 50.0
        )

    def summary(self) -> dict[str, Any]:
        """Compact, PII-free summary for logging."""
        return {
            "role": str(self.role),
            "overall": self.image_quality_score,
            "usable": self.usable,
            "limiting": self.limiting_factor,
            "blur": self.blur_score,
            "sharpness": self.sharpness_score,
            "brightness": self.brightness_score,
            "noise": self.noise_score,
            "resolution": self.resolution_score,
            "critical": list(self.critical_failures),
            "duration_ms": round(self.duration_ms, 1),
        }


__all__ = ["MetricDetail", "QualityResult"]
