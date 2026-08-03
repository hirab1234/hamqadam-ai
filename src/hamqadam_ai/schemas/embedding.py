"""MODULE 3 response contract - face embeddings.

The vector itself is **not** part of the default API response. An embedding is
biometric data under GDPR Article 9 and is reversible enough, through model
inversion, that returning it over HTTP would be handing out the template
rather than the decision made from it. It is carried internally between
pipeline stages and written to the vector store; the response reports the
metadata a caller legitimately needs - dimension, confidence, provenance -
and includes the vector only when a caller explicitly asks, which the Backend
does when enrolling a user into the duplicate-detection index.
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


class EmbeddingResult(OutputModel):
    """MODULE 3 output for one face."""

    success: bool = Field(
        description="Whether a usable embedding was produced for this face."
    )
    role: ImageRole = Field(description="Which image in the request this describes.")

    dimension: int = Field(
        default=0, ge=0, description="Length of the produced vector."
    )
    vector: list[float] | None = Field(
        default=None,
        description=(
            "The L2-normalised embedding. Omitted unless the caller explicitly "
            "requested it - this is biometric data, not a diagnostic."
        ),
    )

    # -- Confidence ------------------------------------------------------- #
    raw_norm: float = Field(
        default=0.0,
        description=(
            "L2 norm of the network output before normalisation. Reported as a "
            "diagnostic only. It is tempting to read this as a quality signal - "
            "that is what MagFace and AdaFace exploit - but measurement on this "
            "artefact refutes it: across a twelve-step degradation ladder the "
            "norm varied only 22.8-25.5, and a heavily blurred face measured "
            "higher than the pristine one. Nothing is decided on this value."
        ),
    )
    confidence: UnitScore = Field(
        default=0.0,
        description=(
            "How well the face aligned onto the recognition template, in "
            "[0, 1]. Derived from the alignment residual, which measurement "
            "shows is both stable and meaningful, and set to a fixed penalised "
            "value when the box-only fallback ran."
        ),
    )
    confidence_score: PercentScore = Field(
        default=0.0, description="The same confidence on the 0-100 reporting scale."
    )
    low_confidence: bool = Field(
        default=False,
        description=(
            "True when confidence fell below the configured floor. The "
            "embedding is still returned and still compared; the decision "
            "engine weighs the resulting match score less."
        ),
    )

    # -- Provenance -------------------------------------------------------- #
    aligned: bool = Field(
        default=False,
        description=(
            "True when the face was warped onto the landmark template. False "
            "means the box-only fallback ran, which recovers scale but not "
            "rotation, and accuracy is materially lower."
        ),
    )
    alignment_residual: float | None = Field(
        default=None,
        description=(
            "Mean landmark-to-template distance after warping, in units of "
            "interocular distance. A well-detected frontal face lands around "
            "0.02-0.08. Null on the fallback path."
        ),
    )
    flip_averaged: bool = Field(
        default=False,
        description="Whether the mirrored crop was embedded and averaged in.",
    )
    cache_hit: bool = Field(
        default=False, description="Whether this came from the cache."
    )
    model_key: str = Field(default="", description="Registry key of the recogniser.")
    model_version: str = Field(
        default="",
        description=(
            "Pinned model version. Embeddings from different versions occupy "
            "unrelated spaces and are never compared."
        ),
    )

    # -- Failure ------------------------------------------------------------ #
    error_code: ErrorCode | None = Field(
        default=None, description="Why no embedding could be produced."
    )
    error_message: str | None = Field(
        default=None, description="Human-readable explanation."
    )
    warnings: list[AnalysisWarning] = Field(
        default_factory=list, description="Non-fatal observations."
    )
    duration_ms: float = Field(
        default=0.0, description="Time spent on this face, in milliseconds."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def trustworthy(self) -> bool:
        """Whether this embedding should carry full weight downstream."""
        return self.success and self.aligned and not self.low_confidence

    def summary(self) -> dict[str, Any]:
        """Compact, PII-free summary for logging. Never includes the vector."""
        return {
            "role": str(self.role),
            "success": self.success,
            "dimension": self.dimension,
            "confidence": round(self.confidence, 3),
            "raw_norm": round(self.raw_norm, 2),
            "aligned": self.aligned,
            "cache_hit": self.cache_hit,
            "error": str(self.error_code) if self.error_code else None,
            "duration_ms": round(self.duration_ms, 1),
        }


class EmbeddingBatchResult(OutputModel):
    """Result of embedding several faces in one call."""

    results: list[EmbeddingResult] = Field(
        default_factory=list,
        description="One entry per submitted face, in submission order.",
    )
    total_duration_ms: float = Field(
        default=0.0, description="Wall-clock time for the whole batch."
    )
    cache_hits: int = Field(default=0, description="How many were served from cache.")
    forward_passes: int = Field(
        default=0, description="How many crops actually reached the network."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def succeeded(self) -> int:
        """Number of faces that produced a usable embedding."""
        return sum(1 for result in self.results if result.success)


__all__ = ["EmbeddingBatchResult", "EmbeddingResult"]
