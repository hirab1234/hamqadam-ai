"""Shared schema building blocks.

Score convention
----------------
Two scales coexist deliberately and are never mixed:

* **Internal** scores are ``float`` in ``[0, 1]`` (or ``[-1, 1]`` for cosine
  similarity). Every algorithm works in this space.
* **External** scores are ``float`` in ``[0, 100]`` rounded to two decimals,
  because the sample response in section 16 of the requirements document uses
  that scale and the Backend's rules engine is written against it.

:func:`to_percent` is the single conversion point. Any field whose name ends in
``_score`` in an API response is on the 0-100 scale.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hamqadam_ai.core.constants import SCORE_DECIMALS
from hamqadam_ai.core.errors import ErrorCode

#: A confidence or probability in the internal ``[0, 1]`` space.
UnitScore = Annotated[float, Field(ge=0.0, le=1.0)]

#: A score on the external ``[0, 100]`` reporting scale.
PercentScore = Annotated[float, Field(ge=0.0, le=100.0)]

#: A cosine similarity in ``[-1, 1]``.
Similarity = Annotated[float, Field(ge=-1.0, le=1.0)]


def to_percent(value: float, *, decimals: int = SCORE_DECIMALS) -> float:
    """Convert an internal ``[0, 1]`` score to the external ``[0, 100]`` scale.

    Values outside the unit interval are clamped rather than rejected: an
    upstream metric that overshoots by a floating-point epsilon should not
    fail a whole verification.

    Args:
        value: Internal score.
        decimals: Rounding precision.

    Returns:
        The score on the 0-100 scale.
    """
    return round(max(0.0, min(1.0, float(value))) * 100.0, decimals)


def from_percent(value: float) -> float:
    """Convert an external ``[0, 100]`` score back to the internal ``[0, 1]``."""
    return max(0.0, min(100.0, float(value))) / 100.0


class StrictModel(BaseModel):
    """Base class for every schema in this service.

    Rejects unknown fields on input, which turns a Backend typo such as
    ``profile_imge`` into an immediate, precise 400 instead of a silently
    ignored field and a mystifying verification result.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        populate_by_name=True,
        protected_namespaces=(),
    )


class OutputModel(BaseModel):
    """Base class for response schemas.

    Unlike :class:`StrictModel` this permits extra fields, so a newer service
    version can add a field without a strict Backend client rejecting the whole
    response.
    """

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=False,
        protected_namespaces=(),
        ser_json_inf_nan="null",
    )


class AnalysisWarning(OutputModel):
    """A non-fatal observation attached to a result.

    Warnings are how the service reports "this worked, but you should know
    something" - a face near the pose limit, a fallback detector in use, an
    optional model unavailable. They never change the outcome on their own but
    they feed the fraud engine and give a human reviewer context.
    """

    code: str = Field(description="Stable machine-readable warning identifier.")
    message: str = Field(description="Human-readable explanation.")
    stage: str | None = Field(
        default=None, description="Pipeline stage that raised the warning."
    )
    detail: dict[str, Any] = Field(
        default_factory=dict, description="Structured, PII-free context."
    )


class ErrorEnvelope(OutputModel):
    """The error body returned for any non-2xx response."""

    success: bool = Field(default=False, description="Always false for an error.")
    error_code: ErrorCode = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable description of the failure.")
    severity: str = Field(description="client | business | transient | fatal.")
    retryable: bool = Field(
        description="Whether an identical retry could plausibly succeed."
    )
    verification_id: str | None = Field(
        default=None, description="Echoed from the request when it was parseable."
    )
    request_id: str | None = Field(
        default=None, description="Correlation id for support and log lookup."
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Structured, PII-free diagnostic context."
    )


class ProcessingTime(OutputModel):
    """Wall-clock breakdown of a request, in milliseconds."""

    total: float = Field(description="End-to-end duration in milliseconds.")
    unit: str = Field(default="ms", description="Unit of every duration here.")
    stages: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Per-stage duration, call count and average. Stage totals may sum "
            "to more than `total` when stages ran concurrently."
        ),
    )
    overhead: float = Field(
        default=0.0,
        description="Time not attributed to any measured stage.",
    )


class ModelVersions(OutputModel):
    """Version of every model that contributed to a result.

    Present in every response so a decision can be reproduced later against the
    exact artefacts that produced it - a hard requirement for disputing or
    auditing a rejected verification.
    """

    versions: dict[str, str] = Field(
        default_factory=dict, description="Registry key to pinned version string."
    )
    service_version: str = Field(description="Version of the AI service contract.")
    device: str = Field(description="Device family the models ran on.")


class ScoreBreakdown(OutputModel):
    """A composite score together with the components that produced it.

    Returning only the composite makes a rejection impossible to explain to the
    user or to a human reviewer. The breakdown is what turns "quality 42" into
    "your photo is too dark and slightly out of focus".
    """

    score: PercentScore = Field(description="Composite score on the 0-100 scale.")
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Named sub-scores on the 0-100 scale, before weighting.",
    )
    weights: dict[str, float] = Field(
        default_factory=dict, description="Weight applied to each component."
    )
    limiting_factor: str | None = Field(
        default=None,
        description="Component that contributed the largest deficit to the score.",
    )

    @field_validator("components", "weights")
    @classmethod
    def _round_values(cls, value: dict[str, float]) -> dict[str, float]:
        return {key: round(float(item), 4) for key, item in value.items()}


__all__ = [
    "ErrorEnvelope",
    "ModelVersions",
    "OutputModel",
    "PercentScore",
    "ProcessingTime",
    "ScoreBreakdown",
    "Similarity",
    "StrictModel",
    "UnitScore",
    "AnalysisWarning",
    "from_percent",
    "to_percent",
]
