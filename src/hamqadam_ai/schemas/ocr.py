"""MODULE 5 response contract - CNIC OCR.

A privacy note that shapes this whole schema
--------------------------------------------
The contents of a CNIC are, in their entirety, personal data: name, father's
name, date of birth and a national identity number. The Backend needs those
values - that is the point of the module - so they are returned. But they are
returned **only here**, in the response body, and never appear in a log line,
a metric label or an error detail. Every ``summary()`` in this module reports
which fields were found and how confidently, never what they contained.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import Field, computed_field

from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.schemas.common import (
    AnalysisWarning,
    OutputModel,
    PercentScore,
    UnitScore,
)


class ExtractedField(OutputModel):
    """One field read from the card."""

    value: str | None = Field(
        default=None, description="The parsed value, or null when unreadable."
    )
    confidence: UnitScore = Field(
        default=0.0,
        description=(
            "Confidence in this field specifically. Reported per-field rather "
            "than only per-document because the fields fail independently - a "
            "card can yield a perfect identity number and no gender glyph."
        ),
    )
    source: str | None = Field(
        default=None,
        description=(
            "How the value was located: `label` when anchored on the card's "
            "printed label, `pattern` by shape alone, `chronological` by date "
            "ordering, `derived` when inferred from another field."
        ),
    )
    corrected: bool = Field(
        default=False,
        description=(
            "Whether OCR character-confusion correction was needed, which "
            "lowers confidence in the reading."
        ),
    )
    present: bool = Field(default=False, description="Whether a value was read.")


class ValidationFindingModel(OutputModel):
    """One inconsistency the validator noticed."""

    code: str = Field(description="Stable machine-readable identifier.")
    severity: str = Field(description="error, warning or info.")
    message: str = Field(description="Human-readable explanation.")
    fields: list[str] = Field(
        default_factory=list, description="Which fields the finding concerns."
    )


class CnicOcrResult(OutputModel):
    """MODULE 5 output for one CNIC image."""

    success: bool = Field(
        description="Whether enough of the card could be read to be usable."
    )
    is_cnic: bool = Field(
        default=False,
        description=(
            "Whether the image appears to be a Pakistani CNIC at all. "
            "Distinguishes 'a CNIC we could not read' from 'not a CNIC', which "
            "need different messages to the user and carry different fraud "
            "weight."
        ),
    )

    # -- The extracted fields ------------------------------------------- #
    cnic_number: str | None = Field(
        default=None, description="Thirteen digits, formatted XXXXX-XXXXXXX-X."
    )
    name: str | None = Field(default=None, description="The holder's name.")
    father_name: str | None = Field(
        default=None, description="Father's or husband's name, as printed."
    )
    gender: str | None = Field(default=None, description="M or F.")
    date_of_birth: dt.date | None = Field(default=None, description="As printed.")
    issue_date: dt.date | None = Field(default=None, description="As printed.")
    expiry_date: dt.date | None = Field(default=None, description="As printed.")
    country_of_stay: str | None = Field(default=None, description="As printed.")

    # -- Derived ---------------------------------------------------------- #
    province: str | None = Field(
        default=None,
        description="Region implied by the identity number's leading digit.",
    )
    implied_gender: str | None = Field(
        default=None,
        description=(
            "Gender implied by the parity of the identity number's final "
            "digit - odd is male, even is female. Reported separately from the "
            "printed value so the two can be compared."
        ),
    )
    is_expired: bool | None = Field(
        default=None,
        description="Whether the expiry date has passed. Null when unreadable.",
    )
    is_lifetime: bool = Field(
        default=False,
        description="Whether the card carries lifetime validity instead of an expiry.",
    )

    # -- Confidence -------------------------------------------------------- #
    ocr_confidence_score: PercentScore = Field(
        default=0.0,
        description=(
            "Document confidence on the 0-100 scale: a weighted mean over the "
            "fields that were read, scaled by completeness and reduced by "
            "validation penalties."
        ),
    )
    field_confidence: UnitScore = Field(
        default=0.0,
        description="Mean confidence across the required fields that were read.",
    )
    completeness: UnitScore = Field(
        default=0.0, description="Share of the six required fields that were read."
    )
    fields_present: list[str] = Field(
        default_factory=list, description="Which required fields were read."
    )
    fields_missing: list[str] = Field(
        default_factory=list, description="Which required fields could not be read."
    )

    # -- Validation --------------------------------------------------------- #
    consistent: bool = Field(
        default=True,
        description=(
            "Whether the card is internally consistent. False means a "
            "cross-field check failed - most usefully, the printed gender "
            "disagreeing with the identity number's final digit."
        ),
    )
    findings: list[ValidationFindingModel] = Field(
        default_factory=list, description="Every inconsistency noticed."
    )

    # -- Evidence ------------------------------------------------------------ #
    fields: dict[str, ExtractedField] = Field(
        default_factory=dict,
        description="Per-field detail, including confidence and how it was found.",
    )
    engine: str = Field(default="", description="Which OCR adapter was used.")
    engine_version: str = Field(default="", description="Its version.")
    used_fallback_engine: bool = Field(
        default=False,
        description="Whether the configured primary engine was unavailable.",
    )
    rotation_applied: int = Field(
        default=0, description="Page rotation applied before recognition, degrees."
    )
    preprocessing: dict[str, Any] = Field(
        default_factory=dict, description="What was done to the image."
    )
    lines_detected: int = Field(default=0, description="Recognised text lines.")
    orientation_attempts: int = Field(
        default=1, description="How many rotations had to be tried."
    )

    error_code: ErrorCode | None = Field(
        default=None, description="Set to CNIC_OCR_FAILED when unusable."
    )
    error_message: str | None = Field(
        default=None, description="Human-readable explanation."
    )
    warnings: list[AnalysisWarning] = Field(
        default_factory=list, description="Non-fatal observations."
    )
    duration_ms: float = Field(
        default=0.0, description="Time spent reading, in milliseconds."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gender_cross_check_passed(self) -> bool | None:
        """Whether the printed gender agrees with the number's final digit.

        Null when the comparison could not be made - either value missing, or
        the gender having been derived from the number in the first place,
        which would make the check circular.
        """
        if self.gender is None or self.implied_gender is None:
            return None
        detail = self.fields.get("gender")
        if detail is not None and detail.source == "derived":
            return None
        return self.gender == self.implied_gender

    def summary(self) -> dict[str, Any]:
        """PII-free summary for logging.

        Never includes a field value: the whole content of a CNIC is personal
        data, and this is the object that reaches the log sink.
        """
        return {
            "success": self.success,
            "is_cnic": self.is_cnic,
            "confidence": self.ocr_confidence_score,
            "completeness": round(self.completeness, 3),
            "present": self.fields_present,
            "missing": self.fields_missing,
            "consistent": self.consistent,
            "engine": self.engine,
            "rotation": self.rotation_applied,
            "attempts": self.orientation_attempts,
            "lines": self.lines_detected,
            "duration_ms": round(self.duration_ms, 1),
        }


__all__ = ["CnicOcrResult", "ExtractedField", "ValidationFindingModel"]
