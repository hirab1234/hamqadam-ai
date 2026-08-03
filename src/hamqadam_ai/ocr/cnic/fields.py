"""The structured result of reading a CNIC.

Every field carries its own confidence and provenance rather than sharing one
document-level number. That matters because the fields fail independently: the
probe that motivated this module read the identity number, all three dates and
both names correctly while missing the gender glyph entirely. A single
"OCR confidence: 0.86" would have concealed exactly the one field that failed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FieldValue(Generic[T]):
    """One extracted field, with how it was obtained and how much to trust it.

    Attributes:
        value: The parsed value, or ``None`` when the field could not be read.
        confidence: Confidence in ``[0, 1]``, combining the OCR line score with
            whatever validation the field supports.
        raw: The recognised text the value came from, before parsing.
        source: How it was located - ``label`` when anchored on the card's
            printed label, ``pattern`` when found by shape alone, ``derived``
            when inferred from another field.
        corrected: Whether OCR character-confusion correction was applied,
            which lowers confidence in the reading.
    """

    value: T | None = None
    confidence: float = 0.0
    raw: str | None = None
    source: str | None = None
    corrected: bool = False

    @property
    def present(self) -> bool:
        """Whether a value was extracted at all."""
        return self.value is not None

    def as_dict(self, *, redact: bool = True) -> dict[str, Any]:
        """Serialisable form.

        Args:
            redact: Omit the raw recognised text. On by default because on a
                CNIC that text is the holder's name and identity number, and
                the parsed value is what a caller needs.
        """
        payload: dict[str, Any] = {
            "value": _serialise(self.value),
            "confidence": round(self.confidence, 4),
            "present": self.present,
        }
        if self.source:
            payload["source"] = self.source
        if self.corrected:
            payload["corrected"] = True
        if not redact and self.raw is not None:
            payload["raw"] = self.raw
        return payload


def _serialise(value: Any) -> Any:
    """Render a field value as something JSON can carry."""
    if isinstance(value, dt.date):
        return value.isoformat()
    return value


@dataclass(slots=True)
class CnicFields:
    """Every field the specification asks to be extracted from the card.

    Attributes:
        full_name: The holder's name as printed.
        father_name: Father's or husband's name. Not requested by the
            specification but printed on every card and extracted because it
            is a further cross-check against the Backend's records.
        cnic_number: The thirteen-digit identity number, hyphen-formatted.
        gender: ``M`` or ``F``.
        date_of_birth: As printed, day-first.
        date_of_issue: As printed.
        date_of_expiry: As printed.
        country_of_stay: Printed on the card; extracted for completeness.
        implied_gender: Gender implied by the parity of the final CNIC digit.
            An independent signal, kept separate from the printed value so the
            two can be compared.
        province: Region implied by the leading CNIC digit.
        is_lifetime: Whether the card carries a lifetime validity marker
            instead of an expiry date, which NADRA issues to holders over 65.
    """

    full_name: FieldValue[str] = field(default_factory=FieldValue)
    father_name: FieldValue[str] = field(default_factory=FieldValue)
    cnic_number: FieldValue[str] = field(default_factory=FieldValue)
    gender: FieldValue[str] = field(default_factory=FieldValue)
    date_of_birth: FieldValue[dt.date] = field(default_factory=FieldValue)
    date_of_issue: FieldValue[dt.date] = field(default_factory=FieldValue)
    date_of_expiry: FieldValue[dt.date] = field(default_factory=FieldValue)
    country_of_stay: FieldValue[str] = field(default_factory=FieldValue)

    implied_gender: str | None = None
    province: str | None = None
    is_lifetime: bool = False

    #: The fields the specification names. Completeness is measured against
    #: this set, not against every field the parser happens to find.
    REQUIRED = (
        "full_name",
        "cnic_number",
        "gender",
        "date_of_birth",
        "date_of_issue",
        "date_of_expiry",
    )

    def get(self, name: str) -> FieldValue[Any]:
        """Return a field by name."""
        value = getattr(self, name, None)
        if not isinstance(value, FieldValue):
            raise AttributeError(f"{name!r} is not an extractable CNIC field")
        return value

    def set(self, name: str, value: FieldValue[Any]) -> None:
        """Replace a field by name."""
        if not hasattr(self, name):
            raise AttributeError(f"{name!r} is not a CNIC field")
        setattr(self, name, value)

    @property
    def present_fields(self) -> list[str]:
        """Names of the required fields that were successfully extracted."""
        return [name for name in self.REQUIRED if self.get(name).present]

    @property
    def missing_fields(self) -> list[str]:
        """Names of the required fields that could not be read."""
        return [name for name in self.REQUIRED if not self.get(name).present]

    @property
    def completeness(self) -> float:
        """Share of the required fields that were extracted, in ``[0, 1]``."""
        return len(self.present_fields) / len(self.REQUIRED)

    def mean_confidence(self) -> float:
        """Mean confidence across the required fields that were found.

        Averaged over *found* fields only; absence is measured by
        :attr:`completeness`, and folding it in here would conflate "read
        badly" with "not read at all".
        """
        confidences = [
            self.get(name).confidence for name in self.REQUIRED if self.get(name).present
        ]
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)

    def as_dict(self, *, redact: bool = True) -> dict[str, Any]:
        """Serialisable form of every field."""
        payload: dict[str, Any] = {
            name: self.get(name).as_dict(redact=redact)
            for name in (
                "full_name",
                "father_name",
                "cnic_number",
                "gender",
                "date_of_birth",
                "date_of_issue",
                "date_of_expiry",
                "country_of_stay",
            )
        }
        payload["implied_gender"] = self.implied_gender
        payload["province"] = self.province
        payload["is_lifetime"] = self.is_lifetime
        return payload

    def summary(self) -> dict[str, Any]:
        """PII-free summary for logging.

        Reports which fields were found and how confidently, never their
        values. The whole content of a CNIC is personal data.
        """
        return {
            "present": self.present_fields,
            "missing": self.missing_fields,
            "completeness": round(self.completeness, 3),
            "mean_confidence": round(self.mean_confidence(), 3),
            "province": self.province,
        }


__all__ = ["CnicFields", "FieldValue"]
