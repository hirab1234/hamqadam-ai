"""Cross-field validation of an extracted CNIC.

What validation can and cannot do here
--------------------------------------
There is **no checksum** on a Pakistani CNIC. A syntactically correct number
cannot be verified offline against anything, and this module does not pretend
otherwise. What it can do is check the document against *itself*: the fields on
a genuine card are mutually constrained, and OCR errors and crude alterations
both tend to break those constraints.

The checks, strongest first
---------------------------
**Gender parity.** The final digit of the CNIC is odd for men and even for
women. Comparing it against the printed gender field is a genuinely
independent cross-check that costs nothing. Disagreement means either a misread
digit or an altered document.

**Date ordering.** Birth precedes issue precedes expiry, necessarily. A
violation is almost always a misread digit, and identifying *which* ordering
broke narrows down which date to distrust.

**Issue-to-expiry span.** NADRA issues cards for a fixed term. A span far from
that is evidence of a misread year.

**Plausible age.** A birth date implying an age outside human range, or a card
issued before its holder was born, is self-evidently wrong.

**Expiry.** An expired card is not an error - it is a valid reading of an
invalid document, and the Backend's rules engine decides what to do about it.
It is reported as a finding rather than a defect.

Every finding is returned with a severity and a confidence adjustment. Nothing
here rejects a document on its own; the module reports what is inconsistent and
lets Module 9 and Module 10 weigh it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hamqadam_ai.core.config import OcrConfig
from hamqadam_ai.ocr.cnic.fields import CnicFields


class Severity(StrEnum):
    """How much weight a finding should carry."""

    #: Something is definitely wrong with the reading or the document.
    ERROR = "error"
    #: Something is unusual and worth a human look.
    WARNING = "warning"
    #: Worth recording, not worth acting on.
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    """One thing the validator noticed.

    Attributes:
        code: Stable machine-readable identifier.
        severity: How much weight to give it.
        message: Human-readable explanation.
        fields: Which fields the finding concerns.
        confidence_penalty: How much to subtract from the document confidence,
            in ``[0, 1]``.
    """

    code: str
    severity: Severity
    message: str
    fields: tuple[str, ...] = ()
    confidence_penalty: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "code": self.code,
            "severity": str(self.severity),
            "message": self.message,
            "fields": list(self.fields),
            "confidence_penalty": round(self.confidence_penalty, 4),
        }


@dataclass(slots=True)
class ValidationOutcome:
    """Everything the validator concluded.

    Attributes:
        findings: What was noticed, most severe first.
        consistent: Whether any ERROR-level finding was raised.
        expired: Whether the card's expiry date has passed.
        confidence_penalty: Total penalty to apply to document confidence.
    """

    findings: list[ValidationFinding] = field(default_factory=list)
    consistent: bool = True
    expired: bool | None = None
    confidence_penalty: float = 0.0

    @property
    def errors(self) -> list[ValidationFinding]:
        """Only the ERROR-level findings."""
        return [f for f in self.findings if f.severity is Severity.ERROR]

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "consistent": self.consistent,
            "expired": self.expired,
            "confidence_penalty": round(self.confidence_penalty, 4),
            "findings": [finding.as_dict() for finding in self.findings],
        }


#: Youngest and oldest plausible holder. NADRA issues a CNIC from 18 and a
#: child registration certificate below that, but B-form holders do appear in
#: verification flows, so the floor is set well below 18 rather than at it.
_MIN_AGE_YEARS = 5
_MAX_AGE_YEARS = 120

#: Typical CNIC validity terms. Ten years is standard; five and seven are
#: issued in some circumstances, and lifetime cards carry no expiry at all.
_EXPECTED_VALIDITY_YEARS = (5, 7, 10, 15)

#: Tolerance on the issue-to-expiry span, in days. Generous, because renewal
#: and re-issue dates shift the span legitimately.
_VALIDITY_TOLERANCE_DAYS = 200


class CnicValidator:
    """Checks an extracted CNIC against itself.

    Args:
        config: The OCR section of the settings.
        today: Reference date for expiry and age checks. Injectable so the
            tests do not change behaviour as the calendar advances.
    """

    __slots__ = ("_config", "_today")

    def __init__(self, config: OcrConfig, *, today: dt.date | None = None) -> None:
        self._config = config
        self._today = today

    def validate(self, fields: CnicFields) -> ValidationOutcome:
        """Run every applicable check.

        Args:
            fields: The extracted fields.

        Returns:
            The findings and the resulting confidence penalty.
        """
        today = self._today or dt.date.today()
        findings: list[ValidationFinding] = []

        findings.extend(self._check_gender_parity(fields))
        findings.extend(self._check_date_order(fields))
        findings.extend(self._check_validity_span(fields))
        findings.extend(self._check_age(fields, today))
        findings.extend(self._check_province(fields))
        findings.extend(self._check_completeness(fields))

        expired = self._expiry_state(fields, today)
        if expired is True:
            findings.append(
                ValidationFinding(
                    code="CNIC_EXPIRED",
                    severity=Severity.WARNING,
                    message=(
                        "The card's expiry date has passed. This is a correct "
                        "reading of an out-of-date document, not an OCR error; "
                        "whether it is acceptable is a policy decision."
                    ),
                    fields=("date_of_expiry",),
                    confidence_penalty=0.0,
                )
            )

        order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        findings.sort(key=lambda finding: order[finding.severity])

        penalty = min(1.0, sum(finding.confidence_penalty for finding in findings))

        return ValidationOutcome(
            findings=findings,
            consistent=not any(f.severity is Severity.ERROR for f in findings),
            expired=expired,
            confidence_penalty=penalty,
        )

    # -- Individual checks ---------------------------------------------------- #

    @staticmethod
    def _check_gender_parity(fields: CnicFields) -> list[ValidationFinding]:
        """Compare the printed gender against the CNIC's final digit."""
        printed = fields.gender.value
        implied = fields.implied_gender

        if printed is None or implied is None:
            return []

        # A gender that was *derived* from the number cannot cross-check the
        # number. Saying it agrees would be circular.
        if fields.gender.source == "derived":
            return [
                ValidationFinding(
                    code="CNIC_GENDER_DERIVED",
                    severity=Severity.INFO,
                    message=(
                        "The printed gender glyph could not be read, so gender "
                        "was taken from the parity of the identity number's "
                        "final digit. The usual cross-check between the two is "
                        "therefore unavailable for this document."
                    ),
                    fields=("gender", "cnic_number"),
                    confidence_penalty=0.03,
                )
            ]

        if printed == implied:
            return [
                ValidationFinding(
                    code="CNIC_GENDER_CONSISTENT",
                    severity=Severity.INFO,
                    message=(
                        "The printed gender agrees with the parity of the "
                        "identity number's final digit."
                    ),
                    fields=("gender", "cnic_number"),
                )
            ]

        return [
            ValidationFinding(
                code="CNIC_GENDER_MISMATCH",
                severity=Severity.ERROR,
                message=(
                    f"The card prints gender {printed} but the identity number "
                    f"ends in a digit implying {implied}. On a genuine CNIC the "
                    f"final digit is odd for men and even for women, so either a "
                    f"digit was misread or the document has been altered."
                ),
                fields=("gender", "cnic_number"),
                confidence_penalty=0.25,
            )
        ]

    @staticmethod
    def _check_date_order(fields: CnicFields) -> list[ValidationFinding]:
        """Birth precedes issue precedes expiry, necessarily."""
        findings: list[ValidationFinding] = []
        birth = fields.date_of_birth.value
        issue = fields.date_of_issue.value
        expiry = fields.date_of_expiry.value

        if birth is not None and issue is not None and issue <= birth:
            findings.append(
                ValidationFinding(
                    code="CNIC_ISSUE_BEFORE_BIRTH",
                    severity=Severity.ERROR,
                    message=(
                        f"The card was issued on {issue.isoformat()}, on or "
                        f"before the stated birth date {birth.isoformat()}. One "
                        f"of the two dates was misread."
                    ),
                    fields=("date_of_birth", "date_of_issue"),
                    confidence_penalty=0.25,
                )
            )

        if issue is not None and expiry is not None and expiry <= issue:
            findings.append(
                ValidationFinding(
                    code="CNIC_EXPIRY_BEFORE_ISSUE",
                    severity=Severity.ERROR,
                    message=(
                        f"The card expires on {expiry.isoformat()}, on or before "
                        f"its issue date {issue.isoformat()}. One of the two "
                        f"dates was misread."
                    ),
                    fields=("date_of_issue", "date_of_expiry"),
                    confidence_penalty=0.25,
                )
            )

        return findings

    @staticmethod
    def _check_validity_span(fields: CnicFields) -> list[ValidationFinding]:
        """Check the issue-to-expiry term against NADRA's usual ones."""
        issue = fields.date_of_issue.value
        expiry = fields.date_of_expiry.value
        if issue is None or expiry is None or expiry <= issue:
            return []

        span_days = (expiry - issue).days
        for years in _EXPECTED_VALIDITY_YEARS:
            if abs(span_days - years * 365.25) <= _VALIDITY_TOLERANCE_DAYS:
                return []

        return [
            ValidationFinding(
                code="CNIC_UNUSUAL_VALIDITY_TERM",
                severity=Severity.WARNING,
                message=(
                    f"The card runs {span_days / 365.25:.1f} years from issue to "
                    f"expiry, which is not one of the usual terms "
                    f"({', '.join(str(y) for y in _EXPECTED_VALIDITY_YEARS)}). "
                    f"Most likely a misread year digit."
                ),
                fields=("date_of_issue", "date_of_expiry"),
                confidence_penalty=0.10,
            )
        ]

    @staticmethod
    def _check_age(fields: CnicFields, today: dt.date) -> list[ValidationFinding]:
        """Check the birth date implies a plausible living person."""
        birth = fields.date_of_birth.value
        if birth is None:
            return []

        if birth > today:
            return [
                ValidationFinding(
                    code="CNIC_BIRTH_IN_FUTURE",
                    severity=Severity.ERROR,
                    message=(
                        f"The stated birth date {birth.isoformat()} is in the "
                        f"future, so at least one digit was misread."
                    ),
                    fields=("date_of_birth",),
                    confidence_penalty=0.30,
                )
            ]

        age = (today - birth).days / 365.25
        if age < _MIN_AGE_YEARS or age > _MAX_AGE_YEARS:
            return [
                ValidationFinding(
                    code="CNIC_IMPLAUSIBLE_AGE",
                    severity=Severity.ERROR,
                    message=(
                        f"The stated birth date implies an age of {age:.0f}, "
                        f"outside the plausible range "
                        f"{_MIN_AGE_YEARS}-{_MAX_AGE_YEARS}. A year digit was "
                        f"most likely misread."
                    ),
                    fields=("date_of_birth",),
                    confidence_penalty=0.25,
                )
            ]

        return []

    @staticmethod
    def _check_province(fields: CnicFields) -> list[ValidationFinding]:
        """Check the leading digit against the known region allocation."""
        number = fields.cnic_number.value
        if number is None:
            return []
        if fields.province is not None:
            return []
        return [
            ValidationFinding(
                code="CNIC_UNKNOWN_REGION_CODE",
                severity=Severity.WARNING,
                message=(
                    f"The identity number begins with {number[0]}, which is not "
                    f"an allocated region code. NADRA does reallocate district "
                    f"codes, so this is weak evidence of a misread digit rather "
                    f"than proof of forgery."
                ),
                fields=("cnic_number",),
                confidence_penalty=0.08,
            )
        ]

    def _check_completeness(self, fields: CnicFields) -> list[ValidationFinding]:
        """Check enough of the required fields were read to be useful."""
        missing = fields.missing_fields
        if len(fields.present_fields) >= self._config.min_required_fields:
            if missing:
                return [
                    ValidationFinding(
                        code="CNIC_FIELDS_MISSING",
                        severity=Severity.WARNING,
                        message=(
                            f"Some fields could not be read: "
                            f"{', '.join(missing)}."
                        ),
                        fields=tuple(missing),
                        confidence_penalty=0.05 * len(missing),
                    )
                ]
            return []

        return [
            ValidationFinding(
                code="CNIC_TOO_FEW_FIELDS",
                severity=Severity.ERROR,
                message=(
                    f"Only {len(fields.present_fields)} of "
                    f"{len(fields.REQUIRED)} required fields could be read; at "
                    f"least {self._config.min_required_fields} are needed for "
                    f"the document to be usable."
                ),
                fields=tuple(missing),
                confidence_penalty=0.30,
            )
        ]

    @staticmethod
    def _expiry_state(fields: CnicFields, today: dt.date) -> bool | None:
        """Whether the card has expired, or ``None`` when undeterminable."""
        if fields.is_lifetime:
            return False
        expiry = fields.date_of_expiry.value
        if expiry is None:
            return None
        return expiry < today


__all__ = ["CnicValidator", "Severity", "ValidationFinding", "ValidationOutcome"]
