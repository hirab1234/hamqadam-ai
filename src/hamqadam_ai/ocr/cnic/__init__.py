"""Pakistani CNIC document intelligence.

Separated from the OCR engines because it is the part that does not commoditise.
Any modern recogniser reads clean printed Latin text; what decides whether this
module is useful is knowing that the last digit of a CNIC encodes gender, that
the three printed dates are necessarily ordered, and that ``ldentity`` is a
misread ``Identity`` rather than a different word.
"""

from __future__ import annotations

from hamqadam_ai.ocr.cnic.fields import CnicFields, FieldValue
from hamqadam_ai.ocr.cnic.parser import CnicParser, looks_like_cnic
from hamqadam_ai.ocr.cnic.patterns import (
    PROVINCE_CODES,
    CnicNumber,
    normalise_confusions,
    parse_cnic_date,
    parse_cnic_number,
    parse_gender,
)
from hamqadam_ai.ocr.cnic.validation import (
    CnicValidator,
    Severity,
    ValidationFinding,
    ValidationOutcome,
)

__all__ = [
    "PROVINCE_CODES",
    "CnicFields",
    "CnicNumber",
    "CnicParser",
    "CnicValidator",
    "FieldValue",
    "Severity",
    "ValidationFinding",
    "ValidationOutcome",
    "looks_like_cnic",
    "normalise_confusions",
    "parse_cnic_date",
    "parse_cnic_number",
    "parse_gender",
]
