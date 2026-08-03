"""MODULE 5 - CNIC optical character recognition.

Reads the Pakistani National Identity Card and returns the structured fields
the Backend needs: name, father's name, CNIC number, gender, date of birth,
date of issue and date of expiry.

Structure
---------
``base``
    The :class:`~hamqadam_ai.ocr.base.OcrEngine` port, plus the
    :class:`~hamqadam_ai.ocr.base.TextLine` value object every adapter emits.

``engines``
    Interchangeable OCR adapters. ONNX PP-OCR is the default because it runs
    through the same ONNX Runtime the rest of the service already uses;
    PaddleOCR and EasyOCR are the engines the specification names and are
    fully implemented behind the same port.

``preprocessing``
    Card rectification, page-orientation resolution and enhancement for
    low-quality captures. Engine-independent, and the part that most affects
    whether anything is readable at all.

``cnic``
    The document-specific intelligence: field extraction anchored on the
    card's printed labels, Pakistani CNIC number semantics, date parsing that
    survives OCR character confusion, and the cross-field validation that
    turns a pile of strings into an assessment.

Why the domain layer matters more than the engine
-------------------------------------------------
Any modern OCR reads clean printed Latin text well. What decides whether this
module is useful is everything around it: finding the card in a photograph
taken at an angle, working out which way up it is, recognising that
``National ldentity Card`` is a lowercase-L substitution rather than a
different document, knowing that the last digit of a CNIC encodes gender and
must agree with the printed gender field, and reporting a confidence that
reflects all of it.
"""

from __future__ import annotations

from hamqadam_ai.ocr.base import (
    OcrEngine,
    OcrOutput,
    TextLine,
)
from hamqadam_ai.ocr.cnic.fields import CnicFields, FieldValue
from hamqadam_ai.ocr.cnic.parser import CnicParser
from hamqadam_ai.ocr.cnic.patterns import (
    PROVINCE_CODES,
    normalise_confusions,
    parse_cnic_date,
    parse_cnic_number,
)
from hamqadam_ai.ocr.cnic.validation import CnicValidator, ValidationFinding
from hamqadam_ai.ocr.preprocessing import (
    Orientation,
    enhance_for_ocr,
    rectify_document,
)

__all__ = [
    "PROVINCE_CODES",
    "CnicFields",
    "CnicParser",
    "CnicValidator",
    "FieldValue",
    "OcrEngine",
    "OcrOutput",
    "Orientation",
    "TextLine",
    "ValidationFinding",
    "enhance_for_ocr",
    "normalise_confusions",
    "parse_cnic_date",
    "parse_cnic_number",
    "rectify_document",
]
