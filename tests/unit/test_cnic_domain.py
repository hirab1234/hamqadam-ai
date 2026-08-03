"""Pakistani CNIC semantics: number structure, dates, gender and validation.

Pure domain logic - no OCR engine, no images, no model weights. This is the
layer that does not commoditise, so it is tested exhaustively against cases
whose answers follow from how the document is actually constructed.
"""

from __future__ import annotations

import datetime as dt

import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.ocr.cnic.fields import CnicFields, FieldValue
from hamqadam_ai.ocr.cnic.patterns import (
    PROVINCE_CODES,
    normalise_confusions,
    parse_cnic_date,
    parse_cnic_number,
    parse_gender,
)
from hamqadam_ai.ocr.cnic.validation import CnicValidator, Severity

TODAY = dt.date(2026, 7, 27)


# --------------------------------------------------------------------------- #
# CNIC number
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_hyphenated_number_parses() -> None:
    parsed = parse_cnic_number("42101-8375926-4")
    assert parsed is not None
    assert parsed.digits == "4210183759264"
    assert parsed.formatted == "42101-8375926-4"


@pytest.mark.unit
def test_an_unhyphenated_number_parses() -> None:
    parsed = parse_cnic_number("4210183759264")
    assert parsed is not None
    assert parsed.formatted == "42101-8375926-4"


@pytest.mark.unit
def test_a_number_is_found_inside_a_label_line() -> None:
    """The card prints the label and value on one row, and the OCR often
    returns them as a single line."""
    parsed = parse_cnic_number("Identity Number 42101-8375926-4")
    assert parsed is not None
    assert parsed.formatted == "42101-8375926-4"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("last_digit", "expected"),
    [("1", "M"), ("3", "M"), ("5", "M"), ("7", "M"), ("9", "M"),
     ("0", "F"), ("2", "F"), ("4", "F"), ("6", "F"), ("8", "F")],
)
def test_the_final_digit_encodes_gender(last_digit: str, expected: str) -> None:
    """Odd is male, even is female. The single most useful validation this
    module has, because it is independent of the printed gender field."""
    parsed = parse_cnic_number(f"42101-837592{last_digit}-{last_digit}")
    assert parsed is not None
    assert parsed.implied_gender == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("first_digit", "region"),
    [("1", "Khyber Pakhtunkhwa"), ("3", "Punjab"), ("4", "Sindh"),
     ("5", "Balochistan"), ("6", "Islamabad Capital Territory")],
)
def test_the_leading_digit_identifies_the_region(first_digit: str, region: str) -> None:
    parsed = parse_cnic_number(f"{first_digit}2101-8375926-4")
    assert parsed is not None
    assert parsed.province == region


@pytest.mark.unit
def test_an_unallocated_leading_digit_has_no_region() -> None:
    parsed = parse_cnic_number("92101-8375926-4")
    assert parsed is not None
    assert parsed.province is None
    assert "9" not in PROVINCE_CODES


@pytest.mark.unit
def test_glyph_confusion_is_corrected() -> None:
    """The probe read ``Identity`` as ``ldentity``; digits suffer the same
    substitutions and can be corrected because the context is numeric."""
    parsed = parse_cnic_number("4210I-83759Z6-4")
    assert parsed is not None
    assert parsed.formatted == "42101-8375926-4"
    assert parsed.corrected is True


@pytest.mark.unit
def test_a_clean_read_is_not_marked_corrected() -> None:
    parsed = parse_cnic_number("42101-8375926-4")
    assert parsed is not None
    assert parsed.corrected is False


@pytest.mark.unit
def test_correction_can_be_refused() -> None:
    assert parse_cnic_number("4210I-83759Z6-4", allow_correction=False) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    ["", "no digits here", "1234", "421018375926", "421018375926412345",
     "the year 2024 and 1994"],
)
def test_non_numbers_are_rejected(text: str) -> None:
    assert parse_cnic_number(text) is None


@pytest.mark.unit
def test_a_longer_digit_run_is_not_matched() -> None:
    """Anchored on non-digit boundaries so a fragment of a longer number - a
    serial printed elsewhere on the card - is not mistaken for the CNIC."""
    assert parse_cnic_number("999942101837592649999") is None


@pytest.mark.unit
def test_there_is_no_checksum_to_verify() -> None:
    """Documented explicitly because it is tempting to assume otherwise.

    A CNIC carries no check digit, so any thirteen digits of the right shape
    parse. Nothing in this module claims to verify a number against anything
    beyond its structure, region code and gender parity.
    """
    assert parse_cnic_number("11111-1111111-1") is not None
    assert parse_cnic_number("42101-0000000-2") is not None


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_printed_date_format_parses() -> None:
    assert parse_cnic_date("14.03.1994") == dt.date(1994, 3, 14)


@pytest.mark.unit
def test_day_comes_first() -> None:
    """Fixed rather than guessed. Interpreting 03.04.1994 as March in a system
    that reads Pakistani documents would be a silent, systematic error on every
    ambiguous date."""
    assert parse_cnic_date("03.04.1994") == dt.date(1994, 4, 3)


@pytest.mark.unit
@pytest.mark.parametrize("separator", [".", "/", "-"])
def test_separators_are_tolerated(separator: str) -> None:
    text = f"14{separator}03{separator}1994"
    assert parse_cnic_date(text) == dt.date(1994, 3, 14)


@pytest.mark.unit
def test_a_date_is_found_inside_a_label_line() -> None:
    assert parse_cnic_date("Date of Birth 14.03.1994") == dt.date(1994, 3, 14)


@pytest.mark.unit
def test_garbled_digits_are_corrected() -> None:
    assert parse_cnic_date("I4.O3.1994") == dt.date(1994, 3, 14)


@pytest.mark.unit
@pytest.mark.parametrize("text", ["31.02.2020", "14.13.1994", "00.03.1994", "14.03.0000"])
def test_impossible_calendar_dates_are_refused(text: str) -> None:
    """A real calendar rejection, not a parse failure. Returning None is
    correct; inventing a nearby valid date would be worse."""
    assert parse_cnic_date(text) is None


@pytest.mark.unit
def test_a_two_digit_year_is_not_accepted() -> None:
    """The card prints four digits. Accepting two would force a century guess."""
    assert parse_cnic_date("14.03.94") is None


# --------------------------------------------------------------------------- #
# Gender and confusions
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [("M", "M"), ("F", "F"), ("Male", "M"), ("FEMALE", "F"), ("  f  ", "F")],
)
def test_gender_values_normalise(text: str, expected: str) -> None:
    assert parse_gender(text) == expected


@pytest.mark.unit
@pytest.mark.parametrize("text", ["", "xyz", "1234", "unknown"])
def test_unrecognised_gender_returns_none(text: str) -> None:
    """Rather than guessing. The CNIC's final digit independently encodes
    gender, so a missing value is recoverable and a wrong guess is not."""
    assert parse_gender(text) is None


@pytest.mark.unit
def test_digit_confusion_correction() -> None:
    assert normalise_confusions("4210I-83759Z6-4", target="digits") == "42101-8375926-4"


@pytest.mark.unit
def test_correction_is_directional() -> None:
    """``O`` must become ``0`` inside a number and must emphatically not
    inside the name ROBERT, which is why it is applied per-field."""
    assert normalise_confusions("O", target="digits") == "0"
    assert normalise_confusions("0", target="letters") == "O"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def build_fields(**overrides: object) -> CnicFields:
    """A consistent, fully-populated set of fields, before any override."""
    fields = CnicFields()
    fields.cnic_number = FieldValue(value="42101-8375926-4", confidence=0.9,
                                    source="pattern")
    fields.full_name = FieldValue(value="AYESHA KHAN", confidence=0.9, source="label")
    fields.gender = FieldValue(value="F", confidence=0.9, source="label")
    fields.date_of_birth = FieldValue(value=dt.date(1994, 3, 14), confidence=0.9,
                                      source="label")
    fields.date_of_issue = FieldValue(value=dt.date(2019, 7, 22), confidence=0.9,
                                      source="label")
    fields.date_of_expiry = FieldValue(value=dt.date(2029, 7, 22), confidence=0.9,
                                       source="label")
    fields.implied_gender = "F"
    fields.province = "Sindh"
    for name, value in overrides.items():
        setattr(fields, name, value)
    return fields


@pytest.fixture
def validator() -> CnicValidator:
    """A validator pinned to a fixed date, so tests do not drift."""
    return CnicValidator(get_settings().ocr, today=TODAY)


@pytest.mark.unit
def test_a_consistent_card_passes(validator: CnicValidator) -> None:
    outcome = validator.validate(build_fields())
    assert outcome.consistent is True
    assert not outcome.errors
    assert outcome.confidence_penalty == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_gender_parity_agreement_is_recorded(validator: CnicValidator) -> None:
    outcome = validator.validate(build_fields())
    codes = [finding.code for finding in outcome.findings]
    assert "CNIC_GENDER_CONSISTENT" in codes


@pytest.mark.unit
def test_gender_parity_disagreement_is_an_error(validator: CnicValidator) -> None:
    """The card says male; the number's final digit is even, meaning female.
    Either a digit was misread or the document was altered."""
    fields = build_fields()
    fields.gender = FieldValue(value="M", confidence=0.9, source="label")

    outcome = validator.validate(fields)

    assert outcome.consistent is False
    codes = [finding.code for finding in outcome.errors]
    assert "CNIC_GENDER_MISMATCH" in codes
    assert outcome.confidence_penalty > 0.2


@pytest.mark.unit
def test_a_derived_gender_cannot_cross_check_itself(validator: CnicValidator) -> None:
    """Circular: the value came *from* the number, so agreeing with it proves
    nothing. The check is withheld and its absence reported."""
    fields = build_fields()
    fields.gender = FieldValue(value="F", confidence=0.5, source="derived")

    outcome = validator.validate(fields)
    codes = [finding.code for finding in outcome.findings]

    assert "CNIC_GENDER_DERIVED" in codes
    assert "CNIC_GENDER_CONSISTENT" not in codes


@pytest.mark.unit
def test_an_issue_date_before_birth_is_an_error(validator: CnicValidator) -> None:
    fields = build_fields()
    fields.date_of_issue = FieldValue(value=dt.date(1990, 1, 1), confidence=0.9)

    outcome = validator.validate(fields)
    assert "CNIC_ISSUE_BEFORE_BIRTH" in [f.code for f in outcome.errors]


@pytest.mark.unit
def test_an_expiry_before_issue_is_an_error(validator: CnicValidator) -> None:
    fields = build_fields()
    fields.date_of_expiry = FieldValue(value=dt.date(2015, 1, 1), confidence=0.9)

    outcome = validator.validate(fields)
    assert "CNIC_EXPIRY_BEFORE_ISSUE" in [f.code for f in outcome.errors]


@pytest.mark.unit
def test_an_unusual_validity_term_is_a_warning(validator: CnicValidator) -> None:
    """Not an error: NADRA does issue non-standard terms. Most likely a
    misread year digit, which is worth flagging without rejecting."""
    fields = build_fields()
    fields.date_of_expiry = FieldValue(value=dt.date(2022, 7, 22), confidence=0.9)

    outcome = validator.validate(fields)
    codes = [f.code for f in outcome.findings]
    assert "CNIC_UNUSUAL_VALIDITY_TERM" in codes
    assert outcome.consistent is True


@pytest.mark.unit
@pytest.mark.parametrize("years", [5, 7, 10, 15])
def test_standard_validity_terms_pass(validator: CnicValidator, years: int) -> None:
    fields = build_fields()
    issue = dt.date(2015, 6, 1)
    fields.date_of_issue = FieldValue(value=issue, confidence=0.9)
    fields.date_of_expiry = FieldValue(
        value=dt.date(2015 + years, 6, 1), confidence=0.9
    )

    outcome = validator.validate(fields)
    assert "CNIC_UNUSUAL_VALIDITY_TERM" not in [f.code for f in outcome.findings]


@pytest.mark.unit
def test_a_future_birth_date_is_an_error(validator: CnicValidator) -> None:
    fields = build_fields()
    fields.date_of_birth = FieldValue(value=dt.date(2030, 1, 1), confidence=0.9)

    outcome = validator.validate(fields)
    assert "CNIC_BIRTH_IN_FUTURE" in [f.code for f in outcome.errors]


@pytest.mark.unit
def test_an_implausible_age_is_an_error(validator: CnicValidator) -> None:
    fields = build_fields()
    fields.date_of_birth = FieldValue(value=dt.date(1850, 1, 1), confidence=0.9)
    fields.date_of_issue = FieldValue(value=dt.date(2019, 7, 22), confidence=0.9)

    outcome = validator.validate(fields)
    assert "CNIC_IMPLAUSIBLE_AGE" in [f.code for f in outcome.errors]


@pytest.mark.unit
def test_an_expired_card_is_a_finding_not_a_defect(validator: CnicValidator) -> None:
    """A correct reading of an out-of-date document. Whether that is
    acceptable is a policy decision for the rules engine, not an OCR error."""
    fields = build_fields()
    fields.date_of_issue = FieldValue(value=dt.date(2010, 1, 1), confidence=0.9)
    fields.date_of_expiry = FieldValue(value=dt.date(2020, 1, 1), confidence=0.9)

    outcome = validator.validate(fields)

    assert outcome.expired is True
    assert "CNIC_EXPIRED" in [f.code for f in outcome.findings]
    assert outcome.consistent is True
    expired = next(f for f in outcome.findings if f.code == "CNIC_EXPIRED")
    assert expired.confidence_penalty == 0.0


@pytest.mark.unit
def test_a_lifetime_card_never_expires(validator: CnicValidator) -> None:
    fields = build_fields()
    fields.is_lifetime = True
    fields.date_of_expiry = FieldValue()

    outcome = validator.validate(fields)
    assert outcome.expired is False


@pytest.mark.unit
def test_too_few_fields_is_an_error(validator: CnicValidator) -> None:
    fields = CnicFields()
    fields.cnic_number = FieldValue(value="42101-8375926-4", confidence=0.9)

    outcome = validator.validate(fields)
    assert "CNIC_TOO_FEW_FIELDS" in [f.code for f in outcome.errors]


@pytest.mark.unit
def test_findings_are_ordered_by_severity(validator: CnicValidator) -> None:
    fields = build_fields()
    fields.gender = FieldValue(value="M", confidence=0.9, source="label")
    fields.date_of_expiry = FieldValue(value=dt.date(2022, 7, 22), confidence=0.9)

    outcome = validator.validate(fields)
    severities = [f.severity for f in outcome.findings]
    assert severities[0] is Severity.ERROR


@pytest.mark.unit
def test_the_outcome_serialises(validator: CnicValidator) -> None:
    import json

    json.dumps(validator.validate(build_fields()).as_dict())


# --------------------------------------------------------------------------- #
# Field container
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_completeness_counts_only_required_fields() -> None:
    fields = build_fields()
    assert fields.completeness == pytest.approx(1.0)
    assert not fields.missing_fields


@pytest.mark.unit
def test_missing_fields_are_reported() -> None:
    fields = build_fields()
    fields.gender = FieldValue()
    fields.date_of_expiry = FieldValue()

    assert set(fields.missing_fields) == {"gender", "date_of_expiry"}
    assert fields.completeness == pytest.approx(4 / 6)


@pytest.mark.unit
def test_mean_confidence_ignores_absent_fields() -> None:
    """Absence is measured by completeness; folding it in here would conflate
    'read badly' with 'not read at all'."""
    fields = CnicFields()
    fields.cnic_number = FieldValue(value="42101-8375926-4", confidence=0.8)
    assert fields.mean_confidence() == pytest.approx(0.8)


@pytest.mark.unit
def test_the_summary_never_contains_a_field_value() -> None:
    """The whole content of a CNIC is personal data, and this is the object
    that reaches the log sink."""
    summary = build_fields().summary()
    serialised = repr(summary)

    assert "AYESHA" not in serialised
    assert "42101" not in serialised
    assert "1994" not in serialised
    assert set(summary) == {
        "present", "missing", "completeness", "mean_confidence", "province"
    }


@pytest.mark.unit
def test_raw_text_is_redacted_by_default() -> None:
    value = FieldValue(value="AYESHA KHAN", confidence=0.9, raw="Name AYESHA KHAN")
    assert "raw" not in value.as_dict()
    assert value.as_dict(redact=False)["raw"] == "Name AYESHA KHAN"
