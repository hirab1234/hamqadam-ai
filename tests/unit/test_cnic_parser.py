"""Field extraction from recognised text, with no OCR engine in the loop.

Every test here builds :class:`TextLine` objects by hand. That is deliberate:
it means the parser - which is where the CNIC-specific knowledge lives - is
tested against exactly the failure modes real engines produce (a dropped
label, a merged row, a substituted glyph, a wrapped value) rather than only
against whatever a particular engine happens to emit today.
"""

from __future__ import annotations

import datetime as dt

import pytest

from hamqadam_ai.ocr.base import OcrOutput, TextLine, quad_from_box
from hamqadam_ai.ocr.cnic.parser import CnicParser, looks_like_cnic

LINE_HEIGHT = 30.0
ROW_PITCH = 60.0
LABEL_X = 60.0
VALUE_X = 420.0


def line(text: str, *, row: int, x: float = LABEL_X, width: float = 300.0,
         confidence: float = 0.95, height: float = LINE_HEIGHT) -> TextLine:
    """One text line placed on a notional card at a given row."""
    top = 100.0 + row * ROW_PITCH
    return TextLine(
        text=text,
        confidence=confidence,
        quad=quad_from_box(x, top, x + width, top + height),
    )


def card(rows: list[tuple[str, str | None]], **kwargs: object) -> OcrOutput:
    """Lay out label/value pairs as an engine would return them.

    A ``None`` value means the label was detected but its value was not, which
    is the single most common real failure and the one the recovery path
    exists for.
    """
    lines: list[TextLine] = []
    for index, (label, value) in enumerate(rows):
        lines.append(line(label, row=index))
        if value is not None:
            lines.append(line(value, row=index, x=VALUE_X, width=360.0))
    return OcrOutput(lines=lines, engine="synthetic", **kwargs)  # type: ignore[arg-type]


STANDARD = [
    ("Name", "AYESHA KHAN"),
    ("Father Name", "MUHAMMAD KHAN"),
    ("Gender", "F"),
    ("Country of Stay", "Pakistan"),
    ("Identity Number", "42101-8375926-4"),
    ("Date of Birth", "14.03.1994"),
    ("Date of Issue", "22.07.2019"),
    ("Date of Expiry", "22.07.2029"),
]


@pytest.fixture
def parser() -> CnicParser:
    return CnicParser()


# --------------------------------------------------------------------------- #
# The straightforward case
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_well_read_card_yields_every_field(parser: CnicParser) -> None:
    fields = parser.parse(card(STANDARD))

    assert fields.cnic_number.value == "42101-8375926-4"
    assert fields.full_name.value == "AYESHA KHAN"
    assert fields.father_name.value == "MUHAMMAD KHAN"
    assert fields.gender.value == "F"
    assert fields.date_of_birth.value == dt.date(1994, 3, 14)
    assert fields.date_of_issue.value == dt.date(2019, 7, 22)
    assert fields.date_of_expiry.value == dt.date(2029, 7, 22)
    assert fields.completeness == pytest.approx(1.0)


@pytest.mark.unit
def test_the_number_drives_the_derived_attributes(parser: CnicParser) -> None:
    fields = parser.parse(card(STANDARD))
    assert fields.province == "Sindh"
    assert fields.implied_gender == "F"


@pytest.mark.unit
def test_a_read_gender_is_not_marked_derived(parser: CnicParser) -> None:
    """Provenance matters: only a gender actually read off the card can be
    cross-checked against the number's parity."""
    fields = parser.parse(card(STANDARD))
    assert fields.gender.source != "derived"


# --------------------------------------------------------------------------- #
# Labels the engine mangled
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "label",
    ["Identity Number", "ldentity Number", "Identlty Numbor", "IDENTITY NUMBER",
     "Identity  Number"],
)
def test_a_misread_label_still_anchors(parser: CnicParser, label: str) -> None:
    """Labels are matched by similarity, not equality. The probe genuinely
    returned ``ldentity`` - capital I read as lowercase l - and an exact match
    would have lost the whole row."""
    rows = [(label if name == "Identity Number" else name, value)
            for name, value in STANDARD]
    fields = parser.parse(card(rows))
    assert fields.cnic_number.value == "42101-8375926-4"


@pytest.mark.unit
def test_a_label_too_far_gone_does_not_anchor(parser: CnicParser) -> None:
    """The similarity threshold has to reject as well as accept, or every
    line on the card matches every label."""
    rows = [("Qxzvbn Plmk" if name == "Gender" else name, value)
            for name, value in STANDARD]
    fields = parser.parse(card(rows))
    # The gender value 'F' is now an orphan line with no label beside it.
    assert fields.gender.source != "label"


# --------------------------------------------------------------------------- #
# Values the engine misplaced
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_label_and_value_merged_into_one_line(parser: CnicParser) -> None:
    """Common when the two sit close together: the detector boxes them as a
    single region."""
    output = OcrOutput(
        lines=[
            line("Name AYESHA KHAN", row=0, width=600.0),
            line("Identity Number 42101-8375926-4", row=1, width=600.0),
            line("Date of Birth 14.03.1994", row=2, width=600.0),
        ],
        engine="synthetic",
    )
    fields = parser.parse(output)

    assert fields.cnic_number.value == "42101-8375926-4"
    assert fields.date_of_birth.value == dt.date(1994, 3, 14)
    assert fields.full_name.value == "AYESHA KHAN"


@pytest.mark.unit
def test_a_value_printed_below_its_label(parser: CnicParser) -> None:
    """Some card layouts stack rather than pair. Nothing beside the label, so
    the parser looks underneath it."""
    output = OcrOutput(
        lines=[
            line("Name", row=0),
            line("AYESHA KHAN", row=1, x=LABEL_X),
            line("Identity Number", row=2),
            line("42101-8375926-4", row=3, x=LABEL_X),
        ],
        engine="synthetic",
    )
    fields = parser.parse(output)

    assert fields.full_name.value == "AYESHA KHAN"
    assert fields.cnic_number.value == "42101-8375926-4"


@pytest.mark.unit
def test_the_number_is_found_with_no_label_at_all(parser: CnicParser) -> None:
    """Its shape is unambiguous, so it needs no anchor - which is why it is
    the most reliable field on a badly-read card."""
    output = OcrOutput(lines=[line("42101-8375926-4", row=0)], engine="synthetic")
    fields = parser.parse(output)

    assert fields.cnic_number.value == "42101-8375926-4"
    assert fields.cnic_number.source == "pattern"


# --------------------------------------------------------------------------- #
# Dates without labels
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_unlabelled_dates_are_assigned_chronologically(parser: CnicParser) -> None:
    """With no labels, order is the only signal - and it is a sound one:
    birth precedes issue precedes expiry on every card ever issued."""
    output = OcrOutput(
        lines=[
            line("42101-8375926-4", row=0),
            line("22.07.2029", row=1),
            line("14.03.1994", row=2),
            line("22.07.2019", row=3),
        ],
        engine="synthetic",
    )
    fields = parser.parse(output)

    assert fields.date_of_birth.value == dt.date(1994, 3, 14)
    assert fields.date_of_issue.value == dt.date(2019, 7, 22)
    assert fields.date_of_expiry.value == dt.date(2029, 7, 22)
    assert fields.date_of_birth.source == "chronological"


@pytest.mark.unit
def test_two_unlabelled_dates_are_not_guessed(parser: CnicParser) -> None:
    """Two dates could be birth+issue, birth+expiry or issue+expiry. Assigning
    them would be a coin toss dressed up as a reading, so the parser declines
    and lets completeness report the gap."""
    output = OcrOutput(
        lines=[line("14.03.1994", row=0), line("22.07.2019", row=1)],
        engine="synthetic",
    )
    fields = parser.parse(output)
    assert not fields.date_of_expiry.present


@pytest.mark.unit
def test_a_labelled_date_wins_over_chronology(parser: CnicParser) -> None:
    """The label pass runs first and its results are not overwritten."""
    output = OcrOutput(
        lines=[
            line("Date of Expiry", row=0),
            line("22.07.2029", row=0, x=VALUE_X),
            line("14.03.1994", row=1),
            line("22.07.2019", row=2),
        ],
        engine="synthetic",
    )
    fields = parser.parse(output)

    assert fields.date_of_expiry.value == dt.date(2029, 7, 22)
    assert fields.date_of_expiry.source == "label"


@pytest.mark.unit
def test_a_lifetime_card_is_recognised(parser: CnicParser) -> None:
    rows = [(name, "Lifetime" if name == "Date of Expiry" else value)
            for name, value in STANDARD]
    fields = parser.parse(card(rows))

    assert fields.is_lifetime is True
    assert not fields.date_of_expiry.present


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_two_name_fields_are_kept_apart(parser: CnicParser) -> None:
    """``Name`` is a substring of ``Father Name``, so a naive match assigns
    the father's name to both."""
    fields = parser.parse(card(STANDARD))
    assert fields.full_name.value == "AYESHA KHAN"
    assert fields.father_name.value == "MUHAMMAD KHAN"


@pytest.mark.unit
def test_glyph_confusion_is_never_corrected_in_a_name(parser: CnicParser) -> None:
    """The correction table maps ``0`` to ``O`` and back. Applying it to a
    name would silently alter the holder's identity, so it is not applied -
    an OCR error the reviewer can see beats a plausible-looking invention."""
    rows = [("Name", "AYESHA 0KHAN")] + STANDARD[1:]
    fields = parser.parse(card(rows))
    assert fields.full_name.value is not None
    assert "0KHAN" in fields.full_name.value


@pytest.mark.unit
def test_a_misread_digit_costs_the_name_confidence_not_the_name(
    parser: CnicParser,
) -> None:
    """Keeping a name that is right but for one glyph is worth more to the
    caller than discarding it - provided the score admits the doubt."""
    clean = parser.parse(card(STANDARD))
    rows = [("Name", "AYESHA 0KHAN")] + STANDARD[1:]
    smudged = parser.parse(card(rows))

    assert smudged.full_name.confidence < clean.full_name.confidence


@pytest.mark.unit
def test_a_name_of_digits_is_rejected(parser: CnicParser) -> None:
    rows = [("Name", "12345678")] + STANDARD[1:]
    fields = parser.parse(card(rows))
    assert not fields.full_name.present


@pytest.mark.unit
def test_a_line_that_is_mostly_digits_is_rejected(parser: CnicParser) -> None:
    """It passes the leading-letter check but is plainly a serial, not a
    name."""
    rows = [("Name", "A4210183759264")] + STANDARD[1:]
    fields = parser.parse(card(rows))
    assert not fields.full_name.present


@pytest.mark.unit
def test_a_name_merged_with_a_two_word_label(parser: CnicParser) -> None:
    """``Father Name`` must consume exactly two words, however the recogniser
    spelled them."""
    output = OcrOutput(
        lines=[line("Fathor Namo MUHAMMAD KHAN", row=0, width=600.0)],
        engine="synthetic",
    )
    fields = parser.parse(output)
    assert fields.father_name.value == "MUHAMMAD KHAN"


@pytest.mark.unit
def test_a_bare_label_yields_no_inline_value(parser: CnicParser) -> None:
    """Nothing follows it, so the parser must fall through to looking beside
    and below rather than reading the label as its own value."""
    output = OcrOutput(
        lines=[line("Name", row=0), line("AYESHA KHAN", row=0, x=VALUE_X)],
        engine="synthetic",
    )
    fields = parser.parse(output)
    assert fields.full_name.value == "AYESHA KHAN"
    assert fields.full_name.source == "label"


# --------------------------------------------------------------------------- #
# Gender
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_gender_falls_back_to_the_number(parser: CnicParser) -> None:
    """When the glyph is unreadable the number still carries the answer, and
    the field records where it came from so nothing later mistakes it for an
    independent reading."""
    rows = [(name, None if name == "Gender" else value) for name, value in STANDARD]
    fields = parser.parse(card(rows))

    assert fields.gender.value == "F"
    assert fields.gender.source == "derived"


@pytest.mark.unit
def test_a_read_gender_is_kept_even_when_it_contradicts(parser: CnicParser) -> None:
    """Overwriting it with the derived value would erase the disagreement -
    which is precisely the signal a tampered card produces."""
    rows = [(name, "M" if name == "Gender" else value) for name, value in STANDARD]
    fields = parser.parse(card(rows))

    assert fields.gender.value == "M"
    assert fields.implied_gender == "F"


# --------------------------------------------------------------------------- #
# Recovery helpers, used by the service's crop pass
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_gender_recovery_picks_the_most_confident_candidate(
    parser: CnicParser,
) -> None:
    recovered = parser.recover_gender([("x", 0.9), ("F", 0.3), ("f", 0.1)])
    assert recovered is not None
    assert recovered.value == "F"
    assert recovered.source == "crop"


@pytest.mark.unit
def test_gender_recovery_returns_nothing_when_nothing_matches(
    parser: CnicParser,
) -> None:
    assert parser.recover_gender([("###", 0.9), ("", 0.8)]) is None


@pytest.mark.unit
def test_name_recovery_ignores_dates_and_numbers(parser: CnicParser) -> None:
    recovered = parser.recover_name([("14.03.1994", 0.9), ("AYESHA KHAN", 0.5)])
    assert recovered is not None
    assert recovered.value == "AYESHA KHAN"


# --------------------------------------------------------------------------- #
# Is this even a CNIC?
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_cnic_is_identified_as_one() -> None:
    matched, score = looks_like_cnic(card(STANDARD))
    assert matched is True
    assert score > 0.5


@pytest.mark.unit
def test_an_unrelated_document_is_not() -> None:
    """A user uploading a utility bill should be told the document is wrong,
    not handed a confident reading of six fields that are not there."""
    output = OcrOutput(
        lines=[
            line("ELECTRICITY BILL", row=0),
            line("Consumer Reference 04 11223 3445566", row=1),
            line("Units Consumed 312", row=2),
            line("Amount Payable 8,420", row=3),
        ],
        engine="synthetic",
    )
    matched, score = looks_like_cnic(output)

    assert matched is False
    assert score < 0.5


@pytest.mark.unit
def test_a_blank_page_is_not_a_cnic() -> None:
    matched, score = looks_like_cnic(OcrOutput(engine="synthetic"))
    assert matched is False
    assert score == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Value regions, used to crop for recovery
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_region_beside_a_label_is_locatable(parser: CnicParser) -> None:
    _fields, anchors = parser.parse_with_anchors(card(STANDARD))
    anchor = anchors["gender"]

    region = parser.value_region(anchor, width=1000, height=800)

    assert region is not None
    x1, y1, x2, y2 = region
    assert x1 >= anchor.line.x2 - 1  # begins at or after the label
    assert x2 <= 1000 and y2 <= 800  # clipped to the image
    assert x2 > x1 and y2 > y1


@pytest.mark.unit
def test_a_label_at_the_right_edge_has_no_region(parser: CnicParser) -> None:
    """Nothing to crop, and returning a zero-width box would crash the
    recogniser rather than simply reporting the field missing."""
    output = OcrOutput(
        lines=[TextLine("Gender", 0.9, quad_from_box(960.0, 100.0, 1000.0, 130.0))],
        engine="synthetic",
    )
    _fields, anchors = parser.parse_with_anchors(output)

    assert parser.value_region(anchors["gender"], width=1000, height=800) is None


# --------------------------------------------------------------------------- #
# Degenerate input
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_no_text_parses_to_no_fields(parser: CnicParser) -> None:
    fields = parser.parse(OcrOutput(engine="synthetic"))
    assert fields.completeness == pytest.approx(0.0)
    assert fields.mean_confidence() == pytest.approx(0.0)


@pytest.mark.unit
def test_pure_noise_does_not_invent_fields(parser: CnicParser) -> None:
    output = OcrOutput(
        lines=[line(text, row=index) for index, text in
               enumerate(["#$%^&", "|||", "~~~", "\\/\\/"])],
        engine="synthetic",
    )
    fields = parser.parse(output)
    assert not fields.cnic_number.present
    assert not fields.date_of_birth.present


class TestCardBoilerplateIsNeverAName:
    """The card's own pre-printed text cannot be anybody's name.

    From a real submission: a CNIC photograph whose outline could not be
    isolated returned `name = "ISLAMIC REPUBLIC OF PAKISTAN"` at **0.94
    confidence** - the card's header line, present on every genuine Pakistani
    CNIC ever issued.

    A confidently-wrong value is worse than a missing one. A Backend can handle
    "the name could not be read"; it has no way to detect that a high-confidence
    string is the document's letterhead.
    """

    @pytest.mark.parametrize(
        "boilerplate",
        [
            "Islamic Republic of Pakistan",
            "ISLAMIC REPUBLIC OF PAKISTAN",
            "National Identity Card",
            "NADRA",
        ],
    )
    def test_boilerplate_is_refused_as_a_name(self, boilerplate: str) -> None:
        parser = CnicParser()
        assert parser._clean_name(boilerplate) is None  # noqa: SLF001

    @pytest.mark.parametrize(
        "name", ["MUHAMMAD KHAN", "AYESHA BIBI", "HIRA BUKHARI", "ALI RAZA SHAH"]
    )
    def test_real_names_still_accepted(self, name: str) -> None:
        parser = CnicParser()
        assert parser._clean_name(name) == name.upper()  # noqa: SLF001
