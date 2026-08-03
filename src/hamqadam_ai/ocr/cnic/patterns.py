"""Pakistani CNIC semantics: number structure, dates and OCR confusions.

What a CNIC number actually encodes
-----------------------------------
Thirteen digits, conventionally printed ``XXXXX-XXXXXXX-X``:

===========  =========================================================
``[0]``      Province / region of registration
``[0:5]``    Locality code - province, division, district, tehsil
``[5:12]``   Family and individual serial within that locality
``[12]``     **Gender digit: odd is male, even is female**
===========  =========================================================

That last property is the single most useful validation in this module. It is
an independent cross-check on the printed gender field, and it costs nothing.
When the two disagree, either the OCR misread a digit or the document has been
altered - both worth surfacing.

**There is no checksum.** Unlike many national identifiers the CNIC carries no
check digit, so a syntactically valid number cannot be verified offline. This
module does not invent one. Anything claiming to "validate" a CNIC beyond
structure, province code and gender parity is guessing.

OCR character confusion
-----------------------
The probe that motivated this module read ``National Identity Card`` as
``National ldentity Card`` - a capital I taken for a lowercase L. That family
of substitution is systematic and predictable, and correcting it in the right
context is the difference between finding a field and missing it. Crucially
the correction is **context-dependent**: ``O`` must become ``0`` inside a
number and must not inside a name, so the substitutions are applied per-field
rather than globally.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Final, NamedTuple

# --------------------------------------------------------------------------- #
# Province codes
# --------------------------------------------------------------------------- #

#: First digit of the CNIC to region of registration.
#:
#: Informational and a weak structural check only. NADRA reallocates district
#: codes over time, so an unexpected leading digit is reported as a finding
#: rather than treated as proof of forgery.
PROVINCE_CODES: Final[dict[str, str]] = {
    "1": "Khyber Pakhtunkhwa",
    "2": "FATA",
    "3": "Punjab",
    "4": "Sindh",
    "5": "Balochistan",
    "6": "Islamabad Capital Territory",
    "7": "Gilgit-Baltistan",
    "8": "Azad Jammu and Kashmir",
}


# --------------------------------------------------------------------------- #
# Character confusion
# --------------------------------------------------------------------------- #

#: Glyphs an OCR reads as digits when the context is numeric.
_TO_DIGIT: Final[dict[str, str]] = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "|": "1", "!": "1", "i": "1",
    "Z": "2", "z": "2",
    "E": "3",
    "A": "4",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7", "?": "7",
    "B": "8",
    "g": "9", "q": "9",
}

#: Glyphs an OCR reads as letters when the context is alphabetic.
_TO_LETTER: Final[dict[str, str]] = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "|": "I",
}


def normalise_confusions(text: str, *, target: str = "digits") -> str:
    """Correct systematic OCR glyph substitutions.

    Args:
        text: The recognised string.
        target: ``digits`` to resolve ambiguous glyphs towards numerals,
            ``letters`` towards alphabetic characters.

    Returns:
        The corrected string.

    Note:
        Applied per-field, never globally. ``O`` must become ``0`` inside a
        CNIC number and must emphatically not inside the name *ROBERT*.

    Example:
        >>> normalise_confusions("4210I-83759Z6-4", target="digits")
        '42101-8375926-4'
        >>> normalise_confusions("Nati0nal", target="letters")
        'NatiOnal'
    """
    table = _TO_DIGIT if target == "digits" else _TO_LETTER
    return "".join(table.get(character, character) for character in text)


# --------------------------------------------------------------------------- #
# CNIC number
# --------------------------------------------------------------------------- #

#: A CNIC as printed, with or without the conventional hyphens. Anchored on
#: non-digit boundaries so it does not match a fragment of a longer run.
CNIC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(\d{5})[\s\-‐-―.]?(\d{7})[\s\-‐-―.]?(\d)(?!\d)"
)

#: A looser pass used only after confusion correction, when the strict pattern
#: found nothing.
CNIC_LOOSE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![0-9])([0-9OoIlSsBZzGT|]{5})[\s\-‐-―._]?"
    r"([0-9OoIlSsBZzGT|]{7})[\s\-‐-―._]?([0-9OoIlSsBZzGT|])(?![0-9])"
)


class CnicNumber(NamedTuple):
    """A parsed CNIC number and what its structure implies.

    Attributes:
        digits: The thirteen digits with no separators.
        formatted: Conventional ``XXXXX-XXXXXXX-X`` presentation.
        province: Region implied by the leading digit, or ``None`` when the
            digit is outside the known allocation.
        implied_gender: ``M`` for an odd final digit, ``F`` for an even one.
        corrected: Whether confusion correction was needed to parse it, which
            lowers confidence in the reading.
    """

    digits: str
    formatted: str
    province: str | None
    implied_gender: str
    corrected: bool

    @property
    def locality_code(self) -> str:
        """The first five digits - province, division, district, tehsil."""
        return self.digits[:5]


def parse_cnic_number(text: str, *, allow_correction: bool = True) -> CnicNumber | None:
    """Find and parse a CNIC number in a line of recognised text.

    Args:
        text: A recognised text line, which may contain a label as well as the
            number.
        allow_correction: Retry with glyph-confusion correction when the strict
            pattern finds nothing.

    Returns:
        The parsed number, or ``None`` when the text contains none.

    Example:
        >>> parsed = parse_cnic_number("Identity Number 42101-8375926-4")
        >>> parsed.formatted, parsed.implied_gender, parsed.province
        ('42101-8375926-4', 'F', 'Sindh')
    """
    match = CNIC_PATTERN.search(text)
    corrected = False

    if match is None and allow_correction:
        loose = CNIC_LOOSE_PATTERN.search(text)
        if loose is None:
            return None
        groups = [normalise_confusions(group, target="digits") for group in loose.groups()]
        if not all(group.isdigit() for group in groups):
            return None
        match_groups = tuple(groups)
        corrected = True
    elif match is None:
        return None
    else:
        match_groups = match.groups()

    digits = "".join(match_groups)
    if len(digits) != 13 or not digits.isdigit():
        return None

    return CnicNumber(
        digits=digits,
        formatted=f"{digits[:5]}-{digits[5:12]}-{digits[12]}",
        province=PROVINCE_CODES.get(digits[0]),
        implied_gender="M" if int(digits[12]) % 2 == 1 else "F",
        corrected=corrected,
    )


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

#: CNIC dates are printed ``DD.MM.YYYY``. Slashes and hyphens are accepted
#: because a photographed card sometimes renders the dot as either.
DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(\d{1,2})\s*[./\-‐-―]\s*(\d{1,2})\s*[./\-‐-―]\s*(\d{4})(?!\d)"
)

#: Same shape, but tolerating glyphs that confusion correction can resolve.
DATE_LOOSE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![0-9])([0-9OoIlSsBZzGT|]{1,2})\s*[./\-‐-―_]\s*"
    r"([0-9OoIlSsBZzGT|]{1,2})\s*[./\-‐-―_]\s*([0-9OoIlSsBZzGT|]{4})(?![0-9])"
)


def parse_cnic_date(text: str, *, allow_correction: bool = True) -> dt.date | None:
    """Find and parse a ``DD.MM.YYYY`` date in recognised text.

    Day-first is assumed because that is what the card prints. Interpreting
    ``03.04.1994`` as March in a system that reads Pakistani documents would be
    a silent, systematic error affecting every ambiguous date, so the
    convention is fixed rather than guessed.

    Args:
        text: A recognised text line.
        allow_correction: Retry with glyph-confusion correction.

    Returns:
        The parsed date, or ``None`` when the text contains no valid one.

    Example:
        >>> parse_cnic_date("Date of Birth 14.03.1994")
        datetime.date(1994, 3, 14)
    """
    match = DATE_PATTERN.search(text)
    groups: tuple[str, ...] | None = None

    if match is not None:
        groups = match.groups()
    elif allow_correction:
        loose = DATE_LOOSE_PATTERN.search(text)
        if loose is None:
            return None
        candidate = tuple(
            normalise_confusions(group, target="digits") for group in loose.groups()
        )
        if all(group.isdigit() for group in candidate):
            groups = candidate

    if groups is None:
        return None

    try:
        day, month, year = (int(group) for group in groups)
        return dt.date(year, month, day)
    except ValueError:
        # A real calendar rejection - 31.02, month 13, year 0000 - not a parse
        # failure. Returning None is correct; the caller reports the field as
        # unreadable rather than inventing a nearby valid date.
        return None


# --------------------------------------------------------------------------- #
# Gender
# --------------------------------------------------------------------------- #

#: Printed gender values seen on the card, normalised to ``M`` or ``F``.
_GENDER_VALUES: Final[dict[str, str]] = {
    "M": "M", "MALE": "M", "MAN": "M",
    "F": "F", "FEMALE": "F", "WOMAN": "F",
    # The card prints a single glyph, and these are its common misreadings.
    "N": "M", "H": "M",
    "E": "F", "P": "F",
}


def parse_gender(text: str) -> str | None:
    """Normalise a printed gender value to ``M`` or ``F``.

    Returns ``None`` for anything unrecognised rather than guessing. The
    CNIC's final digit independently encodes gender, so a missing printed
    value is recoverable and a wrongly-guessed one is not.
    """
    cleaned = re.sub(r"[^A-Za-z]", "", text).upper()
    if not cleaned:
        return None
    if cleaned in _GENDER_VALUES:
        return _GENDER_VALUES[cleaned]
    # A label line such as "Gender M" - take the trailing token.
    tokens = cleaned.split()
    for token in reversed(tokens):
        if token in _GENDER_VALUES:
            return _GENDER_VALUES[token]
    return None


# --------------------------------------------------------------------------- #
# Field labels
# --------------------------------------------------------------------------- #

#: Labels printed on the English side of the card, with the variants OCR
#: produces. Matching is fuzzy, so these are anchors rather than exact keys.
FIELD_LABELS: Final[dict[str, tuple[str, ...]]] = {
    "name": ("name",),
    "father_name": ("father name", "fathername", "father's name", "husband name"),
    "gender": ("gender", "sex"),
    "country_of_stay": ("country of stay", "country"),
    "cnic_number": ("identity number", "identity no", "cnic", "id number"),
    "date_of_birth": ("date of birth", "birth", "dob"),
    "date_of_issue": ("date of issue", "issue"),
    "date_of_expiry": ("date of expiry", "expiry", "date of expiny"),
}

#: Text that identifies the document as a Pakistani CNIC at all. Used to
#: distinguish "this is a CNIC we could not read" from "this is not a CNIC".
DOCUMENT_MARKERS: Final[tuple[str, ...]] = (
    "islamic republic of pakistan",
    "national identity card",
    "pakistan",
    "nadra",
    "identity number",
)


def label_similarity(candidate: str, label: str) -> float:
    """Fuzzy match a recognised line against a known printed label.

    Uses a normalised edit ratio rather than equality because OCR routinely
    substitutes one character in a label - the probe read ``Identity`` as
    ``ldentity`` - and an exact match would miss the anchor entirely.

    Args:
        candidate: A recognised text line, already lowercased.
        label: The expected label.

    Returns:
        Similarity in ``[0, 1]``.
    """
    from difflib import SequenceMatcher

    return SequenceMatcher(None, candidate, label).ratio()


__all__ = [
    "CNIC_LOOSE_PATTERN",
    "CNIC_PATTERN",
    "DATE_PATTERN",
    "DOCUMENT_MARKERS",
    "FIELD_LABELS",
    "PROVINCE_CODES",
    "CnicNumber",
    "label_similarity",
    "normalise_confusions",
    "parse_cnic_date",
    "parse_cnic_number",
    "parse_gender",
]
