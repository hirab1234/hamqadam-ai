"""Turning recognised text lines into CNIC fields.

Three strategies, applied in order of reliability
-------------------------------------------------
**Pattern extraction** runs first for the fields that have an unmistakable
shape. A thirteen-digit group in ``5-7-1`` form is a CNIC number wherever it
appears on the card; a ``DD.MM.YYYY`` group is a date. These need no layout
assumption at all, which makes them robust to a card photographed at an angle,
partly cropped, or in an unfamiliar template revision.

**Label anchoring** then handles the free-text fields. The card prints
``Name``, ``Father Name``, ``Gender`` beside their values, so the parser finds
the label and takes what sits to its right on the same row. Label matching is
fuzzy, because the probe read ``Identity Number`` as ``ldentity Number`` and an
exact comparison would have missed the anchor entirely.

**Positional fallback** is last. When a label is found but nothing sits to its
right - the value landed on the following line, or the row grouping split - the
next line in reading order is taken, provided it does not look like another
label.

Why the three dates need disambiguating
---------------------------------------
Pattern extraction finds three dates and cannot tell which is which. They are
assigned by label where labels were read, and otherwise **by chronological
order**: on a CNIC the date of birth necessarily precedes the date of issue,
which necessarily precedes the date of expiry. That ordering is a property of
the document, not a heuristic, so it holds even when every label was missed.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.ocr.base import OcrOutput, TextLine
from hamqadam_ai.ocr.cnic.fields import CnicFields, FieldValue
from hamqadam_ai.ocr.cnic.patterns import (
    DOCUMENT_MARKERS,
    FIELD_LABELS,
    label_similarity,
    normalise_confusions,
    parse_cnic_date,
    parse_cnic_number,
    parse_gender,
)

log = get_logger(__name__)

#: Fuzzy-match score at or above which a line is treated as a printed label.
#: 0.78 accepts the one-character substitutions OCR actually makes
#: (``Identity`` for ``ldentity`` scores 0.875) while rejecting unrelated text.
LABEL_MATCH_THRESHOLD = 0.78

#: A value must start within this many line-heights of the label's right edge
#: to count as being on the same row. Generous, because the gap between label
#: and value on a CNIC is wide.
SAME_ROW_GAP_LIMIT = 14.0

#: Characters a printed name may contain. Anything else means the line is not
#: a name, which is what stops a date or an identity number being accepted.
#:
#: Digits are permitted, but only as a small minority - see
#: :data:`_MAX_NAME_DIGIT_SHARE`. Forbidding them outright discards a name that
#: is right but for one glyph the recogniser turned into a ``0``, and a name
#: with one wrong character is far more use to the caller than no name at all.
_NAME_ALLOWED = re.compile(r"^[A-Za-z][A-Za-z0-9\s.'\-]*$")

#: Largest share of a name's alphanumeric characters that may be digits before
#: the line is rejected as not a name. One stray digit in a twelve-character
#: name is an OCR slip; a line that is a third digits is an identity number, a
#: date or a serial that happened to start with a letter.
_MAX_NAME_DIGIT_SHARE = 0.3

#: Confidence multiplier applied to a name containing a digit. The value is
#: probably right and is reported, but something in it was certainly misread,
#: and the score has to say so.
_NAME_DIGIT_CONFIDENCE_FACTOR = 0.7

#: Text indicating lifetime validity in place of an expiry date.
_LIFETIME_MARKERS = ("lifetime", "life time", "life-time")


def _prefix_similarity(text: str, variant: str) -> float:
    """How closely the opening words of ``text`` match a known label.

    Word-count-wise rather than character-wise, so a two-word label consumes
    exactly two words however badly either was spelled. Returns ``0.0`` when
    the line is not longer than the label, because then it is the label alone
    and the whole-line comparison already covers it.
    """
    words = text.split()
    count = len(variant.split())
    if count >= len(words):
        return 0.0
    return label_similarity(" ".join(words[:count]).lower(), variant)


@dataclass(slots=True)
class _Anchor:
    """A recognised line identified as one of the card's printed labels."""

    field: str
    line: TextLine
    score: float


class CnicParser:
    """Extracts structured fields from recognised CNIC text.

    Stateless and free of model dependencies, so it is directly unit-testable
    against hand-written line sets - which is how the layout rules below are
    pinned, rather than by round-tripping a rendered card.
    """

    def parse(self, output: OcrOutput) -> CnicFields:
        """Extract every field from one OCR pass.

        Args:
            output: The recognised lines.

        Returns:
            The populated fields. Anything unreadable is left absent rather
            than guessed.
        """
        return self.parse_with_anchors(output)[0]

    def parse_with_anchors(
        self, output: OcrOutput
    ) -> tuple[CnicFields, dict[str, _Anchor]]:
        """Extract every field, returning the located labels as well.

        The caller needs the anchors to retry a field that was labelled but
        whose value the detector missed - see
        :meth:`~hamqadam_ai.services.ocr_service.OcrService._recover_missing`.
        """
        fields = CnicFields()
        lines = output.in_reading_order()
        if not lines:
            return fields, {}

        anchors = self._find_anchors(lines)

        self._extract_cnic_number(lines, anchors, fields)
        self._extract_dates(lines, anchors, fields)
        self._extract_gender(lines, anchors, fields)
        self._extract_names(lines, anchors, fields)
        self._extract_country(lines, anchors, fields)
        self._derive_from_number(fields)

        return fields, anchors

    # -- Targeted recovery ---------------------------------------------------- #

    def value_region(
        self, anchor: _Anchor, *, width: int, height: int
    ) -> tuple[int, int, int, int] | None:
        """The pixel region a label's value should occupy.

        Used to crop and re-recognise a value the full-page detector missed.
        That happens routinely for the gender field, whose printed value is a
        single glyph: text detectors are trained on lines and are poor at
        isolating one character against a wide margin. Handing the recogniser
        a crop that contains nothing else removes the detection problem
        entirely.

        Args:
            anchor: The located label.
            width: Source image width.
            height: Source image height.

        Returns:
            ``(x1, y1, x2, y2)``, or ``None`` when the region would be empty.
        """
        band = max(anchor.line.height, 1.0)
        x1 = int(max(0.0, anchor.line.x2 + band * 0.2))
        x2 = int(min(float(width), anchor.line.x2 + band * SAME_ROW_GAP_LIMIT))
        y1 = int(max(0.0, anchor.line.y1 - band * 0.6))
        y2 = int(min(float(height), anchor.line.y2 + band * 0.6))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return x1, y1, x2, y2

    def recover_gender(self, texts: list[tuple[str, float]]) -> FieldValue[str] | None:
        """Parse a gender value from a targeted crop's recognised text."""
        for text, confidence in texts:
            parsed = parse_gender(text)
            if parsed is not None:
                return FieldValue(
                    value=parsed,
                    # Slightly discounted: recognised in isolation, without the
                    # surrounding context that would corroborate it.
                    confidence=confidence * 0.9,
                    raw=text,
                    source="crop",
                )
        return None

    def recover_name(self, texts: list[tuple[str, float]]) -> FieldValue[str] | None:
        """Parse a name from a targeted crop's recognised text."""
        for text, confidence in texts:
            cleaned = self._clean_name(text)
            if cleaned is not None:
                return FieldValue(
                    value=cleaned,
                    confidence=confidence * 0.9,
                    raw=text,
                    source="crop",
                )
        return None

    # -- Anchors -------------------------------------------------------------- #

    def _find_anchors(self, lines: list[TextLine]) -> dict[str, _Anchor]:
        """Locate the card's printed labels among the recognised lines.

        Each label is claimed by its single best match, so a line reading
        ``Date of Birth`` cannot also anchor ``Date of Issue`` merely because
        the two strings are similar.
        """
        anchors: dict[str, _Anchor] = {}

        for field_name, variants in FIELD_LABELS.items():
            best: _Anchor | None = None
            for line in lines:
                text = line.normalised
                if not text:
                    continue
                score = max(label_similarity(text, variant) for variant in variants)
                # Also accept a label that leads a line containing its value,
                # such as "Identity Number 42101-8375926-4". Compared fuzzily
                # and word-count-wise, for the same reason the whole-line
                # comparison is fuzzy: the label is exactly as likely to be
                # misread when it shares a box with its value.
                score = max(
                    score,
                    max(_prefix_similarity(line.text, v) for v in variants),
                )
                if score < LABEL_MATCH_THRESHOLD:
                    continue
                # On a tie, prefer the shorter line: that is the bare label,
                # whose value needs no splitting out. Applying the preference
                # here rather than by discounting the score keeps the
                # acceptance threshold meaning what it says.
                if best is None or (score, -len(line.text)) > (
                    best.score, -len(best.line.text)
                ):
                    best = _Anchor(field=field_name, line=line, score=score)
            if best is not None:
                anchors[field_name] = best

        return anchors

    @staticmethod
    def _value_beside(
        anchor: _Anchor, lines: list[TextLine]
    ) -> TextLine | None:
        """Return the line sitting to the right of a label on the same row."""
        band = max(anchor.line.height, 1.0)
        candidates = [
            line
            for line in lines
            if line is not anchor.line
            and line.x1 >= anchor.line.x2 - band * 0.5
            and abs(line.centre_y - anchor.line.centre_y) <= band * 0.7
            and (line.x1 - anchor.line.x2) <= band * SAME_ROW_GAP_LIMIT
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda line: line.x1)

    @staticmethod
    def _value_below(
        anchor: _Anchor, lines: list[TextLine]
    ) -> TextLine | None:
        """Return the next line beneath a label, when it is not another label."""
        band = max(anchor.line.height, 1.0)
        below = [
            line
            for line in lines
            if line.centre_y > anchor.line.centre_y + band * 0.5
            and abs(line.x1 - anchor.line.x1) <= band * 4.0
        ]
        if not below:
            return None
        candidate = min(below, key=lambda line: line.centre_y)

        text = candidate.normalised
        for variants in FIELD_LABELS.values():
            if any(label_similarity(text, variant) >= LABEL_MATCH_THRESHOLD
                   for variant in variants):
                return None
        return candidate

    def _resolve_value(
        self, anchor: _Anchor, lines: list[TextLine]
    ) -> tuple[TextLine | None, str]:
        """Find a label's value, reporting which strategy located it."""
        beside = self._value_beside(anchor, lines)
        if beside is not None:
            return beside, "label"
        below = self._value_below(anchor, lines)
        if below is not None:
            return below, "label_below"
        return None, "none"

    # -- Structured fields ---------------------------------------------------- #

    def _extract_cnic_number(
        self, lines: list[TextLine], anchors: dict[str, _Anchor], fields: CnicFields
    ) -> None:
        """Find the identity number by shape, anywhere on the card."""
        best: tuple[float, TextLine, Any] = (0.0, lines[0], None)

        for line in lines:
            parsed = parse_cnic_number(line.text)
            if parsed is None:
                continue
            # A correction-assisted read is real but less certain.
            confidence = line.confidence * (0.85 if parsed.corrected else 1.0)
            # A number found on the line the label anchors is more likely the
            # identity number and not, say, a serial printed elsewhere.
            anchor = anchors.get("cnic_number")
            if anchor is not None and abs(
                line.centre_y - anchor.line.centre_y
            ) <= max(anchor.line.height, 1.0):
                confidence = min(1.0, confidence * 1.1)
            if confidence > best[0]:
                best = (confidence, line, parsed)

        confidence, line, parsed = best
        if parsed is None:
            return

        fields.cnic_number = FieldValue(
            value=parsed.formatted,
            confidence=confidence,
            raw=line.text,
            source="pattern",
            corrected=parsed.corrected,
        )
        fields.implied_gender = parsed.implied_gender
        fields.province = parsed.province

    def _extract_dates(
        self, lines: list[TextLine], anchors: dict[str, _Anchor], fields: CnicFields
    ) -> None:
        """Assign the three dates, by label where possible and otherwise by order."""
        found: list[tuple[dt.date, TextLine]] = []
        for line in lines:
            parsed = parse_cnic_date(line.text)
            if parsed is not None:
                found.append((parsed, line))

        if any(
            marker in line.normalised
            for line in lines
            for marker in _LIFETIME_MARKERS
        ):
            fields.is_lifetime = True

        assigned: set[dt.date] = set()

        # Pass one: labels, which are authoritative when present.
        for field_name in ("date_of_birth", "date_of_issue", "date_of_expiry"):
            anchor = anchors.get(field_name)
            if anchor is None:
                continue

            # The value may sit on the label's own line.
            inline = parse_cnic_date(anchor.line.text)
            if inline is not None:
                fields.set(
                    field_name,
                    FieldValue(
                        value=inline,
                        confidence=anchor.line.confidence,
                        raw=anchor.line.text,
                        source="label",
                    ),
                )
                assigned.add(inline)
                continue

            value_line, source = self._resolve_value(anchor, lines)
            if value_line is None:
                continue
            parsed = parse_cnic_date(value_line.text)
            if parsed is None:
                continue
            fields.set(
                field_name,
                FieldValue(
                    value=parsed,
                    confidence=value_line.confidence,
                    raw=value_line.text,
                    source=source,
                ),
            )
            assigned.add(parsed)

        # Pass two: chronology, for whatever the labels did not cover.
        #
        # On a CNIC the birth date necessarily precedes the issue date, which
        # necessarily precedes the expiry date. That is a property of the
        # document rather than a guess, so it holds even with no labels read.
        remaining = sorted(
            {date for date, _ in found} - assigned
        )
        slots = [
            name
            for name in ("date_of_birth", "date_of_issue", "date_of_expiry")
            if not fields.get(name).present
        ]
        if remaining and len(remaining) == len(slots):
            for name, value in zip(slots, remaining, strict=True):
                line = next(line for date, line in found if date == value)
                fields.set(
                    name,
                    FieldValue(
                        value=value,
                        # Lower than a labelled read: chronological assignment
                        # is sound but weaker evidence than the printed label.
                        confidence=line.confidence * 0.8,
                        raw=line.text,
                        source="chronological",
                    ),
                )

    def _extract_gender(
        self, lines: list[TextLine], anchors: dict[str, _Anchor], fields: CnicFields
    ) -> None:
        """Read the printed gender glyph.

        Frequently missed: the value is a single character, and text detectors
        are poor at isolating one glyph. When it is missed the field is left
        absent and the CNIC's final digit supplies the value instead, which the
        validator records as derived rather than read.
        """
        anchor = anchors.get("gender")
        if anchor is not None:
            inline = anchor.line.normalised
            for variant in FIELD_LABELS["gender"]:
                if inline.startswith(variant) and len(inline) > len(variant):
                    parsed = parse_gender(inline[len(variant):])
                    if parsed is not None:
                        fields.gender = FieldValue(
                            value=parsed,
                            confidence=anchor.line.confidence,
                            raw=anchor.line.text,
                            source="label",
                        )
                        return

            value_line, source = self._resolve_value(anchor, lines)
            if value_line is not None:
                parsed = parse_gender(value_line.text)
                if parsed is not None:
                    fields.gender = FieldValue(
                        value=parsed,
                        confidence=value_line.confidence,
                        raw=value_line.text,
                        source=source,
                    )

    def _extract_names(
        self, lines: list[TextLine], anchors: dict[str, _Anchor], fields: CnicFields
    ) -> None:
        """Read the holder's name and father's name from their labels."""
        for field_name in ("name", "father_name"):
            anchor = anchors.get(field_name)
            if anchor is None:
                continue
            # The value may be a separate line, or it may have been boxed
            # together with the label - which is what the detector does
            # whenever the two sit close on the card. Try the inline case
            # first: it is unambiguous when it fires, because the label was
            # already matched against this very line.
            inline = self._strip_label(anchor)
            if inline is not None:
                text, confidence, source = inline, anchor.line.confidence, "inline"
            else:
                value_line, source = self._resolve_value(anchor, lines)
                if value_line is None:
                    continue
                text, confidence = value_line.text, value_line.confidence

            cleaned = self._clean_name(text)
            if cleaned is None:
                continue
            if any(character.isdigit() for character in cleaned):
                confidence *= _NAME_DIGIT_CONFIDENCE_FACTOR

            target = "full_name" if field_name == "name" else "father_name"
            fields.set(
                target,
                FieldValue(
                    value=cleaned,
                    confidence=confidence,
                    raw=text,
                    source=source,
                ),
            )

    @staticmethod
    def _strip_label(anchor: _Anchor) -> str | None:
        """Return whatever follows the label on the label's own line.

        The anchor matched because the line *starts* with something close to a
        known label; if the line is longer than the label, the remainder is the
        value. Matching is done word-count-wise against each known variant so
        that a two-word label consumes exactly two words, whatever the
        recogniser did to their spelling.

        Returns:
            The trailing text, or ``None`` when the line is the label alone.
        """
        words = anchor.line.text.split()
        if len(words) < 2:
            return None

        for variant in FIELD_LABELS.get(anchor.field, ()):
            if _prefix_similarity(anchor.line.text, variant) >= LABEL_MATCH_THRESHOLD:
                remainder = " ".join(words[len(variant.split()):]).strip(" :-")
                return remainder or None
        return None

    @staticmethod
    def _clean_name(text: str) -> str | None:
        """Validate and tidy a candidate name.

        Rejects anything that is not plausibly a printed name. Digit-to-letter
        confusion correction is *not* applied: a name is free text, and
        "correcting" it would silently alter the holder's identity. A name the
        OCR read badly should be reported with low confidence, not repaired
        into a different name - so a stray digit is kept, visible, and costs
        the field confidence rather than the field itself.
        """
        candidate = " ".join(text.split())
        if len(candidate) < 2 or len(candidate) > 80:
            return None
        if not _NAME_ALLOWED.match(candidate):
            return None

        alphanumeric = [c for c in candidate if c.isalnum()]
        digits = sum(1 for c in alphanumeric if c.isdigit())
        if alphanumeric and digits / len(alphanumeric) > _MAX_NAME_DIGIT_SHARE:
            return None
        # A line that is entirely a known label is not a value.
        lowered = candidate.lower()
        for variants in FIELD_LABELS.values():
            if any(label_similarity(lowered, variant) >= 0.9 for variant in variants):
                return None

        # Nor is the card's own pre-printed text. Every Pakistani CNIC carries
        # "Islamic Republic of Pakistan" and "National Identity Card", so these
        # lines are guaranteed present on a genuine document and can never be
        # anybody's name.
        #
        # Without this, a real submission whose card outline could not be
        # isolated returned name="ISLAMIC REPUBLIC OF PAKISTAN" at 0.94
        # confidence - a wrong answer asserted confidently, which is worse than
        # reporting the field as missing, because the Backend has no way to tell
        # it is wrong.
        if any(
            label_similarity(lowered, marker) >= 0.85 for marker in DOCUMENT_MARKERS
        ):
            return None

        return candidate.upper()

    def _extract_country(
        self, lines: list[TextLine], anchors: dict[str, _Anchor], fields: CnicFields
    ) -> None:
        """Read the country-of-stay field."""
        anchor = anchors.get("country_of_stay")
        if anchor is None:
            return
        value_line, source = self._resolve_value(anchor, lines)
        if value_line is None:
            return
        cleaned = self._clean_name(value_line.text)
        if cleaned is not None:
            fields.country_of_stay = FieldValue(
                value=cleaned.title(),
                confidence=value_line.confidence,
                raw=value_line.text,
                source=source,
            )

    @staticmethod
    def _derive_from_number(fields: CnicFields) -> None:
        """Fill the gender field from the CNIC's final digit when unread.

        The parity of the last digit encodes gender, so a missed glyph is
        recoverable. Marked ``derived`` and given a deliberately moderate
        confidence: the inference is sound but it is an inference, and the
        validator can no longer use it as an independent cross-check on itself.
        """
        if fields.gender.present or fields.implied_gender is None:
            return
        fields.gender = FieldValue(
            value=fields.implied_gender,
            confidence=0.6 * fields.cnic_number.confidence,
            raw=None,
            source="derived",
        )


def looks_like_cnic(output: OcrOutput) -> tuple[bool, float]:
    """Decide whether the image is a Pakistani CNIC at all.

    Distinguishes "this is a CNIC we could not read" from "this is a gas bill",
    which need different messages to the user and different fraud weight.

    Args:
        output: The recognised lines.

    Returns:
        ``(is_cnic, confidence)``.
    """
    if not output.lines:
        return False, 0.0

    haystack = " ".join(line.normalised for line in output.lines)
    hits = sum(1 for marker in DOCUMENT_MARKERS if marker in haystack)

    # A valid identity number is by itself strong evidence, since the 5-7-1
    # shape is distinctive.
    has_number = any(parse_cnic_number(line.text) is not None for line in output.lines)

    score = min(1.0, hits / 3.0)
    if has_number:
        score = min(1.0, score + 0.4)

    return score >= 0.4, score


def normalise_numeric_line(text: str) -> str:
    """Apply digit-confusion correction to a line expected to be numeric.

    Exposed for the demo and for tests; the parser applies it internally via
    the pattern helpers.
    """
    return normalise_confusions(text, target="digits")


__all__ = ["CnicParser", "looks_like_cnic", "normalise_numeric_line"]
