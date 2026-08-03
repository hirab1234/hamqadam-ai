"""The OCR engine port and its value objects.

Contract every adapter honours
------------------------------
* Input is a BGR ``uint8`` image; output is a list of
  :class:`TextLine` in **source-image pixel coordinates**.
* Every line carries a confidence in ``[0, 1]``. An engine that does not
  natively produce one must map its own score into that range and say so.
* Lines are returned in reading order - top to bottom, then left to right -
  which the field parser depends on for its "value follows label" fallback.
* An engine never raises for "no text here". That is an empty list, and a
  perfectly normal result for a photograph of a thumb over the lens.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.constants import EPSILON

BgrImage = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class TextLine:
    """One recognised line of text and where it sits on the page.

    Attributes:
        text: The recognised string, as the engine produced it.
        confidence: Recognition confidence in ``[0, 1]``.
        quad: The four corners of the text region in source pixels, ordered
            clockwise from the top-left. A quadrilateral rather than a
            rectangle because a photographed card is rarely axis-aligned.
    """

    text: str
    confidence: float
    quad: tuple[tuple[float, float], ...]

    @property
    def x1(self) -> float:
        """Left edge of the axis-aligned bounding box."""
        return min(point[0] for point in self.quad)

    @property
    def y1(self) -> float:
        """Top edge of the axis-aligned bounding box."""
        return min(point[1] for point in self.quad)

    @property
    def x2(self) -> float:
        """Right edge of the axis-aligned bounding box."""
        return max(point[0] for point in self.quad)

    @property
    def y2(self) -> float:
        """Bottom edge of the axis-aligned bounding box."""
        return max(point[1] for point in self.quad)

    @property
    def width(self) -> float:
        """Bounding-box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Bounding-box height."""
        return self.y2 - self.y1

    @property
    def centre_y(self) -> float:
        """Vertical centre, used to group lines into rows."""
        return (self.y1 + self.y2) / 2.0

    @property
    def aspect(self) -> float:
        """Width over height.

        Below 1.0 means the text region is taller than it is wide, which for a
        line of Latin script means the page is rotated a quarter turn. The
        distribution of this across all lines is what
        :func:`~hamqadam_ai.ocr.preprocessing.detect_orientation` keys on.
        """
        return self.width / max(self.height, EPSILON)

    @property
    def normalised(self) -> str:
        """Lowercased, whitespace-collapsed text, for label matching."""
        return " ".join(self.text.lower().split())

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form. Used only in diagnostics, never in the response.

        The recognised text of a CNIC is the holder's name and identity
        number, so it never reaches a log line - see the redaction rules in
        :mod:`hamqadam_ai.logging.processors`.
        """
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "box": [round(self.x1, 1), round(self.y1, 1),
                    round(self.x2, 1), round(self.y2, 1)],
        }


@dataclass(slots=True)
class OcrOutput:
    """Everything one OCR pass produced.

    Attributes:
        lines: Recognised lines in reading order.
        engine: Which adapter produced them.
        engine_version: Version of the underlying models.
        duration_ms: How long the pass took.
        rotation_applied: Page rotation applied before recognition, in
            degrees clockwise.
        detail: Engine-specific diagnostics.
    """

    lines: list[TextLine] = field(default_factory=list)
    engine: str = ""
    engine_version: str = ""
    duration_ms: float = 0.0
    rotation_applied: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def line_count(self) -> int:
        """How many lines were recognised."""
        return len(self.lines)

    @property
    def mean_confidence(self) -> float:
        """Mean recognition confidence across all lines, or zero if none."""
        if not self.lines:
            return 0.0
        return float(np.mean([line.confidence for line in self.lines]))

    @property
    def text(self) -> str:
        """Every line joined by newlines, in reading order."""
        return "\n".join(line.text for line in self.lines)

    def in_reading_order(self, *, row_tolerance: float = 0.6) -> list[TextLine]:
        """Sort lines top-to-bottom, then left-to-right within a row.

        A naive sort by ``y`` alone scatters a label and its value onto
        different rows whenever the card is a degree or two off level, which
        breaks the parser's row grouping. Lines whose vertical centres fall
        within ``row_tolerance`` of a shared line height are treated as one row.

        Args:
            row_tolerance: Row band height as a multiple of median line height.

        Returns:
            A new list in reading order.
        """
        if not self.lines:
            return []

        heights = [line.height for line in self.lines if line.height > 0]
        band = float(np.median(heights)) * row_tolerance if heights else 1.0

        ordered = sorted(self.lines, key=lambda line: line.centre_y)
        rows: list[list[TextLine]] = []
        for line in ordered:
            if rows and abs(line.centre_y - rows[-1][0].centre_y) <= band:
                rows[-1].append(line)
            else:
                rows.append([line])

        result: list[TextLine] = []
        for row in rows:
            result.extend(sorted(row, key=lambda line: line.x1))
        return result

    def describe(self) -> dict[str, Any]:
        """PII-free summary for logging.

        Deliberately omits the recognised text: on a CNIC that is the holder's
        name, father's name and identity number.
        """
        return {
            "engine": self.engine,
            "lines": self.line_count,
            "mean_confidence": round(self.mean_confidence, 4),
            "rotation": self.rotation_applied,
            "duration_ms": round(self.duration_ms, 1),
        }


class OcrEngine(abc.ABC):
    """Abstract text recogniser.

    Args:
        name: Short adapter identifier.
        version: Version of the underlying models.
    """

    def __init__(self, *, name: str, version: str) -> None:
        self.name = name
        self.version = version

    @abc.abstractmethod
    def recognise(self, image: BgrImage) -> list[TextLine]:
        """Detect and recognise every text line in an image.

        Args:
            image: ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Recognised lines in source-image coordinates. Empty when the image
            contains no legible text, which is a normal result rather than an
            error.
        """

    def recognise_crop(self, image: BgrImage) -> list[TextLine]:
        """Recognise a crop known to contain one line, skipping detection.

        Exists because text *detection* is the weak link for short values.
        Measured on a synthetic CNIC, PP-OCR's detector returned nothing at
        all for the gender field - a lone ``F`` in a wide margin - at every
        padding and every scale tried, while the recogniser handed the same
        crop read it correctly. Detectors are trained on lines of text and a
        single glyph does not look like one.

        So when a label has been located and its value is missing, the
        caller crops the region beside the label and calls this, which puts
        the crop straight into the recognition head.

        Confidence from this path is **not comparable** with confidence from
        :meth:`recognise`: an isolated glyph scores far lower than the same
        glyph in a line, so it needs its own, lower acceptance floor.

        Args:
            image: A tight crop containing at most one line of text.

        Returns:
            Recognised lines. The default implementation falls back to full
            detection, which is correct but will usually find nothing for
            the very cases this method exists for.
        """
        return self.recognise(image)

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources held by the adapter."""

    def read(self, image: BgrImage) -> OcrOutput:
        """Recognise and wrap the result with timing and provenance."""
        import time

        started = time.perf_counter()
        lines = self.recognise(image)
        return OcrOutput(
            lines=lines,
            engine=self.name,
            engine_version=self.version,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def read_async(self, image: BgrImage) -> OcrOutput:
        """Recognise without blocking the event loop."""
        return await asyncio.to_thread(self.read, image)

    def __enter__(self) -> OcrEngine:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, version={self.version!r})"


def quad_from_box(x1: float, y1: float, x2: float, y2: float) -> tuple[tuple[float, float], ...]:
    """Build a clockwise quadrilateral from an axis-aligned box.

    For adapters whose engine returns rectangles rather than quads.
    """
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def rotate_quad(
    quad: Sequence[tuple[float, float]],
    *,
    rotation: int,
    width: int,
    height: int,
) -> tuple[tuple[float, float], ...]:
    """Map a quadrilateral from a rotated frame back to the source frame.

    When the page is rotated before recognition, every returned coordinate is
    in the rotated frame. Mapping them back is what lets the caller draw a box
    on the original photograph and lets the parser reason spatially about the
    card as the user actually holds it.

    Args:
        quad: Corners in the rotated frame.
        rotation: Clockwise rotation that was applied, in degrees: 0, 90, 180
            or 270.
        width: Width of the **rotated** image.
        height: Height of the **rotated** image.

    Returns:
        Corners in the original frame.
    """
    normalised = rotation % 360
    if normalised == 0:
        return tuple((float(x), float(y)) for x, y in quad)
    if normalised == 90:
        # The source was rotated 90 clockwise to produce this frame, so undo it.
        return tuple((float(y), float(width - x)) for x, y in quad)
    if normalised == 180:
        return tuple((float(width - x), float(height - y)) for x, y in quad)
    if normalised == 270:
        return tuple((float(height - y), float(x)) for x, y in quad)
    raise ValueError(f"rotation must be a multiple of 90 degrees, got {rotation}")


__all__ = [
    "BgrImage",
    "OcrEngine",
    "OcrOutput",
    "TextLine",
    "quad_from_box",
    "rotate_quad",
]
