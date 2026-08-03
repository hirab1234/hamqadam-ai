"""Getting a photographed card into a state an OCR can read.

Three problems, in the order they have to be solved:

**Where is the card?** A user photographs their CNIC on a desk, at an angle,
with a quarter of the frame being the desk. Detecting the card's quadrilateral
and perspective-correcting it to the ID-1 aspect ratio removes the background,
makes the text horizontal, and normalises scale - which in turn makes every
downstream threshold meaningful across wildly different captures.

**Which way up is it?** Established below, because the answer is not what one
would guess.

**Can the text be seen at all?** A dim phone photograph of a laminated card
has glare on one half and shadow on the other. Adaptive local contrast fixes
that in a way global adjustment cannot.

A note on orientation
---------------------
Modern OCR engines classify and correct the angle of each detected *text line*
independently. Measured on the probe that motivated this module, PP-OCR read a
card rotated 90 degrees and one rotated 180 degrees with the same eighteen
lines and the same text as the upright original.

So recognition does not need the page straightened. **Spatial reasoning does.**
The field parser locates a value by looking to the right of its printed label,
and on a rotated page "to the right" is a different direction. Orientation is
therefore resolved from the geometry the engine returns rather than by
re-running recognition four times - which at four seconds a pass would cost
seventeen seconds per document.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.constants import EPSILON
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.ocr.base import TextLine

log = get_logger(__name__)

BgrImage = npt.NDArray[np.uint8]

#: ISO/IEC 7810 ID-1, the format every national identity card uses:
#: 85.60 x 53.98 mm.
ID1_ASPECT_RATIO = 85.60 / 53.98

#: Width the rectified card is normalised to. At this width the printed text is
#: roughly 24 px tall, comfortably above the ~10 px floor below which PP-OCR's
#: recognition accuracy falls away.
RECTIFIED_WIDTH = 1012


class Orientation(IntEnum):
    """Clockwise page rotation needed to bring the card upright."""

    UPRIGHT = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = 270


@dataclass(frozen=True, slots=True)
class RectifyResult:
    """The outcome of trying to isolate the card from its background.

    Attributes:
        image: The rectified card, or the input unchanged when no card was
            found.
        rectified: Whether a card quadrilateral was located and warped.
        quad: The detected corners in source coordinates.
        coverage: Fraction of the source frame the card occupied. Very small
            values mean the user photographed the card from far away and the
            text may be too small to read.
        reason: Why rectification was skipped, when it was.
    """

    image: BgrImage
    rectified: bool
    quad: npt.NDArray[np.float32] | None = None
    coverage: float = 1.0
    reason: str | None = None


def _order_corners(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Order four corners clockwise from the top-left.

    Uses coordinate sums and differences rather than angles: the top-left has
    the smallest ``x + y`` and the top-right the smallest ``y - x``. That is
    robust to the arbitrary ordering ``approxPolyDP`` produces and does not
    care which way round the contour was traced.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]

    difference = np.diff(points, axis=1).reshape(-1)
    ordered[1] = points[np.argmin(difference)]
    ordered[3] = points[np.argmax(difference)]
    return ordered


def rectify_document(
    image: BgrImage,
    *,
    target_width: int = RECTIFIED_WIDTH,
    min_coverage: float = 0.12,
    aspect_tolerance: float = 0.35,
) -> RectifyResult:
    """Locate the card in a photograph and perspective-correct it.

    Finds the largest four-sided contour whose aspect ratio is plausibly ID-1
    and warps it to a canonical frame. When no such contour exists - the card
    fills the frame already, or the edges are lost against a similar-coloured
    surface - the input is returned unchanged rather than a bad crop being
    forced. A wrong crop is far worse than no crop: it removes fields entirely.

    Args:
        image: The source photograph, BGR uint8.
        target_width: Width of the rectified output.
        min_coverage: Smallest share of the frame a candidate may occupy. Below
            this it is more likely a card-shaped object in the background.
        aspect_tolerance: Permitted relative deviation from the ID-1 ratio.

    Returns:
        The rectification outcome.
    """
    height, width = image.shape[:2]
    frame_area = float(height * width)
    if frame_area <= 0:
        return RectifyResult(image=image, rectified=False, reason="empty image")

    # Work at a reduced size: contour finding gains nothing from 12 megapixels
    # and costs proportionally.
    scale = 900.0 / max(height, width)
    working = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )

    grey = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    # Bilateral rather than Gaussian: it suppresses the laminate's texture
    # while keeping the card's outer edge sharp, which is the edge we want.
    smoothed = cv2.bilateralFilter(grey, 9, 75, 75)
    edges = cv2.Canny(smoothed, 40, 130)
    # Close small gaps where the card edge passes over a similar-toned region.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return RectifyResult(
            image=image, rectified=False, reason="no contours found"
        )

    working_area = float(working.shape[0] * working.shape[1])
    best: tuple[float, npt.NDArray[np.float32]] | None = None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour)
        coverage = area / working_area
        if coverage < min_coverage:
            break  # sorted by area, so everything after is smaller still

        perimeter = cv2.arcLength(contour, closed=True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, closed=True)
        if len(approximation) != 4 or not cv2.isContourConvex(approximation):
            continue

        corners = _order_corners(approximation.reshape(4, 2).astype(np.float32))
        top = float(np.linalg.norm(corners[1] - corners[0]))
        bottom = float(np.linalg.norm(corners[2] - corners[3]))
        left = float(np.linalg.norm(corners[3] - corners[0]))
        right = float(np.linalg.norm(corners[2] - corners[1]))

        quad_width = max(top, bottom)
        quad_height = max(left, right)
        if quad_height < EPSILON:
            continue

        aspect = quad_width / quad_height
        # Accept either orientation: a card photographed sideways is still a
        # card, and the orientation step deals with which way up it is.
        deviation = min(
            abs(aspect - ID1_ASPECT_RATIO) / ID1_ASPECT_RATIO,
            abs(aspect - 1.0 / ID1_ASPECT_RATIO) * ID1_ASPECT_RATIO,
        )
        if deviation > aspect_tolerance:
            continue

        if best is None or coverage > best[0]:
            best = (coverage, corners)

    if best is None:
        return RectifyResult(
            image=image,
            rectified=False,
            reason="no card-shaped quadrilateral found; using the whole frame",
        )

    coverage, corners = best
    if scale < 1.0:
        corners = corners / scale

    quad_width = max(
        float(np.linalg.norm(corners[1] - corners[0])),
        float(np.linalg.norm(corners[2] - corners[3])),
    )
    quad_height = max(
        float(np.linalg.norm(corners[3] - corners[0])),
        float(np.linalg.norm(corners[2] - corners[1])),
    )
    landscape = quad_width >= quad_height

    output_width = target_width
    output_height = int(round(target_width / ID1_ASPECT_RATIO))
    if not landscape:
        output_width, output_height = output_height, output_width

    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(image, transform, (output_width, output_height))

    return RectifyResult(
        image=np.asarray(warped, dtype=np.uint8),
        rectified=True,
        quad=corners,
        coverage=coverage,
    )


def detect_orientation(lines: list[TextLine]) -> tuple[Orientation, float]:
    """Infer page rotation from the geometry of recognised text.

    Reads the shape of the detected regions rather than their content. A line
    of Latin script is wider than it is tall; when most detected regions are
    taller than wide, the page is a quarter turn out.

    Distinguishing 90 from 270, and 0 from 180, cannot be done from shape
    alone - both members of each pair produce identical box geometry. The
    caller resolves that by content, which is what
    :func:`orientation_candidates` orders for it.

    Args:
        lines: Recognised lines from a single pass over the unrotated image.

    Returns:
        ``(orientation, confidence)`` where the orientation is either
        ``UPRIGHT`` or ``ROTATE_90``, and confidence is the share of lines
        agreeing.
    """
    if not lines:
        return Orientation.UPRIGHT, 0.0

    # Ignore near-square regions: a single glyph carries no orientation signal
    # and would only add noise.
    informative = [line for line in lines if abs(line.aspect - 1.0) > 0.25]
    if not informative:
        return Orientation.UPRIGHT, 0.0

    portrait = sum(1 for line in informative if line.aspect < 1.0)
    fraction = portrait / len(informative)

    if fraction > 0.6:
        return Orientation.ROTATE_90, fraction
    return Orientation.UPRIGHT, 1.0 - fraction


def orientation_candidates(lines: list[TextLine]) -> list[Orientation]:
    """Order the four rotations by how likely each is, cheapest first.

    Recognition costs seconds per pass, so the caller tries these in order and
    stops as soon as the content confirms one - typically on the first.

    Args:
        lines: Recognised lines from an initial pass over the unrotated image.

    Returns:
        Every orientation, most promising first.
    """
    guess, _ = detect_orientation(lines)
    if guess is Orientation.ROTATE_90:
        # Geometry says quarter-turned but cannot say which way; 270 is the
        # other quarter turn, and the half turn is the remaining possibility.
        return [
            Orientation.ROTATE_90,
            Orientation.ROTATE_270,
            Orientation.UPRIGHT,
            Orientation.ROTATE_180,
        ]
    return [
        Orientation.UPRIGHT,
        Orientation.ROTATE_180,
        Orientation.ROTATE_90,
        Orientation.ROTATE_270,
    ]


def rotate_image(image: BgrImage, orientation: Orientation) -> BgrImage:
    """Rotate an image clockwise by a multiple of ninety degrees."""
    if orientation is Orientation.UPRIGHT:
        return image
    if orientation is Orientation.ROTATE_90:
        return np.asarray(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), dtype=np.uint8)
    if orientation is Orientation.ROTATE_180:
        return np.asarray(cv2.rotate(image, cv2.ROTATE_180), dtype=np.uint8)
    return np.asarray(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), dtype=np.uint8)


def deskew(image: BgrImage, *, max_angle: float = 12.0) -> tuple[BgrImage, float]:
    """Correct a small residual tilt after rectification.

    Estimates the dominant text-line angle with a Hough transform over the
    horizontal edge structure and rotates it out. Bounded to ``max_angle``
    because anything larger is a quarter-turn problem, not a tilt, and forcing
    a large rotation here would fight the orientation step.

    Args:
        image: The rectified card.
        max_angle: Largest correction applied, in degrees.

    Returns:
        ``(image, angle_applied)``.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey, 50, 150)
    # Typed as non-optional by the cv2 stubs, but it genuinely returns None
    # when no line survives the accumulator threshold - which is the normal
    # result for a blank frame.
    lines: Any = cv2.HoughLinesP(
        edges, 1, np.pi / 360, threshold=100,
        minLineLength=image.shape[1] // 4, maxLineGap=20,
    )
    if lines is None:
        return image, 0.0

    angles: list[float] = []
    for line in lines[:200]:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        # Only near-horizontal structure: vertical edges are the card border.
        if abs(angle) <= max_angle:
            angles.append(angle)

    if not angles:
        return image, 0.0

    correction = float(np.median(angles))
    if abs(correction) < 0.25:
        # Below a quarter degree the resampling costs more sharpness than the
        # straightening gains.
        return image, 0.0

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), correction, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return np.asarray(rotated, dtype=np.uint8), correction


def enhance_for_ocr(
    image: BgrImage,
    *,
    clip_limit: float = 2.5,
    denoise: bool = True,
) -> BgrImage:
    """Improve legibility of a poorly-captured card.

    Applies CLAHE to the lightness channel only. Local rather than global
    equalisation because the characteristic failure of a photographed laminated
    card is *uneven* illumination - glare across one corner, shadow in the
    other - which a global curve cannot fix without destroying the good half.

    Operating on L in CIELAB rather than on RGB keeps the hue intact, which
    matters because the parser has no colour dependency but a human reviewing
    a rejected capture does.

    Args:
        image: The rectified card.
        clip_limit: CLAHE contrast ceiling. Above ~4 it starts amplifying
            JPEG blocking into something the detector reads as text.
        denoise: Apply a light bilateral filter first.

    Returns:
        The enhanced image.
    """
    working = image
    if denoise:
        # Small sigma: enough to take the grain off a high-ISO capture without
        # softening the strokes of 20 px text.
        working = np.asarray(
            cv2.bilateralFilter(working, 5, 40, 40), dtype=np.uint8
        )

    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    equalised = clahe.apply(lightness)

    merged = cv2.merge([equalised, a_channel, b_channel])
    return np.asarray(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR), dtype=np.uint8)


def upscale_if_small(
    image: BgrImage, *, min_width: int = 700
) -> tuple[BgrImage, float]:
    """Enlarge a card that is too small for the recogniser to read.

    Adds no information, but PP-OCR's recognition head has a fixed 48-pixel
    input height and text below roughly 10 px in the source degrades sharply
    once resampled into it. Recovering a readable field from an upscale beats
    reporting the document unreadable; the quality module independently
    penalises the low native resolution.

    Returns:
        ``(image, scale)``.
    """
    width = image.shape[1]
    if width >= min_width:
        return image, 1.0
    scale = min_width / float(width)
    enlarged = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    return np.asarray(enlarged, dtype=np.uint8), scale


def describe_preprocessing(
    rectify: RectifyResult, *, deskew_angle: float, upscale: float, enhanced: bool
) -> dict[str, Any]:
    """Serialisable summary of what was done to the image."""
    return {
        "rectified": rectify.rectified,
        "coverage": round(rectify.coverage, 4),
        "rectify_reason": rectify.reason,
        "deskew_degrees": round(deskew_angle, 3),
        "upscale": round(upscale, 3),
        "enhanced": enhanced,
    }


__all__ = [
    "ID1_ASPECT_RATIO",
    "RECTIFIED_WIDTH",
    "Orientation",
    "RectifyResult",
    "describe_preprocessing",
    "deskew",
    "detect_orientation",
    "enhance_for_ocr",
    "orientation_candidates",
    "rectify_document",
    "rotate_image",
    "upscale_if_small",
]
