"""Deciding which detected face is the portrait printed on the card.

The attack this exists to stop
------------------------------
A user holds their CNIC up in front of their own face and photographs it. That
is not an exotic attack - it is how people naturally photograph a card they are
holding. A detector run over the result finds **two** faces: the small printed
portrait, and the large live face behind it.

Take the largest face, as almost every naive implementation does, and you take
the live face. The comparison then becomes "is this selfie the same person as
this selfie", which passes at near-perfect similarity **whoever the card
belongs to**. The entire document check is bypassed by holding the card in the
obvious way.

Three layers stop it, in order of strength:

1. **Rectification.** Warping the card to its own detected quadrilateral crops
   away everything that is not the card. A face behind it ceases to exist. This
   is the real defence; the other two cover the case where no card border could
   be found.
2. **An upper area bound.** An ID-1 card is 85.6 x 53.98 mm and its photo box
   is a fixed physical size, so the face inside that box occupies a *bounded
   fraction of the card* however many pixels the photograph has. That makes the
   bound a geometric property of the document rather than a tuned threshold.
   Measured: the reference card's printed portrait sits at 1.2% of card area,
   the live face in the held-up attack at 18.3%, against a 10% ceiling derived
   from the physical proportions.

   The *lower* bound is deliberately **not** a ratio but a pixel count. Whether
   a face can be recognised depends on how many pixels it has, and a ratio
   floor would reject the genuine portrait on any card photographed small -
   which is most of them.
3. **Template bands.** The photo box lies in a known horizontal band. A face
   centred in the middle of the card, over the text, is not the portrait.

Priors rank and validate; they never crop
-----------------------------------------
Nothing here blind-crops the expected region. Template revisions move the photo
box, and a prior that overrode the detector would fail closed on a layout
nobody anticipated - silently, and on genuine cards. The bands are used to
*score* real detections and to *reject* implausible ones, so an unfamiliar
layout degrades to "found it, lower confidence" rather than "found the wrong
thing, full confidence".

Two faces on the card is normal
-------------------------------
Modern NADRA cards print a faded secondary reproduction of the portrait as a
security feature. Treating "more than one face" as tampering would reject every
one of them. A second face *inside* the card is expected and reported as a
ghost; a second face *outside* it is the thing worth alarming about.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hamqadam_ai.core.config import CnicPortraitGeometry
from hamqadam_ai.utils.geometry import BoundingBox

#: Why a candidate was refused. Stable strings - they reach the response and
#: the fraud engine keys on them.
PortraitRejection = str

#: Height/width bounds for a box to be considered a face at all.
#:
#: Measured on genuine detections: 1.18, 1.18, 1.20, 1.20, 1.20, 1.21, 1.35,
#: 1.40. The bounds are deliberately loose either side of that - a portrait
#: crop can be squarer and a tilted one taller - while still excluding the 0.68
#: landscape blob that caused a false CNIC_FACE_MISMATCH on a real submission.
_MIN_FACE_ASPECT = 0.95
_MAX_FACE_ASPECT = 1.80

#: A candidate below this is reported as "no usable portrait" rather than
#: matched. Plausibility used to rank candidates only, with no floor, so the
#: single worst candidate still won by default and was handed to matching as
#: though it were a face.
_MIN_PLAUSIBILITY = 0.20

REJECT_TOO_SMALL: PortraitRejection = "face_too_few_pixels_to_recognise"
REJECT_TOO_LARGE: PortraitRejection = "face_too_large_to_be_on_the_card"
REJECT_OFF_TEMPLATE: PortraitRejection = "face_outside_the_photo_box_region"
REJECT_LOW_CONFIDENCE: PortraitRejection = "detector_confidence_below_floor"
REJECT_NOT_FACE_SHAPED: PortraitRejection = "box_shape_is_not_face_like"


@dataclass(frozen=True, slots=True)
class PortraitCandidate:
    """One detected face, scored against what a printed portrait looks like.

    Attributes:
        box: The face bounds, in the coordinates of the image searched.
        detector_confidence: Objectness from the detector.
        area_ratio: Face area as a fraction of the searched image's area.
        short_side: Length of the face box's shorter edge, in pixels. The
            measure recognisability actually depends on.
        band_score: How well the face centre sits inside a template band,
            1.0 for squarely inside and falling to 0.0 outside the tolerance.
        plausibility: Combined score used to rank candidates.
        rejections: Why this candidate cannot be the portrait. Empty means it
            is admissible.
        index: Position in the detector's output, for cross-referencing.
    """

    box: BoundingBox
    detector_confidence: float
    area_ratio: float
    short_side: float
    band_score: float
    plausibility: float
    rejections: tuple[PortraitRejection, ...] = ()
    index: int = 0

    @property
    def admissible(self) -> bool:
        """Whether this face could be the portrait printed on the card."""
        return not self.rejections

    def as_dict(self) -> dict[str, Any]:
        """Serialisable summary. Carries geometry only, never pixels."""
        return {
            "box": self.box.as_dict(),
            "detector_confidence": round(self.detector_confidence, 4),
            "area_ratio": round(self.area_ratio, 5),
            "short_side": round(self.short_side, 1),
            "band_score": round(self.band_score, 4),
            "plausibility": round(self.plausibility, 4),
            "rejections": list(self.rejections),
        }


@dataclass(frozen=True, slots=True)
class PortraitLocation:
    """The outcome of searching a card image for its printed portrait.

    Attributes:
        portrait: The chosen candidate, or ``None`` when none was admissible.
        ghost: A second admissible face on the card - the security-feature
            reproduction that modern cards carry. Expected, not suspicious.
        off_card: Faces rejected for being too large or off-template. These
            are the fraud-relevant ones: a live face in the frame lands here.
        candidates: Every face considered, admissible or not.
        searched_size: ``(width, height)`` of the image that was searched.
    """

    portrait: PortraitCandidate | None
    ghost: PortraitCandidate | None = None
    off_card: tuple[PortraitCandidate, ...] = ()
    candidates: tuple[PortraitCandidate, ...] = ()
    searched_size: tuple[int, int] = (0, 0)

    @property
    def found(self) -> bool:
        """Whether a usable portrait was located."""
        return self.portrait is not None

    @property
    def has_ghost(self) -> bool:
        """Whether a second face was found on the card itself."""
        return self.ghost is not None

    @property
    def foreign_face_count(self) -> int:
        """Faces present but not plausibly printed on the card.

        The signal for a card held in front of a face, or photographed beside
        someone. Non-zero does not prove fraud - a poster on the wall behind
        would do it - but it is exactly the kind of thing Module 9 aggregates.
        """
        return len(self.off_card)

    def describe(self) -> dict[str, Any]:
        """PII-free summary for logging and for the fraud engine."""
        return {
            "found": self.found,
            "has_ghost": self.has_ghost,
            "foreign_faces": self.foreign_face_count,
            "candidates": len(self.candidates),
            "plausibility": (
                round(self.portrait.plausibility, 4) if self.portrait else None
            ),
            "rejections": sorted(
                {reason for c in self.candidates for reason in c.rejections}
            ),
        }


@dataclass(frozen=True, slots=True)
class _Band:
    """A horizontal strip of the card, as fractions of its width."""

    start: float
    end: float
    vertical: tuple[float, float] = (0.0, 1.0)

    def region(self, width: int, height: int) -> BoundingBox:
        """The band as pixel bounds on an image of the given size."""
        return BoundingBox(
            x1=self.start * width,
            y1=self.vertical[0] * height,
            x2=self.end * width,
            y2=self.vertical[1] * height,
        )


def portrait_regions(
    width: int, height: int, geometry: CnicPortraitGeometry | None = None
) -> list[BoundingBox]:
    """The regions of a card image where the photo box may sit.

    Exposed for the demo's overlay renderer and for tests. Not used to crop:
    see the module docstring on why priors rank rather than decide.

    Args:
        width: Card image width in pixels.
        height: Card image height in pixels.
        geometry: Template priors. Defaults are loaded when omitted.

    Returns:
        One box per candidate band, in the card's pixel coordinates.
    """
    resolved = geometry or CnicPortraitGeometry()
    return [
        _Band(start, end, resolved.vertical_band).region(width, height)
        for start, end in resolved.candidate_bands
    ]


def band_containment(
    box: BoundingBox,
    width: int,
    height: int,
    geometry: CnicPortraitGeometry | None = None,
) -> float:
    """How well a face centre sits inside a template band, in ``[0, 1]``.

    Graded rather than binary. A hard in-or-out test would discard a genuine
    portrait whose card was cropped a few percent tight, which is common; a
    graded score lets such a card through with the confidence penalty it
    deserves and leaves the decision to the caller.

    Args:
        box: The face bounds.
        width: Card image width.
        height: Card image height.
        geometry: Template priors.

    Returns:
        1.0 when the centre is inside a band, falling linearly to 0.0 at
        ``band_tolerance`` beyond its edge.
    """
    resolved = geometry or CnicPortraitGeometry()
    if width <= 0 or height <= 0:
        return 0.0

    centre_x, centre_y = box.center
    fraction_x = centre_x / width
    fraction_y = centre_y / height
    tolerance = max(resolved.band_tolerance, 1e-6)

    vertical_top, vertical_bottom = resolved.vertical_band
    vertical = _axis_containment(fraction_y, vertical_top, vertical_bottom, tolerance)
    if vertical <= 0.0:
        return 0.0

    horizontal = max(
        (
            _axis_containment(fraction_x, start, end, tolerance)
            for start, end in resolved.candidate_bands
        ),
        default=0.0,
    )
    return horizontal * vertical


def _axis_containment(
    value: float, start: float, end: float, tolerance: float
) -> float:
    """Graded membership of ``value`` in ``[start, end]`` along one axis."""
    if start <= value <= end:
        return 1.0
    distance = start - value if value < start else value - end
    if distance >= tolerance:
        return 0.0
    return 1.0 - distance / tolerance


def locate_portrait(
    faces: Sequence[tuple[BoundingBox, float]],
    *,
    width: int,
    height: int,
    geometry: CnicPortraitGeometry | None = None,
    min_detector_confidence: float = 0.35,
) -> PortraitLocation:
    """Choose which detected face is the portrait printed on the card.

    Args:
        faces: ``(box, detector_confidence)`` for every face found, in the
            coordinates of the image searched. A plain pair rather than the
            detector's own model: this package has no business importing
            Module 1's schema, and the explicit tuple makes the unit tests
            free of fakes.
        width: Width of the image the faces were detected in.
        height: Its height.
        geometry: Template priors.
        min_detector_confidence: Objectness floor. Lower than the live-image
            floor, because a laminated sub-300-dpi print photographed through
            glare is a genuinely harder detection than a selfie.

    Returns:
        The chosen portrait, any ghost, and every face judged not to be on the
        card - which is the part the fraud engine reads.
    """
    resolved = geometry or CnicPortraitGeometry()
    frame_area = float(max(width * height, 1))

    candidates: list[PortraitCandidate] = []
    for index, (box, raw_confidence) in enumerate(faces):
        confidence = float(raw_confidence)
        area_ratio = box.area / frame_area
        band = band_containment(box, width, height, resolved)

        rejections: list[PortraitRejection] = []
        if confidence < min_detector_confidence:
            rejections.append(REJECT_LOW_CONFIDENCE)
        if box.short_side < resolved.min_face_pixels:
            rejections.append(REJECT_TOO_SMALL)
        if area_ratio > resolved.max_face_area_ratio:
            rejections.append(REJECT_TOO_LARGE)
        if band <= 0.0:
            rejections.append(REJECT_OFF_TEMPLATE)

        # Shape. A face box is taller than it is wide; measured across genuine
        # SCRFD detections (selfies, profile photos and printed card portraits)
        # the height/width ratio sits in a tight 1.18-1.40 band.
        #
        # This gate exists because of a real false accusation. On a photograph
        # where the card outline could not be found, the detector returned a
        # 237x161 *landscape* region - aspect 0.68, plausibility 0.13 - from
        # somewhere on the card. Nothing rejected it, because size and position
        # were plausible and shape was never checked. It was embedded, scored
        # 0.007 cosine against the applicant's selfie, and the service told a
        # legitimate user that "the document names somebody else".
        #
        # A wrong accusation of document fraud is far worse than declining to
        # read the card, so shape is now a hard gate.
        aspect = box.height / max(box.width, 1e-6)
        if not (_MIN_FACE_ASPECT <= aspect <= _MAX_FACE_ASPECT):
            rejections.append(REJECT_NOT_FACE_SHAPED)

        candidates.append(
            PortraitCandidate(
                box=box,
                detector_confidence=confidence,
                area_ratio=area_ratio,
                short_side=box.short_side,
                band_score=band,
                plausibility=_plausibility(confidence, area_ratio, band, resolved),
                rejections=tuple(rejections),
                index=index,
            )
        )

    admissible = sorted(
        (c for c in candidates if c.admissible),
        key=lambda c: c.plausibility,
        reverse=True,
    )
    # Plausibility was a ranking key with no floor, so when only one candidate
    # survived it won however weak it was - a 0.13 scored the same as a 0.95 for
    # the purpose of being selected. Declining to read the card is a recoverable
    # outcome the applicant can fix by retaking the photograph; asserting a
    # portrait mismatch against them is not.
    admissible = [c for c in admissible if c.plausibility >= _MIN_PLAUSIBILITY]
    # Only the size and template rejections mean "not on the card". A face
    # refused for low detector confidence is probably still on the card and
    # merely faint, and calling that a foreign face would raise a fraud signal
    # on every dim photograph.
    off_card = tuple(
        c
        for c in candidates
        if REJECT_TOO_LARGE in c.rejections or REJECT_OFF_TEMPLATE in c.rejections
    )

    return PortraitLocation(
        portrait=admissible[0] if admissible else None,
        ghost=admissible[1] if len(admissible) > 1 else None,
        off_card=off_card,
        candidates=tuple(candidates),
        searched_size=(width, height),
    )


def _plausibility(
    confidence: float,
    area_ratio: float,
    band_score: float,
    geometry: CnicPortraitGeometry,
) -> float:
    """Rank admissible candidates: how portrait-like is this face?

    Multiplicative in the band score, so a face outside the photo box cannot
    win on detector confidence alone. Within the band, larger and more
    confident wins - which is what separates the real portrait from the faded
    ghost printed beside it.

    Size contributes as a *fraction of the largest plausible portrait* rather
    than as a raw pixel count, so the term stays comparable across card
    resolutions. It is deliberately the lighter of the two weights: the
    difference between a portrait and its ghost is mostly contrast, which the
    detector's own confidence reflects better than area does.

    Worked example, the reference card: the printed portrait scores
    ``1.0 * (0.55 * 0.87 + 0.45 * 0.125) = 0.534`` and its ghost
    ``1.0 * (0.55 * 0.50 + 0.45 * 0.035) = 0.291``.
    """
    size_term = min(max(area_ratio / max(geometry.max_face_area_ratio, 1e-9), 0.0), 1.0)
    return band_score * (0.55 * confidence + 0.45 * size_term)


__all__ = [
    "REJECT_LOW_CONFIDENCE",
    "REJECT_NOT_FACE_SHAPED",
    "REJECT_OFF_TEMPLATE",
    "REJECT_TOO_LARGE",
    "REJECT_TOO_SMALL",
    "PortraitCandidate",
    "PortraitLocation",
    "PortraitRejection",
    "band_containment",
    "locate_portrait",
    "portrait_regions",
]
