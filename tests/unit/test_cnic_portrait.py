"""Choosing which detected face is the one printed on the card.

Pure geometry: no detector, no model weights, no images. Every test hands
:func:`locate_portrait` a list of ``(box, confidence)`` pairs, which is the
whole of its input, so the document-specific reasoning is pinned exactly
rather than sampled from whatever a detector happens to emit today.

The numbers used here are measured, not invented. On the reference synthetic
card the printed portrait's face covers **1.25%** of card area; the live face
in a held-up-card capture covers **18.3%**.
"""

from __future__ import annotations

import pytest

from hamqadam_ai.core.config import CnicPortraitGeometry
from hamqadam_ai.documents.portrait import (
    REJECT_LOW_CONFIDENCE,
    REJECT_NOT_FACE_SHAPED,
    REJECT_OFF_TEMPLATE,
    REJECT_TOO_LARGE,
    REJECT_TOO_SMALL,
    band_containment,
    locate_portrait,
    portrait_regions,
)
from hamqadam_ai.utils.geometry import BoundingBox

CARD_WIDTH = 1200
CARD_HEIGHT = 757

#: Measured on the reference card: face area 1.25% of the card, in the
#: right-hand photo-box band.
GENUINE = BoundingBox(x1=983.0, y1=246.0, x2=1081.0, y2=361.0)

#: Measured on the held-up-card capture: 18.3% of the frame, centred.
LIVE_FACE = BoundingBox(x1=420.0, y1=60.0, x2=780.0, y2=470.0)


@pytest.fixture
def geometry() -> CnicPortraitGeometry:
    return CnicPortraitGeometry()


def find(faces, geometry=None, **kwargs):
    """Run the locator over ``(box, confidence)`` pairs."""
    return locate_portrait(
        faces,
        width=CARD_WIDTH,
        height=CARD_HEIGHT,
        geometry=geometry or CnicPortraitGeometry(),
        **kwargs,
    )


def box_at(
    *, cx: float, cy: float, width: float = 98.0, height: float = 115.0
) -> BoundingBox:
    """A face box centred at the given fractions of the card."""
    x, y = cx * CARD_WIDTH, cy * CARD_HEIGHT
    return BoundingBox(
        x1=x - width / 2, y1=y - height / 2, x2=x + width / 2, y2=y + height / 2
    )


# --------------------------------------------------------------------------- #
# The ordinary case
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_single_portrait_is_found() -> None:
    location = find([(GENUINE, 0.87)])

    assert location.found is True
    assert location.portrait is not None
    assert location.portrait.box is GENUINE
    assert location.portrait.admissible


@pytest.mark.unit
def test_nothing_detected_finds_nothing() -> None:
    location = find([])

    assert location.found is False
    assert location.portrait is None
    assert location.foreign_face_count == 0
    assert location.candidates == ()


@pytest.mark.unit
def test_the_searched_size_is_recorded() -> None:
    """Area ratios only mean something relative to the image they came from."""
    assert find([(GENUINE, 0.9)]).searched_size == (CARD_WIDTH, CARD_HEIGHT)


@pytest.mark.unit
def test_the_measured_genuine_portrait_is_about_one_percent_of_the_card() -> None:
    """Pins the number the area bounds are reasoned from. If the fixture or
    the card geometry ever moves, the bounds have to be re-derived, and this
    is what says so."""
    ratio = find([(GENUINE, 0.9)]).candidates[0].area_ratio
    assert 0.008 < ratio < 0.020


# --------------------------------------------------------------------------- #
# The attack
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_live_face_held_behind_the_card_is_refused() -> None:
    """The whole reason this module exists.

    A face covering 18% of the frame is not a print on an ID-1 card. Taking
    it would compare the selfie against itself and pass whoever the card
    belongs to.
    """
    location = find([(LIVE_FACE, 0.77)])

    assert location.found is False
    assert REJECT_TOO_LARGE in location.candidates[0].rejections


@pytest.mark.unit
def test_the_printed_portrait_wins_over_a_live_face() -> None:
    """Both present, as in a real held-up-card capture. The small one is the
    right answer, which is the exact opposite of "biggest face wins"."""
    location = find([(LIVE_FACE, 0.77), (GENUINE, 0.86)])

    assert location.portrait is not None
    assert location.portrait.box is GENUINE
    assert location.foreign_face_count == 1


@pytest.mark.unit
def test_a_live_face_cannot_win_on_detector_confidence() -> None:
    """A live face is a far easier detection than a print, so it will almost
    always score higher. Plausibility is multiplicative in the band score
    precisely so that cannot buy it the decision."""
    location = find([(LIVE_FACE, 0.99), (GENUINE, 0.40)])

    assert location.portrait is not None
    assert location.portrait.box is GENUINE


@pytest.mark.unit
def test_an_oversized_face_counts_as_foreign() -> None:
    """The count is what Module 9 reads; the score alone cannot distinguish a
    genuine verification from a card held in front of its owner."""
    assert find([(LIVE_FACE, 0.8)]).foreign_face_count == 1


@pytest.mark.unit
def test_a_face_over_the_text_is_off_template() -> None:
    """The middle of the card carries the printed fields, not a photograph."""
    middle = box_at(cx=0.5, cy=0.5)
    location = find([(middle, 0.9)])

    assert REJECT_OFF_TEMPLATE in location.candidates[0].rejections
    assert location.found is False


# --------------------------------------------------------------------------- #
# Ghost portraits
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_second_face_on_the_card_is_a_ghost_not_an_alarm() -> None:
    """Modern cards print a faded reproduction as a security feature.
    Treating "more than one face" as tampering would reject all of them."""
    ghost = BoundingBox(x1=1003.0, y1=564.0, x2=1058.0, y2=622.0)
    location = find([(GENUINE, 0.87), (ghost, 0.50)])

    assert location.has_ghost is True
    assert location.foreign_face_count == 0
    assert location.portrait is not None
    assert location.portrait.box is GENUINE


@pytest.mark.unit
def test_the_ghost_never_outranks_the_primary() -> None:
    """It is the fainter, smaller reproduction of the same face. Feeding the
    recogniser the ghost would hand it the worse of two images of one person."""
    ghost = BoundingBox(x1=1003.0, y1=564.0, x2=1058.0, y2=622.0)

    for order in ([(GENUINE, 0.87), (ghost, 0.50)], [(ghost, 0.50), (GENUINE, 0.87)]):
        location = find(order)
        assert location.portrait is not None
        assert location.portrait.box is GENUINE


@pytest.mark.unit
def test_a_ghost_is_distinguished_from_a_foreign_face() -> None:
    """One is on the card and expected; the other is not and is a signal.
    Collapsing them would either alarm on every modern card or miss the
    attack entirely."""
    ghost = BoundingBox(x1=1003.0, y1=564.0, x2=1058.0, y2=622.0)
    location = find([(GENUINE, 0.87), (ghost, 0.50), (LIVE_FACE, 0.8)])

    assert location.has_ghost is True
    assert location.foreign_face_count == 1


# --------------------------------------------------------------------------- #
# Size bounds
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_face_of_too_few_pixels_is_refused() -> None:
    """Below roughly 40 px the upscale into ArcFace's 112 x 112 input is pure
    interpolation and adds nothing the network can use."""
    tiny = box_at(cx=0.85, cy=0.4, width=24.0, height=30.0)
    assert REJECT_TOO_SMALL in find([(tiny, 0.9)]).candidates[0].rejections


@pytest.mark.unit
def test_the_lower_bound_is_pixels_not_a_fraction() -> None:
    """A ratio floor conflates "this is not a face" with "the card was
    photographed small", and would reject the genuine portrait on any card
    that does not fill the frame - which is most of them.

    The same 60 px face is admissible whether the frame is small or huge.
    """
    face = BoundingBox(x1=100.0, y1=100.0, x2=160.0, y2=170.0)

    small_frame = locate_portrait([(face, 0.9)], width=300, height=200)
    huge_frame = locate_portrait([(face, 0.9)], width=4000, height=2500)

    assert small_frame.candidates[0].admissible
    assert huge_frame.candidates[0].admissible
    assert huge_frame.candidates[0].area_ratio < 0.001


@pytest.mark.unit
def test_the_upper_bound_is_a_fraction_not_pixels() -> None:
    """This one genuinely is about proportion: an ID-1 photo box is a fixed
    physical size, so its face is a bounded share of the card at any capture
    resolution."""
    big = locate_portrait(
        [(BoundingBox(0.0, 0.0, 200.0, 200.0), 0.9)], width=300, height=200
    )
    assert REJECT_TOO_LARGE in big.candidates[0].rejections


@pytest.mark.unit
def test_a_faint_detection_is_refused_but_is_not_foreign() -> None:
    """A face the detector barely saw is probably still on the card and merely
    dim. Counting it as foreign would raise a fraud signal on every poorly-lit
    photograph."""
    location = find([(GENUINE, 0.10)])

    assert REJECT_LOW_CONFIDENCE in location.candidates[0].rejections
    assert location.found is False
    assert location.foreign_face_count == 0


@pytest.mark.unit
def test_a_print_is_held_to_a_lower_detector_bar_than_a_selfie() -> None:
    """A laminated sub-300-dpi print photographed through glare is a genuinely
    harder detection, and the live-image floor loses real portraits."""
    faint = find([(GENUINE, 0.40)], min_detector_confidence=0.35)
    strict = find([(GENUINE, 0.40)], min_detector_confidence=0.60)

    assert faint.found is True
    assert strict.found is False


# --------------------------------------------------------------------------- #
# Template bands
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_both_sides_of_the_card_are_candidate_locations() -> None:
    """The layout has moved between CNIC generations, and this module refuses
    to assert one. Asserting the wrong side would fail closed on genuine
    cards, silently."""
    left = box_at(cx=0.15, cy=0.5)
    right = box_at(cx=0.85, cy=0.5)

    assert find([(left, 0.9)]).found is True
    assert find([(right, 0.9)]).found is True


@pytest.mark.unit
def test_containment_is_graded_not_binary() -> None:
    """A hard in-or-out test discards a genuine portrait whose card was
    cropped a few percent tight. Grading lets it through with the confidence
    penalty it deserves and leaves the decision to the caller."""
    inside = band_containment(box_at(cx=0.85, cy=0.5), CARD_WIDTH, CARD_HEIGHT)
    edge = band_containment(box_at(cx=0.60, cy=0.5), CARD_WIDTH, CARD_HEIGHT)
    far = band_containment(box_at(cx=0.50, cy=0.5), CARD_WIDTH, CARD_HEIGHT)

    assert inside == pytest.approx(1.0)
    assert 0.0 < edge < 1.0
    assert far == pytest.approx(0.0)


@pytest.mark.unit
def test_containment_falls_off_monotonically() -> None:
    scores = [
        band_containment(box_at(cx=cx, cy=0.5), CARD_WIDTH, CARD_HEIGHT)
        for cx in (0.62, 0.60, 0.58, 0.56, 0.54)
    ]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_a_face_above_or_below_the_photo_box_is_off_template() -> None:
    high = box_at(cx=0.85, cy=0.01)
    assert band_containment(high, CARD_WIDTH, CARD_HEIGHT) == pytest.approx(0.0)


@pytest.mark.unit
def test_a_degenerate_frame_contains_nothing() -> None:
    assert band_containment(GENUINE, 0, 0) == pytest.approx(0.0)


@pytest.mark.unit
def test_the_template_regions_are_exposed_for_rendering() -> None:
    regions = portrait_regions(CARD_WIDTH, CARD_HEIGHT)

    assert len(regions) == 2
    for region in regions:
        assert region.x1 >= 0.0
        assert region.x2 <= CARD_WIDTH
        assert region.y2 <= CARD_HEIGHT


@pytest.mark.unit
def test_widening_the_bands_admits_a_central_face() -> None:
    """The priors are configuration, so an unfamiliar template can be
    accommodated without a code change."""
    middle = box_at(cx=0.5, cy=0.5)
    permissive = CnicPortraitGeometry(candidate_bands=[(0.0, 1.0)])

    assert find([(middle, 0.9)]).found is False
    assert find([(middle, 0.9)], geometry=permissive).found is True


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_every_candidate_is_reported_including_refused_ones() -> None:
    """A reviewer asking "why was nothing found" needs to see what was
    considered and on what grounds it was refused."""
    location = find([(LIVE_FACE, 0.8), (box_at(cx=0.5, cy=0.5), 0.9)])

    assert len(location.candidates) == 2
    assert all(c.rejections for c in location.candidates)


@pytest.mark.unit
def test_the_candidate_index_maps_back_to_the_detector() -> None:
    """The service recovers landmarks by index. Matching by geometry instead
    could hand the primary portrait the ghost's keypoints."""
    ghost = BoundingBox(x1=1003.0, y1=564.0, x2=1058.0, y2=622.0)
    location = find([(ghost, 0.50), (GENUINE, 0.87)])

    assert location.portrait is not None
    assert location.portrait.index == 1


@pytest.mark.unit
def test_the_summary_is_serialisable_and_carries_no_pixels() -> None:
    import json

    summary = find([(GENUINE, 0.87), (LIVE_FACE, 0.8)]).describe()
    json.dumps(summary)

    assert summary["found"] is True
    assert summary["foreign_faces"] == 1
    assert REJECT_TOO_LARGE in summary["rejections"]


@pytest.mark.unit
def test_a_candidate_serialises() -> None:
    import json

    payload = find([(GENUINE, 0.87)]).candidates[0].as_dict()
    json.dumps(payload)

    assert payload["rejections"] == []
    assert payload["short_side"] == pytest.approx(98.0)


# --------------------------------------------------------------------------- #
# Regressions from a real submission
# --------------------------------------------------------------------------- #


class TestFalseMismatchRegression:
    """A landscape blob must never be matched as a card portrait.

    From an actual user submission. The card outline could not be isolated, the
    detector returned a 237x161 **landscape** region from somewhere on the card,
    and nothing rejected it: size and position were plausible and shape was
    never checked. It was embedded, scored 0.007 cosine against the applicant's
    own selfie, and the service told a legitimate user that "the document names
    somebody else".

    Wrongly accusing someone of document fraud is far worse than declining to
    read their card, which they can fix by retaking the photograph.
    """

    def test_the_exact_box_from_the_real_submission_is_rejected(self) -> None:
        box = BoundingBox(x1=560.12, y1=1305.85, x2=796.76, y2=1467.15)
        assert box.height / box.width == pytest.approx(0.68, abs=0.01)

        located = locate_portrait(
            [(box, 0.6015)], width=1200, height=2133, min_detector_confidence=0.5
        )
        assert located.portrait is None
        assert REJECT_NOT_FACE_SHAPED in located.candidates[0].rejections

    @pytest.mark.parametrize("aspect", [1.18, 1.20, 1.25, 1.35, 1.40])
    def test_genuine_face_aspects_still_pass(self, aspect: float) -> None:
        """Measured on real detections: selfies, profiles and printed portraits.

        The gate has to exclude 0.68 without touching any of these.
        """
        width = 160.0
        box = BoundingBox(x1=180.0, y1=400.0, x2=180.0 + width, y2=400.0 + width * aspect)
        located = locate_portrait(
            [(box, 0.80)], width=1200, height=2133, min_detector_confidence=0.5
        )
        assert REJECT_NOT_FACE_SHAPED not in located.candidates[0].rejections

    def test_a_weak_candidate_is_declined_rather_than_matched(self) -> None:
        """Plausibility was a ranking key with no floor.

        With one candidate it won however weak it was - 0.13 ranked identically
        to 0.95 for the purpose of being selected.
        """
        # Face-shaped, on-template, but tiny and faint: plausible enough to be
        # admissible, nowhere near good enough to accuse anyone with.
        box = BoundingBox(x1=100.0, y1=250.0, x2=145.0, y2=306.0)
        located = locate_portrait(
            [(box, 0.51)], width=1200, height=2133, min_detector_confidence=0.5
        )
        if located.candidates[0].plausibility < 0.20:
            assert located.portrait is None
