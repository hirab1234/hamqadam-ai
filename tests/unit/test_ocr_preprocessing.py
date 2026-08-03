"""Geometry and contrast correction, measured on rendered pixels.

These run against synthetic cards rather than mocks: the whole point of
rectification and deskew is what they do to an image, and an assertion that
does not look at pixels cannot check that.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from hamqadam_ai.ocr.base import TextLine, quad_from_box
from hamqadam_ai.ocr.preprocessing import (
    ID1_ASPECT_RATIO,
    RECTIFIED_WIDTH,
    Orientation,
    describe_preprocessing,
    deskew,
    detect_orientation,
    enhance_for_ocr,
    orientation_candidates,
    rectify_document,
    rotate_image,
    upscale_if_small,
)
from tests.fixtures.synthetic_cnic import (
    dim_lighting,
    photograph_on_desk,
    render_cnic,
    rotated,
)


@pytest.fixture(scope="module")
def clean_card() -> np.ndarray:
    """A flat, well-lit synthetic card."""
    return render_cnic()


def landscape_line(index: int) -> TextLine:
    """A wide-and-short region, the shape of a line of upright Latin text."""
    top = 50.0 + index * 40.0
    return TextLine("x", 0.9, quad_from_box(20.0, top, 320.0, top + 24.0))


def portrait_line(index: int) -> TextLine:
    """The same region a quarter turn out."""
    left = 50.0 + index * 40.0
    return TextLine("x", 0.9, quad_from_box(left, 20.0, left + 24.0, 320.0))


# --------------------------------------------------------------------------- #
# Rectification
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_card_on_a_desk_is_isolated_and_flattened(clean_card: np.ndarray) -> None:
    photo = photograph_on_desk(clean_card)
    result = rectify_document(photo)

    assert result.rectified is True
    assert result.quad is not None
    assert result.image.shape[1] == RECTIFIED_WIDTH


@pytest.mark.unit
def test_rectification_restores_the_id1_aspect_ratio(clean_card: np.ndarray) -> None:
    """An ID-1 card is a fixed physical size, so the corrected image must have
    a fixed shape whatever angle it was photographed from."""
    result = rectify_document(photograph_on_desk(clean_card))
    height, width = result.image.shape[:2]

    assert width / height == pytest.approx(ID1_ASPECT_RATIO, rel=0.02)


@pytest.mark.unit
def test_coverage_reports_how_much_of_the_frame_the_card_filled(
    clean_card: np.ndarray,
) -> None:
    """A card photographed from far away yields text too small to read, and
    the caller needs to be able to say so."""
    close = rectify_document(photograph_on_desk(clean_card, scale=0.85))
    distant = rectify_document(photograph_on_desk(clean_card, scale=0.45))

    assert close.rectified and distant.rectified
    assert close.coverage > distant.coverage


@pytest.mark.unit
def test_coverage_is_one_when_no_card_was_found(clean_card: np.ndarray) -> None:
    """The whole frame is being used as the card, so it covers all of itself.

    Worth pinning because the value collides numerically with "the card fills
    the frame": callers must read ``rectified`` before trusting ``coverage``,
    and this is the case that makes that necessary.
    """
    result = rectify_document(clean_card)

    assert result.rectified is False
    assert result.coverage == pytest.approx(1.0)


@pytest.mark.unit
def test_an_already_flat_scan_is_passed_through(clean_card: np.ndarray) -> None:
    """No border to find, and warping a full-frame card would only resample it
    for nothing."""
    result = rectify_document(clean_card)

    assert result.rectified is False
    assert result.reason is not None
    assert np.array_equal(result.image, clean_card)


@pytest.mark.unit
def test_a_frame_with_no_card_is_returned_unchanged() -> None:
    """A photograph of a wall: no border, no quadrilateral, nothing to warp.

    Square rather than card-shaped on purpose. A frame that is already close
    to ID-1 proportions can be "rectified" into itself, which would let this
    test pass without the rejection path ever running.
    """
    rng = np.random.default_rng(0)
    wall = np.clip(
        180 + rng.normal(0.0, 4.0, (700, 700, 3)), 0, 255
    ).astype(np.uint8)

    result = rectify_document(wall)

    assert result.rectified is False
    assert np.array_equal(result.image, wall)


@pytest.mark.unit
def test_a_wrongly_shaped_quadrilateral_is_refused() -> None:
    """A tall dark rectangle on a light desk is a strong contour but the wrong
    shape for an ID-1 card - a phone, or a book. Warping it into card
    proportions would distort whatever text it carries."""
    frame = np.full((700, 700, 3), 235, dtype=np.uint8)
    cv2.rectangle(frame, (240, 60), (420, 640), (30, 30, 30), thickness=-1)

    result = rectify_document(frame)

    assert result.rectified is False
    assert result.reason is not None


@pytest.mark.unit
def test_the_rectified_card_stays_readable(clean_card: np.ndarray) -> None:
    """A cheap sanity check that the warp did not scramble the content:
    edge density in the text band should survive it."""
    photo = photograph_on_desk(clean_card)
    result = rectify_document(photo)

    edges = cv2.Canny(cv2.cvtColor(result.image, cv2.COLOR_BGR2GRAY), 60, 160)
    assert float(edges.mean()) > 1.0


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_wide_regions_read_as_upright() -> None:
    orientation, confidence = detect_orientation(
        [landscape_line(index) for index in range(6)]
    )
    assert orientation is Orientation.UPRIGHT
    assert confidence == pytest.approx(1.0)


@pytest.mark.unit
def test_tall_regions_read_as_quarter_turned() -> None:
    orientation, confidence = detect_orientation(
        [portrait_line(index) for index in range(6)]
    )
    assert orientation is Orientation.ROTATE_90
    assert confidence == pytest.approx(1.0)


@pytest.mark.unit
def test_a_mixed_page_favours_the_majority() -> None:
    lines = [portrait_line(index) for index in range(5)]
    lines += [landscape_line(index) for index in range(2)]

    orientation, confidence = detect_orientation(lines)

    assert orientation is Orientation.ROTATE_90
    assert 0.6 < confidence < 1.0


@pytest.mark.unit
def test_near_square_regions_carry_no_signal() -> None:
    """A lone glyph is as tall as it is wide whichever way up the page is, so
    counting it would only add noise."""
    squares = [
        TextLine("F", 0.9, quad_from_box(10.0, 10.0 + i * 40, 34.0, 34.0 + i * 40))
        for i in range(6)
    ]
    orientation, confidence = detect_orientation(squares)

    assert orientation is Orientation.UPRIGHT
    assert confidence == pytest.approx(0.0)


@pytest.mark.unit
def test_no_text_defaults_to_upright() -> None:
    assert detect_orientation([]) == (Orientation.UPRIGHT, 0.0)


@pytest.mark.unit
def test_every_rotation_is_offered_as_a_candidate() -> None:
    """Because shape cannot separate 90 from 270, or 0 from 180, the search
    must be able to reach all four - the ordering is an optimisation, not a
    filter."""
    for lines in ([landscape_line(i) for i in range(4)],
                  [portrait_line(i) for i in range(4)]):
        assert set(orientation_candidates(lines)) == set(Orientation)


@pytest.mark.unit
def test_the_shape_guess_is_tried_first() -> None:
    assert orientation_candidates(
        [portrait_line(i) for i in range(4)]
    )[0] is Orientation.ROTATE_90
    assert orientation_candidates(
        [landscape_line(i) for i in range(4)]
    )[0] is Orientation.UPRIGHT


@pytest.mark.unit
def test_the_other_quarter_turn_is_tried_second() -> None:
    """When geometry says quarter-turned, the two quarter turns are the only
    live hypotheses; spending the second pass on 180 would waste it."""
    candidates = orientation_candidates([portrait_line(i) for i in range(4)])
    assert candidates[1] is Orientation.ROTATE_270


@pytest.mark.unit
@pytest.mark.parametrize(
    "orientation",
    [Orientation.UPRIGHT, Orientation.ROTATE_90,
     Orientation.ROTATE_180, Orientation.ROTATE_270],
)
def test_rotation_is_lossless(clean_card: np.ndarray, orientation: Orientation) -> None:
    """Multiples of ninety degrees are pure transposition, so a full turn must
    return the original pixel for pixel."""
    once = rotate_image(clean_card, orientation)
    back = once
    for _ in range(3):
        back = rotate_image(back, orientation)

    assert np.array_equal(back, clean_card)


@pytest.mark.unit
def test_a_quarter_turn_swaps_the_axes(clean_card: np.ndarray) -> None:
    height, width = clean_card.shape[:2]
    turned = rotate_image(clean_card, Orientation.ROTATE_90)
    assert turned.shape[:2] == (width, height)


@pytest.mark.unit
def test_rotation_direction_is_clockwise(clean_card: np.ndarray) -> None:
    """Fixed rather than assumed: the whole rotation search, and the quad
    mapping back to source coordinates, depend on the sign."""
    turned = rotate_image(clean_card, Orientation.ROTATE_90)
    # Under a clockwise turn the source top-left corner lands top-right.
    assert np.array_equal(turned[0, -1], clean_card[0, 0])


@pytest.mark.unit
def test_the_fixture_rotation_and_the_correction_are_inverse(
    clean_card: np.ndarray,
) -> None:
    """Guards the integration tests: if the fixture turned the card one way
    and preprocessing corrected the other, every rotation case would pass for
    the wrong reason."""
    for degrees in (90, 180, 270):
        turned = rotated(clean_card, degrees)
        corrected = rotate_image(turned, Orientation(360 - degrees if degrees else 0))
        assert np.array_equal(corrected, clean_card)


# --------------------------------------------------------------------------- #
# Deskew
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_tilted_card_is_levelled(clean_card: np.ndarray) -> None:
    height, width = clean_card.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 5.0, 1.0)
    tilted = cv2.warpAffine(
        clean_card, matrix, (width, height), borderValue=(255, 255, 255)
    )

    _levelled, angle = deskew(tilted)

    assert abs(angle) > 1.0
    assert angle == pytest.approx(-5.0, abs=1.5)


@pytest.mark.unit
def test_a_level_card_is_left_alone(clean_card: np.ndarray) -> None:
    """Warping costs a resample, and resampling text that is already straight
    only softens it."""
    levelled, angle = deskew(clean_card)

    assert abs(angle) < 0.6
    assert np.array_equal(levelled, clean_card)


@pytest.mark.unit
def test_a_large_tilt_is_not_treated_as_skew(clean_card: np.ndarray) -> None:
    """Past the limit it is a rotation, not a skew, and correcting it as one
    would fight the orientation search."""
    turned = rotated(clean_card, 90)
    _result, angle = deskew(turned, max_angle=12.0)
    assert abs(angle) <= 12.0


@pytest.mark.unit
def test_deskew_preserves_the_frame_size(clean_card: np.ndarray) -> None:
    height, width = clean_card.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 4.0, 1.0)
    tilted = cv2.warpAffine(
        clean_card, matrix, (width, height), borderValue=(255, 255, 255)
    )

    levelled, _angle = deskew(tilted)
    assert levelled.shape == tilted.shape


# --------------------------------------------------------------------------- #
# Contrast and scale
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_dim_capture_gains_local_contrast(clean_card: np.ndarray) -> None:
    dim = dim_lighting(clean_card)
    enhanced = enhance_for_ocr(dim)

    before = float(cv2.cvtColor(dim, cv2.COLOR_BGR2GRAY).std())
    after = float(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).std())

    assert after > before


@pytest.mark.unit
def test_enhancement_preserves_shape_and_type(clean_card: np.ndarray) -> None:
    enhanced = enhance_for_ocr(clean_card)
    assert enhanced.shape == clean_card.shape
    assert enhanced.dtype == np.uint8


@pytest.mark.unit
def test_enhancement_does_not_shift_colour(clean_card: np.ndarray) -> None:
    """Equalisation runs on lightness alone. Touching the colour channels
    would tint the card without helping the recogniser, which reads grey."""
    enhanced = enhance_for_ocr(clean_card)
    source = cv2.cvtColor(clean_card, cv2.COLOR_BGR2LAB)
    result = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)

    for channel in (1, 2):
        assert float(np.abs(
            source[:, :, channel].astype(np.int16)
            - result[:, :, channel].astype(np.int16)
        ).mean()) < 2.0


@pytest.mark.unit
def test_a_small_card_is_enlarged() -> None:
    small = cv2.resize(render_cnic(), (320, 202))
    enlarged, scale = upscale_if_small(small, min_width=700)

    assert scale > 1.0
    assert enlarged.shape[1] >= 700


@pytest.mark.unit
def test_a_large_card_is_not_enlarged(clean_card: np.ndarray) -> None:
    """Upscaling adds no information and costs recognition time linearly in
    pixel count."""
    result, scale = upscale_if_small(clean_card, min_width=700)

    assert scale == pytest.approx(1.0)
    assert result is clean_card


@pytest.mark.unit
def test_upscaling_preserves_the_aspect_ratio() -> None:
    small = cv2.resize(render_cnic(), (320, 202))
    enlarged, _scale = upscale_if_small(small, min_width=700)

    assert (enlarged.shape[1] / enlarged.shape[0]) == pytest.approx(
        320 / 202, rel=0.01
    )


@pytest.mark.unit
def test_the_preprocessing_summary_serialises(clean_card: np.ndarray) -> None:
    import json

    result = rectify_document(photograph_on_desk(clean_card))
    summary = describe_preprocessing(
        result, deskew_angle=1.25, upscale=1.0, enhanced=True
    )

    json.dumps(summary)
    assert summary["rectified"] is True
    assert summary["deskew_degrees"] == pytest.approx(1.25)
