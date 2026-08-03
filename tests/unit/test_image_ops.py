"""Image transforms.

The coordinate round-trip is the important part. Every detector works in
network space and must map its output back to source-image pixels; a broken
un-map produces boxes that are subtly, consistently wrong on non-square inputs
— which is exactly the shape of every phone photo this service will ever see.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.constants import ARCFACE_REFERENCE_LANDMARKS_112
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5
from hamqadam_ai.utils.image_ops import (
    align_face,
    build_blob,
    convert_to_bgr,
    crop_with_margin,
    ensure_min_size,
    gradient_energy,
    letterbox,
    resize_long_side,
    skin_mask,
    to_grayscale,
)

BgrImage = npt.NDArray[np.uint8]


def textured(width: int, height: int) -> BgrImage:
    """A deterministic non-uniform test image."""
    rng = np.random.default_rng(4242)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    image[:, :, 2] = rng.integers(0, 256, (height, width), dtype=np.uint8)
    return image


# --------------------------------------------------------------------------- #
# Letterbox and the coordinate round-trip
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_letterbox_produces_the_requested_canvas() -> None:
    padded, _ = letterbox(textured(1280, 720), (640, 640))
    assert padded.shape == (640, 640, 3)


@pytest.mark.unit
def test_letterbox_preserves_aspect_ratio() -> None:
    """Stretching a portrait to square costs several points of recall."""
    source = textured(1280, 720)
    _, transform = letterbox(source, (640, 640))
    assert transform.scale == pytest.approx(0.5)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("width", "height"), [(1280, 720), (720, 1280), (640, 640), (1000, 333)]
)
def test_box_round_trips_through_the_letterbox(width: int, height: int) -> None:
    """The property every detector depends on."""
    source = textured(width, height)
    _, transform = letterbox(source, (640, 640))

    original = BoundingBox(0.15 * width, 0.20 * height, 0.55 * width, 0.70 * height)
    mapped = BoundingBox(
        original.x1 * transform.scale + transform.pad_x,
        original.y1 * transform.scale + transform.pad_y,
        original.x2 * transform.scale + transform.pad_x,
        original.y2 * transform.scale + transform.pad_y,
    )
    recovered = transform.unmap_box(mapped)

    assert recovered.as_tuple() == pytest.approx(original.as_tuple(), abs=0.01)


@pytest.mark.unit
def test_landmarks_round_trip_through_the_letterbox() -> None:
    source = textured(900, 1600)
    _, transform = letterbox(source, (640, 640))

    original = Landmarks5(
        np.array(
            [[300, 500], [420, 505], [360, 580], [310, 660], [410, 663]],
            dtype=np.float32,
        )
    )
    mapped = Landmarks5(
        original.points * transform.scale
        + np.array([transform.pad_x, transform.pad_y], dtype=np.float32)
    )
    recovered = transform.unmap_landmarks(mapped)

    assert recovered.points == pytest.approx(original.points, abs=0.01)


@pytest.mark.unit
def test_centred_padding_is_symmetric() -> None:
    _, transform = letterbox(textured(1280, 640), (640, 640), center=True)
    # 1280x640 scales to 640x320, leaving 320 rows split evenly.
    assert transform.pad_y == pytest.approx(160.0)
    assert transform.pad_x == pytest.approx(0.0)


@pytest.mark.unit
def test_top_left_padding_has_no_offset() -> None:
    """SCRFD's reference pre-processing pads bottom-right only."""
    _, transform = letterbox(textured(1280, 640), (640, 640), center=False)
    assert transform.pad_x == 0.0
    assert transform.pad_y == 0.0


@pytest.mark.unit
def test_pad_value_is_applied() -> None:
    padded, _ = letterbox(textured(1280, 320), (640, 640), pad_value=114, center=False)
    assert int(padded[600, 320, 0]) == 114


@pytest.mark.unit
def test_unmap_array_handles_batched_points() -> None:
    _, transform = letterbox(textured(1280, 720), (640, 640))
    points = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float32)
    recovered = transform.unmap_array(points)
    assert recovered.shape == points.shape
    expected_x = (10.0 - transform.pad_x) / transform.scale
    assert recovered[0, 0, 0] == pytest.approx(expected_x)


# --------------------------------------------------------------------------- #
# Rescaling guards
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_oversized_image_is_downscaled() -> None:
    resized, scale = resize_long_side(textured(4000, 3000), 1920)
    assert max(resized.shape[:2]) == 1920
    assert scale == pytest.approx(0.48)


@pytest.mark.unit
def test_image_within_the_limit_is_untouched() -> None:
    source = textured(1280, 720)
    resized, scale = resize_long_side(source, 1920)
    assert scale == 1.0
    assert resized is source


@pytest.mark.unit
def test_undersized_image_is_upscaled() -> None:
    resized, scale = ensure_min_size(textured(200, 120), 160)
    assert min(resized.shape[:2]) >= 160
    assert scale > 1.0


@pytest.mark.unit
def test_image_above_the_floor_is_untouched() -> None:
    source = textured(640, 480)
    resized, scale = ensure_min_size(source, 160)
    assert scale == 1.0
    assert resized is source


# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_crop_returns_the_requested_region() -> None:
    source = textured(640, 480)
    crop, effective = crop_with_margin(source, BoundingBox(100, 80, 300, 320))
    assert crop.shape[:2] == (240, 200)
    assert effective.as_int_tuple() == (100, 80, 300, 320)


@pytest.mark.unit
def test_margin_expands_the_crop() -> None:
    source = textured(640, 480)
    crop, _ = crop_with_margin(source, BoundingBox(200, 150, 300, 250), margin=0.2)
    assert crop.shape[1] > 100
    assert crop.shape[0] > 100


@pytest.mark.unit
def test_crop_at_the_edge_is_padded_not_clipped() -> None:
    """Clipping silently changes the aspect ratio and shifts the face
    off-centre, which shifts the alignment landmarks and degrades the
    embedding."""
    source = textured(640, 480)
    crop, _ = crop_with_margin(source, BoundingBox(-50, -40, 150, 200))
    assert crop.shape[:2] == (240, 200)
    # The out-of-frame corner carries the fill value, not image content.
    assert int(crop[5, 5, 0]) == 114


@pytest.mark.unit
def test_square_crop_preserves_aspect_ratio() -> None:
    source = textured(640, 480)
    crop, _ = crop_with_margin(source, BoundingBox(100, 100, 200, 300), square=True)
    assert crop.shape[0] == crop.shape[1]


# --------------------------------------------------------------------------- #
# Colour conversion
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_grayscale_conversion_is_idempotent() -> None:
    gray = to_grayscale(textured(64, 64))
    assert to_grayscale(gray).shape == gray.shape


@pytest.mark.unit
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_convert_to_bgr_normalises_channel_count(channels: int) -> None:
    source = np.zeros((32, 32, channels), dtype=np.uint8)
    assert convert_to_bgr(source).shape == (32, 32, 3)


@pytest.mark.unit
def test_convert_to_bgr_handles_a_2d_array() -> None:
    assert convert_to_bgr(np.zeros((32, 32), dtype=np.uint8)).shape == (32, 32, 3)


@pytest.mark.unit
def test_convert_to_bgr_clips_a_float_array() -> None:
    source = np.full((16, 16, 3), 300.0, dtype=np.float32)
    assert int(convert_to_bgr(source).max()) == 255


@pytest.mark.unit
def test_unsupported_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="Cannot interpret"):
        convert_to_bgr(np.zeros((4, 4, 7), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Blob construction
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_blob_shape_is_nchw() -> None:
    blob = build_blob(
        textured(640, 640), (640, 640), mean=(127.5,) * 3, scale=1 / 128, swap_rb=True
    )
    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32


@pytest.mark.unit
def test_blob_applies_mean_and_scale() -> None:
    source = np.full((32, 32, 3), 200, dtype=np.uint8)
    blob = build_blob(source, (32, 32), mean=(100.0,) * 3, scale=0.5, swap_rb=False)
    assert float(blob.max()) == pytest.approx((200 - 100) * 0.5)


@pytest.mark.unit
def test_blob_swaps_channels_when_asked() -> None:
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    source[:, :, 0] = 255  # blue in BGR

    without = build_blob(source, (8, 8), mean=(0.0,) * 3, scale=1.0, swap_rb=False)
    with_swap = build_blob(source, (8, 8), mean=(0.0,) * 3, scale=1.0, swap_rb=True)

    assert float(without[0, 0].max()) == 255.0  # channel 0 populated
    assert float(with_swap[0, 2].max()) == 255.0  # moved to channel 2


@pytest.mark.unit
def test_blob_resizes_a_mismatched_input() -> None:
    blob = build_blob(
        textured(320, 240), (640, 640), mean=(0.0,) * 3, scale=1.0, swap_rb=False
    )
    assert blob.shape == (1, 3, 640, 640)


@pytest.mark.unit
def test_blob_is_contiguous() -> None:
    """ONNX Runtime copies a non-contiguous buffer on every call."""
    blob = build_blob(
        textured(64, 64), (64, 64), mean=(0.0,) * 3, scale=1.0, swap_rb=True
    )
    assert blob.flags["C_CONTIGUOUS"]


# --------------------------------------------------------------------------- #
# Face alignment
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_alignment_maps_landmarks_onto_the_template() -> None:
    """The ArcFace pre-processing step. Getting it wrong silently degrades
    every downstream similarity score."""
    source = textured(400, 400)
    landmarks = Landmarks5(
        np.array(
            [[150, 170], [250, 170], [200, 220], [160, 270], [240, 270]],
            dtype=np.float32,
        )
    )
    aligned = align_face(
        source, landmarks, ARCFACE_REFERENCE_LANDMARKS_112, (112, 112)
    )
    assert aligned.shape == (112, 112, 3)


@pytest.mark.unit
def test_alignment_removes_in_plane_rotation() -> None:
    """A tilted face and its upright twin must align to the same result."""
    upright = Landmarks5(
        np.array(
            [[150, 170], [250, 170], [200, 220], [160, 270], [240, 270]],
            dtype=np.float32,
        )
    )

    angle = np.radians(20.0)
    centre = np.array([200.0, 220.0], dtype=np.float32)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    tilted_points = (upright.points - centre) @ rotation.T + centre
    tilted = Landmarks5(tilted_points.astype(np.float32))

    matrix, _ = cv2.estimateAffinePartial2D(
        tilted.points, ARCFACE_REFERENCE_LANDMARKS_112, method=cv2.LMEDS
    )
    assert matrix is not None
    projected = (
        np.hstack([tilted.points, np.ones((5, 1), dtype=np.float32)]) @ matrix.T
    )
    assert projected == pytest.approx(ARCFACE_REFERENCE_LANDMARKS_112, abs=2.0)


@pytest.mark.unit
def test_alignment_survives_degenerate_landmarks() -> None:
    """LMEDS can fail; the two-point eye fallback must always have a solution."""
    source = textured(200, 200)
    collinear = Landmarks5(
        np.array(
            [[50, 100], [150, 100], [100, 100], [60, 100], [140, 100]],
            dtype=np.float32,
        )
    )
    aligned = align_face(
        source, collinear, ARCFACE_REFERENCE_LANDMARKS_112, (112, 112)
    )
    assert aligned.shape == (112, 112, 3)


# --------------------------------------------------------------------------- #
# Analysis primitives
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_skin_mask_accepts_a_skin_tone() -> None:
    """Chrominance-only, so it must work across skin tones and lighting."""
    for bgr in [(150, 175, 205), (90, 115, 150), (60, 80, 110)]:
        patch = np.full((64, 64, 3), bgr, dtype=np.uint8)
        coverage = float(np.mean(skin_mask(patch) > 0))
        assert coverage > 0.5, f"{bgr} was not classified as skin"


@pytest.mark.unit
def test_skin_mask_rejects_obvious_non_skin() -> None:
    for bgr in [(20, 200, 20), (220, 30, 30), (15, 15, 15)]:
        patch = np.full((64, 64, 3), bgr, dtype=np.uint8)
        coverage = float(np.mean(skin_mask(patch) > 0))
        assert coverage < 0.5, f"{bgr} was wrongly classified as skin"


@pytest.mark.unit
def test_gradient_energy_is_zero_on_a_flat_patch() -> None:
    flat = np.full((64, 64), 128, dtype=np.uint8)
    assert float(gradient_energy(flat).mean()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_gradient_energy_rises_with_texture() -> None:
    flat = np.full((64, 64), 128, dtype=np.uint8)
    rng = np.random.default_rng(8)
    noisy = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    blocks = ((np.indices((64, 64)) // 8).sum(axis=0) % 2 * 255).astype(np.uint8)

    assert gradient_energy(noisy).mean() > gradient_energy(flat).mean()
    assert gradient_energy(blocks).mean() > gradient_energy(flat).mean()


@pytest.mark.unit
def test_gradient_energy_is_blind_at_the_nyquist_frequency() -> None:
    """A documented property of the 3x3 Sobel kernel, not a defect.

    On a single-pixel checkerboard the kernel's +1 and -1 taps sample identical
    values and cancel exactly, so the response is zero despite maximal local
    contrast. It matters here because the occlusion analyser's flat-fraction
    signal is built on this operator: a synthetic pattern at exactly this
    frequency would read as perfectly flat. Real photographs never contain it
    (the camera's anti-alias filter and JPEG chroma subsampling both remove it),
    so the blind spot is harmless in production - but it must not surprise
    anyone reading a diagnostic.
    """
    checker = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
    assert float(gradient_energy(checker).mean()) == pytest.approx(0.0, abs=1e-6)

    # A 3x3 Gaussian is *also* nulled by this pattern - it averages to a
    # uniform 128 - so blurring does not rescue it. Rescaling does, because it
    # moves the fundamental away from Nyquist.
    assert float(gradient_energy(cv2.GaussianBlur(checker, (3, 3), 0)).mean()) == (
        pytest.approx(0.0, abs=1e-6)
    )

    enlarged = cv2.resize(checker, (192, 192), interpolation=cv2.INTER_NEAREST)
    assert float(gradient_energy(enlarged).mean()) > 1.0
