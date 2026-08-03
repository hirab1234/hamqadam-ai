"""Face alignment - the largest controllable factor in recognition accuracy.

Measured on the reference portrait, the same face embedded through the
landmark template and through the box-only fallback scores just 0.786 against
itself, and on a 25-degree rotated capture the fallback collapses to 0.573 -
below any sane impostor threshold. These tests exist because a silent
regression here would not crash anything; it would quietly start calling
people strangers.

None of them need model weights.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.constants import ARCFACE_REFERENCE_LANDMARKS_112
from hamqadam_ai.embeddings.alignment import (
    align_face_for_recognition,
    alignment_residual,
    estimate_similarity_transform,
    mirror,
)
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

BgrImage = npt.NDArray[np.uint8]


def template_landmarks(
    *, scale: float = 1.0, rotation_deg: float = 0.0, offset: tuple[float, float] = (0.0, 0.0)
) -> Landmarks5:
    """The canonical template, optionally transformed by a known similarity."""
    points = np.asarray(ARCFACE_REFERENCE_LANDMARKS_112, dtype=np.float32).copy()
    angle = math.radians(rotation_deg)
    matrix = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    transformed = (points * scale) @ matrix.T + np.asarray(offset, dtype=np.float32)
    return Landmarks5(transformed.astype(np.float32))


def textured(width: int = 400, height: int = 400, seed: int = 3) -> BgrImage:
    """A detailed test image."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (height, width), dtype=np.uint8)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    smooth = 128 + 55 * np.sin(x / 25.0) + 45 * np.cos(y / 18.0)
    blended = np.clip(0.5 * smooth + 0.5 * base, 0, 255).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- #
# The similarity transform
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_identity_landmarks_give_the_identity_transform() -> None:
    matrix = estimate_similarity_transform(template_landmarks())
    assert matrix is not None
    np.testing.assert_allclose(matrix[:, :2], np.eye(2), atol=1e-4)
    np.testing.assert_allclose(matrix[:, 2], np.zeros(2), atol=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize("scale", [0.4, 1.0, 2.5, 7.0])
def test_uniform_scale_is_recovered(scale: float) -> None:
    matrix = estimate_similarity_transform(template_landmarks(scale=scale))
    assert matrix is not None
    recovered = math.sqrt(abs(float(np.linalg.det(matrix[:, :2]))))
    assert recovered == pytest.approx(1.0 / scale, rel=0.02)


@pytest.mark.unit
@pytest.mark.parametrize("degrees", [-40.0, -12.0, 7.0, 25.0])
def test_rotation_is_undone(degrees: float) -> None:
    """The whole reason alignment matters: the warp must remove in-plane tilt."""
    landmarks = template_landmarks(rotation_deg=degrees)
    matrix = estimate_similarity_transform(landmarks)
    assert matrix is not None
    assert alignment_residual(landmarks, matrix) < 0.01


@pytest.mark.unit
def test_translation_is_undone() -> None:
    landmarks = template_landmarks(offset=(320.0, -95.0))
    matrix = estimate_similarity_transform(landmarks)
    assert matrix is not None
    assert alignment_residual(landmarks, matrix) < 0.01


@pytest.mark.unit
def test_a_combined_similarity_is_fully_undone() -> None:
    landmarks = template_landmarks(scale=3.2, rotation_deg=-17.0, offset=(210.0, 88.0))
    matrix = estimate_similarity_transform(landmarks)
    assert matrix is not None
    assert alignment_residual(landmarks, matrix) < 0.01


@pytest.mark.unit
def test_a_degenerate_landmark_set_returns_none() -> None:
    collapsed = Landmarks5(np.full((5, 2), 50.0, dtype=np.float32))
    assert estimate_similarity_transform(collapsed) is None


@pytest.mark.unit
def test_the_transform_never_mirrors_the_face() -> None:
    """A reflection would "fit" a mirrored landmark set by flipping the face,
    which the recogniser would then embed as somebody else."""
    points = np.asarray(ARCFACE_REFERENCE_LANDMARKS_112, dtype=np.float32).copy()
    points[:, 0] = 112.0 - points[:, 0]
    matrix = estimate_similarity_transform(Landmarks5(points))
    assert matrix is not None
    assert float(np.linalg.det(matrix[:, :2])) > 0.0, "the fit reflected the face"


@pytest.mark.unit
def test_the_transform_is_deterministic() -> None:
    """The embedding cache is keyed on the aligned crop, so a non-deterministic
    warp would silently produce a different vector for the same face."""
    landmarks = template_landmarks(scale=2.1, rotation_deg=9.0, offset=(31.0, 47.0))
    first = estimate_similarity_transform(landmarks)
    second = estimate_similarity_transform(landmarks)
    assert first is not None and second is not None
    np.testing.assert_array_equal(first, second)


# --------------------------------------------------------------------------- #
# The residual
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_perfect_fit_has_a_near_zero_residual() -> None:
    landmarks = template_landmarks(scale=1.7, rotation_deg=22.0)
    matrix = estimate_similarity_transform(landmarks)
    assert matrix is not None
    assert alignment_residual(landmarks, matrix) < 0.005


@pytest.mark.unit
def test_the_residual_grows_as_the_landmarks_stop_being_face_like() -> None:
    base = np.asarray(ARCFACE_REFERENCE_LANDMARKS_112, dtype=np.float32)
    residuals = []
    for perturbation in (0.0, 3.0, 9.0, 22.0):
        points = base.copy()
        points[2, 0] += perturbation  # slide the nose sideways
        landmarks = Landmarks5(points)
        matrix = estimate_similarity_transform(landmarks)
        assert matrix is not None
        residuals.append(alignment_residual(landmarks, matrix))
    assert residuals == sorted(residuals)


@pytest.mark.unit
def test_the_residual_is_scale_invariant() -> None:
    """Expressed in interocular units, so a big and a small face with the same
    proportional error report the same number."""
    base = np.asarray(ARCFACE_REFERENCE_LANDMARKS_112, dtype=np.float32)
    results = []
    for scale in (1.0, 4.0, 11.0):
        points = base.copy()
        points[2, 0] += 6.0
        points = points * scale
        landmarks = Landmarks5(points.astype(np.float32))
        matrix = estimate_similarity_transform(landmarks)
        assert matrix is not None
        results.append(alignment_residual(landmarks, matrix))
    assert max(results) - min(results) < 0.01


# --------------------------------------------------------------------------- #
# The full alignment path
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_landmark_alignment_produces_the_template_size() -> None:
    image = textured()
    result = align_face_for_recognition(
        image, landmarks=template_landmarks(scale=2.0, offset=(120.0, 110.0))
    )
    assert result.crop.shape == (112, 112, 3)
    assert result.crop.dtype == np.uint8
    assert result.aligned is True
    assert result.residual is not None
    assert result.trustworthy is True


@pytest.mark.unit
def test_alignment_places_the_eyes_on_the_template_points() -> None:
    """The functional check: after warping, a landmark must land where the
    template says it should."""
    landmarks = template_landmarks(scale=2.4, rotation_deg=15.0, offset=(90.0, 70.0))
    result = align_face_for_recognition(textured(600, 600), landmarks=landmarks)

    assert result.matrix is not None
    homogeneous = np.hstack([landmarks.points, np.ones((5, 1), dtype=np.float32)])
    projected = homogeneous @ result.matrix.T

    np.testing.assert_allclose(
        projected, ARCFACE_REFERENCE_LANDMARKS_112, atol=1.0
    )


@pytest.mark.unit
def test_an_implausible_landmark_set_falls_back_to_the_box() -> None:
    base = np.asarray(ARCFACE_REFERENCE_LANDMARKS_112, dtype=np.float32).copy()
    base[2] = [400.0, -300.0]  # a nose nowhere near the face
    result = align_face_for_recognition(
        textured(),
        box=BoundingBox(40.0, 40.0, 200.0, 200.0),
        landmarks=Landmarks5(base * 2.0),
        max_residual=0.28,
    )
    assert result.aligned is False
    assert result.residual is None
    assert result.reason is not None
    assert result.trustworthy is False


@pytest.mark.unit
def test_the_box_fallback_produces_a_usable_crop() -> None:
    result = align_face_for_recognition(
        textured(), box=BoundingBox(60.0, 60.0, 260.0, 260.0), landmarks=None
    )
    assert result.crop.shape == (112, 112, 3)
    assert result.aligned is False


@pytest.mark.unit
def test_the_fallback_can_be_refused() -> None:
    """Some callers would rather fail than embed an unaligned face."""
    with pytest.raises(ValueError, match="disabled"):
        align_face_for_recognition(
            textured(),
            box=BoundingBox(60.0, 60.0, 260.0, 260.0),
            landmarks=None,
            allow_box_fallback=False,
        )


@pytest.mark.unit
def test_alignment_needs_something_to_work_with() -> None:
    with pytest.raises(ValueError, match="landmarks or a bounding box"):
        align_face_for_recognition(textured())


@pytest.mark.unit
def test_a_face_at_the_frame_edge_is_padded_not_clipped() -> None:
    """Clipping would change the crop's framing and shift every landmark
    relative to the template."""
    result = align_face_for_recognition(
        textured(200, 200), landmarks=template_landmarks(scale=1.6, offset=(150.0, 150.0))
    )
    assert result.crop.shape == (112, 112, 3)


@pytest.mark.unit
def test_alignment_is_deterministic_pixel_for_pixel() -> None:
    """What makes the embedding cache sound."""
    image = textured()
    landmarks = template_landmarks(scale=2.2, rotation_deg=-8.0, offset=(100.0, 95.0))
    first = align_face_for_recognition(image, landmarks=landmarks)
    second = align_face_for_recognition(image, landmarks=landmarks)
    np.testing.assert_array_equal(first.crop, second.crop)


@pytest.mark.unit
def test_a_non_arcface_output_size_is_rejected_in_configuration() -> None:
    """The template's coordinates are absolute pixels at 112x112, so another
    size would silently mis-place every reference point."""
    from hamqadam_ai.core.config import AlignmentConfig

    with pytest.raises(ValueError, match=r"112, 112"):
        AlignmentConfig(output_size=(128, 128))


# --------------------------------------------------------------------------- #
# Mirroring
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_mirror_flips_horizontally_and_is_its_own_inverse() -> None:
    image = textured(64, 64)
    np.testing.assert_array_equal(mirror(mirror(image)), image)
    np.testing.assert_array_equal(mirror(image)[:, ::-1], image)
