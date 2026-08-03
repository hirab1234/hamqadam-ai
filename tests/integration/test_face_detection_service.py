"""MODULE 1 integration - the real service against real SCRFD weights.

These tests need the model store populated, so they skip cleanly on a fresh
checkout. Run ``python scripts/download_models.py`` first.

Test imagery is the US Navy portrait of Rear Admiral Grace Hopper bundled
inside matplotlib: a genuine photograph of a real face, public domain, already
on disk. Every rejection scenario is composed from it, so the whole suite runs
against real detector output without a single identity document entering the
repository.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import Settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import (
    FaceNotDetectedError,
    MultipleFacesDetectedError,
)
from hamqadam_ai.models.registry import ModelRegistry
from hamqadam_ai.services import build_face_detection_service
from hamqadam_ai.services.face_detection_service import FaceDetectionService
from hamqadam_ai.utils.image_io import load_image

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def portrait_path() -> Path:
    """The public-domain portrait bundled with matplotlib."""
    matplotlib = pytest.importorskip("matplotlib")
    path = (
        Path(matplotlib.__file__).parent
        / "mpl-data"
        / "sample_data"
        / "grace_hopper.jpg"
    )
    if not path.is_file():
        pytest.skip("matplotlib sample portrait unavailable")
    return path


@pytest.fixture(scope="module")
def portrait(portrait_path: Path) -> BgrImage:
    """A real photograph of a real, clearly-visible face."""
    return load_image(portrait_path, role="test").pixels


@pytest.fixture(scope="module")
def service(settings: Settings, has_scrfd: bool) -> FaceDetectionService:
    """The fully-wired detection service on real weights."""
    if not has_scrfd:
        pytest.skip("SCRFD weights absent; run scripts/download_models.py")
    registry = ModelRegistry(settings)
    built = build_face_detection_service(settings, registry)
    yield built
    built.close()
    registry.close()


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_real_portrait_passes_every_gate(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    result = service.detect(portrait, role=ImageRole.LIVE_SELFIE)

    assert result.passed is True
    assert result.error_code is None
    assert result.face_detected is True
    assert result.face_count == 1
    assert result.detector == "scrfd"
    assert result.used_fallback is False


def test_specification_mandated_fields_are_populated(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    """face_detected, face_count, face_visibility_score, bounding_box, pose,
    confidence - the six outputs Module 1 must return."""
    result = service.detect(portrait, role=ImageRole.LIVE_SELFIE)
    face = result.primary_face

    assert face is not None
    assert isinstance(result.face_detected, bool)
    assert result.face_count == 1
    assert 0.0 <= result.face_visibility_score <= 100.0
    assert face.bounding_box.width > 0 and face.bounding_box.height > 0
    assert face.pose is not None
    assert 0.0 <= face.confidence <= 1.0


def test_bounding_box_is_plausible_for_a_portrait(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    face = service.detect(portrait).primary_face
    assert face is not None
    box = face.bounding_box

    assert 0 <= box.x1 < box.x2 <= result_width(portrait)
    assert 0 <= box.y1 < box.y2 <= result_height(portrait)
    # A head is taller than it is wide, but not by much.
    assert 0.55 <= (box.width / box.height) <= 1.15
    # The face should occupy a meaningful but not dominant share of a portrait.
    assert 0.02 <= face.face_area_ratio <= 0.60


def test_five_landmarks_are_anatomically_ordered(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    face = service.detect(portrait).primary_face
    assert face is not None
    assert len(face.landmarks) == 5

    points = {landmark.name: (landmark.x, landmark.y) for landmark in face.landmarks}
    assert points["left_eye"][0] < points["right_eye"][0]
    assert points["mouth_left"][0] < points["mouth_right"][0]
    eye_y = (points["left_eye"][1] + points["right_eye"][1]) / 2
    mouth_y = (points["mouth_left"][1] + points["mouth_right"][1]) / 2
    assert eye_y < points["nose_tip"][1] < mouth_y


def test_pose_of_a_frontal_portrait_is_near_zero(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    face = service.detect(portrait).primary_face
    assert face is not None
    pose = face.pose
    assert pose is not None

    assert pose.method == "pnp"
    assert pose.frontal is True
    assert abs(pose.yaw) < 20.0
    assert abs(pose.roll) < 20.0
    assert pose.reprojection_error is not None
    assert pose.reprojection_error < 10.0


def test_clear_face_is_not_reported_as_occluded(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    face = service.detect(portrait).primary_face
    assert face is not None
    assert face.occlusion is not None
    assert face.occlusion.occluded is False
    assert face.occlusion.occluded_regions == []
    assert len(face.occlusion.regions) == 6


def test_visibility_breakdown_reconstructs_the_composite(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    """The published breakdown must actually explain the published score."""
    face = service.detect(portrait).primary_face
    assert face is not None
    breakdown = face.visibility_breakdown
    assert breakdown is not None

    recomputed = sum(
        breakdown.weights[name] * getattr(breakdown, name)
        for name in ("detector_confidence", "occlusion", "pose", "face_size", "framing")
    )
    assert recomputed * 100.0 == pytest.approx(face.face_visibility_score, abs=0.5)


# --------------------------------------------------------------------------- #
# Rejection paths
# --------------------------------------------------------------------------- #


def test_image_with_no_face_is_rejected(service: FaceDetectionService) -> None:
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    result = service.detect(noise, role=ImageRole.LIVE_SELFIE)

    assert result.passed is False
    assert result.face_detected is False
    assert result.face_count == 0
    assert result.error_code is ErrorCode.FACE_NOT_DETECTED
    assert result.face_visibility_score == 0.0


def test_two_people_are_rejected(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    height, width = portrait.shape[:2]
    pair = np.zeros((height, width * 2, 3), dtype=np.uint8)
    pair[:, :width] = portrait
    pair[:, width:] = portrait

    result = service.detect(pair, role=ImageRole.PROFILE_IMAGE)

    assert result.passed is False
    assert result.face_count == 2
    assert result.error_code is ErrorCode.MULTIPLE_FACES_DETECTED
    assert result.multiple_faces is True


def test_sunglasses_are_detected_as_eye_occlusion(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    covered = portrait.copy()
    probe = service.detect(portrait)
    assert probe.primary_face is not None
    box = probe.primary_face.bounding_box

    # A flat dark band across the eye line, which is what sunglasses are.
    eye_top = int(box.y1 + box.height * 0.20)
    eye_bottom = int(box.y1 + box.height * 0.45)
    covered[eye_top:eye_bottom, int(box.x1) : int(box.x2)] = 12

    result = service.detect(covered, role=ImageRole.LIVE_SELFIE)

    assert result.passed is False
    assert result.error_code is ErrorCode.FACE_OCCLUDED
    assert result.primary_face is not None
    occlusion = result.primary_face.occlusion
    assert occlusion is not None
    assert occlusion.occluded is True
    assert {"left_eye", "right_eye"} <= {str(r) for r in occlusion.occluded_regions}


def test_face_too_small_is_rejected_with_its_own_code(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    canvas = np.full((1600, 1600, 3), 40, dtype=np.uint8)
    rng = np.random.default_rng(3)
    canvas += rng.integers(0, 25, canvas.shape, dtype=np.uint8)
    small = cv2.resize(portrait, (110, 138), interpolation=cv2.INTER_AREA)
    canvas[700:838, 700:810] = small

    result = service.detect(canvas, role=ImageRole.PROFILE_IMAGE)

    assert result.passed is False
    assert result.error_code in {
        ErrorCode.FACE_TOO_SMALL,
        ErrorCode.FACE_NOT_DETECTED,
    }


def test_extreme_downscale_raises_rather_than_returning(
    service: FaceDetectionService,
) -> None:
    from hamqadam_ai.core.exceptions import ImageTooSmallError

    with pytest.raises(ImageTooSmallError):
        service.detect(np.zeros((40, 40, 3), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Determinism, invariance and robustness
# --------------------------------------------------------------------------- #


def test_detection_is_deterministic(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    """A verification decision must be reproducible for an audit."""
    first = service.detect(portrait)
    second = service.detect(portrait)

    assert first.face_count == second.face_count
    assert first.face_visibility_score == second.face_visibility_score
    assert first.primary_face is not None and second.primary_face is not None
    assert first.primary_face.bounding_box.x1 == second.primary_face.bounding_box.x1
    assert first.primary_face.confidence == second.primary_face.confidence


def test_upscaling_the_source_does_not_change_the_verdict(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    doubled = cv2.resize(portrait, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    original = service.detect(portrait)
    scaled = service.detect(doubled)

    assert scaled.passed == original.passed
    assert scaled.face_count == original.face_count
    # The face occupies the same fraction of the frame either way.
    assert scaled.primary_face is not None and original.primary_face is not None
    assert scaled.primary_face.face_area_ratio == pytest.approx(
        original.primary_face.face_area_ratio, abs=0.02
    )


def test_moderate_head_tilt_is_still_accepted(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    height, width = portrait.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 12.0, 1.0)
    tilted = cv2.warpAffine(
        portrait, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
    )
    result = service.detect(tilted, role=ImageRole.LIVE_SELFIE)

    assert result.passed is True
    assert result.primary_face is not None
    assert result.primary_face.pose is not None
    # Rotating the image anticlockwise raises the viewer-right eye: negative roll.
    assert result.primary_face.pose.roll < -4.0


def test_grayscale_source_still_detects(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    gray = cv2.cvtColor(portrait, cv2.COLOR_BGR2GRAY)
    three_channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert service.detect(three_channel).face_detected is True


def test_heavy_jpeg_compression_still_detects(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    ok, buffer = cv2.imencode(".jpg", portrait, [int(cv2.IMWRITE_JPEG_QUALITY), 18])
    assert ok
    degraded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    assert service.detect(degraded).face_detected is True


# --------------------------------------------------------------------------- #
# Fail-fast variant and async surface
# --------------------------------------------------------------------------- #


def test_detect_or_raise_returns_on_success(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    assert service.detect_or_raise(portrait).passed is True


def test_detect_or_raise_raises_for_no_face(service: FaceDetectionService) -> None:
    rng = np.random.default_rng(11)
    noise = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    with pytest.raises(FaceNotDetectedError):
        service.detect_or_raise(noise, role=ImageRole.CNIC_PORTRAIT)


def test_detect_or_raise_raises_for_multiple_faces(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    height, width = portrait.shape[:2]
    pair = np.zeros((height, width * 2, 3), dtype=np.uint8)
    pair[:, :width] = portrait
    pair[:, width:] = portrait
    with pytest.raises(MultipleFacesDetectedError) as excinfo:
        service.detect_or_raise(pair)
    assert excinfo.value.details["face_count"] == 2


async def test_async_detection_matches_the_sync_path(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    sync = service.detect(portrait)
    asynchronous = await service.detect_async(portrait)
    assert asynchronous.face_count == sync.face_count
    assert asynchronous.face_visibility_score == sync.face_visibility_score


async def test_concurrent_batch_isolates_failures(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    """One bad image must not abort the other six in a verification request."""
    rng = np.random.default_rng(5)
    noise = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    broken = np.zeros((10, 10, 3), dtype=np.uint8)  # below the hard floor

    results = await service.detect_many_async(
        [
            (portrait, ImageRole.LIVE_SELFIE),
            (noise, ImageRole.PROFILE_IMAGE),
            (broken, ImageRole.SECONDARY_IMAGE),
            (portrait, ImageRole.SECONDARY_IMAGE),
        ]
    )

    assert len(results) == 4
    assert getattr(results[0], "passed", None) is True
    assert getattr(results[1], "passed", None) is False
    assert isinstance(results[2], Exception), "the broken image should yield an error"
    assert getattr(results[3], "passed", None) is True


# --------------------------------------------------------------------------- #
# Registry and provenance
# --------------------------------------------------------------------------- #


def test_model_version_is_reported_for_audit(
    service: FaceDetectionService, portrait: BgrImage
) -> None:
    result = service.detect(portrait)
    assert result.detector_version.startswith("scrfd_10g")


def test_registry_reports_ready_with_verified_digests(settings: Settings) -> None:
    registry = ModelRegistry(settings)
    try:
        registry.preload(["face_detector_scrfd"])
        report = registry.status_report()
        scrfd = next(m for m in report["models"] if m["key"] == "face_detector_scrfd")
        assert scrfd["state"] == "loaded"
        assert registry.version_map()["face_detector_scrfd"].startswith("scrfd_10g")
    finally:
        registry.close()


def test_tampered_artifact_is_refused(settings: Settings, tmp_path: Path) -> None:
    """A modified detector is a total compromise of the verification decision."""
    from hamqadam_ai.core.exceptions import ModelChecksumError

    registry = ModelRegistry(settings)
    try:
        genuine = registry.resolve_path("face_detector_scrfd")
        if not genuine.is_file():
            pytest.skip("SCRFD weights absent")
        if settings.models["face_detector_scrfd"].sha256 is None:
            pytest.skip("no pinned digest to violate")

        tampered = tmp_path / "det_10g.onnx"
        data = bytearray(genuine.read_bytes())
        data[-1] ^= 0xFF
        tampered.write_bytes(bytes(data))

        patched = settings.model_copy(deep=True)
        patched.models["face_detector_scrfd"].path = str(tampered)
        hostile = ModelRegistry(patched)
        try:
            with pytest.raises(ModelChecksumError):
                hostile.verify_artifact("face_detector_scrfd")
        finally:
            hostile.close()
    finally:
        registry.close()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def result_width(image: BgrImage) -> int:
    """Width of a BGR array."""
    return int(image.shape[1])


def result_height(image: BgrImage) -> int:
    """Height of a BGR array."""
    return int(image.shape[0])
