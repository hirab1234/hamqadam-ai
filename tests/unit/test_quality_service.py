"""The MODULE 2 service: image in, QualityResult out.

Covers the orchestration rather than the individual metrics - context
construction, detection hand-off, role policy, the response contract, error
containment and the async surface.
"""

from __future__ import annotations

import asyncio
import json

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import HamqadamError, QualityRejectedError
from hamqadam_ai.schemas.quality import QualityResult
from hamqadam_ai.services.quality_service import build_quality_service
from hamqadam_ai.utils.geometry import BoundingBox

BgrImage = npt.NDArray[np.uint8]

ALL_FAMILIES = {
    "blur",
    "sharpness",
    "brightness",
    "contrast",
    "noise",
    "resolution",
    "pixelation",
    "distortion",
}


def textured(width: int = 512, height: int = 512, seed: int = 11) -> BgrImage:
    """A broadband test image with detail at every spatial frequency."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (height, width), dtype=np.uint8)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    smooth = 128 + 60 * np.sin(x / 40.0) + 40 * np.cos(y / 25.0)
    blended = np.clip(0.55 * smooth + 0.45 * base, 0, 255).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="module")
def service():  # noqa: ANN201 - pytest fixture
    """The quality service. Needs no model weights, so it always builds."""
    return build_quality_service()


# --------------------------------------------------------------------------- #
# Response contract
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_assess_returns_every_specified_field(service) -> None:
    """The four scalars the requirements name explicitly, plus the five further
    dimensions it asks to be detected."""
    result = service.assess(textured(), role=ImageRole.PROFILE_IMAGE)

    assert isinstance(result, QualityResult)
    for field in (
        "image_quality_score",
        "blur_score",
        "brightness_score",
        "noise_score",
        "contrast_score",
        "resolution_score",
        "sharpness_score",
        "distortion_score",
        "pixelation_score",
    ):
        value = getattr(result, field)
        assert 0.0 <= value <= 100.0, field


@pytest.mark.unit
def test_every_dimension_is_reported_with_its_evidence(service) -> None:
    result = service.assess(textured())
    assert set(result.metrics) >= ALL_FAMILIES

    blur = result.metrics["blur"]
    assert blur.measurements, "raw measurements are part of the contract"
    assert blur.sub_scores


@pytest.mark.unit
def test_the_result_is_json_serialisable(service) -> None:
    """It crosses an HTTP boundary, so it must survive serialisation."""
    result = service.assess(textured())
    payload = result.model_dump(mode="json")
    json.dumps(payload)
    assert payload["image_quality_score"] == result.image_quality_score


@pytest.mark.unit
def test_summary_is_compact_and_pii_free(service) -> None:
    summary = service.assess(textured()).summary()
    json.dumps(summary)
    assert "measurements" not in summary


@pytest.mark.unit
def test_image_dimensions_are_echoed(service) -> None:
    result = service.assess(textured(320, 240))
    assert (result.image_width, result.image_height) == (320, 240)


@pytest.mark.unit
def test_duration_is_recorded(service) -> None:
    assert service.assess(textured()).duration_ms > 0.0


# --------------------------------------------------------------------------- #
# Face awareness
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_without_a_face_sharpness_is_unmeasured(service) -> None:
    result = service.assess(textured())
    assert result.face_analysed is False
    assert "sharpness" in result.unmeasured
    assert result.metrics["sharpness"].measured is False


@pytest.mark.unit
def test_with_a_face_box_sharpness_is_measured(service) -> None:
    result = service.assess(
        textured(), face_box=BoundingBox(120.0, 120.0, 380.0, 380.0)
    )
    assert result.face_analysed is True
    assert result.metrics["sharpness"].measured is True
    assert "sharpness" not in result.unmeasured


@pytest.mark.unit
def test_exposure_is_measured_on_the_face_when_one_is_present(service) -> None:
    """A selfie against a bright window has a healthy global histogram and a
    silhouetted, unusable face. What matters is the exposure of the biometric.
    """
    # A well-exposed scene with a silhouetted subject. The dark region is
    # drawn larger than the face box so the analysis crop's 15% margin still
    # lands on dark pixels rather than dragging in the bright surround.
    image = np.full((400, 400, 3), 140, dtype=np.uint8)
    image[100:300, 100:300] = (textured(200, 200, seed=3) * 0.09).astype(np.uint8)

    with_face = service.assess(image, face_box=BoundingBox(140, 140, 260, 260))
    without_face = service.assess(image)

    assert with_face.metrics["brightness"].measurements["measured_on_face"] == 1.0
    assert without_face.metrics["brightness"].measurements["measured_on_face"] == 0.0

    # The scene reads as correctly exposed; the biometric does not.
    assert without_face.brightness_score > 90.0
    assert with_face.brightness_score < 40.0


@pytest.mark.unit
def test_landmarks_enable_the_interocular_and_geometric_terms(
    service, synthetic_face
) -> None:
    result = service.assess(
        synthetic_face.image,
        face_box=synthetic_face.box,
        landmarks=synthetic_face.landmarks,
        yaw_degrees=0.0,
    )
    assert "interocular" in result.metrics["resolution"].sub_scores
    assert "geometric" in result.metrics["distortion"].sub_scores


@pytest.mark.unit
def test_detection_result_supplies_the_face_geometry(service, synthetic_face) -> None:
    """The normal path: Module 1 runs first and hands its primary face over."""
    from hamqadam_ai.core.config import get_settings
    from hamqadam_ai.models.registry import get_registry
    from hamqadam_ai.services import build_face_detection_service

    settings = get_settings()
    detector = build_face_detection_service(settings, get_registry(settings))
    detection = detector.detect(synthetic_face.image, role=ImageRole.LIVE_SELFIE)

    if not detection.face_detected:
        pytest.skip("detector found no face in the synthetic fixture")

    result = service.assess(
        synthetic_face.image, role=ImageRole.LIVE_SELFIE, detection=detection
    )
    assert result.face_analysed is True


@pytest.mark.unit
def test_a_detection_without_a_face_degrades_gracefully(service, synthetic_face) -> None:
    from hamqadam_ai.schemas.detection import FaceDetectionResult

    empty = FaceDetectionResult(
        face_detected=False,
        face_count=0,
        face_visibility_score=0.0,
        role=ImageRole.LIVE_SELFIE,
        primary_face=None,
        faces=[],
        raw_detection_count=0,
        detector="scrfd",
        detector_version="test",
        image_width=synthetic_face.image.shape[1],
        image_height=synthetic_face.image.shape[0],
        passed=False,
    )
    result = service.assess(synthetic_face.image, detection=empty)
    assert result.face_analysed is False


@pytest.mark.unit
def test_explicit_geometry_overrides_the_detection_result(
    service, synthetic_face
) -> None:
    box = BoundingBox(10.0, 10.0, 90.0, 90.0)
    result = service.assess(synthetic_face.image, face_box=box)
    assert result.metrics["resolution"].measurements["face_short_side"] == pytest.approx(
        80.0
    )


# --------------------------------------------------------------------------- #
# Role policy
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_role_selects_the_acceptance_threshold(service) -> None:
    image = textured()
    selfie = service.assess(image, role=ImageRole.LIVE_SELFIE)
    cnic = service.assess(image, role=ImageRole.CNIC_PORTRAIT)

    assert selfie.image_quality_score == pytest.approx(cnic.image_quality_score)
    assert selfie.min_required > cnic.min_required


@pytest.mark.unit
def test_the_same_pixels_can_pass_as_one_role_and_fail_as_another(service) -> None:
    """A CNIC portrait is a sub-300-dpi print behind a laminate; holding it to
    the live-selfie bar would reject every genuine document."""
    # Blur is the wrong lever here: it is a *critical* dimension, so past a
    # point it rejects at every role and the two bands stop being separable.
    # Exposure and contrast are non-critical by design - a dim but sharp
    # photograph is still workable - so they are what actually straddles.
    base = textured()
    candidates = []
    for factor, offset in ((0.42, 0.0), (0.34, 0.0), (0.28, 0.0), (0.22, 0.0)):
        dim = np.clip(base.astype(np.float32) * factor + offset, 0, 255).astype(np.uint8)
        candidates.append(
            (
                factor,
                service.assess(dim, role=ImageRole.LIVE_SELFIE),
                service.assess(dim, role=ImageRole.CNIC_PORTRAIT),
            )
        )

    matches = [
        (factor, selfie, portrait)
        for factor, selfie, portrait in candidates
        if portrait.usable and not selfie.usable
    ]
    assert matches, (
        "no exposure level fell between the CNIC-portrait and live-selfie "
        "thresholds. Scores: "
        + repr(
            [
                (f, s.image_quality_score, s.min_required, p.min_required, s.critical_failures)
                for f, s, p in candidates
            ]
        )
    )

    _, selfie, portrait = matches[0]
    assert selfie.image_quality_score == pytest.approx(portrait.image_quality_score)
    assert selfie.min_required > portrait.min_required
    assert not selfie.critical_failures


# --------------------------------------------------------------------------- #
# Rejection
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_an_unusable_image_reports_the_contract_error_code(service) -> None:
    """A business rejection is a *result*, not an exception - the pipeline has
    six other images to analyse."""
    destroyed = cv2.GaussianBlur(textured(), (31, 31), 0)
    result = service.assess(destroyed, role=ImageRole.LIVE_SELFIE)

    assert result.usable is False
    assert result.error_code is ErrorCode.LOW_IMAGE_QUALITY
    assert result.error_message


@pytest.mark.unit
def test_a_usable_image_carries_no_error(service) -> None:
    result = service.assess(textured(), role=ImageRole.PROFILE_IMAGE)
    assert result.usable is True
    assert result.error_code is None
    assert result.error_message is None


@pytest.mark.unit
def test_the_rejection_message_is_actionable(service) -> None:
    """"Image quality too low" sends the user round a loop they cannot exit."""
    destroyed = cv2.GaussianBlur(textured(), (31, 31), 0)
    message = service.assess(destroyed, role=ImageRole.LIVE_SELFIE).error_message

    assert message is not None
    assert any(
        word in message.lower() for word in ("focus", "steady", "still", "camera")
    )


@pytest.mark.unit
def test_assess_or_raise_raises_on_an_unusable_image(service) -> None:
    destroyed = cv2.GaussianBlur(textured(), (31, 31), 0)
    with pytest.raises(QualityRejectedError) as excinfo:
        service.assess_or_raise(destroyed, role=ImageRole.LIVE_SELFIE)

    assert excinfo.value.code is ErrorCode.LOW_IMAGE_QUALITY
    assert "limiting_factor" in excinfo.value.details


@pytest.mark.unit
def test_assess_or_raise_returns_a_good_image(service) -> None:
    assert service.assess_or_raise(textured()).usable is True


@pytest.mark.unit
def test_degraded_dimensions_lists_the_weak_ones(service) -> None:
    dark = np.clip(textured() * 0.12, 0, 255).astype(np.uint8)
    result = service.assess(dark)
    assert "brightness" in result.degraded_dimensions


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_non_bgr_array_is_rejected_with_a_clear_message(service) -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        service.assess(np.zeros((64, 64), dtype=np.uint8))


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(64, 64), (33, 400), (400, 33), (17, 17)])
def test_extreme_shapes_do_not_raise(service, shape: tuple[int, int]) -> None:
    image = textured(shape[1], shape[0], seed=5)
    result = service.assess(image)
    assert 0.0 <= result.image_quality_score <= 100.0


@pytest.mark.unit
def test_a_uniform_image_degrades_rather_than_crashing(service) -> None:
    flat = np.full((256, 256, 3), 128, dtype=np.uint8)
    result = service.assess(flat, role=ImageRole.LIVE_SELFIE)
    assert result.usable is False
    assert 0.0 <= result.image_quality_score <= 100.0


@pytest.mark.unit
def test_pure_black_and_pure_white_are_handled(service) -> None:
    for value in (0, 255):
        result = service.assess(np.full((200, 200, 3), value, dtype=np.uint8))
        assert result.usable is False


@pytest.mark.unit
def test_a_face_box_outside_the_frame_does_not_crash(service) -> None:
    result = service.assess(
        textured(200, 200), face_box=BoundingBox(180.0, 180.0, 400.0, 400.0)
    )
    assert 0.0 <= result.image_quality_score <= 100.0


@pytest.mark.unit
def test_effective_weights_sum_to_one(service) -> None:
    result = service.assess(textured())
    assert sum(result.effective_weights.values()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.unit
def test_describe_reports_the_active_configuration(service) -> None:
    described = service.describe()
    assert set(described["analysers"]) == ALL_FAMILIES
    assert described["aggregation"]["power"] < 1.0
    json.dumps(described)


# --------------------------------------------------------------------------- #
# Async surface
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_assess_async_matches_the_synchronous_path(service) -> None:
    image = textured()
    sync = service.assess(image, role=ImageRole.LIVE_SELFIE)
    async_result = asyncio.run(service.assess_async(image, role=ImageRole.LIVE_SELFIE))
    assert async_result.image_quality_score == pytest.approx(sync.image_quality_score)


@pytest.mark.unit
def test_assess_many_analyses_every_image(service) -> None:
    images = [
        (textured(seed=1), ImageRole.LIVE_SELFIE),
        (textured(seed=2), ImageRole.PROFILE_IMAGE),
        (textured(seed=3), ImageRole.SECONDARY_IMAGE),
    ]
    results = asyncio.run(service.assess_many_async(images))

    assert len(results) == 3
    assert all(isinstance(r, QualityResult) for r in results)
    assert [r.role for r in results] == [role for _, role in images]  # type: ignore[union-attr]


@pytest.mark.unit
def test_assess_many_isolates_a_single_failure(service) -> None:
    """One unreadable photo must not abort the other six."""
    images = [
        (textured(seed=1), ImageRole.PROFILE_IMAGE),
        (np.zeros((4, 4), dtype=np.uint8), ImageRole.SECONDARY_IMAGE),
        (textured(seed=3), ImageRole.SECONDARY_IMAGE),
    ]
    results = asyncio.run(service.assess_many_async(images))

    assert isinstance(results[0], QualityResult)
    assert isinstance(results[1], HamqadamError)
    assert isinstance(results[2], QualityResult)


@pytest.mark.unit
def test_assess_many_rejects_misaligned_detections(service) -> None:
    with pytest.raises(ValueError, match="positionally aligned"):
        asyncio.run(
            service.assess_many_async(
                [(textured(), ImageRole.PROFILE_IMAGE)], detections=[None, None]
            )
        )


@pytest.mark.unit
def test_concurrent_assessment_is_deterministic(service) -> None:
    """Shared state would show up as scores that vary between runs."""
    image = textured(seed=17)

    async def run_many() -> list[float]:
        results = await asyncio.gather(
            *(service.assess_async(image, role=ImageRole.LIVE_SELFIE) for _ in range(8))
        )
        return [r.image_quality_score for r in results]

    scores = asyncio.run(run_many())
    assert len(set(scores)) == 1
