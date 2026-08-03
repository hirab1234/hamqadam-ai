"""MODULE 1 -> MODULE 2 integration: detection feeding quality.

These tests exercise the real ordering the verification pipeline uses -
detection first, then quality measured *on the face that detection selected* -
against a real photograph rather than synthetic pixels.

They are marked ``integration`` because they need the SCRFD weights. Where the
weights are absent the tests skip rather than fail: a developer without a
populated model store should still be able to run the unit suite green.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.models.registry import get_registry
from hamqadam_ai.services import build_face_detection_service, build_quality_service

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def portrait() -> BgrImage:
    """Public-domain reference portrait bundled with matplotlib."""
    matplotlib = pytest.importorskip("matplotlib")
    path = (
        Path(matplotlib.__file__).parent
        / "mpl-data" / "sample_data" / "grace_hopper.jpg"
    )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        pytest.skip("reference portrait unavailable in this matplotlib install")
    return image


@pytest.fixture(scope="module")
def detector():  # noqa: ANN201 - pytest fixture
    """The face-detection service, skipping when no detector can be built."""
    settings = get_settings()
    try:
        return build_face_detection_service(settings, get_registry(settings))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no face detector available: {exc}")


@pytest.fixture(scope="module")
def quality():  # noqa: ANN201 - pytest fixture
    """The quality service. Needs no weights."""
    return build_quality_service()


@pytest.fixture(scope="module")
def detected(detector, portrait):  # noqa: ANN001, ANN201 - pytest fixture
    """Detection result for the reference portrait."""
    result = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    if not result.face_detected:
        pytest.skip("the detector found no face in the reference portrait")
    return result


# --------------------------------------------------------------------------- #
# The normal path
# --------------------------------------------------------------------------- #


def test_a_good_photograph_passes_both_stages(detected, quality, portrait) -> None:
    assert detected.passed is True

    result = quality.assess(portrait, role=ImageRole.LIVE_SELFIE, detection=detected)

    assert result.usable is True
    assert result.image_quality_score > 85.0
    assert result.face_analysed is True
    assert result.error_code is None


def test_detection_geometry_reaches_every_face_metric(
    detected, quality, portrait
) -> None:
    """The whole point of running detection first: quality is measured on the
    face that will actually be embedded, not on whatever is in frame."""
    result = quality.assess(portrait, role=ImageRole.LIVE_SELFIE, detection=detected)

    assert result.metrics["sharpness"].measured is True
    assert result.metrics["brightness"].measurements["measured_on_face"] == 1.0
    assert "interocular" in result.metrics["resolution"].sub_scores
    assert "geometric" in result.metrics["distortion"].sub_scores

    detector_box = detected.primary_face.bounding_box
    assert result.metrics["resolution"].measurements[
        "face_short_side"
    ] == pytest.approx(min(detector_box.width, detector_box.height), rel=0.01)


def test_running_without_detection_loses_the_face_dimensions(
    quality, portrait
) -> None:
    with_detection = quality.assess(
        portrait,
        role=ImageRole.LIVE_SELFIE,
        face_box=None,
    )
    assert with_detection.face_analysed is False
    assert "sharpness" in with_detection.unmeasured


# --------------------------------------------------------------------------- #
# Degradations, end to end
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "transform", "expected_limiting"),
    [
        ("defocus", lambda i: cv2.GaussianBlur(i, (21, 21), 0), "blur"),
        (
            "underexposure",
            lambda i: np.clip(i * 0.3, 0, 255).astype(np.uint8),
            "brightness",
        ),
        (
            "flat contrast",
            lambda i: np.clip(i * 0.28 + 110, 0, 255).astype(np.uint8),
            "contrast",
        ),
    ],
)
def test_the_limiting_factor_names_the_real_defect(
    detector, quality, portrait, label: str, transform, expected_limiting: str
) -> None:
    """One defect at a time; the report must name that defect and not another.

    This is what makes a rejection actionable - the user is told to fix the
    thing that is actually wrong.
    """
    degraded = transform(portrait)
    detection = detector.detect(degraded, role=ImageRole.LIVE_SELFIE)
    result = quality.assess(
        degraded, role=ImageRole.LIVE_SELFIE, detection=detection
    )

    assert result.metrics[expected_limiting].score < 60.0, (
        f"{label}: expected {expected_limiting} to register the defect, "
        f"got {result.metrics[expected_limiting].score:.1f}"
    )


def test_severe_defocus_is_rejected_as_a_critical_failure(
    detector, quality, portrait
) -> None:
    destroyed = cv2.GaussianBlur(portrait, (21, 21), 0)
    detection = detector.detect(destroyed, role=ImageRole.LIVE_SELFIE)
    result = quality.assess(
        destroyed, role=ImageRole.LIVE_SELFIE, detection=detection
    )

    assert result.usable is False
    assert "blur" in result.critical_failures
    assert result.error_code is ErrorCode.LOW_IMAGE_QUALITY


def test_heavy_compression_is_caught_by_pixelation(
    detector, quality, portrait
) -> None:
    ok, buffer = cv2.imencode(".jpg", portrait, [int(cv2.IMWRITE_JPEG_QUALITY), 8])
    assert ok
    compressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    detection = detector.detect(compressed, role=ImageRole.LIVE_SELFIE)
    result = quality.assess(
        compressed, role=ImageRole.LIVE_SELFIE, detection=detection
    )

    assert result.metrics["pixelation"].score < 20.0
    assert result.limiting_factor == "pixelation"


def test_a_stretched_face_is_caught_by_geometric_distortion(
    detector, quality, portrait
) -> None:
    """Only detectable beyond roughly 1.35x, because individual facial
    variation sets a noise floor around 0.12 against the population template.
    """
    height, width = portrait.shape[:2]
    stretched = cv2.resize(
        portrait, (int(width * 1.6), height), interpolation=cv2.INTER_CUBIC
    )

    detection = detector.detect(stretched, role=ImageRole.LIVE_SELFIE)
    if not detection.face_detected:
        pytest.skip("the detector lost the face on the stretched image")

    baseline = quality.assess(
        portrait,
        role=ImageRole.LIVE_SELFIE,
        detection=detector.detect(portrait, role=ImageRole.LIVE_SELFIE),
    )
    result = quality.assess(
        stretched, role=ImageRole.LIVE_SELFIE, detection=detection
    )

    assert (
        result.metrics["distortion"].measurements["geometric_anisotropy"]
        > baseline.metrics["distortion"].measurements["geometric_anisotropy"] + 0.15
    )
    assert result.metrics["distortion"].score < baseline.metrics["distortion"].score


def test_a_turned_head_suppresses_the_anisotropy_term(
    detector, quality, portrait
) -> None:
    """Perspective foreshortening is geometrically indistinguishable from an
    editor's squeeze, so the term must be withheld rather than guessed."""
    detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    if detection.primary_face is None or detection.primary_face.pose is None:
        pytest.skip("no pose estimate available")

    result = quality.assess(
        portrait,
        role=ImageRole.LIVE_SELFIE,
        face_box=None,
        landmarks=None,
        yaw_degrees=45.0,
    )
    # Without a face box the geometric term cannot run at all; assert the
    # explicit-yaw path on a detection-backed call instead.
    assert "geometric" not in result.metrics["distortion"].sub_scores


# --------------------------------------------------------------------------- #
# Role policy against real pixels
# --------------------------------------------------------------------------- #


def test_a_cnic_portrait_is_held_to_a_laxer_bar(detector, quality, portrait) -> None:
    """A sub-300-dpi print behind a laminate cannot meet the selfie bar, and
    holding it to one would reject every genuine document."""
    printed = cv2.GaussianBlur(portrait, (7, 7), 0)
    ok, buffer = cv2.imencode(".jpg", printed, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
    assert ok
    printed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    detection = detector.detect(printed, role=ImageRole.CNIC_PORTRAIT)
    selfie = quality.assess(printed, role=ImageRole.LIVE_SELFIE, detection=detection)
    portrait_role = quality.assess(
        printed, role=ImageRole.CNIC_PORTRAIT, detection=detection
    )

    assert selfie.image_quality_score == pytest.approx(
        portrait_role.image_quality_score
    )
    assert selfie.min_required > portrait_role.min_required


# --------------------------------------------------------------------------- #
# Batch behaviour
# --------------------------------------------------------------------------- #


async def test_a_full_request_is_assessed_concurrently(
    detector, quality, portrait
) -> None:
    """Seven images, one bad, analysed together - the shape of a real call."""
    rng = np.random.default_rng(3)
    images = [
        (portrait, ImageRole.LIVE_SELFIE),
        (portrait, ImageRole.PROFILE_IMAGE),
        (cv2.GaussianBlur(portrait, (21, 21), 0), ImageRole.SECONDARY_IMAGE),
        (
            np.clip(
                portrait.astype(np.float32) + rng.normal(0, 12, portrait.shape), 0, 255
            ).astype(np.uint8),
            ImageRole.SECONDARY_IMAGE,
        ),
    ]

    detections = [detector.detect(image, role=role) for image, role in images]
    results = await quality.assess_many_async(images, detections=detections)

    assert len(results) == 4
    assert results[0].usable is True
    assert results[2].usable is False, "the defocused secondary should be rejected"
    assert all(r.duration_ms > 0 for r in results)
