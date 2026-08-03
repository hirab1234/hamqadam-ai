"""MODULE 1 -> 3 -> 4 end to end, against real weights.

The unit suite pins the fusion rules against synthetic vectors. This one checks
that the rules produce sensible verdicts when fed embeddings a real model
actually produced - including the case that matters most, an impostor's
photograph presented as the profile image.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import ImageRole, MatchDecision
from hamqadam_ai.models.registry import get_registry
from hamqadam_ai.services import (
    build_embedding_service,
    build_face_detection_service,
    build_matching_service,
)
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.integration

_LANDMARK_ORDER = ("left_eye", "right_eye", "nose_tip", "mouth_left", "mouth_right")


@pytest.fixture(scope="module")
def portrait() -> BgrImage:
    """Person A."""
    matplotlib = pytest.importorskip("matplotlib")
    path = (
        Path(matplotlib.__file__).parent
        / "mpl-data" / "sample_data" / "grace_hopper.jpg"
    )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        pytest.skip("reference portrait unavailable")
    return image


@pytest.fixture(scope="module")
def other_person() -> BgrImage:
    """Person B - a genuinely different identity."""
    skimage_data = pytest.importorskip("skimage.data")
    return cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR)


@pytest.fixture(scope="module")
def pipeline():  # noqa: ANN201 - pytest fixture
    """Detection, embedding and matching, wired together."""
    settings = get_settings()
    registry = get_registry(settings)
    try:
        detector = build_face_detection_service(settings, registry)
        embedder = build_embedding_service(settings, registry)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"models unavailable: {exc}")
    return detector, embedder, build_matching_service(settings)


def embed(pipeline, image: BgrImage, role: ImageRole):  # noqa: ANN001, ANN201
    """Detect and embed one image, or return None when no face is found."""
    detector, embedder, _ = pipeline
    detection = detector.detect(image, role=role)
    if detection.primary_face is None:
        return None

    bb = detection.primary_face.bounding_box
    box = BoundingBox(bb.x1, bb.y1, bb.x2, bb.y2)
    named = {lm.name: (lm.x, lm.y) for lm in detection.primary_face.landmarks}
    marks = (
        Landmarks5(np.array([named[n] for n in _LANDMARK_ORDER], dtype=np.float32))
        if all(n in named for n in _LANDMARK_ORDER)
        else None
    )
    return embedder.embed_to_vector(image, role=role, box=box, landmarks=marks)


def jpeg(image: BgrImage, quality: int) -> BgrImage:
    """JPEG round-trip, standing in for a differently-captured upload."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_a_consistent_identity_scores_high(pipeline, portrait) -> None:  # noqa: ANN001
    """Every image is the same person, lightly varied as real uploads are."""
    _, _, matcher = pipeline

    selfie = embed(pipeline, portrait, ImageRole.LIVE_SELFIE)
    if selfie is None:
        pytest.skip("no face detected in the reference portrait")

    result = matcher.match(
        selfie=selfie,
        profile=embed(pipeline, jpeg(portrait, 85), ImageRole.PROFILE_IMAGE),
        secondaries=[
            embed(pipeline, cv2.GaussianBlur(portrait, (3, 3), 0),
                  ImageRole.SECONDARY_IMAGE),
            embed(pipeline, jpeg(portrait, 60), ImageRole.SECONDARY_IMAGE),
        ],
        cnic=embed(pipeline, jpeg(portrait, 40), ImageRole.CNIC_PORTRAIT),
    )

    assert result.identity_available is True
    assert result.identity_confidence_score is not None
    assert result.identity_confidence_score > 85.0
    assert result.any_comparison_failed is False
    assert result.cnic_identity_match is True
    assert result.capped_by is None


def test_every_specified_field_is_populated(pipeline, portrait) -> None:  # noqa: ANN001
    _, _, matcher = pipeline
    selfie = embed(pipeline, portrait, ImageRole.LIVE_SELFIE)
    if selfie is None:
        pytest.skip("no face detected")

    result = matcher.match(
        selfie=selfie,
        profile=embed(pipeline, jpeg(portrait, 85), ImageRole.PROFILE_IMAGE),
        secondaries=[embed(pipeline, jpeg(portrait, 70), ImageRole.SECONDARY_IMAGE)],
        cnic=embed(pipeline, jpeg(portrait, 45), ImageRole.CNIC_PORTRAIT),
    )

    assert result.face_match_score > 0
    assert result.profile_face_match_score is not None
    assert len(result.secondary_face_match_scores) == 1
    assert result.cnic_face_match_score is not None
    assert result.identity_confidence_score is not None
    assert result.model_version


# --------------------------------------------------------------------------- #
# The cases that must be caught
# --------------------------------------------------------------------------- #


def test_an_impostor_profile_photo_is_rejected(
    pipeline, portrait, other_person
) -> None:  # noqa: ANN001
    """Somebody else's photograph uploaded as the profile image."""
    _, _, matcher = pipeline

    selfie = embed(pipeline, portrait, ImageRole.LIVE_SELFIE)
    impostor = embed(pipeline, other_person, ImageRole.PROFILE_IMAGE)
    if selfie is None or impostor is None:
        pytest.skip("no face detected in one of the images")

    result = matcher.match(selfie=selfie, profile=impostor)

    profile = next(e for e in result.comparisons if e.comparison == "profile")
    assert profile.decision is MatchDecision.FAILED
    assert profile.score < 20.0
    assert result.any_comparison_failed is True


def test_a_stranger_s_document_caps_the_identity_confidence(
    pipeline, portrait, other_person
) -> None:  # noqa: ANN001
    """The rule that matters most.

    The user's own selfies agree with each other perfectly, but the CNIC shows
    somebody else. Internal consistency must not carry the verification.
    """
    _, _, matcher = pipeline

    selfie = embed(pipeline, portrait, ImageRole.LIVE_SELFIE)
    own_profile = embed(pipeline, jpeg(portrait, 85), ImageRole.PROFILE_IMAGE)
    stranger_cnic = embed(pipeline, other_person, ImageRole.CNIC_PORTRAIT)
    if selfie is None or own_profile is None or stranger_cnic is None:
        pytest.skip("no face detected in one of the images")

    result = matcher.match(
        selfie=selfie,
        profile=own_profile,
        secondaries=[embed(pipeline, jpeg(portrait, 70), ImageRole.SECONDARY_IMAGE)],
        cnic=stranger_cnic,
    )
    cap = get_settings().matching.identity.cnic_failure_cap

    assert result.profile_face_match_score is not None
    assert result.profile_face_match_score > 80.0, "the selfies do agree"
    assert result.cnic_identity_match is False
    assert result.identity_confidence_score is not None
    assert result.identity_confidence_score <= cap
    assert result.capped_by == "cnic_failure"


def test_one_impostor_secondary_is_surfaced(
    pipeline, portrait, other_person
) -> None:  # noqa: ANN001
    """A single mismatching secondary image is a fraud signal, and must not be
    averaged into invisibility."""
    _, _, matcher = pipeline

    selfie = embed(pipeline, portrait, ImageRole.LIVE_SELFIE)
    if selfie is None:
        pytest.skip("no face detected")

    result = matcher.match(
        selfie=selfie,
        profile=embed(pipeline, jpeg(portrait, 85), ImageRole.PROFILE_IMAGE),
        secondaries=[
            embed(pipeline, jpeg(portrait, 80), ImageRole.SECONDARY_IMAGE),
            embed(pipeline, other_person, ImageRole.SECONDARY_IMAGE),
        ],
        cnic=embed(pipeline, jpeg(portrait, 45), ImageRole.CNIC_PORTRAIT),
    )

    assert result.any_comparison_failed is True
    assert result.secondary_worst_score is not None
    assert result.secondary_worst_score < 20.0
    # The aggregate is dragged down but not to zero - one bad photo among two.
    assert len(result.secondary_face_match_scores) == 2


# --------------------------------------------------------------------------- #
# Degradation and partial input
# --------------------------------------------------------------------------- #


def test_a_degraded_but_genuine_document_still_matches(
    pipeline, portrait
) -> None:  # noqa: ANN001
    """A CNIC portrait is a low-DPI print behind a laminate; the laxer
    operating point exists so a genuine one is not refused."""
    _, _, matcher = pipeline

    selfie = embed(pipeline, portrait, ImageRole.LIVE_SELFIE)
    if selfie is None:
        pytest.skip("no face detected")

    printed = cv2.GaussianBlur(jpeg(portrait, 35), (3, 3), 0)
    cnic = embed(pipeline, printed, ImageRole.CNIC_PORTRAIT)
    if cnic is None:
        pytest.skip("no face detected in the simulated print")

    result = matcher.match(selfie=selfie, cnic=cnic)
    assert result.cnic_identity_match is True


def test_a_request_with_no_secondaries_is_not_penalised(
    pipeline, portrait
) -> None:  # noqa: ANN001
    """Secondary images are optional; their absence must not lower the score."""
    _, _, matcher = pipeline

    selfie = embed(pipeline, portrait, ImageRole.LIVE_SELFIE)
    if selfie is None:
        pytest.skip("no face detected")

    profile = embed(pipeline, jpeg(portrait, 85), ImageRole.PROFILE_IMAGE)
    cnic = embed(pipeline, jpeg(portrait, 45), ImageRole.CNIC_PORTRAIT)

    without = matcher.match(selfie=selfie, profile=profile, cnic=cnic)
    with_extras = matcher.match(
        selfie=selfie,
        profile=profile,
        secondaries=[embed(pipeline, jpeg(portrait, 75), ImageRole.SECONDARY_IMAGE)],
        cnic=cnic,
    )

    assert without.identity_confidence_score is not None
    assert with_extras.identity_confidence_score is not None
    assert without.identity_confidence_score > 85.0
    assert abs(
        without.identity_confidence_score - with_extras.identity_confidence_score
    ) < 12.0


def test_only_a_cnic_still_produces_a_confidence(pipeline, portrait) -> None:  # noqa: ANN001
    _, _, matcher = pipeline
    selfie = embed(pipeline, portrait, ImageRole.LIVE_SELFIE)
    if selfie is None:
        pytest.skip("no face detected")

    result = matcher.match(
        selfie=selfie, cnic=embed(pipeline, jpeg(portrait, 45), ImageRole.CNIC_PORTRAIT)
    )
    assert result.identity_available is True
    assert result.identity_confidence_score is not None


def test_the_measured_scores_land_where_calibration_predicts(
    pipeline, portrait, other_person
) -> None:  # noqa: ANN001
    """Ties the calibration back to real numbers.

    Module 3 measured genuine self-similarity at 0.898-0.977 and the impostor
    at 0.011. Through the calibration those must land above 90 and below 5.
    """
    _, _, matcher = pipeline

    selfie = embed(pipeline, portrait, ImageRole.LIVE_SELFIE)
    genuine = embed(pipeline, jpeg(portrait, 85), ImageRole.PROFILE_IMAGE)
    impostor = embed(pipeline, other_person, ImageRole.PROFILE_IMAGE)
    if selfie is None or genuine is None or impostor is None:
        pytest.skip("no face detected in one of the images")

    good = matcher.match(selfie=selfie, profile=genuine)
    bad = matcher.match(selfie=selfie, profile=impostor)

    assert good.profile_face_match_score is not None
    assert bad.profile_face_match_score is not None
    assert good.profile_face_match_score > 90.0
    assert bad.profile_face_match_score < 5.0
