"""MODULE 7 against real weights: is this photograph usable as a profile?

Three questions the service keeps separate on purpose - is it a genuine
capture, is there one person in it, is it good enough to recognise - because
"upload a photo instead of a screenshot", "crop this so only you are in it"
and "retake this somewhere brighter" are three different things to tell a user.

Every image is a public-domain reference photograph or is synthesised. No
private individual's photograph appears in this repository.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.services import build_profile_service
from hamqadam_ai.services.profile_service import ProfileAnalysisService
from tests.fixtures.profile_images import (
    as_heavily_compressed,
    as_print_recapture,
    as_recompressed,
    as_screen_recapture,
    as_screenshot,
    as_synthetic_render,
    flat_colour,
    landscape_photo,
    reference_photo,
)

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def service() -> ProfileAnalysisService:
    try:
        return build_profile_service()
    except Exception as exc:  # noqa: BLE001 - a missing model is a skip
        pytest.skip(f"profile service unavailable: {exc}")


@pytest.fixture(scope="module")
def photo() -> BgrImage:
    image = reference_photo()
    if image is None:
        pytest.skip("no public-domain reference photograph installed")
    return image


def letterbox(image: BgrImage, pad: float = 0.25) -> BgrImage:
    rows = int(image.shape[0] * pad)
    return cv2.copyMakeBorder(
        image, rows, rows, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


# --------------------------------------------------------------------------- #
# The ordinary case
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_a_genuine_photograph_is_usable(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    result = service.analyse(photo)

    assert result.usable_as_profile is True
    assert result.authenticity_score == pytest.approx(100.0)
    assert result.face_detected is True
    assert result.face_count == 1
    assert result.error_code is None


@pytest.mark.integration
@pytest.mark.parametrize(
    "maker",
    [
        pytest.param(as_heavily_compressed, id="jpeg_q12"),
        pytest.param(as_recompressed, id="recompressed"),
        pytest.param(letterbox, id="letterboxed"),
    ],
)
def test_the_hard_negatives_stay_usable(
    service: ProfileAnalysisService, photo: BgrImage, maker
) -> None:
    """The three cases most likely to be wrongly accused. A heavily
    compressed photograph and a square-padded one are what real uploads
    look like after a messaging app has had them."""
    result = service.analyse(maker(photo))

    assert result.usable_as_profile is True
    assert not result.findings


@pytest.mark.integration
def test_the_three_questions_are_reported_separately(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    """A single merged score could tell the user none of the three things
    they might need to do."""
    result = service.analyse(photo)

    assert result.authenticity_score > 0
    assert result.face_count >= 0
    assert result.image_quality_score > 0


# --------------------------------------------------------------------------- #
# What people upload instead
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.parametrize(
    ("maker", "expected"),
    [
        pytest.param(as_screenshot, "PROFILE_IMAGE_IS_SCREENSHOT", id="screenshot"),
        pytest.param(
            as_screen_recapture, "PROFILE_IMAGE_IS_SCREEN_RECAPTURE", id="screen"
        ),
        pytest.param(
            as_print_recapture, "PROFILE_IMAGE_IS_PRINT_RECAPTURE", id="print"
        ),
    ],
)
def test_a_recaptured_image_is_refused_with_the_right_reason(
    service: ProfileAnalysisService, photo: BgrImage, maker, expected: str
) -> None:
    result = service.analyse(maker(photo))

    assert result.usable_as_profile is False
    assert result.is_probably_genuine_capture is False
    assert result.findings
    assert result.findings[0].code == expected


@pytest.mark.integration
def test_rendered_artwork_is_refused(service: ProfileAnalysisService) -> None:
    """No face to find and no meaningful quality score, so the useful thing to
    say is not "no face detected" but "that is a drawing" - which is why the
    detectors run on every image whatever else fails."""
    result = service.analyse(as_synthetic_render())

    assert result.usable_as_profile is False
    assert result.findings
    assert result.findings[0].code == "PROFILE_IMAGE_IS_SYNTHETIC"
    assert result.error_message is not None
    assert "illustration" in result.error_message


@pytest.mark.integration
def test_a_refusal_says_what_to_do_about_it(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    """Every one of these accuses somebody of uploading something dishonest.
    The least it can do is say what would fix it."""
    result = service.analyse(as_screenshot(photo))

    assert result.error_message is not None
    assert "Upload" in result.error_message


@pytest.mark.integration
def test_an_authenticity_failure_outranks_a_quality_failure(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    """Ordered by what the user should do first. Retaking the same photograph
    cannot fix a screenshot - they have to upload a different image."""
    result = service.analyse(as_print_recapture(photo))

    assert result.error_code is ErrorCode.INVALID_IMAGE
    assert result.error_message is not None
    assert "printed photo" in result.error_message


# --------------------------------------------------------------------------- #
# Subject
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_a_photograph_with_no_person_is_not_usable(
    service: ProfileAnalysisService,
) -> None:
    image = landscape_photo()
    if image is None:
        pytest.skip("no second reference photograph installed")

    result = service.analyse(image)

    assert result.usable_as_profile is False
    assert result.face_detected is False
    assert result.error_code is ErrorCode.FACE_NOT_DETECTED


@pytest.mark.integration
def test_a_photograph_with_no_person_is_still_authentic(
    service: ProfileAnalysisService,
) -> None:
    """Authenticity and subject are different questions. A landscape is a
    genuine capture; that it is unusable is Module 1's finding, not a fraud
    signal, and conflating them would accuse a user who uploaded their cat."""
    image = landscape_photo()
    if image is None:
        pytest.skip("no second reference photograph installed")

    result = service.analyse(image)

    assert result.authenticity_score == pytest.approx(100.0)
    assert not result.findings


@pytest.mark.integration
def test_no_face_and_a_screenshot_are_distinguished(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    landscape = landscape_photo()
    if landscape is None:
        pytest.skip("no second reference photograph installed")

    assert service.analyse(landscape).error_code is ErrorCode.FACE_NOT_DETECTED
    assert service.analyse(as_screenshot(photo)).error_code is ErrorCode.INVALID_IMAGE


# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.parametrize(
    "role", [ImageRole.PROFILE_IMAGE, ImageRole.SECONDARY_IMAGE, ImageRole.LIVE_SELFIE]
)
def test_authenticity_is_role_independent(
    service: ProfileAnalysisService, photo: BgrImage, role: ImageRole
) -> None:
    """A screenshot is a screenshot whatever slot it was uploaded into. Only
    the quality bar varies by role."""
    result = service.analyse(as_screenshot(photo), role=role)

    assert result.role is role
    assert result.findings
    assert result.findings[0].code == "PROFILE_IMAGE_IS_SCREENSHOT"


@pytest.mark.integration
def test_the_role_is_reported(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    result = service.analyse(photo, role=ImageRole.SECONDARY_IMAGE)
    assert result.role is ImageRole.SECONDARY_IMAGE


# --------------------------------------------------------------------------- #
# Degenerate input
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_a_flat_image_is_refused_without_a_fraud_accusation(
    service: ProfileAnalysisService,
) -> None:
    """Nothing can be measured on a uniform field, so no detector claims
    anything. It fails for the honest reason: there is no face in it."""
    result = service.analyse(flat_colour())

    assert result.usable_as_profile is False
    assert not result.findings
    assert result.error_code is ErrorCode.FACE_NOT_DETECTED


@pytest.mark.integration
def test_noise_is_refused(service: ProfileAnalysisService) -> None:
    noise = np.random.default_rng(3).integers(0, 256, (400, 500, 3), dtype=np.uint8)
    result = service.analyse(noise)

    assert result.usable_as_profile is False


@pytest.mark.integration
def test_a_non_bgr_array_is_a_programming_error(
    service: ProfileAnalysisService,
) -> None:
    with pytest.raises(ValueError, match="BGR"):
        service.analyse(np.zeros((80, 80), dtype=np.uint8))


@pytest.mark.integration
def test_a_tiny_image_does_not_crash(service: ProfileAnalysisService) -> None:
    """Below the detector's own floor, so Module 1 refuses it. The result must
    still be a populated report rather than an exception."""
    tiny = np.random.default_rng(1).integers(0, 256, (20, 24, 3), dtype=np.uint8)
    result = service.analyse(tiny)

    assert result.usable_as_profile is False


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_every_detector_reading_is_returned(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    """Including the silent ones: an engineer debugging a false negative needs
    the readings from the detectors that did not fire."""
    result = service.analyse(photo)

    assert len(result.signals) == 4
    assert all(signal.measurements for signal in result.signals)


@pytest.mark.integration
def test_the_raw_measurements_are_returned(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    """A rejection that cannot be explained cannot be appealed."""
    result = service.analyse(as_screenshot(photo))
    screenshot = next(s for s in result.signals if s.name == "screenshot")

    assert "interior_constant_rows" in screenshot.measurements


@pytest.mark.integration
def test_a_signal_says_whether_it_could_measure_at_all(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    """``measured`` is on the response model, not left to the caller to derive
    from ``note``.

    The distinction is easy to miss and it matters: a detector that looked and
    found nothing is telling you something, and one that could not run is not.
    Every caller re-deriving ``note is None`` is the drift problem in
    miniature - and the demo hitting exactly that is how this was found.
    """
    clean = service.analyse(photo)
    assert all(signal.measured for signal in clean.signals)

    flat = service.analyse(flat_colour())
    assert all(not signal.measured for signal in flat.signals)
    assert all(signal.note for signal in flat.signals)


@pytest.mark.integration
def test_measured_survives_serialisation(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    """A computed field, so it has to be in the JSON an API consumer sees."""
    payload = service.analyse(photo).model_dump(mode="json")
    assert all("measured" in signal for signal in payload["signals"])


@pytest.mark.integration
def test_findings_become_warnings_for_the_fraud_engine(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    result = service.analyse(as_screen_recapture(photo))
    codes = [warning.code for warning in result.warnings]

    assert "PROFILE_IMAGE_IS_SCREEN_RECAPTURE" in codes


@pytest.mark.integration
def test_the_result_serialises(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    import json

    payload = service.analyse(photo).model_dump(mode="json")
    json.dumps(payload)

    assert payload["usable_as_profile"] is True


@pytest.mark.integration
def test_the_summary_is_pii_free(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    import json

    summary = service.analyse(photo).summary()
    json.dumps(summary)

    assert set(summary) >= {"usable", "authenticity", "findings", "faces"}


@pytest.mark.integration
def test_the_service_disclaims_what_it_does_not_do(
    service: ProfileAnalysisService,
) -> None:
    """No content moderation, no deepfake detection, no reverse image search.
    Each needs a trained model or a reference corpus this project does not
    have, and a check that always returns False would look exactly like a
    working one while providing none of the protection.
    """
    claims = service.describe()["claims"]

    assert claims["content_moderation"] is False
    assert claims["deepfake_detection"] is False
    assert claims["reverse_image_search"] is False


@pytest.mark.integration
def test_analysis_is_deterministic(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    first = service.analyse(as_screenshot(photo))
    second = service.analyse(as_screenshot(photo))

    assert first.authenticity_score == pytest.approx(second.authenticity_score)
    assert [f.code for f in first.findings] == [f.code for f in second.findings]


@pytest.mark.integration
def test_authenticity_can_be_assessed_without_the_model_stages(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    """The cheap path: no weights, tens of milliseconds. Module 9 may want it
    for an image the identity path has already rejected."""
    assessment = service.assess_authenticity(as_screenshot(photo))

    assert assessment.clean is False
    assert assessment.strongest is not None
