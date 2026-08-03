"""MODULE 5 against a real OCR engine and rendered cards.

Every card here is synthetic and describes a wholly fictitious holder. No real
identity document appears in this repository, and none may: a CNIC is exactly
the category of personal data the service is built to avoid retaining.

The degradations are the ones a phone camera actually produces - a card lying
on a desk, poor light, laminate glare, a small capture, heavy JPEG, a photo
taken sideways. Skips cleanly when no engine is installed.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.services.ocr_service import OcrService, build_ocr_service
from tests.fixtures.synthetic_cnic import (
    CnicSpec,
    dim_lighting,
    glare,
    jpeg_artifacts,
    low_resolution,
    photograph_on_desk,
    render_cnic,
    rotated,
    sensor_noise,
)

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.integration

SPEC = CnicSpec()


@pytest.fixture(scope="module")
def service() -> OcrService:
    """The service wired from configuration, or a skip if no engine exists."""
    try:
        built = build_ocr_service()
    except Exception as exc:  # noqa: BLE001 - any engine absence is a skip
        pytest.skip(f"no OCR engine available: {exc}")
    yield built
    built.close()


@pytest.fixture(scope="module")
def clean_card() -> BgrImage:
    return render_cnic()


@pytest.fixture(scope="module")
def clean_result(service: OcrService, clean_card: BgrImage):
    """One read of the pristine card, shared across the assertions on it."""
    return service.read(clean_card)


# --------------------------------------------------------------------------- #
# The pristine card - every field, read correctly
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_a_clean_card_is_read(clean_result) -> None:
    assert clean_result.success is True
    assert clean_result.is_cnic is True
    assert clean_result.error_code is None


@pytest.mark.integration
def test_the_identity_number_is_exact(clean_result) -> None:
    """No tolerance here, deliberately. A near-miss on the identity number is
    a different person, and the backend has no way to tell that it happened."""
    assert clean_result.cnic_number == SPEC.cnic_number


@pytest.mark.integration
def test_the_names_are_read(clean_result) -> None:
    assert clean_result.name == SPEC.name
    assert clean_result.father_name == SPEC.father_name


@pytest.mark.integration
def test_every_date_is_read(clean_result) -> None:
    assert clean_result.date_of_birth == SPEC.date_of_birth
    assert clean_result.issue_date == SPEC.date_of_issue
    assert clean_result.expiry_date == SPEC.date_of_expiry


@pytest.mark.integration
def test_the_gender_glyph_is_read_not_inferred(clean_result) -> None:
    """A lone ``F`` in a wide margin is the hardest field on the card: the
    page detector finds no line there, so it is recovered by feeding the
    cropped region straight to the recognition head.

    It matters because a *read* gender is the only thing that makes the
    parity cross-check meaningful - a derived one would just be agreeing
    with itself.
    """
    assert clean_result.gender == SPEC.gender
    assert clean_result.fields["gender"].source != "derived"
    assert clean_result.gender_cross_check_passed is True


@pytest.mark.integration
def test_the_number_yields_province_and_implied_gender(clean_result) -> None:
    assert clean_result.province == "Sindh"
    assert clean_result.implied_gender == "F"


@pytest.mark.integration
def test_the_card_validates_as_internally_consistent(clean_result) -> None:
    assert clean_result.consistent is True
    assert clean_result.is_expired is False
    assert not [f for f in clean_result.findings if f.severity == "error"]


@pytest.mark.integration
def test_completeness_and_confidence_are_reported(clean_result) -> None:
    assert clean_result.completeness == pytest.approx(1.0)
    assert clean_result.ocr_confidence_score > 60.0
    assert not clean_result.fields_missing


@pytest.mark.integration
def test_the_result_serialises(clean_result) -> None:
    payload = clean_result.model_dump(mode="json")
    import json

    json.dumps(payload)
    assert payload["cnic_number"] == SPEC.cnic_number


# --------------------------------------------------------------------------- #
# Degraded captures
# --------------------------------------------------------------------------- #


DEGRADATIONS = [
    pytest.param(photograph_on_desk, id="on_a_desk"),
    pytest.param(dim_lighting, id="dim_light"),
    pytest.param(glare, id="laminate_glare"),
    pytest.param(low_resolution, id="low_resolution"),
    pytest.param(jpeg_artifacts, id="jpeg_q25"),
    pytest.param(sensor_noise, id="sensor_noise"),
]


@pytest.mark.integration
@pytest.mark.parametrize("degrade", DEGRADATIONS)
def test_the_identity_number_survives_degradation(
    service: OcrService, clean_card: BgrImage, degrade
) -> None:
    """The number is the field the backend keys on, so it is the one that has
    to survive a realistic capture rather than an ideal one."""
    result = service.read(degrade(clean_card))

    assert result.success is True
    assert result.cnic_number == SPEC.cnic_number


@pytest.mark.integration
@pytest.mark.parametrize("degrade", DEGRADATIONS)
def test_degraded_captures_stay_internally_consistent(
    service: OcrService, clean_card: BgrImage, degrade
) -> None:
    """A wrong reading usually announces itself as an inconsistency - a date
    out of order, a gender that disagrees with the number's parity. Silence
    here is meaningful."""
    result = service.read(degrade(clean_card))

    assert result.consistent is True


@pytest.mark.integration
@pytest.mark.parametrize("degrade", DEGRADATIONS)
def test_degradation_costs_completeness_or_confidence_but_not_correctness(
    service: OcrService, clean_card: BgrImage, degrade
) -> None:
    result = service.read(degrade(clean_card))

    assert result.completeness >= 0.5
    for name, expected in (
        ("date_of_birth", SPEC.date_of_birth),
        ("issue_date", SPEC.date_of_issue),
        ("expiry_date", SPEC.date_of_expiry),
    ):
        actual = getattr(result, name)
        # Absent is acceptable under degradation; wrong is not.
        assert actual is None or actual == expected


# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.parametrize("degrees", [90, 180, 270])
def test_a_card_photographed_sideways_is_still_read(
    service: OcrService, clean_card: BgrImage, degrees: int
) -> None:
    """Users hold cards whichever way is convenient, and the recogniser reads
    only upright text. The search over rotations is what covers that."""
    result = service.read(rotated(clean_card, degrees))

    assert result.success is True
    assert result.cnic_number == SPEC.cnic_number
    assert result.name == SPEC.name


@pytest.mark.integration
def test_the_applied_rotation_is_reported(
    service: OcrService, clean_card: BgrImage
) -> None:
    """So the caller can tell a rotated capture from an upright one - useful
    both for user guidance and as a fraud signal."""
    result = service.read(rotated(clean_card, 180))
    assert result.rotation_applied == 180


@pytest.mark.integration
def test_an_upright_card_costs_one_pass(
    service: OcrService, clean_card: BgrImage
) -> None:
    """Each rotation is a full recognition pass. The common case must not pay
    for the rare one."""
    result = service.read(clean_card)

    assert result.rotation_applied == 0
    assert result.orientation_attempts == 1


# --------------------------------------------------------------------------- #
# Variant cards
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_a_male_card_reads_and_cross_checks(service: OcrService) -> None:
    """The parity rule is only convincing if it works both ways round."""
    spec = CnicSpec(
        name="BILAL AHMED",
        father_name="RASHID AHMED",
        gender="M",
        cnic_number="35202-4471938-7",
    )
    result = service.read(render_cnic(spec))

    assert result.cnic_number == spec.cnic_number
    assert result.implied_gender == "M"
    assert result.province == "Punjab"
    if result.fields["gender"].source != "derived":
        assert result.gender_cross_check_passed is True


@pytest.mark.integration
def test_a_lifetime_card_has_no_expiry(service: OcrService) -> None:
    result = service.read(render_cnic(CnicSpec(lifetime=True)))

    assert result.is_lifetime is True
    assert result.expiry_date is None
    assert result.is_expired is False


@pytest.mark.integration
def test_an_expired_card_is_reported_not_rejected(service: OcrService) -> None:
    """Reading it correctly is the OCR module's job; deciding whether an
    expired document is acceptable is the rules engine's."""
    spec = CnicSpec(
        date_of_issue=dt.date(2010, 5, 4), date_of_expiry=dt.date(2020, 5, 4)
    )
    result = service.read(render_cnic(spec))

    assert result.success is True
    assert result.is_expired is True
    assert "CNIC_EXPIRED" in [f.code for f in result.findings]


@pytest.mark.integration
def test_a_tampered_gender_is_caught(service: OcrService) -> None:
    """The printed gender says male; the number's final digit is even, which
    means female. Nothing else on the card reveals this - it is the one check
    that reads two independent parts of the document against each other.
    """
    spec = CnicSpec(gender="M", cnic_number="42101-8375926-4")
    result = service.read(render_cnic(spec))

    if result.fields["gender"].source == "derived":
        pytest.skip("gender glyph not read; the cross-check needs a read value")

    assert result.gender_cross_check_passed is False
    assert result.consistent is False
    assert "CNIC_GENDER_MISMATCH" in [f.code for f in result.findings]


@pytest.mark.integration
def test_an_inconsistent_card_is_not_blamed_on_the_photograph(
    service: OcrService,
) -> None:
    """Every field was read correctly, so "retake the photograph" is advice
    that cannot work: the contradiction is in the document, and repeating the
    capture reproduces it exactly. The message has to say manual review."""
    spec = CnicSpec(gender="M", cnic_number="42101-8375926-4")
    result = service.read(render_cnic(spec))

    if result.fields["gender"].source == "derived":
        pytest.skip("gender glyph not read; the cross-check needs a read value")

    assert result.success is False
    assert result.error_message is not None
    assert "manual review" in result.error_message
    assert "Retake" not in result.error_message


# --------------------------------------------------------------------------- #
# Things that are not a CNIC
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_a_blank_image_is_reported_as_unreadable(service: OcrService) -> None:
    blank = np.full((640, 1012, 3), 245, dtype=np.uint8)
    result = service.read(blank)

    assert result.success is False
    assert result.error_code in {
        ErrorCode.CNIC_OCR_FAILED,
        ErrorCode.CNIC_NOT_RECOGNISED,
    }
    assert result.cnic_number is None


@pytest.mark.integration
def test_a_different_document_is_not_read_as_a_cnic(service: OcrService) -> None:
    """A user uploading the wrong page should be told so, rather than handed
    a confident reading of fields the document does not contain."""
    from tests.fixtures.synthetic_cnic import render_unrelated_document

    result = service.read(render_unrelated_document())

    assert result.is_cnic is False
    assert result.success is False
    assert result.error_code is ErrorCode.CNIC_NOT_RECOGNISED


@pytest.mark.integration
def test_noise_yields_no_fields(service: OcrService) -> None:
    noise = np.random.default_rng(7).integers(
        0, 256, (500, 800, 3), dtype=np.uint8
    )
    result = service.read(noise)

    assert result.cnic_number is None
    assert result.success is False


# --------------------------------------------------------------------------- #
# Provenance and privacy
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_every_field_reports_where_it_came_from(clean_result) -> None:
    """Downstream modules weigh a value differently depending on whether it
    was read from its label, matched by shape, or inferred."""
    for name in ("cnic_number", "name", "gender", "date_of_birth"):
        assert clean_result.fields[name].source


@pytest.mark.integration
def test_the_field_detail_uses_the_response_field_names(clean_result) -> None:
    """The detail map is keyed the way the response is, not the way the parser
    is - the caller should never have to learn two vocabularies for one card.
    """
    assert set(clean_result.fields) >= {
        "cnic_number", "name", "father_name", "gender",
        "date_of_birth", "issue_date", "expiry_date",
    }
    assert "full_name" not in clean_result.fields


@pytest.mark.integration
def test_raw_recognised_text_is_not_returned(clean_result) -> None:
    """The raw text of a CNIC is the holder's name and identity number. The
    parsed fields are what the caller asked for; the unfiltered transcript is
    additional personal data with no purpose."""
    payload = clean_result.model_dump(mode="json")

    assert "lines" not in payload
    assert "raw_text" not in payload
    for field in payload["fields"].values():
        assert "raw" not in field


@pytest.mark.integration
def test_the_engine_identifies_itself(clean_result) -> None:
    """Reproducibility: a field read differently after an upgrade has to be
    attributable to the version that read it."""
    assert clean_result.engine
    assert clean_result.engine_version


@pytest.mark.integration
def test_the_service_describes_its_configuration(service: OcrService) -> None:
    import json

    json.dumps(service.describe())


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_the_same_card_reads_the_same_way_twice(
    service: OcrService, clean_card: BgrImage
) -> None:
    """A verification decision that changes between two reads of one image is
    not a decision anyone can defend."""
    first = service.read(clean_card)
    second = service.read(clean_card)

    assert first.cnic_number == second.cnic_number
    assert first.name == second.name
    assert first.ocr_confidence_score == pytest.approx(
        second.ocr_confidence_score, abs=1e-6
    )
