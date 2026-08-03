"""MODULE 10 end to end: the HTTP surface against real weights.

Every image is synthetic or a print-degraded public-domain reference portrait.
No real identity document and no private individual's photograph appears in
this repository.

This is the only test that exercises the whole stack as the Backend will meet
it - multipart upload, authentication, the seven-stage pipeline, the decision
engine, and the response model - so it is where the wiring defects surface.
Two were found by writing it:

* ``create_app(settings)`` silently ignored its argument. The lifespan called
  ``get_settings()`` itself, so a caller's API keys never reached the routes
  and every authenticated endpoint answered 500 CONFIGURATION_ERROR. The 500
  named the symptom; nothing named the cause.
* The route handlers' ``UploadFile`` annotations could not be resolved, because
  the fastapi imports were function-local while ``from __future__ import
  annotations`` had turned every annotation into a string.

Neither is reachable from a unit test of any single component.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import get_settings
from tests.fixtures.cnic_portrait import (
    reference_portrait,
    render_cnic_with_portrait,
)

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.integration

#: The key the test app is configured with.
API_KEY = "integration-test-key"


def _encode(image: BgrImage) -> bytes:
    """JPEG-encode, as a phone would upload."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("could not encode the fixture image")
    return bytes(buffer)


@pytest.fixture(scope="module")
def portrait() -> BgrImage:
    image = reference_portrait()
    if image is None:
        pytest.skip("no public-domain reference portrait installed")
    return image


@pytest.fixture(scope="module")
def client() -> Iterator[Any]:
    """A test client over the real application, models and all.

    Module-scoped: building the app loads several hundred megabytes of ONNX,
    and doing that per test would make this file take minutes rather than
    seconds.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from hamqadam_ai.api.app import create_app

    settings = get_settings().model_copy(deep=True)
    settings.security.require_api_key = True
    settings.security.api_keys = [API_KEY]
    settings.security.rate_limit.enabled = False

    try:
        app = create_app(settings)
    except Exception as exc:  # noqa: BLE001 - a missing model is a skip
        pytest.skip(f"application could not be built: {exc}")

    with TestClient(app) as opened:
        if opened.get("/ready").status_code != 200:
            pytest.skip("models unavailable; run scripts/download_models.py")
        yield opened


class TestOperationalEndpoints:
    """Liveness, readiness and metrics, none of which need a key."""

    def test_health_is_unauthenticated(self, client: Any) -> None:
        """A liveness probe cannot hold an API key.

        Kubernetes' probe sends no headers. Requiring one here would fail every
        probe and put the pod into a permanent restart loop.
        """
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_ready_reports_the_loaded_components(self, client: Any) -> None:
        body = client.get("/ready").json()
        assert body["status"] == "ready"
        assert body["components"]

    def test_metrics_are_scrapeable_without_a_key(self, client: Any) -> None:
        """Deliberately unauthenticated.

        A scrape must keep working when key rotation goes wrong, which is
        precisely when the metrics matter most.
        """
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "hamqadam_" in response.text

    def test_every_response_carries_a_request_id(self, client: Any) -> None:
        """What makes an incident reconstructable from logs alone."""
        response = client.get("/health")
        assert response.headers.get("X-Request-ID")
        assert response.headers.get("X-Response-Time-Ms")


class TestAuthentication:
    """The key check, over real HTTP."""

    def test_a_missing_key_is_refused(self, client: Any, portrait: BgrImage) -> None:
        response = client.post(
            "/v1/verify",
            data={"verification_id": "v-no-key"},
            files={"live_selfie": ("s.jpg", _encode(portrait), "image/jpeg")},
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"

    def test_a_wrong_key_is_refused(self, client: Any, portrait: BgrImage) -> None:
        response = client.post(
            "/v1/verify",
            headers={"X-API-Key": "not-the-key"},
            data={"verification_id": "v-bad-key"},
            files={"live_selfie": ("s.jpg", _encode(portrait), "image/jpeg")},
        )
        assert response.status_code == 401

    def test_the_refusal_leaks_nothing(
        self, client: Any, portrait: BgrImage
    ) -> None:
        """A rejection must not say how many keys exist or which one was close."""
        body = client.post(
            "/v1/verify",
            headers={"X-API-Key": "not-the-key"},
            data={"verification_id": "v-leak"},
            files={"live_selfie": ("s.jpg", _encode(portrait), "image/jpeg")},
        ).text
        assert API_KEY not in body
        assert "Traceback" not in body


class TestRequestValidation:
    """Malformed submissions, refused before any model runs."""

    def test_a_request_with_no_images_is_refused(self, client: Any) -> None:
        response = client.post(
            "/v1/verify",
            headers={"X-API-Key": API_KEY},
            data={"verification_id": "v-empty"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_a_corrupt_upload_is_refused_by_code(self, client: Any) -> None:
        """Rejected at decode, before anything is allocated at full resolution.

        That ordering is the point: a decompression bomb is only a bomb if you
        decode it first.
        """
        response = client.post(
            "/v1/verify",
            headers={"X-API-Key": API_KEY},
            data={"verification_id": "v-corrupt"},
            files={"live_selfie": ("s.jpg", b"this is not a JPEG", "image/jpeg")},
        )
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "UNSUPPORTED_IMAGE_FORMAT"

    def test_an_error_response_carries_no_traceback(self, client: Any) -> None:
        body = client.post(
            "/v1/verify",
            headers={"X-API-Key": API_KEY},
            data={"verification_id": "v-corrupt-2"},
            files={"live_selfie": ("s.jpg", b"nonsense", "image/jpeg")},
        ).json()
        assert set(body["error"]) == {"code", "message", "details", "retryable"}
        assert "Traceback" not in body["error"]["message"]


class TestFullVerification:
    """The path the Backend actually calls."""

    @pytest.fixture(scope="class")
    def approved(self, client: Any, portrait: BgrImage) -> dict[str, Any]:
        """One coherent submission: same person as selfie, profile and card."""
        card = render_cnic_with_portrait(face=portrait)
        if card is None:
            pytest.skip("no reference portrait available for the card")
        response = client.post(
            "/v1/verify",
            headers={"X-API-Key": API_KEY},
            data={"verification_id": "v-coherent", "user_reference": "acct-1"},
            files={
                "live_selfie": ("s.jpg", _encode(portrait), "image/jpeg"),
                "profile_image": ("p.jpg", _encode(portrait), "image/jpeg"),
                "cnic_image": ("c.jpg", _encode(card), "image/jpeg"),
            },
        )
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        return body

    def test_a_coherent_identity_is_approved(
        self, approved: dict[str, Any]
    ) -> None:
        assert approved["recommendation"] == "APPROVE"
        assert approved["identity_confidence_score"] is not None
        assert approved["identity_confidence_score"] >= 75.0
        assert approved["fraud_risk_score"] <= 30.0

    def test_the_response_carries_every_contracted_field(
        self, approved: dict[str, Any]
    ) -> None:
        """The Backend's contract. A missing key here is a broken integration."""
        for field in (
            "verification_id",
            "recommendation",
            "recommendation_reasons",
            "identity_confidence_score",
            "fraud_risk_score",
            "fraud_risk_level",
            "assessment_confidence",
            "stages",
            "warnings",
            "processing_time",
        ):
            assert field in approved, f"missing {field}"

    def test_every_stage_is_accounted_for(self, approved: dict[str, Any]) -> None:
        """Ran, succeeded and cost, per stage.

        A stage that silently did not run is indistinguishable from one that
        ran and found nothing, unless the response says so.
        """
        stages = {stage["stage"]: stage for stage in approved["stages"]}
        assert "selfie" in stages
        for stage in stages.values():
            assert isinstance(stage["ran"], bool)
            assert isinstance(stage["succeeded"], bool)
            if stage["ran"]:
                assert stage["duration_ms"] >= 0.0

    def test_the_decision_explains_itself(self, approved: dict[str, Any]) -> None:
        reasons = approved["recommendation_reasons"]
        assert reasons
        for reason in reasons:
            assert reason["code"]
            assert reason["message"]

    def test_a_selfie_alone_is_not_approved(
        self, client: Any, portrait: BgrImage
    ) -> None:
        """The thin-evidence defect, over HTTP.

        A single selfie once scored identity 100 / fraud 0 and was approved:
        nothing contradicted the applicant because almost nothing had been
        checked. The evidence floor now sends this to review.
        """
        response = client.post(
            "/v1/verify",
            headers={"X-API-Key": API_KEY},
            data={"verification_id": "v-thin"},
            files={"live_selfie": ("s.jpg", _encode(portrait), "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["recommendation"] != "APPROVE"
        assert body["assessment_confidence"] < 0.70

    def test_an_unreadable_cnic_degrades_rather_than_failing(
        self, client: Any, portrait: BgrImage
    ) -> None:
        """Degrade, never abort.

        A verification whose card could not be read still has a face comparison
        worth reporting. A Backend that receives an exception learns nothing
        about the images that were fine.
        """
        noise = np.random.default_rng(11).integers(
            0, 255, (400, 640, 3), dtype=np.uint8
        )
        response = client.post(
            "/v1/verify",
            headers={"X-API-Key": API_KEY},
            data={"verification_id": "v-bad-cnic"},
            files={
                "live_selfie": ("s.jpg", _encode(portrait), "image/jpeg"),
                "profile_image": ("p.jpg", _encode(portrait), "image/jpeg"),
                "cnic_image": ("c.jpg", _encode(noise), "image/jpeg"),
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["recommendation"] in {"APPROVE", "MANUAL_REVIEW", "REJECT"}
        assert body["complete"] is False or body["warnings"]

    def test_no_personal_data_is_echoed_back_verbatim(
        self, client: Any, portrait: BgrImage
    ) -> None:
        """The user reference must not reappear in the response.

        The Backend already knows which account it asked about; repeating the
        identifier only widens where it is written down.
        """
        response = client.post(
            "/v1/verify",
            headers={"X-API-Key": API_KEY},
            data={
                "verification_id": "v-privacy",
                "user_reference": "account-9f3a-secret",
            },
            files={"live_selfie": ("s.jpg", _encode(portrait), "image/jpeg")},
        )
        assert "account-9f3a-secret" not in response.text


class TestGallery:
    """Enrolment and erasure."""

    def test_enrol_then_erase_is_idempotent(
        self, client: Any, portrait: BgrImage
    ) -> None:
        """A retried erasure must not be told it failed.

        That is how erasure requests get abandoned half-done, which for
        biometric templates is a compliance failure rather than a nuisance.
        """
        headers = {"X-API-Key": API_KEY}
        enrolled = client.post(
            "/v1/duplicate/enrol",
            headers=headers,
            data={"reference": "acct-erase-me"},
            files={"live_selfie": ("s.jpg", _encode(portrait), "image/jpeg")},
        )
        assert enrolled.status_code == 200, enrolled.text

        first = client.delete("/v1/duplicate/acct-erase-me", headers=headers)
        assert first.status_code == 200
        assert first.json()["erased"] is True

        second = client.delete("/v1/duplicate/acct-erase-me", headers=headers)
        assert second.status_code == 200
        assert second.json()["erased"] is True
        assert second.json()["removed"] is False

    def test_enrolling_a_faceless_image_is_refused(self, client: Any) -> None:
        """Textured but faceless, so it reaches the detector.

        A *uniform* image never gets that far - see the test below - so using
        one here would have asserted the right outcome for the wrong reason.
        """
        noise = np.random.default_rng(3).integers(
            0, 255, (320, 320, 3), dtype=np.uint8
        )
        response = client.post(
            "/v1/duplicate/enrol",
            headers={"X-API-Key": API_KEY},
            data={"reference": "acct-no-face"},
            files={"live_selfie": ("s.jpg", _encode(noise), "image/jpeg")},
        )
        assert response.status_code >= 400
        assert response.json()["error"]["code"] == "FACE_NOT_DETECTED"

    def test_a_uniform_image_is_caught_before_the_detector(
        self, client: Any
    ) -> None:
        """Rejected at decode, with a message that names the real problem.

        "This image is a single flat colour" tells a user what to fix. Letting
        it reach the detector and answering "no face detected" is true but
        useless, and it costs a model inference to say.
        """
        blank = np.full((320, 320, 3), 200, dtype=np.uint8)
        response = client.post(
            "/v1/duplicate/enrol",
            headers={"X-API-Key": API_KEY},
            data={"reference": "acct-blank"},
            files={"live_selfie": ("s.jpg", _encode(blank), "image/jpeg")},
        )
        assert response.status_code >= 400
        assert response.json()["error"]["code"] == "IMAGE_DECODE_FAILED"

    def test_erasure_requires_authentication(self, client: Any) -> None:
        """Erasure is a write. An unauthenticated caller must not reach it."""
        assert client.delete("/v1/duplicate/anything").status_code == 401


class TestOpenApiSecurity:
    """The OpenAPI document must declare the auth the service enforces.

    Enforcement and declaration are separate things, and only one of them was
    ever true. Authentication worked - an unauthenticated request was correctly
    refused with 401 - but it was implemented by reading the header directly::

        request.headers.get(settings.security.api_key_header)

    FastAPI's schema generator records security schemes it can see in a
    signature, and it cannot see a dictionary lookup inside a function body. So
    the document contained no `securitySchemes`, no operation carried a
    `security` requirement, and Swagger UI had nothing to build from: no
    Authorize button, no header field, and a generated cURL with no `-H`. The
    documentation page could not call the API it documented.

    These tests pin the declaration, because the enforcement tests above pass
    either way and would not have caught it.
    """

    def test_the_document_declares_an_api_key_scheme(self, client: Any) -> None:
        schemes = client.get("/openapi.json").json()["components"][
            "securitySchemes"
        ]
        assert "ApiKeyAuth" in schemes
        assert schemes["ApiKeyAuth"]["type"] == "apiKey"
        assert schemes["ApiKeyAuth"]["in"] == "header"

    def test_the_declared_header_matches_the_enforced_one(
        self, client: Any
    ) -> None:
        """Swagger must send the header the service actually reads.

        Both come from `security.api_key_header`. Hard-coding either would let
        a renamed header work in curl and fail in the documentation, or vice
        versa.
        """
        declared = client.get("/openapi.json").json()["components"][
            "securitySchemes"
        ]["ApiKeyAuth"]["name"]
        assert declared == get_settings().security.api_key_header

    @pytest.mark.parametrize(
        ("path", "method"),
        [
            ("/v1/verify", "post"),
            ("/v1/duplicate/enrol", "post"),
            ("/v1/duplicate/{reference}", "delete"),
        ],
    )
    def test_every_protected_route_requires_the_scheme(
        self, client: Any, path: str, method: str
    ) -> None:
        operation = client.get("/openapi.json").json()["paths"][path][method]
        assert operation.get("security") == [{"ApiKeyAuth": []}]

    @pytest.mark.parametrize("path", ["/health", "/ready", "/metrics"])
    def test_operational_routes_declare_no_requirement(
        self, client: Any, path: str
    ) -> None:
        """A probe cannot carry a key, so it must not be documented as needing one."""
        operation = client.get("/openapi.json").json()["paths"][path]["get"]
        assert "security" not in operation


class TestSwaggerFormQuirks:
    """Swagger's own generated form must be submittable."""

    def test_empty_file_parts_are_treated_as_absent(self, client: Any) -> None:
        """Swagger ticks "Send empty value" on optional file fields by default.

        It then posts those parts as an empty *string*, not as an omitted field.
        Typed `UploadFile | None`, FastAPI rejected the whole request with
        `Expected UploadFile, received: <class 'str'>` before any handler ran -
        so the documentation page could not submit the form it had generated.
        """
        response = client.post(
            "/v1/verify",
            headers={"X-API-Key": API_KEY},
            data={
                "verification_id": "v-empty-parts",
                "live_selfie": "",
                "profile_image": "",
                "cnic_image": "",
            },
        )
        # Reaches the handler and gets the real, actionable refusal rather than
        # a schema error about types the caller never chose.
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"
        assert "live selfie" in response.json()["error"]["message"]

    def test_a_real_upload_still_works_alongside_empty_parts(
        self, client: Any, portrait: BgrImage
    ) -> None:
        """The mixed case: one file chosen, the rest left empty."""
        response = client.post(
            "/v1/verify",
            headers={"X-API-Key": API_KEY},
            data={"verification_id": "v-mixed", "cnic_image": ""},
            files={"live_selfie": ("s.jpg", _encode(portrait), "image/jpeg")},
        )
        assert response.status_code == 200

    def test_body_validation_errors_use_the_standard_envelope(
        self, client: Any
    ) -> None:
        """FastAPI's own 422 must not escape in its native shape.

        Otherwise the service speaks two error dialects, and the one a client
        meets first during integration is the one it cannot parse.
        """
        response = client.post("/v1/verify", headers={"X-API-Key": API_KEY})
        assert "error" in response.json()
        assert "detail" not in response.json()
        assert set(response.json()["error"]) == {
            "code",
            "message",
            "details",
            "retryable",
        }
