"""HTTP routes.

Image transport
---------------
``/v1/verify`` accepts multipart uploads. Multipart rather than base64-in-JSON
because seven photographs base64-encode to roughly a third more bytes than they
need, and because the multipart parser streams rather than buffering the whole
body as one string before parsing it.

Every upload goes through :func:`decode_image`, which enforces the size,
format, dimension and decompression-bomb limits **before** anything is
allocated at full resolution. That ordering matters: a decompression bomb is
only a bomb if you decode it first.

Erasure is a first-class route
------------------------------
``DELETE /v1/duplicate/{reference}`` exists because the service stores
biometric templates, and one that cannot delete them on request cannot lawfully
be deployed. It is idempotent - deleting an absent reference returns 200, not
404 - because a caller retrying an erasure must not be told it failed the
second time.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, File, Form, Request, Response, Security, UploadFile
from fastapi.responses import PlainTextResponse

from hamqadam_ai.api.security import (
    build_api_key_scheme,
    verify_api_key,
    verify_signature,
)
from hamqadam_ai.core.context import RequestContext, pseudonymise, request_context
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import HamqadamError, RequestValidationError
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.schemas.verification import VerificationRequest, VerificationResult
from hamqadam_ai.utils.image_io import decode_image

log = get_logger(__name__)


def _binary(description: str) -> dict[str, Any]:
    """One optional image field, rendered by Swagger as a file picker."""
    return {"type": "string", "format": "binary", "description": description}


#: The multipart body of ``POST /v1/verify``, declared by hand.
#:
#: Hand-written because the handler parses the form itself - see the note on
#: :func:`verify`. ``format: binary`` is what makes Swagger draw a "Choose File"
#: button; an array of them draws a repeatable selector, which is what
#: ``secondary_images`` needs and what a ``str`` union destroyed.
VERIFY_FORM_SCHEMA: dict[str, Any] = {
    "schema": {
        "type": "object",
        "required": ["verification_id"],
        "properties": {
            "verification_id": {
                "type": "string",
                "description": (
                    "Your identifier for this attempt. Echoed back and used as "
                    "the log correlation id."
                ),
                "example": "ver_01HQ8XZ",
            },
            "user_reference": {
                "type": "string",
                "description": (
                    "Your account identifier. Pseudonymised before it reaches "
                    "a log and never echoed in the response. Also the key the "
                    "duplicate gallery stores against."
                ),
            },
            "enrol_on_success": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Add this face to the duplicate gallery if the result is "
                    "APPROVE. Needs user_reference; without one there is no key "
                    "to store against."
                ),
            },
            "live_selfie": _binary(
                "The biometric reference for the whole decision."
            ),
            "profile_image": _binary("The account's main photograph."),
            "cnic_image": _binary("Front of the CNIC."),
            "secondary_images": {
                "type": "array",
                "items": {"type": "string", "format": "binary"},
                "description": "Additional photographs of the same person.",
            },
        },
    }
}


def register_routes(app: Any, state: dict[str, Any]) -> None:
    """Attach every route to the application.

    Args:
        app: The FastAPI instance.
        state: The shared application state built at startup.

    Note:
        FastAPI resolves handler annotations against the **module** namespace,
        and ``from __future__ import annotations`` makes every annotation a
        string. Importing ``UploadFile`` inside this function therefore leaves
        FastAPI unable to resolve it, with an error about an invalid response
        field that names the wrong culprit. The fastapi imports stay at module
        level for that reason.
    """
    def _settings() -> Any:
        from hamqadam_ai.core.config import get_settings

        return state.get("settings") or get_settings()

    api_key_scheme = build_api_key_scheme(_settings().security)

    def _authoriser(*, analyze: bool) -> Any:
        """Build an authentication dependency.

        A **dependency**, not a helper the handlers call themselves. That
        distinction is the whole fix: FastAPI records security schemes it can
        see in a signature, and it cannot see a `request.headers.get()` buried
        in a function body. Declaring the key through `Security` is what puts
        `securitySchemes` into the OpenAPI document, which is what gives Swagger
        UI its Authorize button and its `-H` in the generated cURL.

        Args:
            analyze: Whether this route spends the expensive rate budget. A
                verification costs seconds of CPU across seven images; a health
                probe does not, and they must not share an allowance.

        Returns:
            An async dependency yielding the caller's key fingerprint.
        """

        async def dependency(
            request: Request,
            api_key: str | None = Security(api_key_scheme),
        ) -> str:
            settings = _settings()
            fingerprint = verify_api_key(api_key, settings.security)

            if settings.security.hmac.enabled:
                body = await request.body()
                verify_signature(
                    body,
                    request.headers.get(settings.security.hmac.header),
                    request.headers.get(settings.security.hmac.timestamp_header),
                    settings.security,
                )

            limiters = state.get("limiters")
            if limiters is not None:
                limiters.check(fingerprint, analyze=analyze)
            return fingerprint

        return dependency

    #: Two dependencies rather than one, so the cheap routes do not draw on the
    #: verification budget.
    authorise_analyze = _authoriser(analyze=True)
    authorise_cheap = _authoriser(analyze=False)

    def _pipeline() -> Any:
        """The pipeline, or a typed error explaining why there is not one."""
        pipeline = state.get("pipeline")
        if pipeline is None:
            raise HamqadamError(
                state.get("ready_error")
                or "The service is still starting and cannot analyse yet.",
                code=ErrorCode.MODEL_NOT_LOADED,
            )
        return pipeline

    def _publish_gallery_size(duplicate: Any) -> None:
        """Republish the template count after the gallery changes.

        Polled from the service rather than tracked incrementally, because an
        incremental counter drifts the moment anything else writes to the store
        - another replica, or an operator running the calibration script.
        """
        from hamqadam_ai.observability import set_gallery_size

        try:
            set_gallery_size(int(duplicate.gallery_size()))
        except Exception as exc:  # noqa: BLE001 - a metric must not fail a request
            log.debug("api.gallery_size_unavailable", reason=str(exc))

    def _decode(upload: Any, role: str) -> Any:
        """Decode one upload, or explain which photograph was rejected.

        Accepts a string as well as an upload, and treats it as "not supplied".
        Swagger UI ships a "Send empty value" checkbox ticked by default on
        optional file fields, and it posts those parts as an empty *string*
        rather than omitting them. Typed as `UploadFile | None`, FastAPI
        rejected the whole request with `Expected UploadFile, received: <class
        'str'>` before any handler ran - so the documentation page could not
        submit the form it had itself generated.

        An empty file part means the same thing as an absent one: the caller did
        not send that photograph.
        """
        if upload is None or isinstance(upload, str):
            return None
        data = upload.file.read()
        if not data:
            return None
        return decode_image(data, role=role).pixels

    # -- Verification --------------------------------------------------------- #

    @app.post(
        "/v1/verify",
        response_model=VerificationResult,
        summary="Run a full identity verification",
        tags=["verification"],
        openapi_extra={"requestBody": {"content": {"multipart/form-data": VERIFY_FORM_SCHEMA}}},
    )
    async def verify(
        request: Request,
        fingerprint: str = Depends(authorise_analyze),
    ) -> VerificationResult:
        """Analyse one submission and return a recommendation.

        Every image is optional at the transport layer, deliberately. A missing
        one is a finding the pipeline reports rather than a request the API
        refuses - the Backend usually still wants the analysis of whatever did
        arrive.

        Note:
            The multipart body is parsed here rather than declared as `Form`/
            `File` parameters, and the schema Swagger renders comes from
            ``openapi_extra`` above. Two problems forced that, and neither is
            solvable with plain parameter annotations:

            1. Swagger's "Send empty value" checkbox, ticked by default on
               optional file fields, posts an empty **string** for a field the
               user left blank. Declared as ``UploadFile | None`` FastAPI
               refuses the whole request with "Expected UploadFile, received:
               <class 'str'>" - so the documentation page could not submit its
               own form.
            2. Widening the annotation to ``UploadFile | str`` fixed that and
               broke something worse: Swagger renders a union as a *text box*,
               so `secondary_images` became ``array<(string | string)>`` with no
               file picker at all. The upload UI was traded for the ability to
               submit.

            Parsing the form directly gives both: the schema below declares
            honest ``format: binary`` fields, so Swagger shows real file
            pickers and a repeatable selector for `secondary_images`, while the
            parser treats an empty part exactly as an absent one.
        """
        from hamqadam_ai.pipelines import VerificationImages

        pipeline = _pipeline()
        form = await request.form()

        verification_id = str(form.get("verification_id") or "").strip()
        if not verification_id:
            raise RequestValidationError(
                "verification_id is required.",
                details={"field": "verification_id"},
            )

        raw_reference = str(form.get("user_reference") or "").strip()
        # Swagger pre-fills optional string fields with the literal word
        # "string"; treating that as a real account identifier would key the
        # gallery on it.
        user_reference = (
            None if raw_reference in {"", "string"} else raw_reference
        )
        enrol_on_success = str(
            form.get("enrol_on_success") or "false"
        ).strip().lower() in {"true", "1", "yes", "on"}

        images = VerificationImages(
            live_selfie=_decode(form.get("live_selfie"), "live_selfie"),
            profile=_decode(form.get("profile_image"), "profile_image"),
            secondaries=[
                image
                for image in (
                    _decode(upload, f"secondary_image[{index}]")
                    for index, upload in enumerate(
                        form.getlist("secondary_images")
                    )
                )
                if image is not None
            ],
            cnic=_decode(form.get("cnic_image"), "cnic_image"),
        )

        if images.count == 0:
            raise RequestValidationError(
                "No images were supplied. A verification needs at least a live "
                "selfie.",
                details={"field": "live_selfie"},
            )

        context = RequestContext(
            request_id=request.headers.get("X-Request-ID") or verification_id,
            verification_id=verification_id,
            user_pseudonym=(
                pseudonymise(user_reference) if user_reference else None
            ),
            api_key_id=fingerprint,
        )
        with request_context(context):
            result: VerificationResult = await pipeline.verify(
                VerificationRequest(
                    verification_id=verification_id,
                    user_reference=user_reference,
                    enrol_on_success=enrol_on_success,
                ),
                images,
            )
        return result

    # -- Gallery --------------------------------------------------------------- #

    @app.post(
        "/v1/duplicate/enrol",
        summary="Add a face to the duplicate gallery",
        tags=["gallery"],
    )
    async def enrol(
        _fingerprint: str = Depends(authorise_analyze),
        reference: str = Form(...),
        live_selfie: UploadFile = File(...),
    ) -> dict[str, Any]:
        """Store a face template against a reference.

        Separate from ``/v1/verify`` on purpose. Enrolling as a side effect of
        verifying would put a **rejected** applicant's face in the gallery,
        where it would match their next legitimate attempt.
        """
        from hamqadam_ai.core.constants import ImageRole

        pipeline = _pipeline()
        services = pipeline._services  # noqa: SLF001 - same package, one owner

        image = _decode(live_selfie, "live_selfie")
        if image is None:
            raise RequestValidationError("No image was supplied.")

        detection = services["detection"].detect(image, role=ImageRole.LIVE_SELFIE)
        if not detection.face_detected:
            raise HamqadamError(
                "No face was found in the supplied image, so there is nothing "
                "to enrol.",
                code=ErrorCode.FACE_NOT_DETECTED,
            )

        embedding = services["embedding"].embed_to_vector(
            image, role=ImageRole.LIVE_SELFIE, detection=detection
        )
        enrolment = services["duplicate"].enrol(embedding, reference=reference)
        _publish_gallery_size(services["duplicate"])
        payload: dict[str, Any] = enrolment.model_dump(mode="json")
        return payload

    @app.delete(
        "/v1/duplicate/{reference}",
        summary="Erase a face from the duplicate gallery",
        tags=["gallery"],
    )
    async def forget(
        reference: str,
        _fingerprint: str = Depends(authorise_cheap),
    ) -> dict[str, Any]:
        """Erase a stored template.

        Idempotent: erasing an absent reference succeeds. A caller retrying an
        erasure request must not be told it failed the second time, because
        that is how erasure requests get abandoned half-done.
        """
        pipeline = _pipeline()
        duplicate = pipeline._services["duplicate"]  # noqa: SLF001
        removed = bool(duplicate.forget(reference))
        _publish_gallery_size(duplicate)
        return {"reference": reference, "removed": removed, "erased": True}

    # -- Operational ----------------------------------------------------------- #

    @app.get("/health", summary="Liveness", tags=["operations"])
    async def health() -> dict[str, Any]:
        """Is this process alive?

        Answers yes whenever it can. A liveness probe that fails while models
        are loading gets the pod killed and restarted, forever.
        """
        settings = _settings()
        return {
            "status": "alive",
            "service": settings.app.name,
            "version": settings.app.version,
            "environment": settings.app.environment,
        }

    @app.get("/ready", summary="Readiness", tags=["operations"])
    async def ready(response: Response) -> dict[str, Any]:
        """Should traffic be sent here?

        A different question from liveness, and answering both with the same
        endpoint is how a warming-up pod ends up in a crash loop.
        """
        pipeline = state.get("pipeline")
        if pipeline is None:
            response.status_code = 503
            return {
                "status": "not_ready",
                "reason": state.get("ready_error") or "still starting",
            }

        described: dict[str, Any] = {}
        try:
            described = pipeline.describe()
        except Exception as exc:  # noqa: BLE001 - readiness must always answer
            response.status_code = 503
            return {"status": "not_ready", "reason": str(exc)}

        return {"status": "ready", "components": described}

    @app.get("/metrics", summary="Prometheus metrics", tags=["operations"])
    async def metrics() -> Any:
        """Prometheus exposition.

        Deliberately unauthenticated and deliberately separate from the API
        key: a scrape must keep working when key rotation goes wrong, which is
        exactly when the metrics matter most. Expose the port only inside the
        cluster.
        """
        settings = _settings()
        if not settings.observability.metrics.enabled:
            return PlainTextResponse(
                "metrics are disabled\n", status_code=404
            )

        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return PlainTextResponse(
            generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST
        )

    log.info("api.routes_registered", count=7)


__all__ = ["register_routes"]
