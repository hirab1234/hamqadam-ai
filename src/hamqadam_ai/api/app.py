"""The HTTP surface.

Endpoints
---------
========================================  ====================================
``POST /v1/verify``                       verify, check duplicates, enrol
``DELETE /v1/duplicate/{reference}``      erase a face from the gallery
``GET  /health``                          liveness - is the process alive
``GET  /ready``                           readiness - can it serve traffic
``GET  /metrics``                         Prometheus exposition
========================================  ====================================

Liveness and readiness are different questions
----------------------------------------------
``/health`` answers "is this process running", and returns 200 as long as it
can. ``/ready`` answers "should traffic be sent here", and returns 503 while
models are still loading or a required one failed. Conflating them is a classic
way to build a crash loop: the orchestrator restarts a pod that was merely
still warming up, and it never finishes warming up.

Errors carry codes, not stack traces
------------------------------------
Every :class:`HamqadamError` already knows its HTTP status and its stable code,
so the handler maps them mechanically. An unexpected exception becomes a
generic 500 with the request id, and the detail goes to the log rather than to
the caller - an internal traceback in an API response is an information leak
with a debugging excuse.

Images are never persisted
--------------------------
Uploads are decoded in memory and dropped when the request ends. Nothing here
writes an image to disk.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from hamqadam_ai.core.config import Settings, get_settings
from hamqadam_ai.core.context import RequestContext, pseudonymise, request_context
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import HamqadamError
from hamqadam_ai.logging.setup import configure_logging, get_logger
from hamqadam_ai.observability import record_error

log = get_logger(__name__)

#: Set once at startup and read by the routes. A module-level holder rather
#: than a global because FastAPI's dependency system needs something stable to
#: read, and the alternative - rebuilding services per request - would reload
#: several hundred megabytes of model weights on every call.
_state: dict[str, Any] = {}

#: Settings the app was created with, so ``lifespan`` uses them rather than
#: re-reading the environment.
#:
#: Without this, ``create_app(settings)`` silently ignored everything but the
#: title and version: the lifespan called ``get_settings()`` itself, so a
#: caller's API keys, thresholds and limits never reached the running app. It
#: presented as every authenticated route returning CONFIGURATION_ERROR, which
#: named the symptom and not the cause.
_configured: dict[str, Settings] = {}


def _build_state(settings: Settings) -> dict[str, Any]:
    """Construct everything the API serves, once."""
    from hamqadam_ai.api.security import Limiters
    from hamqadam_ai.observability import configure_metrics
    from hamqadam_ai.pipelines import build_pipeline

    configure_metrics(settings.observability.metrics)
    pipeline = build_pipeline(settings)

    # Publish the starting gallery size so the gauge is populated before the
    # first enrolment. Without it a freshly restarted replica reports zero
    # templates, which reads as data loss rather than as a metric not yet set.
    with contextlib.suppress(Exception):
        from hamqadam_ai.observability import set_gallery_size

        set_gallery_size(int(pipeline._services["duplicate"].gallery_size()))  # noqa: SLF001

    return {
        "settings": settings,
        "pipeline": pipeline,
        "limiters": Limiters.from_config(settings.security),
        "started_at": time.time(),
        "ready": True,
        "ready_error": None,
    }


#: Serialises recovery attempts. Sync endpoints run in Starlette's threadpool,
#: so several requests can find a broken state at the same moment; without this
#: each would rebuild the pipeline concurrently.
_recovery_lock = threading.Lock()

#: Monotonic timestamp of the last recovery attempt.
_recovery: dict[str, float] = {}

#: Minimum gap between recovery attempts. A backend that is still down should
#: cost one reconnect per interval, not one per request.
RECOVERY_COOLDOWN_SECONDS = 10.0


def recover_state() -> Any:
    """Retry a failed startup, at most once per cooldown.

    Startup builds the pipeline once, and a backend that is unreachable at that
    moment - Qdrant during a Docker restart is the common case - leaves the
    process serving ``MODEL_NOT_LOADED`` for the rest of its life. The backend
    coming back does not heal it, because nothing tries again. That is a
    restart an operator has to know to perform, prompted by an error naming a
    model rather than a socket.

    Retrying is cheap here, which is what makes this worth doing rather than
    just documenting the restart: the model registry is process-wide and holds
    its loaded weights, and the Qdrant connection is made *after* every model
    has loaded, so the second attempt reuses the sessions already in memory and
    only redials the socket.

    Returns:
        The pipeline, or None if it still cannot be built. None keeps the
        caller's existing error path intact - this widens no contract and
        introduces no fallback store.
    """
    pipeline = _state.get("pipeline")
    if pipeline is not None:
        return pipeline

    with _recovery_lock:
        # Another thread may have rebuilt it while this one waited on the lock.
        pipeline = _state.get("pipeline")
        if pipeline is not None:
            return pipeline

        now = time.monotonic()
        if now - _recovery.get("attempted_at", -RECOVERY_COOLDOWN_SECONDS) < (
            RECOVERY_COOLDOWN_SECONDS
        ):
            return None
        _recovery["attempted_at"] = now

        settings = (
            _state.get("settings") or _configured.get("settings") or get_settings()
        )
        try:
            _state.update(_build_state(settings))
        except Exception as exc:  # noqa: BLE001 - report, do not crash the server
            _state.update(
                {
                    "pipeline": None,
                    "ready": False,
                    "ready_error": f"{type(exc).__name__}: {exc}",
                }
            )
            log.warning("api.recovery_failed", reason=str(exc))
            return None

        log.info("api.recovered", note="a previously failed startup now succeeded")
        return _state.get("pipeline")


@asynccontextmanager
async def lifespan(_app: Any) -> AsyncIterator[None]:
    """Load models at startup, release them at shutdown.

    Eagerly, not lazily. Lazy loading makes the first verification on every
    cold pod seconds slower and turns a missing model into a 500 on a real
    user's request rather than a failed readiness probe.
    """
    settings = _configured.get("settings") or get_settings()
    configure_logging(settings)

    _state.clear()
    try:
        _state.update(_build_state(settings))
        log.info(
            "api.ready",
            environment=settings.app.environment,
            version=settings.app.version,
        )
    except Exception as exc:  # noqa: BLE001 - a failed start must still serve /health
        # The process stays up so the orchestrator can read /ready and see
        # *why*. A pod that exits immediately tells an operator nothing beyond
        # "it crashed".
        _state.update(
            {
                "settings": settings,
                "pipeline": None,
                "limiters": None,
                "started_at": time.time(),
                "ready": False,
                "ready_error": f"{type(exc).__name__}: {exc}",
            }
        )
        log.error(
            "api.startup_failed",
            reason=str(exc),
            note="the first request will retry; see recover_state",
        )

    yield

    pipeline = _state.get("pipeline")
    if pipeline is not None:
        for service in getattr(pipeline, "_services", {}).values():  # noqa: SLF001
            close = getattr(service, "close", None)
            if callable(close):
                # Shutdown must not raise: an exception here loses whatever
                # real error caused the shutdown in the first place.
                with contextlib.suppress(Exception):
                    close()
    _state.clear()
    log.info("api.stopped")


def create_app(settings: Settings | None = None) -> Any:
    """Build the FastAPI application.

    Args:
        settings: Configuration. Loaded from the environment when omitted.

    Returns:
        A configured ``FastAPI`` instance.
    """
    from fastapi import FastAPI, Request
    from fastapi.exceptions import (
        RequestValidationError as FastAPIRequestValidationError,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    from hamqadam_ai.api.apidocs import register_api_docs
    from hamqadam_ai.api.routes import register_routes

    resolved = settings or get_settings()
    _configured["settings"] = resolved

    app = FastAPI(
        title="Hamqadam AI Identity Verification",
        version=resolved.app.version,
        description=(
            "Face, document and fraud analysis for identity verification. "
            "Returns a recommendation; the Backend owns the decision."
        ),
        root_path=resolved.server.root_path,
        lifespan=lifespan,
        swagger_ui_parameters={
            # Keep the Authorize credential across page reloads.
            #
            # Swagger UI defaults to discarding it, and FastAPI does not
            # override that. The failure is silent and looks exactly like a
            # broken API: you authorize, the padlocks close, you reload or
            # follow a deep link to an operation, and the key is gone - the
            # padlocks reopen, the generated cURL quietly drops its
            # `-H 'X-API-Key: ...'` line, and every call returns 401 "Missing
            # X-API-Key header" while the page still looks set up.
            #
            # Persisting it puts the credential in the browser's localStorage,
            # which is acceptable for an interactive documentation page holding
            # a developer's own key and is why the production overlay serves
            # /docs to nobody by default.
            "persistAuthorization": True,
        },
    )

    if resolved.server.cors.enabled:
        cors = resolved.server.cors
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors.allow_origins,
            allow_credentials=cors.allow_credentials,
            allow_methods=cors.allow_methods,
            allow_headers=cors.allow_headers,
        )

    @app.middleware("http")
    async def bind_request_context(
        request: Request, call_next: Callable[..., Any]
    ) -> Any:
        """Give every request an id and bind it to the logging context.

        The id is echoed on the response and appears on every log line the
        request produces, which is what makes a production incident
        reconstructable from logs alone.
        """
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        context = RequestContext(request_id=request_id)

        started = time.perf_counter()
        with request_context(context):
            response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = (
            f"{(time.perf_counter() - started) * 1000:.1f}"
        )
        return response

    @app.exception_handler(HamqadamError)
    async def handle_service_error(
        request: Request, exc: HamqadamError
    ) -> JSONResponse:
        """Map a typed service error onto its HTTP status and stable code."""
        record_error(str(exc.code))
        log.info(
            "api.request_rejected",
            code=str(exc.code),
            status=exc.http_status,
            path=request.url.path,
        )
        headers: dict[str, str] = {}
        retry_after = exc.details.get("retry_after_seconds")
        if retry_after is not None:
            headers["Retry-After"] = str(int(float(retry_after)) + 1)

        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": {
                    "code": str(exc.code),
                    "message": exc.message,
                    "details": exc.details,
                    "retryable": exc.code.retryable,
                }
            },
            headers=headers,
        )

    @app.exception_handler(FastAPIRequestValidationError)
    async def handle_request_validation(
        request: Request, exc: FastAPIRequestValidationError
    ) -> JSONResponse:
        """Render FastAPI's own body validation failures in the usual envelope.

        Without this the service speaks two error dialects: every deliberate
        refusal returns ``{"error": {...}}`` with a stable code, while a
        malformed body returns FastAPI's raw ``{"detail": [...]}`` and a 422
        that appears in no error table. A client written against the documented
        contract cannot parse it, so the one failure most likely to occur during
        integration is the one it cannot read.
        """
        record_error(str(ErrorCode.VALIDATION_ERROR))
        fields = [
            {
                "field": ".".join(str(part) for part in error.get("loc", [])[1:]),
                "problem": error.get("msg", ""),
            }
            for error in exc.errors()
        ]
        log.info(
            "api.request_invalid", path=request.url.path, fields=len(fields)
        )
        return JSONResponse(
            status_code=ErrorCode.VALIDATION_ERROR.http_status,
            content={
                "error": {
                    "code": str(ErrorCode.VALIDATION_ERROR),
                    "message": "The request body did not validate.",
                    "details": {"fields": fields},
                    "retryable": False,
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Turn anything unforeseen into a generic 500.

        The detail goes to the log, not to the caller. An internal traceback in
        an API response is an information leak with a debugging excuse.
        """
        record_error(str(ErrorCode.AI_SERVICE_ERROR))
        log.error(
            "api.unhandled_error",
            error=f"{type(exc).__name__}: {exc}",
            path=request.url.path,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": str(ErrorCode.AI_SERVICE_ERROR),
                    "message": (
                        "An unexpected error occurred inside the AI "
                        "verification service."
                    ),
                    "details": {},
                    "retryable": True,
                }
            },
        )

    register_routes(app, _state)
    # After the API routes, and reading `resolved` rather than the lifespan
    # state: the reference must be readable while the service is still loading
    # models or has failed to reach a backend. That is precisely when somebody
    # is looking up what /ready means.
    register_api_docs(app, resolved)
    return app


def get_state() -> dict[str, Any]:
    """The application state, for routes and tests."""
    return _state


__all__ = [
    "RECOVERY_COOLDOWN_SECONDS",
    "create_app",
    "get_state",
    "lifespan",
    "pseudonymise",
    "recover_state",
]
