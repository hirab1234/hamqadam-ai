"""One-shot structlog + stdlib logging configuration.

:func:`configure_logging` is idempotent and must be called once during
start-up, before any other subsystem emits a log line. It routes stdlib logging
(uvicorn, onnxruntime, paddleocr) through the same structlog pipeline, so a
third-party warning is redacted and JSON-rendered exactly like a first-party
event.
"""

from __future__ import annotations

import logging
import logging.config
import sys
import threading
from typing import Any

import structlog

from hamqadam_ai.core.config import LoggingConfig, Settings, get_settings
from hamqadam_ai.logging.processors import (
    RedactionProcessor,
    add_context_fields,
    add_service_metadata,
    drop_color_message_key,
    rename_event_key,
)

_configured = False
_configure_lock = threading.Lock()

_LEVEL_ORDER: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def is_configured() -> bool:
    """Return whether :func:`configure_logging` has already run."""
    return _configured


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Install the structured logging pipeline for the whole process.

    Args:
        settings: Settings to configure from. Loaded from the environment when
            omitted.
        force: Reconfigure even if already configured. Used by tests.

    The processor chain, in execution order:

    1. Merge structlog context variables bound by middleware.
    2. Add the log level and a UTC ISO-8601 timestamp.
    3. Merge the ambient :class:`~hamqadam_ai.core.context.RequestContext`.
    4. Stamp static service metadata.
    5. Format any exception into a ``exception`` string field.
    6. Add ``filename``/``lineno`` for WARNING and above.
    7. **Redact PII** - always last before rendering.
    8. Rename ``event`` to ``message`` and render.
    """
    global _configured
    with _configure_lock:
        if _configured and not force:
            return

        settings = settings or get_settings()
        config = settings.logging

        shared: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(
                fmt="iso" if config.timestamp_format == "iso" else None,
                utc=config.utc,
            ),
            add_context_fields,
            add_service_metadata(
                service=settings.app.name,
                version=settings.app.version,
                environment=settings.app.environment,
                instance=settings.app.instance_id or "unknown",
            ),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            drop_color_message_key,
            _CallSiteAdder(config.include_call_site_from),
            # Redaction is deliberately the final transform: nothing may add a
            # field after PII has been stripped.
            RedactionProcessor(config.redaction),
            rename_event_key,
        ]

        structlog.configure(
            processors=[
                *shared,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        renderer: Any = (
            structlog.processors.JSONRenderer(sort_keys=False)
            if config.renderer == "json"
            else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )

        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(formatter)

        root = logging.getLogger()
        for existing in list(root.handlers):
            root.removeHandler(existing)
        root.addHandler(handler)
        root.setLevel(_LEVEL_ORDER[config.level])

        _apply_logger_levels(config)

        _configured = True

    get_logger(__name__).info(
        "logging.configured",
        level=config.level,
        renderer=config.renderer,
        redaction_enabled=config.redaction.enabled,
        redacted_key_count=len(config.redaction.drop_keys) + len(config.redaction.mask_keys),
        pattern_count=len(config.redaction.patterns),
    )


def _apply_logger_levels(config: LoggingConfig) -> None:
    """Apply per-logger level overrides and stop them propagating twice."""
    for name, level in config.levels.items():
        logger = logging.getLogger(name)
        logger.setLevel(_LEVEL_ORDER[level])
        # Uvicorn installs its own handlers; clear them so records travel
        # through the root handler and get the structlog formatter exactly once.
        logger.handlers.clear()
        logger.propagate = True


class _CallSiteAdder:
    """Attach ``filename``/``lineno``/``func`` for events at or above a level.

    Call-site resolution walks the stack and is comparatively expensive, so it
    is skipped for INFO and DEBUG where the event name already identifies the
    code path.
    """

    __slots__ = ("_threshold", "_delegate")

    def __init__(self, threshold: str) -> None:
        self._threshold = _LEVEL_ORDER[threshold]
        self._delegate = structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        )

    def __call__(
        self, logger: Any, method: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        level_name = str(event_dict.get("level", method)).upper()
        if _LEVEL_ORDER.get(level_name, logging.INFO) >= self._threshold:
            return dict(self._delegate(logger, method, event_dict))
        return event_dict


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Safe to call at import time: structlog defers configuration lookup until
    the first log call, so a module-level ``log = get_logger(__name__)`` picks
    up whatever configuration is installed later.

    Args:
        name: Logger name, conventionally ``__name__``.
    """
    return structlog.stdlib.get_logger(name or "hamqadam")


def bind_log_context(**fields: Any) -> None:
    """Bind fields to the current coroutine's structlog context.

    Complements :class:`~hamqadam_ai.core.context.RequestContext` for values
    that are relevant to a sub-section of the pipeline only (``stage``,
    ``model_key``) and should not live in the request context proper.
    """
    structlog.contextvars.bind_contextvars(**fields)


def unbind_log_context(*keys: str) -> None:
    """Remove previously bound structlog context fields."""
    structlog.contextvars.unbind_contextvars(*keys)


def clear_log_context() -> None:
    """Clear every structlog context variable for the current coroutine."""
    structlog.contextvars.clear_contextvars()


__all__ = [
    "bind_log_context",
    "clear_log_context",
    "configure_logging",
    "get_logger",
    "is_configured",
    "unbind_log_context",
]
