"""Structured logging with mandatory PII redaction.

Every log line the service emits is a structured event, JSON-rendered in
production. Before serialisation each event passes through a redaction
processor that removes image payloads, embeddings and identity fields, so a
CNIC number physically cannot reach a log sink even if a developer passes one
to ``logger.info``.

Usage::

    from hamqadam_ai.logging import get_logger

    log = get_logger(__name__)
    log.info("face.detected", face_count=1, confidence=0.97)
"""

from __future__ import annotations

from hamqadam_ai.logging.processors import (
    RedactionProcessor,
    add_context_fields,
    add_service_metadata,
)
from hamqadam_ai.logging.setup import configure_logging, get_logger, is_configured

__all__ = [
    "RedactionProcessor",
    "add_context_fields",
    "add_service_metadata",
    "configure_logging",
    "get_logger",
    "is_configured",
]
