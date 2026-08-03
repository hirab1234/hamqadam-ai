"""structlog processors: context injection, service metadata and PII redaction.

The redaction processor is the security-critical component here. It runs
immediately before serialisation, after every other processor has finished
adding fields, so nothing can slip in behind it.

Three independent mechanisms, applied in order:

1. **Drop keys** - the value is replaced wholesale. Used for anything that is
   large or entirely sensitive: image bytes, embedding vectors, API keys.
2. **Mask keys** - the value is partially masked, keeping the first and last
   two characters. Enough to correlate two log lines about the same CNIC
   without disclosing the number.
3. **Value patterns** - regexes applied to every remaining string, at any
   nesting depth. Catches a CNIC embedded in a free-text OCR dump or an
   exception message, which key-based rules alone would miss.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any, Final

from hamqadam_ai.core.config import RedactionConfig
from hamqadam_ai.core.context import context_log_fields

#: Maximum depth the redactor will walk into nested structures. A malicious or
#: buggy payload with a deeply recursive structure must not blow the stack
#: inside the logging path, which would take the whole process down.
_MAX_DEPTH: Final[int] = 8

#: Strings longer than this are truncated before being written, regardless of
#: redaction rules. Guards against a base64 image accidentally logged as text.
_MAX_STRING_LENGTH: Final[int] = 2048

#: Sequences longer than this are summarised rather than serialised in full.
_MAX_SEQUENCE_LENGTH: Final[int] = 64

EventDict = MutableMapping[str, Any]


def add_service_metadata(
    service: str, version: str, environment: str, instance: str
) -> Callable[[Any, str, EventDict], EventDict]:
    """Build a processor that stamps static service identity onto every event.

    Args:
        service: Service name.
        version: Service contract version.
        environment: Deployment environment.
        instance: Instance identifier (hostname or pod name).

    Returns:
        A structlog processor closure.
    """

    def processor(_logger: Any, _method: str, event_dict: EventDict) -> EventDict:
        event_dict.setdefault("service", service)
        event_dict.setdefault("version", version)
        event_dict.setdefault("env", environment)
        event_dict.setdefault("instance", instance)
        return event_dict

    return processor


def add_context_fields(_logger: Any, _method: str, event_dict: EventDict) -> EventDict:
    """Merge the ambient :class:`~hamqadam_ai.core.context.RequestContext`.

    Explicit keyword arguments on the call site win over context fields, so a
    nested operation can override ``verification_id`` when it genuinely differs.
    """
    for key, value in context_log_fields().items():
        event_dict.setdefault(key, value)
    return event_dict


def rename_event_key(_logger: Any, _method: str, event_dict: EventDict) -> EventDict:
    """Rename structlog's ``event`` key to ``message``.

    Log aggregators (Loki, Elasticsearch, CloudWatch) universally expect the
    human-readable line under ``message``; ``event`` collides with reserved
    fields in several of them.
    """
    if "event" in event_dict:
        event_dict["message"] = event_dict.pop("event")
    return event_dict


class RedactionProcessor:
    """Removes personally identifiable information from a log event.

    Compiled once at configuration time and then called on the hot path for
    every log line, so all rule preparation happens in ``__init__``.

    Args:
        config: The redaction section of the logging configuration.
    """

    __slots__ = ("_drop_keys", "_enabled", "_mask", "_mask_keys", "_patterns")

    def __init__(self, config: RedactionConfig) -> None:
        self._enabled = config.enabled
        self._mask = config.mask
        self._drop_keys = frozenset(key.lower() for key in config.drop_keys)
        self._mask_keys = frozenset(key.lower() for key in config.mask_keys)
        self._patterns: tuple[tuple[re.Pattern[str], str], ...] = tuple(
            (re.compile(pattern.regex), pattern.replacement) for pattern in config.patterns
        )

    def __call__(self, _logger: Any, _method: str, event_dict: EventDict) -> EventDict:
        """Redact the event in place and return it."""
        if not self._enabled:
            return event_dict
        redacted = self._walk_mapping(event_dict, depth=0)
        # structlog requires the same mutable mapping type back.
        event_dict.clear()
        event_dict.update(redacted)
        return event_dict

    # -- Internals -------------------------------------------------------- #

    def _walk_mapping(self, mapping: Mapping[str, Any], depth: int) -> dict[str, Any]:
        if depth > _MAX_DEPTH:
            return {"_truncated": "max_depth_exceeded"}
        result: dict[str, Any] = {}
        for raw_key, value in mapping.items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered in self._drop_keys:
                result[key] = self._mask
            elif lowered in self._mask_keys:
                result[key] = self._partial_mask(value)
            else:
                result[key] = self._walk_value(value, depth + 1)
        return result

    def _walk_value(self, value: Any, depth: int) -> Any:
        if depth > _MAX_DEPTH:
            return "[TRUNCATED:max_depth]"
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            return self._scrub_string(value)
        if isinstance(value, bytes):
            return f"[BYTES:{len(value)}]"
        if isinstance(value, Mapping):
            return self._walk_mapping(value, depth)
        if isinstance(value, list | tuple | set | frozenset):
            return self._walk_sequence(value, depth)
        # numpy arrays and similar: report shape, never contents. An embedding
        # is biometric data and must never be serialised into a log.
        shape = getattr(value, "shape", None)
        if shape is not None:
            return f"[ARRAY:shape={tuple(shape)},dtype={getattr(value, 'dtype', '?')}]"
        return self._scrub_string(str(value))

    def _walk_sequence(self, value: Sequence[Any] | set[Any] | frozenset[Any], depth: int) -> Any:
        items = list(value)
        if len(items) > _MAX_SEQUENCE_LENGTH:
            head = [self._walk_value(item, depth + 1) for item in items[:_MAX_SEQUENCE_LENGTH]]
            return [*head, f"[+{len(items) - _MAX_SEQUENCE_LENGTH} more]"]
        return [self._walk_value(item, depth + 1) for item in items]

    def _scrub_string(self, value: str) -> str:
        if len(value) > _MAX_STRING_LENGTH:
            value = f"{value[:_MAX_STRING_LENGTH]}...[+{len(value) - _MAX_STRING_LENGTH} chars]"
        for pattern, replacement in self._patterns:
            value = pattern.sub(replacement, value)
        return value

    def _partial_mask(self, value: Any) -> str:
        """Keep two characters at each end so lines remain correlatable."""
        text = str(value)
        if len(text) <= 6:
            return self._mask
        return f"{text[:2]}{'*' * max(3, len(text) - 4)}{text[-2:]}"


def drop_color_message_key(_logger: Any, _method: str, event_dict: EventDict) -> EventDict:
    """Discard uvicorn's ANSI-coloured duplicate of the message."""
    event_dict.pop("color_message", None)
    return event_dict


__all__ = [
    "RedactionProcessor",
    "add_context_fields",
    "add_service_metadata",
    "drop_color_message_key",
    "rename_event_key",
]
