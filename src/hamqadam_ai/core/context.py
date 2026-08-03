"""Per-request execution context propagated through the whole pipeline.

Every log line, metric label and error envelope produced while handling a
request carries the same correlation identifiers without any function having to
thread them through its signature. Implemented with :mod:`contextvars`, so it
is coroutine-safe and survives ``await`` boundaries and ``asyncio.to_thread``
hand-offs alike.

Identifiers only. The context never holds image bytes, embeddings or plaintext
PII: the ``user_id`` is stored as a truncated salted digest so that even a
misconfigured log sink cannot leak it.
"""

from __future__ import annotations

import contextvars
import hashlib
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

#: Salt for the user-id digest. Set ``HQ_PSEUDONYM_SALT`` in production so that
#: digests cannot be reversed with a rainbow table over a known user-id space.
_PSEUDONYM_SALT: bytes = os.environ.get(
    "HQ_PSEUDONYM_SALT", "hamqadam-default-pseudonym-salt"
).encode("utf-8")

_DIGEST_LENGTH = 16


def pseudonymise(value: str) -> str:
    """Return a stable, non-reversible short digest of an identifier.

    Used for ``user_id`` so that operators can correlate a user's requests
    across log lines without the plaintext identifier ever being written.

    Args:
        value: The identifier to pseudonymise.

    Returns:
        A lowercase hex digest truncated to 16 characters, prefixed ``u_``.
    """
    digest = hashlib.blake2b(
        value.encode("utf-8"), key=_PSEUDONYM_SALT, digest_size=_DIGEST_LENGTH // 2
    ).hexdigest()
    return f"u_{digest}"


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Immutable correlation identifiers for one verification request.

    Attributes:
        request_id: Identifier for this HTTP call or queue message. Generated
            by the service when the caller does not supply ``X-Request-ID``.
        verification_id: The Backend's identifier for the verification attempt.
            Present in the response so results can be reconciled.
        user_pseudonym: Salted digest of the user id. Never the raw value.
        api_key_id: Short fingerprint of the calling API key, for audit.
        attempt: 1-based retry counter, incremented by the worker on redelivery.
        extra: Free-form additional labels bound to every log line.
    """

    request_id: str
    verification_id: str | None = None
    user_pseudonym: str | None = None
    api_key_id: str | None = None
    attempt: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        request_id: str | None = None,
        verification_id: str | None = None,
        user_id: str | None = None,
        api_key: str | None = None,
        attempt: int = 1,
        **extra: Any,
    ) -> RequestContext:
        """Build a context, generating a request id and hashing sensitive input.

        Args:
            request_id: Caller-supplied correlation id; generated when absent.
            verification_id: Backend verification identifier.
            user_id: Raw user identifier. Stored only as a pseudonym.
            api_key: Raw API key. Stored only as an 8-character fingerprint.
            attempt: Delivery attempt number.
            **extra: Additional labels to bind to every log line.
        """
        return cls(
            request_id=request_id or f"req_{uuid.uuid4().hex[:20]}",
            verification_id=verification_id,
            user_pseudonym=pseudonymise(user_id) if user_id else None,
            api_key_id=fingerprint_api_key(api_key) if api_key else None,
            attempt=attempt,
            extra=dict(extra),
        )

    def with_(self, **updates: Any) -> RequestContext:
        """Return a copy with the given fields replaced.

        Unknown keyword arguments are folded into :attr:`extra` rather than
        raising, so call sites can attach ad-hoc labels (``stage="detection"``)
        without the dataclass needing to know about them.
        """
        known = {key: value for key, value in updates.items() if key in _FIELD_NAMES}
        unknown = {key: value for key, value in updates.items() if key not in _FIELD_NAMES}
        if unknown:
            known["extra"] = {**self.extra, **unknown}
        return replace(self, **known)

    def as_log_fields(self) -> dict[str, Any]:
        """Render the context as the label set bound to every log event."""
        fields: dict[str, Any] = {"request_id": self.request_id}
        if self.verification_id:
            fields["verification_id"] = self.verification_id
        if self.user_pseudonym:
            fields["user"] = self.user_pseudonym
        if self.api_key_id:
            fields["api_key_id"] = self.api_key_id
        if self.attempt != 1:
            fields["attempt"] = self.attempt
        fields.update(self.extra)
        return fields


_FIELD_NAMES = frozenset(
    {"request_id", "verification_id", "user_pseudonym", "api_key_id", "attempt", "extra"}
)


def fingerprint_api_key(api_key: str) -> str:
    """Return a short, non-reversible fingerprint of an API key for audit logs."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:8]


#: The ambient context. ``None`` outside a request (start-up, CLI scripts).
_current_context: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar(
    "hamqadam_request_context", default=None
)


def current_context() -> RequestContext | None:
    """Return the context bound to the current coroutine, if any."""
    return _current_context.get()


def require_context() -> RequestContext:
    """Return the current context, synthesising a detached one if unbound.

    Library code should call this rather than :func:`current_context` when it
    unconditionally needs a request id: a background janitor sweep or a unit
    test still gets a usable, uniquely-labelled context.
    """
    context = _current_context.get()
    if context is None:
        return RequestContext.create(request_id=f"detached_{uuid.uuid4().hex[:16]}")
    return context


def bind_context(context: RequestContext) -> contextvars.Token[RequestContext | None]:
    """Bind ``context`` to the current coroutine.

    Returns:
        A token that must be passed to :func:`unbind_context` to restore the
        previous value. Prefer the :func:`request_context` context manager.
    """
    return _current_context.set(context)


def unbind_context(token: contextvars.Token[RequestContext | None]) -> None:
    """Restore the context that was active before the matching :func:`bind_context`."""
    _current_context.reset(token)


@contextmanager
def request_context(context: RequestContext) -> Iterator[RequestContext]:
    """Bind ``context`` for the duration of the ``with`` block.

    Example:
        >>> ctx = RequestContext.create(verification_id="VER-1")
        >>> with request_context(ctx):
        ...     current_context().verification_id
        'VER-1'
    """
    token = bind_context(context)
    try:
        yield context
    finally:
        unbind_context(token)


def context_log_fields() -> dict[str, Any]:
    """Return the current context's log labels, or an empty mapping."""
    context = _current_context.get()
    return context.as_log_fields() if context is not None else {}


#: Read-only view used by tests to assert nothing mutates the module state.
CONTEXT_FIELD_NAMES = MappingProxyType({name: True for name in sorted(_FIELD_NAMES)})


__all__ = [
    "CONTEXT_FIELD_NAMES",
    "RequestContext",
    "bind_context",
    "context_log_fields",
    "current_context",
    "fingerprint_api_key",
    "pseudonymise",
    "request_context",
    "require_context",
    "unbind_context",
]
