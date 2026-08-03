"""API-key authentication, HMAC signing and rate limiting.

Three layers, each solving a different problem
----------------------------------------------
**The API key** says which caller this is. Compared with
``hmac.compare_digest`` rather than ``==``, because a naive comparison returns
early on the first differing byte and leaks the key one character at a time to
anyone who can measure response times.

**The HMAC signature** says the request body has not been altered in transit
and is not a replay. Optional, because it needs shared-secret management the
Backend may not want; on when the deployment can support it.

**The rate limit** stops one caller consuming the whole inference pool. A
verification is seconds of CPU across seven images, so the limit here is not
about bandwidth - a modest request rate is enough to saturate the service.

Failing closed, and where that is wrong
---------------------------------------
Authentication fails closed: an unverifiable request is refused. Rate limiting
fails **open** when its backing store is unreachable, and that asymmetry is
deliberate. Refusing every request because Redis is down converts a
nice-to-have into a hard dependency and turns a degraded service into no
service. A brief window of un-throttled traffic is the lesser failure.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from hamqadam_ai.core.config import SecurityConfig
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    HamqadamError,
    RateLimitExceededError,
)
from hamqadam_ai.logging.setup import get_logger

log = get_logger(__name__)


def build_api_key_scheme(config: SecurityConfig) -> Any:
    """Declare API-key auth so it appears in the OpenAPI document.

    Why this exists
    ---------------
    Authentication was previously enforced by reading the header directly::

        request.headers.get(settings.security.api_key_header)

    That works - an unauthenticated request was correctly refused with 401 - but
    it is **invisible to FastAPI's schema generator**, which only records
    parameters and security schemes declared as typed dependencies. The
    consequence was a documented API that never mentioned its own
    authentication: no ``components.securitySchemes``, no ``security``
    requirement on any operation, therefore no Authorize button in Swagger UI,
    no header field, and a "Copy as cURL" that omitted the key. Every request
    sent from the documentation page failed, and the page gave no way to fix it.

    ``auto_error=False`` is deliberate. Letting FastAPI raise would return its
    own ``{"detail": ...}`` body, breaking the ``{"error": {...}}`` envelope
    every other failure in this service uses. A missing key arrives here as
    ``None`` and :func:`verify_api_key` raises the typed error instead.

    Args:
        config: The security section of the settings. The header **name** comes
            from configuration, which is why the scheme is built at startup
            rather than declared at module import: a deployment that renames the
            header must have Swagger send the renamed one.

    Returns:
        An ``APIKeyHeader`` instance to be used with ``fastapi.Security``.
    """
    from fastapi.security import APIKeyHeader

    return APIKeyHeader(
        name=config.api_key_header,
        scheme_name="ApiKeyAuth",
        description=(
            "Per-caller API key issued by the Backend. Compared in constant "
            "time; the key itself is never written to a log."
        ),
        auto_error=False,
    )


def _signature_error() -> HamqadamError:
    """Build the single response every signature failure gets.

    Absent, malformed, stale and simply wrong all return the same thing.
    Distinguishing them would tell an attacker which part of their forgery to
    fix next.
    """
    return HamqadamError(
        "The request signature is not valid.", code=ErrorCode.FORBIDDEN
    )


def verify_api_key(supplied: str | None, config: SecurityConfig) -> str:
    """Check an API key and return its fingerprint.

    Args:
        supplied: The value of the configured header, if present.
        config: The security section of the settings.

    Returns:
        A short non-reversible fingerprint of the key, for audit logging. The
        key itself never reaches a log line.

    Raises:
        AuthenticationError: when the key is missing or unrecognised.
        ConfigurationError: when authentication is required and no keys are
            configured. Deliberately an error rather than an open door - a
            deployment that forgot to set its keys should fail to start, not
            silently accept everything.
    """
    from hamqadam_ai.core.context import fingerprint_api_key

    if not config.require_api_key:
        return "anonymous"

    if not config.api_keys:
        raise ConfigurationError(
            "API-key authentication is required but no keys are configured. "
            "Set security.api_keys, or turn security.require_api_key off "
            "deliberately.",
        )

    if not supplied:
        raise AuthenticationError(
            f"Missing {config.api_key_header} header.",
        )

    # Every configured key is compared, and all of them are compared even
    # after a match, so the time taken does not reveal which key matched or
    # how many are configured.
    matched = False
    for candidate in config.api_keys:
        if hmac.compare_digest(supplied, candidate):
            matched = True

    if not matched:
        raise AuthenticationError("The supplied API key is not recognised.")

    return fingerprint_api_key(supplied)


def verify_signature(
    body: bytes,
    signature: str | None,
    timestamp: str | None,
    config: SecurityConfig,
) -> None:
    """Check an HMAC signature over the request body.

    Args:
        body: The raw request body, exactly as received.
        signature: The signature header, hex-encoded.
        timestamp: The timestamp header, Unix seconds.
        config: The security section of the settings.

    Raises:
        HamqadamError: FORBIDDEN when the signature is absent, malformed, stale or
            wrong. All four are the same answer to the caller: this request
            will not be processed. Distinguishing them in the response would
            tell an attacker which part of their forgery to fix.
    """
    hmac_config = config.hmac
    if not hmac_config.enabled:
        return

    if not hmac_config.secret:
        raise ConfigurationError(
            "HMAC signing is enabled but security.hmac.secret is not set."
        )

    if not signature or not timestamp:
        raise _signature_error()

    try:
        sent_at = float(timestamp)
    except ValueError as exc:
        raise _signature_error() from exc

    # The timestamp is inside the signed payload, so an attacker cannot replay
    # an old body under a fresh timestamp without the secret.
    skew = abs(time.time() - sent_at)
    if skew > hmac_config.max_clock_skew_seconds:
        raise _signature_error()

    expected = hmac.new(
        hmac_config.secret.encode("utf-8"),
        f"{timestamp}.".encode() + body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(signature, expected):
        raise _signature_error()


@dataclass(slots=True)
class _Bucket:
    """One caller's token bucket."""

    tokens: float
    updated: float


class TokenBucketLimiter:
    """Per-caller rate limiting, in process.

    A token bucket rather than a fixed window, because a fixed window lets a
    caller send its whole allowance in the last second of one window and again
    in the first second of the next - twice the intended rate, at the worst
    possible moment.

    In-process, which means each replica enforces the limit independently: N
    replicas allow N times the configured rate. That is a real limitation and
    the right fix is the Redis-backed limiter the configuration anticipates;
    this one is correct for a single node and is a bound rather than no bound
    for several.

    Args:
        capacity: Burst size - the most a caller may spend at once.
        refill_per_second: Sustained rate.
    """

    __slots__ = ("_buckets", "_capacity", "_lock", "_refill")

    def __init__(self, *, capacity: float, refill_per_second: float) -> None:
        self._capacity = capacity
        self._refill = refill_per_second
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, caller: str, *, cost: float = 1.0) -> None:
        """Spend from a caller's bucket.

        Args:
            caller: Key fingerprint, or another stable caller identifier.
            cost: How many tokens this request costs. A verification costs
                more than a health check because it consumes far more of the
                thing being protected.

        Raises:
            RateLimitExceededError: when the bucket cannot cover the cost.
        """
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(caller)
            if bucket is None:
                bucket = _Bucket(tokens=self._capacity, updated=now)
                self._buckets[caller] = bucket

            elapsed = now - bucket.updated
            bucket.tokens = min(
                self._capacity, bucket.tokens + elapsed * self._refill
            )
            bucket.updated = now

            if bucket.tokens < cost:
                deficit = cost - bucket.tokens
                retry_after = deficit / self._refill if self._refill > 0 else 60.0
                raise RateLimitExceededError(retry_after, scope="api")

            bucket.tokens -= cost

    def forget(self, caller: str) -> None:
        """Drop a caller's bucket. Used by tests and on key rotation."""
        with self._lock:
            self._buckets.pop(caller, None)

    @property
    def tracked_callers(self) -> int:
        """How many callers currently have a bucket."""
        with self._lock:
            return len(self._buckets)


@dataclass
class Limiters:
    """The rate limiters a deployment uses.

    Two buckets rather than one, because a health probe and a verification are
    not the same load. Sharing a bucket would let Kubernetes' liveness polling
    consume an applicant's allowance.
    """

    general: TokenBucketLimiter
    analyze: TokenBucketLimiter
    enabled: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_config(cls, config: SecurityConfig) -> Limiters:
        """Build both limiters from the rate-limit settings."""
        limit = config.rate_limit
        return cls(
            general=TokenBucketLimiter(
                capacity=float(limit.burst),
                refill_per_second=limit.requests_per_minute / 60.0,
            ),
            analyze=TokenBucketLimiter(
                capacity=max(float(limit.burst) / 2.0, 1.0),
                refill_per_second=limit.analyze_requests_per_minute / 60.0,
            ),
            enabled=limit.enabled,
        )

    def check(self, caller: str, *, analyze: bool = False) -> None:
        """Apply the appropriate limit.

        Fails open on an unexpected internal error. Refusing every request
        because the limiter itself broke would turn a protective measure into
        an outage, which is a worse failure than briefly not throttling.
        """
        if not self.enabled:
            return
        try:
            self.general.check(caller)
            if analyze:
                self.analyze.check(caller)
        except RateLimitExceededError:
            raise
        except Exception as exc:  # noqa: BLE001 - protection must not become an outage
            log.warning("api.rate_limiter_failed_open", reason=str(exc))


__all__ = ["Limiters", "TokenBucketLimiter", "verify_api_key", "verify_signature"]
