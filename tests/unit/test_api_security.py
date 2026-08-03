"""API-key checking, HMAC signing and rate limiting.

Pure functions over configuration, so they need no models and no fixtures
beyond a settings object. Which is the point: authentication that can only be
tested by standing up the whole service does not get tested.
"""

from __future__ import annotations

import hashlib
import hmac
import time

import pytest
from pydantic import ValidationError

from hamqadam_ai.api.security import (
    Limiters,
    TokenBucketLimiter,
    verify_api_key,
    verify_signature,
)
from hamqadam_ai.core.config import (
    HmacConfig,
    RateLimitConfig,
    SecurityConfig,
)
from hamqadam_ai.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    HamqadamError,
    RateLimitExceededError,
)


class TestApiKey:
    """Key authentication."""

    def test_valid_key_returns_a_fingerprint(self) -> None:
        config = SecurityConfig(require_api_key=True, api_keys=["secret-alpha"])
        fingerprint = verify_api_key("secret-alpha", config)
        assert fingerprint
        # The fingerprint may be logged, so the key itself must not survive
        # inside it.
        assert "secret-alpha" not in fingerprint

    def test_unknown_key_is_refused(self) -> None:
        config = SecurityConfig(require_api_key=True, api_keys=["secret-alpha"])
        with pytest.raises(AuthenticationError):
            verify_api_key("secret-beta", config)

    def test_missing_key_is_refused(self) -> None:
        config = SecurityConfig(require_api_key=True, api_keys=["secret-alpha"])
        with pytest.raises(AuthenticationError):
            verify_api_key(None, config)

    def test_required_but_unconfigured_is_an_error_not_an_open_door(self) -> None:
        """The failure mode that matters most.

        A deployment that turns authentication on and forgets to set any keys
        must fail loudly. Treating "no keys configured" as "accept everything"
        would silently expose the service, and it would look healthy while
        doing it.
        """
        config = SecurityConfig(require_api_key=True, api_keys=[])
        with pytest.raises(ConfigurationError):
            verify_api_key("anything", config)

    def test_authentication_can_be_disabled_deliberately(self) -> None:
        config = SecurityConfig(require_api_key=False, api_keys=[])
        assert verify_api_key(None, config) == "anonymous"

    def test_several_keys_are_all_accepted(self) -> None:
        """Rotation needs an overlap window where both keys work."""
        config = SecurityConfig(
            require_api_key=True, api_keys=["old-key", "new-key"]
        )
        assert verify_api_key("old-key", config)
        assert verify_api_key("new-key", config)


class TestSignature:
    """HMAC body signing."""

    @staticmethod
    def _config(secret: str = "shared-secret") -> SecurityConfig:
        return SecurityConfig(hmac=HmacConfig(enabled=True, secret=secret))

    @staticmethod
    def _sign(body: bytes, secret: str, timestamp: str) -> str:
        return hmac.new(
            secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256
        ).hexdigest()

    def test_a_correct_signature_passes(self) -> None:
        body = b'{"verification_id":"v-1"}'
        stamp = str(int(time.time()))
        config = self._config()
        verify_signature(
            body, self._sign(body, "shared-secret", stamp), stamp, config
        )

    def test_a_tampered_body_fails(self) -> None:
        stamp = str(int(time.time()))
        config = self._config()
        signature = self._sign(b"original", "shared-secret", stamp)
        with pytest.raises(HamqadamError):
            verify_signature(b"tampered", signature, stamp, config)

    def test_a_replayed_request_fails_once_it_is_stale(self) -> None:
        """The timestamp is inside the signed payload, which is what stops replay.

        An attacker who captures a valid request cannot re-send it after the
        skew window: changing the timestamp invalidates the signature, and
        keeping it makes the request stale.
        """
        body = b"payload"
        stale = str(int(time.time()) - 3600)
        config = self._config()
        signature = self._sign(body, "shared-secret", stale)
        with pytest.raises(HamqadamError):
            verify_signature(body, signature, stale, config)

    @pytest.mark.parametrize(
        ("signature", "timestamp"),
        [
            (None, "1700000000"),
            ("abc", None),
            ("not-hex", "1700000000"),
            ("abc", "not-a-number"),
        ],
    )
    def test_every_malformed_case_gives_the_same_answer(
        self, signature: str | None, timestamp: str | None
    ) -> None:
        """Absent, malformed, stale and wrong are one response.

        Distinguishing them would tell an attacker which part of their forgery
        to fix next, which is a free oracle.
        """
        config = self._config()
        with pytest.raises(HamqadamError) as caught:
            verify_signature(b"body", signature, timestamp, config)
        assert "not valid" in str(caught.value)

    def test_enabled_without_a_secret_cannot_even_be_configured(self) -> None:
        """Caught at construction, which is better than at first request.

        `verify_signature` also checks, but it can never see this state through
        a normally loaded configuration: the model refuses to validate. That
        means the deployment fails at startup rather than serving traffic until
        the first signed request arrives and 500s.
        """
        with pytest.raises(ValidationError, match="secret must be set"):
            HmacConfig(enabled=True, secret=None)

    def test_the_runtime_check_still_holds_if_the_object_is_mutated(self) -> None:
        """Defence in depth for the path validation cannot reach.

        Unreachable through configuration loading, reachable if something
        mutates the object in process. A signature check that silently passes
        because the secret went missing is the one failure this must not have.
        """
        config = SecurityConfig(hmac=HmacConfig(enabled=True, secret="x"))
        object.__setattr__(config.hmac, "secret", None)
        with pytest.raises(ConfigurationError):
            verify_signature(b"body", "abc", str(int(time.time())), config)

    def test_disabled_signing_checks_nothing(self) -> None:
        config = SecurityConfig(hmac=HmacConfig(enabled=False))
        verify_signature(b"body", None, None, config)


class TestTokenBucket:
    """Rate limiting."""

    def test_a_burst_is_allowed_then_refused(self) -> None:
        limiter = TokenBucketLimiter(capacity=3.0, refill_per_second=1.0)
        for _ in range(3):
            limiter.check("caller")
        with pytest.raises(RateLimitExceededError):
            limiter.check("caller")

    def test_tokens_refill_over_time(self) -> None:
        limiter = TokenBucketLimiter(capacity=2.0, refill_per_second=100.0)
        limiter.check("caller")
        limiter.check("caller")
        time.sleep(0.05)  # 100/s for 50 ms is 5 tokens, capped at capacity
        limiter.check("caller")

    def test_callers_do_not_share_a_bucket(self) -> None:
        """One noisy caller must not exhaust everyone else's allowance."""
        limiter = TokenBucketLimiter(capacity=1.0, refill_per_second=0.001)
        limiter.check("noisy")
        with pytest.raises(RateLimitExceededError):
            limiter.check("noisy")
        limiter.check("quiet")

    def test_the_error_says_when_to_retry(self) -> None:
        limiter = TokenBucketLimiter(capacity=1.0, refill_per_second=0.5)
        limiter.check("caller")
        with pytest.raises(RateLimitExceededError) as caught:
            limiter.check("caller")
        retry_after = caught.value.details.get("retry_after_seconds")
        assert retry_after is not None
        # A caller told only "too many requests" retries immediately, which is
        # how a rate limit turns into a hot loop.
        assert 0.0 < float(retry_after) <= 3.0

    def test_a_costly_request_spends_more(self) -> None:
        limiter = TokenBucketLimiter(capacity=5.0, refill_per_second=0.001)
        limiter.check("caller", cost=4.0)
        with pytest.raises(RateLimitExceededError):
            limiter.check("caller", cost=2.0)


class TestLimiters:
    """The two-bucket arrangement the API uses."""

    def test_verification_and_probe_budgets_are_separate(self) -> None:
        """Health polling must not consume an applicant's allowance.

        Kubernetes polls liveness every few seconds forever. Sharing one bucket
        with verification would let the orchestrator's own probes rate-limit
        real users.
        """
        limiters = Limiters.from_config(
            SecurityConfig(
                rate_limit=RateLimitConfig(
                    enabled=True,
                    requests_per_minute=600,
                    analyze_requests_per_minute=1,
                    # General capacity is the burst; analyze is half of it,
                    # floored at one. burst=10 therefore gives 10 general
                    # tokens and 5 analyze ones, so the analyze budget is the
                    # binding constraint with general capacity still to spare -
                    # which is the whole property under test.
                    burst=10,
                )
            )
        )
        for _ in range(5):
            limiters.check("caller", analyze=True)

        with pytest.raises(RateLimitExceededError):
            limiters.check("caller", analyze=True)

        # The general bucket is checked first, so the refused verification
        # spent a general token on its way to being refused - six spent of ten.
        # Probes keep working on what is left, which is the point: an exhausted
        # verification budget must not take liveness polling down with it.
        limiters.check("caller", analyze=False)

    def test_disabled_limiting_permits_everything(self) -> None:
        limiters = Limiters.from_config(
            SecurityConfig(rate_limit=RateLimitConfig(enabled=False, burst=1))
        )
        for _ in range(50):
            limiters.check("caller", analyze=True)

    def test_a_broken_limiter_fails_open(self) -> None:
        """Deliberate asymmetry: authentication fails closed, throttling opens.

        Refusing every request because the limiter itself broke converts a
        protective measure into an outage - a worse failure than briefly not
        throttling.
        """

        class Broken:
            def check(self, caller: str, *, cost: float = 1.0) -> None:
                raise RuntimeError("backing store unreachable")

        limiters = Limiters(
            general=Broken(),  # type: ignore[arg-type]
            analyze=Broken(),  # type: ignore[arg-type]
            enabled=True,
        )
        limiters.check("caller", analyze=True)
