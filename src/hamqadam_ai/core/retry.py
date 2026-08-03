"""Retry, timeout and circuit-breaking primitives.

Only *transient* failures are retried. Retrying a malformed image or an
occluded face wastes latency budget and produces the identical answer, so the
predicate keys off :attr:`~hamqadam_ai.core.errors.ErrorCode.retryable` rather
than on exception type. Every retry is jittered to avoid synchronised
thundering herds when a shared dependency recovers.
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

from hamqadam_ai.core.exceptions import HamqadamError, PipelineTimeoutError

P = ParamSpec("P")
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    Attributes:
        attempts: Total number of tries, including the first. ``1`` disables
            retrying entirely.
        base_delay: Delay before the second attempt, in seconds.
        max_delay: Ceiling applied to the exponential growth.
        multiplier: Growth factor per attempt.
        jitter: When true the actual sleep is drawn uniformly from
            ``[0, computed_delay]`` (AWS "full jitter"), which decorrelates
            clients far better than a fixed +/- percentage.
        retry_on: Extra exception types treated as transient regardless of
            what their error code says.
    """

    attempts: int = 3
    base_delay: float = 0.25
    max_delay: float = 5.0
    multiplier: float = 2.0
    jitter: bool = True
    retry_on: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError, OSError)

    def delay_for(self, attempt: int) -> float:
        """Return the sleep duration before ``attempt`` (1-based, >= 2)."""
        raw = min(self.base_delay * (self.multiplier ** (attempt - 2)), self.max_delay)
        if not self.jitter:
            return raw
        return random.uniform(0.0, raw)  # noqa: S311 - jitter, not cryptography

    def should_retry(self, exc: BaseException) -> bool:
        """Decide whether ``exc`` is worth another attempt."""
        if isinstance(exc, HamqadamError):
            return exc.retryable
        return isinstance(exc, self.retry_on)


#: Sensible defaults for the two dependency classes the service talks to.
NETWORK_RETRY = RetryPolicy(attempts=3, base_delay=0.2, max_delay=2.0)
INFERENCE_RETRY = RetryPolicy(attempts=2, base_delay=0.05, max_delay=0.5)
NO_RETRY = RetryPolicy(attempts=1)


class RetryExhaustedError(HamqadamError):
    """Every attempt permitted by the policy failed.

    The final underlying exception is preserved as ``__cause__`` and its error
    code is adopted, so a caller that switches on ``code`` sees the real reason
    rather than a generic wrapper.
    """

    def __init__(self, operation: str, attempts: int, last: BaseException) -> None:
        code = last.code if isinstance(last, HamqadamError) else None
        super().__init__(
            f"Operation {operation!r} failed after {attempts} attempt(s): {last}",
            code=code,
            details={"operation": operation, "attempts": attempts},
            cause=last,
        )


def with_retry(
    policy: RetryPolicy = NETWORK_RETRY,
    *,
    operation: str | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorate a synchronous callable with the given retry policy.

    Args:
        policy: The backoff schedule and retry predicate.
        operation: Label used in errors and in the ``on_retry`` callback.
            Defaults to the wrapped function's qualified name.
        on_retry: Invoked as ``(attempt, exception, sleep_seconds)`` before each
            sleep. The natural place to emit a log line or bump a counter.

    Returns:
        The decorated callable, raising :class:`RetryExhaustedError` when every
        attempt fails.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        label = operation or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last: BaseException | None = None
            for attempt in range(1, policy.attempts + 1):
                try:
                    return func(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001 - re-raised below
                    last = exc
                    if attempt >= policy.attempts or not policy.should_retry(exc):
                        break
                    sleep_for = policy.delay_for(attempt + 1)
                    if on_retry is not None:
                        on_retry(attempt, exc, sleep_for)
                    time.sleep(sleep_for)
            assert last is not None  # noqa: S101 - unreachable unless attempts < 1
            if policy.attempts == 1 or not policy.should_retry(last):
                raise last
            raise RetryExhaustedError(label, policy.attempts, last) from last

        return wrapper

    return decorator


def with_async_retry(
    policy: RetryPolicy = NETWORK_RETRY,
    *,
    operation: str | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Asynchronous counterpart of :func:`with_retry`.

    ``asyncio.CancelledError`` is never retried: cancellation is a control-flow
    signal from the event loop, not a transient dependency failure.
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        label = operation or func.__qualname__

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last: BaseException | None = None
            for attempt in range(1, policy.attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:  # noqa: BLE001 - re-raised below
                    last = exc
                    if attempt >= policy.attempts or not policy.should_retry(exc):
                        break
                    sleep_for = policy.delay_for(attempt + 1)
                    if on_retry is not None:
                        on_retry(attempt, exc, sleep_for)
                    await asyncio.sleep(sleep_for)
            assert last is not None  # noqa: S101
            if policy.attempts == 1 or not policy.should_retry(last):
                raise last
            raise RetryExhaustedError(label, policy.attempts, last) from last

        return wrapper

    return decorator


async def run_with_timeout(
    awaitable: Awaitable[T],
    timeout_seconds: float,
    *,
    stage: str,
) -> T:
    """Await ``awaitable`` under a deadline, converting expiry to a typed error.

    Args:
        awaitable: The coroutine or future to await.
        timeout_seconds: Wall-clock budget.
        stage: Pipeline stage name, reported in the error details so an
            operator can see *which* step blew the budget.

    Raises:
        PipelineTimeoutError: if the deadline elapses first.
    """
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except TimeoutError as exc:
        raise PipelineTimeoutError(
            f"Stage {stage!r} exceeded its {timeout_seconds:.1f}s budget.",
            details={"stage": stage, "timeout_seconds": timeout_seconds},
            cause=exc,
        ) from exc


async def gather_resilient(
    tasks: Iterable[Awaitable[T]],
    *,
    stage: str,
    timeout_seconds: float | None = None,
) -> list[T | HamqadamError]:
    """Run awaitables concurrently, returning failures inline instead of raising.

    The verification pipeline analyses up to seven images. One unreadable
    secondary photo must not abort the whole request: the caller receives the
    successful results plus a typed error in the failed slot and decides what
    that means for the final recommendation.

    Args:
        tasks: The awaitables to run concurrently.
        stage: Stage label used when a task times out.
        timeout_seconds: Optional per-task deadline.

    Returns:
        A list positionally aligned with ``tasks``; each element is either the
        result or a :class:`~hamqadam_ai.core.exceptions.HamqadamError`.
    """

    async def guarded(task: Awaitable[T], index: int) -> T | HamqadamError:
        try:
            if timeout_seconds is not None:
                return await run_with_timeout(
                    task, timeout_seconds, stage=f"{stage}[{index}]"
                )
            return await task
        except asyncio.CancelledError:
            raise
        except HamqadamError as exc:
            return exc
        except Exception as exc:  # noqa: BLE001 - normalised into the result list
            return HamqadamError(
                f"Unhandled failure in stage {stage!r}: {exc}",
                details={"stage": stage, "index": index, "exception": type(exc).__name__},
                cause=exc,
            )

    return await asyncio.gather(*(guarded(task, i) for i, task in enumerate(tasks)))


@dataclass(slots=True)
class CircuitBreaker:
    """Trip after repeated failures so a dead dependency is not hammered.

    Guards the optional external services (Qdrant, Redis, RabbitMQ). When open,
    calls fail immediately with the last error instead of paying a connection
    timeout on every request, and one probe is allowed through after
    ``reset_timeout`` to detect recovery.

    Attributes:
        failure_threshold: Consecutive failures required to open the circuit.
        reset_timeout: Seconds the circuit stays open before a probe is allowed.
        name: Label for diagnostics.
    """

    failure_threshold: int = 5
    reset_timeout: float = 30.0
    name: str = "dependency"

    _failures: int = 0
    _opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        """True when calls should be short-circuited right now."""
        if self._opened_at is None:
            return False
        # Once reset_timeout has elapsed the breaker goes half-open: exactly one
        # call is let through to probe whether the dependency has recovered.
        return (time.monotonic() - self._opened_at) < self.reset_timeout

    @property
    def state(self) -> str:
        """``closed``, ``open`` or ``half_open``, for metrics and health output."""
        if self._opened_at is None:
            return "closed"
        return "open" if self.is_open else "half_open"

    def record_success(self) -> None:
        """Reset the breaker after a successful call."""
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Count a failure and open the circuit once the threshold is reached."""
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()

    def guard(self) -> None:
        """Raise if the circuit is open.

        Raises:
            HamqadamError: with a retryable ``DEPENDENCY_UNAVAILABLE`` code.
        """
        if not self.is_open:
            return
        from hamqadam_ai.core.errors import ErrorCode

        assert self._opened_at is not None  # noqa: S101 - implied by is_open
        remaining = self.reset_timeout - (time.monotonic() - self._opened_at)
        raise HamqadamError(
            f"Circuit breaker for {self.name!r} is open; "
            f"retrying in {remaining:.1f}s.",
            code=ErrorCode.DEPENDENCY_UNAVAILABLE,
            details={
                "dependency": self.name,
                "consecutive_failures": self._failures,
                "retry_after_seconds": round(max(0.0, remaining), 2),
            },
        )


__all__ = [
    "INFERENCE_RETRY",
    "NETWORK_RETRY",
    "NO_RETRY",
    "CircuitBreaker",
    "RetryExhaustedError",
    "RetryPolicy",
    "gather_resilient",
    "run_with_timeout",
    "with_async_retry",
    "with_retry",
]
