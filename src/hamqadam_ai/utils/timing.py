"""Latency measurement for the pipeline and its stages.

Everything uses :func:`time.perf_counter`, which is monotonic and unaffected by
NTP steps or DST. Wall-clock time is unsuitable for measuring a 200 ms stage.

The recorded per-stage breakdown is returned in the API response, so the
Backend can see *where* a slow verification spent its time without needing
access to the AI service's traces.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Stopwatch:
    """A restartable monotonic timer.

    Example:
        >>> watch = Stopwatch().start()
        >>> _ = sum(range(1000))
        >>> watch.stop() >= 0.0
        True
    """

    _started_at: float | None = None
    _elapsed: float = 0.0

    def start(self) -> Stopwatch:
        """Begin or resume timing. Returns ``self`` for chaining."""
        if self._started_at is None:
            self._started_at = time.perf_counter()
        return self

    def stop(self) -> float:
        """Pause timing and return the total accumulated seconds."""
        if self._started_at is not None:
            self._elapsed += time.perf_counter() - self._started_at
            self._started_at = None
        return self._elapsed

    def reset(self) -> None:
        """Clear the accumulated time and stop the watch."""
        self._started_at = None
        self._elapsed = 0.0

    @property
    def elapsed(self) -> float:
        """Seconds accumulated so far, including any in-flight interval."""
        if self._started_at is None:
            return self._elapsed
        return self._elapsed + (time.perf_counter() - self._started_at)

    @property
    def elapsed_ms(self) -> float:
        """Accumulated time in milliseconds."""
        return self.elapsed * 1000.0

    @property
    def running(self) -> bool:
        """Whether the watch is currently accumulating."""
        return self._started_at is not None


@dataclass(slots=True)
class StageTimings:
    """Accumulates per-stage durations for one verification request.

    A stage measured more than once (seven images through the detector) has its
    durations summed and its invocation count tracked, so the report shows both
    total time in the stage and average time per call.
    """

    _totals: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    _overall: Stopwatch = field(default_factory=Stopwatch)

    def __post_init__(self) -> None:
        self._overall.start()

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Time the ``with`` block and attribute it to ``stage``.

        The duration is recorded even when the block raises, so a stage that
        times out still shows how long it consumed.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, time.perf_counter() - started)

    def record(self, stage: str, seconds: float) -> None:
        """Add ``seconds`` to a stage's total."""
        if stage not in self._totals:
            self._totals[stage] = 0.0
            self._counts[stage] = 0
            self._order.append(stage)
        self._totals[stage] += seconds
        self._counts[stage] += 1

    def total_for(self, stage: str) -> float:
        """Total seconds attributed to ``stage``; zero if never measured."""
        return self._totals.get(stage, 0.0)

    def count_for(self, stage: str) -> int:
        """Number of times ``stage`` was measured."""
        return self._counts.get(stage, 0)

    @property
    def total_seconds(self) -> float:
        """Wall-clock seconds since this object was created."""
        return self._overall.elapsed

    @property
    def total_ms(self) -> float:
        """Wall-clock milliseconds since this object was created."""
        return self._overall.elapsed_ms

    @property
    def accounted_seconds(self) -> float:
        """Sum of all stage totals.

        Compared against :attr:`total_seconds` this reveals unaccounted
        overhead - or, when it *exceeds* the wall clock, confirms that stages
        genuinely ran concurrently.
        """
        return sum(self._totals.values())

    def finish(self) -> None:
        """Stop the overall timer. Idempotent."""
        self._overall.stop()

    def as_dict(self, *, milliseconds: bool = True) -> dict[str, Any]:
        """Render the breakdown for the API response.

        Args:
            milliseconds: Report in milliseconds rather than seconds.

        Returns:
            A mapping with ``total``, ``unit``, ``stages`` (in the order they
            were first measured) and ``overhead``.
        """
        factor = 1000.0 if milliseconds else 1.0
        stages = {
            name: {
                "duration": round(self._totals[name] * factor, 2),
                "calls": self._counts[name],
                "average": round((self._totals[name] / self._counts[name]) * factor, 2),
            }
            for name in self._order
        }
        overhead = max(0.0, self.total_seconds - self.accounted_seconds)
        return {
            "total": round(self.total_seconds * factor, 2),
            "unit": "ms" if milliseconds else "s",
            "stages": stages,
            "overhead": round(overhead * factor, 2),
        }

    def slowest_stage(self) -> tuple[str, float] | None:
        """Return the ``(stage, seconds)`` pair that consumed the most time."""
        if not self._totals:
            return None
        name = max(self._totals, key=lambda key: self._totals[key])
        return name, self._totals[name]


@contextmanager
def perf_timer(label: str, *, sink: dict[str, float] | None = None) -> Iterator[Stopwatch]:
    """Time a block and optionally store the result in ``sink``.

    A lighter alternative to :class:`StageTimings` for ad-hoc measurement
    inside a single function.

    Args:
        label: Key under which the duration is stored in ``sink``.
        sink: Optional mapping to write the duration (in seconds) into.

    Yields:
        The running :class:`Stopwatch`, so the block can read intermediate
        elapsed time.
    """
    watch = Stopwatch().start()
    try:
        yield watch
    finally:
        duration = watch.stop()
        if sink is not None:
            sink[label] = duration


class Deadline:
    """A shared wall-clock budget for a multi-stage operation.

    The pipeline gets one overall budget from ``server.request_timeout_seconds``
    and each stage asks the deadline how much is left. This prevents the
    failure mode where every stage has a generous individual timeout and their
    sum quietly exceeds the caller's own timeout.

    Args:
        budget_seconds: Total time available from construction.
    """

    __slots__ = ("_budget", "_started")

    def __init__(self, budget_seconds: float) -> None:
        if budget_seconds <= 0:
            raise ValueError("Deadline budget must be positive")
        self._budget = budget_seconds
        self._started = time.perf_counter()

    @property
    def budget(self) -> float:
        """The total budget this deadline was created with, in seconds."""
        return self._budget

    @property
    def elapsed(self) -> float:
        """Seconds consumed since the deadline was created."""
        return time.perf_counter() - self._started

    @property
    def remaining(self) -> float:
        """Seconds left, floored at zero."""
        return max(0.0, self._budget - self.elapsed)

    @property
    def expired(self) -> bool:
        """Whether the budget is exhausted."""
        return self.remaining <= 0.0

    def slice_for(self, stage: str, requested: float) -> float:
        """Return the time a stage may take: ``min(requested, remaining)``.

        Args:
            stage: Stage name, used in the error message on expiry.
            requested: The stage's own preferred timeout.

        Raises:
            PipelineTimeoutError: if the budget is already exhausted.
        """
        remaining = self.remaining
        if remaining <= 0.0:
            from hamqadam_ai.core.exceptions import PipelineTimeoutError

            raise PipelineTimeoutError(
                f"Time budget exhausted before stage {stage!r} could start.",
                details={
                    "stage": stage,
                    "budget_seconds": self._budget,
                    "elapsed_seconds": round(self.elapsed, 3),
                },
            )
        return min(requested, remaining)


__all__ = ["Deadline", "StageTimings", "Stopwatch", "perf_timer"]
