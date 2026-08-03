"""Prometheus instrumentation.

What is measured, and what is deliberately not
----------------------------------------------
Metrics are aggregate and non-identifying. There is no label carrying a user
reference, a CNIC number, a filename or an image hash: a metric is scraped,
stored for a month and read by anyone with dashboard access, which makes it
exactly the wrong place for anything about a specific applicant.

Label cardinality is the other constraint. Prometheus keeps one time series per
distinct label combination, so a label with unbounded values - a request id, a
verification id, an error message - multiplies the series count without bound
until the server falls over. Every label here draws from a small fixed set:
stage names, recommendation values, error codes, decision reasons.

The four questions these answer
-------------------------------
1. *Is it working?* ``verifications_total`` by recommendation, and
   ``errors_total`` by code.
2. *Is it fast enough?* ``verification_seconds`` and ``stage_seconds``, as
   histograms. Not gauges: a mean latency hides the tail, and the tail is what
   a caller's timeout actually hits.
3. *Is it deciding sensibly?* ``identity_confidence`` and ``fraud_risk``
   distributions, plus ``decision_reasons_total``. A sudden shift in the score
   distribution is the earliest visible sign that a model or an input pipeline
   has changed underneath you.
4. *Is it degraded?* ``stage_skips_total`` and ``gallery_size``.

No-op when the client is absent
-------------------------------
``prometheus_client`` is an optional dependency. Every function here degrades
to a no-op rather than raising, because a service must not fall over because
its monitoring library is missing.
"""

from __future__ import annotations

from typing import Any

from hamqadam_ai.core.config import MetricsConfig

#: Populated by :func:`configure_metrics`. Empty means metrics are off, and
#: every recording function returns immediately.
_metrics: dict[str, Any] = {}


def configure_metrics(config: MetricsConfig) -> bool:
    """Create the collectors. Idempotent.

    Args:
        config: The metrics section of the observability settings.

    Returns:
        ``True`` when metrics are active, ``False`` when disabled or when
        ``prometheus_client`` is not installed.
    """
    if not config.enabled:
        _metrics.clear()
        return False

    if _metrics:
        return True

    try:
        from prometheus_client import Counter, Gauge, Histogram
    except ImportError:
        return False

    buckets = tuple(config.latency_buckets)

    _metrics.update(
        {
            "verifications": Counter(
                "hamqadam_verifications_total",
                "Verifications completed, by recommendation.",
                ["recommendation", "fraud_level"],
            ),
            "errors": Counter(
                "hamqadam_errors_total",
                "Requests that failed, by stable error code.",
                ["code"],
            ),
            "latency": Histogram(
                "hamqadam_verification_seconds",
                "End-to-end verification latency.",
                buckets=buckets,
            ),
            "stage_latency": Histogram(
                "hamqadam_stage_seconds",
                "Per-stage latency inside the pipeline.",
                ["stage"],
                buckets=buckets,
            ),
            "stage_status": Counter(
                "hamqadam_stage_status_total",
                "Pipeline stage outcomes.",
                ["stage", "status"],
            ),
            "identity_confidence": Histogram(
                "hamqadam_identity_confidence",
                "Identity confidence distribution, 0-100.",
                # Ten equal buckets. A shift in this distribution is the
                # earliest visible sign that a model or an input source has
                # changed, and it shows up here long before it shows up in a
                # complaint.
                buckets=(10, 20, 30, 40, 50, 60, 70, 75, 85, 95, 100),
            ),
            "fraud_risk": Histogram(
                "hamqadam_fraud_risk",
                "Fraud risk distribution, 0-100.",
                buckets=(5, 10, 20, 30, 40, 50, 65, 80, 90, 100),
            ),
            "decision_reasons": Counter(
                "hamqadam_decision_reasons_total",
                "Reasons cited by the decision engine.",
                ["reason"],
            ),
            "fraud_signals": Counter(
                "hamqadam_fraud_signals_total",
                "Fraud signals raised, by code.",
                ["code", "family"],
            ),
            "gallery_size": Gauge(
                "hamqadam_gallery_vectors",
                "Face templates currently enrolled in the duplicate gallery.",
            ),
            "budget_exhausted": Counter(
                "hamqadam_budget_exhausted_total",
                "Verifications that ran out of time budget before finishing.",
            ),
        }
    )
    return True


def record_verification(result: Any) -> None:
    """Record everything one finished verification tells us.

    Args:
        result: A :class:`VerificationResult`.

    Never raises. Instrumentation that can fail a request is worse than no
    instrumentation, because it converts an observability problem into an
    availability problem.
    """
    if not _metrics:
        return

    try:
        _metrics["verifications"].labels(
            recommendation=str(result.recommendation),
            fraud_level=str(result.fraud_risk_level),
        ).inc()
        _metrics["fraud_risk"].observe(result.fraud_risk_score)

        if result.processing_time is not None:
            _metrics["latency"].observe(result.processing_time.total / 1000.0)

        # Optional, because a request with no usable face has no identity
        # confidence at all. Observing a placeholder zero would put a spike at
        # the bottom of the histogram and make a shortage of evidence look
        # like a wave of impostors.
        if result.identity_confidence_score is not None:
            _metrics["identity_confidence"].observe(
                result.identity_confidence_score
            )

        for reason in result.recommendation_reasons:
            code = reason.get("code") if isinstance(reason, dict) else None
            if code:
                _metrics["decision_reasons"].labels(reason=str(code)).inc()

        for stage in result.stages:
            # Three states, from two booleans: a stage that did not run is not
            # the same as one that ran and failed, and collapsing them would
            # hide degradation behind an unchanged failure count.
            status = (
                "succeeded"
                if stage.succeeded
                else ("failed" if stage.ran else "skipped")
            )
            _metrics["stage_status"].labels(
                stage=stage.stage, status=status
            ).inc()
            if stage.ran and stage.duration_ms is not None:
                _metrics["stage_latency"].labels(stage=stage.stage).observe(
                    stage.duration_ms / 1000.0
                )

        if result.fraud is not None:
            for signal in result.fraud.signals:
                _metrics["fraud_signals"].labels(
                    code=str(signal.code), family=str(signal.family)
                ).inc()
    except Exception:  # noqa: BLE001 - see the docstring
        return


def record_error(code: str) -> None:
    """Count one failed request against its stable error code."""
    if not _metrics:
        return
    try:
        _metrics["errors"].labels(code=code).inc()
    except Exception:  # noqa: BLE001
        return


def record_budget_exhausted() -> None:
    """Count a verification that ran out of time before finishing."""
    if not _metrics:
        return
    try:
        _metrics["budget_exhausted"].inc()
    except Exception:  # noqa: BLE001
        return


def set_gallery_size(count: int) -> None:
    """Publish the current template count.

    A gauge rather than a counter: it goes down when a template is erased, and
    a counter that decreases breaks every rate() query over it.
    """
    if not _metrics:
        return
    try:
        _metrics["gallery_size"].set(float(count))
    except Exception:  # noqa: BLE001
        return


def metrics_active() -> bool:
    """Whether collectors were created."""
    return bool(_metrics)


def reset_metrics() -> None:
    """Drop the collectors. For tests, which need a clean registry."""
    _metrics.clear()


__all__ = [
    "configure_metrics",
    "metrics_active",
    "record_budget_exhausted",
    "record_error",
    "record_verification",
    "reset_metrics",
    "set_gallery_size",
]
