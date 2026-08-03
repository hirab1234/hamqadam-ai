"""Mapping raw measurements onto the ``[0, 1]`` quality scale.

Why not thresholds
------------------
A bare pass/fail threshold discards exactly the information that matters most
downstream. "Blur = fail" tells a user nothing; "blur 0.31, limited by
defocus" tells them their photo is well short of usable and tells the fraud
engine that this attempt is a poor-quality one rather than a marginal one.

Why smoothstep
--------------
A linear ramp has a discontinuous derivative at both anchors, so a measurement
sitting near an anchor makes the composite score jitter on sensor noise alone.
The cubic smoothstep ``3t^2 - 2t^3`` is monotone, bounded to ``[0, 1]``, and
has zero derivative at both ends, which damps that jitter without introducing
any new parameter to tune.
"""

from __future__ import annotations

import math

from hamqadam_ai.core.config import BandMapping, RampMapping


def smoothstep(t: float) -> float:
    """Cubic Hermite interpolation of ``t`` clamped to ``[0, 1]``.

    Args:
        t: Position along the ramp. Values outside ``[0, 1]`` are clamped.

    Returns:
        ``3t^2 - 2t^3``, which is 0 at ``t=0``, 1 at ``t=1``, and has zero
        gradient at both.

    Example:
        >>> smoothstep(0.0), smoothstep(0.5), smoothstep(1.0)
        (0.0, 0.5, 1.0)
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def ramp_score(value: float, mapping: RampMapping) -> float:
    """Score a monotone measurement against a two-anchor ramp.

    Handles both directions transparently: when ``mapping.floor`` is greater
    than ``mapping.good`` the metric is lower-is-better (noise sigma, JPEG
    blockiness) and the ramp inverts.

    Args:
        value: The raw measurement.
        mapping: The configured anchor pair.

    Returns:
        A score in ``[0, 1]``.

    Example:
        >>> from hamqadam_ai.core.config import RampMapping
        >>> higher_better = RampMapping(floor=10.0, good=110.0)
        >>> round(ramp_score(60.0, higher_better), 3)
        0.5
        >>> lower_better = RampMapping(floor=15.0, good=1.5)
        >>> ramp_score(20.0, lower_better)
        0.0
    """
    if not math.isfinite(value):
        # A non-finite measurement means the analyser hit a degenerate input.
        # Scoring it zero is the safe direction: it can only reject, never
        # approve, an image we failed to measure.
        return 0.0

    if not mapping.log_scale:
        span = mapping.good - mapping.floor
        return smoothstep((value - mapping.floor) / span)

    # Log interpolation. A measurement of zero or below is off the bottom of a
    # log scale entirely: for a higher-is-better metric that is the worst
    # possible reading, for a lower-is-better one it is the best.
    if value <= 0.0:
        return 1.0 if mapping.lower_is_better else 0.0

    log_value = math.log10(value)
    log_floor = math.log10(mapping.floor)
    log_good = math.log10(mapping.good)
    return smoothstep((log_value - log_floor) / (log_good - log_floor))


def band_score(value: float, mapping: BandMapping) -> float:
    """Score a measurement where both extremes are bad.

    Zero outside ``[min_acceptable, max_acceptable]``, one across the ideal
    plateau, smoothstep on each shoulder.

    The plateau is the point: an image at mean luminance 120 and one at 160 are
    both simply well exposed, and ramping towards a single ideal value would
    penalise one of them for a difference no photographer would call a defect.

    Args:
        value: The raw measurement.
        mapping: The configured four-point band.

    Returns:
        A score in ``[0, 1]``.

    Example:
        >>> from hamqadam_ai.core.config import BandMapping
        >>> band = BandMapping(
        ...     min_acceptable=30.0, ideal_low=95.0,
        ...     ideal_high=180.0, max_acceptable=230.0,
        ... )
        >>> band_score(130.0, band)
        1.0
        >>> band_score(20.0, band)
        0.0
    """
    if not math.isfinite(value):
        return 0.0

    if value <= mapping.min_acceptable or value >= mapping.max_acceptable:
        return 0.0
    if mapping.ideal_low <= value <= mapping.ideal_high:
        return 1.0

    if value < mapping.ideal_low:
        span = mapping.ideal_low - mapping.min_acceptable
        if span <= 0.0:
            return 1.0
        return smoothstep((value - mapping.min_acceptable) / span)

    span = mapping.max_acceptable - mapping.ideal_high
    if span <= 0.0:
        return 1.0
    return smoothstep((mapping.max_acceptable - value) / span)


def weighted_mean(components: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted arithmetic mean over the components present in both mappings.

    Weights are renormalised across whatever components are actually supplied,
    so a metric that could not be measured shrinks the denominator rather than
    silently contributing zero.

    Args:
        components: Component name to score in ``[0, 1]``.
        weights: Component name to weight.

    Returns:
        The weighted mean, or 0.0 when nothing could be measured.
    """
    total_weight = 0.0
    accumulated = 0.0
    for name, score in components.items():
        weight = weights.get(name)
        if weight is None or weight <= 0.0:
            continue
        accumulated += weight * score
        total_weight += weight
    return accumulated / total_weight if total_weight > 0.0 else 0.0


def weighted_power_mean(
    components: dict[str, float], weights: dict[str, float], power: float
) -> float:
    """Weighted generalised mean with exponent ``power``.

    ``power = 1`` recovers the arithmetic mean. Below 1 the mean is pulled
    towards its smallest component, which is the behaviour the composite
    quality score needs: a plain average lets five healthy metrics conceal one
    fatal one, so a perfectly exposed but completely out-of-focus photograph
    scores around 70 instead of being rejected.

    A zero component drives the whole result to zero for ``power <= 0``, which
    is why the exponent is constrained positive in configuration.

    Args:
        components: Component name to score in ``[0, 1]``.
        weights: Component name to weight.
        power: The exponent, strictly positive.

    Returns:
        The weighted power mean in ``[0, 1]``.

    Example:
        >>> scores = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 0.0}
        >>> equal = dict.fromkeys(scores, 0.2)
        >>> round(weighted_mean(scores, equal), 3)
        0.8
        >>> round(weighted_power_mean(scores, equal, 0.5), 3)
        0.64
    """
    if power <= 0.0:
        raise ValueError("power must be strictly positive")

    total_weight = 0.0
    accumulated = 0.0
    for name, score in components.items():
        weight = weights.get(name)
        if weight is None or weight <= 0.0:
            continue
        accumulated += weight * (max(0.0, min(1.0, score)) ** power)
        total_weight += weight

    if total_weight <= 0.0:
        return 0.0
    return float((accumulated / total_weight) ** (1.0 / power))


def limiting_component(
    components: dict[str, float], weights: dict[str, float]
) -> str | None:
    """Return the component responsible for the largest weighted deficit.

    This is what turns a bare number into actionable feedback - the single
    thing most worth telling the user to fix.

    Args:
        components: Component name to score in ``[0, 1]``.
        weights: Component name to weight.

    Returns:
        The component name, or ``None`` when nothing was measured.
    """
    if not components:
        return None
    deficits = {
        name: weights.get(name, 0.0) * (1.0 - score)
        for name, score in components.items()
    }
    return max(deficits, key=lambda name: deficits[name])


__all__ = [
    "band_score",
    "limiting_component",
    "ramp_score",
    "smoothstep",
    "weighted_mean",
    "weighted_power_mean",
]
