"""Choosing a 1:N threshold, and why the 1:1 one cannot be reused.

The mistake this module exists to prevent
-----------------------------------------
Module 4 asks "are these two the same person?" - a 1:1 verification against one
claimed identity. This module asks "is this person already in the gallery?" -
a 1:N identification against everyone enrolled. Reusing the verification
threshold is the standard way to get this wrong, because **a query that
compares against N templates has N chances to match one of them by accident**:

    system false-match rate  ~  1 - (1 - per_comparison_far) ** N

At a per-comparison FAR of 1e-4 - unremarkable for a face recogniser - a
gallery of 10,000 produces a false duplicate on roughly **63%** of queries. The
per-comparison FAR needed for a 1% system rate is ``0.01 / N``, so:

    gallery       FAR for 1% system rate    for 0.1%
      1,000                     1.0e-05     1.0e-06
     10,000                     1.0e-06     1.0e-07
    100,000                     1.0e-07     1.0e-08
  1,000,000                     1.0e-08     1.0e-09

What the simulation says, and why it is not reassuring
------------------------------------------------------
Modelled as isotropic random unit vectors in 512 dimensions, the largest cosine
across a gallery of 100,000 is about **0.21** - nowhere near the configured
0.68, and the conclusion would be that gallery growth is harmless.

That model is wrong, and it is wrong in the unsafe direction. Real face
embeddings do not fill the sphere: every enrolled vector is a *face*, so they
share a manifold of far lower intrinsic dimension. Repeating the simulation
with the query drawn from the same subspace as the gallery, the largest cosine
at 100,000 records becomes:

    effective dimension    512    128     64     32     16      8
    max impostor cosine  0.263  0.397  0.538  0.690  0.863  0.964

At an effective dimension of 32 the gallery reaches **0.690** - above the
configured 0.68 - on essentially every query. The answer swings from "entirely
safe" to "fails always" across a range of one parameter, and that parameter is
a property of the recogniser and the enrolled population that **cannot be
measured from this repository**. Two public-domain reference faces do not
estimate a manifold.

So this module does not ship a calibrated threshold. It ships the tool to
derive one - :func:`recommend_threshold` - and reports ``thresholds_validated:
false`` until somebody runs it against a real gallery.

The tail that matters more than the arithmetic
----------------------------------------------
Everything above concerns *random* impostors. The pairs that actually defeat a
duplicate check are not random: identical twins, siblings, cousins, parents and
children at the same age. Face recognition cannot separate identical twins at
any threshold, and a matrimonial platform serving extended families will enrol
exactly the population where this is most common. No threshold fixes that; only
a second factor does, and choosing one is the Client's decision, not this
service's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from hamqadam_ai.duplicate_detection.base import FloatArray


@dataclass(frozen=True, slots=True)
class ImpostorStatistics:
    """The measured distribution of similarities between different people.

    Attributes:
        sample_count: How many impostor pairs were measured.
        mean: Mean impostor similarity.
        std: Its standard deviation.
        percentiles: Similarity at selected upper percentiles, keyed by the
            percentile. The tail is the whole story, so these matter far more
            than the mean.
        maximum: The largest impostor similarity seen.
    """

    sample_count: int
    mean: float
    std: float
    percentiles: dict[float, float]
    maximum: float

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "sample_count": self.sample_count,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "percentiles": {
                str(key): round(value, 6) for key, value in self.percentiles.items()
            },
            "maximum": round(self.maximum, 6),
        }


@dataclass(frozen=True, slots=True)
class ThresholdRecommendation:
    """A threshold derived from a measured gallery, and its caveats.

    Attributes:
        threshold: The recommended cosine threshold.
        gallery_size: Size the recommendation was derived for.
        target_system_far: The system-level false-match rate aimed at.
        per_comparison_far: The per-comparison rate that implies.
        extrapolated: Whether the threshold sits beyond the measured data and
            was extrapolated from a Gaussian tail rather than observed. An
            extrapolated threshold is a guess with arithmetic attached.
        statistics: The impostor distribution it came from.
    """

    threshold: float
    gallery_size: int
    target_system_far: float
    per_comparison_far: float
    extrapolated: bool
    statistics: ImpostorStatistics

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "threshold": round(self.threshold, 6),
            "gallery_size": self.gallery_size,
            "target_system_far": self.target_system_far,
            "per_comparison_far": self.per_comparison_far,
            "extrapolated": self.extrapolated,
            "statistics": self.statistics.as_dict(),
        }


def per_comparison_far(target_system_far: float, gallery_size: int) -> float:
    """The per-comparison FAR needed for a target system-level rate.

    Inverts ``system = 1 - (1 - far) ** n`` exactly rather than using the
    ``far ~ target / n`` approximation, which drifts once the target is not
    small relative to one.

    Evaluated through ``log1p`` and ``expm1`` rather than written out directly.
    The direct form suffers catastrophic cancellation for a large gallery: at
    ``n = 1e12`` the intermediate ``(1 - target) ** (1 / n)`` rounds to exactly
    1.0, so subtracting it from one gives **zero** - which reads as "no
    threshold can achieve this target" and is off by eighteen orders of
    magnitude. The stable form returns 1e-18, which is correct.

    Args:
        target_system_far: Acceptable probability that a query returns at
            least one false duplicate, in ``(0, 1)``.
        gallery_size: Number of comparable templates enrolled.

    Returns:
        The per-comparison false-accept rate. Returns the target unchanged for
        an empty or single-entry gallery, where the two coincide.
    """
    if not 0.0 < target_system_far < 1.0:
        raise ValueError("target_system_far must lie strictly between 0 and 1")
    if gallery_size <= 1:
        return target_system_far
    return float(-math.expm1(math.log1p(-target_system_far) / gallery_size))


def system_far(per_comparison: float, gallery_size: int) -> float:
    """The system-level false-match rate implied by a per-comparison rate.

    The forward direction of :func:`per_comparison_far`, exposed because it is
    the number that makes the problem visible: it is what turns "one in ten
    thousand" into "most queries".
    """
    if not 0.0 <= per_comparison <= 1.0:
        raise ValueError("per_comparison must lie in [0, 1]")
    if gallery_size <= 0:
        return 0.0
    return 1.0 - (1.0 - per_comparison) ** gallery_size


def measure_impostor_distribution(
    vectors: FloatArray,
    *,
    max_pairs: int = 2_000_000,
    percentiles: tuple[float, ...] = (50.0, 90.0, 99.0, 99.9, 99.99),
    seed: int = 0,
) -> ImpostorStatistics:
    """Measure how similar different people's templates actually are.

    Assumes every row belongs to a **different** person. That is the caller's
    responsibility and it is the one assumption that matters: a gallery
    containing two templates of one person would put a genuine match into the
    impostor distribution and inflate the recommended threshold, which fails
    silently in the unsafe direction.

    Args:
        vectors: ``(n, d)`` L2-normalised templates, one per person.
        max_pairs: Cap on sampled pairs. All pairs are used when the gallery is
            small enough; above that a random sample is taken, because the full
            set grows quadratically and 100,000 templates is five billion pairs.
        percentiles: Upper percentiles to report.
        seed: Sampling seed, so a recommendation is reproducible.

    Returns:
        The distribution.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return ImpostorStatistics(
            sample_count=0,
            mean=0.0,
            std=0.0,
            percentiles=dict.fromkeys(percentiles, 0.0),
            maximum=0.0,
        )

    count = matrix.shape[0]
    total_pairs = count * (count - 1) // 2

    if total_pairs <= max_pairs:
        similarities = matrix @ matrix.T
        upper = np.triu_indices(count, k=1)
        samples = similarities[upper]
    else:
        rng = np.random.default_rng(seed)
        left = rng.integers(0, count, size=max_pairs)
        right = rng.integers(0, count, size=max_pairs)
        keep = left != right
        left, right = left[keep], right[keep]
        samples = np.einsum("ij,ij->i", matrix[left], matrix[right])

    return ImpostorStatistics(
        sample_count=int(samples.size),
        mean=float(samples.mean()),
        std=float(samples.std()),
        percentiles={
            level: float(np.percentile(samples, level)) for level in percentiles
        },
        maximum=float(samples.max()),
    )


def recommend_threshold(
    vectors: FloatArray,
    *,
    gallery_size: int | None = None,
    target_system_far: float = 0.01,
    max_pairs: int = 2_000_000,
    seed: int = 0,
) -> ThresholdRecommendation:
    """Derive a 1:N threshold from a real gallery.

    Reads the threshold off the measured impostor distribution wherever the
    data reaches, and extrapolates a Gaussian tail only when it does not -
    flagging that it did, because an extrapolated threshold is a guess with
    arithmetic attached rather than a measurement.

    Args:
        vectors: ``(n, d)`` L2-normalised templates, one per **distinct**
            person.
        gallery_size: Size to size the threshold for. Defaults to the number
            of vectors supplied; pass a larger figure to plan for growth.
        target_system_far: Acceptable probability that a query returns at
            least one false duplicate.
        max_pairs: Cap on sampled pairs.
        seed: Sampling seed.

    Returns:
        The recommendation, with everything behind it.
    """
    statistics = measure_impostor_distribution(
        vectors, max_pairs=max_pairs, seed=seed
    )
    size = gallery_size if gallery_size is not None else int(
        np.asarray(vectors).shape[0]
    )
    far = per_comparison_far(target_system_far, size)

    if statistics.sample_count == 0:
        return ThresholdRecommendation(
            threshold=1.0,
            gallery_size=size,
            target_system_far=target_system_far,
            per_comparison_far=far,
            extrapolated=True,
            statistics=statistics,
        )

    # The percentile the target FAR corresponds to. A per-comparison FAR of
    # 1e-6 is the 99.9999th percentile of the impostor distribution.
    required_percentile = 100.0 * (1.0 - far)
    # A sample of n pairs cannot resolve a percentile finer than 1 - 1/n.
    resolvable = 100.0 * (1.0 - 1.0 / statistics.sample_count)

    if required_percentile <= resolvable:
        threshold = float(
            np.percentile(
                _resample(vectors, max_pairs, seed), required_percentile
            )
        )
        return ThresholdRecommendation(
            threshold=threshold,
            gallery_size=size,
            target_system_far=target_system_far,
            per_comparison_far=far,
            extrapolated=False,
            statistics=statistics,
        )

    # Beyond the data. A Gaussian tail is the conventional extrapolation and is
    # very likely optimistic: real impostor distributions are right-skewed by
    # look-alikes, so the true threshold for this FAR is higher than this.
    z = _inverse_normal_tail(far)
    threshold = statistics.mean + z * statistics.std
    return ThresholdRecommendation(
        threshold=float(min(max(threshold, -1.0), 1.0)),
        gallery_size=size,
        target_system_far=target_system_far,
        per_comparison_far=far,
        extrapolated=True,
        statistics=statistics,
    )


def _resample(vectors: FloatArray, max_pairs: int, seed: int) -> np.ndarray:
    """Re-derive the pair samples for a percentile read.

    Recomputed rather than carried on :class:`ImpostorStatistics`, which would
    otherwise hold a two-million-element array of biometric-derived data for
    the lifetime of the recommendation.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    count = matrix.shape[0]
    if count * (count - 1) // 2 <= max_pairs:
        return (matrix @ matrix.T)[np.triu_indices(count, k=1)]
    rng = np.random.default_rng(seed)
    left = rng.integers(0, count, size=max_pairs)
    right = rng.integers(0, count, size=max_pairs)
    keep = left != right
    return np.asarray(
        np.einsum("ij,ij->i", matrix[left[keep]], matrix[right[keep]])
    )


def _inverse_normal_tail(tail_probability: float) -> float:
    """The z-score whose upper tail holds ``tail_probability`` of the mass.

    Bisection on ``erfc``. Slower than a rational approximation and exact to
    machine precision, which matters here: the tails involved are 1e-8 and
    below, where a rational fit designed for the middle of the distribution
    loses several digits.
    """
    if not 0.0 <= tail_probability < 1.0:
        raise ValueError("tail_probability must lie in [0, 1)")
    if tail_probability == 0.0:
        # Only reachable if a caller asks for a rate below the smallest
        # positive double. Saturate rather than raise: the answer is "further
        # into the tail than this arithmetic can express", and the bisection
        # bound below says the same thing.
        return 10.0

    def upper_tail(z: float) -> float:
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    low, high = -10.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if upper_tail(middle) > tail_probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


__all__ = [
    "ImpostorStatistics",
    "ThresholdRecommendation",
    "measure_impostor_distribution",
    "per_comparison_far",
    "recommend_threshold",
    "system_far",
]
