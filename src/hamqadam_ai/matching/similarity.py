"""Cosine similarity and its calibration onto the reported scale.

The calibration problem
-----------------------
The API contract reports match scores on a 0-100 scale, but the underlying
quantity is a cosine similarity in ``[-1, 1]``. The obvious conversion is a
linear rescaling::

    score = (cosine + 1) / 2 * 100

which is actively misleading. Two embeddings of different people are close to
*orthogonal*, not opposed - the measured impostor pair in Module 3 scored
0.011 - so that formula reports **50** for a confident non-match. A reviewer
reading "50" reasonably infers "half a match, borderline"; the truth is
"certainly a different person".

What is done instead
--------------------
The score is interpolated piecewise-linearly through the **decision boundaries
themselves**:

===========================  ==================
cosine                       reported score
===========================  ==================
``<= floor_similarity``      0
``review`` threshold         ``review_score`` (50)
``strong_match`` threshold   ``strong_match_score`` (75)
``1.0``                      100
===========================  ==================

Two properties fall out of this, both of them the point:

1. **The score is self-describing.** 75+ is a strong match, 50-75 needs review,
   below 50 failed - regardless of which comparison produced it. A reader does
   not need to know that the CNIC threshold is 0.42 and the profile threshold
   0.62 to interpret the number.
2. **The comparison types become commensurable.** A CNIC score of 80 and a
   profile score of 80 mean the same thing about the strength of the evidence,
   even though they came from very different cosine values. That is what makes
   the weighted identity aggregation in :mod:`hamqadam_ai.matching.aggregator`
   defensible.

The cost is that the reported score is *not* a linear function of the cosine,
so it must never be reverse-engineered into one. :func:`uncalibrate_score`
exists for tests and for the evaluation harness, which needs to move back into
cosine space to compute an ROC.
"""

from __future__ import annotations

from hamqadam_ai.core.config import MatchThresholds, ScoreCalibration
from hamqadam_ai.core.constants import EPSILON, MatchDecision


def decide(similarity: float, thresholds: MatchThresholds) -> MatchDecision:
    """Classify a cosine similarity against the configured operating points.

    Args:
        similarity: Cosine similarity in ``[-1, 1]``.
        thresholds: The operating points for this comparison type.

    Returns:
        ``STRONG_MATCH``, ``REVIEW`` or ``FAILED``.

    Example:
        >>> from hamqadam_ai.core.config import MatchThresholds
        >>> points = MatchThresholds(strong_match=0.62, review=0.45)
        >>> decide(0.9, points)
        <MatchDecision.STRONG_MATCH: 'STRONG_MATCH'>
        >>> decide(0.5, points)
        <MatchDecision.REVIEW: 'REVIEW'>
        >>> decide(0.1, points)
        <MatchDecision.FAILED: 'FAILED'>
    """
    if similarity >= thresholds.strong_match:
        return MatchDecision.STRONG_MATCH
    if similarity >= thresholds.review:
        return MatchDecision.REVIEW
    return MatchDecision.FAILED


def calibrate_score(
    similarity: float,
    thresholds: MatchThresholds,
    calibration: ScoreCalibration,
) -> float:
    """Map a cosine similarity onto the reported 0-100 scale.

    Piecewise-linear through the decision boundaries - see the module
    docstring for why this rather than a linear rescaling of the raw metric.

    Args:
        similarity: Cosine similarity in ``[-1, 1]``.
        thresholds: The operating points for this comparison type.
        calibration: The shared anchor scores.

    Returns:
        A score in ``[0, 100]``, monotonically increasing in ``similarity``.

    Example:
        >>> from hamqadam_ai.core.config import MatchThresholds, ScoreCalibration
        >>> points = MatchThresholds(strong_match=0.62, review=0.45)
        >>> calib = ScoreCalibration()
        >>> round(calibrate_score(0.62, points, calib), 1)
        75.0
        >>> round(calibrate_score(0.45, points, calib), 1)
        50.0
        >>> round(calibrate_score(0.0, points, calib), 1)
        0.0
    """
    floor = calibration.floor_similarity
    review = thresholds.review
    strong = thresholds.strong_match

    if similarity <= floor:
        return 0.0
    if similarity >= 1.0:
        return 100.0

    if similarity < review:
        span = max(review - floor, EPSILON)
        fraction = (similarity - floor) / span
        return float(fraction * calibration.review_score)

    if similarity < strong:
        span = max(strong - review, EPSILON)
        fraction = (similarity - review) / span
        return float(
            calibration.review_score
            + fraction * (calibration.strong_match_score - calibration.review_score)
        )

    span = max(1.0 - strong, EPSILON)
    fraction = (similarity - strong) / span
    return float(
        calibration.strong_match_score
        + fraction * (100.0 - calibration.strong_match_score)
    )


def uncalibrate_score(
    score: float,
    thresholds: MatchThresholds,
    calibration: ScoreCalibration,
) -> float:
    """Invert :func:`calibrate_score`, recovering the cosine similarity.

    Needed by the evaluation harness, which computes ROC curves in cosine
    space, and by tests asserting the mapping round-trips. Production code
    should never need it: the calibrated score is the reporting format and the
    cosine is the internal one, and mixing them is how a threshold ends up
    applied on the wrong scale.

    Args:
        score: A calibrated score in ``[0, 100]``.
        thresholds: The operating points the score was produced with.
        calibration: The anchor scores the score was produced with.

    Returns:
        The cosine similarity that would calibrate to ``score``.
    """
    floor = calibration.floor_similarity
    review = thresholds.review
    strong = thresholds.strong_match

    if score <= 0.0:
        return floor
    if score >= 100.0:
        return 1.0

    if score < calibration.review_score:
        fraction = score / max(calibration.review_score, EPSILON)
        return float(floor + fraction * (review - floor))

    if score < calibration.strong_match_score:
        span = max(calibration.strong_match_score - calibration.review_score, EPSILON)
        fraction = (score - calibration.review_score) / span
        return float(review + fraction * (strong - review))

    span = max(100.0 - calibration.strong_match_score, EPSILON)
    fraction = (score - calibration.strong_match_score) / span
    return float(strong + fraction * (1.0 - strong))


def score_for_decision(
    decision: MatchDecision, calibration: ScoreCalibration
) -> float:
    """Return the calibrated score at the boundary of a decision band.

    Used by the aggregator when it needs to reason about "what would a bare
    pass have looked like" without a concrete similarity to hand.
    """
    if decision is MatchDecision.STRONG_MATCH:
        return calibration.strong_match_score
    if decision is MatchDecision.REVIEW:
        return calibration.review_score
    return 0.0


__all__ = [
    "calibrate_score",
    "decide",
    "score_for_decision",
    "uncalibrate_score",
]
