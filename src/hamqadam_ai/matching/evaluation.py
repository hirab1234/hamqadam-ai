"""Verification accuracy metrics: ROC, EER, TAR@FAR and confusion matrices.

Why this lives in the package rather than in a script
-----------------------------------------------------
The operating points in ``configs/thresholds.yaml`` decide whether a real
person is admitted or refused. Choosing them by intuition is not acceptable,
and re-deriving them requires exactly these curves, so the machinery is a
first-class, unit-tested part of the service rather than a one-off notebook.

It is also the honest counterweight to Module 3's caveat: the separation
measured there came from a two-identity sample, which demonstrates the model is
wired up correctly and says nothing about where genuine and impostor
distributions begin to overlap on real traffic.

Implemented in NumPy alone - no scikit-learn - so it can run inside the
production image without adding a dependency to the inference path.

Terminology
-----------
The biometric literature and the machine-learning literature use different
words for the same quantities. Both appear here because both audiences read
these reports:

=========================  ====================  ==============================
Biometrics                 ML                    Meaning
=========================  ====================  ==============================
FAR / FMR                  False positive rate   Impostors wrongly accepted
FRR / FNMR                 False negative rate   Genuine users wrongly refused
TAR / GAR                  True positive rate    Genuine users correctly accepted
EER                        -                     Threshold where FAR == FRR
=========================  ====================  ==============================

For identity verification **FAR is the expensive error**: admitting an impostor
onto somebody else's account is a security incident, while refusing a genuine
user is an inconvenience they can retry. Thresholds should therefore be chosen
at a fixed low FAR rather than at the EER, which weights the two equally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.constants import EPSILON

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Counts and derived rates at one decision threshold.

    Attributes:
        threshold: The similarity at or above which a pair is accepted.
        true_positives: Genuine pairs correctly accepted.
        false_positives: Impostor pairs wrongly accepted - the expensive error.
        true_negatives: Impostor pairs correctly refused.
        false_negatives: Genuine pairs wrongly refused.
    """

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def genuine_count(self) -> int:
        """Total genuine pairs evaluated."""
        return self.true_positives + self.false_negatives

    @property
    def impostor_count(self) -> int:
        """Total impostor pairs evaluated."""
        return self.true_negatives + self.false_positives

    @property
    def false_accept_rate(self) -> float:
        """FAR / FMR: the share of impostors wrongly accepted."""
        return self.false_positives / max(self.impostor_count, 1)

    @property
    def false_reject_rate(self) -> float:
        """FRR / FNMR: the share of genuine users wrongly refused."""
        return self.false_negatives / max(self.genuine_count, 1)

    @property
    def true_accept_rate(self) -> float:
        """TAR / GAR: the share of genuine users correctly accepted."""
        return self.true_positives / max(self.genuine_count, 1)

    @property
    def precision(self) -> float:
        """Of the pairs accepted, the share that were genuine."""
        accepted = self.true_positives + self.false_positives
        return self.true_positives / max(accepted, 1)

    @property
    def recall(self) -> float:
        """Same quantity as :attr:`true_accept_rate`, under the ML name."""
        return self.true_accept_rate

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        denominator = self.precision + self.recall
        if denominator <= EPSILON:
            return 0.0
        return 2.0 * self.precision * self.recall / denominator

    @property
    def accuracy(self) -> float:
        """Share of all pairs classified correctly.

        Reported because it is asked for, but it is the least useful number
        here: impostor pairs outnumber genuine ones quadratically in any
        all-pairs protocol, so a system that refused everybody would score
        highly on it.
        """
        total = self.genuine_count + self.impostor_count
        correct = self.true_positives + self.true_negatives
        return correct / max(total, 1)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form."""
        return {
            "threshold": round(self.threshold, 6),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "far": round(self.false_accept_rate, 6),
            "frr": round(self.false_reject_rate, 6),
            "tar": round(self.true_accept_rate, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "accuracy": round(self.accuracy, 6),
        }


@dataclass(slots=True)
class RocCurve:
    """A receiver operating characteristic over similarity thresholds.

    Attributes:
        thresholds: Descending similarity thresholds.
        false_accept_rate: FAR at each threshold.
        true_accept_rate: TAR at each threshold.
        auc: Area under the curve.
    """

    thresholds: FloatArray
    false_accept_rate: FloatArray
    true_accept_rate: FloatArray
    auc: float

    def tar_at_far(self, target_far: float) -> tuple[float, float]:
        """The best TAR achievable at or below a target FAR.

        The operating point a verification system is actually specified at:
        "accept 97% of genuine users while admitting at most one impostor in
        ten thousand" is a requirement one can hold a system to, whereas an
        EER is not.

        Args:
            target_far: The FAR ceiling, e.g. ``1e-4``.

        Returns:
            ``(tar, threshold)``. When the target cannot be met without
            refusing everybody - impostors scoring above every genuine pair -
            the returned threshold is one that accepts nothing, and the TAR is
            zero. The ``(0.0, 1.0)`` branch below is a guard for a curve built
            by hand without the sentinel thresholds :func:`roc_curve` adds; a
            curve from that function always contains a zero-FAR point.
        """
        eligible = self.false_accept_rate <= target_far
        if not eligible.any():
            return 0.0, 1.0
        index = int(np.argmax(np.where(eligible, self.true_accept_rate, -1.0)))
        return float(self.true_accept_rate[index]), float(self.thresholds[index])

    def as_dict(self, *, max_points: int = 200) -> dict[str, Any]:
        """Serialisable form, subsampled so a report stays readable."""
        count = self.thresholds.size
        step = max(1, count // max_points)
        return {
            "auc": round(self.auc, 6),
            "points": [
                {
                    "threshold": round(float(self.thresholds[i]), 6),
                    "far": round(float(self.false_accept_rate[i]), 6),
                    "tar": round(float(self.true_accept_rate[i]), 6),
                }
                for i in range(0, count, step)
            ],
        }


@dataclass(slots=True)
class EvaluationReport:
    """Everything needed to choose an operating point.

    Attributes:
        genuine: Similarity scores for same-identity pairs.
        impostor: Similarity scores for different-identity pairs.
        roc: The ROC curve.
        equal_error_rate: The rate where FAR equals FRR.
        eer_threshold: The similarity at which that happens.
        operating_points: TAR and threshold at each requested FAR.
        separation: Gap between the worst genuine and best impostor score.
            Positive means the two populations do not overlap at all on this
            corpus.
    """

    genuine: FloatArray
    impostor: FloatArray
    roc: RocCurve
    equal_error_rate: float
    eer_threshold: float
    operating_points: dict[str, dict[str, float]] = field(default_factory=dict)
    separation: float = 0.0

    @property
    def genuine_count(self) -> int:
        """Number of genuine pairs."""
        return int(self.genuine.size)

    @property
    def impostor_count(self) -> int:
        """Number of impostor pairs."""
        return int(self.impostor.size)

    @property
    def separable(self) -> bool:
        """Whether some threshold classifies this corpus perfectly."""
        return self.separation > 0.0

    def confusion_at(self, threshold: float) -> ConfusionMatrix:
        """Build the confusion matrix at a given similarity threshold."""
        return confusion_matrix(self.genuine, self.impostor, threshold)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable summary."""
        return {
            "pairs": {
                "genuine": self.genuine_count,
                "impostor": self.impostor_count,
            },
            "genuine_distribution": _describe(self.genuine),
            "impostor_distribution": _describe(self.impostor),
            "separation": round(self.separation, 6),
            "separable": self.separable,
            "auc": round(self.roc.auc, 6),
            "eer": round(self.equal_error_rate, 6),
            "eer_threshold": round(self.eer_threshold, 6),
            "operating_points": self.operating_points,
        }


def _describe(scores: FloatArray) -> dict[str, float]:
    """Summary statistics of a score distribution."""
    if scores.size == 0:
        return {}
    return {
        "count": int(scores.size),
        "min": round(float(scores.min()), 6),
        "p01": round(float(np.percentile(scores, 1)), 6),
        "p05": round(float(np.percentile(scores, 5)), 6),
        "median": round(float(np.median(scores)), 6),
        "p95": round(float(np.percentile(scores, 95)), 6),
        "p99": round(float(np.percentile(scores, 99)), 6),
        "max": round(float(scores.max()), 6),
        "mean": round(float(scores.mean()), 6),
        "std": round(float(scores.std()), 6),
    }


def confusion_matrix(
    genuine: npt.NDArray[Any], impostor: npt.NDArray[Any], threshold: float
) -> ConfusionMatrix:
    """Count outcomes at one decision threshold.

    A pair is accepted when its similarity is **at or above** the threshold,
    matching the comparison in
    :func:`hamqadam_ai.matching.similarity.decide`.
    """
    genuine_scores = np.asarray(genuine, dtype=np.float64).reshape(-1)
    impostor_scores = np.asarray(impostor, dtype=np.float64).reshape(-1)

    true_positives = int(np.count_nonzero(genuine_scores >= threshold))
    false_negatives = int(genuine_scores.size - true_positives)
    false_positives = int(np.count_nonzero(impostor_scores >= threshold))
    true_negatives = int(impostor_scores.size - false_positives)

    return ConfusionMatrix(
        threshold=float(threshold),
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
    )


def roc_curve(
    genuine: npt.NDArray[Any], impostor: npt.NDArray[Any]
) -> RocCurve:
    """Compute the ROC over every threshold the data distinguishes.

    Thresholds are taken from the observed scores themselves rather than a
    fixed grid, so the curve has full resolution wherever the distributions
    actually live and wastes no points where nothing happens.

    Args:
        genuine: Similarity scores for same-identity pairs.
        impostor: Similarity scores for different-identity pairs.

    Returns:
        The curve, with thresholds in descending order.

    Raises:
        ValueError: if either population is empty. An ROC needs both.
    """
    genuine_scores = np.asarray(genuine, dtype=np.float64).reshape(-1)
    impostor_scores = np.asarray(impostor, dtype=np.float64).reshape(-1)

    if genuine_scores.size == 0 or impostor_scores.size == 0:
        raise ValueError(
            "An ROC needs both genuine and impostor pairs; got "
            f"{genuine_scores.size} genuine and {impostor_scores.size} impostor."
        )

    combined = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.unique(combined)[::-1]
    # Extend past the extremes so the curve reaches (0, 0) and (1, 1).
    thresholds = np.concatenate(
        [[thresholds[0] + 1e-9], thresholds, [thresholds[-1] - 1e-9]]
    )

    # Vectorised: for each threshold, how many of each population clear it.
    genuine_sorted = np.sort(genuine_scores)
    impostor_sorted = np.sort(impostor_scores)

    tar = 1.0 - np.searchsorted(genuine_sorted, thresholds, side="left") / (
        genuine_sorted.size
    )
    far = 1.0 - np.searchsorted(impostor_sorted, thresholds, side="left") / (
        impostor_sorted.size
    )

    # Trapezoidal area, integrating TAR over FAR which increases as the
    # threshold falls.
    order = np.argsort(far)
    area = float(np.trapezoid(tar[order], far[order]))

    return RocCurve(
        thresholds=thresholds,
        false_accept_rate=far,
        true_accept_rate=tar,
        auc=area,
    )


def equal_error_rate(curve: RocCurve) -> tuple[float, float]:
    """Find where the false-accept and false-reject rates cross.

    Args:
        curve: The ROC.

    Returns:
        ``(eer, threshold)``.

    Note:
        The EER weights admitting an impostor and refusing a genuine user
        equally. For identity verification they are not equal - one is a
        security incident and the other an inconvenience - so the EER is
        reported as a summary of separability rather than as a recommended
        operating point.
    """
    frr = 1.0 - curve.true_accept_rate
    difference = np.abs(curve.false_accept_rate - frr)
    index = int(np.argmin(difference))
    eer = float((curve.false_accept_rate[index] + frr[index]) / 2.0)
    return eer, float(curve.thresholds[index])


def evaluate(
    genuine: npt.NDArray[Any],
    impostor: npt.NDArray[Any],
    *,
    far_targets: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5),
) -> EvaluationReport:
    """Produce the full accuracy report for a set of scored pairs.

    Args:
        genuine: Similarity scores for same-identity pairs.
        impostor: Similarity scores for different-identity pairs.
        far_targets: FAR ceilings to report operating points at.

    Returns:
        The report.
    """
    genuine_scores = np.asarray(genuine, dtype=np.float64).reshape(-1)
    impostor_scores = np.asarray(impostor, dtype=np.float64).reshape(-1)

    curve = roc_curve(genuine_scores, impostor_scores)
    eer, eer_threshold = equal_error_rate(curve)

    operating: dict[str, dict[str, float]] = {}
    for target in far_targets:
        # A FAR of 1e-4 cannot be resolved with fewer than 10,000 impostor
        # pairs; reporting a number from 200 would be spurious precision.
        resolvable = impostor_scores.size >= (1.0 / target)
        tar, threshold = curve.tar_at_far(target)
        operating[f"far_{target:g}"] = {
            "target_far": target,
            "tar": round(tar, 6),
            "threshold": round(threshold, 6),
            "resolvable": resolvable,
            "impostor_pairs_needed": int(np.ceil(1.0 / target)),
        }

    separation = float(genuine_scores.min() - impostor_scores.max())

    return EvaluationReport(
        genuine=genuine_scores,
        impostor=impostor_scores,
        roc=curve,
        equal_error_rate=eer,
        eer_threshold=eer_threshold,
        operating_points=operating,
        separation=separation,
    )


def recommend_thresholds(
    report: EvaluationReport,
    *,
    strong_far: float = 1e-4,
    review_far: float = 1e-2,
) -> dict[str, float]:
    """Suggest operating points from a measured report.

    The strong-match boundary is set at a strict FAR because crossing it
    admits somebody to an account. The review boundary is set at a looser one
    because crossing it only routes the case to a human.

    Args:
        report: A measured evaluation report.
        strong_far: FAR ceiling for the strong-match boundary.
        review_far: FAR ceiling for the review boundary.

    Returns:
        ``{"strong_match": ..., "review": ...}``.
    """
    _, strong = report.roc.tar_at_far(strong_far)
    _, review = report.roc.tar_at_far(review_far)

    # Guard the invariant MatchThresholds enforces. With too few impostor
    # pairs both queries can return the same threshold, which would produce an
    # invalid configuration.
    if review >= strong:
        review = strong - 0.05

    return {"strong_match": round(strong, 4), "review": round(review, 4)}


__all__ = [
    "ConfusionMatrix",
    "EvaluationReport",
    "RocCurve",
    "confusion_matrix",
    "equal_error_rate",
    "evaluate",
    "recommend_thresholds",
    "roc_curve",
]
