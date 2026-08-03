"""Verification accuracy metrics.

These are the numbers thresholds get chosen from, so they are checked against
cases whose answers are known analytically rather than against a reference
implementation. A subtly wrong ROC would produce a confident, wrong operating
point - the worst kind of failure here, because nothing would look broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from hamqadam_ai.matching.evaluation import (
    ConfusionMatrix,
    confusion_matrix,
    equal_error_rate,
    evaluate,
    recommend_thresholds,
    roc_curve,
)

# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_counts_on_a_hand_worked_example() -> None:
    genuine = np.array([0.9, 0.8, 0.7, 0.4])
    impostor = np.array([0.6, 0.3, 0.2, 0.1, 0.05])

    matrix = confusion_matrix(genuine, impostor, threshold=0.5)

    assert matrix.true_positives == 3
    assert matrix.false_negatives == 1
    assert matrix.false_positives == 1
    assert matrix.true_negatives == 4


@pytest.mark.unit
def test_the_threshold_is_inclusive() -> None:
    """Matches ``decide``, which accepts at or above the threshold."""
    matrix = confusion_matrix(np.array([0.5]), np.array([0.5]), threshold=0.5)
    assert matrix.true_positives == 1
    assert matrix.false_positives == 1


@pytest.mark.unit
def test_derived_rates() -> None:
    matrix = ConfusionMatrix(
        threshold=0.5,
        true_positives=90,
        false_positives=10,
        true_negatives=890,
        false_negatives=10,
    )
    assert matrix.false_accept_rate == pytest.approx(10 / 900)
    assert matrix.false_reject_rate == pytest.approx(10 / 100)
    assert matrix.true_accept_rate == pytest.approx(90 / 100)
    assert matrix.precision == pytest.approx(90 / 100)
    assert matrix.recall == pytest.approx(0.9)
    assert matrix.f1 == pytest.approx(0.9)


@pytest.mark.unit
def test_f1_is_zero_when_nothing_is_accepted() -> None:
    matrix = ConfusionMatrix(
        threshold=1.1, true_positives=0, false_positives=0,
        true_negatives=100, false_negatives=50,
    )
    assert matrix.f1 == 0.0


@pytest.mark.unit
def test_accuracy_is_misleading_under_class_imbalance() -> None:
    """Documented in the module as the least useful metric here, and this is
    why: refusing everybody scores 99.5% when impostors outnumber genuine
    pairs 200 to 1."""
    matrix = ConfusionMatrix(
        threshold=1.1, true_positives=0, false_positives=0,
        true_negatives=20_000, false_negatives=100,
    )
    assert matrix.accuracy > 0.99
    assert matrix.recall == 0.0


@pytest.mark.unit
def test_a_perfect_threshold_is_reported_as_perfect() -> None:
    matrix = confusion_matrix(np.array([0.9, 0.8]), np.array([0.1, 0.2]), 0.5)
    assert matrix.false_accept_rate == 0.0
    assert matrix.false_reject_rate == 0.0
    assert matrix.f1 == pytest.approx(1.0)


@pytest.mark.unit
def test_confusion_matrix_serialises() -> None:
    import json

    matrix = confusion_matrix(np.array([0.9]), np.array([0.1]), 0.5)
    json.dumps(matrix.as_dict())


# --------------------------------------------------------------------------- #
# ROC
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_perfect_separation_gives_auc_one() -> None:
    curve = roc_curve(np.array([0.8, 0.9, 1.0]), np.array([0.0, 0.1, 0.2]))
    assert curve.auc == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_identical_distributions_give_auc_one_half() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(0.5, 0.1, 4000)
    other = rng.normal(0.5, 0.1, 4000)
    curve = roc_curve(scores, other)
    assert curve.auc == pytest.approx(0.5, abs=0.03)


@pytest.mark.unit
def test_inverted_separation_gives_auc_near_zero() -> None:
    """Impostors scoring higher than genuine pairs - a wired-up-backwards
    detector would look exactly like this."""
    curve = roc_curve(np.array([0.1, 0.2]), np.array([0.8, 0.9]))
    assert curve.auc < 0.1


@pytest.mark.unit
def test_the_curve_spans_the_full_range() -> None:
    curve = roc_curve(np.array([0.9, 0.7]), np.array([0.3, 0.1]))
    assert curve.false_accept_rate.min() == pytest.approx(0.0)
    assert curve.false_accept_rate.max() == pytest.approx(1.0)
    assert curve.true_accept_rate.min() == pytest.approx(0.0)
    assert curve.true_accept_rate.max() == pytest.approx(1.0)


@pytest.mark.unit
def test_both_rates_are_monotone_in_the_threshold() -> None:
    rng = np.random.default_rng(3)
    curve = roc_curve(rng.normal(0.7, 0.1, 500), rng.normal(0.2, 0.1, 500))
    # Thresholds descend, so both rates must be non-decreasing.
    assert np.all(np.diff(curve.true_accept_rate) >= -1e-12)
    assert np.all(np.diff(curve.false_accept_rate) >= -1e-12)


@pytest.mark.unit
def test_an_roc_needs_both_populations() -> None:
    with pytest.raises(ValueError, match="both genuine and impostor"):
        roc_curve(np.array([0.9]), np.array([]))
    with pytest.raises(ValueError, match="both genuine and impostor"):
        roc_curve(np.array([]), np.array([0.1]))


# --------------------------------------------------------------------------- #
# EER and operating points
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_eer_is_zero_under_perfect_separation() -> None:
    curve = roc_curve(np.array([0.8, 0.9]), np.array([0.1, 0.2]))
    eer, threshold = equal_error_rate(curve)
    assert eer == pytest.approx(0.0, abs=1e-9)
    assert 0.2 < threshold <= 0.8


@pytest.mark.unit
def test_the_eer_is_one_half_for_identical_distributions() -> None:
    rng = np.random.default_rng(1)
    curve = roc_curve(rng.normal(0.5, 0.1, 3000), rng.normal(0.5, 0.1, 3000))
    eer, _ = equal_error_rate(curve)
    assert eer == pytest.approx(0.5, abs=0.05)


@pytest.mark.unit
def test_tar_at_far_respects_the_ceiling() -> None:
    rng = np.random.default_rng(2)
    genuine = rng.normal(0.7, 0.1, 2000)
    impostor = rng.normal(0.1, 0.1, 20000)
    curve = roc_curve(genuine, impostor)

    for target in (1e-2, 1e-3):
        tar, threshold = curve.tar_at_far(target)
        matrix = confusion_matrix(genuine, impostor, threshold)
        assert matrix.false_accept_rate <= target + 1e-9
        assert matrix.true_accept_rate == pytest.approx(tar, abs=1e-6)


@pytest.mark.unit
def test_a_stricter_far_never_yields_a_higher_tar() -> None:
    rng = np.random.default_rng(4)
    curve = roc_curve(rng.normal(0.7, 0.12, 2000), rng.normal(0.1, 0.12, 20000))
    loose, _ = curve.tar_at_far(1e-2)
    strict, _ = curve.tar_at_far(1e-3)
    assert strict <= loose + 1e-12


@pytest.mark.unit
def test_an_unachievable_far_accepts_nothing() -> None:
    """With impostors scoring above every genuine pair, the only threshold
    meeting a tight FAR is one that admits nobody at all.

    Asserted on the semantics rather than on the literal threshold value: the
    curve is extended just past its extremes, so the returned threshold sits
    fractionally above the top score rather than at exactly 1.0.
    """
    genuine = np.array([0.1])
    impostor = np.array([0.9])
    curve = roc_curve(genuine, impostor)

    tar, threshold = curve.tar_at_far(1e-6)

    assert tar == 0.0
    matrix = confusion_matrix(genuine, impostor, threshold)
    assert matrix.true_positives == 0
    assert matrix.false_positives == 0


# --------------------------------------------------------------------------- #
# The full report
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_report_records_resolvability() -> None:
    """A FAR of 1e-4 cannot be resolved with 200 impostor pairs, and reporting
    a number from them would be spurious precision."""
    rng = np.random.default_rng(5)
    report = evaluate(rng.normal(0.7, 0.1, 100), rng.normal(0.1, 0.1, 200))

    assert report.operating_points["far_0.01"]["resolvable"] is True
    assert report.operating_points["far_0.0001"]["resolvable"] is False
    assert report.operating_points["far_0.0001"]["impostor_pairs_needed"] == 10_000


@pytest.mark.unit
def test_separation_is_positive_when_populations_do_not_overlap() -> None:
    report = evaluate(np.array([0.8, 0.9]), np.array([0.1, 0.2]))
    assert report.separation == pytest.approx(0.6)
    assert report.separable is True


@pytest.mark.unit
def test_separation_is_negative_when_populations_overlap() -> None:
    report = evaluate(np.array([0.4, 0.9]), np.array([0.1, 0.6]))
    assert report.separation < 0.0
    assert report.separable is False


@pytest.mark.unit
def test_the_report_describes_both_distributions() -> None:
    rng = np.random.default_rng(6)
    report = evaluate(rng.normal(0.7, 0.1, 500), rng.normal(0.1, 0.1, 2000))
    described = report.as_dict()

    assert described["genuine_distribution"]["count"] == 500
    assert described["impostor_distribution"]["count"] == 2000
    assert described["genuine_distribution"]["median"] > (
        described["impostor_distribution"]["median"]
    )


@pytest.mark.unit
def test_the_report_serialises() -> None:
    import json

    rng = np.random.default_rng(7)
    report = evaluate(rng.normal(0.7, 0.1, 200), rng.normal(0.1, 0.1, 1000))
    payload = report.as_dict()
    payload["roc"] = report.roc.as_dict()
    json.dumps(payload)


@pytest.mark.unit
def test_the_roc_is_subsampled_for_reporting() -> None:
    rng = np.random.default_rng(8)
    report = evaluate(rng.normal(0.7, 0.1, 2000), rng.normal(0.1, 0.1, 8000))
    assert len(report.roc.as_dict(max_points=50)["points"]) <= 60


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_recommendations_are_ordered() -> None:
    """`MatchThresholds` enforces review < strong_match, so a recommendation
    that violated it would produce an unloadable configuration."""
    rng = np.random.default_rng(9)
    report = evaluate(rng.normal(0.7, 0.12, 3000), rng.normal(0.08, 0.1, 40000))
    recommended = recommend_thresholds(report)
    assert recommended["review"] < recommended["strong_match"]


@pytest.mark.unit
def test_recommendations_stay_ordered_on_a_tiny_corpus() -> None:
    """With too few impostor pairs both FAR queries can return the same
    threshold; the guard must still produce a valid pair."""
    report = evaluate(np.array([0.8, 0.85, 0.9]), np.array([0.1, 0.15]))
    recommended = recommend_thresholds(report)
    assert recommended["review"] < recommended["strong_match"]


@pytest.mark.unit
def test_recommendations_can_build_a_valid_config() -> None:
    from hamqadam_ai.core.config import MatchThresholds

    rng = np.random.default_rng(10)
    report = evaluate(rng.normal(0.7, 0.12, 2000), rng.normal(0.08, 0.1, 30000))
    recommended = recommend_thresholds(report)

    thresholds = MatchThresholds(
        strong_match=recommended["strong_match"], review=recommended["review"]
    )
    assert thresholds.strong_match > thresholds.review


@pytest.mark.unit
def test_a_stricter_far_gives_a_higher_strong_threshold() -> None:
    rng = np.random.default_rng(11)
    report = evaluate(rng.normal(0.7, 0.12, 3000), rng.normal(0.08, 0.1, 50000))

    lenient = recommend_thresholds(report, strong_far=1e-2, review_far=1e-1)
    strict = recommend_thresholds(report, strong_far=1e-4, review_far=1e-2)

    assert strict["strong_match"] >= lenient["strong_match"]
