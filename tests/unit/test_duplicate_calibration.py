"""1:N threshold arithmetic, and the simulation behind the module's caveats.

Pure numerics - no store, no models, no images. What is pinned here is the
reasoning the module documents, so that a future change which quietly makes the
threshold look safer than it is has to break a test to do it.
"""

from __future__ import annotations

import numpy as np
import pytest

from hamqadam_ai.duplicate_detection.calibration import (
    measure_impostor_distribution,
    per_comparison_far,
    recommend_threshold,
    system_far,
)

DIMENSION = 512


def isotropic(count: int, *, seed: int = 0, dimension: int = DIMENSION) -> np.ndarray:
    """Unit vectors filling the sphere - the naive impostor model."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(count, dimension)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def on_manifold(count: int, effective_dim: int, *, seed: int = 0) -> np.ndarray:
    """Unit vectors confined to a low-dimensional subspace.

    The realistic model: every enrolled vector is a *face*, so they share a
    manifold whose intrinsic dimension is far below the nominal 512.
    """
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(effective_dim, DIMENSION))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    vectors = rng.normal(size=(count, effective_dim)) @ basis
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(
        np.float32
    )


# --------------------------------------------------------------------------- #
# The compounding
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_single_comparison_is_the_target_itself() -> None:
    assert per_comparison_far(0.01, 1) == pytest.approx(0.01)
    assert per_comparison_far(0.01, 0) == pytest.approx(0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("gallery", "expected"),
    [(1_000, 1.0e-05), (10_000, 1.0e-06), (100_000, 1.0e-07)],
)
def test_the_required_far_falls_with_gallery_size(
    gallery: int, expected: float
) -> None:
    """The whole reason a 1:1 threshold cannot be reused: a query against N
    templates has N chances to match one by accident."""
    assert per_comparison_far(0.01, gallery) == pytest.approx(expected, rel=0.01)


@pytest.mark.unit
def test_an_unremarkable_far_becomes_most_queries() -> None:
    """A per-comparison FAR of 1e-4 is ordinary for a face recogniser. Against
    ten thousand enrolled faces it produces a false duplicate on roughly two
    queries in three."""
    assert system_far(1e-4, 10_000) == pytest.approx(0.632, abs=0.005)
    assert system_far(1e-4, 100) == pytest.approx(0.010, abs=0.001)


@pytest.mark.unit
def test_the_two_directions_invert_each_other() -> None:
    for gallery in (1, 10, 1_000, 50_000):
        far = per_comparison_far(0.01, gallery)
        assert system_far(far, gallery) == pytest.approx(0.01, rel=1e-6)


@pytest.mark.unit
def test_an_empty_gallery_has_no_false_matches() -> None:
    assert system_far(0.5, 0) == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.parametrize("target", [0.0, 1.0, -0.1, 1.5])
def test_an_impossible_target_is_refused(target: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        per_comparison_far(target, 100)


# --------------------------------------------------------------------------- #
# The simulation the module's caveats rest on
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_isotropic_impostors_look_harmless() -> None:
    """The naive model, and the reason it is quoted before being dismissed:
    it says a gallery of thousands never approaches the threshold."""
    statistics = measure_impostor_distribution(isotropic(2_000))

    assert statistics.mean == pytest.approx(0.0, abs=0.01)
    assert statistics.maximum < 0.35


@pytest.mark.unit
@pytest.mark.parametrize(
    ("effective_dim", "at_least"),
    [(128, 0.15), (64, 0.20), (32, 0.30), (16, 0.45)],
)
def test_a_shared_manifold_raises_impostor_similarity(
    effective_dim: int, at_least: float
) -> None:
    """The finding that overturns the naive model. Confining the population to
    a lower-dimensional subspace - which is what "everyone here is a face"
    means - drives impostor similarity up sharply."""
    statistics = measure_impostor_distribution(on_manifold(1_500, effective_dim))
    assert statistics.percentiles[99.9] >= at_least


@pytest.mark.unit
def test_impostor_similarity_rises_monotonically_as_dimension_falls() -> None:
    """The sensitivity that makes the threshold unknowable here: the answer
    moves steadily across a parameter nobody in this repository can measure."""
    peaks = [
        measure_impostor_distribution(on_manifold(1_000, dim)).percentiles[99.9]
        for dim in (256, 128, 64, 32, 16)
    ]
    assert peaks == sorted(peaks)


@pytest.mark.unit
def test_a_low_dimensional_manifold_defeats_the_configured_threshold() -> None:
    """At an effective dimension of 16 the impostor tail passes 0.68 - the
    shipped duplicate threshold - so a gallery of that shape would report a
    duplicate for almost every query."""
    statistics = measure_impostor_distribution(on_manifold(2_000, 16))
    assert statistics.maximum > 0.68


# --------------------------------------------------------------------------- #
# Measuring a distribution
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_gallery_of_one_yields_no_pairs() -> None:
    statistics = measure_impostor_distribution(isotropic(1))
    assert statistics.sample_count == 0


@pytest.mark.unit
def test_an_empty_gallery_yields_no_pairs() -> None:
    statistics = measure_impostor_distribution(
        np.zeros((0, DIMENSION), dtype=np.float32)
    )
    assert statistics.sample_count == 0


@pytest.mark.unit
def test_all_pairs_are_used_when_the_gallery_is_small() -> None:
    statistics = measure_impostor_distribution(isotropic(50))
    assert statistics.sample_count == 50 * 49 // 2


@pytest.mark.unit
def test_pairs_are_sampled_when_the_gallery_is_large() -> None:
    """The full set grows quadratically: a hundred thousand templates is five
    billion pairs."""
    statistics = measure_impostor_distribution(isotropic(400), max_pairs=1_000)
    assert statistics.sample_count <= 1_000


@pytest.mark.unit
def test_measurement_is_deterministic() -> None:
    vectors = isotropic(300)
    first = measure_impostor_distribution(vectors, max_pairs=500, seed=7)
    second = measure_impostor_distribution(vectors, max_pairs=500, seed=7)

    assert first.as_dict() == second.as_dict()


@pytest.mark.unit
def test_the_statistics_serialise() -> None:
    import json

    json.dumps(measure_impostor_distribution(isotropic(100)).as_dict())


# --------------------------------------------------------------------------- #
# Recommending a threshold
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_recommendation_rises_with_gallery_size() -> None:
    """The point of the exercise: the same population needs a stricter
    threshold once there is more of it to be wrong about."""
    vectors = isotropic(1_500)
    small = recommend_threshold(vectors, gallery_size=1_000)
    large = recommend_threshold(vectors, gallery_size=1_000_000)

    assert large.threshold > small.threshold


@pytest.mark.unit
def test_a_recommendation_rises_as_the_target_tightens() -> None:
    vectors = isotropic(1_500)
    loose = recommend_threshold(vectors, gallery_size=1_000, target_system_far=0.05)
    tight = recommend_threshold(vectors, gallery_size=1_000, target_system_far=0.0001)

    assert tight.threshold > loose.threshold


@pytest.mark.unit
def test_a_threshold_within_the_data_is_not_flagged_extrapolated() -> None:
    """A million pairs resolve a 1e-5 tail directly, so nothing is guessed."""
    result = recommend_threshold(
        isotropic(2_000), gallery_size=1_000, target_system_far=0.01
    )

    assert result.extrapolated is False
    assert result.threshold <= result.statistics.maximum


@pytest.mark.unit
def test_a_threshold_beyond_the_data_says_so() -> None:
    """An extrapolated threshold is a guess with arithmetic attached, and the
    caller has to be able to tell."""
    result = recommend_threshold(
        isotropic(200), gallery_size=10_000_000, target_system_far=0.001
    )

    assert result.extrapolated is True


@pytest.mark.unit
def test_an_empty_gallery_recommends_refusing_everything() -> None:
    """With nothing measured, the only defensible threshold is one nothing
    clears."""
    result = recommend_threshold(np.zeros((0, DIMENSION), dtype=np.float32))

    assert result.threshold == pytest.approx(1.0)
    assert result.extrapolated is True


@pytest.mark.unit
def test_a_recommendation_carries_its_evidence() -> None:
    result = recommend_threshold(isotropic(500), gallery_size=5_000)

    assert result.statistics.sample_count > 0
    assert result.gallery_size == 5_000
    assert 0.0 < result.per_comparison_far < 1.0


@pytest.mark.unit
def test_a_recommendation_serialises() -> None:
    import json

    json.dumps(recommend_threshold(isotropic(200), gallery_size=1_000).as_dict())


@pytest.mark.unit
def test_the_threshold_stays_within_the_cosine_range() -> None:
    """Extrapolating a Gaussian tail far enough would otherwise recommend a
    similarity above 1, which nothing can ever reach."""
    result = recommend_threshold(
        isotropic(100), gallery_size=10**12, target_system_far=1e-6
    )

    assert -1.0 <= result.threshold <= 1.0


@pytest.mark.unit
@pytest.mark.parametrize("gallery", [10**6, 10**9, 10**12, 10**15])
def test_a_huge_gallery_does_not_underflow_to_zero(gallery: int) -> None:
    """Written directly, ``1 - (1 - target) ** (1 / n)`` cancels catastrophically:
    at n = 1e12 the intermediate rounds to exactly 1.0 and the result is
    **zero**, which reads as "no threshold can achieve this" and is off by
    eighteen orders of magnitude. Computed through log1p/expm1 it is 1e-18.
    """
    far = per_comparison_far(1e-6, gallery)

    assert far > 0.0
    assert far == pytest.approx(1e-6 / gallery, rel=1e-6)


@pytest.mark.unit
def test_a_manifold_gallery_needs_a_higher_threshold_than_an_isotropic_one() -> None:
    """The practical consequence of the sensitivity: two galleries of the same
    size can need very different operating points, and only measurement
    distinguishes them."""
    flat = recommend_threshold(isotropic(1_000), gallery_size=10_000)
    clustered = recommend_threshold(on_manifold(1_000, 32), gallery_size=10_000)

    assert clustered.threshold > flat.threshold
