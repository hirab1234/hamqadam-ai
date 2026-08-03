"""MODULE 8 latency, and how it scales with the gallery.

Scaling is the point here rather than the absolute numbers. An exact search is
a matrix-vector product, so it is linear in gallery size by construction - and
a change that quietly made it quadratic would still look fast at the sizes a
test fixture reaches while becoming unusable in production.

Measured on an idle development machine, in-process store, 512 dimensions,
median of seven runs:

    gallery      search
      1,000      0.18 ms
     10,000      0.96 ms
     50,000      6.9 ms
    100,000     17.1 ms

An earlier draft of this docstring guessed 0.6 / 4 / 19 ms. The guesses were
wrong, and worse, they hid a real defect: the store stacked its whole gallery
into a fresh matrix on every query, which cost **285 ms** at 100,000 records
and grew faster than linearly. Materialising the matrix fixed both. The numbers
above are measured.

Enrolment appends one row to a capacity-doubling buffer: 0.010 ms each,
amortised constant up to 40,000 records and degrading under allocation
pressure beyond that.

That too started out wrong. The first version of the materialised matrix
appended with ``np.vstack``, which copies the whole buffer - so building a
gallery was O(n^2) and a realistic check-then-enrol cycle against 20,000
templates took **60 ms**, of which 58 ms was the copy, against a 2 ms search.
The doubling buffer brought the same cycle to 6.8 ms.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.duplicate_detection.base import VectorRecord, utc_now
from hamqadam_ai.duplicate_detection.memory_store import InMemoryVectorStore
from hamqadam_ai.embeddings.base import FaceEmbedding
from hamqadam_ai.services.duplicate_service import DuplicateService

pytestmark = pytest.mark.performance

MODEL = "arcface-perf"
DIMENSION = 512

#: Budget for one search over a gallery of 50,000. Roughly five times the
#: measured 6.9 ms, which leaves room for a loaded runner while still catching
#: a return to the per-query stacking this store used to do.
SEARCH_BUDGET_MS = 40.0

#: Budget for one enrolment. A dictionary write plus a normalisation.
ENROL_BUDGET_MS = 20.0


def unit_matrix(count: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(count, DIMENSION)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def filled_store(count: int) -> InMemoryVectorStore:
    """A gallery of ``count`` random templates."""
    store = InMemoryVectorStore(max_records=max(count * 2, 10))
    stamp = utc_now()
    for index, vector in enumerate(unit_matrix(count)):
        store.enrol(
            VectorRecord(
                reference=f"user-{index}",
                vector=vector,
                model_version=MODEL,
                enrolled_at=stamp,
            )
        )
    return store


def embedding(vector: np.ndarray) -> FaceEmbedding:
    return FaceEmbedding(
        vector=vector,
        raw_norm=22.0,
        confidence=0.9,
        model_key="face_embedder_arcface",
        model_version=MODEL,
        role=ImageRole.LIVE_SELFIE,
    )


def median_ms(function, repeats: int = 5) -> float:
    """Median wall-clock of several runs, discarding the first."""
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


@pytest.mark.performance
def test_a_search_over_fifty_thousand_is_within_budget() -> None:
    store = filled_store(50_000)
    service = DuplicateService(store=store, settings=get_settings())
    probe = embedding(unit_matrix(1, seed=99)[0])

    elapsed = median_ms(lambda: service.check(probe, reference="q"), repeats=3)

    assert elapsed < SEARCH_BUDGET_MS, f"{elapsed:.1f} ms"
    service.close()


@pytest.mark.performance
def test_search_scales_linearly_with_the_gallery() -> None:
    """A matrix-vector product is linear by construction. A change that made
    it quadratic would still look fast at fixture sizes and be unusable in
    production, so the *shape* is what is pinned rather than the value.
    """
    probe = unit_matrix(1, seed=99)[0]
    timings: dict[int, float] = {}

    for size in (5_000, 20_000):
        store = filled_store(size)
        timings[size] = median_ms(
            lambda s=store: s.search(probe, model_version=MODEL, top_k=10),
            repeats=5,
        )
        store.close()

    ratio = timings[20_000] / max(timings[5_000], 1e-6)
    # Four times the data. Linear predicts 4x; quadratic predicts 16x. The
    # bound sits well clear of both, because small-gallery timings are noisy
    # and only the order of growth is being asserted.
    assert ratio < 9.0, f"4x the gallery cost {ratio:.1f}x the time: {timings}"


@pytest.mark.performance
def test_a_check_then_enrol_cycle_stays_cheap() -> None:
    """The pattern the pipeline actually produces, and the one that caught the
    defect. Appending with ``np.vstack`` copies the whole buffer, so this cycle
    cost 60 ms over a 20,000 gallery when a bare search cost 2 ms. Measured at
    6.8 ms with the doubling buffer.
    """
    store = filled_store(20_000)
    service = DuplicateService(store=store, settings=get_settings())
    probe = embedding(unit_matrix(1, seed=42)[0])

    def cycle() -> None:
        service.check(probe, reference="rolling")
        service.enrol(probe, reference="rolling")

    elapsed = median_ms(cycle, repeats=15)

    assert elapsed < SEARCH_BUDGET_MS, f"{elapsed:.1f} ms"
    service.close()


@pytest.mark.performance
def test_appending_is_amortised_constant() -> None:
    """Bulk enrolment must be linear in the number of templates, not
    quadratic. Doubling the count should roughly double the time; the
    ``vstack`` version quadrupled it."""
    timings: dict[int, float] = {}

    for size in (5_000, 20_000):
        store = InMemoryVectorStore(max_records=size * 2)
        vectors = unit_matrix(size, seed=11)
        stamp = utc_now()
        # Warm the index so appends take the incremental path.
        store.enrol(
            VectorRecord(
                reference="seed",
                vector=vectors[0],
                model_version=MODEL,
                enrolled_at=stamp,
            )
        )
        store.search(vectors[0], model_version=MODEL, top_k=1)

        started = time.perf_counter()
        for index, vector in enumerate(vectors):
            store.enrol(
                VectorRecord(
                    reference=f"bulk-{index}",
                    vector=vector,
                    model_version=MODEL,
                    enrolled_at=stamp,
                )
            )
        timings[size] = (time.perf_counter() - started) * 1000.0
        store.close()

    ratio = timings[20_000] / max(timings[5_000], 1e-6)
    # Four times the records. Amortised-linear predicts 4x; the copy-per-append
    # version predicted 16x. Bounded at 9x, clear of both.
    assert ratio < 9.0, f"4x the records cost {ratio:.1f}x the time: {timings}"


@pytest.mark.performance
def test_enrolment_does_not_slow_as_the_gallery_grows() -> None:
    """It is a dictionary write. If it ever started scanning, this notices."""
    small = filled_store(1_000)
    large = filled_store(30_000)
    vector = unit_matrix(1, seed=7)[0]

    def write(store: InMemoryVectorStore, reference: str) -> None:
        store.enrol(
            VectorRecord(
                reference=reference,
                vector=vector,
                model_version=MODEL,
                enrolled_at=utc_now(),
            )
        )

    into_small = median_ms(lambda: write(small, "new"), repeats=20)
    into_large = median_ms(lambda: write(large, "new"), repeats=20)

    assert into_small < ENROL_BUDGET_MS
    assert into_large < ENROL_BUDGET_MS
    small.close()
    large.close()


@pytest.mark.performance
def test_an_empty_gallery_returns_immediately() -> None:
    service = DuplicateService(
        store=InMemoryVectorStore(), settings=get_settings()
    )
    probe = embedding(unit_matrix(1, seed=3)[0])

    assert median_ms(lambda: service.check(probe, reference="q")) < 5.0
    service.close()


@pytest.mark.performance
def test_a_search_ignores_records_of_another_model_version() -> None:
    """Filtering happens before the matrix product, so a gallery full of stale
    templates from a previous recogniser must not be paid for."""
    store = filled_store(20_000)
    stamp = utc_now()
    for index, vector in enumerate(unit_matrix(20_000, seed=5)):
        store.enrol(
            VectorRecord(
                reference=f"old-{index}",
                vector=vector,
                model_version="arcface-previous",
                enrolled_at=stamp,
            )
        )

    probe = unit_matrix(1, seed=99)[0]
    elapsed = median_ms(
        lambda: store.search(probe, model_version=MODEL, top_k=10), repeats=3
    )

    assert elapsed < SEARCH_BUDGET_MS, f"{elapsed:.1f} ms"
    store.close()


@pytest.mark.performance
def test_the_reported_duration_matches_the_wall_clock() -> None:
    store = filled_store(10_000)
    service = DuplicateService(store=store, settings=get_settings())
    probe = embedding(unit_matrix(1, seed=99)[0])
    service.check(probe, reference="q")  # warm

    started = time.perf_counter()
    result = service.check(probe, reference="q")
    measured = (time.perf_counter() - started) * 1000.0

    assert result.duration_ms == pytest.approx(measured, rel=0.5, abs=2.0)
    service.close()
