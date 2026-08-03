"""The gallery contract, run against every adapter.

Parametrised over both stores on purpose. Two adapters that disagree about
what "already enrolled" means, or about whether a deleted reference is gone,
would give a deployment a different duplicate verdict depending on how it was
configured - and that is precisely the sort of difference nobody notices until
it matters.

The Qdrant cases use the client's **embedded engine**, which runs the real
query path rather than a hand-written fake. That distinction earned its keep
immediately: the real engine returns cosine ``1.0000000158616884`` for a
vector matched against itself, which the response schema rejected outright. A
fake would have returned exactly 1.0 and hidden it until production.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from hamqadam_ai.duplicate_detection.base import (
    SearchHit,
    VectorRecord,
    VectorStore,
    utc_now,
)
from hamqadam_ai.duplicate_detection.memory_store import InMemoryVectorStore

MODEL = "arcface-test-1"
OTHER_MODEL = "arcface-test-2"
DIMENSION = 64


def unit(seed: int, dimension: int = DIMENSION) -> np.ndarray:
    """A deterministic unit vector."""
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dimension).astype(np.float32)
    return vector / np.linalg.norm(vector)


def record(
    reference: str, seed: int, *, model_version: str = MODEL, **metadata: object
) -> VectorRecord:
    """One enrolled template."""
    return VectorRecord(
        reference=reference,
        vector=unit(seed),
        model_version=model_version,
        enrolled_at=utc_now(),
        metadata=dict(metadata),
    )


def make_memory() -> VectorStore:
    return InMemoryVectorStore(max_records=1000)


def make_qdrant() -> VectorStore:
    pytest.importorskip("qdrant_client")
    from hamqadam_ai.duplicate_detection.qdrant_store import QdrantVectorStore

    return QdrantVectorStore(
        url=":memory:", collection="contract", dimension=DIMENSION
    )


@pytest.fixture(params=["memory", "qdrant"])
def store(request: pytest.FixtureRequest):  # noqa: ANN201 - pytest fixture
    """One gallery adapter, torn down after the test."""
    built = make_memory() if request.param == "memory" else make_qdrant()
    yield built
    built.close()


# --------------------------------------------------------------------------- #
# Enrolment
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_an_enrolled_template_is_found(store: VectorStore) -> None:
    store.enrol(record("alice", 1))
    hits = store.search(unit(1), model_version=MODEL, top_k=5)

    assert len(hits) == 1
    assert hits[0].reference == "alice"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-4)


@pytest.mark.unit
def test_a_similarity_never_exceeds_one(store: VectorStore) -> None:
    """A cosine cannot exceed 1, but a backend computing one in float32 can
    report that it did. Qdrant returned 1.0000000158616884 matching a vector
    against itself, and the response schema rejected it - taking the whole
    verification down rather than the one field."""
    store.enrol(record("alice", 1))
    hits = store.search(unit(1), model_version=MODEL, top_k=5)

    assert -1.0 <= hits[0].similarity <= 1.0


@pytest.mark.unit
def test_re_enrolling_replaces_rather_than_appends(store: VectorStore) -> None:
    """Several templates for one person would each match the next query, so a
    returning user would accumulate their own false duplicates."""
    store.enrol(record("alice", 1))
    store.enrol(record("alice", 2))

    assert store.count(model_version=MODEL) == 1
    hits = store.search(unit(2), model_version=MODEL, top_k=5)
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-4)


@pytest.mark.unit
def test_exists_is_a_lookup_not_a_nearest_neighbour_search(
    store: VectorStore,
) -> None:
    """The bug this method replaced: inferring existence from a top-1 search
    reports "new" whenever somebody else's template happens to be nearer."""
    store.enrol(record("alice", 1))
    store.enrol(record("bob", 2))

    assert store.exists("alice") is True
    assert store.exists("bob") is True
    assert store.exists("carol") is False


@pytest.mark.unit
def test_a_zero_vector_cannot_be_enrolled() -> None:
    """It carries no direction, so every cosine against it is undefined."""
    with pytest.raises(ValueError, match="zero vector"):
        VectorRecord(
            reference="alice",
            vector=np.zeros(DIMENSION, dtype=np.float32),
            model_version=MODEL,
            enrolled_at=utc_now(),
        )


@pytest.mark.unit
def test_a_record_normalises_its_vector() -> None:
    """So the two stores agree, and so the calibration script can read vectors
    out of either and get comparable numbers."""
    unnormalised = np.full(DIMENSION, 3.0, dtype=np.float32)
    stored = VectorRecord(
        reference="alice",
        vector=unnormalised,
        model_version=MODEL,
        enrolled_at=utc_now(),
    )

    assert float(np.linalg.norm(stored.vector)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_an_empty_gallery_returns_nothing(store: VectorStore) -> None:
    """The normal state of a new deployment, not an error."""
    assert store.search(unit(1), model_version=MODEL, top_k=5) == []


@pytest.mark.unit
def test_hits_come_back_best_first(store: VectorStore) -> None:
    query = unit(1)
    for index in range(6):
        vector = query * (1.0 - index * 0.15) + unit(100 + index) * (index * 0.15)
        store.enrol(
            VectorRecord(
                reference=f"user-{index}",
                vector=vector,
                model_version=MODEL,
                enrolled_at=utc_now(),
            )
        )

    hits = store.search(query, model_version=MODEL, top_k=6)
    similarities = [hit.similarity for hit in hits]

    assert similarities == sorted(similarities, reverse=True)


@pytest.mark.unit
def test_top_k_bounds_the_result(store: VectorStore) -> None:
    for index in range(10):
        store.enrol(record(f"user-{index}", index))

    assert len(store.search(unit(0), model_version=MODEL, top_k=3)) == 3


@pytest.mark.unit
def test_the_querying_user_can_be_excluded(store: VectorStore) -> None:
    """Load-bearing. Without it a returning user matches their own template at
    cosine 1.0 and every re-verification is a duplicate."""
    store.enrol(record("alice", 1))
    store.enrol(record("bob", 2))

    hits = store.search(unit(1), model_version=MODEL, top_k=5, exclude="alice")

    assert [hit.reference for hit in hits] == ["bob"]


@pytest.mark.unit
def test_excluding_the_only_entry_leaves_nothing(store: VectorStore) -> None:
    store.enrol(record("alice", 1))
    assert store.search(unit(1), model_version=MODEL, top_k=5, exclude="alice") == []


@pytest.mark.unit
def test_exclusion_does_not_cost_the_caller_a_result(store: VectorStore) -> None:
    """Asking for three and excluding one must still return three when the
    gallery holds them."""
    for index in range(5):
        store.enrol(record(f"user-{index}", index))

    hits = store.search(unit(0), model_version=MODEL, top_k=3, exclude="user-0")

    assert len(hits) == 3
    assert "user-0" not in {hit.reference for hit in hits}


@pytest.mark.unit
def test_a_search_never_crosses_model_versions(store: VectorStore) -> None:
    """Two embeddings from different recogniser builds are not comparable.
    Comparing them anyway yields scores that look ordinary and mean nothing."""
    store.enrol(record("alice", 1, model_version=MODEL))
    store.enrol(record("bob", 1, model_version=OTHER_MODEL))

    hits = store.search(unit(1), model_version=MODEL, top_k=5)

    assert [hit.reference for hit in hits] == ["alice"]
    assert all(hit.model_version == MODEL for hit in hits)


@pytest.mark.unit
def test_a_zero_query_finds_nothing(store: VectorStore) -> None:
    store.enrol(record("alice", 1))
    zero = np.zeros(DIMENSION, dtype=np.float32)

    assert store.search(zero, model_version=MODEL, top_k=5) == []


@pytest.mark.unit
def test_metadata_survives_the_round_trip(store: VectorStore) -> None:
    store.enrol(record("alice", 1, tier="gold", region="north"))
    hits = store.search(unit(1), model_version=MODEL, top_k=1)

    assert hits[0].metadata["tier"] == "gold"


# --------------------------------------------------------------------------- #
# Erasure
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_template_can_be_erased(store: VectorStore) -> None:
    """A service that stores face templates and cannot delete one on request
    cannot lawfully be deployed."""
    store.enrol(record("alice", 1))

    assert store.delete("alice") is True
    assert store.search(unit(1), model_version=MODEL, top_k=5) == []
    assert store.exists("alice") is False


@pytest.mark.unit
def test_erasure_is_idempotent(store: VectorStore) -> None:
    """A caller retrying an erasure must not be told it failed the second
    time; that is how erasure requests get abandoned half-done."""
    store.enrol(record("alice", 1))

    assert store.delete("alice") is True
    assert store.delete("alice") is False


@pytest.mark.unit
def test_erasing_one_leaves_the_others(store: VectorStore) -> None:
    store.enrol(record("alice", 1))
    store.enrol(record("bob", 2))

    store.delete("alice")

    assert store.count(model_version=MODEL) == 1
    assert store.exists("bob") is True


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_counting_is_per_model_version(store: VectorStore) -> None:
    """A caller cannot interpret a similarity without knowing how many
    *comparable* entries it beat - and entries from another model version were
    never in the running."""
    store.enrol(record("alice", 1, model_version=MODEL))
    store.enrol(record("bob", 2, model_version=MODEL))
    store.enrol(record("carol", 3, model_version=OTHER_MODEL))

    assert store.count(model_version=MODEL) == 2
    assert store.count(model_version=OTHER_MODEL) == 1
    assert store.count() == 3


@pytest.mark.unit
def test_an_empty_gallery_counts_zero(store: VectorStore) -> None:
    assert store.count() == 0


@pytest.mark.unit
def test_health_reports_the_adapter(store: VectorStore) -> None:
    import json

    health = store.health()
    json.dumps(health)

    assert health["store"] == store.name
    assert "durable" in health


# --------------------------------------------------------------------------- #
# Memory-store specifics
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_memory_gallery_refuses_to_grow_without_bound() -> None:
    """An unbounded in-process gallery is a memory leak with a plausible
    excuse. Evicting silently would be worse: it would stop detecting
    duplicates of long-standing users without saying so."""
    store = InMemoryVectorStore(max_records=3)
    for index in range(3):
        store.enrol(record(f"user-{index}", index))

    with pytest.raises(MemoryError, match="Qdrant"):
        store.enrol(record("one-too-many", 99))


@pytest.mark.unit
def test_replacing_at_capacity_is_allowed() -> None:
    """The cap is on distinct people, not on writes. A user at capacity must
    still be able to re-verify."""
    store = InMemoryVectorStore(max_records=2)
    store.enrol(record("alice", 1))
    store.enrol(record("bob", 2))

    store.enrol(record("alice", 3))  # must not raise

    assert store.count() == 2


@pytest.mark.unit
def test_closing_the_memory_gallery_drops_the_templates() -> None:
    """The gallery is biometric data. Releasing the store releases it, rather
    than leaving it for the garbage collector to get to eventually."""
    store = InMemoryVectorStore()
    store.enrol(record("alice", 1))

    store.close()

    assert store.count() == 0


@pytest.mark.unit
def test_the_memory_gallery_exposes_vectors_as_a_copy() -> None:
    """The calibration script needs the whole gallery; mutating it through
    that handle must not corrupt the store."""
    store = InMemoryVectorStore()
    store.enrol(record("alice", 1))

    matrix = store.vectors_for(MODEL)
    matrix[0, 0] = 99.0

    assert store.search(unit(1), model_version=MODEL, top_k=1)[0].similarity == (
        pytest.approx(1.0, abs=1e-4)
    )


@pytest.mark.unit
def test_the_memory_gallery_is_thread_safe() -> None:
    """Uvicorn dispatches to a thread pool, so concurrent enrolment is real.
    A torn read would produce a score against half-written memory rather than
    an error anybody would notice."""
    import concurrent.futures

    store = InMemoryVectorStore(max_records=500)

    def work(index: int) -> None:
        store.enrol(record(f"user-{index}", index))
        store.search(unit(index), model_version=MODEL, top_k=3)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(200)))

    assert store.count() == 200


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_record_summary_never_carries_the_vector() -> None:
    """An embedding is biometric data. It is not a diagnostic and it does not
    go in a log line."""
    summary = record("alice", 1).describe()

    assert "vector" not in summary
    assert set(summary) == {
        "reference", "model_version", "dimension", "enrolled_at"
    }


@pytest.mark.unit
def test_a_hit_serialises() -> None:
    import json

    hit = SearchHit(
        reference="alice",
        similarity=0.9,
        model_version=MODEL,
        enrolled_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
    )
    json.dumps(hit.as_dict())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"), [(1.0000001, 1.0), (-1.0000001, -1.0), (0.5, 0.5)]
)
def test_a_hit_clamps_its_similarity(raw: float, expected: float) -> None:
    hit = SearchHit(reference="a", similarity=raw, model_version=MODEL)
    assert hit.similarity == pytest.approx(expected)
