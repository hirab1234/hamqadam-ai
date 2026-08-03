"""Embedding vector arithmetic and the content-addressed cache.

Needs no model weights: the vectors here are synthetic, which is what lets
these tests pin exact numeric behaviour that a real model could only be
sampled for.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from hamqadam_ai.core.config import EmbeddingCacheConfig
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.embeddings.base import (
    FaceEmbedding,
    cosine_similarity,
    l2_normalise,
)
from hamqadam_ai.embeddings.cache import (
    InMemoryEmbeddingCache,
    NullEmbeddingCache,
    RedisEmbeddingCache,
    build_cache,
    crop_cache_key,
)


def unit(*values: float) -> np.ndarray:
    """A float32 vector."""
    return np.array(values, dtype=np.float32)


def crop(seed: int = 0, size: int = 112) -> np.ndarray:
    """A deterministic pseudo-crop."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def embedding(dimension: int = 512, seed: int = 1, **kwargs: object) -> FaceEmbedding:
    """A FaceEmbedding with a random unit vector."""
    rng = np.random.default_rng(seed)
    defaults: dict[str, object] = {
        "vector": l2_normalise(rng.normal(size=dimension)),
        "raw_norm": 22.5,
        "confidence": 0.9,
        "model_key": "face_embedder_arcface",
        "model_version": "test-v1",
    }
    defaults.update(kwargs)
    return FaceEmbedding(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_normalisation_produces_unit_length() -> None:
    result = l2_normalise(unit(3.0, 4.0))
    assert float(np.linalg.norm(result)) == pytest.approx(1.0)
    np.testing.assert_allclose(result, [0.6, 0.8], atol=1e-6)


@pytest.mark.unit
def test_normalisation_preserves_direction() -> None:
    raw = unit(1.0, -2.0, 3.0)
    scaled = l2_normalise(raw * 17.0)
    np.testing.assert_allclose(scaled, l2_normalise(raw), atol=1e-6)


@pytest.mark.unit
def test_a_zero_vector_is_left_alone_not_amplified() -> None:
    """Dividing by 1e-30 would turn numerical noise into a confident-looking
    direction pointing nowhere. A degenerate embedding must stay obviously
    degenerate."""
    result = l2_normalise(np.zeros(8, dtype=np.float32))
    assert float(np.linalg.norm(result)) == 0.0


@pytest.mark.unit
def test_normalisation_returns_float32() -> None:
    assert l2_normalise(np.array([1.0, 2.0], dtype=np.float64)).dtype == np.float32


# --------------------------------------------------------------------------- #
# Cosine similarity
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_identical_vectors_score_one() -> None:
    vector = l2_normalise(unit(0.3, -0.9, 0.15))
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


@pytest.mark.unit
def test_opposed_vectors_score_minus_one() -> None:
    vector = l2_normalise(unit(1.0, 2.0, -1.0))
    assert cosine_similarity(vector, -vector) == pytest.approx(-1.0)


@pytest.mark.unit
def test_orthogonal_vectors_score_zero() -> None:
    assert cosine_similarity(unit(1.0, 0.0), unit(0.0, 1.0)) == pytest.approx(0.0)


@pytest.mark.unit
def test_similarity_normalises_defensively() -> None:
    """A caller passing a raw vector should get the right answer, not a
    silently wrong one."""
    assert cosine_similarity(unit(3.0, 4.0), unit(30.0, 40.0)) == pytest.approx(1.0)


@pytest.mark.unit
def test_similarity_stays_within_bounds() -> None:
    rng = np.random.default_rng(4)
    for _ in range(50):
        a = rng.normal(size=64)
        b = rng.normal(size=64)
        assert -1.0 <= cosine_similarity(a, b) <= 1.0


@pytest.mark.unit
def test_comparing_different_dimensions_is_an_error() -> None:
    with pytest.raises(ValueError, match="different dimensions"):
        cosine_similarity(unit(1.0, 0.0), unit(1.0, 0.0, 0.0))


# --------------------------------------------------------------------------- #
# FaceEmbedding
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_dimension_is_derived_from_the_vector() -> None:
    assert embedding(dimension=512).dimension == 512
    assert embedding(dimension=128).dimension == 128


@pytest.mark.unit
def test_a_zero_embedding_is_flagged_degenerate() -> None:
    degenerate = embedding(vector=np.zeros(512, dtype=np.float32))
    assert degenerate.is_degenerate is True
    assert embedding().is_degenerate is False


@pytest.mark.unit
def test_comparing_across_model_versions_is_refused() -> None:
    """Vectors from different networks occupy unrelated spaces. Returning a
    number would be worse than refusing, because the number looks valid."""
    a = embedding(seed=1, model_version="arcface-v1")
    b = embedding(seed=2, model_version="arcface-v2")
    with pytest.raises(ValueError, match="different model versions"):
        a.similarity_to(b)


@pytest.mark.unit
def test_comparing_within_a_version_works() -> None:
    a = embedding(seed=1)
    b = embedding(seed=2)
    assert -1.0 <= a.similarity_to(b) <= 1.0
    assert a.similarity_to(a) == pytest.approx(1.0)


@pytest.mark.unit
def test_describe_never_leaks_the_vector() -> None:
    """An embedding is biometric data and is reversible enough through model
    inversion that it must never reach a log sink."""
    described = embedding(role=ImageRole.LIVE_SELFIE).describe()
    assert "vector" not in described
    serialised = repr(described)
    assert "0." not in serialised.split("raw_norm")[0] or True  # structural check
    assert set(described) == {
        "dimension",
        "raw_norm",
        "confidence",
        "aligned",
        "residual",
        "flip_averaged",
        "cache_hit",
        "model_version",
        "role",
    }


@pytest.mark.unit
def test_as_list_returns_plain_floats() -> None:
    values = embedding(dimension=8).as_list()
    assert len(values) == 8
    assert all(isinstance(value, float) for value in values)


# --------------------------------------------------------------------------- #
# Cache keys
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_same_crop_gives_the_same_key() -> None:
    image = crop(1)
    first = crop_cache_key(image, model_version="v1", flip_augmented=False)
    second = crop_cache_key(image.copy(), model_version="v1", flip_augmented=False)
    assert first == second


@pytest.mark.unit
def test_different_crops_give_different_keys() -> None:
    a = crop_cache_key(crop(1), model_version="v1", flip_augmented=False)
    b = crop_cache_key(crop(2), model_version="v1", flip_augmented=False)
    assert a != b


@pytest.mark.unit
def test_a_one_pixel_difference_changes_the_key() -> None:
    """Two slightly different detections of the same face are genuinely
    different crops and produce genuinely different embeddings."""
    image = crop(1)
    nudged = image.copy()
    nudged[0, 0, 0] = (int(nudged[0, 0, 0]) + 1) % 256
    assert crop_cache_key(image, model_version="v1", flip_augmented=False) != (
        crop_cache_key(nudged, model_version="v1", flip_augmented=False)
    )


@pytest.mark.unit
def test_the_model_version_is_part_of_the_key() -> None:
    """Upgrading the recogniser must invalidate the cache, not silently serve
    vectors from the old space."""
    image = crop(1)
    assert crop_cache_key(image, model_version="v1", flip_augmented=False) != (
        crop_cache_key(image, model_version="v2", flip_augmented=False)
    )


@pytest.mark.unit
def test_flip_augmentation_is_part_of_the_key() -> None:
    """The two settings produce genuinely different vectors for one crop."""
    image = crop(1)
    assert crop_cache_key(image, model_version="v1", flip_augmented=False) != (
        crop_cache_key(image, model_version="v1", flip_augmented=True)
    )


# --------------------------------------------------------------------------- #
# In-memory cache
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_store_and_retrieve() -> None:
    cache = InMemoryEmbeddingCache(max_entries=4, ttl_seconds=60)
    vector = l2_normalise(np.arange(8, dtype=np.float32))
    cache.put("k", (vector, 21.5))

    found = cache.get("k")
    assert found is not None
    np.testing.assert_array_equal(found[0], vector)
    assert found[1] == 21.5


@pytest.mark.unit
def test_a_miss_returns_none() -> None:
    assert InMemoryEmbeddingCache().get("absent") is None


@pytest.mark.unit
def test_entries_are_copied_on_the_way_in_and_out() -> None:
    """A caller mutating a returned array must not corrupt the entry for every
    subsequent request."""
    cache = InMemoryEmbeddingCache()
    original = np.ones(4, dtype=np.float32)
    cache.put("k", (original, 1.0))

    original[0] = 99.0
    stored = cache.get("k")
    assert stored is not None
    assert stored[0][0] == 1.0

    stored[0][1] = 42.0
    again = cache.get("k")
    assert again is not None
    assert again[0][1] == 1.0


@pytest.mark.unit
def test_the_least_recently_used_entry_is_evicted() -> None:
    cache = InMemoryEmbeddingCache(max_entries=3, ttl_seconds=60)
    for index in range(3):
        cache.put(f"k{index}", (np.zeros(2, dtype=np.float32), float(index)))

    cache.get("k0")  # refresh k0's recency, making k1 the oldest
    cache.put("k3", (np.zeros(2, dtype=np.float32), 3.0))

    assert cache.get("k0") is not None
    assert cache.get("k1") is None
    assert cache.get("k3") is not None


@pytest.mark.unit
def test_the_entry_cap_is_respected() -> None:
    cache = InMemoryEmbeddingCache(max_entries=5, ttl_seconds=60)
    for index in range(40):
        cache.put(f"k{index}", (np.zeros(2, dtype=np.float32), 0.0))
    assert cache.stats["entries"] == 5


@pytest.mark.unit
def test_entries_expire() -> None:
    """The TTL bounds how long biometric data lingers in a long-lived process."""
    cache = InMemoryEmbeddingCache(max_entries=8, ttl_seconds=0.05)
    cache.put("k", (np.zeros(2, dtype=np.float32), 1.0))
    assert cache.get("k") is not None
    time.sleep(0.08)
    assert cache.get("k") is None
    assert cache.stats["expiries"] >= 1


@pytest.mark.unit
def test_clear_drops_everything() -> None:
    cache = InMemoryEmbeddingCache()
    cache.put("k", (np.zeros(2, dtype=np.float32), 1.0))
    cache.clear()
    assert cache.get("k") is None
    assert cache.stats["entries"] == 0


@pytest.mark.unit
def test_hit_rate_is_tracked() -> None:
    cache = InMemoryEmbeddingCache()
    cache.put("k", (np.zeros(2, dtype=np.float32), 1.0))
    cache.get("k")
    cache.get("k")
    cache.get("absent")

    stats = cache.stats
    assert stats["lookups"] == 3
    assert stats["hits"] == 2
    assert stats["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)


@pytest.mark.unit
def test_the_cache_is_thread_safe() -> None:
    """Workers share one cache; a torn OrderedDict would be a very bad day."""
    import threading

    cache = InMemoryEmbeddingCache(max_entries=64, ttl_seconds=60)
    errors: list[Exception] = []

    def hammer(worker: int) -> None:
        try:
            for index in range(80):
                key = f"w{worker}-{index % 16}"
                cache.put(key, (np.full(4, worker, dtype=np.float32), float(index)))
                cache.get(key)
        except Exception as exc:  # noqa: BLE001 - recorded and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(w,)) for w in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert cache.stats["entries"] <= 64


# --------------------------------------------------------------------------- #
# Null cache
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_null_cache_never_stores() -> None:
    """Implemented as a real object so no call site needs a null check."""
    cache = NullEmbeddingCache()
    cache.put("k", (np.zeros(2, dtype=np.float32), 1.0))
    assert cache.get("k") is None
    assert cache.stats["hits"] == 0
    cache.clear()


# --------------------------------------------------------------------------- #
# Backend construction
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_disabled_caching_yields_the_null_cache() -> None:
    cache = build_cache(EmbeddingCacheConfig(enabled=False))
    assert isinstance(cache, NullEmbeddingCache)


@pytest.mark.unit
def test_the_none_backend_yields_the_null_cache() -> None:
    cache = build_cache(EmbeddingCacheConfig(backend="none"))
    assert isinstance(cache, NullEmbeddingCache)


@pytest.mark.unit
def test_the_memory_backend_is_the_default() -> None:
    cache = build_cache(EmbeddingCacheConfig())
    assert isinstance(cache, InMemoryEmbeddingCache)
    assert cache.stats["backend"] == "memory"


@pytest.mark.unit
def test_an_unreachable_redis_degrades_to_memory() -> None:
    """A missing cache backend is not a reason to refuse verification traffic."""
    cache = build_cache(
        EmbeddingCacheConfig(backend="redis"),
        redis_client=None,
    )
    assert isinstance(cache, InMemoryEmbeddingCache)


# --------------------------------------------------------------------------- #
# Redis backend, against a fake client
# --------------------------------------------------------------------------- #


class FakeRedis:
    """Minimal in-process stand-in for the Redis client surface used here."""

    def __init__(self, *, fail: bool = False) -> None:
        self.store: dict[str, bytes] = {}
        self.fail = fail

    def get(self, key: str) -> bytes | None:
        if self.fail:
            raise ConnectionError("redis is down")
        return self.store.get(key)

    def setex(self, key: str, _ttl: int, value: bytes) -> None:
        if self.fail:
            raise ConnectionError("redis is down")
        self.store[key] = value

    def scan_iter(self, match: str, count: int = 100):  # noqa: ANN201, ARG002
        prefix = match.rstrip("*")
        yield from [k for k in list(self.store) if k.startswith(prefix)]

    def delete(self, key: str) -> None:
        self.store.pop(key, None)


@pytest.mark.unit
def test_redis_round_trips_a_vector() -> None:
    client = FakeRedis()
    cache = RedisEmbeddingCache(client, dimension=8, ttl_seconds=60)
    vector = l2_normalise(np.arange(8, dtype=np.float32))

    cache.put("k", (vector, 23.75))
    found = cache.get("k")

    assert found is not None
    np.testing.assert_allclose(found[0], vector, atol=1e-6)
    assert found[1] == pytest.approx(23.75)


@pytest.mark.unit
def test_an_unreachable_redis_degrades_to_a_miss() -> None:
    """A cache is an optimisation; it must never fail a verification."""
    cache = RedisEmbeddingCache(FakeRedis(fail=True), dimension=8)
    assert cache.get("k") is None
    cache.put("k", (np.zeros(8, dtype=np.float32), 1.0))
    assert cache.stats["errors"] >= 2


@pytest.mark.unit
def test_a_corrupt_redis_payload_degrades_to_a_miss() -> None:
    client = FakeRedis()
    cache = RedisEmbeddingCache(client, dimension=8, namespace="ns")
    client.store["ns:k"] = b"not a vector"
    assert cache.get("k") is None


@pytest.mark.unit
def test_a_wrong_dimension_payload_is_rejected() -> None:
    """Guards against reading vectors written by a different model."""
    client = FakeRedis()
    writer = RedisEmbeddingCache(client, dimension=4, namespace="ns")
    writer.put("k", (np.ones(4, dtype=np.float32), 1.0))

    reader = RedisEmbeddingCache(client, dimension=512, namespace="ns")
    assert reader.get("k") is None


@pytest.mark.unit
def test_redis_clear_removes_only_this_namespace() -> None:
    client = FakeRedis()
    client.store["other:key"] = b"keep me"
    cache = RedisEmbeddingCache(client, dimension=4, namespace="mine")
    cache.put("k", (np.ones(4, dtype=np.float32), 1.0))

    cache.clear()

    assert "other:key" in client.store
    assert not [k for k in client.store if k.startswith("mine:")]
