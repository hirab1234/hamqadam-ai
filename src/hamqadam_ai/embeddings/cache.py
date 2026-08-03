"""Content-addressed embedding cache.

Why cache at all
----------------
A single ArcFace forward pass costs roughly 150 ms on CPU, and the pipeline
runs one per image plus one for the CNIC portrait. Any repeat - a user
retrying a failed verification, a worker redelivering a queue message, a
Backend replaying a request - pays that again for a bit-identical answer.

What the key is
---------------
The SHA-256 of the **aligned crop**, not the source image. Two slightly
different detections of the same face produce different crops and genuinely
different embeddings, so they must not share an entry; conversely a genuine
repeat produces a byte-identical crop and is a guaranteed hit. The model
version is mixed into the key, so upgrading the recogniser invalidates
everything rather than silently serving vectors from the old space.

A note on caching biometrics
----------------------------
An embedding is biometric data and is reversible enough, through model
inversion, to be treated as personal data. The in-memory backend keeps it in
process behind a short TTL and it dies with the pod; the Redis backend writes
it to a shared store with its own lifecycle, retention and access controls.
That is a data-protection decision rather than a performance one, which is why
``memory`` is the default and Redis has to be turned on deliberately.
"""

from __future__ import annotations

import abc
import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import EmbeddingCacheConfig
from hamqadam_ai.logging.setup import get_logger

log = get_logger(__name__)

BgrImage = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float32]

#: What a cache stores per entry: the unit vector and its pre-normalisation
#: magnitude. Everything else on a FaceEmbedding is provenance the caller
#: already knows.
CachedEmbedding = tuple[FloatArray, float]


def crop_cache_key(
    crop: BgrImage, *, model_version: str, flip_augmented: bool
) -> str:
    """Build the cache key for an aligned crop.

    Args:
        crop: The aligned crop, exactly as it will be fed to the network.
        model_version: Pinned version of the recogniser.
        flip_augmented: Whether the flip-averaged variant will be produced.
            Included because the two settings yield genuinely different
            vectors for the same crop and must not collide.

    Returns:
        A hex digest string.
    """
    digest = hashlib.sha256()
    digest.update(model_version.encode("utf-8"))
    digest.update(b"|flip|" if flip_augmented else b"|noflip|")
    contiguous = np.ascontiguousarray(crop)
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


class EmbeddingCache(abc.ABC):
    """Abstract embedding cache."""

    @abc.abstractmethod
    def get(self, key: str) -> CachedEmbedding | None:
        """Return the cached entry, or ``None`` on a miss."""

    @abc.abstractmethod
    def put(self, key: str, value: CachedEmbedding) -> None:
        """Store an entry."""

    @abc.abstractmethod
    def clear(self) -> None:
        """Drop every entry."""

    @property
    @abc.abstractmethod
    def stats(self) -> dict[str, Any]:
        """Hit/miss counters for the health endpoint."""

    def get_many(self, keys: list[str]) -> list[CachedEmbedding | None]:
        """Look up several keys. Backends with a batch API override this."""
        return [self.get(key) for key in keys]


class NullEmbeddingCache(EmbeddingCache):
    """A cache that stores nothing.

    Used when caching is disabled. Implemented as a real object rather than an
    ``Optional`` so no call site needs a null check.
    """

    __slots__ = ("_lookups",)

    def __init__(self) -> None:
        self._lookups = 0

    def get(self, key: str) -> CachedEmbedding | None:  # noqa: ARG002
        """Always a miss."""
        self._lookups += 1
        return None

    def put(self, key: str, value: CachedEmbedding) -> None:
        """Discard."""

    def clear(self) -> None:
        """No-op."""

    @property
    def stats(self) -> dict[str, Any]:
        """Counters, with a zero hit rate by construction."""
        return {"backend": "none", "lookups": self._lookups, "hits": 0, "entries": 0}


class InMemoryEmbeddingCache(EmbeddingCache):
    """Thread-safe LRU cache with a TTL.

    Bounded on both axes deliberately. The entry cap bounds memory - 2048
    float32 vectors of 512 elements is about 4 MB, which is affordable per
    worker - and the TTL bounds how long biometric data lingers in a process
    that may serve many users.

    Args:
        max_entries: Hard cap on stored entries; the least-recently-used is
            evicted past it.
        ttl_seconds: Age at which an entry is treated as absent.
    """

    __slots__ = (
        "_entries",
        "_evictions",
        "_expiries",
        "_hits",
        "_lock",
        "_lookups",
        "_max_entries",
        "_ttl",
    )

    def __init__(self, *, max_entries: int = 2048, ttl_seconds: float = 600.0) -> None:
        self._max_entries = max(1, max_entries)
        self._ttl = float(ttl_seconds)
        self._entries: OrderedDict[str, tuple[CachedEmbedding, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._lookups = 0
        self._hits = 0
        self._evictions = 0
        self._expiries = 0

    def get(self, key: str) -> CachedEmbedding | None:
        """Return the entry if present and unexpired, refreshing its recency."""
        now = time.monotonic()
        with self._lock:
            self._lookups += 1
            found = self._entries.get(key)
            if found is None:
                return None
            value, stored_at = found
            if (now - stored_at) > self._ttl:
                del self._entries[key]
                self._expiries += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            # A copy, so a caller mutating the returned array cannot corrupt
            # the cached entry for every subsequent request.
            vector, norm = value
            return vector.copy(), norm

    def put(self, key: str, value: CachedEmbedding) -> None:
        """Store an entry, evicting the least recently used past the cap."""
        vector, norm = value
        with self._lock:
            self._entries[key] = ((vector.copy(), norm), time.monotonic())
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        """Drop every entry. Called on shutdown so vectors do not outlive use."""
        with self._lock:
            self._entries.clear()

    @property
    def stats(self) -> dict[str, Any]:
        """Hit rate and occupancy."""
        with self._lock:
            lookups = self._lookups
            hits = self._hits
            return {
                "backend": "memory",
                "lookups": lookups,
                "hits": hits,
                "hit_rate": round(hits / lookups, 4) if lookups else 0.0,
                "entries": len(self._entries),
                "max_entries": self._max_entries,
                "evictions": self._evictions,
                "expiries": self._expiries,
                "ttl_seconds": self._ttl,
            }


class RedisEmbeddingCache(EmbeddingCache):
    """Shared cache backed by Redis.

    Off by default. Turning it on means biometric templates leave this
    process's memory and land in a store with its own retention and access
    controls, which needs a deliberate data-protection decision rather than a
    performance one.

    Every Redis call is wrapped: a cache is an optimisation, and an unreachable
    one must degrade to a miss rather than fail the verification.

    Args:
        client: A connected ``redis.Redis`` instance.
        namespace: Key prefix.
        ttl_seconds: Expiry applied to every write.
        dimension: Vector length, used to validate what comes back.
    """

    __slots__ = ("_client", "_dimension", "_errors", "_hits", "_lookups", "_namespace", "_ttl")

    def __init__(
        self,
        client: Any,
        *,
        namespace: str = "hqai:emb",
        ttl_seconds: int = 600,
        dimension: int = 512,
    ) -> None:
        self._client = client
        self._namespace = namespace
        self._ttl = ttl_seconds
        self._dimension = dimension
        self._lookups = 0
        self._hits = 0
        self._errors = 0

    def _redis_key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    def get(self, key: str) -> CachedEmbedding | None:
        """Fetch and decode, degrading to a miss on any failure."""
        self._lookups += 1
        try:
            payload = self._client.get(self._redis_key(key))
        except Exception as exc:  # noqa: BLE001 - a cache must never fail a request
            self._errors += 1
            log.warning("embedding_cache.redis_get_failed", reason=str(exc))
            return None

        if not payload:
            return None

        try:
            # float32 vector followed by a float64 norm.
            vector = np.frombuffer(
                payload[: self._dimension * 4], dtype=np.float32
            ).copy()
            norm = float(
                np.frombuffer(payload[self._dimension * 4 :], dtype=np.float64)[0]
            )
        except (ValueError, IndexError) as exc:
            self._errors += 1
            log.warning("embedding_cache.redis_decode_failed", reason=str(exc))
            return None

        if vector.shape[0] != self._dimension:
            self._errors += 1
            return None

        self._hits += 1
        return vector, norm

    def put(self, key: str, value: CachedEmbedding) -> None:
        """Encode and store with the configured TTL."""
        vector, norm = value
        payload = (
            np.ascontiguousarray(vector, dtype=np.float32).tobytes()
            + np.array([norm], dtype=np.float64).tobytes()
        )
        try:
            self._client.setex(self._redis_key(key), self._ttl, payload)
        except Exception as exc:  # noqa: BLE001 - a cache must never fail a request
            self._errors += 1
            log.warning("embedding_cache.redis_put_failed", reason=str(exc))

    def clear(self) -> None:
        """Delete every key under this namespace.

        Uses ``scan_iter`` rather than ``KEYS``: the latter blocks the Redis
        event loop for the duration of the scan, which on a shared instance is
        an outage.
        """
        try:
            for found in self._client.scan_iter(match=f"{self._namespace}:*", count=500):
                self._client.delete(found)
        except Exception as exc:  # noqa: BLE001
            log.warning("embedding_cache.redis_clear_failed", reason=str(exc))

    @property
    def stats(self) -> dict[str, Any]:
        """Hit rate and error count."""
        return {
            "backend": "redis",
            "lookups": self._lookups,
            "hits": self._hits,
            "hit_rate": (
                round(self._hits / self._lookups, 4) if self._lookups else 0.0
            ),
            "errors": self._errors,
            "namespace": self._namespace,
            "ttl_seconds": self._ttl,
        }


def build_cache(
    config: EmbeddingCacheConfig, *, dimension: int = 512, redis_client: Any = None
) -> EmbeddingCache:
    """Construct the configured cache backend.

    Falls back to the in-memory cache when Redis is requested but no client
    was supplied or the package is absent - a missing cache backend is not a
    reason to refuse verification traffic.

    Args:
        config: The cache section of the embedding configuration.
        dimension: Vector length, needed by the Redis decoder.
        redis_client: A pre-connected client. When ``None`` and the backend is
            ``redis``, one is created from ``HQ_REDIS__URL``.

    Returns:
        A ready cache.
    """
    if not config.enabled or config.backend == "none":
        return NullEmbeddingCache()

    if config.backend == "memory":
        return InMemoryEmbeddingCache(
            max_entries=config.max_entries, ttl_seconds=config.ttl_seconds
        )

    client = redis_client
    if client is None:
        try:
            import os

            import redis

            client = redis.Redis.from_url(
                os.environ.get("HQ_REDIS__URL", "redis://localhost:6379/0"),
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            client.ping()
        except Exception as exc:  # noqa: BLE001 - degrade rather than refuse traffic
            log.warning(
                "embedding_cache.redis_unavailable",
                reason=str(exc),
                action="falling_back_to_in_memory_cache",
            )
            return InMemoryEmbeddingCache(
                max_entries=config.max_entries, ttl_seconds=config.ttl_seconds
            )

    return RedisEmbeddingCache(
        client,
        namespace=config.namespace,
        ttl_seconds=config.ttl_seconds,
        dimension=dimension,
    )


__all__ = [
    "CachedEmbedding",
    "EmbeddingCache",
    "InMemoryEmbeddingCache",
    "NullEmbeddingCache",
    "RedisEmbeddingCache",
    "build_cache",
    "crop_cache_key",
]
