"""Latency and memory budget for MODULE 3.

Embedding is the most expensive stage in the pipeline: roughly 148 ms per face
on this CPU against ~10 ms for the whole of Module 2. These are regression
alarms for an algorithmic mistake - a lost cache, an accidental double forward
pass, flip augmentation switched on by accident - not a benchmark of the host,
so the ceilings are deliberately generous.
"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.embeddings.alignment import align_face_for_recognition
from hamqadam_ai.embeddings.base import EmbeddingRequest
from hamqadam_ai.models.registry import get_registry
from hamqadam_ai.services import build_embedding_service
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.performance

#: Ceiling for one face, cold. Measured ~148 ms.
MAX_SINGLE_MS = 900.0

#: Ceiling for a full seven-image request.
MAX_REQUEST_MS = 5000.0

#: Ceiling for a fully-cached repeat of the same request. Measured ~2 ms.
MAX_CACHED_MS = 120.0


@pytest.fixture(scope="module")
def service():  # noqa: ANN201 - pytest fixture
    """The embedding service, skipping when the recogniser is absent."""
    settings = get_settings()
    try:
        return build_embedding_service(settings, get_registry(settings))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ArcFace weights unavailable: {exc}")


def face_image(seed: int = 0, size: int = 400) -> BgrImage:
    """A detailed synthetic image; no detection is involved in these tests."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (size, size), dtype=np.uint8)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    smooth = 128 + 55 * np.sin(x / 25.0) + 45 * np.cos(y / 18.0)
    blended = np.clip(0.5 * smooth + 0.5 * base, 0, 255).astype(np.uint8)
    import cv2

    return cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)


def landmarks_for(size: int = 400) -> Landmarks5:
    """Template landmarks scaled into a synthetic frame."""
    from hamqadam_ai.core.constants import ARCFACE_REFERENCE_LANDMARKS_112

    scale = size / 112.0 * 0.55
    offset = size * 0.22
    points = ARCFACE_REFERENCE_LANDMARKS_112 * scale + offset
    return Landmarks5(points.astype(np.float32))


def request_for(seed: int) -> EmbeddingRequest:
    """One embedding request over a synthetic face."""
    return EmbeddingRequest(
        image=face_image(seed),
        box=BoundingBox(60.0, 60.0, 340.0, 340.0),
        landmarks=landmarks_for(),
        role=ImageRole.SECONDARY_IMAGE,
    )


@pytest.mark.performance
def test_a_single_face_is_within_budget(service) -> None:  # noqa: ANN001
    service._cache.clear()  # noqa: SLF001
    request = request_for(1)

    service.embed_many([request])  # warm the kernels
    service._cache.clear()  # noqa: SLF001

    started = time.perf_counter()
    batch = service.embed_many([request])
    elapsed = (time.perf_counter() - started) * 1000.0

    assert batch.succeeded == 1
    assert elapsed < MAX_SINGLE_MS, f"single face took {elapsed:.0f} ms"


@pytest.mark.performance
def test_a_full_seven_image_request_is_within_budget(service) -> None:  # noqa: ANN001
    """One selfie, one profile, four secondaries and a CNIC portrait."""
    requests = [request_for(seed) for seed in range(7)]

    service.embed_many(requests[:1])  # warm
    service._cache.clear()  # noqa: SLF001

    started = time.perf_counter()
    batch = service.embed_many(requests)
    elapsed = (time.perf_counter() - started) * 1000.0

    assert batch.succeeded == 7
    assert batch.forward_passes == 7
    assert elapsed < MAX_REQUEST_MS, f"seven faces took {elapsed:.0f} ms"


@pytest.mark.performance
def test_a_cached_repeat_is_nearly_free(service) -> None:  # noqa: ANN001
    """Measured 591 ms cold against 2 ms warm. If this regresses, the cache key
    has probably stopped being stable."""
    requests = [request_for(seed) for seed in range(3)]

    service._cache.clear()  # noqa: SLF001
    service.embed_many(requests)

    started = time.perf_counter()
    warm = service.embed_many(requests)
    elapsed = (time.perf_counter() - started) * 1000.0

    assert warm.cache_hits == 3
    assert warm.forward_passes == 0
    assert elapsed < MAX_CACHED_MS, f"cached repeat took {elapsed:.0f} ms"


@pytest.mark.performance
def test_flip_augmentation_is_off_by_default() -> None:
    """It costs exactly 2x inference for a measured +0.0065 mean cosine. On the
    CPU path that is a bad trade, and switching it on by accident would double
    the pipeline's most expensive stage."""
    assert get_settings().embedding.flip_augmentation is False


@pytest.mark.performance
def test_flip_augmentation_really_does_double_the_work(service) -> None:  # noqa: ANN001
    """Confirms the cost is what the configuration comment claims."""
    from hamqadam_ai.embeddings.arcface import ArcFaceEmbedder

    settings = get_settings()
    registry = get_registry(settings)
    spec = settings.model_spec(settings.embedding.model)
    model = registry.get(settings.embedding.model)

    crop = align_face_for_recognition(
        face_image(3), box=BoundingBox(60.0, 60.0, 340.0, 340.0),
        landmarks=landmarks_for(),
    ).crop

    plain = ArcFaceEmbedder(model, spec, max_batch=8, flip_augmentation=False)
    flipped = ArcFaceEmbedder(model, spec, max_batch=8, flip_augmentation=True)

    for embedder in (plain, flipped):
        embedder.embed_aligned([crop])

    started = time.perf_counter()
    plain.embed_aligned([crop] * 4)
    without = time.perf_counter() - started

    started = time.perf_counter()
    flipped.embed_aligned([crop] * 4)
    with_flip = time.perf_counter() - started

    # Generous bounds: the point is that it is roughly double, not exactly.
    assert with_flip > without * 1.4, (
        f"flip augmentation cost only {with_flip / without:.2f}x, which suggests "
        f"the mirrored pass is not actually running"
    )


@pytest.mark.performance
def test_repeated_embedding_does_not_leak(service) -> None:  # noqa: ANN001
    """A long-lived worker must not accumulate. The cache is bounded, so growth
    here would mean something else is holding references."""
    import gc
    import tracemalloc

    service._cache.clear()  # noqa: SLF001
    service.embed_many([request_for(0)])
    gc.collect()

    tracemalloc.start()
    baseline = tracemalloc.take_snapshot()
    for seed in range(10):
        service.embed_many([request_for(100 + seed)])
    gc.collect()
    current = tracemalloc.take_snapshot()
    tracemalloc.stop()

    growth = sum(stat.size_diff for stat in current.compare_to(baseline, "filename"))
    assert growth < 64 * 1024 * 1024, f"ten embeddings grew the heap by {growth / 1e6:.1f} MB"


@pytest.mark.performance
def test_the_cache_stays_bounded_under_churn(service) -> None:  # noqa: ANN001
    """Distinct crops must evict rather than accumulate."""
    configured = get_settings().embedding.cache.max_entries
    service._cache.clear()  # noqa: SLF001

    for seed in range(24):
        service.embed_many([request_for(500 + seed)])

    entries = service._cache.stats.get("entries")  # noqa: SLF001
    if entries is not None:
        assert entries <= configured
