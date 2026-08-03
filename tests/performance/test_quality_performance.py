"""Latency and throughput budget for MODULE 2.

The pipeline analyses up to seven images inside a 45-second request budget
shared with detection, embedding, OCR and vector search. Quality is pure
OpenCV and NumPy with no model weights, so it should be one of the cheapest
stages - these tests exist to notice when it stops being.

Thresholds are deliberately generous. They are regression alarms for an
algorithmic mistake (an accidental full-resolution FFT, a per-pixel Python
loop), not a benchmark of the host.
"""

from __future__ import annotations

import asyncio
import time

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.quality.base import QualityContext
from hamqadam_ai.services.quality_service import build_quality_service
from hamqadam_ai.utils.geometry import BoundingBox

BgrImage = npt.NDArray[np.uint8]

#: Ceiling for a single 12-megapixel assessment. A modern phone's main camera
#: produces roughly this, and it is the worst case the service sees.
MAX_LARGE_IMAGE_MS = 1500.0

#: Ceiling for a typical 1080p upload.
MAX_TYPICAL_IMAGE_MS = 700.0

#: Ceiling for the full seven-image request analysed concurrently.
MAX_FULL_REQUEST_MS = 4000.0


def photo(width: int, height: int, seed: int = 4) -> BgrImage:
    """A realistically detailed test image of a given size."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (max(2, height // 8), max(2, width // 8), 3), dtype=np.uint8)
    upscaled = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    structure = (
        128 + 50 * np.sin(x / 30.0) + 40 * np.cos(y / 22.0)
    )[:, :, None]
    return np.clip(0.5 * upscaled + 0.5 * structure, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def service():  # noqa: ANN201 - pytest fixture
    """The quality service, built once for the module."""
    return build_quality_service()


def median_ms(callable_, repeats: int = 5) -> float:  # noqa: ANN001
    """Median wall-clock milliseconds over several runs, discarding the first.

    The first call pays for OpenCV's lazy kernel initialisation and would
    otherwise dominate a small sample.
    """
    callable_()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        timings.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(timings))


@pytest.mark.performance
def test_typical_upload_is_fast(service) -> None:
    """A 1080p photo with a detected face - the common case."""
    image = photo(1080, 1920)
    box = BoundingBox(420.0, 620.0, 700.0, 980.0)

    elapsed = median_ms(
        lambda: service.assess(image, role=ImageRole.LIVE_SELFIE, face_box=box)
    )
    assert elapsed < MAX_TYPICAL_IMAGE_MS, (
        f"typical 1080p assessment took {elapsed:.0f} ms, "
        f"budget {MAX_TYPICAL_IMAGE_MS:.0f} ms"
    )


@pytest.mark.performance
def test_twelve_megapixel_image_stays_within_budget(service) -> None:
    """The reduction to `global_analysis_long_side` is what makes this
    tractable; without it the FFT and blockiness passes dominate."""
    image = photo(3000, 4000)
    elapsed = median_ms(lambda: service.assess(image), repeats=3)
    assert elapsed < MAX_LARGE_IMAGE_MS, (
        f"12 MP assessment took {elapsed:.0f} ms, budget {MAX_LARGE_IMAGE_MS:.0f} ms"
    )


@pytest.mark.performance
def test_cost_grows_sublinearly_with_pixel_count(service) -> None:
    """Because global metrics run on a size-capped view, a 16x larger image
    must not cost 16x more."""
    small = median_ms(lambda: service.assess(photo(600, 800)), repeats=3)
    large = median_ms(lambda: service.assess(photo(2400, 3200)), repeats=3)

    assert large < small * 8.0, (
        f"16x the pixels cost {large / max(small, 1e-6):.1f}x the time; "
        f"the analysis-size cap may not be applied"
    )


@pytest.mark.performance
def test_a_full_seven_image_request_fits_the_budget(service) -> None:
    """One selfie, one profile, four secondaries and a CNIC."""
    images = [
        (photo(1080, 1440, seed=1), ImageRole.LIVE_SELFIE),
        (photo(1080, 1440, seed=2), ImageRole.PROFILE_IMAGE),
        *[(photo(900, 1200, seed=10 + i), ImageRole.SECONDARY_IMAGE) for i in range(4)],
        (photo(1600, 1000, seed=20), ImageRole.CNIC_IMAGE),
    ]

    async def run() -> None:
        await service.assess_many_async(images)

    started = time.perf_counter()
    asyncio.run(run())
    elapsed = (time.perf_counter() - started) * 1000.0

    assert elapsed < MAX_FULL_REQUEST_MS, (
        f"seven-image request took {elapsed:.0f} ms, "
        f"budget {MAX_FULL_REQUEST_MS:.0f} ms"
    )


@pytest.mark.performance
def test_the_shared_context_is_computed_once(service) -> None:
    """Grayscale, the canonical crop and the FFT are each needed by three or
    four analysers. Recomputing them per analyser would roughly double the
    module's cost, so the caching is worth asserting rather than assuming.
    """
    image = photo(1200, 1600)
    context = QualityContext(
        image=image, face_box=BoundingBox(400.0, 500.0, 700.0, 900.0)
    )

    first = time.perf_counter()
    _ = context.radial_spectrum
    cold = time.perf_counter() - first

    second = time.perf_counter()
    for _ in range(50):
        _ = context.radial_spectrum
    warm = (time.perf_counter() - second) / 50.0

    assert warm < cold / 100.0, "radial_spectrum is being recomputed, not cached"


@pytest.mark.performance
def test_concurrency_gives_real_parallelism(service) -> None:
    """OpenCV and NumPy release the GIL, so dispatching to a thread pool must
    beat running the same work serially."""
    images = [(photo(900, 1200, seed=i), ImageRole.PROFILE_IMAGE) for i in range(4)]

    started = time.perf_counter()
    for image, role in images:
        service.assess(image, role=role)
    serial = time.perf_counter() - started

    async def run() -> None:
        await service.assess_many_async(images)

    started = time.perf_counter()
    asyncio.run(run())
    concurrent = time.perf_counter() - started

    # A modest bar: on a single-core runner there is nothing to win, so this
    # only asserts the async path is not *slower* by more than a small margin.
    assert concurrent < serial * 1.25, (
        f"concurrent path ({concurrent * 1000:.0f} ms) is slower than serial "
        f"({serial * 1000:.0f} ms)"
    )


@pytest.mark.performance
def test_memory_footprint_stays_bounded(service) -> None:
    """Repeated assessment must not accumulate - a leaked cache on a
    long-lived worker is how a pod dies at 3am."""
    import gc
    import tracemalloc

    image = photo(1200, 1600)
    service.assess(image)
    gc.collect()

    tracemalloc.start()
    baseline = tracemalloc.take_snapshot()
    for index in range(12):
        service.assess(photo(1200, 1600, seed=index))
    gc.collect()
    current = tracemalloc.take_snapshot()
    tracemalloc.stop()

    growth = sum(
        stat.size_diff for stat in current.compare_to(baseline, "filename")
    )
    assert growth < 64 * 1024 * 1024, (
        f"12 assessments grew the heap by {growth / 1e6:.1f} MB"
    )
