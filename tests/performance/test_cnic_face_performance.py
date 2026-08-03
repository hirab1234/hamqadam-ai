"""MODULE 6 latency, measured rather than assumed.

Budgets are roughly twice the figures measured on the development machine
(12 logical cores, ONNX Runtime on CPU). Regression guards, not targets: they
should fail when a change makes the pipeline several times slower, and should
not fail on slower CI hardware.

Measured, one card:

    rectify + upscale                    ~40 ms
    SCRFD over a 1200 px card           ~600 ms
    quality on the CNIC scale           ~250 ms
    ArcFace on the portrait             ~250 ms
    -----------------------------------------
    end to end                          ~1.1-1.7 s

Detection dominates, which is why the card is upscaled only to
``detection_min_width`` and not further: every extra pixel is paid for in the
stage that already costs the most.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.embeddings.base import FaceEmbedding
from hamqadam_ai.services import (
    build_cnic_face_service,
    build_embedding_service,
    build_face_detection_service,
)
from hamqadam_ai.services.cnic_face_service import CnicFaceService
from tests.fixtures.cnic_portrait import (
    blank_card,
    reference_portrait,
    render_cnic_with_portrait,
)

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.performance

#: Budget for locating the portrait and comparing it against a selfie.
MATCH_BUDGET_MS = 4_000.0

#: Budget for a card with no portrait on it. Lower, because the quality and
#: embedding stages never run - and a user waiting to be told "wrong side of
#: the card" should not wait as long as one being verified.
NOT_FOUND_BUDGET_MS = 2_500.0


@pytest.fixture(scope="module")
def service() -> CnicFaceService:
    try:
        return build_cnic_face_service()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"CNIC face service unavailable: {exc}")


@pytest.fixture(scope="module")
def face() -> BgrImage:
    image = reference_portrait()
    if image is None:
        pytest.skip("no public-domain reference portrait installed")
    return image


@pytest.fixture(scope="module")
def card(face: BgrImage) -> BgrImage:
    rendered = render_cnic_with_portrait(face=face)
    if rendered is None:
        pytest.skip("card fixture unavailable")
    return rendered


@pytest.fixture(scope="module")
def selfie(face: BgrImage) -> FaceEmbedding:
    detector = build_face_detection_service()
    embedder = build_embedding_service()
    return embedder.embed_to_vector(
        face,
        role=ImageRole.LIVE_SELFIE,
        detection=detector.detect(face, role=ImageRole.LIVE_SELFIE),
    )


def median_ms(function, repeats: int = 3) -> float:
    """Median wall-clock of several runs, discarding the first.

    The first call pays for lazy session setup and page faults on the model
    weights - real cost, but not per-request cost.
    """
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


@pytest.mark.performance
def test_a_card_matches_within_budget(
    service: CnicFaceService, card: BgrImage, selfie: FaceEmbedding
) -> None:
    elapsed = median_ms(lambda: service.match(card, selfie))
    assert elapsed < MATCH_BUDGET_MS, f"{elapsed:.0f} ms"


@pytest.mark.performance
def test_a_card_with_no_portrait_fails_faster_than_one_with(
    service: CnicFaceService, card: BgrImage, selfie: FaceEmbedding
) -> None:
    """Quality and embedding are skipped when there is nothing to embed. If
    that ever stopped being true, this is what would notice."""
    found = median_ms(lambda: service.match(card, selfie))
    missing = median_ms(lambda: service.match(blank_card(), selfie), repeats=2)

    assert missing < NOT_FOUND_BUDGET_MS, f"{missing:.0f} ms"
    assert missing < found


@pytest.mark.performance
def test_extraction_dominates_the_comparison(
    service: CnicFaceService, card: BgrImage, selfie: FaceEmbedding
) -> None:
    """Pins the assumption the design rests on: the cost is finding and
    encoding the portrait, not comparing two vectors. If comparison ever
    became significant, caching the portrait embedding across comparisons
    would stop being an obvious win.
    """
    full = median_ms(lambda: service.match(card, selfie))
    extraction = median_ms(lambda: service.extract_portrait(card))

    assert extraction > full * 0.85


@pytest.mark.performance
def test_the_reported_duration_matches_the_wall_clock(
    service: CnicFaceService, card: BgrImage, selfie: FaceEmbedding
) -> None:
    """A caller budgeting on the reported figure deserves one that means
    something."""
    service.match(card, selfie)  # warm

    started = time.perf_counter()
    result = service.match(card, selfie)
    measured = (time.perf_counter() - started) * 1000.0

    assert result.duration_ms == pytest.approx(measured, rel=0.35)


@pytest.mark.performance
def test_noise_does_not_take_longer_than_a_card(
    service: CnicFaceService, selfie: FaceEmbedding
) -> None:
    """A frame full of spurious detections must not turn into a long walk
    through candidate scoring."""
    noise = np.random.default_rng(5).integers(
        0, 256, (760, 1200, 3), dtype=np.uint8
    )
    elapsed = median_ms(lambda: service.match(noise, selfie), repeats=2)

    assert elapsed < MATCH_BUDGET_MS, f"{elapsed:.0f} ms"
