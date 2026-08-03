"""MODULE 5 latency, measured rather than assumed.

The budgets below are deliberately loose - roughly twice the figures measured
on the development machine (12 logical cores, ONNX Runtime on CPU). They are
regression guards, not targets: their job is to fail when a change makes the
pipeline several times slower, without failing on slower CI hardware.

Measured, single-threaded, 1012 px wide card:

    upright, one recognition pass      ~3.5 s
    half turn, two passes              ~6.2 s
    quarter turn, three passes         ~10.8 s

Recognition dominates completely. Preprocessing is 35-40 ms in a fresh process
and 150-175 ms in a warm one - see ``PREPROCESS_BUDGET_MS`` for why, and why the
warm figure is the one that describes production. Either way recognition is more
than twenty times larger, which is why the orientation search is ordered by
likelihood and exits early rather than always trying all four.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.ocr.preprocessing import (
    deskew,
    enhance_for_ocr,
    rectify_document,
    upscale_if_small,
)
from hamqadam_ai.services.ocr_service import OcrService, build_ocr_service
from tests.fixtures.synthetic_cnic import photograph_on_desk, render_cnic, rotated

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.performance

#: Budget for one recognition pass over an upright card.
UPRIGHT_BUDGET_MS = 8_000.0

#: Budget for the worst case, a quarter-turned card needing three passes.
ROTATED_BUDGET_MS = 26_000.0

#: Budget for the whole image-preparation chain, which must stay negligible
#: beside recognition or the ordering of the pipeline stops making sense.
#:
#: 600 ms rather than the 250 ms this started at, and the reason is worth
#: recording because it is not "the machine was busy".
#:
#: Preprocessing measures 35-40 ms in a freshly started process and roughly
#: 150-175 ms in one that has already done a few thousand OpenCV operations -
#: which is what it is by the time the full suite reaches this test, and what
#: a long-lived service process is at all times. Measured:
#:
#:     round 0-3      35-49 ms
#:     round 4        54 ms
#:     round 5-7      128-145 ms      and stable thereafter
#:
#: The degradation is uniform across every stage (rectify 5.8x, deskew 2.5x,
#: enhance 3.6x), involves no ONNX, is not released by ``gc.collect()``, and
#: does not recover after 25 seconds of complete idle - so it is neither a leak
#: in this code nor thermal throttling. It is a property of sustained OpenCV
#: work in a long-lived process, and the warm figure is the one that describes
#: production.
#:
#: The assumption this test exists to defend survives either way: recognition
#: costs ~3.5 s, so preprocessing is still a twentieth of it warm.
PREPROCESS_BUDGET_MS = 600.0


@pytest.fixture(scope="module")
def service() -> OcrService:
    try:
        built = build_ocr_service()
    except Exception as exc:  # noqa: BLE001 - any engine absence is a skip
        pytest.skip(f"no OCR engine available: {exc}")
    yield built
    built.close()


@pytest.fixture(scope="module")
def clean_card() -> BgrImage:
    return render_cnic()


def median_ms(function, repeats: int = 3) -> float:
    """Median wall-clock of several runs, discarding the first.

    The first call pays for lazy session setup and page faults on the model
    weights, which is real cost but not per-request cost.
    """
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


@pytest.mark.performance
def test_an_upright_card_reads_within_budget(
    service: OcrService, clean_card: BgrImage
) -> None:
    elapsed = median_ms(lambda: service.read(clean_card))
    assert elapsed < UPRIGHT_BUDGET_MS, f"{elapsed:.0f} ms"


@pytest.mark.performance
def test_the_worst_rotation_stays_within_budget(
    service: OcrService, clean_card: BgrImage
) -> None:
    """Three passes rather than one. Bounded because it is the tail latency a
    user actually waits through, and because an unbounded search would grow
    silently if a future change stopped the early exit from firing."""
    turned = rotated(clean_card, 90)
    elapsed = median_ms(lambda: service.read(turned), repeats=2)
    assert elapsed < ROTATED_BUDGET_MS, f"{elapsed:.0f} ms"


@pytest.mark.performance
def test_rotation_costs_passes_not_multiples_of_the_budget(
    service: OcrService, clean_card: BgrImage
) -> None:
    """A quarter turn should cost about three upright reads, not thirty. This
    is what fails if the early exit breaks and the search always runs to the
    end of the candidate list."""
    upright = median_ms(lambda: service.read(clean_card))
    turned_image = rotated(clean_card, 90)
    turned = median_ms(lambda: service.read(turned_image), repeats=2)

    assert turned < upright * 6.0, f"upright {upright:.0f} ms, turned {turned:.0f} ms"


@pytest.mark.performance
def test_preprocessing_is_negligible_beside_recognition(
    clean_card: BgrImage,
) -> None:
    """Pins the assumption the whole design rests on. If preparation ever
    became comparable to recognition, trying several orientations would stop
    being the obviously right trade.

    Budgeted for a **warm** process - see ``PREPROCESS_BUDGET_MS``. A cold
    figure would pass here and fail in the full suite, which is exactly what it
    did before the budget was re-derived.
    """
    photo = photograph_on_desk(clean_card)

    def prepare() -> None:
        result = rectify_document(photo)
        levelled, _angle = deskew(result.image)
        enlarged, _scale = upscale_if_small(levelled)
        enhance_for_ocr(enlarged)

    elapsed = median_ms(prepare, repeats=5)
    assert elapsed < PREPROCESS_BUDGET_MS, f"{elapsed:.0f} ms"


@pytest.mark.performance
def test_a_realistic_photograph_reads_within_budget(
    service: OcrService, clean_card: BgrImage
) -> None:
    """The case that actually arrives: a card on a desk, needing rectification
    before anything else can work."""
    photo = photograph_on_desk(clean_card)
    elapsed = median_ms(lambda: service.read(photo))
    assert elapsed < UPRIGHT_BUDGET_MS, f"{elapsed:.0f} ms"


@pytest.mark.performance
def test_the_reported_duration_matches_the_wall_clock(
    service: OcrService, clean_card: BgrImage
) -> None:
    """The response carries its own timing, and a caller budgeting on it
    deserves a number that means something."""
    started = time.perf_counter()
    result = service.read(clean_card)
    measured = (time.perf_counter() - started) * 1000.0

    assert result.duration_ms == pytest.approx(measured, rel=0.35)


@pytest.mark.performance
def test_an_unreadable_image_fails_fast(service: OcrService) -> None:
    """A blank frame must not pay for a full search over four orientations:
    there is no text to find in any of them, and the user is waiting."""
    blank = np.full((640, 1012, 3), 245, dtype=np.uint8)

    elapsed = median_ms(lambda: service.read(blank), repeats=2)

    assert elapsed < UPRIGHT_BUDGET_MS, f"{elapsed:.0f} ms"
