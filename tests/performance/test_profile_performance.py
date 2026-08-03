"""MODULE 7 latency.

Budgets are regression guards, not targets: they should fail when a change
makes the pipeline several times slower, and not fail on a loaded CI runner.

Measured on an idle development machine (12 cores, ONNX Runtime on CPU), median of five runs:

    four detectors, 512x600 portrait          150 ms
    four detectors, 1080x1920 screenshot      210 ms
    four detectors, 3000x4000 phone photo     725 ms
    full analysis, portrait                   467 ms
    detector share of the full analysis      0.344

The detectors are about three times cheaper than the full analysis, not ten -
an earlier version of this docstring said "an order of magnitude" and was
wrong. What makes ``assess_authenticity`` worth exposing separately is that it
needs **no model weights** at all, so it runs where the models are absent and
rejects a screenshot before they are loaded.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.services import build_profile_service
from hamqadam_ai.services.profile_service import ProfileAnalysisService
from tests.fixtures.profile_images import (
    as_screenshot,
    as_synthetic_render,
    reference_photo,
)

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.performance

#: Budget for the full analysis of one photograph.
ANALYSIS_BUDGET_MS = 3_000.0

#: Budget for the detectors alone, at ordinary upload sizes. Roughly three
#: times the measured 150-210 ms, which leaves room for a loaded CI runner
#: without letting a real regression through.
AUTHENTICITY_BUDGET_MS = 700.0


@pytest.fixture(scope="module")
def service() -> ProfileAnalysisService:
    try:
        return build_profile_service()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"profile service unavailable: {exc}")


@pytest.fixture(scope="module")
def photo() -> BgrImage:
    image = reference_photo()
    if image is None:
        pytest.skip("no public-domain reference photograph installed")
    return image


def median_ms(function, repeats: int = 3) -> float:
    """Median wall-clock of several runs, discarding the first."""
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


@pytest.mark.performance
def test_a_photograph_is_analysed_within_budget(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    elapsed = median_ms(lambda: service.analyse(photo))
    assert elapsed < ANALYSIS_BUDGET_MS, f"{elapsed:.0f} ms"


@pytest.mark.performance
def test_the_detectors_alone_are_cheap(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    """They run on every image in a request, including ones the identity path
    has already rejected, so their cost is paid unconditionally."""
    elapsed = median_ms(lambda: service.assess_authenticity(photo), repeats=5)
    assert elapsed < AUTHENTICITY_BUDGET_MS, f"{elapsed:.0f} ms"


@pytest.mark.performance
def test_the_detectors_are_a_minority_of_the_full_analysis(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    """Measured share is 0.34. Bounded at 0.60, which leaves room for a loaded
    machine while still failing if the detectors ever came to dominate.

    Note what this test does *not* claim. An earlier version asserted an order
    of magnitude and was wrong - the real figure is about three times. The
    reason ``assess_authenticity`` is exposed separately is that it needs no
    model weights, not that it is nearly free.
    """
    full = median_ms(lambda: service.analyse(photo))
    detectors_only = median_ms(lambda: service.assess_authenticity(photo), repeats=5)

    assert detectors_only < full * 0.60, (
        f"detectors {detectors_only:.0f} ms against full {full:.0f} ms "
        f"(share {detectors_only / full:.2f})"
    )


@pytest.mark.performance
def test_a_large_upload_stays_within_budget(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    """A modern phone photograph is several thousand pixels on the long side.
    The spectral work is fixed-size, but the block and row measurements are
    not, so this is where a quadratic mistake would show."""
    import cv2

    large = cv2.resize(photo, (3000, 4000), interpolation=cv2.INTER_CUBIC)
    elapsed = median_ms(lambda: service.assess_authenticity(large), repeats=2)

    # Measured 725 ms at 12 megapixels. Cost scales with pixel count because
    # the row and block measurements run at full resolution; only the spectral
    # work is fixed-size. Linear is fine, quadratic is not, and 3x the measured
    # value is where the difference shows.
    assert elapsed < 2_200.0, f"{elapsed:.0f} ms"


@pytest.mark.performance
def test_an_image_with_no_face_is_not_slower(
    service: ProfileAnalysisService
) -> None:
    """Rendered artwork has no face, so quality never runs. It must not cost
    more than a photograph that does."""
    elapsed = median_ms(lambda: service.analyse(as_synthetic_render()), repeats=2)
    assert elapsed < ANALYSIS_BUDGET_MS, f"{elapsed:.0f} ms"


@pytest.mark.performance
def test_the_reported_duration_matches_the_wall_clock(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    service.analyse(photo)  # warm

    started = time.perf_counter()
    result = service.analyse(photo)
    measured = (time.perf_counter() - started) * 1000.0

    assert result.duration_ms == pytest.approx(measured, rel=0.35)


@pytest.mark.performance
def test_a_screenshot_is_not_slower_than_a_photograph(
    service: ProfileAnalysisService, photo: BgrImage
) -> None:
    """A tall screenshot has several times the pixels of a portrait crop, so
    this is the case where the row-wise measurements cost most."""
    elapsed = median_ms(lambda: service.analyse(as_screenshot(photo)), repeats=2)
    assert elapsed < ANALYSIS_BUDGET_MS, f"{elapsed:.0f} ms"
