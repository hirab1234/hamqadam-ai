"""MODULE 10 latency: the number the Backend's timeout is actually set against.

Every other module measures its own stage. This measures the whole thing, which
is the only figure a caller experiences, and the only one that tells you whether
the concurrency is doing anything.

Budgets are roughly twice the figures measured on the development machine (12
logical cores, ONNX Runtime on CPU). Regression guards, not targets: they should
fail when a change makes a verification several times slower, and should not fail
on slower CI hardware.

Measured, one coherent submission (selfie + profile + CNIC):

    stage sum                           ~15.5 s
    wall clock                           ~7.9 s
    concurrency saving                   ~49%

    cnic_ocr                            ~7946 ms   <- sets the floor
    profile                             ~2689 ms
    cnic_portrait                       ~2347 ms
    selfie                              ~2256 ms
    cnic_authenticity                    ~234 ms
    duplicate                              ~2 ms
    matching                               ~2 ms

The CNIC read dominates so completely that the wall clock is essentially "the
OCR, plus whatever did not fit alongside it". Duplicate search and matching are
arithmetic over vectors that already exist and are free by comparison.

The concurrency assertion is the one worth having. A refactor that accidentally
serialises phase A would not fail any correctness test - every number would
still be right - and would roughly double the latency of every verification.
"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.pipelines import VerificationImages, build_pipeline
from hamqadam_ai.schemas.verification import VerificationRequest
from tests.fixtures.cnic_portrait import (
    reference_portrait,
    render_cnic_with_portrait,
)

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.performance

#: Budget for a full submission: selfie, profile and CNIC.
FULL_BUDGET_MS = 20_000.0

#: Budget for a selfie alone. Far lower, because every CNIC stage is skipped -
#: and it is worth pinning separately, since a user submitting one image should
#: not wait as long as one submitting three.
SELFIE_ONLY_BUDGET_MS = 6_000.0

#: The stages that must overlap. If their durations sum to no more than the
#: wall clock, phase A has been serialised.
MIN_CONCURRENCY_SAVING = 1.25


@pytest.fixture(scope="module")
def pipeline() -> object:
    try:
        return build_pipeline(get_settings())
    except Exception as exc:  # noqa: BLE001 - a missing model is a skip
        pytest.skip(f"pipeline unavailable: {exc}")


@pytest.fixture(scope="module")
def portrait() -> BgrImage:
    image = reference_portrait()
    if image is None:
        pytest.skip("no public-domain reference portrait installed")
    return image


@pytest.fixture(scope="module")
def submission(portrait: BgrImage) -> VerificationImages:
    card = render_cnic_with_portrait(face=portrait)
    if card is None:
        pytest.skip("no reference portrait available for the card")
    return VerificationImages(live_selfie=portrait, profile=portrait, cnic=card)


def _time(pipeline: object, images: VerificationImages, name: str) -> tuple[object, float]:
    """Run one verification, returning the result and the wall clock in ms."""
    started = time.perf_counter()
    result = pipeline.verify_sync(  # type: ignore[attr-defined]
        VerificationRequest(verification_id=name), images
    )
    return result, (time.perf_counter() - started) * 1000.0


def test_a_full_submission_completes_within_budget(
    pipeline: object, submission: VerificationImages
) -> None:
    """The figure the Backend's own timeout has to accommodate."""
    result, elapsed_ms = _time(pipeline, submission, "perf-full")

    assert elapsed_ms < FULL_BUDGET_MS, (
        f"a full verification took {elapsed_ms:.0f} ms against a "
        f"{FULL_BUDGET_MS:.0f} ms budget"
    )
    # A verification that failed fast would pass a latency budget while being
    # useless, so the outcome is checked too.
    assert result.complete is True  # type: ignore[attr-defined]


def test_a_selfie_alone_is_much_faster(
    pipeline: object, portrait: BgrImage
) -> None:
    """Skipped stages must actually cost nothing.

    Every CNIC stage is skipped here. If this is not dramatically faster than
    the full submission, stages are running on absent input.
    """
    _, elapsed_ms = _time(
        pipeline, VerificationImages(live_selfie=portrait), "perf-selfie"
    )
    assert elapsed_ms < SELFIE_ONLY_BUDGET_MS, (
        f"a selfie-only verification took {elapsed_ms:.0f} ms against a "
        f"{SELFIE_ONLY_BUDGET_MS:.0f} ms budget"
    )


def test_phase_a_actually_runs_concurrently(
    pipeline: object, submission: VerificationImages
) -> None:
    """The assertion no correctness test would catch.

    A refactor that serialises phase A leaves every number in the response
    correct and roughly doubles the latency of every verification. The only
    evidence is that the stage durations stop exceeding the wall clock.
    """
    result, elapsed_ms = _time(pipeline, submission, "perf-concurrency")

    ran = [stage for stage in result.stages if stage.ran]  # type: ignore[attr-defined]
    stage_sum = sum(stage.duration_ms for stage in ran)
    saving = stage_sum / elapsed_ms

    assert saving > MIN_CONCURRENCY_SAVING, (
        f"stages summed to {stage_sum:.0f} ms against {elapsed_ms:.0f} ms of "
        f"wall clock (ratio {saving:.2f}); phase A appears to be serialised"
    )


def test_the_reported_time_matches_the_measured_time(
    pipeline: object, submission: VerificationImages
) -> None:
    """`processing_time.total` must be the truth.

    The Backend uses this for its own metrics and SLA accounting, so a figure
    that drifts from reality is worse than no figure: it would be trusted.
    """
    result, elapsed_ms = _time(pipeline, submission, "perf-selfreport")
    reported = result.processing_time.total  # type: ignore[attr-defined]

    # Within 15%: the pipeline's own clock starts a little after ours and stops
    # a little before, so exact equality would be a flaky assertion about
    # function-call overhead rather than about timing.
    assert abs(reported - elapsed_ms) / elapsed_ms < 0.15, (
        f"reported {reported:.0f} ms against {elapsed_ms:.0f} ms measured"
    )


def test_repeated_verifications_do_not_degrade(
    pipeline: object, submission: VerificationImages
) -> None:
    """Three runs, and the third must not be dramatically slower than the first.

    This guards a real, previously-observed failure mode: OpenCV degrades by
    roughly 4x after some tens of iterations in the same process, which was once
    misattributed to ONNX Runtime and produced a wrong configuration change. The
    ratio here is deliberately loose - it is looking for a leak or an unbounded
    cache, not for jitter.
    """
    timings = [
        _time(pipeline, submission, f"perf-repeat-{index}")[1] for index in range(3)
    ]
    assert timings[-1] < timings[0] * 2.5, (
        f"latency grew across runs: {[f'{ms:.0f}' for ms in timings]} ms"
    )
