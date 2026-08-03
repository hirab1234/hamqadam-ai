"""MODULE 10: the pipeline's failure behaviour, with stages deliberately broken.

Every image is synthetic or a print-degraded public-domain reference portrait.

The central claim of the pipeline is *degrade, never abort*: no single stage
failing may end the request, because a verification whose CNIC was unreadable
still has a face comparison worth reporting, and a Backend that receives an
exception learns nothing about the six images that were fine.

That claim cannot be tested by feeding in bad images - a bad image is a
finding, not a failure. It is tested by breaking a stage outright and checking
that a full result still comes back, correctly marked. Each test here replaces
one real service method with one that raises, which is the failure mode the
``_run`` wrapper exists for.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import Recommendation
from hamqadam_ai.pipelines import VerificationImages, build_pipeline
from hamqadam_ai.schemas.verification import VerificationRequest
from tests.fixtures.cnic_portrait import (
    reference_portrait,
    render_cnic_with_portrait,
)

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pipeline() -> Any:
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
    """A coherent submission: one person, selfie, profile and card."""
    card = render_cnic_with_portrait(face=portrait)
    if card is None:
        pytest.skip("no reference portrait available for the card")
    return VerificationImages(
        live_selfie=portrait, profile=portrait, cnic=card, secondaries=[]
    )


@pytest.fixture
def broken(pipeline: Any) -> Iterator[Any]:
    """Break one service method, then put it back.

    Yields a callable taking a service name and a method name.
    """
    restore: list[tuple[type, str, Any]] = []

    def break_it(service: str, method: str) -> None:
        # Patched on the class, not the instance: every service defines
        # `__slots__`, so assigning an attribute to one raises AttributeError.
        # The pipeline is module-scoped and each patch is undone below, so the
        # blast radius is this test.
        target = type(pipeline._services[service])  # noqa: SLF001
        original = getattr(target, method)
        restore.append((target, method, original))

        def explode(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError(f"{service}.{method} is deliberately broken")

        setattr(target, method, explode)

    yield break_it

    for target, method, original in reversed(restore):
        setattr(target, method, original)


def _run(pipeline: Any, images: VerificationImages, name: str) -> Any:
    return pipeline.verify_sync(
        VerificationRequest(verification_id=name), images
    )


class TestDegradation:
    """One broken stage must not take the request with it."""

    @pytest.mark.parametrize(
        ("service", "method", "stage"),
        [
            ("ocr", "read", "cnic_ocr"),
            ("cnic_face", "extract_portrait", "cnic_portrait"),
            ("duplicate", "check", "duplicate"),
            ("profile", "analyse", "profile"),
        ],
    )
    def test_a_broken_stage_still_returns_a_result(
        self,
        pipeline: Any,
        submission: VerificationImages,
        broken: Any,
        service: str,
        method: str,
        stage: str,
    ) -> None:
        broken(service, method)
        result = _run(pipeline, submission, f"v-broken-{stage}")

        assert result.recommendation in set(Recommendation)
        # The failure must be *visible*. A stage that silently did not run is
        # indistinguishable from one that ran and found nothing.
        status = {entry.stage: entry for entry in result.stages}[stage]
        assert status.succeeded is False
        assert status.error

    def test_a_broken_stage_lowers_assessment_confidence(
        self, pipeline: Any, submission: VerificationImages, broken: Any
    ) -> None:
        """Less evidence must be reported as less evidence.

        Otherwise a partially-failed verification looks exactly like a complete
        one, and the decision engine's evidence floor has nothing to act on.
        """
        healthy = _run(pipeline, submission, "v-healthy")
        broken("ocr", "read")
        degraded = _run(pipeline, submission, "v-degraded")

        assert degraded.assessment_confidence < healthy.assessment_confidence
        assert degraded.complete is False

    def test_a_broken_stage_never_raises_to_the_caller(
        self, pipeline: Any, submission: VerificationImages, broken: Any
    ) -> None:
        """Every stage broken at once still produces a response.

        The worst case, and the one where an exception would be most tempting.
        The Backend still gets a structured answer saying nothing could be
        checked, which it can route to manual review.
        """
        for service, method in (
            ("ocr", "read"),
            ("cnic_face", "extract_portrait"),
            ("duplicate", "check"),
            ("profile", "analyse"),
            ("matching", "match"),
        ):
            broken(service, method)

        result = _run(pipeline, submission, "v-all-broken")
        assert result.recommendation is not Recommendation.APPROVE
        assert result.complete is False
        assert result.warnings


class TestHealthyRun:
    """The undegraded path, for comparison."""

    def test_a_coherent_submission_is_approved(
        self, pipeline: Any, submission: VerificationImages
    ) -> None:
        result = _run(pipeline, submission, "v-coherent")
        assert result.recommendation is Recommendation.APPROVE
        assert result.complete is True
        assert result.assessment_confidence >= 0.70

    def test_every_stage_reports_a_duration(
        self, pipeline: Any, submission: VerificationImages
    ) -> None:
        result = _run(pipeline, submission, "v-timing")
        ran = [stage for stage in result.stages if stage.ran]
        assert ran
        assert all(stage.duration_ms >= 0.0 for stage in ran)

        # The stages run concurrently, so their durations sum to more than the
        # wall clock. Asserting the opposite would pin the pipeline to running
        # them one after another.
        total = result.processing_time.total
        assert total > 0.0
        assert sum(stage.duration_ms for stage in ran) > total * 0.5

    def test_no_images_at_all_is_answered_not_raised(self, pipeline: Any) -> None:
        """An empty submission is a finding the pipeline reports.

        The API refuses this earlier, but the pipeline must not depend on that:
        the worker calls it too.
        """
        result = _run(pipeline, VerificationImages(), "v-nothing")
        assert result.recommendation is not Recommendation.APPROVE
        assert result.identity_confidence_score is None or (
            result.identity_confidence_score == 0.0
        )
