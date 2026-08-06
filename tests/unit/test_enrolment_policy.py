"""Which outcomes write to the duplicate gallery.

`/v1/verify` is now the only route that enrols - `POST /v1/duplicate/enrol` was
removed - so this decision table is the whole of the enrolment contract. It is
tested directly rather than over HTTP because forcing a specific recommendation
through the pipeline needs images that reliably produce it, and one of the four
outcomes (REJECT) is awkward to provoke on demand.

The matrix that matters:

===============  ==========  ==============  ===============
policy           override    recommendation  enrols?
===============  ==========  ==============  ===============
never            None        APPROVE         no
on_approve       None        APPROVE         yes
on_approve       None        MANUAL_REVIEW   no
unless_rejected  None        MANUAL_REVIEW   yes
any              True        REJECT          **no**
any              False       APPROVE         no
never            True        APPROVE         yes
===============  ==========  ==============  ===============
"""

from __future__ import annotations

import pytest

from hamqadam_ai.core.config import Settings
from hamqadam_ai.core.constants import Recommendation
from hamqadam_ai.pipelines.verification import VerificationPipeline

pytestmark = pytest.mark.unit


def _pipeline(policy: str) -> VerificationPipeline:
    """A pipeline with no services - only `_should_enrol` is exercised."""
    settings = Settings()
    settings.duplicate.enrol_policy = policy  # type: ignore[assignment]
    return VerificationPipeline(services={}, settings=settings)


class TestPolicyDrivesEnrolment:
    """Configuration decides when the request says nothing.

    This is what makes one API call sufficient. Previously enrolment needed the
    caller to pass `enrol_on_success=true`, so a Backend that did not know the
    flag existed never populated the gallery - and every duplicate search then
    ran against nothing and found nothing, silently.
    """

    @pytest.mark.parametrize(
        ("policy", "recommendation", "expected"),
        [
            ("never", Recommendation.APPROVE, False),
            ("never", Recommendation.MANUAL_REVIEW, False),
            ("on_approve", Recommendation.APPROVE, True),
            ("on_approve", Recommendation.MANUAL_REVIEW, False),
            ("unless_rejected", Recommendation.APPROVE, True),
            ("unless_rejected", Recommendation.MANUAL_REVIEW, True),
        ],
    )
    def test_matrix(
        self, policy: str, recommendation: Recommendation, expected: bool
    ) -> None:
        assert (
            _pipeline(policy)._should_enrol(recommendation, None)  # noqa: SLF001
            is expected
        )

    def test_on_approve_leaves_the_multi_account_hole(self) -> None:
        """Documents the reason `unless_rejected` exists.

        Under `on_approve` a reviewed applicant is never enrolled, so when they
        open a second account there is nothing to collide with - the
        multi-account case the gallery exists to catch is the one it misses.
        """
        assert not _pipeline("on_approve")._should_enrol(  # noqa: SLF001
            Recommendation.MANUAL_REVIEW, None
        )
        assert _pipeline("unless_rejected")._should_enrol(  # noqa: SLF001
            Recommendation.MANUAL_REVIEW, None
        )


class TestRejectIsNeverEnrolled:
    """The one rule no policy or override may break."""

    @pytest.mark.parametrize(
        "policy", ["never", "on_approve", "unless_rejected"]
    )
    @pytest.mark.parametrize("override", [None, True, False])
    def test_a_rejection_never_enrols(
        self, policy: str, override: bool | None
    ) -> None:
        """Under every policy, and even when the caller forces enrolment.

        Storing a refused applicant's template would make it collide with their
        next legitimate attempt - the service would manufacture a duplicate out
        of its own earlier refusal, and the person could never get in.
        """
        assert not _pipeline(policy)._should_enrol(  # noqa: SLF001
            Recommendation.REJECT, override
        )


class TestRequestOverride:
    """`enrol_on_success` overrides policy, but only for non-rejections."""

    def test_false_suppresses_under_a_permissive_policy(self) -> None:
        assert not _pipeline("unless_rejected")._should_enrol(  # noqa: SLF001
            Recommendation.APPROVE, False
        )

    def test_true_forces_under_a_restrictive_policy(self) -> None:
        assert _pipeline("never")._should_enrol(  # noqa: SLF001
            Recommendation.APPROVE, True
        )

    def test_none_means_defer_to_policy(self) -> None:
        """The distinction the form parser was destroying.

        It coerced an absent field to `False`, which read as an explicit "do not
        enrol" - so `enrol_policy` was unreachable for any caller that simply
        did not send the field, which is every caller unaware of it.
        """
        assert _pipeline("on_approve")._should_enrol(  # noqa: SLF001
            Recommendation.APPROVE, None
        )
        assert not _pipeline("never")._should_enrol(  # noqa: SLF001
            Recommendation.APPROVE, None
        )
