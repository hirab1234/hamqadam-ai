"""A photograph the applicant supplied and the service could not read must not
be treated as evidence in their favour.

Reported: a live selfie of the account holder, a profile photograph of a
*different* person at 185x272, a matching CNIC, and a secondary image at
179x282. Both small images were too small for the detector, so both analysed
successfully and reported FACE_NOT_DETECTED. The result was APPROVE.

    profile    compared=false  NOT_COMPARED  "could not be embedded"
    secondary  compared=false  NOT_COMPARED  "could not be embedded"
    cnic       compared=true   STRONG_MATCH  77.98
    effective_weights {"cnic": 1.0}      identity 77.98      APPROVE

Two existing guards both missed it, and the reason each missed is the point:

* `MANDATORY_STAGE_INCOMPLETE` reads the stage record, which tracks execution.
  The profile stage recorded ran=true, succeeded=true - concluding "no face
  here" *is* a successful analysis.
* The Module 4 failure caps fire on a decision of FAILED. NOT_COMPARED is not
  FAILED, so nothing capped, and the weights renormalised onto the CNIC alone.

Together those made degrading an image strictly more effective than submitting
an honest one: a mismatching face caps identity at 45, the same face shrunk
below the detector's reach costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.decision.engine import BLOCKING_CONDITIONS
from hamqadam_ai.pipelines.verification import (
    VerificationImages,
    VerificationPipeline,
    _Stage,
)


@dataclass
class _Photo:
    """Stands in for the pipeline's per-image record."""

    embedding: Any = None


def _image() -> Any:
    """Any array; only its presence is read."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture
def pipeline() -> VerificationPipeline:
    """The helpers under test read only `settings`, so services stay empty."""
    return VerificationPipeline(services={}, settings=get_settings())


def _complete_stages() -> dict[str, _Stage]:
    """Every mandatory stage ran and succeeded - as it did in the report."""
    return {
        name: _Stage(name=name, ran=True, succeeded=True)
        for name in get_settings().decision.approve.mandatory_stages
    }


class TestWhichImagesAreUncomparable:
    """The request decides what was expected; the result decides what arrived."""

    def test_a_supplied_profile_with_no_template_is_reported(
        self, pipeline: VerificationPipeline
    ) -> None:
        missing = pipeline._uncomparable_supplied_images(
            VerificationImages(live_selfie=_image(), profile=_image()),
            _Photo(embedding=None),
            [],
        )
        assert missing == ["profile"]

    def test_a_supplied_secondary_with_no_template_is_reported(
        self, pipeline: VerificationPipeline
    ) -> None:
        missing = pipeline._uncomparable_supplied_images(
            VerificationImages(live_selfie=_image(), secondaries=[_image()]),
            None,
            [_Photo(embedding=None)],
        )
        assert missing == ["secondary[0]"]

    def test_the_reported_submission_reports_both(
        self, pipeline: VerificationPipeline
    ) -> None:
        missing = pipeline._uncomparable_supplied_images(
            VerificationImages(
                live_selfie=_image(),
                profile=_image(),
                cnic=_image(),
                secondaries=[_image()],
            ),
            _Photo(embedding=None),
            [_Photo(embedding=None)],
        )
        assert missing == ["profile", "secondary[0]"]

    def test_only_the_unreadable_secondary_is_named(
        self, pipeline: VerificationPipeline
    ) -> None:
        """Indices must survive - reporting the wrong one sends a reviewer to
        the wrong photograph."""
        missing = pipeline._uncomparable_supplied_images(
            VerificationImages(
                live_selfie=_image(), secondaries=[_image(), _image(), _image()]
            ),
            None,
            [_Photo("t"), _Photo(embedding=None), _Photo("t")],
        )
        assert missing == ["secondary[1]"]

    def test_a_stage_that_returned_nothing_at_all_counts(
        self, pipeline: VerificationPipeline
    ) -> None:
        """A crashed stage yields None rather than a record with no template."""
        missing = pipeline._uncomparable_supplied_images(
            VerificationImages(live_selfie=_image(), profile=_image()), None, []
        )
        assert missing == ["profile"]

    def test_images_that_were_never_supplied_are_not_reported(
        self, pipeline: VerificationPipeline
    ) -> None:
        """Absence is a different finding, handled by the stage record. Counting
        it here would block every selfie-plus-CNIC submission twice over."""
        missing = pipeline._uncomparable_supplied_images(
            VerificationImages(live_selfie=_image(), cnic=_image()), None, []
        )
        assert missing == []

    def test_usable_images_are_not_reported(
        self, pipeline: VerificationPipeline
    ) -> None:
        missing = pipeline._uncomparable_supplied_images(
            VerificationImages(
                live_selfie=_image(), profile=_image(), secondaries=[_image()]
            ),
            _Photo(embedding="template"),
            [_Photo(embedding="template")],
        )
        assert missing == []


class TestItBlocksTheApproval:
    """The finding has to reach the decision, not just the log."""

    def test_an_uncomparable_image_blocks(
        self, pipeline: VerificationPipeline
    ) -> None:
        blocking = pipeline._blocking_conditions(
            {"embedding": "t"},
            _Matching(identity_available=True),
            None,
            _complete_stages(),
            ["profile"],
        )
        assert "SUPPLIED_IMAGE_NOT_COMPARED" in blocking

    def test_nothing_blocks_when_every_supplied_image_was_used(
        self, pipeline: VerificationPipeline
    ) -> None:
        """Guards the guard: if this ever returns a blocker, the test above
        proves nothing."""
        blocking = pipeline._blocking_conditions(
            {"embedding": "t"},
            _Matching(identity_available=True),
            None,
            _complete_stages(),
            [],
        )
        assert blocking == []

    def test_it_blocks_even_though_every_mandatory_stage_succeeded(
        self, pipeline: VerificationPipeline
    ) -> None:
        """The exact hole. Stage bookkeeping was clean in the report."""
        stages = _complete_stages()
        assert all(s.ran and s.succeeded for s in stages.values())

        blocking = pipeline._blocking_conditions(
            {"embedding": "t"},
            _Matching(identity_available=True),
            None,
            stages,
            ["profile", "secondary[0]"],
        )
        assert "MANDATORY_STAGE_INCOMPLETE" not in blocking
        assert "SUPPLIED_IMAGE_NOT_COMPARED" in blocking


class TestTheReasonIsExplained:
    """A blocker with no message reads as an internal error to the Backend."""

    @pytest.mark.parametrize(
        "code", ["SUPPLIED_IMAGE_NOT_COMPARED", "MANDATORY_STAGE_INCOMPLETE"]
    )
    def test_the_code_has_a_message(self, code: str) -> None:
        assert code in BLOCKING_CONDITIONS
        assert len(BLOCKING_CONDITIONS[code]) > 40


@dataclass
class _Matching:
    """Minimal stand-in for the matching result."""

    identity_available: bool = True
