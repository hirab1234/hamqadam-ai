"""MODULE 10 - the end-to-end verification pipeline.

What runs, and in what order
----------------------------
::

    phase A, all concurrent
      live selfie   detect -> quality -> embed
      profile       detect -> embed ; analyse (authenticity, subject, quality)
      secondary[i]  detect -> embed ; analyse
      CNIC          OCR ; portrait extract ; authenticity detectors
    phase B, concurrent
      duplicate     needs the selfie template
      matching      needs every template
    phase C
      fraud         needs every finding
      decision      needs identity confidence and fraud risk

Phase A is where the wall-clock goes, and every item in it is independent, so
it is dispatched together. ONNX Runtime releases the GIL inside its kernels, so
``asyncio.to_thread`` over a bounded pool gives real parallelism rather than
the appearance of it.

Degrade, never abort
--------------------
No stage failure ends the request. A verification where the CNIC was
unreadable still has a face comparison worth reporting, and a Backend that
receives an exception learns nothing about the six images that were fine. Every
stage records whether it ran, whether it succeeded, and what it cost; the
recommendation is then made from whatever evidence exists, and
``assessment_confidence`` says how much of it there was.

One deadline, shared
--------------------
The whole request gets a single budget from ``server.request_timeout_seconds``
and every stage asks how much is left. The alternative - a generous timeout per
stage - lets their sum quietly exceed the caller's own timeout, which is how a
service ends up holding connections nobody is waiting on any more.

Two optimisations, one taken and one refused
--------------------------------------------
**Taken**: the pipeline detects each face once and hands the result to both the
quality/embedding path and Module 7, which would otherwise detect again -
roughly 300 ms per image for an identical answer.

**Refused**: sharing the rectified CNIC between Modules 5 and 6. It sounds
obviously right, and Module 6 exposes ``prepare_card`` partly for it. Measured,
``rectify_document`` costs 90 ms against a 4293 ms CNIC read - **2.1%** - and
buying that back means coupling two services that currently share nothing but
an image. Not worth it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import Settings, get_settings
from hamqadam_ai.core.constants import ImageRole, Recommendation, RiskLevel
from hamqadam_ai.core.exceptions import HamqadamError
from hamqadam_ai.decision.engine import DecisionEngine
from hamqadam_ai.embeddings.base import FaceEmbedding
from hamqadam_ai.fraud_detection import SignalCollector
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.observability import (
    record_budget_exhausted,
    record_verification,
    set_gallery_size,
)
from hamqadam_ai.schemas.common import AnalysisWarning, ModelVersions, ProcessingTime
from hamqadam_ai.schemas.verification import (
    StageStatus,
    VerificationRequest,
    VerificationResult,
)
from hamqadam_ai.utils.timing import Deadline

log = get_logger(__name__)

BgrImage = npt.NDArray[np.uint8]


@dataclass(slots=True)
class VerificationImages:
    """The images one verification submits.

    Attributes:
        live_selfie: The biometric reference for the whole decision.
        profile: The account's main photograph.
        secondaries: Additional photographs, in upload order.
        cnic: A photograph of the identity card's front.
    """

    live_selfie: BgrImage | None = None
    profile: BgrImage | None = None
    secondaries: list[BgrImage] = field(default_factory=list)
    cnic: BgrImage | None = None

    @property
    def count(self) -> int:
        """How many images were actually supplied."""
        supplied = [self.live_selfie, self.profile, self.cnic]
        return sum(1 for image in supplied if image is not None) + len(
            self.secondaries
        )


@dataclass(slots=True)
class _Photo:
    """A user photograph's analysis and its template.

    A small carrier rather than an attribute bolted onto the analysis, because
    :class:`ProfileAnalysisResult` is a strict response model and an embedding
    is biometric data that has no business on one.
    """

    analysis: Any
    embedding: FaceEmbedding | None = None


@dataclass(slots=True)
class _Stage:
    """Bookkeeping for one stage of the pipeline."""

    name: str
    ran: bool = False
    succeeded: bool = False
    duration_ms: float = 0.0
    error: str | None = None

    def as_status(self) -> StageStatus:
        """Render for the response."""
        return StageStatus(
            stage=self.name,
            ran=self.ran,
            succeeded=self.succeeded,
            duration_ms=round(self.duration_ms, 2),
            error=self.error,
        )


class VerificationPipeline:
    """Runs one verification end to end.

    Args:
        services: The constructed module services, keyed by name.
        settings: Service configuration.
    """

    __slots__ = ("_config", "_decision", "_services", "_settings")

    def __init__(self, *, services: dict[str, Any], settings: Settings) -> None:
        self._services = services
        self._settings = settings
        self._config = settings.decision
        self._decision = DecisionEngine(settings.decision)

    # -- Public surface ------------------------------------------------------ #

    async def verify(
        self, request: VerificationRequest, images: VerificationImages
    ) -> VerificationResult:
        """Run the whole verification.

        Args:
            request: Identifiers and per-request options.
            images: The submitted images.

        Returns:
            The complete result. Never raises for a business outcome, and
            never raises for a single stage failing.
        """
        started = time.perf_counter()
        deadline = Deadline(self._settings.server.request_timeout_seconds)
        stages: dict[str, _Stage] = {}
        warnings: list[AnalysisWarning] = []

        # -- Phase A: everything that depends only on one image -------------- #
        selfie_task = self._run(
            stages, deadline, "selfie", self._analyse_selfie, images.live_selfie
        )
        profile_task = self._run(
            stages, deadline, "profile", self._analyse_photo, images.profile,
            ImageRole.PROFILE_IMAGE,
        )
        secondary_tasks = [
            self._run(
                stages, deadline, f"secondary[{index}]", self._analyse_photo,
                image, ImageRole.SECONDARY_IMAGE,
            )
            for index, image in enumerate(images.secondaries)
        ]
        ocr_task = self._run(
            stages, deadline, "cnic_ocr", self._read_cnic, images.cnic
        )
        portrait_task = self._run(
            stages, deadline, "cnic_portrait", self._extract_portrait, images.cnic
        )
        cnic_auth_task = self._run(
            stages, deadline, "cnic_authenticity", self._cnic_authenticity,
            images.cnic,
        )

        (
            selfie,
            profile,
            secondaries,
            ocr,
            portrait,
            cnic_authenticity,
        ) = await asyncio.gather(
            selfie_task,
            profile_task,
            asyncio.gather(*secondary_tasks) if secondary_tasks else _none_list(),
            ocr_task,
            portrait_task,
            cnic_auth_task,
        )

        # -- Phase B: needs the templates phase A produced ------------------- #
        selfie_embedding = selfie.get("embedding") if selfie else None
        portrait_embedding = portrait.get("embedding") if portrait else None
        profile_analysis = profile.analysis if profile else None
        secondary_analyses = [
            entry.analysis for entry in secondaries if entry is not None
        ]

        duplicate_task = self._run(
            stages, deadline, "duplicate", self._check_duplicate,
            selfie_embedding, request.user_reference,
        )
        matching_task = self._run(
            stages, deadline, "matching", self._match,
            selfie_embedding,
            profile.embedding if profile else None,
            [entry.embedding if entry is not None else None for entry in secondaries],
            portrait_embedding,
        )
        duplicate, matching = await asyncio.gather(duplicate_task, matching_task)

        # -- Phase C: aggregate and recommend -------------------------------- #
        fraud = self._assess_fraud(
            stages=stages,
            selfie=selfie,
            profile=profile_analysis,
            secondaries=secondary_analyses,
            ocr=ocr,
            portrait=portrait,
            cnic_authenticity=cnic_authenticity,
            matching=matching,
            duplicate=duplicate,
        )

        identity_confidence = (
            matching.identity_confidence_score if matching is not None else None
        )
        outcome = self._decision.decide(
            identity_confidence=identity_confidence,
            fraud_risk=fraud.fraud_risk_score if fraud else 0.0,
            fraud_level=fraud.fraud_risk_level if fraud else RiskLevel.LOW,
            assessment_confidence=fraud.assessment_confidence if fraud else 0.0,
            blocking=self._blocking_conditions(
                selfie, matching, duplicate, stages
            ),
            rejecting=self._rejecting_conditions(duplicate),
        )

        if cnic_authenticity is not None and cnic_authenticity.triggered:
            warnings.append(
                AnalysisWarning(
                    code="CNIC_IMAGE_MAY_BE_SCREEN_RECAPTURE",
                    message=(
                        "The identity document shows the periodic interference "
                        "left by photographing a screen. Reported for review "
                        "rather than scored: the detector was calibrated on "
                        "photographs of people, and on document images its "
                        "margin over an honest capture is thin."
                    ),
                    stage="cnic_authenticity",
                    detail={
                        "confidence": round(cnic_authenticity.confidence, 4)
                    },
                )
            )

        portrait_result = portrait.get("result") if portrait else None
        for source in (
            selfie.get("result") if selfie else None,
            profile_analysis,
            ocr,
            portrait_result,
            matching,
            duplicate,
        ):
            warnings.extend(getattr(source, "warnings", None) or [])
        for entry in secondary_analyses:
            warnings.extend(getattr(entry, "warnings", None) or [])

        result = self._to_schema(
            request=request,
            stages=stages,
            selfie=selfie,
            profile=profile_analysis,
            secondaries=secondary_analyses,
            ocr=ocr,
            portrait=portrait,
            matching=matching,
            duplicate=duplicate,
            fraud=fraud,
            outcome=outcome,
            warnings=warnings,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            deadline=deadline,
        )

        if self._should_enrol(result.recommendation, request.enrol_on_success):
            self._enrol(selfie_embedding, request.user_reference)

        record_verification(result)
        log.info("verification.completed", **result.summary())
        return result

    def verify_sync(
        self, request: VerificationRequest, images: VerificationImages
    ) -> VerificationResult:
        """Run a verification from synchronous code, for scripts and workers."""
        return asyncio.run(self.verify(request, images))

    def describe(self) -> dict[str, Any]:
        """Serialisable summary of every wired service, for ``/health``."""
        described: dict[str, Any] = {
            "decision": self._decision.describe(),
            "request_timeout_seconds": self._settings.server.request_timeout_seconds,
        }
        for name, service in self._services.items():
            describe = getattr(service, "describe", None)
            if callable(describe):
                try:
                    described[name] = describe()
                except Exception as exc:  # noqa: BLE001 - health must not fail
                    described[name] = {"error": str(exc)}
        return described

    # -- Stage runner --------------------------------------------------------- #

    async def _run(
        self,
        stages: dict[str, _Stage],
        deadline: Deadline,
        name: str,
        work: Any,
        *args: Any,
    ) -> Any:
        """Run one stage on a worker thread, recording what happened.

        A stage that raises is recorded and returns ``None``. That is the whole
        degradation strategy: the pipeline continues, the result says the stage
        failed, and the fraud engine treats it as an absence rather than a
        pass.

        A stage that would start after the shared budget is exhausted does not
        start. Beginning a four-second OCR read when nobody is still waiting
        for the answer costs a worker thread and delivers nothing.
        """
        stage = _Stage(name=name)
        stages[name] = stage

        # The first argument is always the stage's subject - an image, or a
        # template. Without it there is nothing to run, which is recorded as
        # "did not run" rather than as a failure.
        if args and args[0] is None:
            stage.error = "no input supplied"
            return None

        if deadline.expired:
            stage.error = (
                f"skipped: the {deadline.budget:.0f}s request budget was "
                f"exhausted before this stage could start"
            )
            log.warning("verification.stage_skipped", stage=name, reason="deadline")
            return None

        started = time.perf_counter()
        stage.ran = True
        try:
            outcome = await asyncio.to_thread(work, *args)
        except HamqadamError as exc:
            stage.error = f"{type(exc).__name__}: {exc}"
            log.warning("verification.stage_failed", stage=name, reason=str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 - one stage must not end the request
            stage.error = f"{type(exc).__name__}: {exc}"
            log.error("verification.stage_error", stage=name, reason=str(exc))
            return None
        finally:
            stage.duration_ms = (time.perf_counter() - started) * 1000.0

        stage.succeeded = outcome is not None
        return outcome

    # -- Stage implementations ------------------------------------------------ #

    def _analyse_selfie(self, image: BgrImage) -> dict[str, Any] | None:
        """Detect, assess and embed the live selfie.

        The reference for every comparison, so it is the one image whose
        failure genuinely blocks a recommendation.
        """
        detection = self._services["detection"].detect(
            image, role=ImageRole.LIVE_SELFIE
        )
        quality = self._services["quality"].assess(
            image, role=ImageRole.LIVE_SELFIE, detection=detection
        )

        embedding = self._embed_or_none(
            image, role=ImageRole.LIVE_SELFIE, detection=detection
        )
        return {"result": detection, "quality": quality, "embedding": embedding}

    def _analyse_photo(self, image: BgrImage, role: ImageRole) -> _Photo:
        """Analyse a profile or secondary photograph, and embed it.

        Detects once and passes the result to Module 7, which would otherwise
        detect again for the same answer.
        """
        detection = self._services["detection"].detect(image, role=role)
        analysis = self._services["profile"].analyse(
            image, role=role, detection=detection
        )

        embedding = self._embed_or_none(image, role=role, detection=detection)
        return _Photo(analysis=analysis, embedding=embedding)

    def _embed_or_none(
        self, image: BgrImage, *, role: ImageRole, detection: Any
    ) -> FaceEmbedding | None:
        """Embed a detected face, or return None with the reason logged.

        Two guards, both learned from a real failure.

        **A primary face must exist.** A frame containing two equally-sized
        people sets `face_detected` True - faces *were* found - while the policy
        selects no primary face. Embedding then reached the aligner with nothing
        to align and raised
        `ValueError: Alignment needs either landmarks or a bounding box`.

        The guard is `primary_face`, deliberately **not** `passed`. Gating on
        `passed` was the first attempt and it was too strict: a badly-posed but
        perfectly single face also fails the policy, and refusing to embed it
        threw away a real comparison. Measured on an impostor submission, that
        mistake turned a correct "identity 43.8, CNIC_FACE_MISMATCH" into "no
        identity established" - the system stopped catching the impostor for
        the stated reason and fell back to a generic review. A pose warning
        should colour the result, not erase it.

        **A bare `except HamqadamError` was not enough.** That ValueError is not
        a HamqadamError, so it escaped, failed the whole selfie stage, and threw
        away the detection result with it. The response then carried
        `selfie_detection: null` and told the reviewer "No usable live selfie
        was supplied" - which is wrong and unactionable, because a perfectly
        good selfie *was* supplied and the real problem was that it had two
        people in it. Worse, MULTIPLE_FACES_DETECTED never reached the fraud
        engine, so a genuine fraud signal was lost.

        Returning None keeps the detection result - and its error code - in the
        response, where both the reviewer and the fraud engine can see it.
        """
        if not detection.face_detected or detection.primary_face is None:
            return None
        try:
            embedding: FaceEmbedding = self._services[
                "embedding"
            ].embed_to_vector(image, role=role, detection=detection)
        except HamqadamError as exc:
            log.info(
                "verification.embedding_failed", role=str(role), reason=str(exc)
            )
            return None
        except Exception as exc:  # noqa: BLE001 - must not discard the detection
            log.warning(
                "verification.embedding_error",
                role=str(role),
                reason=f"{type(exc).__name__}: {exc}",
            )
            return None
        return embedding

    def _read_cnic(self, image: BgrImage) -> Any:
        """Read the identity card."""
        return self._services["ocr"].read(image)

    def _extract_portrait(self, image: BgrImage) -> dict[str, Any] | None:
        """Locate and embed the portrait printed on the card."""
        portrait, embedding = self._services["cnic_face"].extract_portrait(image)
        return {"result": portrait, "embedding": embedding}

    def _cnic_authenticity(self, image: BgrImage) -> Any:
        """Check the card image for signs of a screen recapture.

        **Only the moire detector**, and this is the interesting part. Module
        7's detectors look role-independent, so running all four on the card
        seemed obviously right. Measured, they do not transfer:

            case                          screenshot  moire  print  synthetic
            a rendered card layout             1.000  0.000  0.000      0.231
            a card photographed on a desk      0.000  0.393  0.000      0.000
            a card shot off a screen           0.000  1.000  0.000      0.000
            a card shot off a print            0.000  0.000  0.063      0.000

        The screenshot detector keys on flat full-width bands, and a document
        **is** a layout of flat bands - it fires at full confidence on a
        perfectly legitimate card. Print recapture fails the other way, missing
        an actual print at 0.063, because its high-frequency measure was
        anchored on facial texture that a document does not have.

        Only moire transfers, because it keys on the physics of the capture
        rather than on the content. Even that has a thin margin - 0.393 on an
        honestly photographed card against a 0.55 trigger - so its finding is
        surfaced as a warning for a human and is **not** scored by Module 9.
        """
        from hamqadam_ai.authenticity import AuthenticityContext, MoireDetector

        detector = MoireDetector(self._settings.profile.moire)
        return detector.safe_analyse(AuthenticityContext(image=image))

    def _check_duplicate(
        self, embedding: FaceEmbedding | None, reference: str | None
    ) -> Any:
        """Search the gallery for this face."""
        return self._services["duplicate"].check(embedding, reference=reference)

    def _match(
        self,
        selfie: FaceEmbedding | None,
        profile: FaceEmbedding | None,
        secondaries: list[FaceEmbedding | None],
        cnic: FaceEmbedding | None,
    ) -> Any:
        """Compare every template against the live selfie."""
        return self._services["matching"].match(
            selfie=selfie,
            profile=profile,
            secondaries=secondaries,
            cnic=cnic,
            secondary_labels=[f"secondary[{i}]" for i in range(len(secondaries))],
        )

    def _should_enrol(
        self, recommendation: Recommendation, override: bool | None
    ) -> bool:
        """Whether this outcome writes to the gallery.

        Configuration decides; the request may override. That ordering is what
        makes one ``/v1/verify`` call sufficient - previously enrolment required
        the caller to pass ``enrol_on_success=true`` *and* the result to be an
        APPROVE, so a Backend that did not know to set the flag silently never
        populated the gallery, and every duplicate search ran against nothing.

        Args:
            recommendation: What the decision engine concluded.
            override: The request's ``enrol_on_success``. ``None`` - the normal
                case - defers to policy. ``False`` suppresses enrolment for this
                one submission; ``True`` forces it even where policy is
                ``never``.

        A REJECT is never enrolled under any policy, including a forced
        override: storing a refused applicant's face would make it collide with
        their next legitimate attempt.
        """
        if recommendation is Recommendation.REJECT:
            return False
        if override is not None:
            return override

        policy = self._settings.duplicate.enrol_policy
        if policy == "never":
            return False
        if policy == "unless_rejected":
            return True
        return recommendation is Recommendation.APPROVE

    def _enrol(
        self, embedding: FaceEmbedding | None, reference: str | None
    ) -> None:
        """Add the selfie to the duplicate gallery, on request."""
        if embedding is None or not reference:
            return
        try:
            self._services["duplicate"].enrol(embedding, reference=reference)
        except Exception as exc:  # noqa: BLE001 - enrolment must not fail a verify
            log.warning("verification.enrolment_failed", reason=str(exc))
            return

        # Keep the gauge honest now that this is the only place that enrols.
        try:
            set_gallery_size(int(self._services["duplicate"].gallery_size()))
        except Exception as exc:  # noqa: BLE001 - a metric must not fail a verify
            log.debug("verification.gallery_size_unavailable", reason=str(exc))

    # -- Aggregation ---------------------------------------------------------- #

    def _assess_fraud(
        self,
        *,
        stages: dict[str, _Stage],
        selfie: dict[str, Any] | None,
        profile: Any,
        secondaries: list[Any],
        ocr: Any,
        portrait: dict[str, Any] | None,
        cnic_authenticity: Any,
        matching: Any,
        duplicate: Any,
    ) -> Any:
        """Collect every finding and score it.

        Driven through the collector rather than
        :meth:`FraudRiskService.assess` because the pipeline knows which stages
        ran and which merely returned nothing - a distinction the results
        themselves cannot always express.
        """
        service = self._services["fraud"]
        collector = SignalCollector(self._settings.fraud.signal_weights)

        collector.collect_warnings(
            selfie.get("result") if selfie else None, stage="detection"
        )

        # A *rejected* detection reports through `error_code`, not `warnings`,
        # so collecting only warnings loses it. That mattered concretely: a
        # selfie containing two equally-sized people yields
        # MULTIPLE_FACES_DETECTED - a catalogued, scored fraud signal - and the
        # fraud engine never saw it. The verification still went to review, but
        # for the wrong reason ("no usable selfie") and with the actual finding
        # absent from the risk score.
        selfie_detection = selfie.get("result") if selfie else None
        detection_error = getattr(selfie_detection, "error_code", None)
        if detection_error is not None:
            collector.add_code(str(detection_error), stage="detection")

        collector.collect_warnings(
            selfie.get("quality") if selfie else None, stage="quality"
        )
        collector.collect_warnings(profile, stage="profile")
        for index, entry in enumerate(secondaries):
            collector.collect_warnings(entry, stage=f"secondary[{index}]")
        collector.collect_warnings(ocr, stage="ocr")

        portrait_result = portrait.get("result") if portrait else None
        collector.collect_warnings(portrait_result, stage="cnic_face")
        if portrait_result is not None:
            foreign = int(getattr(portrait_result, "foreign_face_count", 0) or 0)
            if foreign:
                collector.add_code(
                    "CNIC_FOREIGN_FACE_PRESENT",
                    stage="cnic_face",
                    detail={"count": foreign},
                )
            if not getattr(portrait_result, "found", True):
                collector.add_code("CNIC_FACE_NOT_FOUND", stage="cnic_face")

        # The card's own capture medium. Surfaced, never scored - see
        # `_cnic_authenticity` for the measurement that decided that.
        if cnic_authenticity is None:
            collector.mark_unavailable("cnic_authenticity")

        collector.collect_matching(matching)
        collector.collect_duplicate(duplicate)

        for name, stage in stages.items():
            if not stage.ran or stage.error:
                collector.mark_unavailable(name)

        return service.assess_from_signals(collector)

    def _blocking_conditions(
        self,
        selfie: dict[str, Any] | None,
        matching: Any,
        duplicate: Any,
        stages: dict[str, _Stage],
    ) -> list[str]:
        """Conditions that make a recommendation impossible rather than negative.

        A confirmed duplicate is here rather than left to the fraud score alone.
        The score route does work - a duplicate carries weight 0.70, which
        aggregates to about 70 and clears the reject threshold of 65 - but it
        works by arithmetic coincidence. Retune the weight, raise the reject
        threshold, or let a family cap bite, and the same face on two accounts
        could quietly start being approved with no test failing.

        A confirmed duplicate is a categorical finding: this face is already
        enrolled under a different reference. Stating that directly means the
        outcome no longer depends on where a threshold happens to sit.
        """
        blocking: list[str] = []
        if selfie is None or selfie.get("embedding") is None:
            blocking.append("NO_LIVE_SELFIE")
        elif matching is None or not getattr(matching, "identity_available", False):
            blocking.append("NO_IDENTITY_COMPARISON")

        if (
            duplicate is not None
            and getattr(duplicate, "duplicate_found", False)
            and self._settings.duplicate.on_duplicate != "reject"
        ):
            blocking.append("DUPLICATE_FACE_NEEDS_REVIEW")

        # Every mandatory stage must have run *and* succeeded before an
        # automatic approval is possible.
        #
        # This closes a demonstrated false approval. A submission carried a
        # profile photograph of a different person and the profile stage failed
        # mid-request. Module 4 renormalises identity weights over the
        # comparisons that produced a result, so the absent profile comparison
        # was dropped, the weights collapsed to `{cnic: 1.0}`, and identity
        # confidence became the CNIC score alone - 95.0. With
        # `assessment_confidence` at 5/6 = 0.83, over the 0.70 floor, every
        # approve condition was satisfied and the answer was APPROVE.
        #
        # The wrong photograph was never compared, so it never counted against
        # the applicant. `assessment_confidence` did not save it either: losing
        # one of six checks still leaves 0.83, and that field is a *ratio*, so
        # it cannot distinguish which check was lost. Losing the profile
        # comparison and losing the moiré observation score identically.
        #
        # Named stages, not a ratio. An absent mandatory stage is not a smaller
        # number - it is a different question, and the only honest answer is
        # that the verification did not happen.
        incomplete = self._incomplete_mandatory_stages(stages)
        if incomplete:
            log.warning(
                "verification.mandatory_stage_incomplete",
                stages=incomplete,
                note="approval is impossible; routed to manual review",
            )
            blocking.append("MANDATORY_STAGE_INCOMPLETE")
        return blocking

    def _incomplete_mandatory_stages(
        self, stages: dict[str, _Stage]
    ) -> list[str]:
        """Mandatory stages that did not run, or ran and failed.

        Reads the pipeline's own stage record rather than inferring from scores.
        That is the point: a score cannot distinguish "this evidence was never
        requested" from "this evidence was requested and did not arrive", and
        those must lead to opposite outcomes. Only the stage record knows.

        A stage absent from the record entirely counts as incomplete - if the
        pipeline never created it, its input was never supplied, and a
        verification missing a mandatory input is not one that can be approved.
        """
        failed: list[str] = []
        for name in self._settings.decision.approve.mandatory_stages:
            stage = stages.get(name)
            if stage is None or not stage.ran or not stage.succeeded:
                failed.append(name)
        return failed

    def _rejecting_conditions(self, duplicate: Any) -> list[str]:
        """Categorical adverse findings that refuse the verification outright.

        Only reached when `duplicate.on_duplicate == "reject"`. Kept apart from
        the blocking list because blocking yields MANUAL_REVIEW - routing a
        duplicate through it would have inverted a deliberate reject policy into
        a review.
        """
        if (
            duplicate is not None
            and getattr(duplicate, "duplicate_found", False)
            and self._settings.duplicate.on_duplicate == "reject"
        ):
            return ["DUPLICATE_FACE_CONFIRMED"]
        return []

    def _to_schema(
        self,
        *,
        request: VerificationRequest,
        stages: dict[str, _Stage],
        selfie: dict[str, Any] | None,
        profile: Any,
        secondaries: list[Any],
        ocr: Any,
        portrait: dict[str, Any] | None,
        matching: Any,
        duplicate: Any,
        fraud: Any,
        outcome: Any,
        warnings: list[AnalysisWarning],
        duration_ms: float,
        deadline: Deadline,
    ) -> VerificationResult:
        """Render everything as the API response model."""
        statuses = [stage.as_status() for stage in stages.values()]
        if deadline.expired:
            record_budget_exhausted()
            warnings.append(
                AnalysisWarning(
                    code="VERIFICATION_BUDGET_EXHAUSTED",
                    message=(
                        f"The {deadline.budget:.0f}s request budget ran out "
                        f"before every stage completed. The recommendation was "
                        f"made from the evidence that was available."
                    ),
                    stage="pipeline",
                    detail={"elapsed_seconds": round(deadline.elapsed, 2)},
                )
            )
        # A stage that raised contributes no sub-result, so it contributed no
        # warning either: with five stages broken this list came back empty
        # while `stages[].error` held five failures. `warnings` is the field a
        # caller reads to find out what went wrong, and it was silent in
        # exactly the case where the most had. Each failure is surfaced here.
        for status in statuses:
            if status.ran and not status.succeeded:
                warnings.append(
                    AnalysisWarning(
                        code="VERIFICATION_STAGE_FAILED",
                        message=(
                            f"The {status.stage} stage failed, so its evidence "
                            f"is missing from this recommendation."
                        ),
                        stage=status.stage,
                        detail={"error": status.error or "unknown"},
                    )
                )

        # "Every intended stage ran", which is what the field is documented to
        # mean. The earlier version considered only stages that *had* run, so a
        # selfie-only submission with four skipped stages reported complete=True
        # beside an assessment confidence of 0% - two fields contradicting each
        # other in the same response. A stage exists here only because the
        # pipeline intended to run it, so a skipped one counts against
        # completeness exactly as a failed one does.
        complete = bool(statuses) and all(
            status.ran and status.succeeded for status in statuses
        )

        return VerificationResult(
            verification_id=request.verification_id,
            completed_at=dt.datetime.now(dt.UTC),
            recommendation=outcome.recommendation,
            recommendation_reasons=[
                reason.as_dict() for reason in outcome.reasons
            ],
            automated=outcome.automated,
            identity_confidence_score=(
                matching.identity_confidence_score if matching else None
            ),
            fraud_risk_score=fraud.fraud_risk_score if fraud else 0.0,
            fraud_risk_level=fraud.fraud_risk_level if fraud else RiskLevel.LOW,
            selfie_detection=selfie.get("result") if selfie else None,
            selfie_quality=selfie.get("quality") if selfie else None,
            profile_analysis=profile,
            secondary_analyses=list(secondaries),
            cnic_ocr=ocr,
            cnic_portrait=portrait.get("result") if portrait else None,
            matching=matching,
            duplicate=duplicate,
            fraud=fraud,
            stages=statuses,
            complete=complete,
            assessment_confidence=(
                fraud.assessment_confidence if fraud else 0.0
            ),
            warnings=warnings,
            model_versions=self._model_versions(),
            processing_time=ProcessingTime(
                total=round(duration_ms, 2),
                stages={
                    status.stage: {
                        "duration_ms": status.duration_ms,
                        "calls": 1.0,
                        "average_ms": status.duration_ms,
                    }
                    for status in statuses
                    if status.ran
                },
                overhead=round(
                    max(
                        duration_ms - max(
                            (s.duration_ms for s in statuses if s.ran), default=0.0
                        ),
                        0.0,
                    ),
                    2,
                ),
            ),
            thresholds_validated=False,
        )

    def _model_versions(self) -> ModelVersions | None:
        """Every model that contributed, for reproducibility."""
        try:
            registry = self._services.get("registry")
            if registry is None:
                return None
            return ModelVersions(
                versions=registry.version_map(),
                service_version=self._settings.app.version,
                device=str(self._settings.runtime.device),
            )
        except Exception:  # noqa: BLE001 - provenance must not fail a request
            return None


async def _none_list() -> list[Any]:
    """An awaitable empty list, so the gather above stays uniform."""
    return []


def build_pipeline(settings: Settings | None = None) -> VerificationPipeline:
    """Wire up the pipeline and every service beneath it.

    Raises:
        HamqadamError: if a required model could not be loaded. Deliberately
            fail-fast: a pod that cannot detect a face should fail its
            readiness probe rather than accept traffic it cannot serve.
    """
    from hamqadam_ai.models.registry import get_registry
    from hamqadam_ai.services import (
        build_cnic_face_service,
        build_duplicate_service,
        build_embedding_service,
        build_face_detection_service,
        build_fraud_service,
        build_matching_service,
        build_ocr_service,
        build_profile_service,
        build_quality_service,
    )

    settings = settings or get_settings()
    services: dict[str, Any] = {
        "registry": get_registry(settings),
        "detection": build_face_detection_service(settings),
        "quality": build_quality_service(settings),
        "embedding": build_embedding_service(settings),
        "matching": build_matching_service(settings),
        "profile": build_profile_service(settings),
        "cnic_face": build_cnic_face_service(settings),
        "duplicate": build_duplicate_service(settings),
        "fraud": build_fraud_service(settings),
    }

    try:
        services["ocr"] = build_ocr_service(settings)
    except HamqadamError as exc:
        # The only optional stage. An OCR engine cannot be improvised from
        # OpenCV primitives, but a verification without document text is still
        # worth running - the face comparison is the larger part of it.
        log.error(
            "verification.ocr_unavailable",
            reason=str(exc),
            note="document text will not be read; face verification continues",
        )
        services["ocr"] = None

    log.info("verification.pipeline_ready", services=sorted(services))
    return VerificationPipeline(services=services, settings=settings)


__all__ = ["VerificationImages", "VerificationPipeline", "build_pipeline"]
