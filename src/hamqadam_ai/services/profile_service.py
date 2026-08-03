"""MODULE 7 service - is this photograph usable as a profile picture?

Three questions, answered by three different layers::

    authenticity  ->  is it a genuine camera capture?      (this module)
    subject       ->  is there exactly one person in it?   (Module 1)
    quality       ->  is it good enough to recognise?      (Module 2)

Kept separate because a caller needs to tell them apart. "Upload a photo
instead of a screenshot", "crop this so only you are in it" and "retake this
somewhere brighter" are three different things to say to a user, and a single
merged score can say none of them.

Why authenticity runs first and independently
---------------------------------------------
It is the only one of the three that does not need a face. A cartoon avatar
has no face for Module 1 to find and no meaningful quality score, but the
useful thing to tell the user is not "no face detected" - it is "that is a
drawing". Running the detectors on every image, whatever else fails, is what
makes that possible.

What this module does not claim
-------------------------------
No content moderation, no deepfake detection, no matching against a database
of stock photography or celebrity images. Each needs a trained model or a
reference corpus this project does not have, and a function that returns
``False`` for all of them would look exactly like a working check while
providing none of the protection somebody would rely on.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from hamqadam_ai.authenticity import (
    AuthenticityAssessment,
    AuthenticityContext,
    AuthenticityDetector,
    aggregate,
    build_detectors,
)
from hamqadam_ai.core.config import Settings, get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import HamqadamError
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.schemas.common import AnalysisWarning
from hamqadam_ai.schemas.profile import (
    AuthenticityFindingModel,
    AuthenticitySignalModel,
    ProfileAnalysisResult,
)
from hamqadam_ai.services.face_detection_service import FaceDetectionService
from hamqadam_ai.services.quality_service import QualityService

log = get_logger(__name__)

BgrImage = npt.NDArray[np.uint8]


class ProfileAnalysisService:
    """Analyses a user-supplied photograph for use as a profile picture.

    Args:
        detectors: The authenticity detectors.
        detection: Module 1, for subject presence and count.
        quality: Module 2.
        settings: Service configuration.
    """

    __slots__ = ("_config", "_detection", "_detectors", "_quality", "_settings")

    def __init__(
        self,
        *,
        detectors: list[AuthenticityDetector],
        detection: FaceDetectionService,
        quality: QualityService,
        settings: Settings,
    ) -> None:
        self._detectors = detectors
        self._detection = detection
        self._quality = quality
        self._settings = settings
        self._config = settings.profile

    # -- Public surface ------------------------------------------------------ #

    def analyse(
        self,
        image: BgrImage,
        *,
        role: ImageRole = ImageRole.PROFILE_IMAGE,
        detection: Any = None,
    ) -> ProfileAnalysisResult:
        """Analyse one photograph.

        Args:
            image: The photograph, BGR uint8.
            role: Which image in the verification request this is. Selects the
                quality bar; the authenticity detectors are role-independent,
                because a screenshot is a screenshot whatever slot it was
                uploaded into.
            detection: A Module 1 result for this image, when the caller
                already has one. The pipeline does - it needs the same
                detection to produce an embedding - and detecting a second time
                would cost roughly 300 ms per image for an identical answer.

        Returns:
            The populated result. A business rejection is reported through
            ``usable_as_profile`` and ``error_code``, never raised.
        """
        started = time.perf_counter()

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Profile analysis expects a (H, W, 3) BGR array; got {image.shape}"
            )

        assessment = self.assess_authenticity(image)

        if detection is None:
            detection = self._safe_detect(image, role)
        face_count = detection.face_count if detection is not None else 0
        face_detected = bool(detection is not None and detection.face_detected)

        quality_score = 0.0
        quality_usable = False
        if detection is not None and face_detected:
            quality = self._safe_quality(image, role, detection)
            if quality is not None:
                quality_score = quality.image_quality_score
                quality_usable = quality.usable

        result = self._to_schema(
            assessment=assessment,
            role=role,
            image=image,
            face_detected=face_detected,
            face_count=face_count,
            face_visibility=(
                detection.face_visibility_score if detection is not None else 0.0
            ),
            quality_score=quality_score,
            quality_usable=quality_usable,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

        log.info("profile.completed", **result.summary())
        return result

    def assess_authenticity(self, image: BgrImage) -> AuthenticityAssessment:
        """Run the detectors alone, with no face or quality analysis.

        Separated because it is the cheap part - no model weights, tens of
        milliseconds - and because Module 9 may want it for an image the
        identity path has already rejected.
        """
        context = AuthenticityContext(image=image)
        signals = [detector.safe_analyse(context) for detector in self._detectors]
        return aggregate(signals)

    async def analyse_async(
        self,
        image: BgrImage,
        *,
        role: ImageRole = ImageRole.PROFILE_IMAGE,
        detection: Any = None,
    ) -> ProfileAnalysisResult:
        """Analyse without blocking the event loop."""
        return await asyncio.to_thread(
            self.analyse, image, role=role, detection=detection
        )

    def describe(self) -> dict[str, Any]:
        """Serialisable summary of the active configuration, for ``/health``."""
        return {
            "detectors": [detector.name for detector in self._detectors],
            "min_authenticity_score": self._config.min_authenticity_score,
            "require_face": self._config.require_face,
            "flag_multiple_faces": self._config.flag_multiple_faces,
            "claims": {
                "content_moderation": False,
                "deepfake_detection": False,
                "reverse_image_search": False,
            },
        }

    # -- Internals ----------------------------------------------------------- #

    def _safe_detect(self, image: BgrImage, role: ImageRole) -> Any | None:
        """Detect faces, treating a failure as "no subject information".

        A cartoon avatar or a landscape has no face, and the detector may also
        reject the image outright for being too small. Neither should lose the
        authenticity findings, which are the useful part for exactly those
        images.
        """
        try:
            return self._detection.detect(image, role=role)
        except HamqadamError as exc:
            log.info("profile.detection_unavailable", reason=str(exc))
            return None

    def _safe_quality(
        self, image: BgrImage, role: ImageRole, detection: Any
    ) -> Any | None:
        """Assess quality, treating a failure as "no quality information"."""
        try:
            return self._quality.assess(image, role=role, detection=detection)
        except HamqadamError as exc:
            log.info("profile.quality_unavailable", reason=str(exc))
            return None

    def _to_schema(
        self,
        *,
        assessment: AuthenticityAssessment,
        role: ImageRole,
        image: BgrImage,
        face_detected: bool,
        face_count: int,
        face_visibility: float,
        quality_score: float,
        quality_usable: bool,
        duration_ms: float,
    ) -> ProfileAnalysisResult:
        """Render everything as the API response model."""
        is_group = face_count > 1
        genuine = assessment.score >= self._config.min_authenticity_score

        usable = genuine
        if self._config.require_face:
            usable = usable and face_detected and quality_usable
        if self._config.flag_multiple_faces:
            usable = usable and not is_group

        warnings = self._warnings_for(
            assessment=assessment,
            face_detected=face_detected,
            is_group=is_group,
            quality_usable=quality_usable,
        )

        error_code, error_message = self._failure(
            genuine=genuine,
            assessment=assessment,
            face_detected=face_detected,
            is_group=is_group,
            quality_usable=quality_usable,
            usable=usable,
        )

        return ProfileAnalysisResult(
            usable_as_profile=usable,
            authenticity_score=assessment.score,
            is_probably_genuine_capture=genuine,
            findings=[
                AuthenticityFindingModel(**finding.as_dict())
                for finding in assessment.findings
            ],
            signals=[
                AuthenticitySignalModel(**signal.as_dict())
                for signal in assessment.signals
            ],
            face_detected=face_detected,
            face_count=face_count,
            is_group_photo=is_group,
            face_visibility_score=face_visibility,
            image_quality_score=quality_score,
            quality_usable=quality_usable,
            role=role,
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
            error_code=error_code,
            error_message=error_message,
            warnings=warnings,
            duration_ms=duration_ms,
        )

    def _warnings_for(
        self,
        *,
        assessment: AuthenticityAssessment,
        face_detected: bool,
        is_group: bool,
        quality_usable: bool,
    ) -> list[AnalysisWarning]:
        """Non-fatal findings for the Backend and the fraud engine."""
        warnings: list[AnalysisWarning] = []

        for finding in assessment.findings:
            warnings.append(
                AnalysisWarning(
                    code=finding.code,
                    message=finding.message,
                    stage="profile",
                    detail={"confidence": round(finding.confidence, 4)},
                )
            )

        if is_group:
            warnings.append(
                AnalysisWarning(
                    code="PROFILE_IMAGE_IS_GROUP_PHOTO",
                    message=(
                        "More than one person is in this photograph, so which "
                        "one the account belongs to is ambiguous. Not "
                        "dishonest - but it cannot be verified as it stands."
                    ),
                    stage="profile",
                    detail={},
                )
            )

        if not face_detected:
            warnings.append(
                AnalysisWarning(
                    code="PROFILE_IMAGE_HAS_NO_FACE",
                    message=(
                        "No face was found in this photograph. It cannot be "
                        "compared against the live selfie."
                    ),
                    stage="profile",
                    detail={},
                )
            )
        elif not quality_usable:
            warnings.append(
                AnalysisWarning(
                    code="PROFILE_IMAGE_QUALITY_LOW",
                    message=(
                        "The photograph is too degraded for reliable face "
                        "comparison."
                    ),
                    stage="profile",
                    detail={},
                )
            )

        for note in assessment.unmeasured:
            warnings.append(
                AnalysisWarning(
                    code="PROFILE_DETECTOR_UNAVAILABLE",
                    message=f"An authenticity check could not run: {note}",
                    stage="profile",
                    detail={},
                )
            )

        return warnings

    @staticmethod
    def _failure(
        *,
        genuine: bool,
        assessment: AuthenticityAssessment,
        face_detected: bool,
        is_group: bool,
        quality_usable: bool,
        usable: bool,
    ) -> tuple[ErrorCode | None, str | None]:
        """Which failure to report, and what to tell the user about it.

        Ordered by what the user should do first. The authenticity findings
        come before everything else because they are the only ones for which
        retaking the same photograph cannot possibly help - the user has to
        upload a *different* image.
        """
        if usable:
            return None, None

        if not genuine and assessment.strongest is not None:
            return ErrorCode.INVALID_IMAGE, assessment.strongest.message

        if not face_detected:
            return (
                ErrorCode.FACE_NOT_DETECTED,
                "No face could be found in this photograph. Upload a clear "
                "photo of yourself, facing the camera.",
            )

        if is_group:
            return (
                ErrorCode.MULTIPLE_FACES_DETECTED,
                "More than one person is in this photograph. Upload one where "
                "you are the only person, or crop it so only you are in frame.",
            )

        if not quality_usable:
            return (
                ErrorCode.LOW_IMAGE_QUALITY,
                "This photograph is too blurred or too poorly lit to compare "
                "reliably. Retake it in better light, holding the camera "
                "steady.",
            )

        return ErrorCode.INVALID_IMAGE, "This photograph cannot be used."


def build_profile_service(settings: Settings | None = None) -> ProfileAnalysisService:
    """Wire up a :class:`ProfileAnalysisService` from configuration.

    Args:
        settings: Service configuration. Loaded from the environment when
            omitted.

    Returns:
        A ready service, with Modules 1 and 2 constructed beneath it.
    """
    from hamqadam_ai.services import (
        build_face_detection_service,
        build_quality_service,
    )

    settings = settings or get_settings()
    service = ProfileAnalysisService(
        detectors=build_detectors(settings.profile),
        detection=build_face_detection_service(settings),
        quality=build_quality_service(settings),
        settings=settings,
    )

    log.info(
        "profile.service_ready",
        detectors=[detector.name for detector in service._detectors],  # noqa: SLF001
        min_authenticity_score=settings.profile.min_authenticity_score,
    )
    return service


__all__ = ["ProfileAnalysisService", "build_profile_service"]
