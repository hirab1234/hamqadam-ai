"""MODULE 1 service - orchestrates detection, analysis and policy.

Pipeline for one image::

    normalise size  ->  detect  ->  per-face pose
                                 ->  per-face occlusion
                                 ->  per-face visibility
                                 ->  acceptance policy
                                 ->  FaceDetectionResult

Design notes
------------
**Analysis is bounded.** Pose and occlusion run only on faces that survived
admissibility, and occlusion - the expensive one, a warp plus four full-frame
colour conversions - runs only on the faces that could plausibly be selected as
the subject. On a group photo with eleven faces that is the difference between
one warp and eleven.

**Failures never propagate.** A pose solve or occlusion analysis that raises
degrades that one face's score and adds a warning; it does not fail the image.
The only genuinely fatal condition is every detector in the chain failing.

**Business outcomes are not exceptions.** "No face detected" is returned as a
populated :class:`~hamqadam_ai.schemas.detection.FaceDetectionResult` with
``passed=False``, not raised. The pipeline needs to analyse the other six
images and hand the Backend a complete picture. ``detect_or_raise`` exists for
the callers that genuinely want fail-fast.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import Settings, get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import (
    FaceNotDetectedError,
    HamqadamError,
    ImageTooSmallError,
    MultipleFacesDetectedError,
)
from hamqadam_ai.detectors.base import DetectedFace, RawDetection
from hamqadam_ai.detectors.factory import DetectorChain, build_detector_chain
from hamqadam_ai.detectors.occlusion import OcclusionAnalyzer, OcclusionResult
from hamqadam_ai.detectors.policy import DetectionPolicy
from hamqadam_ai.detectors.pose import PoseEstimator
from hamqadam_ai.detectors.visibility import VisibilityScorer
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.models.registry import ModelRegistry, get_registry
from hamqadam_ai.schemas.common import AnalysisWarning
from hamqadam_ai.schemas.detection import FaceDetectionResult
from hamqadam_ai.utils.image_ops import ensure_min_size, resize_long_side

log = get_logger(__name__)

#: Faces are ranked by area and only this many are analysed in full. The
#: subject of a verification photo is never the ninth-largest face, and
#: capping the count bounds worst-case latency on an adversarial crowd image.
_MAX_FACES_ANALYSED = 8


class FaceDetectionService:
    """Detects and analyses faces in a single image, then applies policy.

    Args:
        chain: The detector fallback chain.
        pose_estimator: Head-pose estimator.
        occlusion_analyzer: Occlusion analyser.
        visibility_scorer: Composite visibility scorer.
        policy: The acceptance policy.
        settings: Service configuration.
    """

    __slots__ = (
        "_chain",
        "_occlusion",
        "_policy",
        "_pose",
        "_settings",
        "_visibility",
    )

    def __init__(
        self,
        *,
        chain: DetectorChain,
        pose_estimator: PoseEstimator,
        occlusion_analyzer: OcclusionAnalyzer,
        visibility_scorer: VisibilityScorer,
        policy: DetectionPolicy,
        settings: Settings,
    ) -> None:
        self._chain = chain
        self._pose = pose_estimator
        self._occlusion = occlusion_analyzer
        self._visibility = visibility_scorer
        self._policy = policy
        self._settings = settings

    # -- Public surface ----------------------------------------------------- #

    def detect(
        self,
        image: npt.NDArray[np.uint8],
        *,
        role: ImageRole = ImageRole.PROFILE_IMAGE,
        analyse_all_faces: bool = False,
    ) -> FaceDetectionResult:
        """Detect and analyse faces in one image.

        Args:
            image: ``(H, W, 3)`` BGR uint8 array.
            role: Which image in the verification request this is. Recorded on
                the result and used by the caller to pick the right policy.
            analyse_all_faces: Run pose and occlusion on every admissible face
                rather than only the candidates for primary selection. Slower;
                used by the evaluation harness and the demo renderer.

        Returns:
            A populated result. A business rejection (no face, multiple faces,
            occluded) is reported through ``passed`` and ``error_code``, not
            raised.

        Raises:
            InferenceError: if every detector in the chain failed.
            ImageTooSmallError: if the image is below the configured floor.
        """
        started = time.perf_counter()
        detection_config = self._settings.detection

        source_height, source_width = image.shape[:2]
        if min(source_height, source_width) < detection_config.min_source_short_side // 2:
            raise ImageTooSmallError(
                f"Image is {source_width}x{source_height}, far below the usable minimum.",
                details={
                    "role": str(role),
                    "width": source_width,
                    "height": source_height,
                    "minimum_short_side": detection_config.min_source_short_side,
                },
            )

        working, scale, warnings = self._normalise(image, role)
        raw_detections, detector = self._chain.detect(working)

        # Everything downstream reports in *source* coordinates, so undo the
        # normalisation scale immediately rather than threading it through.
        detections = self._rescale(raw_detections, scale) if scale != 1.0 else raw_detections

        faces = self._analyse(
            image, detections, analyse_all_faces=analyse_all_faces
        )

        verdict = self._policy.apply(
            faces,
            image_width=source_width,
            image_height=source_height,
            detector_provides_landmarks=detector.provides_landmarks,
        )
        warnings.extend(verdict.warnings)

        if self._chain.using_fallback:
            warnings.append(
                AnalysisWarning(
                    code="FALLBACK_DETECTOR_ACTIVE",
                    message=(
                        f"The configured primary detector "
                        f"({self._chain.requested_primary}) is unavailable; "
                        f"{detector.name} was used instead. Accuracy is lower than "
                        f"the calibrated operating point."
                    ),
                    stage="detection",
                    detail={
                        "requested": self._chain.requested_primary,
                        "active": detector.name,
                    },
                )
            )

        primary = verdict.primary_face
        duration_ms = (time.perf_counter() - started) * 1000.0

        result = FaceDetectionResult(
            face_detected=verdict.qualifying_count > 0,
            face_count=verdict.qualifying_count,
            face_visibility_score=(
                round(primary.visibility_score * 100.0, 2) if primary else 0.0
            ),
            role=role,
            primary_face=primary.to_schema() if primary else None,
            faces=[face.to_schema() for face in faces],
            raw_detection_count=len(detections),
            detector=detector.name,
            detector_version=detector.version,
            used_fallback=self._chain.using_fallback,
            image_width=source_width,
            image_height=source_height,
            passed=verdict.passed,
            error_code=verdict.error_code,
            error_message=verdict.message,
            warnings=warnings,
            duration_ms=duration_ms,
        )

        log.info("detection.completed", **result.summary())
        return result

    async def detect_async(
        self,
        image: npt.NDArray[np.uint8],
        *,
        role: ImageRole = ImageRole.PROFILE_IMAGE,
        analyse_all_faces: bool = False,
    ) -> FaceDetectionResult:
        """Detect without blocking the event loop.

        The whole pipeline - not just the forward pass - runs on a worker
        thread. Occlusion analysis is a few milliseconds of pure OpenCV per
        face and would otherwise stall the loop under concurrency.
        """
        return await asyncio.to_thread(
            self.detect, image, role=role, analyse_all_faces=analyse_all_faces
        )

    async def detect_many_async(
        self,
        images: Sequence[tuple[npt.NDArray[np.uint8], ImageRole]],
    ) -> list[FaceDetectionResult | HamqadamError]:
        """Analyse several images concurrently, isolating failures.

        A verification request carries up to seven images. One unreadable
        secondary photo must not abort the other six, so a failure is returned
        in place rather than raised.

        Args:
            images: ``(image, role)`` pairs.

        Returns:
            A list positionally aligned with ``images``, each element either a
            result or the error that occurred.
        """
        from hamqadam_ai.core.retry import gather_resilient

        return await gather_resilient(
            [self.detect_async(image, role=role) for image, role in images],
            stage="face_detection",
        )

    def detect_or_raise(
        self,
        image: npt.NDArray[np.uint8],
        *,
        role: ImageRole = ImageRole.PROFILE_IMAGE,
    ) -> FaceDetectionResult:
        """Detect, raising a typed exception on a business rejection.

        For callers that genuinely want fail-fast semantics - the CNIC portrait
        extractor, for instance, where there is nothing useful to do with a
        result containing no face.

        Raises:
            FaceNotDetectedError: no qualifying face.
            MultipleFacesDetectedError: more than one qualifying face.
            HamqadamError: any other policy rejection, carrying its code.
        """
        result = self.detect(image, role=role)
        if result.passed:
            return result

        details: dict[str, Any] = {"role": str(role), "detector": result.detector}
        message = result.error_message or "Face detection failed."

        if result.error_code is ErrorCode.FACE_NOT_DETECTED:
            raise FaceNotDetectedError(message, details=details)
        if result.error_code is ErrorCode.MULTIPLE_FACES_DETECTED:
            details["face_count"] = result.face_count
            raise MultipleFacesDetectedError(message, details=details)
        raise HamqadamError(
            message, code=result.error_code or ErrorCode.FACE_NOT_DETECTED, details=details
        )

    def describe(self) -> dict[str, Any]:
        """Serialisable summary of the active configuration, for ``/health``."""
        return {
            "chain": self._chain.describe(),
            "thresholds": {
                "score": self._settings.detection.score_threshold,
                "nms_iou": self._settings.detection.nms_iou_threshold,
                "min_confidence": self._settings.detection.policy.min_confidence,
                "max_faces": self._settings.detection.policy.max_faces_allowed,
                "min_visibility": self._settings.detection.visibility.min_acceptable,
            },
        }

    def close(self) -> None:
        """Release the detector chain."""
        self._chain.close()

    # -- Internals ----------------------------------------------------------- #

    def _normalise(
        self, image: npt.NDArray[np.uint8], role: ImageRole
    ) -> tuple[npt.NDArray[np.uint8], float, list[AnalysisWarning]]:
        """Bring the image into the detector's preferred size range.

        Returns the working image, the cumulative scale factor applied, and any
        warnings raised. Coordinates in the working image are converted back to
        source coordinates by dividing by the returned scale.
        """
        config = self._settings.detection
        warnings: list[AnalysisWarning] = []

        working, downscale = resize_long_side(image, config.max_source_long_side)
        scale = downscale

        height, width = working.shape[:2]
        if min(height, width) < config.min_source_short_side:
            working, upscale = ensure_min_size(working, config.min_source_short_side)
            scale *= upscale
            warnings.append(
                AnalysisWarning(
                    code="IMAGE_UPSCALED",
                    message=(
                        "The image was below the detector's minimum working size and "
                        "was upscaled. Detection is less reliable and no detail was "
                        "recovered by the upscale."
                    ),
                    stage="detection",
                    detail={
                        "role": str(role),
                        "original_short_side": min(image.shape[:2]),
                        "minimum": config.min_source_short_side,
                    },
                )
            )

        return working, scale, warnings

    @staticmethod
    def _rescale(detections: list[RawDetection], scale: float) -> list[RawDetection]:
        """Map detections from working-image space back to source coordinates."""
        inverse = 1.0 / scale
        rescaled: list[RawDetection] = []
        for detection in detections:
            rescaled.append(
                RawDetection(
                    box=detection.box.scale(inverse),
                    confidence=detection.confidence,
                    landmarks=(
                        detection.landmarks.scale(inverse)
                        if detection.landmarks is not None
                        else None
                    ),
                    landmark_confidence=detection.landmark_confidence,
                    landmarks_derived=detection.landmarks_derived,
                )
            )
        return rescaled

    def _analyse(
        self,
        image: npt.NDArray[np.uint8],
        detections: list[RawDetection],
        *,
        analyse_all_faces: bool,
    ) -> list[DetectedFace]:
        """Enrich detections with pose, occlusion and visibility.

        Faces are analysed in descending area order and the analysis budget is
        spent on the largest few, because the subject of a verification photo
        is never a small background face. Faces beyond the budget still get a
        visibility score from the components that need no analysis, so they
        remain visible in the result and still count towards the
        single-person rule.
        """
        height, width = image.shape[:2]
        policy = self._settings.detection.policy

        faces = [
            DetectedFace(raw=detection, image_width=width, image_height=height)
            for detection in detections
        ]
        faces.sort(key=lambda face: face.box.area, reverse=True)

        budget = len(faces) if analyse_all_faces else _MAX_FACES_ANALYSED

        for index, face in enumerate(faces):
            deep = index < budget
            # Skip the expensive analysis for faces that cannot pass anyway.
            admissible = (
                face.confidence >= policy.min_confidence
                and face.box.short_side >= policy.min_face_pixels
            )

            pose_result = None
            occlusion_result: OcclusionResult | None = None

            if deep and admissible and face.landmarks is not None:
                pose_result = self._safe_pose(face, (width, height))
                occlusion_result = self._safe_occlusion(image, face)

            face.pose = pose_result
            face.occlusion = occlusion_result

            visibility = self._visibility.score(
                detector_confidence=face.confidence,
                occlusion_score=(
                    occlusion_result.overall_score
                    if occlusion_result is not None
                    and occlusion_result.method != "unavailable"
                    else None
                ),
                pose_deviation=(
                    pose_result.deviation_score if pose_result is not None else None
                ),
                face_area_ratio=face.area_ratio,
                truncation_ratio=face.truncation_ratio,
            )
            face.visibility_score = visibility.score
            face.visibility_components = visibility.components
            face.visibility_weights = visibility.weights
            face.limiting_factor = visibility.limiting_factor

        return faces

    def _safe_pose(self, face: DetectedFace, image_size: tuple[int, int]) -> Any | None:
        """Estimate pose, degrading to ``None`` rather than failing the image."""
        assert face.landmarks is not None  # noqa: S101 - guarded by the caller
        try:
            return self._pose.estimate(
                face.landmarks,
                image_size,
                box=face.box,
                landmarks_derived=face.raw.landmarks_derived,
            )
        except Exception as exc:  # noqa: BLE001 - analysis must not fail the request
            log.warning("detection.pose_failed", reason=str(exc))
            return None

    def _safe_occlusion(
        self, image: npt.NDArray[np.uint8], face: DetectedFace
    ) -> OcclusionResult | None:
        """Analyse occlusion, degrading to ``None`` rather than failing."""
        try:
            return self._occlusion.analyze(image, face.box, face.landmarks)
        except Exception as exc:  # noqa: BLE001 - analysis must not fail the request
            log.warning("detection.occlusion_failed", reason=str(exc))
            return None


def build_face_detection_service(
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
) -> FaceDetectionService:
    """Wire up a :class:`FaceDetectionService` from configuration.

    Args:
        settings: Service configuration. Loaded from the environment when
            omitted.
        registry: The model registry. The process-wide one is used when
            omitted.

    Returns:
        A ready service.

    Raises:
        ModelLoadError: if no detector at all could be constructed.
    """
    settings = settings or get_settings()
    registry = registry or get_registry(settings)

    chain = build_detector_chain(settings, registry)

    classifier = None
    occlusion_spec = settings.models.get("face_occlusion_classifier")
    if occlusion_spec is not None and occlusion_spec.enabled:
        classifier = registry.try_get("face_occlusion_classifier")

    return FaceDetectionService(
        chain=chain,
        pose_estimator=PoseEstimator(settings.detection.pose),
        occlusion_analyzer=OcclusionAnalyzer(settings.detection.occlusion, classifier),
        visibility_scorer=VisibilityScorer(settings.detection.visibility),
        policy=DetectionPolicy(settings.detection),
        settings=settings,
    )


__all__ = ["FaceDetectionService", "build_face_detection_service"]
