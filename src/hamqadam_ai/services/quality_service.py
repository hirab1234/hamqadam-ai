"""MODULE 2 service - orchestrates the eight quality analysers.

Pipeline for one image::

    build shared context  ->  8 analysers  ->  aggregate  ->  QualityResult
    (grayscale, canonical                     (power mean +
     face crop, FFT, ...)                      critical floor)

Design notes
------------
**The context is built once.** Grayscale conversion, the canonical face crop
and the radial power spectrum are each needed by three or four analysers.
Computing them once removes roughly 60% of the module's arithmetic and, more
importantly, guarantees every metric is measured on identical pixels - a metric
computed on a slightly different crop is not comparable with its own configured
anchors.

**Detection feeds quality, not the other way round.** The service accepts an
optional :class:`~hamqadam_ai.schemas.detection.FaceDetectionResult` and uses
its box, landmarks and yaw. Without one it still works, measuring globally and
reporting sharpness as unmeasured. That ordering matters for the full pipeline:
running detection first means quality is measured *on the face that will
actually be used*, not on whatever happens to be in frame.

**Analyser failure is contained.** Each runs through
:meth:`~hamqadam_ai.quality.base.QualityMetric.safe_analyse`, so a numerical
edge case in one dimension costs that dimension and nothing else.
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
from hamqadam_ai.core.exceptions import HamqadamError, QualityRejectedError
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.quality.aggregator import QualityAggregator, QualityAssessment
from hamqadam_ai.quality.artifacts import DistortionAnalyzer, PixelationAnalyzer
from hamqadam_ai.quality.base import MetricResult, QualityContext, QualityMetric
from hamqadam_ai.quality.blur import BlurAnalyzer, SharpnessAnalyzer
from hamqadam_ai.quality.exposure import BrightnessAnalyzer, ContrastAnalyzer
from hamqadam_ai.quality.noise import NoiseAnalyzer
from hamqadam_ai.quality.resolution import ResolutionAnalyzer
from hamqadam_ai.schemas.common import AnalysisWarning
from hamqadam_ai.schemas.detection import FaceDetectionResult
from hamqadam_ai.schemas.quality import MetricDetail, QualityResult
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

log = get_logger(__name__)


class QualityService:
    """Assesses image and face quality for one image.

    Args:
        analysers: The metric analysers to run, in report order.
        aggregator: Combines their results into a composite.
        settings: Service configuration.
    """

    __slots__ = ("_aggregator", "_analysers", "_settings")

    def __init__(
        self,
        *,
        analysers: Sequence[QualityMetric],
        aggregator: QualityAggregator,
        settings: Settings,
    ) -> None:
        self._analysers = list(analysers)
        self._aggregator = aggregator
        self._settings = settings

    # -- Public surface ----------------------------------------------------- #

    def assess(
        self,
        image: npt.NDArray[np.uint8],
        *,
        role: ImageRole = ImageRole.PROFILE_IMAGE,
        detection: FaceDetectionResult | None = None,
        face_box: BoundingBox | None = None,
        landmarks: Landmarks5 | None = None,
        yaw_degrees: float | None = None,
    ) -> QualityResult:
        """Assess one image.

        Args:
            image: ``(H, W, 3)`` BGR uint8 array.
            role: Which image in the verification request this is. Selects the
                acceptance threshold - a CNIC portrait is held to a far laxer
                bar than a live selfie, because it is a sub-300-dpi print
                photographed through a laminate.
            detection: The Module 1 result for this image. Supplies the face
                box, landmarks and yaw; strongly preferred over passing them
                individually.
            face_box: Face bounds, when no detection result is available.
            landmarks: Five keypoints, when no detection result is available.
            yaw_degrees: Head yaw, used to suppress the geometric anisotropy
                measurement on a turned head.

        Returns:
            A populated result. A quality rejection is reported through
            ``usable`` and ``error_code``, never raised.
        """
        started = time.perf_counter()
        config = self._settings.quality

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Quality assessment expects a (H, W, 3) BGR array; got {image.shape}"
            )

        box, marks, yaw = self._resolve_face(
            detection, face_box, landmarks, yaw_degrees
        )

        context = QualityContext(
            image=image,
            face_box=box,
            landmarks=marks,
            yaw_degrees=yaw,
            canonical_size=config.canonical_face_size,
            crop_margin=config.face_crop_margin,
            analysis_long_side=config.global_analysis_long_side,
        )

        results = [analyser.safe_analyse(context) for analyser in self._analysers]
        assessment = self._aggregator.aggregate(
            results,
            min_required=config.min_overall_for(str(role)),
            critical_components=config.critical_components_for(str(role)),
        )

        duration_ms = (time.perf_counter() - started) * 1000.0
        result = self._to_schema(
            assessment,
            role=role,
            context=context,
            duration_ms=duration_ms,
        )

        log.info("quality.completed", **result.summary())
        return result

    async def assess_async(
        self,
        image: npt.NDArray[np.uint8],
        *,
        role: ImageRole = ImageRole.PROFILE_IMAGE,
        detection: FaceDetectionResult | None = None,
        **kwargs: Any,
    ) -> QualityResult:
        """Assess without blocking the event loop.

        The whole assessment - not merely one operator - runs on a worker
        thread. It is pure OpenCV and NumPy, both of which release the GIL, so
        this gives real parallelism across concurrent requests.
        """
        return await asyncio.to_thread(
            self.assess, image, role=role, detection=detection, **kwargs
        )

    async def assess_many_async(
        self,
        images: Sequence[tuple[npt.NDArray[np.uint8], ImageRole]],
        detections: Sequence[FaceDetectionResult | None] | None = None,
    ) -> list[QualityResult | HamqadamError]:
        """Assess several images concurrently, isolating failures.

        Args:
            images: ``(image, role)`` pairs.
            detections: Optional per-image Module 1 results, positionally
                aligned with ``images``.

        Returns:
            A list positionally aligned with ``images``, each element either a
            result or the error that occurred.
        """
        from hamqadam_ai.core.retry import gather_resilient

        paired = list(detections) if detections is not None else [None] * len(images)
        if len(paired) != len(images):
            raise ValueError("`detections` must be positionally aligned with `images`")

        return await gather_resilient(
            [
                self.assess_async(image, role=role, detection=detection)
                for (image, role), detection in zip(images, paired, strict=True)
            ],
            stage="quality_assessment",
        )

    def assess_or_raise(
        self,
        image: npt.NDArray[np.uint8],
        *,
        role: ImageRole = ImageRole.PROFILE_IMAGE,
        detection: FaceDetectionResult | None = None,
    ) -> QualityResult:
        """Assess, raising when the image is unusable.

        For callers that genuinely want fail-fast semantics.

        Raises:
            QualityRejectedError: when the image does not clear its threshold.
        """
        result = self.assess(image, role=role, detection=detection)
        if not result.usable:
            raise QualityRejectedError(
                result.error_message or "Image quality is below the required minimum.",
                details={
                    "role": str(role),
                    "score": result.image_quality_score,
                    "minimum": result.min_required,
                    "limiting_factor": result.limiting_factor,
                    "critical_failures": result.critical_failures,
                },
            )
        return result

    def describe(self) -> dict[str, Any]:
        """Serialisable summary of the active configuration, for ``/health``."""
        config = self._settings.quality
        return {
            "analysers": [analyser.name for analyser in self._analysers],
            "canonical_face_size": config.canonical_face_size,
            "aggregation": {
                "power": config.aggregation.power,
                "weights": config.aggregation.weights,
                "critical_floor": config.aggregation.critical_floor,
                "critical_components": config.aggregation.critical_components,
            },
            "role_thresholds": {
                name: entry.min_overall for name, entry in config.roles.items()
            },
        }

    # -- Internals ----------------------------------------------------------- #

    @staticmethod
    def _resolve_face(
        detection: FaceDetectionResult | None,
        face_box: BoundingBox | None,
        landmarks: Landmarks5 | None,
        yaw_degrees: float | None,
    ) -> tuple[BoundingBox | None, Landmarks5 | None, float | None]:
        """Extract the face geometry, preferring an explicit override.

        Only the *primary* face is used. Quality is a property of the biometric
        that will actually be compared, not an average over everyone in frame.
        """
        if face_box is not None or landmarks is not None or yaw_degrees is not None:
            return face_box, landmarks, yaw_degrees

        if detection is None or detection.primary_face is None:
            return None, None, None

        primary = detection.primary_face
        box = BoundingBox(
            primary.bounding_box.x1,
            primary.bounding_box.y1,
            primary.bounding_box.x2,
            primary.bounding_box.y2,
        )

        marks: Landmarks5 | None = None
        if len(primary.landmarks) == 5:
            ordered = {landmark.name: landmark for landmark in primary.landmarks}
            names = ("left_eye", "right_eye", "nose_tip", "mouth_left", "mouth_right")
            if all(name in ordered for name in names):
                marks = Landmarks5(
                    np.array(
                        [[ordered[name].x, ordered[name].y] for name in names],
                        dtype=np.float32,
                    )
                )

        yaw = primary.pose.yaw if primary.pose is not None else None
        return box, marks, yaw

    def _to_schema(
        self,
        assessment: QualityAssessment,
        *,
        role: ImageRole,
        context: QualityContext,
        duration_ms: float,
    ) -> QualityResult:
        """Render the assessment as the API response model."""
        details = {
            name: MetricDetail(**result.as_dict())
            for name, result in assessment.metrics.items()
        }

        warnings = [
            AnalysisWarning(
                code="QUALITY_OBSERVATION",
                message=note,
                stage="quality",
            )
            for note in assessment.notes
        ]
        if assessment.unmeasured:
            warnings.append(
                AnalysisWarning(
                    code="QUALITY_DIMENSION_UNMEASURED",
                    message=(
                        "Some quality dimensions could not be assessed and were "
                        "excluded from the composite; the remaining weights were "
                        "renormalised around them."
                    ),
                    stage="quality",
                    detail={"dimensions": list(assessment.unmeasured)},
                )
            )

        error_code = None
        error_message = None
        if not assessment.usable:
            error_code = ErrorCode.LOW_IMAGE_QUALITY
            error_message = self._rejection_message(assessment)

        return QualityResult(
            image_quality_score=assessment.overall_score,
            blur_score=assessment.score_for("blur"),
            brightness_score=assessment.score_for("brightness"),
            noise_score=assessment.score_for("noise"),
            contrast_score=assessment.score_for("contrast"),
            resolution_score=assessment.score_for("resolution"),
            sharpness_score=assessment.score_for("sharpness"),
            distortion_score=assessment.score_for("distortion"),
            pixelation_score=assessment.score_for("pixelation"),
            role=role,
            usable=assessment.usable,
            min_required=assessment.min_required,
            limiting_factor=assessment.limiting_factor,
            critical_failures=list(assessment.critical_failures),
            unmeasured=list(assessment.unmeasured),
            metrics=details,
            effective_weights={
                key: round(value, 4)
                for key, value in assessment.effective_weights.items()
            },
            arithmetic_score=assessment.arithmetic_score,
            error_code=error_code,
            error_message=error_message,
            warnings=warnings,
            image_width=context.width,
            image_height=context.height,
            face_analysed=context.has_face,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _rejection_message(assessment: QualityAssessment) -> str:
        """Build user-facing guidance from the failing dimension.

        The advice is keyed off *what* failed, because "image quality too low"
        is unactionable and sends the user round a loop they cannot exit.
        """
        guidance = {
            "blur": "Hold the camera steady and make sure it has focused.",
            "sharpness": "Keep still while the photo is taken.",
            "brightness": "Move somewhere more evenly lit.",
            "contrast": "Avoid shooting against a bright background.",
            "noise": "Move somewhere brighter so the camera does not have to "
            "amplify the signal.",
            "resolution": "Move closer to the camera, and send the original "
            "photo rather than a resized copy.",
            "pixelation": "Send the original photo rather than a compressed or "
            "enlarged copy.",
            "distortion": "Send the original photo without filters or resizing.",
        }

        if assessment.critical_failures:
            failing = assessment.critical_failures[0]
            return (
                f"Image quality is too low to use: {failing} is far below the "
                f"minimum. {guidance.get(failing, '')}".strip()
            )

        limiting = assessment.limiting_factor
        advice = guidance.get(limiting or "", "")
        return (
            f"Image quality is {assessment.overall_score:.0f}, below the required "
            f"{assessment.min_required:.0f}"
            + (f", limited by {limiting}. {advice}" if limiting else ".")
        ).strip()


def build_quality_service(settings: Settings | None = None) -> QualityService:
    """Wire up a :class:`QualityService` from configuration.

    Args:
        settings: Service configuration. Loaded from the environment when
            omitted.

    Returns:
        A ready service. Needs no model weights, so it cannot fail to start.
    """
    settings = settings or get_settings()
    config = settings.quality

    analysers: list[QualityMetric] = [
        BlurAnalyzer(config.blur),
        SharpnessAnalyzer(config.sharpness),
        BrightnessAnalyzer(config.brightness),
        ContrastAnalyzer(config.contrast),
        NoiseAnalyzer(config.noise),
        ResolutionAnalyzer(config.resolution),
        PixelationAnalyzer(config.pixelation),
        DistortionAnalyzer(config.distortion),
    ]

    log.info(
        "quality.service_ready",
        analysers=[analyser.name for analyser in analysers],
        canonical_face_size=config.canonical_face_size,
        power=config.aggregation.power,
    )

    return QualityService(
        analysers=analysers,
        aggregator=QualityAggregator(config.aggregation),
        settings=settings,
    )


__all__ = ["MetricResult", "QualityService", "build_quality_service"]
