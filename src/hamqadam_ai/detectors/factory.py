"""Detector construction and the fallback chain.

The chain is the reason this service degrades instead of failing. Adapters are
built in the order given by ``detection.fallback_chain``; each is attempted in
turn and the first one that constructs successfully becomes active. An adapter
whose weights are absent, whose digest fails, or whose runtime cannot
initialise is skipped with a log line, not an exception.

At runtime the chain also handles *inference* failure: if the active detector
raises mid-request, the next adapter is tried for that image. A single
corrupted forward pass therefore costs a few hundred milliseconds rather than
the whole verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hamqadam_ai.core.config import Settings
from hamqadam_ai.core.exceptions import (
    HamqadamError,
    InferenceError,
    ModelLoadError,
)
from hamqadam_ai.detectors.base import BgrImage, FaceDetector, RawDetection
from hamqadam_ai.detectors.cascades import cascade_path as resolve_cascade
from hamqadam_ai.detectors.haar import HaarCascadeDetector
from hamqadam_ai.detectors.opencv_dnn import OpenCvDnnDetector
from hamqadam_ai.detectors.scrfd import ScrfdDetector
from hamqadam_ai.detectors.yolo_face import YoloFaceDetector
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.models.registry import ModelRegistry

log = get_logger(__name__)

#: Chain name -> the registry key holding that adapter's weights.
DETECTOR_MODEL_KEYS: dict[str, str] = {
    "scrfd": "face_detector_scrfd",
    "yolo": "face_detector_yolo",
    "opencv_dnn": "face_detector_opencv_dnn",
    "haar": "face_detector_haar",
}


def create_detector(
    name: str, settings: Settings, registry: ModelRegistry
) -> FaceDetector | None:
    """Build one detector adapter, or return ``None`` if it is unavailable.

    Args:
        name: Chain name (``scrfd``, ``yolo``, ``opencv_dnn``, ``haar``).
        settings: Service configuration.
        registry: The model registry, used to obtain ONNX sessions.

    Returns:
        The constructed adapter, or ``None`` when its weights are missing, it
        is disabled, or construction failed. Never raises for an unavailable
        model - that is the whole point of the chain.

    Raises:
        ValueError: for an unknown chain name, which is a configuration bug
            rather than a runtime condition.
    """
    if name not in DETECTOR_MODEL_KEYS:
        raise ValueError(
            f"Unknown detector {name!r}. Supported: {sorted(DETECTOR_MODEL_KEYS)}"
        )

    detection = settings.detection
    model_key = DETECTOR_MODEL_KEYS[name]
    spec = settings.models.get(model_key)
    if spec is None or not spec.enabled:
        log.info("detector.disabled", detector=name, model=model_key)
        return None

    try:
        if name == "scrfd":
            model = registry.try_get(model_key)
            if model is None:
                return None
            return ScrfdDetector(
                model,
                spec,
                score_threshold=detection.score_threshold,
                nms_iou_threshold=detection.nms_iou_threshold,
                input_size=detection.detection_input_size,
                max_candidates=detection.max_candidates,
            )

        if name == "yolo":
            model = registry.try_get(model_key)
            if model is None:
                return None
            return YoloFaceDetector(
                model,
                spec,
                score_threshold=detection.score_threshold,
                nms_iou_threshold=detection.nms_iou_threshold,
                input_size=detection.detection_input_size,
                max_candidates=detection.max_candidates,
            )

        if name == "opencv_dnn":
            weights = registry.resolve_path(model_key)
            if spec.config_path is None:
                log.warning("detector.missing_config_path", detector=name)
                return None
            config_path = registry.store_root / spec.config_path
            if not weights.is_file() or not config_path.is_file():
                log.info(
                    "detector.weights_absent",
                    detector=name,
                    weights_present=weights.is_file(),
                    config_present=config_path.is_file(),
                )
                return None
            return OpenCvDnnDetector(
                weights,
                config_path,
                spec,
                score_threshold=detection.score_threshold,
                nms_iou_threshold=detection.nms_iou_threshold,
                prefer_cuda=registry.device_plan.is_gpu,
                max_candidates=detection.max_candidates,
            )

        # haar - vendored inside the package, needs no model store at all.
        return HaarCascadeDetector(
            cascade_path=resolve_cascade(spec.path),
            eye_cascade_path=(
                resolve_cascade(spec.eye_cascade_path) if spec.eye_cascade_path else None
            ),
            version=spec.version,
            # Haar's pseudo-confidence is not on the same scale as a neural
            # detector's objectness, so it gets its own, lower floor. The
            # policy layer applies the real confidence gate afterwards.
            score_threshold=min(detection.score_threshold, 0.30),
            nms_iou_threshold=detection.nms_iou_threshold,
            max_candidates=detection.max_candidates,
        )

    except (ModelLoadError, HamqadamError, ValueError, OSError) as exc:
        log.warning(
            "detector.construction_failed",
            detector=name,
            model=model_key,
            reason=str(exc),
        )
        return None


@dataclass(slots=True)
class DetectorChain:
    """An ordered set of detectors with runtime failover.

    Attributes:
        detectors: Available adapters in preference order. Never empty in a
            successfully-constructed chain.
        requested_primary: The detector the configuration asked for first.
        unavailable: Chain names that could not be constructed, for diagnostics.
    """

    detectors: list[FaceDetector]
    requested_primary: str
    unavailable: list[str] = field(default_factory=list)

    @property
    def primary(self) -> FaceDetector:
        """The detector that will be tried first."""
        return self.detectors[0]

    @property
    def using_fallback(self) -> bool:
        """Whether the configured primary detector is unavailable."""
        return self.detectors[0].name != self.requested_primary

    def detect(self, image: BgrImage) -> tuple[list[RawDetection], FaceDetector]:
        """Detect faces, failing over to the next adapter on inference error.

        Args:
            image: ``(H, W, 3)`` BGR uint8 array.

        Returns:
            ``(detections, detector)`` - the detections and which adapter
            actually produced them.

        Raises:
            InferenceError: only if *every* adapter in the chain failed.
        """
        errors: list[str] = []
        for detector in self.detectors:
            try:
                return detector.detect(image), detector
            except (InferenceError, ValueError) as exc:
                errors.append(f"{detector.name}: {exc}")
                log.warning(
                    "detector.inference_failed",
                    detector=detector.name,
                    reason=str(exc),
                    action="trying_next_in_chain",
                )
        raise InferenceError(
            "Every detector in the fallback chain failed on this image.",
            details={"attempts": errors, "chain": [d.name for d in self.detectors]},
        )

    async def detect_async(
        self, image: BgrImage
    ) -> tuple[list[RawDetection], FaceDetector]:
        """Async counterpart of :meth:`detect`."""
        errors: list[str] = []
        for detector in self.detectors:
            try:
                return await detector.detect_async(image), detector
            except (InferenceError, ValueError) as exc:
                errors.append(f"{detector.name}: {exc}")
                log.warning(
                    "detector.inference_failed",
                    detector=detector.name,
                    reason=str(exc),
                    action="trying_next_in_chain",
                )
        raise InferenceError(
            "Every detector in the fallback chain failed on this image.",
            details={"attempts": errors, "chain": [d.name for d in self.detectors]},
        )

    def describe(self) -> dict[str, Any]:
        """Serialisable summary for health checks and start-up logging."""
        return {
            "active": self.detectors[0].name,
            "active_version": self.detectors[0].version,
            "chain": [
                {"name": d.name, "version": d.version, "landmarks": d.provides_landmarks}
                for d in self.detectors
            ],
            "requested_primary": self.requested_primary,
            "using_fallback": self.using_fallback,
            "unavailable": list(self.unavailable),
        }

    def close(self) -> None:
        """Close every adapter in the chain."""
        for detector in self.detectors:
            try:
                detector.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                log.warning("detector.close_failed", detector=detector.name, reason=str(exc))


def build_detector_chain(
    settings: Settings, registry: ModelRegistry
) -> DetectorChain:
    """Construct the detector fallback chain from configuration.

    Args:
        settings: Service configuration.
        registry: The model registry.

    Returns:
        A chain with at least one working detector.

    Raises:
        ModelLoadError: if no detector at all could be constructed. This is a
            genuine fatal condition - the Haar fallback ships inside the OpenCV
            wheel, so reaching it means the installation itself is broken.
    """
    requested = settings.detection.primary
    order = list(settings.detection.fallback_chain)[: settings.detection.max_fallback_depth]

    detectors: list[FaceDetector] = []
    unavailable: list[str] = []

    for name in order:
        detector = create_detector(name, settings, registry)
        if detector is None:
            unavailable.append(name)
            continue
        detectors.append(detector)

    if not detectors:
        raise ModelLoadError(
            "No face detector could be constructed. Even the bundled Haar "
            "cascade failed, which indicates a broken OpenCV installation.",
            details={"attempted": order, "unavailable": unavailable},
        )

    chain = DetectorChain(
        detectors=detectors, requested_primary=requested, unavailable=unavailable
    )

    if chain.using_fallback:
        log.warning(
            "detector.primary_unavailable",
            requested=requested,
            active=chain.primary.name,
            unavailable=unavailable,
            note="verification accuracy will be lower than specified",
        )
    else:
        log.info(
            "detector.chain_ready",
            active=chain.primary.name,
            depth=len(detectors),
            unavailable=unavailable,
        )

    return chain


__all__ = [
    "DETECTOR_MODEL_KEYS",
    "DetectorChain",
    "build_detector_chain",
    "create_detector",
]
