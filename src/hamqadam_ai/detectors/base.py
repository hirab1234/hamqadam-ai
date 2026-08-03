"""The face-detector port and its value objects.

Contract every adapter honours
------------------------------
* :meth:`FaceDetector.detect` takes a BGR ``uint8`` array and returns
  :class:`RawDetection` objects in **source-image pixel coordinates**. Every
  network-space to source-space mapping is the adapter's responsibility;
  no consumer ever has to know the network's input size.
* Results are sorted by descending confidence.
* Scores are already thresholded and NMS-suppressed by the adapter.
* An adapter that cannot produce landmarks returns ``None`` for them rather
  than fabricating points. Downstream code degrades explicitly on missing
  landmarks instead of silently computing pose from invented data.
* :meth:`detect` never raises for "no faces here" - that is an empty list, a
  perfectly normal result. It raises only on genuine inference failure.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.constants import FaceRegion
from hamqadam_ai.schemas.detection import (
    BoundingBoxModel,
    DetectedFaceModel,
    LandmarkModel,
)
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

BgrImage = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class RawDetection:
    """One face as reported by a detector, before any analysis.

    Attributes:
        box: Face bounds in source-image pixels.
        confidence: Detector objectness in ``[0, 1]``.
        landmarks: Five keypoints, or ``None`` for landmark-free detectors.
        landmark_confidence: Per-keypoint confidence when the detector supplies
            it. YOLO-pose does; SCRFD does not.
        landmarks_derived: True when the keypoints were *constructed* from a
            canonical template rather than predicted by the model. The Haar
            fallback does this from a detected eye pair. Derived points carry
            real information about roll and scale but none about pitch or yaw,
            so the pose estimator must not run a PnP solve on them - doing so
            would report a confident frontal pose for every face regardless of
            its actual orientation.
    """

    box: BoundingBox
    confidence: float
    landmarks: Landmarks5 | None = None
    landmark_confidence: tuple[float, ...] | None = None
    landmarks_derived: bool = False

    @property
    def has_landmarks(self) -> bool:
        """Whether usable keypoints are attached."""
        return self.landmarks is not None

    @property
    def has_predicted_landmarks(self) -> bool:
        """Whether the keypoints were predicted by a model, not constructed."""
        return self.landmarks is not None and not self.landmarks_derived

    def area_ratio(self, image_width: int, image_height: int) -> float:
        """Face area as a fraction of the image area."""
        total = float(image_width * image_height)
        return float(self.box.area / total) if total > 0 else 0.0


@dataclass(slots=True)
class DetectedFace:
    """A detection enriched with pose, occlusion and visibility analysis.

    Mutable by design: the detection service builds it incrementally as each
    analyser runs, then freezes it into the immutable
    :class:`~hamqadam_ai.schemas.detection.DetectedFaceModel` response schema.
    """

    raw: RawDetection
    image_width: int
    image_height: int

    pose: Any | None = None  # PoseResult; typed loosely to avoid a cycle
    occlusion: Any | None = None  # OcclusionResult
    visibility_score: float = 0.0
    visibility_components: dict[str, float] = field(default_factory=dict)
    visibility_weights: dict[str, float] = field(default_factory=dict)
    limiting_factor: str = "detector_confidence"

    is_primary: bool = False
    is_bystander: bool = False
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def box(self) -> BoundingBox:
        """The face bounds in source-image pixels."""
        return self.raw.box

    @property
    def confidence(self) -> float:
        """Detector objectness score."""
        return self.raw.confidence

    @property
    def landmarks(self) -> Landmarks5 | None:
        """Five facial keypoints, when available."""
        return self.raw.landmarks

    @property
    def area_ratio(self) -> float:
        """Face area as a fraction of the image area."""
        return self.raw.area_ratio(self.image_width, self.image_height)

    @property
    def truncation_ratio(self) -> float:
        """Fraction of the face box lying outside the frame."""
        return self.box.truncation_ratio(self.image_width, self.image_height)

    @property
    def accepted(self) -> bool:
        """Whether this face passed every policy check."""
        return not self.rejection_reasons

    def to_schema(self) -> DetectedFaceModel:
        """Convert to the API response model."""
        from hamqadam_ai.schemas.common import to_percent
        from hamqadam_ai.schemas.detection import (
            OcclusionReport,
            PoseEstimate,
            RegionOcclusion,
            VisibilityBreakdown,
        )

        landmark_models: list[LandmarkModel] = []
        if self.raw.landmarks is not None:
            landmark_models = [
                LandmarkModel(**entry) for entry in self.raw.landmarks.as_list()
            ]

        pose_model: PoseEstimate | None = None
        if self.pose is not None:
            pose_model = PoseEstimate(
                yaw=round(self.pose.yaw, 2),
                pitch=round(self.pose.pitch, 2),
                roll=round(self.pose.roll, 2),
                frontal=self.pose.frontal,
                within_hard_limits=self.pose.within_hard_limits,
                deviation_score=round(self.pose.deviation_score, 4),
                method=self.pose.method,
                reprojection_error=(
                    round(self.pose.reprojection_error, 3)
                    if self.pose.reprojection_error is not None
                    else None
                ),
            )

        occlusion_model: OcclusionReport | None = None
        if self.occlusion is not None:
            occlusion_model = OcclusionReport(
                occluded=self.occlusion.occluded,
                overall_score=round(self.occlusion.overall_score, 4),
                regions=[
                    RegionOcclusion(
                        region=FaceRegion(region),
                        occlusion_probability=round(evidence.probability, 4),
                        occluded=evidence.occluded,
                        flat_fraction=round(evidence.flat_fraction, 4),
                        texture_energy=round(evidence.texture_energy, 4),
                        skin_coverage=round(evidence.skin_coverage, 4),
                    )
                    for region, evidence in self.occlusion.regions.items()
                ],
                occluded_regions=[
                    FaceRegion(name) for name in self.occlusion.occluded_regions
                ],
                symmetry_delta=round(self.occlusion.symmetry_delta, 4),
                method=self.occlusion.method,
            )

        breakdown: VisibilityBreakdown | None = None
        if self.visibility_components:
            breakdown = VisibilityBreakdown(
                detector_confidence=round(
                    self.visibility_components.get("detector_confidence", 0.0), 4
                ),
                occlusion=round(self.visibility_components.get("occlusion", 0.0), 4),
                pose=round(self.visibility_components.get("pose", 0.0), 4),
                face_size=round(self.visibility_components.get("face_size", 0.0), 4),
                framing=round(self.visibility_components.get("framing", 0.0), 4),
                weights={k: round(v, 4) for k, v in self.visibility_weights.items()},
                limiting_factor=self.limiting_factor,
            )

        return DetectedFaceModel(
            bounding_box=BoundingBoxModel(**self.box.as_dict()),
            confidence=round(min(1.0, max(0.0, self.confidence)), 4),
            landmarks=landmark_models,
            pose=pose_model,
            occlusion=occlusion_model,
            face_visibility_score=to_percent(self.visibility_score),
            visibility_breakdown=breakdown,
            face_area_ratio=round(self.area_ratio, 6),
            truncation_ratio=round(self.truncation_ratio, 4),
            is_primary=self.is_primary,
            is_bystander=self.is_bystander,
            rejection_reasons=list(self.rejection_reasons),
        )


class FaceDetector(abc.ABC):
    """Abstract face detector.

    Args:
        name: Short adapter identifier used in configuration and in results.
        version: Version string of the underlying model artefact.
        score_threshold: Minimum objectness for a candidate to be returned.
        nms_iou_threshold: IoU above which overlapping boxes are suppressed.
        max_candidates: Hard cap on candidates considered before NMS. Guards
            against an adversarial image producing tens of thousands of weak
            activations and turning NMS into a denial of service.
    """

    #: Whether this adapter emits five-point landmarks.
    provides_landmarks: bool = False

    def __init__(
        self,
        *,
        name: str,
        version: str,
        score_threshold: float,
        nms_iou_threshold: float,
        max_candidates: int = 200,
    ) -> None:
        self.name = name
        self.version = version
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_candidates = max_candidates

    # -- Subclass contract ------------------------------------------------ #

    @abc.abstractmethod
    def _detect_impl(self, image: BgrImage) -> list[RawDetection]:
        """Run detection on a single BGR image. Called on a worker thread."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources held by the adapter."""

    # -- Public surface ---------------------------------------------------- #

    def detect(self, image: BgrImage) -> list[RawDetection]:
        """Detect faces in one image.

        Args:
            image: ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Detections in source-image coordinates, sorted by descending
            confidence. Empty when no face is present, which is a normal
            result and not an error.

        Raises:
            InferenceError: if the underlying model fails.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Detector expects a (H, W, 3) BGR array; got shape {image.shape}"
            )
        detections = self._detect_impl(image)
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[: self.max_candidates]

    async def detect_async(self, image: BgrImage) -> list[RawDetection]:
        """Detect faces without blocking the event loop.

        The default implementation hands off to the default thread pool.
        Adapters backed by :class:`~hamqadam_ai.models.base.LoadedModel`
        override this to use the registry's bounded inference pool instead.
        """
        return await asyncio.to_thread(self.detect, image)

    def detect_batch(self, images: Sequence[BgrImage]) -> list[list[RawDetection]]:
        """Detect faces in several images.

        The default implementation loops. Adapters whose model accepts a batch
        dimension override this to issue a single forward pass, which is
        materially faster for the small networks in the fallback chain.
        """
        return [self.detect(image) for image in images]

    async def detect_batch_async(
        self, images: Sequence[BgrImage]
    ) -> list[list[RawDetection]]:
        """Async counterpart of :meth:`detect_batch`."""
        return await asyncio.to_thread(self.detect_batch, images)

    def __enter__(self) -> FaceDetector:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, version={self.version!r})"


__all__ = ["BgrImage", "DetectedFace", "FaceDetector", "RawDetection"]
