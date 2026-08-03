"""MODULE 1 response contract - face detection.

Covers every output the specification requires from the detection stage:
``face_detected``, ``face_count``, ``face_visibility_score``, ``bounding_box``,
``pose`` and ``confidence`` - plus the occlusion and framing evidence that
makes a rejection explainable rather than merely final.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, computed_field

from hamqadam_ai.core.constants import FaceRegion, ImageRole
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.schemas.common import (
    AnalysisWarning,
    OutputModel,
    PercentScore,
    UnitScore,
    to_percent,
)


class BoundingBoxModel(OutputModel):
    """An axis-aligned face box in source-image pixel coordinates."""

    x1: float = Field(description="Left edge in source pixels.")
    y1: float = Field(description="Top edge in source pixels.")
    x2: float = Field(description="Right edge in source pixels, exclusive.")
    y2: float = Field(description="Bottom edge in source pixels, exclusive.")
    width: float = Field(description="Box width in pixels.")
    height: float = Field(description="Box height in pixels.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return round(self.width * self.height, 2)


class LandmarkModel(OutputModel):
    """One of the five canonical facial keypoints."""

    name: str = Field(
        description=(
            "One of: left_eye, right_eye, nose_tip, mouth_left, mouth_right. "
            "Left and right are from the viewer's perspective."
        )
    )
    x: float = Field(description="Horizontal position in source pixels.")
    y: float = Field(description="Vertical position in source pixels.")


class PoseEstimate(OutputModel):
    """Head orientation in degrees, and whether it is within tolerance.

    Angles follow the aviation convention applied to a head facing the camera:

    * ``yaw`` - rotation about the vertical axis. Positive means the subject
      turned towards the viewer's right.
    * ``pitch`` - rotation about the horizontal axis. Positive means the chin
      is raised.
    * ``roll`` - in-plane tilt. Positive means the head is tilted so the
      viewer-right eye moves down.
    """

    yaw: float = Field(description="Left/right rotation in degrees.")
    pitch: float = Field(description="Up/down rotation in degrees.")
    roll: float = Field(description="In-plane tilt in degrees.")

    frontal: bool = Field(
        description="True when every axis is inside the configured soft limit."
    )
    within_hard_limits: bool = Field(
        description=(
            "True when every axis is inside the hard limit. False means the "
            "face is unusable for recognition, not merely imperfect."
        )
    )
    deviation_score: UnitScore = Field(
        description=(
            "Normalised distance from frontal, 0 = perfectly frontal, "
            "1 = at or beyond the hard limit on some axis."
        )
    )
    method: str = Field(
        description="How the pose was derived: 'pnp' or 'landmark_geometry'."
    )
    reprojection_error: float | None = Field(
        default=None,
        description=(
            "Mean landmark reprojection error in pixels for the PnP solve. "
            "A large value means the solve did not fit and the estimate is "
            "less trustworthy."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dominant_axis(self) -> str:
        """The axis furthest from frontal, for user-facing guidance."""
        magnitudes = {"yaw": abs(self.yaw), "pitch": abs(self.pitch), "roll": abs(self.roll)}
        return max(magnitudes, key=lambda key: magnitudes[key])


class RegionOcclusion(OutputModel):
    """Occlusion evidence for one anatomical region."""

    region: FaceRegion = Field(description="Which part of the face this describes.")
    occlusion_probability: UnitScore = Field(
        description="Fused probability that this region is obstructed."
    )
    occluded: bool = Field(
        description="Whether the probability exceeds the configured threshold."
    )
    flat_fraction: UnitScore = Field(
        default=0.0,
        description=(
            "Share of the region whose local gradient falls below half the "
            "face-wide median. The dominant occlusion signal: on a real "
            "portrait a bare eye measures around 0.06 and a covered one 1.00, "
            "and unlike chrominance it detects a skin-coloured occluder."
        ),
    )
    texture_energy: float = Field(
        description=(
            "Mean gradient energy of the region relative to the face median. "
            "Reported for diagnosis only. It is deliberately not the driver of "
            "the occlusion verdict, because a hard-edged occluder raises it "
            "through its own boundary gradients."
        )
    )
    skin_coverage: UnitScore = Field(
        description="Fraction of the region classified as skin in YCrCb space."
    )


class OcclusionReport(OutputModel):
    """Aggregate occlusion analysis for one face."""

    occluded: bool = Field(
        description="True when overall occlusion exceeds the configured threshold."
    )
    overall_score: UnitScore = Field(
        description="Weighted occlusion probability across all regions, 0 = clear."
    )
    regions: list[RegionOcclusion] = Field(
        default_factory=list, description="Per-region evidence."
    )
    occluded_regions: list[FaceRegion] = Field(
        default_factory=list, description="Regions flagged as obstructed."
    )
    symmetry_delta: UnitScore = Field(
        default=0.0,
        description=(
            "Normalised left/right texture asymmetry. A high value with a low "
            "overall score is the signature of a hand or object covering one "
            "side of the face."
        ),
    )
    method: str = Field(
        default="geometric",
        description="'classifier' when a dedicated ONNX model was used, else 'geometric'.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def visible_fraction(self) -> float:
        """Complement of the overall occlusion score, on the 0-100 scale."""
        return to_percent(1.0 - self.overall_score)


class VisibilityBreakdown(OutputModel):
    """Components of the composite face-visibility score.

    Exposed in full because ``face_visibility_score`` alone cannot tell a user
    whether to move closer, remove their sunglasses or face the camera.
    """

    detector_confidence: UnitScore = Field(
        description="Raw detector objectness for this face."
    )
    occlusion: UnitScore = Field(description="1 - overall occlusion score.")
    pose: UnitScore = Field(description="1 - pose deviation score.")
    face_size: UnitScore = Field(
        description="How well the face fills the frame, relative to the target ratio."
    )
    framing: UnitScore = Field(
        description="1 - truncation ratio; penalises a face cut off by the frame edge."
    )
    weights: dict[str, float] = Field(
        default_factory=dict, description="Weight applied to each component."
    )
    limiting_factor: str = Field(
        description="Component contributing the largest deficit to the composite."
    )


class DetectedFaceModel(OutputModel):
    """A single detected face and everything known about it."""

    bounding_box: BoundingBoxModel = Field(description="Face box in source pixels.")
    confidence: UnitScore = Field(description="Detector objectness score.")
    landmarks: list[LandmarkModel] = Field(
        default_factory=list,
        description=(
            "Five facial keypoints. Empty when the active detector does not "
            "produce them - the OpenCV DNN and Haar fallbacks do not."
        ),
    )
    pose: PoseEstimate | None = Field(
        default=None, description="Head orientation. Null without landmarks."
    )
    occlusion: OcclusionReport | None = Field(
        default=None, description="Occlusion analysis. Null without landmarks."
    )

    face_visibility_score: PercentScore = Field(
        description="Composite visibility on the 0-100 scale."
    )
    visibility_breakdown: VisibilityBreakdown | None = Field(
        default=None, description="Components of the visibility score."
    )

    face_area_ratio: float = Field(
        description="Face area divided by image area, in [0, 1]."
    )
    truncation_ratio: float = Field(
        description="Fraction of the face box falling outside the frame."
    )
    is_primary: bool = Field(
        default=False,
        description=(
            "True for the face selected as the subject. Exactly one face in a "
            "successful result carries this flag."
        ),
    )
    is_bystander: bool = Field(
        default=False,
        description=(
            "True for a face small enough relative to the subject to be treated "
            "as background rather than a second person."
        ),
    )
    rejection_reasons: list[str] = Field(
        default_factory=list,
        description="Policy checks this face failed, empty when it passed.",
    )


class FaceDetectionResult(OutputModel):
    """MODULE 1 output for one image."""

    # -- Specification-mandated fields ---------------------------------- #
    face_detected: bool = Field(
        description="True when at least one face passed the acceptance policy."
    )
    face_count: int = Field(
        ge=0,
        description=(
            "Number of qualifying faces, after the bystander filter. This is "
            "the count the single-person rule is applied to."
        ),
    )
    face_visibility_score: PercentScore = Field(
        description="Visibility of the primary face, 0-100. Zero when none was found."
    )

    # -- Supporting detail ---------------------------------------------- #
    role: ImageRole = Field(description="Which image in the request this describes.")
    primary_face: DetectedFaceModel | None = Field(
        default=None, description="The face selected as the subject."
    )
    faces: list[DetectedFaceModel] = Field(
        default_factory=list,
        description="Every face considered, including rejected ones and bystanders.",
    )
    raw_detection_count: int = Field(
        ge=0,
        description=(
            "Candidates the detector emitted before the acceptance policy ran. "
            "A large gap from face_count means the policy did the work."
        ),
    )

    detector: str = Field(description="Which adapter produced this result.")
    detector_version: str = Field(description="Pinned version of that adapter's model.")
    used_fallback: bool = Field(
        default=False,
        description="True when the configured primary detector was unavailable.",
    )

    image_width: int = Field(gt=0, description="Source image width in pixels.")
    image_height: int = Field(gt=0, description="Source image height in pixels.")

    passed: bool = Field(
        description=(
            "True when this image satisfies the full detection policy and may "
            "proceed to the next pipeline stage."
        )
    )
    error_code: ErrorCode | None = Field(
        default=None,
        description="Why the image failed. Null when `passed` is true.",
    )
    error_message: str | None = Field(
        default=None, description="Human-readable explanation of the failure."
    )
    warnings: list[AnalysisWarning] = Field(
        default_factory=list, description="Non-fatal observations."
    )
    duration_ms: float = Field(
        default=0.0, description="Time spent detecting in this image, in milliseconds."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def multiple_faces(self) -> bool:
        """Whether more than one qualifying face is present."""
        return self.face_count > 1

    def summary(self) -> dict[str, Any]:
        """Compact, PII-free summary for logging."""
        return {
            "role": str(self.role),
            "detected": self.face_detected,
            "count": self.face_count,
            "visibility": self.face_visibility_score,
            "detector": self.detector,
            "fallback": self.used_fallback,
            "passed": self.passed,
            "error": str(self.error_code) if self.error_code else None,
            "duration_ms": round(self.duration_ms, 1),
        }


__all__ = [
    "BoundingBoxModel",
    "DetectedFaceModel",
    "FaceDetectionResult",
    "LandmarkModel",
    "OcclusionReport",
    "PoseEstimate",
    "RegionOcclusion",
    "VisibilityBreakdown",
]
