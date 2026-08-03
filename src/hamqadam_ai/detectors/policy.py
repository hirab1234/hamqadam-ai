"""The detection acceptance policy.

Separated from the detectors deliberately. A detector answers "where are the
faces"; the policy answers "is this image acceptable for identity
verification", and that second question is pure business logic driven entirely
by ``configs/thresholds.yaml``. Keeping them apart means the rules can be
retuned, unit-tested against synthetic detections and reasoned about without
any model weights present.

Rule order
----------
Checks run cheapest-and-most-actionable first, and the **first** failure
determines the error code. That ordering is a product decision, not an
implementation detail: telling a user "no face detected" when the real problem
is that they are wearing sunglasses sends them round a loop they cannot exit.

1. Any detections at all?
2. Per-face admissibility: confidence, absolute size, relative size, framing.
3. Bystander filtering, then the single-person rule.
4. Primary-face selection.
5. Primary-face quality gates: pose, occlusion, composite visibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hamqadam_ai.core.config import DetectionConfig
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.detectors.base import DetectedFace
from hamqadam_ai.schemas.common import AnalysisWarning

#: How strongly centrality contributes to primary-face selection relative to
#: area. Area dominates - the subject of a selfie is nearly always the largest
#: face - but centrality breaks ties in group shots where two faces are
#: similarly sized and only one is the person actually being verified.
_CENTRALITY_WEIGHT = 0.25
_CONFIDENCE_WEIGHT = 0.15


@dataclass(slots=True)
class PolicyVerdict:
    """Outcome of applying the acceptance policy to one image.

    Attributes:
        passed: Whether the image may proceed to the next pipeline stage.
        error_code: Why it failed. ``None`` when ``passed`` is true.
        message: Human-readable explanation of the failure.
        primary_face: The face selected as the subject, when one was.
        qualifying_count: Faces that survived admissibility and the bystander
            filter. This is the number reported as ``face_count``.
        warnings: Non-fatal observations raised while evaluating.
        detail: Structured, PII-free context for the error envelope.
    """

    passed: bool
    error_code: ErrorCode | None = None
    message: str | None = None
    primary_face: DetectedFace | None = None
    qualifying_count: int = 0
    warnings: list[AnalysisWarning] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


class DetectionPolicy:
    """Applies the configured acceptance rules to a set of detections.

    Args:
        config: The full detection configuration.
    """

    __slots__ = ("_config",)

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config

    def apply(
        self,
        faces: list[DetectedFace],
        *,
        image_width: int,
        image_height: int,
        detector_provides_landmarks: bool = True,
    ) -> PolicyVerdict:
        """Evaluate the detections against the policy.

        Args:
            faces: Every detection, already enriched with pose, occlusion and
                visibility where those were computable.
            image_width: Source image width.
            image_height: Source image height.
            detector_provides_landmarks: Whether the active detector emits
                landmarks. Used to raise an informative warning rather than
                silently skipping the pose and occlusion gates.

        Returns:
            The verdict, with rejection reasons annotated onto each face.
        """
        policy = self._config.policy
        warnings: list[AnalysisWarning] = []

        if not faces:
            return PolicyVerdict(
                passed=False,
                error_code=ErrorCode.FACE_NOT_DETECTED,
                message="No face was found in this image.",
                qualifying_count=0,
                detail={"raw_detections": 0},
            )

        # ---- Step 2: per-face admissibility ----------------------------- #
        admissible: list[DetectedFace] = []
        for face in faces:
            reasons = self._admissibility_reasons(face)
            face.rejection_reasons = reasons
            if not reasons:
                admissible.append(face)

        if not admissible:
            return PolicyVerdict(
                passed=False,
                error_code=self._best_error_for(faces),
                message=self._best_message_for(faces),
                qualifying_count=0,
                detail={
                    "raw_detections": len(faces),
                    "rejection_reasons": sorted(
                        {reason for face in faces for reason in face.rejection_reasons}
                    ),
                    "best_confidence": round(
                        max(face.confidence for face in faces), 4
                    ),
                },
            )

        # ---- Step 3: bystanders, then the single-person rule ------------- #
        largest_area = max(face.box.area for face in admissible)
        qualifying: list[DetectedFace] = []
        for face in admissible:
            relative = face.box.area / largest_area if largest_area > 0 else 0.0
            face.is_bystander = relative < policy.bystander_area_ratio
            if not face.is_bystander:
                qualifying.append(face)

        bystander_count = len(admissible) - len(qualifying)
        if bystander_count:
            warnings.append(
                AnalysisWarning(
                    code="BACKGROUND_FACES_IGNORED",
                    message=(
                        f"{bystander_count} small background face(s) were ignored "
                        f"as bystanders."
                    ),
                    stage="detection",
                    detail={"count": bystander_count},
                )
            )

        if len(qualifying) > policy.max_faces_allowed:
            if policy.on_multiple_faces == "reject":
                return PolicyVerdict(
                    passed=False,
                    error_code=ErrorCode.MULTIPLE_FACES_DETECTED,
                    message=(
                        f"{len(qualifying)} people are present; this image must "
                        f"contain exactly one."
                    ),
                    qualifying_count=len(qualifying),
                    warnings=warnings,
                    detail={
                        "face_count": len(qualifying),
                        "max_allowed": policy.max_faces_allowed,
                        "bystanders_ignored": bystander_count,
                    },
                )
            warnings.append(
                AnalysisWarning(
                    code="MULTIPLE_FACES_PRESENT",
                    message=(
                        f"{len(qualifying)} faces qualified; the dominant one was "
                        f"selected as the subject."
                    ),
                    stage="detection",
                    detail={"face_count": len(qualifying)},
                )
            )

        # ---- Step 4: primary-face selection ------------------------------ #
        primary = self._select_primary(qualifying, image_width, image_height)
        primary.is_primary = True

        if not detector_provides_landmarks or primary.landmarks is None:
            warnings.append(
                AnalysisWarning(
                    code="LANDMARKS_UNAVAILABLE",
                    message=(
                        "The active detector produced no facial landmarks, so head "
                        "pose and occlusion could not be assessed. Visibility was "
                        "scored on the remaining signals."
                    ),
                    stage="detection",
                    detail={"detector_provides_landmarks": detector_provides_landmarks},
                )
            )

        # ---- Step 5: primary-face quality gates -------------------------- #
        gate = self._quality_gate(primary)
        if gate is not None:
            code, message, detail = gate
            return PolicyVerdict(
                passed=False,
                error_code=code,
                message=message,
                primary_face=primary,
                qualifying_count=len(qualifying),
                warnings=warnings,
                detail=detail,
            )

        warnings.extend(self._soft_warnings(primary))

        return PolicyVerdict(
            passed=True,
            primary_face=primary,
            qualifying_count=len(qualifying),
            warnings=warnings,
            detail={
                "raw_detections": len(faces),
                "admissible": len(admissible),
                "bystanders_ignored": bystander_count,
            },
        )

    # -- Step 2 helpers ------------------------------------------------------ #

    def _admissibility_reasons(self, face: DetectedFace) -> list[str]:
        """Return every admissibility rule this face violates."""
        policy = self._config.policy
        reasons: list[str] = []

        if face.confidence < policy.min_confidence:
            reasons.append("low_confidence")
        if face.box.short_side < policy.min_face_pixels:
            reasons.append("too_few_pixels")
        if face.area_ratio < policy.min_face_area_ratio:
            reasons.append("face_too_small")
        if face.area_ratio > policy.max_face_area_ratio:
            reasons.append("face_too_large")
        if face.truncation_ratio > policy.max_truncation_ratio:
            reasons.append("truncated_by_frame")

        return reasons

    @staticmethod
    def _best_error_for(faces: list[DetectedFace]) -> ErrorCode:
        """Choose the most actionable error when every face was inadmissible.

        The face that came closest to passing determines the message, because
        that is the one the user has a realistic chance of fixing.
        """
        best = max(faces, key=lambda face: face.confidence)
        reasons = set(best.rejection_reasons)

        if "truncated_by_frame" in reasons:
            return ErrorCode.FACE_TRUNCATED
        if "too_few_pixels" in reasons or "face_too_small" in reasons:
            return ErrorCode.FACE_TOO_SMALL
        return ErrorCode.FACE_NOT_DETECTED

    @staticmethod
    def _best_message_for(faces: list[DetectedFace]) -> str:
        """Human-readable guidance derived from the closest near-miss."""
        best = max(faces, key=lambda face: face.confidence)
        reasons = set(best.rejection_reasons)

        if "truncated_by_frame" in reasons:
            return "The face is cut off by the edge of the photo."
        if "too_few_pixels" in reasons or "face_too_small" in reasons:
            return "The face is too small in the photo. Move closer to the camera."
        if "face_too_large" in reasons:
            return "The face fills the entire photo. Move further from the camera."
        if "low_confidence" in reasons:
            return "No face could be identified with sufficient confidence."
        return "No usable face was found in this image."

    # -- Step 4 helper -------------------------------------------------------- #

    @staticmethod
    def _select_primary(
        faces: list[DetectedFace], image_width: int, image_height: int
    ) -> DetectedFace:
        """Pick the subject from several qualifying faces.

        Scored on normalised area, centrality and confidence. Area carries most
        of the weight because in a selfie the subject is nearly always both the
        largest and the closest face.
        """
        if len(faces) == 1:
            return faces[0]

        max_area = max(face.box.area for face in faces)
        centre_x = image_width / 2.0
        centre_y = image_height / 2.0
        # Half the frame diagonal: the furthest any face centre can be.
        max_distance = ((centre_x**2) + (centre_y**2)) ** 0.5

        def rank(face: DetectedFace) -> float:
            area_term = face.box.area / max_area if max_area > 0 else 0.0
            face_x, face_y = face.box.center
            distance = (((face_x - centre_x) ** 2) + ((face_y - centre_y) ** 2)) ** 0.5
            centrality = 1.0 - min(1.0, distance / max(max_distance, 1.0))
            return float(
                area_term
                + _CENTRALITY_WEIGHT * centrality
                + _CONFIDENCE_WEIGHT * face.confidence
            )

        return max(faces, key=rank)

    # -- Step 5 helpers ------------------------------------------------------- #

    def _quality_gate(
        self, face: DetectedFace
    ) -> tuple[ErrorCode, str, dict[str, Any]] | None:
        """Apply the hard quality gates to the primary face.

        Returns ``None`` when the face passes, otherwise the error code,
        message and structured detail for the rejection.
        """
        if face.pose is not None and not face.pose.within_hard_limits:
            axis = max(
                ("yaw", "pitch", "roll"),
                key=lambda name: abs(getattr(face.pose, name)),
            )
            guidance = {
                "yaw": "Look straight at the camera rather than to the side.",
                "pitch": "Hold the camera at eye level rather than above or below.",
                "roll": "Hold your head upright rather than tilted.",
            }[axis]
            return (
                ErrorCode.FACE_POSE_OUT_OF_RANGE,
                f"The head is turned too far from the camera. {guidance}",
                {
                    "yaw": round(face.pose.yaw, 1),
                    "pitch": round(face.pose.pitch, 1),
                    "roll": round(face.pose.roll, 1),
                    "dominant_axis": axis,
                    "limits": {
                        "yaw": self._config.pose.hard_max_yaw,
                        "pitch": self._config.pose.hard_max_pitch,
                        "roll": self._config.pose.hard_max_roll,
                    },
                },
            )

        if face.occlusion is not None and face.occlusion.occluded:
            regions = face.occlusion.occluded_regions
            readable = ", ".join(name.replace("_", " ") for name in regions) or "the face"
            return (
                ErrorCode.FACE_OCCLUDED,
                (
                    f"Part of the face is obstructed ({readable}). Remove any mask, "
                    f"sunglasses or covering and retake the photo."
                ),
                {
                    "occlusion_score": round(face.occlusion.overall_score, 3),
                    "occluded_regions": regions,
                    "symmetry_delta": round(face.occlusion.symmetry_delta, 3),
                    "threshold": self._config.occlusion.overall_threshold,
                },
            )

        if face.visibility_score < self._config.visibility.min_acceptable:
            return (
                ErrorCode.FACE_NOT_VISIBLE,
                (
                    "The face is not clearly enough visible for verification "
                    f"(limited by {face.limiting_factor.replace('_', ' ')})."
                ),
                {
                    "visibility_score": round(face.visibility_score, 3),
                    "minimum": self._config.visibility.min_acceptable,
                    "limiting_factor": face.limiting_factor,
                    "components": {
                        key: round(value, 3)
                        for key, value in face.visibility_components.items()
                    },
                },
            )

        return None

    def _soft_warnings(self, face: DetectedFace) -> list[AnalysisWarning]:
        """Raise non-fatal observations about an otherwise-acceptable face.

        These do not change the detection outcome but they are consumed by the
        fraud engine in Module 9 and shown to a human reviewer, so a face that
        only just passed is visibly different from one that sailed through.
        """
        warnings: list[AnalysisWarning] = []

        if face.pose is not None and not face.pose.frontal:
            warnings.append(
                AnalysisWarning(
                    code="POSE_NOT_FRONTAL",
                    message=(
                        f"The head is off-frontal on the {face.pose.exceeded_axes} "
                        f"axis but within the hard limit."
                    ),
                    stage="detection",
                    detail={
                        "yaw": round(face.pose.yaw, 1),
                        "pitch": round(face.pose.pitch, 1),
                        "roll": round(face.pose.roll, 1),
                        "exceeded": list(face.pose.exceeded_axes),
                    },
                )
            )

        if (
            face.occlusion is not None
            and not face.occlusion.occluded
            and face.occlusion.occluded_regions
        ):
            warnings.append(
                AnalysisWarning(
                    code="PARTIAL_OCCLUSION",
                    message=(
                        "Some facial regions appear partially obstructed but the "
                        "overall face is still usable."
                    ),
                    stage="detection",
                    detail={"regions": face.occlusion.occluded_regions},
                )
            )

        margin = face.visibility_score - self._config.visibility.min_acceptable
        if 0.0 <= margin < 0.10:
            warnings.append(
                AnalysisWarning(
                    code="VISIBILITY_MARGINAL",
                    message=(
                        "Face visibility only just cleared the minimum; downstream "
                        "match scores may be less reliable."
                    ),
                    stage="detection",
                    detail={
                        "visibility_score": round(face.visibility_score, 3),
                        "limiting_factor": face.limiting_factor,
                    },
                )
            )

        if face.raw.landmarks_derived:
            warnings.append(
                AnalysisWarning(
                    code="LANDMARKS_DERIVED",
                    message=(
                        "Facial landmarks were constructed from a detected eye pair "
                        "rather than predicted by a model. Head pose beyond in-plane "
                        "tilt could not be measured."
                    ),
                    stage="detection",
                    detail={"detector": "haar"},
                )
            )

        return warnings


__all__ = ["DetectionPolicy", "PolicyVerdict"]
