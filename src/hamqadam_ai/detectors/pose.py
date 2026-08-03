"""Head-pose estimation from five facial landmarks.

Two independent estimators, used in that order:

1. **Perspective-n-Point.** Solves for the rigid transform that best projects a
   canonical 3D face model onto the detected 2D landmarks, then decomposes the
   resulting rotation into Euler angles. This is a genuine 3D solve and is
   accurate to a few degrees on cooperative captures.

2. **Landmark geometry.** A closed-form approximation from the landmark
   configuration alone. Used when PnP fails to converge, when its reprojection
   error is implausibly large, or when the landmarks were *derived* rather than
   predicted (see :class:`~hamqadam_ai.detectors.base.RawDetection`).

Camera model
------------
Phone EXIF rarely survives the Flutter upload path, so the intrinsics are
unknown. The standard substitute is used: a pinhole camera with the focal
length set to the image width and the principal point at the image centre,
which corresponds to a horizontal field of view of about 53 degrees - close to
the main camera of essentially every phone in circulation. The resulting
absolute angles carry a systematic error of a few degrees for unusual optics,
but the *thresholds* in ``configs/thresholds.yaml`` are calibrated against this
same assumption, so the accept/reject boundary is unaffected.

Sign convention
---------------
* ``yaw`` > 0: the subject has turned towards the viewer's right.
* ``pitch`` > 0: the subject's chin is raised (looking up).
* ``roll`` > 0: the head is tilted so the viewer-right eye moves down.

These are validated in ``tests/unit/test_pose.py`` by projecting the 3D model
at known angles and asserting recovery, so the convention is enforced by test
rather than by comment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import PoseConfig
from hamqadam_ai.core.constants import CANONICAL_FACE_3D_5PT, EPSILON
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

#: Reprojection error, in units of interocular distance, above which the PnP
#: solution is rejected as a bad fit and the geometric fallback is used.
_MAX_RELATIVE_REPROJECTION_ERROR = 0.35


def _model_geometry() -> tuple[float, float, float, float, float]:
    """Extract the scalars the geometric fallback needs from the 3D model.

    Derived from :data:`~hamqadam_ai.core.constants.CANONICAL_FACE_3D_5PT` at
    import time rather than hard-coded, so editing the model automatically
    recalibrates the fallback instead of silently invalidating it.

    Returns:
        ``(interocular, nose_protrusion, nose_eye_dy, mouth_eye_dy,
        mouth_eye_dz)`` in model millimetres.
    """
    model = CANONICAL_FACE_3D_5PT
    left_eye, right_eye, nose, mouth_left, mouth_right = model
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (mouth_left + mouth_right) / 2.0

    interocular = float(np.linalg.norm(right_eye[:2] - left_eye[:2]))
    nose_protrusion = float(eye_center[2] - nose[2])
    return (
        interocular,
        nose_protrusion,
        float(nose[1] - eye_center[1]),
        float(mouth_center[1] - eye_center[1]),
        float(mouth_center[2] - eye_center[2]),
    )


(
    _MODEL_INTEROCULAR,
    _MODEL_NOSE_PROTRUSION,
    _MODEL_NOSE_EYE_DY,
    _MODEL_MOUTH_EYE_DY,
    _MODEL_MOUTH_EYE_DZ,
) = _model_geometry()

#: ``E / P`` from the yaw derivation in :meth:`PoseEstimator._from_geometry`.
_YAW_GEOMETRY_GAIN = _MODEL_INTEROCULAR / max(_MODEL_NOSE_PROTRUSION, EPSILON)

#: Nose depth relative to the eye plane, signed to match the pitch derivation.
_MODEL_NOSE_EYE_DZ = -_MODEL_NOSE_PROTRUSION


@dataclass(frozen=True, slots=True)
class PoseResult:
    """Head orientation and how far it is from acceptable.

    Attributes:
        yaw: Left/right rotation in degrees.
        pitch: Up/down rotation in degrees.
        roll: In-plane tilt in degrees.
        frontal: Every axis within the soft limit.
        within_hard_limits: Every axis within the hard limit. False means the
            face is unusable for recognition, not merely imperfect.
        deviation_score: Normalised distance from frontal in ``[0, 1]``, where
            1.0 means at or beyond the hard limit on the worst axis.
        method: ``pnp``, ``landmark_geometry`` or ``roll_only``.
        reprojection_error: Mean landmark reprojection error in pixels, for the
            PnP path only.
        exceeded_axes: Which axes broke their soft limit.
    """

    yaw: float
    pitch: float
    roll: float
    frontal: bool
    within_hard_limits: bool
    deviation_score: float
    method: str
    reprojection_error: float | None = None
    exceeded_axes: tuple[str, ...] = ()

    @property
    def max_absolute_angle(self) -> float:
        """Largest absolute rotation across the three axes."""
        return max(abs(self.yaw), abs(self.pitch), abs(self.roll))

    def describe(self) -> dict[str, Any]:
        """Compact summary for logging."""
        return {
            "yaw": round(self.yaw, 1),
            "pitch": round(self.pitch, 1),
            "roll": round(self.roll, 1),
            "frontal": self.frontal,
            "method": self.method,
            "deviation": round(self.deviation_score, 3),
        }


class PoseEstimator:
    """Estimates head orientation and scores it against the configured limits.

    Args:
        config: The pose section of the detection configuration.
    """

    __slots__ = ("_config", "_model_points")

    def __init__(self, config: PoseConfig) -> None:
        self._config = config
        self._model_points = np.ascontiguousarray(
            CANONICAL_FACE_3D_5PT, dtype=np.float64
        )

    def estimate(
        self,
        landmarks: Landmarks5,
        image_size: tuple[int, int],
        *,
        box: BoundingBox | None = None,
        landmarks_derived: bool = False,
    ) -> PoseResult:
        """Estimate head pose from five landmarks.

        Args:
            landmarks: The five keypoints in source-image pixels.
            image_size: ``(width, height)`` of the source image, for the
                camera intrinsics.
            box: The face box. Unused by the PnP solve; accepted so callers do
                not have to branch, and used to sanity-check the landmark scale.
            landmarks_derived: True when the landmarks were constructed from a
                template. Forces the ``roll_only`` path, because a constructed
                nose tip encodes no yaw or pitch information and a PnP solve on
                it would confidently report a frontal pose for any face.

        Returns:
            The estimated pose, scored against the configured limits.
        """
        if landmarks_derived:
            return self._score(
                yaw=0.0,
                pitch=0.0,
                roll=landmarks.roll_degrees,
                method="roll_only",
                reprojection_error=None,
            )

        pnp = self._solve_pnp(landmarks, image_size)
        if pnp is not None:
            yaw, pitch, roll, error = pnp
            return self._score(
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                method="pnp",
                reprojection_error=error,
            )

        yaw, pitch, roll = self._from_geometry(landmarks, box)
        return self._score(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            method="landmark_geometry",
            reprojection_error=None,
        )

    # -- Estimators --------------------------------------------------------- #

    def _solve_pnp(
        self, landmarks: Landmarks5, image_size: tuple[int, int]
    ) -> tuple[float, float, float, float] | None:
        """Run the PnP solve, returning ``(yaw, pitch, roll, error)`` or None.

        ``SOLVEPNP_SQPNP`` is used rather than the classic iterative solver: it
        is a global, non-iterative method that needs no initial guess and does
        not fall into the mirror-ambiguity local minimum that ``ITERATIVE``
        hits on near-frontal faces - the exact case that dominates this
        workload.
        """
        width, height = image_size
        focal = float(width)
        camera_matrix = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        # No lens-distortion model: phone JPEGs are already rectified by the
        # camera pipeline and an unconstrained distortion estimate from five
        # points would be pure overfitting.
        distortion = np.zeros((4, 1), dtype=np.float64)

        image_points = np.ascontiguousarray(landmarks.points, dtype=np.float64)

        try:
            ok, rotation_vector, translation_vector = cv2.solvePnP(
                self._model_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error:
            return None

        if not ok:
            return None

        rotation = np.asarray(rotation_vector, dtype=np.float64)
        translation = np.asarray(translation_vector, dtype=np.float64)
        error = self._reprojection_error(
            image_points, rotation, translation, camera_matrix, distortion
        )
        interocular = max(landmarks.interocular_distance, EPSILON)
        if error / interocular > _MAX_RELATIVE_REPROJECTION_ERROR:
            # The solve converged on something, but not on this face.
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation)
        pitch, yaw, roll = self._euler_from_rotation(
            np.asarray(rotation_matrix, dtype=np.float64)
        )
        return yaw, pitch, roll, error

    @staticmethod
    def _euler_from_rotation(
        rotation: npt.NDArray[np.float64],
    ) -> tuple[float, float, float]:
        """Decompose a rotation matrix into ``(pitch, yaw, roll)`` in degrees.

        Uses the standard XYZ (Tait-Bryan) decomposition, with the gimbal-lock
        branch handled explicitly. The sign flips at the end convert from the
        camera-frame rotation - which describes how the *model* was rotated to
        match the image - into the subject-centric convention documented in the
        module docstring.
        """
        sy = math.sqrt(
            float(rotation[0, 0]) ** 2 + float(rotation[1, 0]) ** 2
        )
        if sy < 1e-6:
            # Gimbal lock: yaw is +/-90 degrees and roll is indistinguishable
            # from pitch. Attribute the whole in-plane component to pitch.
            pitch = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
            yaw = math.atan2(-float(rotation[2, 0]), sy)
            roll = 0.0
        else:
            pitch = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
            yaw = math.atan2(-float(rotation[2, 0]), sy)
            roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))

        # Camera-frame -> subject-frame. A head turned to the viewer's right
        # corresponds to a negative rotation about the camera's Y axis, and a
        # raised chin to a negative rotation about X.
        return (
            -math.degrees(pitch),
            -math.degrees(yaw),
            math.degrees(roll),
        )

    @staticmethod
    def _reprojection_error(
        image_points: npt.NDArray[np.float64],
        rotation_vector: npt.NDArray[np.float64],
        translation_vector: npt.NDArray[np.float64],
        camera_matrix: npt.NDArray[np.float64],
        distortion: npt.NDArray[np.float64],
    ) -> float:
        """Mean Euclidean distance between observed and reprojected landmarks."""
        projected, _ = cv2.projectPoints(
            CANONICAL_FACE_3D_5PT,
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion,
        )
        projected = projected.reshape(-1, 2)
        return float(np.mean(np.linalg.norm(projected - image_points, axis=1)))

    def _from_geometry(
        self, landmarks: Landmarks5, box: BoundingBox | None
    ) -> tuple[float, float, float]:
        """Closed-form pose approximation from the landmark configuration.

        This is not a curve fit. Under a weak-perspective (orthographic)
        camera, both angles invert analytically from the same 3D model the PnP
        solve uses, with the constants read straight off
        :data:`~hamqadam_ai.core.constants.CANONICAL_FACE_3D_5PT` so the two
        estimators can never drift apart.

        **Yaw.** The nose tip protrudes ``P`` millimetres in front of the eye
        plane. Rotating the head by ``theta`` displaces it horizontally by
        ``P sin(theta)`` while the eye separation ``E`` projects to
        ``E cos(theta)``. Therefore::

            dx / interocular = (P / E) * tan(theta)
            theta = atan( (dx / interocular) * (E / P) )

        **Pitch.** With the eye, nose and mouth heights ``ey, ny, my`` and
        depths ``ez, nz, mz``, the nose's fractional position ``f`` within the
        projected eye-to-mouth band is a ratio of two linear functions of
        ``sin(phi)`` and ``cos(phi)``, which rearranges to::

            tan(phi) = (f * (my - ey) - (ny - ey)) / ((nz - ez) - f * (mz - ez))

        Both are exact under orthography; the residual error against a real
        perspective camera is a few degrees at typical selfie distances. That
        is ample for a fallback whose only job is to catch a grossly
        non-frontal face when the PnP solve has already failed.
        """
        interocular = max(landmarks.interocular_distance, EPSILON)
        roll = landmarks.roll_degrees

        eye_x, eye_y = landmarks.eye_center
        nose_x, nose_y = landmarks.nose_tip
        mouth_x, mouth_y = landmarks.mouth_center

        # Undo in-plane rotation before measuring anything, otherwise a tilted
        # head reads as a turned one and the mouth line is no longer vertical
        # below the eyes. Both offsets must be de-rotated, not just the nose:
        # measuring the eye-to-mouth band in the un-rotated frame was the
        # source of a spurious 6-degree pitch on a 15-degree roll.
        angle = math.radians(-roll)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        def derotate(px: float, py: float) -> tuple[float, float]:
            dx = px - eye_x
            dy = py - eye_y
            return (dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a)

        nose_dx, nose_dy = derotate(nose_x, nose_y)
        _, mouth_dy = derotate(mouth_x, mouth_y)

        yaw = math.degrees(
            math.atan((nose_dx / interocular) * _YAW_GEOMETRY_GAIN)
        )

        band = mouth_dy
        if abs(band) < EPSILON:
            pitch = 0.0
        else:
            fraction = nose_dy / band
            numerator = fraction * _MODEL_MOUTH_EYE_DY - _MODEL_NOSE_EYE_DY
            denominator = _MODEL_NOSE_EYE_DZ - fraction * _MODEL_MOUTH_EYE_DZ
            pitch = math.degrees(math.atan2(numerator, denominator))
            # atan2 returns the full circle; fold the far branch back into the
            # +/- 90 degree range a head can physically occupy.
            if pitch > 90.0:
                pitch -= 180.0
            elif pitch < -90.0:
                pitch += 180.0

        # A face box far from a plausible head aspect ratio is evidence of a
        # detector artefact rather than a real pose; damp the estimate rather
        # than reporting a confident angle derived from a bad box.
        if box is not None and box.height > 0:
            aspect = box.width / box.height
            if aspect > 1.6 or aspect < 0.45:
                yaw *= 0.5
                pitch *= 0.5

        return (
            float(np.clip(yaw, -90.0, 90.0)),
            float(np.clip(pitch, -90.0, 90.0)),
            roll,
        )

    # -- Scoring ------------------------------------------------------------ #

    def _score(
        self,
        *,
        yaw: float,
        pitch: float,
        roll: float,
        method: str,
        reprojection_error: float | None,
    ) -> PoseResult:
        """Compare the angles against the configured limits.

        The deviation score is the maximum per-axis ratio of the measured angle
        to that axis's hard limit. Taking the maximum rather than an average is
        deliberate: 45 degrees of yaw makes a face unusable no matter how
        perfect its pitch and roll are, and an average would dilute exactly the
        signal that matters.
        """
        config = self._config
        measurements = {"yaw": abs(yaw), "pitch": abs(pitch), "roll": abs(roll)}
        soft_limits = {
            "yaw": config.max_yaw,
            "pitch": config.max_pitch,
            "roll": config.max_roll,
        }
        hard_limits = {
            "yaw": config.hard_max_yaw,
            "pitch": config.hard_max_pitch,
            "roll": config.hard_max_roll,
        }

        exceeded = tuple(
            axis for axis, value in measurements.items() if value > soft_limits[axis]
        )
        within_hard = all(
            value <= hard_limits[axis] for axis, value in measurements.items()
        )
        deviation = max(
            min(1.0, measurements[axis] / max(hard_limits[axis], EPSILON))
            for axis in measurements
        )

        return PoseResult(
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            frontal=not exceeded,
            within_hard_limits=within_hard,
            deviation_score=float(deviation),
            method=method,
            reprojection_error=reprojection_error,
            exceeded_axes=exceeded,
        )


__all__ = ["PoseEstimator", "PoseResult"]
