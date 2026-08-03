"""Validate the pose estimator's sign convention by round-tripping known angles.

Projects the canonical 3D face model at a known (yaw, pitch, roll), feeds the
resulting 2D landmarks to the estimator, and prints the recovered angles. Used
to pin the sign convention documented in `hamqadam_ai.detectors.pose`.
"""

from __future__ import annotations

import math
import sys

import cv2
import numpy as np

from hamqadam_ai.core.config import PoseConfig
from hamqadam_ai.core.constants import CANONICAL_FACE_3D_5PT
from hamqadam_ai.detectors.pose import PoseEstimator
from hamqadam_ai.utils.geometry import Landmarks5

WIDTH, HEIGHT = 640, 640


def project(yaw_deg: float, pitch_deg: float, roll_deg: float) -> Landmarks5:
    """Render the 3D model at a known orientation into 2D image points.

    Builds the rotation in the subject-centric convention the estimator claims
    to return, so a correct estimator recovers exactly what was put in.
    """
    # Subject turns right (+yaw) -> negative rotation about the camera Y axis.
    yaw = math.radians(-yaw_deg)
    pitch = math.radians(-pitch_deg)
    roll = math.radians(roll_deg)

    rx = np.array(
        [[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]]
    )
    ry = np.array(
        [[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]]
    )
    rz = np.array(
        [[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]]
    )
    rotation = rz @ ry @ rx

    rvec, _ = cv2.Rodrigues(rotation)
    tvec = np.array([[0.0], [0.0], [600.0]])
    camera = np.array(
        [[WIDTH, 0, WIDTH / 2], [0, WIDTH, HEIGHT / 2], [0, 0, 1]], dtype=np.float64
    )
    points, _ = cv2.projectPoints(
        CANONICAL_FACE_3D_5PT, rvec, tvec, camera, np.zeros((4, 1))
    )
    return Landmarks5(points.reshape(-1, 2).astype(np.float32))


def main() -> int:
    """Print recovered vs expected angles for a grid of orientations."""
    estimator = PoseEstimator(PoseConfig())
    cases = [
        (0.0, 0.0, 0.0),
        (25.0, 0.0, 0.0),
        (-25.0, 0.0, 0.0),
        (0.0, 20.0, 0.0),
        (0.0, -20.0, 0.0),
        (0.0, 0.0, 15.0),
        (0.0, 0.0, -15.0),
        (18.0, 12.0, 8.0),
        (-35.0, -22.0, -14.0),
        (45.0, 0.0, 0.0),
    ]

    print(f"{'expected (y/p/r)':>24} | {'recovered (y/p/r)':>24} | "
          f"{'err':>6} | method | frontal")
    print("-" * 92)

    worst = 0.0
    for yaw, pitch, roll in cases:
        landmarks = project(yaw, pitch, roll)
        result = estimator.estimate(landmarks, (WIDTH, HEIGHT))
        error = max(
            abs(result.yaw - yaw), abs(result.pitch - pitch), abs(result.roll - roll)
        )
        worst = max(worst, error)
        print(
            f"{yaw:7.1f} {pitch:7.1f} {roll:7.1f} | "
            f"{result.yaw:7.1f} {result.pitch:7.1f} {result.roll:7.1f} | "
            f"{error:6.2f} | {result.method:6s} | {result.frontal}"
        )

    print("-" * 92)
    print(f"worst absolute error: {worst:.3f} degrees")

    # Geometric fallback: same landmarks, PnP disabled by forcing the path.
    print()
    print("Geometric fallback (PnP bypassed):")
    for yaw, pitch, roll in [(0.0, 0.0, 0.0), (25.0, 0.0, 0.0), (-25.0, 0.0, 0.0),
                             (0.0, 20.0, 0.0), (0.0, 0.0, 15.0)]:
        landmarks = project(yaw, pitch, roll)
        y, p, r = estimator._from_geometry(landmarks, None)  # noqa: SLF001
        print(f"  expected {yaw:6.1f} {pitch:6.1f} {roll:6.1f}  ->  "
              f"recovered {y:6.1f} {p:6.1f} {r:6.1f}")

    return 0 if worst < 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
