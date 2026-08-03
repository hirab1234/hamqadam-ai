"""Unit tests for head-pose estimation.

The sign convention documented in :mod:`hamqadam_ai.detectors.pose` is enforced
here rather than merely described: known orientations are projected through the
canonical 3D model and the estimator must recover them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hamqadam_ai.core.config import PoseConfig
from hamqadam_ai.detectors.pose import PoseEstimator
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5
from tests.conftest import _project_landmarks

IMAGE_SIZE = 640
DISTANCE = 600.0


@pytest.fixture
def estimator() -> PoseEstimator:
    """Estimator with the default configured limits."""
    return PoseEstimator(PoseConfig())


def project(yaw: float, pitch: float, roll: float) -> Landmarks5:
    """Project the 3D model at a known orientation."""
    return Landmarks5(_project_landmarks(yaw, pitch, roll, IMAGE_SIZE, DISTANCE))


# --------------------------------------------------------------------------- #
# PnP accuracy and sign convention
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("yaw", "pitch", "roll"),
    [
        (0.0, 0.0, 0.0),
        (25.0, 0.0, 0.0),
        (-25.0, 0.0, 0.0),
        (0.0, 20.0, 0.0),
        (0.0, -20.0, 0.0),
        (0.0, 0.0, 15.0),
        (0.0, 0.0, -15.0),
        (18.0, 12.0, 8.0),
        (-35.0, -22.0, -14.0),
        (45.0, 30.0, 20.0),
    ],
)
def test_pnp_recovers_known_orientation(
    estimator: PoseEstimator, yaw: float, pitch: float, roll: float
) -> None:
    """PnP must recover a projected orientation to sub-degree accuracy."""
    result = estimator.estimate(project(yaw, pitch, roll), (IMAGE_SIZE, IMAGE_SIZE))

    assert result.method == "pnp"
    assert result.yaw == pytest.approx(yaw, abs=0.5)
    assert result.pitch == pytest.approx(pitch, abs=0.5)
    assert result.roll == pytest.approx(roll, abs=0.5)


@pytest.mark.unit
def test_yaw_sign_means_turned_to_viewer_right(estimator: PoseEstimator) -> None:
    """Positive yaw is documented as 'turned towards the viewer's right'.

    Verified physically rather than by round-trip: when the subject turns that
    way, the nose tip must move right of the eye midpoint in the image.
    """
    landmarks = project(30.0, 0.0, 0.0)
    eye_x, _ = landmarks.eye_center
    nose_x, _ = landmarks.nose_tip

    assert nose_x > eye_x
    assert estimator.estimate(landmarks, (IMAGE_SIZE, IMAGE_SIZE)).yaw > 0


@pytest.mark.unit
def test_roll_sign_matches_eye_line_geometry(estimator: PoseEstimator) -> None:
    """Positive roll must put the viewer-right eye lower in the image."""
    landmarks = project(0.0, 0.0, 20.0)

    assert landmarks.right_eye[1] > landmarks.left_eye[1]
    assert estimator.estimate(landmarks, (IMAGE_SIZE, IMAGE_SIZE)).roll > 0


@pytest.mark.unit
def test_pitch_sign_means_chin_raised(estimator: PoseEstimator) -> None:
    """Positive pitch is 'chin raised'.

    With the chin up, the mouth-to-eye vertical span foreshortens relative to
    frontal, because the lower face rotates away from the camera.
    """
    frontal = project(0.0, 0.0, 0.0)
    chin_up = project(25.0 * 0.0 + 0.0, 25.0, 0.0)

    def span(marks: Landmarks5) -> float:
        return marks.mouth_center[1] - marks.eye_center[1]

    assert span(chin_up) < span(frontal)
    assert estimator.estimate(chin_up, (IMAGE_SIZE, IMAGE_SIZE)).pitch > 0


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_frontal_face_is_within_all_limits(estimator: PoseEstimator) -> None:
    """A perfectly frontal face is frontal, within limits and zero-deviation."""
    result = estimator.estimate(project(0.0, 0.0, 0.0), (IMAGE_SIZE, IMAGE_SIZE))

    assert result.frontal
    assert result.within_hard_limits
    assert result.deviation_score == pytest.approx(0.0, abs=0.02)
    assert result.exceeded_axes == ()


@pytest.mark.unit
def test_soft_limit_breach_is_not_hard_limit_breach(estimator: PoseEstimator) -> None:
    """40 degrees of yaw exceeds the soft limit of 30 but not the hard 50."""
    result = estimator.estimate(project(40.0, 0.0, 0.0), (IMAGE_SIZE, IMAGE_SIZE))

    assert not result.frontal
    assert result.within_hard_limits
    assert "yaw" in result.exceeded_axes


@pytest.mark.unit
def test_hard_limit_breach_is_reported(estimator: PoseEstimator) -> None:
    """60 degrees of yaw exceeds the hard limit and saturates the deviation."""
    result = estimator.estimate(project(60.0, 0.0, 0.0), (IMAGE_SIZE, IMAGE_SIZE))

    assert not result.frontal
    assert not result.within_hard_limits
    assert result.deviation_score == pytest.approx(1.0)


@pytest.mark.unit
def test_deviation_takes_the_worst_axis_not_the_average(
    estimator: PoseEstimator,
) -> None:
    """One bad axis must not be diluted by two good ones.

    A face at 45 degrees of yaw is unusable however perfect its pitch and roll,
    so the deviation must reflect the yaw alone.
    """
    result = estimator.estimate(project(45.0, 0.0, 0.0), (IMAGE_SIZE, IMAGE_SIZE))

    assert result.deviation_score == pytest.approx(45.0 / 50.0, abs=0.05)


# --------------------------------------------------------------------------- #
# Geometric fallback
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("yaw", "pitch", "roll"),
    [(0.0, 0.0, 0.0), (25.0, 0.0, 0.0), (-25.0, 0.0, 0.0), (0.0, 20.0, 0.0)],
)
def test_geometric_fallback_within_documented_tolerance(
    estimator: PoseEstimator, yaw: float, pitch: float, roll: float
) -> None:
    """The closed-form fallback must stay inside its documented +/- 8 degrees."""
    landmarks = project(yaw, pitch, roll)
    got_yaw, got_pitch, got_roll = estimator._from_geometry(landmarks, None)  # noqa: SLF001

    assert got_yaw == pytest.approx(yaw, abs=8.0)
    assert got_pitch == pytest.approx(pitch, abs=8.0)
    assert got_roll == pytest.approx(roll, abs=2.0)


@pytest.mark.unit
def test_roll_does_not_leak_into_pitch(estimator: PoseEstimator) -> None:
    """De-rotation must be applied to the eye-to-mouth band, not only the nose.

    Regression test: measuring the band in the un-rotated frame produced a
    spurious 6-degree pitch on a purely rolled head.
    """
    landmarks = project(0.0, 0.0, 25.0)
    _, pitch, _ = estimator._from_geometry(landmarks, None)  # noqa: SLF001

    assert abs(pitch) < 2.0


@pytest.mark.unit
def test_derived_landmarks_report_roll_only(estimator: PoseEstimator) -> None:
    """Constructed landmarks must not produce a confident yaw or pitch.

    A template-derived nose tip encodes no out-of-plane information. Running
    PnP on it would report perfect frontality for a face at 40 degrees of yaw,
    which is exactly the failure this guard exists to prevent.
    """
    landmarks = project(40.0, 30.0, 12.0)
    result = estimator.estimate(
        landmarks, (IMAGE_SIZE, IMAGE_SIZE), landmarks_derived=True
    )

    assert result.method == "roll_only"
    assert result.yaw == 0.0
    assert result.pitch == 0.0
    assert result.roll == pytest.approx(landmarks.roll_degrees, abs=0.01)


@pytest.mark.unit
def test_degenerate_landmarks_do_not_raise(estimator: PoseEstimator) -> None:
    """Collinear landmarks must degrade, not explode."""
    collinear = Landmarks5(
        np.array(
            [[100, 100], [200, 100], [150, 100], [120, 100], [180, 100]],
            dtype=np.float32,
        )
    )
    result = estimator.estimate(collinear, (IMAGE_SIZE, IMAGE_SIZE))

    assert math.isfinite(result.yaw)
    assert math.isfinite(result.pitch)
    assert math.isfinite(result.roll)
    assert 0.0 <= result.deviation_score <= 1.0


@pytest.mark.unit
def test_box_aspect_ratio_damps_implausible_geometry(
    estimator: PoseEstimator,
) -> None:
    """An absurd face box halves the geometric estimate rather than trusting it."""
    landmarks = project(30.0, 0.0, 0.0)
    normal = BoundingBox(0, 0, 200, 260)
    absurd = BoundingBox(0, 0, 400, 100)

    yaw_normal, _, _ = estimator._from_geometry(landmarks, normal)  # noqa: SLF001
    yaw_absurd, _, _ = estimator._from_geometry(landmarks, absurd)  # noqa: SLF001

    assert abs(yaw_absurd) == pytest.approx(abs(yaw_normal) / 2.0, rel=0.01)


@pytest.mark.unit
def test_reprojection_error_is_reported_for_pnp(estimator: PoseEstimator) -> None:
    """A clean solve reports a near-zero reprojection error."""
    result = estimator.estimate(project(15.0, 10.0, 5.0), (IMAGE_SIZE, IMAGE_SIZE))

    assert result.reprojection_error is not None
    assert result.reprojection_error < 1.0


@pytest.mark.unit
def test_pnp_result_is_only_trusted_when_it_actually_fits(
    estimator: PoseEstimator,
) -> None:
    """Whenever the PnP path is taken, its fit must be within tolerance.

    This is the real invariant. An earlier version of this test asserted that a
    near-collinear landmark set would be *rejected*, which was wrong: SQPNP
    legitimately explains eyes and mouth on one line as a face at extreme
    pitch, and it reprojects accurately. The contract is not "reject weird
    input", it is "never report a pose the solve could not actually explain".
    """
    awkward = Landmarks5(
        np.array(
            [[100, 300], [400, 300], [250, 305], [150, 302], [350, 298]],
            dtype=np.float32,
        )
    )
    result = estimator.estimate(awkward, (IMAGE_SIZE, IMAGE_SIZE))

    if result.method == "pnp":
        assert result.reprojection_error is not None
        relative = result.reprojection_error / awkward.interocular_distance
        assert relative <= 0.35
    else:
        assert result.method == "landmark_geometry"


@pytest.mark.unit
def test_poor_reprojection_triggers_the_geometric_fallback(
    estimator: PoseEstimator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A solve that converges on the wrong answer must be discarded.

    Driven by forcing ``solvePnP`` to return a pose that does not reproject
    onto the observed landmarks, which is the condition the guard exists for
    and which real input reaches only rarely.
    """
    import cv2

    def bogus_solve(*_args: object, **_kwargs: object):  # noqa: ANN202
        rotation = np.array([[0.9], [0.4], [0.2]], dtype=np.float64)
        translation = np.array([[300.0], [200.0], [900.0]], dtype=np.float64)
        return True, rotation, translation

    monkeypatch.setattr(cv2, "solvePnP", bogus_solve)
    result = estimator.estimate(project(10.0, 5.0, 0.0), (IMAGE_SIZE, IMAGE_SIZE))

    assert result.method == "landmark_geometry"
    assert result.reprojection_error is None


@pytest.mark.unit
def test_solver_exception_falls_back_rather_than_propagating(
    estimator: PoseEstimator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OpenCV failure inside the solve must degrade, not fail the request."""
    import cv2

    def raising_solve(*_args: object, **_kwargs: object):  # noqa: ANN202
        raise cv2.error("synthetic solver failure")

    monkeypatch.setattr(cv2, "solvePnP", raising_solve)
    result = estimator.estimate(project(10.0, 5.0, 0.0), (IMAGE_SIZE, IMAGE_SIZE))

    assert result.method == "landmark_geometry"
