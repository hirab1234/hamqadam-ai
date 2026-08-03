"""Unit tests for the detection acceptance policy.

The policy is pure business logic over synthetic detections, so these tests
need no model weights and no images. Rule *ordering* is tested as carefully as
the rules themselves: which error code a user receives determines whether they
can fix the problem, so "occluded" must beat "not visible" and "too small" must
beat "not detected".
"""

from __future__ import annotations

import numpy as np
import pytest

from hamqadam_ai.core.config import DetectionConfig
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.detectors.base import DetectedFace
from hamqadam_ai.detectors.occlusion import OcclusionResult, RegionEvidence
from hamqadam_ai.detectors.policy import DetectionPolicy
from hamqadam_ai.detectors.pose import PoseResult
from tests.conftest import make_face

IMAGE = (1000, 1000)


@pytest.fixture
def policy() -> DetectionPolicy:
    """Policy with the default configured thresholds."""
    return DetectionPolicy(DetectionConfig())


def apply(policy: DetectionPolicy, faces: list[DetectedFace], **kwargs: object):  # noqa: ANN201
    """Apply the policy against the standard test frame."""
    return policy.apply(
        faces, image_width=IMAGE[0], image_height=IMAGE[1], **kwargs
    )


def big_face(**kwargs: object) -> DetectedFace:
    """A face large enough to clear every size rule."""
    return make_face((300, 300, 620, 700), image_size=IMAGE, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Detection presence
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_no_detections_is_face_not_detected(policy: DetectionPolicy) -> None:
    verdict = apply(policy, [])

    assert not verdict.passed
    assert verdict.error_code is ErrorCode.FACE_NOT_DETECTED
    assert verdict.qualifying_count == 0


@pytest.mark.unit
def test_single_good_face_passes(policy: DetectionPolicy) -> None:
    verdict = apply(policy, [big_face()])

    assert verdict.passed
    assert verdict.error_code is None
    assert verdict.qualifying_count == 1
    assert verdict.primary_face is not None
    assert verdict.primary_face.is_primary


# --------------------------------------------------------------------------- #
# Admissibility rules
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_low_confidence_face_is_inadmissible(policy: DetectionPolicy) -> None:
    verdict = apply(policy, [big_face(confidence=0.30)])

    assert not verdict.passed
    assert verdict.error_code is ErrorCode.FACE_NOT_DETECTED


@pytest.mark.unit
def test_face_below_pixel_floor_is_rejected(policy: DetectionPolicy) -> None:
    """A 40 px face has too few pixels for a usable embedding."""
    verdict = apply(policy, [make_face((100, 100, 140, 150), image_size=IMAGE)])

    assert not verdict.passed
    assert verdict.error_code is ErrorCode.FACE_TOO_SMALL


@pytest.mark.unit
def test_face_below_area_ratio_is_rejected(policy: DetectionPolicy) -> None:
    """80x80 in a 1000x1000 frame is 0.0064, under the 0.008 floor."""
    verdict = apply(policy, [make_face((100, 100, 180, 180), image_size=IMAGE)])

    assert not verdict.passed
    assert verdict.error_code is ErrorCode.FACE_TOO_SMALL


@pytest.mark.unit
def test_face_filling_the_frame_is_rejected(policy: DetectionPolicy) -> None:
    verdict = apply(policy, [make_face((0, 0, 1000, 1000), image_size=IMAGE)])

    assert not verdict.passed
    assert "face_too_large" in verdict.detail["rejection_reasons"]


@pytest.mark.unit
def test_truncated_face_is_rejected_with_its_own_code(
    policy: DetectionPolicy,
) -> None:
    """Half the box outside the frame is a truncation, not an absence."""
    verdict = apply(policy, [make_face((-200, 300, 200, 700), image_size=IMAGE)])

    assert not verdict.passed
    assert verdict.error_code is ErrorCode.FACE_TRUNCATED
    assert "cut off" in (verdict.message or "")


@pytest.mark.unit
def test_rejection_reasons_are_recorded_on_the_face(
    policy: DetectionPolicy,
) -> None:
    """Every violated rule is annotated, not just the first."""
    face = make_face((100, 100, 140, 150), confidence=0.2, image_size=IMAGE)
    apply(policy, [face])

    assert "low_confidence" in face.rejection_reasons
    assert "too_few_pixels" in face.rejection_reasons


# --------------------------------------------------------------------------- #
# Single-person rule and bystanders
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_two_comparable_faces_are_rejected(policy: DetectionPolicy) -> None:
    verdict = apply(
        policy,
        [
            make_face((100, 300, 400, 700), image_size=IMAGE),
            make_face((600, 300, 900, 700), image_size=IMAGE),
        ],
    )

    assert not verdict.passed
    assert verdict.error_code is ErrorCode.MULTIPLE_FACES_DETECTED
    assert verdict.qualifying_count == 2


@pytest.mark.unit
def test_small_background_face_is_ignored_as_a_bystander(
    policy: DetectionPolicy,
) -> None:
    """A distant onlooker must not block a legitimate selfie."""
    subject = make_face((300, 250, 700, 750), image_size=IMAGE)
    onlooker = make_face((60, 60, 160, 190), image_size=IMAGE)

    verdict = apply(policy, [subject, onlooker])

    assert verdict.passed
    assert verdict.qualifying_count == 1
    assert onlooker.is_bystander
    assert not subject.is_bystander
    assert any(w.code == "BACKGROUND_FACES_IGNORED" for w in verdict.warnings)


@pytest.mark.unit
def test_select_primary_mode_picks_the_dominant_face() -> None:
    """With `select_primary` configured, a crowd resolves rather than rejects."""
    config = DetectionConfig()
    config.policy.on_multiple_faces = "select_primary"
    policy = DetectionPolicy(config)

    small = make_face((100, 350, 380, 650), image_size=IMAGE)
    large = make_face((450, 250, 900, 780), image_size=IMAGE)

    verdict = policy.apply(
        [small, large], image_width=IMAGE[0], image_height=IMAGE[1]
    )

    assert verdict.passed
    assert verdict.primary_face is large
    assert any(w.code == "MULTIPLE_FACES_PRESENT" for w in verdict.warnings)


@pytest.mark.unit
def test_primary_selection_prefers_the_larger_face(policy: DetectionPolicy) -> None:
    config = DetectionConfig()
    config.policy.max_faces_allowed = 5
    relaxed = DetectionPolicy(config)

    small = make_face((100, 400, 340, 700), image_size=IMAGE)
    large = make_face((450, 250, 900, 800), image_size=IMAGE)

    verdict = relaxed.apply(
        [small, large], image_width=IMAGE[0], image_height=IMAGE[1]
    )

    assert verdict.primary_face is large


@pytest.mark.unit
def test_primary_selection_breaks_ties_on_centrality() -> None:
    """Equal-sized faces resolve towards the centre of the frame."""
    config = DetectionConfig()
    config.policy.max_faces_allowed = 5
    policy = DetectionPolicy(config)

    edge = make_face((10, 350, 310, 650), image_size=IMAGE)
    centre = make_face((350, 350, 650, 650), image_size=IMAGE)

    verdict = policy.apply(
        [edge, centre], image_width=IMAGE[0], image_height=IMAGE[1]
    )

    assert verdict.primary_face is centre


# --------------------------------------------------------------------------- #
# Quality gates on the primary face
# --------------------------------------------------------------------------- #


def with_pose(face: DetectedFace, **kwargs: object) -> DetectedFace:
    """Attach a pose result to a face."""
    defaults = {
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "frontal": True,
        "within_hard_limits": True,
        "deviation_score": 0.0,
        "method": "pnp",
        "exceeded_axes": (),
    }
    defaults.update(kwargs)
    face.pose = PoseResult(**defaults)  # type: ignore[arg-type]
    return face


def with_occlusion(face: DetectedFace, *, occluded: bool, score: float) -> DetectedFace:
    """Attach an occlusion result to a face."""
    face.occlusion = OcclusionResult(
        occluded=occluded,
        overall_score=score,
        regions={
            "left_eye": RegionEvidence(
                probability=score,
                occluded=occluded,
                flat_fraction=0.9 if occluded else 0.1,
                texture_energy=0.2 if occluded else 1.2,
                skin_coverage=0.05 if occluded else 0.7,
                colour_uniformity=0.9 if occluded else 0.2,
                mean_luminance=0.3 if occluded else 1.0,
            )
        },
        occluded_regions=["left_eye"] if occluded else [],
    )
    return face


@pytest.mark.unit
def test_pose_beyond_hard_limits_is_rejected(policy: DetectionPolicy) -> None:
    face = with_pose(
        big_face(),
        yaw=62.0,
        frontal=False,
        within_hard_limits=False,
        deviation_score=1.0,
        exceeded_axes=("yaw",),
    )
    verdict = apply(policy, [face])

    assert not verdict.passed
    assert verdict.error_code is ErrorCode.FACE_POSE_OUT_OF_RANGE
    assert verdict.detail["dominant_axis"] == "yaw"
    assert "straight at the camera" in (verdict.message or "")


@pytest.mark.unit
def test_pose_within_hard_limits_only_warns(policy: DetectionPolicy) -> None:
    """Off-frontal but usable must pass with a warning, not a rejection."""
    face = with_pose(
        big_face(),
        yaw=38.0,
        frontal=False,
        within_hard_limits=True,
        deviation_score=0.76,
        exceeded_axes=("yaw",),
    )
    verdict = apply(policy, [face])

    assert verdict.passed
    assert any(w.code == "POSE_NOT_FRONTAL" for w in verdict.warnings)


@pytest.mark.unit
def test_occluded_face_is_rejected(policy: DetectionPolicy) -> None:
    face = with_occlusion(with_pose(big_face()), occluded=True, score=0.72)
    verdict = apply(policy, [face])

    assert not verdict.passed
    assert verdict.error_code is ErrorCode.FACE_OCCLUDED
    assert "left eye" in (verdict.message or "")


@pytest.mark.unit
def test_partial_occlusion_below_threshold_only_warns(
    policy: DetectionPolicy,
) -> None:
    face = with_pose(big_face())
    face.occlusion = OcclusionResult(
        occluded=False,
        overall_score=0.20,
        regions={},
        occluded_regions=["chin"],
    )
    verdict = apply(policy, [face])

    assert verdict.passed
    assert any(w.code == "PARTIAL_OCCLUSION" for w in verdict.warnings)


@pytest.mark.unit
def test_low_visibility_is_rejected(policy: DetectionPolicy) -> None:
    face = with_pose(big_face(visibility=0.30))
    face.limiting_factor = "face_size"
    verdict = apply(policy, [face])

    assert not verdict.passed
    assert verdict.error_code is ErrorCode.FACE_NOT_VISIBLE
    assert verdict.detail["limiting_factor"] == "face_size"


@pytest.mark.unit
def test_marginal_visibility_warns(policy: DetectionPolicy) -> None:
    """A face that only just cleared the bar is flagged for the fraud engine."""
    face = with_pose(big_face(visibility=0.58))
    verdict = apply(policy, [face])

    assert verdict.passed
    assert any(w.code == "VISIBILITY_MARGINAL" for w in verdict.warnings)


# --------------------------------------------------------------------------- #
# Rule ordering
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_occlusion_is_reported_before_low_visibility(
    policy: DetectionPolicy,
) -> None:
    """Both conditions hold; the actionable one must be the one reported.

    "Remove your sunglasses" is a fixable instruction. "The face is not
    sufficiently visible" sends the user round a loop they cannot exit.
    """
    face = with_occlusion(
        with_pose(big_face(visibility=0.20)), occluded=True, score=0.80
    )
    verdict = apply(policy, [face])

    assert verdict.error_code is ErrorCode.FACE_OCCLUDED


@pytest.mark.unit
def test_pose_is_reported_before_occlusion(policy: DetectionPolicy) -> None:
    """An extreme angle makes the occlusion measurement unreliable anyway."""
    face = with_occlusion(
        with_pose(
            big_face(),
            yaw=70.0,
            frontal=False,
            within_hard_limits=False,
            deviation_score=1.0,
            exceeded_axes=("yaw",),
        ),
        occluded=True,
        score=0.80,
    )
    verdict = apply(policy, [face])

    assert verdict.error_code is ErrorCode.FACE_POSE_OUT_OF_RANGE


@pytest.mark.unit
def test_multiple_faces_is_reported_before_quality_gates(
    policy: DetectionPolicy,
) -> None:
    """Which face's quality would we even be reporting?"""
    faces = [
        with_occlusion(with_pose(make_face((100, 300, 400, 700), image_size=IMAGE)),
                       occluded=True, score=0.9),
        make_face((600, 300, 900, 700), image_size=IMAGE),
    ]
    verdict = apply(policy, faces)

    assert verdict.error_code is ErrorCode.MULTIPLE_FACES_DETECTED


# --------------------------------------------------------------------------- #
# Landmark-free detectors
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_missing_landmarks_raises_a_warning(policy: DetectionPolicy) -> None:
    verdict = apply(policy, [big_face()], detector_provides_landmarks=False)

    assert verdict.passed
    assert any(w.code == "LANDMARKS_UNAVAILABLE" for w in verdict.warnings)


@pytest.mark.unit
def test_derived_landmarks_are_disclosed(policy: DetectionPolicy) -> None:
    """A Haar-constructed landmark set must be visible to a reviewer."""
    from hamqadam_ai.detectors.base import RawDetection
    from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

    face = DetectedFace(
        raw=RawDetection(
            box=BoundingBox(300, 300, 620, 700),
            confidence=0.9,
            landmarks=Landmarks5(
                np.array(
                    [[380, 420], [540, 420], [460, 500], [400, 580], [520, 580]],
                    dtype=np.float32,
                )
            ),
            landmarks_derived=True,
        ),
        image_width=IMAGE[0],
        image_height=IMAGE[1],
    )
    face.visibility_score = 0.85
    with_pose(face, method="roll_only")

    verdict = apply(policy, [face])

    assert verdict.passed
    assert any(w.code == "LANDMARKS_DERIVED" for w in verdict.warnings)
