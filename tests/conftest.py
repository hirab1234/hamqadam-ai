"""Shared pytest fixtures.

The synthetic-face generator here deserves an explanation. Unit tests must run
with no model weights, no network and no real identity documents - a repository
containing sample CNICs would be a data-protection incident waiting to happen.
So the geometry-dependent components (pose, occlusion, visibility, policy) are
tested against *synthetic* faces: procedurally drawn images whose ground-truth
landmarks are known exactly, and which can be perturbed deterministically to
simulate occlusion, blur and darkness.

That makes these tests fast, hermetic and reproducible. Tests that genuinely
need real weights are marked ``integration`` and skip automatically when the
model store is empty.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import Settings, get_settings
from hamqadam_ai.core.constants import CANONICAL_FACE_3D_5PT
from hamqadam_ai.detectors.base import DetectedFace, RawDetection
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

BgrImage = npt.NDArray[np.uint8]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def settings() -> Settings:
    """The real, fully-validated service configuration."""
    return get_settings()


@pytest.fixture(scope="session")
def model_store(settings: Settings) -> Path:
    """Path to the model store."""
    return settings.storage.resolved_model_dir


@pytest.fixture(scope="session")
def has_scrfd(settings: Settings, model_store: Path) -> bool:
    """Whether the SCRFD artefact is present, gating integration tests."""
    spec = settings.models.get("face_detector_scrfd")
    return spec is not None and (model_store / spec.path).is_file()


@pytest.fixture
def requires_scrfd(has_scrfd: bool) -> None:
    """Skip a test when the SCRFD weights have not been downloaded."""
    if not has_scrfd:
        pytest.skip("SCRFD weights absent; run scripts/download_models.py")


# --------------------------------------------------------------------------- #
# Synthetic face generation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SyntheticFace:
    """A procedurally drawn face with exactly known ground truth.

    Attributes:
        image: The rendered BGR image.
        landmarks: True landmark positions.
        box: True face bounding box.
        yaw: True yaw in degrees.
        pitch: True pitch in degrees.
        roll: True roll in degrees.
    """

    image: BgrImage
    landmarks: Landmarks5
    box: BoundingBox
    yaw: float
    pitch: float
    roll: float

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of the rendered image."""
        return (int(self.image.shape[1]), int(self.image.shape[0]))


def _project_landmarks(
    yaw: float, pitch: float, roll: float, image_size: int, distance: float
) -> npt.NDArray[np.float32]:
    """Project the canonical 3D face model at a known orientation."""
    yaw_r = math.radians(-yaw)
    pitch_r = math.radians(-pitch)
    roll_r = math.radians(roll)

    rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(pitch_r), -math.sin(pitch_r)],
            [0, math.sin(pitch_r), math.cos(pitch_r)],
        ]
    )
    ry = np.array(
        [
            [math.cos(yaw_r), 0, math.sin(yaw_r)],
            [0, 1, 0],
            [-math.sin(yaw_r), 0, math.cos(yaw_r)],
        ]
    )
    rz = np.array(
        [
            [math.cos(roll_r), -math.sin(roll_r), 0],
            [math.sin(roll_r), math.cos(roll_r), 0],
            [0, 0, 1],
        ]
    )

    rvec, _ = cv2.Rodrigues(rz @ ry @ rx)
    tvec = np.array([[0.0], [0.0], [distance]])
    camera = np.array(
        [
            [image_size, 0, image_size / 2],
            [0, image_size, image_size / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    points, _ = cv2.projectPoints(
        CANONICAL_FACE_3D_5PT, rvec, tvec, camera, np.zeros((4, 1))
    )
    return points.reshape(-1, 2).astype(np.float32)


def make_synthetic_face(
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    image_size: int = 512,
    distance: float = 420.0,
    skin_bgr: tuple[int, int, int] = (150, 175, 205),
    background_bgr: tuple[int, int, int] = (120, 120, 120),
    texture_strength: float = 14.0,
    seed: int = 7,
) -> SyntheticFace:
    """Render a face-like image with exactly known landmarks.

    Not photorealistic - it will not fool a neural detector - but it has the
    properties the geometric analysers actually measure: a textured skin-toned
    ellipse with darker, high-contrast eye, nostril and mouth features at
    landmark positions consistent with the 3D model. That is precisely enough
    to test pose recovery, occlusion detection and the visibility scorer
    against ground truth.

    Args:
        yaw: True yaw in degrees.
        pitch: True pitch in degrees.
        roll: True roll in degrees.
        image_size: Edge length of the square canvas.
        distance: Camera distance in model millimetres; smaller means a larger
            face in frame.
        skin_bgr: Base skin colour, chosen inside the YCrCb skin ellipse.
        background_bgr: Background colour, chosen outside it.
        texture_strength: Standard deviation of the skin's additive texture.
        seed: RNG seed, making every rendering deterministic.

    Returns:
        The rendered face and its ground truth.
    """
    rng = np.random.default_rng(seed)
    canvas = np.full((image_size, image_size, 3), background_bgr, dtype=np.uint8)

    points = _project_landmarks(yaw, pitch, roll, image_size, distance)
    landmarks = Landmarks5(points)

    interocular = landmarks.interocular_distance
    centre_x, centre_y = landmarks.eye_center
    # Head ellipse proportions: roughly 1.5 interocular wide, 2.0 tall, with
    # the eye line about 40% of the way down the head.
    axis_x = int(round(interocular * 1.15))
    axis_y = int(round(interocular * 1.55))
    head_centre = (int(round(centre_x)), int(round(centre_y + interocular * 0.45)))

    cv2.ellipse(
        canvas, head_centre, (axis_x, axis_y), roll, 0, 360, skin_bgr, thickness=-1
    )

    # Skin texture. Without it the gradient-energy signal is identically zero
    # and every region reads as occluded.
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.ellipse(mask, head_centre, (axis_x, axis_y), roll, 0, 360, 255, thickness=-1)
    noise = rng.normal(0.0, texture_strength, (image_size, image_size, 3))
    textured = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    canvas = np.where(mask[:, :, None] > 0, textured, canvas)

    eye_radius = max(3, int(interocular * 0.16))
    for eye in (landmarks.left_eye, landmarks.right_eye):
        centre = (int(round(eye[0])), int(round(eye[1])))
        # Sclera then iris: two concentric discs give the strong local gradient
        # a real eye produces.
        cv2.circle(canvas, centre, int(eye_radius * 1.5), (235, 235, 235), -1)
        cv2.circle(canvas, centre, eye_radius, (60, 45, 35), -1)
        cv2.circle(canvas, centre, max(1, eye_radius // 3), (15, 15, 15), -1)

    nose = landmarks.nose_tip
    nostril = max(2, int(interocular * 0.07))
    for offset in (-interocular * 0.11, interocular * 0.11):
        cv2.circle(
            canvas,
            (int(round(nose[0] + offset)), int(round(nose[1]))),
            nostril,
            (95, 105, 130),
            -1,
        )

    mouth_left = landmarks.mouth_left
    mouth_right = landmarks.mouth_right
    cv2.line(
        canvas,
        (int(round(mouth_left[0])), int(round(mouth_left[1]))),
        (int(round(mouth_right[0])), int(round(mouth_right[1]))),
        (70, 60, 110),
        thickness=max(2, int(interocular * 0.10)),
    )

    xs = points[:, 0]
    ys = points[:, 1]
    box = BoundingBox(
        x1=float(max(0.0, xs.min() - interocular * 0.65)),
        y1=float(max(0.0, ys.min() - interocular * 0.95)),
        x2=float(min(image_size, xs.max() + interocular * 0.65)),
        y2=float(min(image_size, ys.max() + interocular * 0.75)),
    )

    return SyntheticFace(
        image=canvas, landmarks=landmarks, box=box, yaw=yaw, pitch=pitch, roll=roll
    )


@pytest.fixture
def synthetic_face() -> SyntheticFace:
    """A frontal synthetic face."""
    return make_synthetic_face()


@pytest.fixture
def synthetic_face_factory():  # noqa: ANN201 - pytest factory fixture
    """Factory for synthetic faces with arbitrary pose and appearance."""
    return make_synthetic_face


# --------------------------------------------------------------------------- #
# Occlusion helpers
# --------------------------------------------------------------------------- #


def occlude_rectangle(
    image: BgrImage,
    centre: tuple[float, float],
    size: tuple[int, int],
    colour: tuple[int, int, int] = (40, 40, 40),
) -> BgrImage:
    """Paste a flat, untextured rectangle over part of an image.

    Flat and untextured is the point: it is what a mask, a lens or a sticker
    looks like to the gradient-energy and colour-uniformity signals.
    """
    result = image.copy()
    half_w, half_h = size[0] // 2, size[1] // 2
    x, y = int(centre[0]), int(centre[1])
    cv2.rectangle(
        result,
        (x - half_w, y - half_h),
        (x + half_w, y + half_h),
        colour,
        thickness=-1,
    )
    return result


@pytest.fixture
def occluder():  # noqa: ANN201 - pytest factory fixture
    """Factory that pastes a flat rectangle over an image region."""
    return occlude_rectangle


# --------------------------------------------------------------------------- #
# Detection value objects
# --------------------------------------------------------------------------- #


def make_detection(
    box: tuple[float, float, float, float],
    confidence: float = 0.95,
    landmarks: Landmarks5 | None = None,
    *,
    derived: bool = False,
) -> RawDetection:
    """Build a :class:`RawDetection` without running a model."""
    return RawDetection(
        box=BoundingBox(*box),
        confidence=confidence,
        landmarks=landmarks,
        landmarks_derived=derived,
    )


def make_face(
    box: tuple[float, float, float, float],
    *,
    confidence: float = 0.95,
    image_size: tuple[int, int] = (512, 512),
    visibility: float = 0.9,
    landmarks: Landmarks5 | None = None,
) -> DetectedFace:
    """Build a fully-formed :class:`DetectedFace` for policy tests."""
    face = DetectedFace(
        raw=make_detection(box, confidence, landmarks),
        image_width=image_size[0],
        image_height=image_size[1],
    )
    face.visibility_score = visibility
    face.visibility_components = {
        "detector_confidence": confidence,
        "face_size": 1.0,
        "framing": 1.0,
    }
    face.visibility_weights = {
        "detector_confidence": 0.34,
        "face_size": 0.33,
        "framing": 0.33,
    }
    return face


@pytest.fixture
def detection_factory():  # noqa: ANN201 - pytest factory fixture
    """Factory for :class:`RawDetection` objects."""
    return make_detection


@pytest.fixture
def face_factory():  # noqa: ANN201 - pytest factory fixture
    """Factory for :class:`DetectedFace` objects."""
    return make_face


# --------------------------------------------------------------------------- #
# Filesystem
# --------------------------------------------------------------------------- #


@pytest.fixture
def temp_root(tmp_path: Path) -> Iterator[Path]:
    """An isolated scratch root for storage tests."""
    root = tmp_path / "scratch"
    root.mkdir()
    yield root

# --------------------------------------------------------------------------- #
# Gallery isolation
# --------------------------------------------------------------------------- #


def pytest_configure(config: pytest.Config) -> None:
    """Keep the duplicate gallery out of the tests.

    The shipped default is an **on-disk** Qdrant, which is right for a
    deployment and wrong for a test run: a face enrolled by one test would be
    found as a duplicate by the next, and by every future run of the suite. The
    tests would pass or fail depending on what an earlier run happened to
    write.

    Pinned to the in-process store, which starts empty every time. The Qdrant
    adapter itself is still covered - `test_vector_store_contract.py` and
    `test_duplicate_service.py` exercise it directly against an embedded
    instance, which is the right place to test a storage adapter.

    Set before any test imports settings, so the singleton is built with it.
    """
    os.environ.setdefault("HQ_DUPLICATE__BACKEND", "memory")
