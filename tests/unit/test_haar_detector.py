"""The Haar terminal fallback.

This adapter is the service's floor: it ships inside the wheel, needs no
download and no GPU, and is what stands between a wiped model volume and a
total outage. It is therefore worth testing carefully despite being the weakest
detector in the chain.

Two properties matter most:

* it produces *something* on a real face, and
* it is honest about what it does not know — specifically, that its landmarks
  are constructed from an eye pair rather than predicted, which is why they
  must be flagged and must not feed a PnP pose solve.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import PoseConfig
from hamqadam_ai.detectors.haar import HaarCascadeDetector, _weight_to_confidence
from hamqadam_ai.detectors.pose import PoseEstimator

BgrImage = npt.NDArray[np.uint8]


@pytest.fixture(scope="module")
def detector() -> HaarCascadeDetector:
    """The detector built from the vendored cascades."""
    return HaarCascadeDetector()


@pytest.fixture(scope="module")
def portrait() -> BgrImage:
    """A real photograph of a real, frontal face."""
    matplotlib = pytest.importorskip("matplotlib")
    path = (
        Path(matplotlib.__file__).parent
        / "mpl-data"
        / "sample_data"
        / "grace_hopper.jpg"
    )
    if not path.is_file():
        pytest.skip("matplotlib sample portrait unavailable")
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- #
# Availability — the whole reason this adapter exists
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_detector_constructs_with_no_model_store(detector: HaarCascadeDetector) -> None:
    """No download, no network, no GPU, no model_store directory."""
    assert detector.name == "haar"
    assert detector.version


@pytest.mark.unit
def test_cascades_are_vendored_inside_the_package() -> None:
    """Relying on cv2.data would tie us to the wheel's layout."""
    from hamqadam_ai.detectors.cascades import cascade_path

    path = cascade_path("haarcascade_frontalface_default.xml")
    assert path.is_file()
    assert path.stat().st_size > 100_000


# --------------------------------------------------------------------------- #
# Detection on a real face
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_finds_a_real_frontal_face(
    detector: HaarCascadeDetector, portrait: BgrImage
) -> None:
    detections = detector.detect(portrait)
    assert len(detections) >= 1

    best = detections[0]
    height, width = portrait.shape[:2]
    assert 0 <= best.box.x1 < best.box.x2 <= width
    assert 0 <= best.box.y1 < best.box.y2 <= height
    # A face box should be a meaningful fraction of a portrait.
    assert best.box.area / (width * height) > 0.01


@pytest.mark.unit
def test_finds_nothing_in_noise(detector: HaarCascadeDetector) -> None:
    rng = np.random.default_rng(19)
    noise = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert detector.detect(noise) == []


@pytest.mark.unit
def test_results_are_sorted_by_confidence(
    detector: HaarCascadeDetector, portrait: BgrImage
) -> None:
    height, width = portrait.shape[:2]
    pair = np.zeros((height, width * 2, 3), dtype=np.uint8)
    pair[:, :width] = portrait
    pair[:, width:] = cv2.GaussianBlur(portrait, (7, 7), 0)

    detections = detector.detect(pair)
    scores = [d.confidence for d in detections]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_confidences_are_bounded(
    detector: HaarCascadeDetector, portrait: BgrImage
) -> None:
    for detection in detector.detect(portrait):
        assert 0.0 <= detection.confidence <= 1.0


@pytest.mark.unit
def test_wrong_input_rank_is_rejected(detector: HaarCascadeDetector) -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        detector.detect(np.zeros((100, 100), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# The pseudo-confidence mapping
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_stage_weight_maps_monotonically_to_confidence() -> None:
    """Viola-Jones emits no probability; the mapping must at least rank."""
    values = [_weight_to_confidence(w) for w in (-2.0, 0.0, 2.6, 5.0, 9.0)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


@pytest.mark.unit
def test_midpoint_weight_maps_to_one_half() -> None:
    assert _weight_to_confidence(2.6) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.unit
def test_a_marginal_detection_lands_below_the_policy_floor() -> None:
    """Deliberate: weak Haar hits must not auto-pass the 0.60 default floor."""
    assert _weight_to_confidence(1.0) < 0.60


# --------------------------------------------------------------------------- #
# Derived landmarks — the honesty requirement
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_landmarks_when_present_are_flagged_as_derived(
    detector: HaarCascadeDetector, portrait: BgrImage
) -> None:
    for detection in detector.detect(portrait):
        if detection.landmarks is not None:
            assert detection.landmarks_derived is True
            assert detection.has_predicted_landmarks is False


@pytest.mark.unit
def test_constructed_points_place_the_eyes_exactly() -> None:
    """The eyes are measured; only the nose and mouth are template-derived."""
    left = (100.0, 200.0)
    right = (160.0, 206.0)
    landmarks = HaarCascadeDetector._construct_from_eyes(left, right)  # noqa: SLF001

    assert landmarks.left_eye == pytest.approx(left, abs=0.5)
    assert landmarks.right_eye == pytest.approx(right, abs=0.5)


@pytest.mark.unit
def test_constructed_points_are_anatomically_ordered() -> None:
    landmarks = HaarCascadeDetector._construct_from_eyes(  # noqa: SLF001
        (100.0, 200.0), (160.0, 200.0)
    )
    assert landmarks.is_plausible()
    assert landmarks.nose_tip[1] > landmarks.eye_center[1]
    assert landmarks.mouth_center[1] > landmarks.nose_tip[1]
    assert landmarks.mouth_left[0] < landmarks.mouth_right[0]


@pytest.mark.unit
def test_construction_follows_the_eye_line_rotation() -> None:
    """A tilted eye pair must produce a tilted face, not an upright one."""
    tilted = HaarCascadeDetector._construct_from_eyes(  # noqa: SLF001
        (100.0, 200.0), (150.0, 240.0)
    )
    assert tilted.roll_degrees == pytest.approx(38.66, abs=1.0)


@pytest.mark.unit
def test_construction_scales_with_eye_separation() -> None:
    near = HaarCascadeDetector._construct_from_eyes((100.0, 200.0), (140.0, 200.0))  # noqa: SLF001
    far = HaarCascadeDetector._construct_from_eyes((100.0, 200.0), (180.0, 200.0))  # noqa: SLF001

    near_span = near.mouth_center[1] - near.eye_center[1]
    far_span = far.mouth_center[1] - far.eye_center[1]
    assert far_span == pytest.approx(near_span * 2.0, rel=0.02)


@pytest.mark.unit
def test_derived_landmarks_never_reach_a_pnp_solve() -> None:
    """A template nose tip would report a confident frontal pose for any face.

    This is the specific dishonesty the ``landmarks_derived`` flag exists to
    prevent, so it is asserted directly.
    """
    estimator = PoseEstimator(PoseConfig())
    landmarks = HaarCascadeDetector._construct_from_eyes(  # noqa: SLF001
        (100.0, 200.0), (160.0, 214.0)
    )

    derived = estimator.estimate(
        landmarks, (640, 480), landmarks_derived=True
    )
    assert derived.method == "roll_only"
    assert derived.yaw == 0.0
    assert derived.pitch == 0.0
    assert derived.roll != 0.0

    # The same points *not* flagged would have been fed to PnP.
    undeclared = estimator.estimate(landmarks, (640, 480), landmarks_derived=False)
    assert undeclared.method != "roll_only"


# --------------------------------------------------------------------------- #
# Landmark recovery guards
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_recovery_returns_none_without_an_eye_cascade(
    portrait: BgrImage, tmp_path: Path
) -> None:
    """Landmark recovery is optional; its absence must degrade, not crash."""
    detector = HaarCascadeDetector()
    detector._eye_cascade = None  # noqa: SLF001
    detector.provides_landmarks = False

    detections = detector.detect(portrait)
    assert all(d.landmarks is None for d in detections)


@pytest.mark.unit
def test_recovery_rejects_an_implausible_eye_pair(
    detector: HaarCascadeDetector,
) -> None:
    """A flat grey box has no eyes; the cascade must not invent a pair."""
    from hamqadam_ai.utils.geometry import BoundingBox

    grey = np.full((300, 300), 128, dtype=np.uint8)
    assert detector._recover_landmarks(  # noqa: SLF001
        grey, BoundingBox(50, 50, 250, 250)
    ) is None


@pytest.mark.unit
def test_recovery_handles_a_degenerate_roi(detector: HaarCascadeDetector) -> None:
    from hamqadam_ai.utils.geometry import BoundingBox

    gray = np.zeros((100, 100), dtype=np.uint8)
    assert detector._recover_landmarks(  # noqa: SLF001
        gray, BoundingBox(0, 0, 4, 4)
    ) is None


@pytest.mark.unit
def test_close_releases_the_cascades() -> None:
    detector = HaarCascadeDetector()
    detector.close()
    assert detector._cascade is None  # noqa: SLF001
