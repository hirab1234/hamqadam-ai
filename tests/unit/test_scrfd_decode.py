"""SCRFD post-processing, exercised against synthetic network output.

The decoder is the part of the detector most likely to be silently wrong: an
off-by-one in the anchor grid, a forgotten stride multiplication or a
transposed keypoint block all produce boxes that are *plausible* rather than
obviously broken, and no amount of eyeballing a demo image reliably catches a
two-pixel systematic bias.

So the decoder is tested by construction. A fake model emits tensors encoding
a box whose position is known exactly, and the test asserts the decoder
recovers it to sub-pixel accuracy. No weights, no network, no images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import ModelSpec, get_settings
from hamqadam_ai.detectors.scrfd import ScrfdDetector
from hamqadam_ai.models.base import ModelHandle
from hamqadam_ai.utils.geometry import build_anchor_centers

INPUT_SIZE = (640, 640)
STRIDES = (8, 16, 32)
NUM_ANCHORS = 2


# --------------------------------------------------------------------------- #
# A fake LoadedModel that returns tensors we constructed by hand
# --------------------------------------------------------------------------- #


@dataclass
class FakeScrfdModel:
    """Stands in for a loaded ONNX session, emitting scripted tensors."""

    outputs: list[npt.NDArray[np.float32]]
    spec: ModelSpec
    handle: ModelHandle = field(
        default_factory=lambda: ModelHandle(
            key="fake", version="fake-1.0", kind="onnx", device="cpu",
            providers=("CPUExecutionProvider",),
        )
    )
    executor: Any = None
    last_input: npt.NDArray[np.float32] | None = None

    @property
    def input_names(self) -> tuple[str, ...]:
        return ("input.1",)

    @property
    def output_names(self) -> tuple[str, ...]:
        # Nine outputs -> the fmc=3, num_anchors=2, use_kps=True topology.
        return tuple(f"out{i}" for i in range(9))

    @property
    def input_shapes(self) -> dict[str, tuple[int | str | None, ...]]:
        return {"input.1": (1, 3, "H", "W")}

    def run(self, inputs: dict[str, npt.NDArray[Any]]) -> list[npt.NDArray[Any]]:
        self.last_input = next(iter(inputs.values()))
        return list(self.outputs)


def blank_outputs() -> list[npt.NDArray[np.float32]]:
    """Nine all-zero tensors shaped for a 640x640 input."""
    width, height = INPUT_SIZE
    scores: list[npt.NDArray[np.float32]] = []
    boxes: list[npt.NDArray[np.float32]] = []
    keypoints: list[npt.NDArray[np.float32]] = []
    for stride in STRIDES:
        cells = (height // stride) * (width // stride) * NUM_ANCHORS
        scores.append(np.zeros((cells, 1), dtype=np.float32))
        boxes.append(np.zeros((cells, 4), dtype=np.float32))
        keypoints.append(np.zeros((cells, 10), dtype=np.float32))
    return [*scores, *boxes, *keypoints]


def plant_face(
    outputs: list[npt.NDArray[np.float32]],
    *,
    stride: int,
    cell_x: int,
    cell_y: int,
    box: tuple[float, float, float, float],
    score: float = 0.95,
    keypoints: npt.NDArray[np.float32] | None = None,
) -> tuple[float, float]:
    """Write a single detection into the synthetic tensors.

    Encodes ``box`` as the four edge distances from the chosen cell centre,
    divided by the stride, exactly as the network would.

    Returns:
        The anchor centre, so the test can assert against it.
    """
    level = STRIDES.index(stride)
    width, _ = INPUT_SIZE
    grid_width = width // stride

    row = (cell_y * grid_width + cell_x) * NUM_ANCHORS
    centre_x = float(cell_x * stride)
    centre_y = float(cell_y * stride)

    x1, y1, x2, y2 = box
    outputs[level][row, 0] = score
    outputs[level + 3][row] = np.array(
        [
            (centre_x - x1) / stride,
            (centre_y - y1) / stride,
            (x2 - centre_x) / stride,
            (y2 - centre_y) / stride,
        ],
        dtype=np.float32,
    )

    if keypoints is not None:
        offsets = (keypoints - np.array([centre_x, centre_y], dtype=np.float32)) / stride
        outputs[level + 6][row] = offsets.reshape(-1)

    return centre_x, centre_y


@pytest.fixture
def spec() -> ModelSpec:
    """The real SCRFD declaration from models.yaml."""
    return get_settings().model_spec("face_detector_scrfd")


def build_detector(outputs: list[npt.NDArray[np.float32]], spec: ModelSpec, **kwargs: Any):  # noqa: ANN201
    """Construct a detector wired to a fake model."""
    defaults: dict[str, Any] = {
        "score_threshold": 0.5,
        "nms_iou_threshold": 0.4,
        "input_size": INPUT_SIZE,
        "max_candidates": 200,
    }
    defaults.update(kwargs)
    return ScrfdDetector(FakeScrfdModel(outputs=outputs, spec=spec), spec, **defaults)


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_topology_is_inferred_from_the_output_count(spec: ModelSpec) -> None:
    detector = build_detector(blank_outputs(), spec)
    assert detector.provides_landmarks is True
    assert detector._strides == STRIDES  # noqa: SLF001
    assert detector._num_anchors == NUM_ANCHORS  # noqa: SLF001
    assert detector._fmc == 3  # noqa: SLF001


@pytest.mark.unit
def test_anchor_grid_row_count_matches_the_score_tensor(spec: ModelSpec) -> None:
    """A mismatch here is the classic source of silently shifted boxes."""
    detector = build_detector(blank_outputs(), spec)
    outputs = blank_outputs()
    for level, stride in enumerate(STRIDES):
        anchors = detector._anchor_cache[stride]  # noqa: SLF001
        assert anchors.shape[0] == outputs[level].shape[0]


@pytest.mark.unit
def test_anchor_centres_are_stride_spaced_and_anchor_repeated() -> None:
    anchors = build_anchor_centers(2, 3, stride=8, num_anchors=2)
    assert anchors.shape == (12, 2)
    # Each cell centre appears num_anchors times, consecutively.
    assert anchors[0].tolist() == [0.0, 0.0]
    assert anchors[1].tolist() == [0.0, 0.0]
    assert anchors[2].tolist() == [8.0, 0.0]
    assert anchors[6].tolist() == [0.0, 8.0]


# --------------------------------------------------------------------------- #
# Box decoding
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_planted_box_is_recovered_exactly(spec: ModelSpec) -> None:
    outputs = blank_outputs()
    expected = (200.0, 150.0, 328.0, 310.0)
    plant_face(outputs, stride=8, cell_x=33, cell_y=28, box=expected)

    detector = build_detector(outputs, spec)
    # A square image means the letterbox is an identity transform, so the
    # decoded coordinates are directly comparable with what was planted.
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = detector.detect(image)

    assert len(detections) == 1
    assert detections[0].box.as_tuple() == pytest.approx(expected, abs=0.51)
    assert detections[0].confidence == pytest.approx(0.95, abs=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("stride", STRIDES)
def test_every_fpn_level_decodes_correctly(spec: ModelSpec, stride: int) -> None:
    """A stride multiplication forgotten on one level only would slip through
    a test that exercised stride 8 alone."""
    outputs = blank_outputs()
    grid = 640 // stride
    cell_x = cell_y = grid // 2
    centre = float(cell_x * stride)
    expected = (centre - 60.0, centre - 75.0, centre + 60.0, centre + 75.0)
    plant_face(outputs, stride=stride, cell_x=cell_x, cell_y=cell_y, box=expected)

    detector = build_detector(outputs, spec)
    detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].box.as_tuple() == pytest.approx(expected, abs=0.51)


@pytest.mark.unit
def test_scores_below_the_threshold_are_discarded(spec: ModelSpec) -> None:
    outputs = blank_outputs()
    plant_face(
        outputs, stride=8, cell_x=40, cell_y=40, box=(280.0, 280.0, 360.0, 380.0),
        score=0.31,
    )
    detector = build_detector(outputs, spec, score_threshold=0.5)
    assert detector.detect(np.zeros((640, 640, 3), dtype=np.uint8)) == []


@pytest.mark.unit
def test_overlapping_duplicates_are_suppressed(spec: ModelSpec) -> None:
    outputs = blank_outputs()
    plant_face(
        outputs, stride=8, cell_x=40, cell_y=40, box=(280.0, 280.0, 380.0, 400.0),
        score=0.95,
    )
    plant_face(
        outputs, stride=8, cell_x=41, cell_y=40, box=(284.0, 283.0, 384.0, 403.0),
        score=0.90,
    )
    detector = build_detector(outputs, spec)
    detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    assert len(detections) == 1, "near-identical boxes should collapse to one"
    assert detections[0].confidence == pytest.approx(0.95, abs=1e-5)


@pytest.mark.unit
def test_distinct_faces_both_survive(spec: ModelSpec) -> None:
    outputs = blank_outputs()
    plant_face(outputs, stride=8, cell_x=15, cell_y=30, box=(80.0, 200.0, 180.0, 330.0))
    plant_face(
        outputs, stride=8, cell_x=55, cell_y=30, box=(400.0, 200.0, 500.0, 330.0),
        score=0.88,
    )
    detector = build_detector(outputs, spec)
    detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    assert len(detections) == 2
    # Sorted by descending confidence.
    assert detections[0].confidence > detections[1].confidence


@pytest.mark.unit
def test_detections_are_clipped_to_the_frame(spec: ModelSpec) -> None:
    outputs = blank_outputs()
    plant_face(outputs, stride=8, cell_x=4, cell_y=4, box=(-90.0, -70.0, 110.0, 140.0))
    detector = build_detector(outputs, spec)
    detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    assert len(detections) == 1
    box = detections[0].box
    assert box.x1 >= 0.0
    assert box.y1 >= 0.0


# --------------------------------------------------------------------------- #
# Keypoint decoding
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_planted_keypoints_are_recovered(spec: ModelSpec) -> None:
    outputs = blank_outputs()
    box = (240.0, 200.0, 400.0, 400.0)
    keypoints = np.array(
        [
            [285.0, 270.0],  # left eye
            [355.0, 270.0],  # right eye
            [320.0, 310.0],  # nose
            [292.0, 355.0],  # mouth left
            [348.0, 355.0],  # mouth right
        ],
        dtype=np.float32,
    )
    plant_face(
        outputs, stride=8, cell_x=40, cell_y=37, box=box, keypoints=keypoints
    )

    detector = build_detector(outputs, spec)
    detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].landmarks is not None
    assert detections[0].landmarks.points == pytest.approx(keypoints, abs=0.51)
    assert detections[0].landmarks_derived is False


@pytest.mark.unit
def test_implausible_keypoints_are_dropped_but_the_box_survives(
    spec: ModelSpec,
) -> None:
    """Detectors emit garbage landmarks on textured non-face regions.

    Dropping the landmarks rather than the detection keeps the box while
    preventing a nonsensical pose solve downstream.
    """
    outputs = blank_outputs()
    box = (240.0, 200.0, 400.0, 400.0)
    upside_down = np.array(
        [
            [285.0, 370.0],  # "eyes" below the "mouth"
            [355.0, 370.0],
            [320.0, 310.0],
            [292.0, 255.0],
            [348.0, 255.0],
        ],
        dtype=np.float32,
    )
    plant_face(
        outputs, stride=8, cell_x=40, cell_y=37, box=box, keypoints=upside_down
    )

    detector = build_detector(outputs, spec)
    detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].landmarks is None


# --------------------------------------------------------------------------- #
# Pre-processing contract
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_blob_matches_the_declared_normalisation(spec: ModelSpec) -> None:
    """SCRFD expects (x - 127.5) / 128 on RGB, in NCHW."""
    model = FakeScrfdModel(outputs=blank_outputs(), spec=spec)
    detector = ScrfdDetector(
        model,
        spec,
        score_threshold=0.5,
        nms_iou_threshold=0.4,
        input_size=INPUT_SIZE,
    )
    image = np.full((640, 640, 3), 255, dtype=np.uint8)
    detector.detect(image)

    blob = model.last_input
    assert blob is not None
    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert float(blob.max()) == pytest.approx((255.0 - 127.5) / 128.0, abs=1e-6)


@pytest.mark.unit
def test_non_square_input_is_letterboxed_top_left(spec: ModelSpec) -> None:
    """InsightFace pads bottom-right with zeros; centred padding would bias
    every box by half the pad width."""
    model = FakeScrfdModel(outputs=blank_outputs(), spec=spec)
    detector = ScrfdDetector(
        model, spec, score_threshold=0.5, nms_iou_threshold=0.4, input_size=INPUT_SIZE
    )
    detector.detect(np.full((480, 640, 3), 200, dtype=np.uint8))

    blob = model.last_input
    assert blob is not None
    # 640x480 scales to 640x480 inside a 640x640 canvas; rows 480+ are padding.
    padded_region = blob[0, :, 500:, :]
    expected_pad = (0.0 - 127.5) / 128.0
    assert float(padded_region.min()) == pytest.approx(expected_pad, abs=1e-6)
    assert float(padded_region.max()) == pytest.approx(expected_pad, abs=1e-6)


@pytest.mark.unit
def test_coordinates_are_unmapped_from_the_letterbox(spec: ModelSpec) -> None:
    """A face planted in network space must come back in *source* coordinates."""
    outputs = blank_outputs()
    # Source is 1280x960; scale into 640x640 is 0.5.
    network_box = (200.0, 150.0, 300.0, 280.0)
    plant_face(outputs, stride=8, cell_x=31, cell_y=27, box=network_box)

    detector = build_detector(outputs, spec)
    detections = detector.detect(np.zeros((960, 1280, 3), dtype=np.uint8))

    assert len(detections) == 1
    expected_source = tuple(value / 0.5 for value in network_box)
    assert detections[0].box.as_tuple() == pytest.approx(expected_source, abs=1.1)


@pytest.mark.unit
def test_no_detections_returns_an_empty_list_not_an_error(spec: ModelSpec) -> None:
    """'No face here' is a normal result, not an exception."""
    detector = build_detector(blank_outputs(), spec)
    assert detector.detect(np.zeros((640, 640, 3), dtype=np.uint8)) == []


@pytest.mark.unit
def test_wrong_input_rank_is_rejected(spec: ModelSpec) -> None:
    detector = build_detector(blank_outputs(), spec)
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        detector.detect(np.zeros((640, 640), dtype=np.uint8))


@pytest.mark.unit
def test_candidate_cap_is_enforced(spec: ModelSpec) -> None:
    """A DoS guard: an adversarial image must not make NMS quadratic."""
    outputs = blank_outputs()
    for index in range(60):
        plant_face(
            outputs,
            stride=32,
            cell_x=(index % 10) * 2,
            cell_y=(index // 10) * 2,
            box=(
                float((index % 10) * 64),
                float((index // 10) * 64),
                float((index % 10) * 64 + 50),
                float((index // 10) * 64 + 60),
            ),
            score=0.9,
        )
    detector = build_detector(outputs, spec, max_candidates=5)
    assert len(detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))) <= 5
