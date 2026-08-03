"""Unit tests for the geometric primitives.

These underpin every detector, so their edge cases are tested exhaustively:
a wrong IoU or an off-by-one in the anchor grid produces subtly wrong boxes
everywhere rather than an obvious failure anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from hamqadam_ai.utils.geometry import (
    BoundingBox,
    Landmarks5,
    batched_nms,
    build_anchor_centers,
    distance2bbox,
    distance2kps,
    iou_matrix,
    nms,
    point_in_polygon_area,
)

# --------------------------------------------------------------------------- #
# BoundingBox
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestBoundingBox:
    """Behaviour of the axis-aligned box value object."""

    def test_basic_geometry(self) -> None:
        box = BoundingBox(10, 20, 110, 170)

        assert box.width == 100
        assert box.height == 150
        assert box.area == 15000
        assert box.center == (60.0, 95.0)
        assert box.short_side == 100
        assert box.long_side == 150
        assert box.aspect_ratio == pytest.approx(100 / 150)

    def test_degenerate_box_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Degenerate"):
            BoundingBox(100, 10, 50, 60)

    def test_zero_area_box_is_permitted(self) -> None:
        """A zero-width box is degenerate but not inverted; clipping produces
        them legitimately when a face lies entirely outside the frame."""
        box = BoundingBox(50, 50, 50, 50)
        assert box.area == 0.0

    def test_clip_constrains_to_frame(self) -> None:
        box = BoundingBox(-20, -30, 150, 200).clip(100, 100)

        assert box.as_tuple() == (0.0, 0.0, 100.0, 100.0)

    def test_expand_grows_about_the_centre(self) -> None:
        box = BoundingBox(100, 100, 200, 200).expand(0.20)

        assert box.center == pytest.approx((150.0, 150.0))
        assert box.width == pytest.approx(120.0)

    def test_to_square_preserves_centre_and_uses_long_side(self) -> None:
        box = BoundingBox(0, 0, 100, 200).to_square()

        assert box.width == pytest.approx(200.0)
        assert box.height == pytest.approx(200.0)
        assert box.center == pytest.approx((50.0, 100.0))

    @pytest.mark.parametrize(
        ("other", "expected"),
        [
            ((10, 10, 110, 110), 1.0),  # identical
            ((200, 200, 300, 300), 0.0),  # disjoint
            ((60, 60, 160, 160), 2500 / 17500),  # partial
            ((110, 10, 210, 110), 0.0),  # edge-touching, exclusive bounds
        ],
    )
    def test_iou(self, other: tuple[float, ...], expected: float) -> None:
        box = BoundingBox(10, 10, 110, 110)

        assert box.iou(BoundingBox(*other)) == pytest.approx(expected)

    def test_truncation_ratio(self) -> None:
        """Half the box outside the frame means a truncation ratio of 0.5."""
        box = BoundingBox(50, 0, 150, 100)

        assert box.truncation_ratio(100, 100) == pytest.approx(0.5)
        assert box.truncation_ratio(200, 200) == pytest.approx(0.0)

    def test_int_tuple_rounds_outward(self) -> None:
        """Outward rounding guarantees the crop contains the whole box."""
        box = BoundingBox(10.4, 20.6, 110.2, 170.9)

        assert box.as_int_tuple() == (10, 20, 111, 171)

    def test_roundtrip_through_xywh(self) -> None:
        original = BoundingBox(10, 20, 110, 170)
        rebuilt = BoundingBox.from_xywh(*original.as_xywh())

        assert rebuilt.as_tuple() == original.as_tuple()


# --------------------------------------------------------------------------- #
# Landmarks5
# --------------------------------------------------------------------------- #


def _valid_points() -> np.ndarray:
    return np.array(
        [[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]],
        dtype=np.float32,
    )


@pytest.mark.unit
class TestLandmarks5:
    """Behaviour of the five-point landmark value object."""

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=r"\(5, 2\)"):
            Landmarks5(np.zeros((3, 2), dtype=np.float32))

    def test_named_accessors(self) -> None:
        marks = Landmarks5(_valid_points())

        assert marks.left_eye == pytest.approx((38.3, 51.7), abs=1e-4)
        assert marks.right_eye == pytest.approx((73.5, 51.5), abs=1e-4)
        assert marks.nose_tip == pytest.approx((56.0, 71.7), abs=1e-4)

    def test_interocular_distance(self) -> None:
        marks = Landmarks5(_valid_points())

        assert marks.interocular_distance == pytest.approx(35.2006, abs=1e-3)

    def test_roll_from_level_eyes_is_zero(self) -> None:
        marks = Landmarks5(
            np.array(
                [[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]], dtype=np.float32
            )
        )

        assert marks.roll_degrees == pytest.approx(0.0)

    def test_roll_positive_when_right_eye_lower(self) -> None:
        marks = Landmarks5(
            np.array(
                [[40, 40], [80, 60], [60, 70], [45, 90], [75, 95]], dtype=np.float32
            )
        )

        assert marks.roll_degrees > 0

    def test_plausibility_accepts_a_real_configuration(self) -> None:
        assert Landmarks5(_valid_points()).is_plausible()

    @pytest.mark.parametrize(
        ("points", "reason"),
        [
            (
                [[40, 90], [80, 90], [60, 70], [45, 50], [75, 50]],
                "mouth above eyes",
            ),
            (
                [[50, 50], [50, 50], [50, 50], [50, 50], [50, 50]],
                "all points collapsed",
            ),
            (
                [[40, 50], [80, 50], [60, 200], [45, 90], [75, 90]],
                "nose far below mouth",
            ),
        ],
    )
    def test_plausibility_rejects_impossible_configurations(
        self, points: list[list[float]], reason: str
    ) -> None:
        marks = Landmarks5(np.array(points, dtype=np.float32))

        assert not marks.is_plausible(), reason

    def test_box_relative_coordinates(self) -> None:
        marks = Landmarks5(
            np.array(
                [[110, 120], [190, 120], [150, 150], [120, 180], [180, 180]],
                dtype=np.float32,
            )
        )
        relative = marks.to_box_relative(BoundingBox(100, 100, 200, 200))

        assert relative[0].tolist() == pytest.approx([0.1, 0.2])
        assert relative[1].tolist() == pytest.approx([0.9, 0.2])


# --------------------------------------------------------------------------- #
# NMS
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestNms:
    """Non-maximum suppression."""

    def test_suppresses_overlapping_keeps_distinct(self) -> None:
        boxes = np.array(
            [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        assert nms(boxes, scores, 0.4).tolist() == [0, 2]

    def test_returns_indices_in_descending_score_order(self) -> None:
        boxes = np.array(
            [[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110]], dtype=np.float32
        )
        scores = np.array([0.3, 0.9, 0.6], dtype=np.float32)

        assert nms(boxes, scores, 0.5).tolist() == [1, 2, 0]

    def test_empty_input(self) -> None:
        result = nms(
            np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), 0.5
        )

        assert result.size == 0

    def test_top_k_limits_candidates(self) -> None:
        boxes = np.array([[i * 100, 0, i * 100 + 10, 10] for i in range(10)], dtype=np.float32)
        scores = np.linspace(0.1, 1.0, 10, dtype=np.float32)

        assert len(nms(boxes, scores, 0.5, top_k=3)) == 3

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="disagree"):
            nms(
                np.zeros((3, 4), dtype=np.float32),
                np.zeros((2,), dtype=np.float32),
                0.5,
            )

    def test_threshold_of_one_suppresses_nothing(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)

        assert len(nms(boxes, scores, 1.0)) == 2

    def test_batched_nms_does_not_suppress_across_classes(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)

        assert len(batched_nms(boxes, scores, np.array([0, 1]), 0.3)) == 2
        assert len(batched_nms(boxes, scores, np.array([0, 0]), 0.3)) == 1


@pytest.mark.unit
def test_iou_matrix_matches_pairwise_iou() -> None:
    """The vectorised matrix must agree with the scalar method."""
    a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)

    matrix = iou_matrix(a, b)

    assert matrix.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            expected = BoundingBox.from_array(a[i]).iou(BoundingBox.from_array(b[j]))
            assert matrix[i, j] == pytest.approx(expected, abs=1e-5)


# --------------------------------------------------------------------------- #
# Anchor-free decoding
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestAnchorDecoding:
    """SCRFD's distance-based box and keypoint parameterisation."""

    def test_distance2bbox(self) -> None:
        centers = np.array([[50, 50]], dtype=np.float32)
        distances = np.array([[10, 20, 30, 40]], dtype=np.float32)

        assert distance2bbox(centers, distances)[0].tolist() == pytest.approx(
            [40.0, 30.0, 80.0, 90.0]
        )

    def test_distance2bbox_clamps_to_shape(self) -> None:
        centers = np.array([[10, 10]], dtype=np.float32)
        distances = np.array([[50, 50, 500, 500]], dtype=np.float32)

        result = distance2bbox(centers, distances, max_shape=(100, 100))

        assert result[0].tolist() == pytest.approx([0.0, 0.0, 100.0, 100.0])

    def test_distance2kps(self) -> None:
        centers = np.array([[50, 50]], dtype=np.float32)
        offsets = np.array([[-10, -5, 10, -5, 0, 5, -8, 15, 8, 15]], dtype=np.float32)

        keypoints = distance2kps(centers, offsets)

        assert keypoints.shape == (1, 5, 2)
        assert keypoints[0, 0].tolist() == pytest.approx([40.0, 45.0])
        assert keypoints[0, 2].tolist() == pytest.approx([50.0, 55.0])

    def test_distance2kps_rejects_odd_width(self) -> None:
        with pytest.raises(ValueError, match=r"\(N, 2K\)"):
            distance2kps(
                np.array([[0, 0]], dtype=np.float32),
                np.array([[1, 2, 3]], dtype=np.float32),
            )

    def test_anchor_grid_shape_and_ordering(self) -> None:
        """Two anchors per cell must be adjacent, matching the flattened output."""
        anchors = build_anchor_centers(2, 3, stride=8, num_anchors=2)

        assert anchors.shape == (12, 2)
        # Cell (0,0) at x=0,y=0 occupies rows 0 and 1.
        assert anchors[0].tolist() == [0.0, 0.0]
        assert anchors[1].tolist() == [0.0, 0.0]
        # Cell (0,1) at x=8,y=0 occupies rows 2 and 3.
        assert anchors[2].tolist() == [8.0, 0.0]
        # Row 1 starts at index 6 with y=8.
        assert anchors[6].tolist() == [0.0, 8.0]

    def test_anchor_grid_covers_the_input(self) -> None:
        """A 640-pixel input at stride 32 yields a 20x20 grid."""
        anchors = build_anchor_centers(20, 20, stride=32, num_anchors=2)

        assert anchors.shape == (800, 2)
        assert float(anchors[:, 0].max()) == pytest.approx(19 * 32)


@pytest.mark.unit
def test_polygon_area_shoelace() -> None:
    """A unit square has area 1 regardless of winding direction."""
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    assert point_in_polygon_area(square) == pytest.approx(1.0)
    assert point_in_polygon_area(square[::-1]) == pytest.approx(1.0)
