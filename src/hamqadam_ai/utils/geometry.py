"""Geometric primitives: boxes, landmarks, IoU, NMS and anchor decoding.

All coordinates are in **source-image pixel space**, floating point, with the
origin at the top-left corner and ``x2``/``y2`` exclusive. Detector adapters
are responsible for mapping their network-space outputs back into this space
before returning; every consumer downstream can then assume one convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.constants import EPSILON, FIVE_POINT_LANDMARK_NAMES

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned rectangle in source-image pixel coordinates.

    Attributes:
        x1: Left edge, inclusive.
        y1: Top edge, inclusive.
        x2: Right edge, exclusive.
        y2: Bottom edge, exclusive.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"Degenerate bounding box: ({self.x1}, {self.y1}) -> ({self.x2}, {self.y2})"
            )

    # -- Derived geometry ------------------------------------------------- #

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Centre point as ``(x, y)``."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def short_side(self) -> float:
        """Length of the shorter edge - the limiting factor for recognition."""
        return min(self.width, self.height)

    @property
    def long_side(self) -> float:
        """Length of the longer edge."""
        return max(self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height. 1.0 is square."""
        return self.width / max(self.height, EPSILON)

    # -- Transforms ------------------------------------------------------- #

    def clip(self, width: int, height: int) -> BoundingBox:
        """Return a copy clamped to the bounds of a ``width`` x ``height`` image."""
        return BoundingBox(
            x1=max(0.0, min(self.x1, float(width))),
            y1=max(0.0, min(self.y1, float(height))),
            x2=max(0.0, min(self.x2, float(width))),
            y2=max(0.0, min(self.y2, float(height))),
        )

    def scale(self, factor_x: float, factor_y: float | None = None) -> BoundingBox:
        """Scale the coordinates about the image origin."""
        fy = factor_x if factor_y is None else factor_y
        return BoundingBox(self.x1 * factor_x, self.y1 * fy, self.x2 * factor_x, self.y2 * fy)

    def translate(self, dx: float, dy: float) -> BoundingBox:
        """Shift the box by ``(dx, dy)`` pixels."""
        return BoundingBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def expand(self, ratio: float) -> BoundingBox:
        """Grow the box outwards by ``ratio`` of its size, about its centre.

        A detector box crops tightly to the face; recognition and occlusion
        analysis both want a little context (hair line, chin, ears), so callers
        typically expand by 0.15-0.30 before cropping.
        """
        dx = self.width * ratio / 2.0
        dy = self.height * ratio / 2.0
        return BoundingBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def to_square(self) -> BoundingBox:
        """Return the smallest centred square containing this box.

        Squaring before resizing to a fixed network input preserves the face's
        aspect ratio, which matters: a horizontally stretched face measurably
        degrades ArcFace similarity.
        """
        cx, cy = self.center
        half = self.long_side / 2.0
        return BoundingBox(cx - half, cy - half, cx + half, cy + half)

    # -- Relationships ---------------------------------------------------- #

    def intersection_area(self, other: BoundingBox) -> float:
        """Area of the overlap with ``other``; zero when disjoint."""
        left = max(self.x1, other.x1)
        top = max(self.y1, other.y1)
        right = min(self.x2, other.x2)
        bottom = min(self.y2, other.y2)
        if right <= left or bottom <= top:
            return 0.0
        return (right - left) * (bottom - top)

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union with ``other``, in ``[0, 1]``."""
        intersection = self.intersection_area(other)
        if intersection <= 0.0:
            return 0.0
        union = self.area + other.area - intersection
        return float(intersection / max(union, EPSILON))

    def contains_point(self, x: float, y: float) -> bool:
        """Whether ``(x, y)`` lies inside the box."""
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2

    def truncation_ratio(self, width: int, height: int) -> float:
        """Fraction of the box area lying outside a ``width`` x ``height`` frame.

        A face that is cut off by the edge of the photo cannot be reliably
        recognised, and the missing region is invisible to the occlusion
        analyser, so it needs its own signal.
        """
        if self.area <= 0.0:
            return 1.0
        visible = self.clip(width, height).area
        return float(max(0.0, 1.0 - visible / self.area))

    # -- Conversions ------------------------------------------------------ #

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(x1, y1, x2, y2)`` as floats."""
        return (self.x1, self.y1, self.x2, self.y2)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        """Return integer ``(x1, y1, x2, y2)``, outward-rounded for slicing."""
        return (
            int(math.floor(self.x1)),
            int(math.floor(self.y1)),
            int(math.ceil(self.x2)),
            int(math.ceil(self.y2)),
        )

    def as_xywh(self) -> tuple[float, float, float, float]:
        """Return ``(x, y, width, height)``."""
        return (self.x1, self.y1, self.width, self.height)

    def as_dict(self) -> dict[str, float]:
        """Serialisable representation used in the API response."""
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> BoundingBox:
        """Build from top-left corner plus size."""
        return cls(x, y, x + width, y + height)

    @classmethod
    def from_array(cls, array: npt.NDArray[Any]) -> BoundingBox:
        """Build from a length-4 array of ``[x1, y1, x2, y2]``."""
        flat = np.asarray(array, dtype=np.float64).reshape(-1)
        if flat.size < 4:
            raise ValueError(f"Expected at least 4 coordinates, got {flat.size}")
        return cls(float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3]))


@dataclass(frozen=True, slots=True)
class Landmarks5:
    """The five canonical facial keypoints in source-image pixel space.

    Order matches :data:`~hamqadam_ai.core.constants.FIVE_POINT_LANDMARK_NAMES`
    and the raw output of SCRFD, RetinaFace and YOLO-face:
    left eye, right eye, nose tip, left mouth corner, right mouth corner.

    "Left" and "right" are from the *viewer's* perspective, matching the
    detectors rather than anatomy.
    """

    points: FloatArray

    def __post_init__(self) -> None:
        array = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)
        if array.shape != (5, 2):
            raise ValueError(f"Landmarks5 requires a (5, 2) array, got {array.shape}")
        object.__setattr__(self, "points", array)

    @property
    def left_eye(self) -> tuple[float, float]:
        """Viewer-left eye centre."""
        return (float(self.points[0, 0]), float(self.points[0, 1]))

    @property
    def right_eye(self) -> tuple[float, float]:
        """Viewer-right eye centre."""
        return (float(self.points[1, 0]), float(self.points[1, 1]))

    @property
    def nose_tip(self) -> tuple[float, float]:
        """Nose tip."""
        return (float(self.points[2, 0]), float(self.points[2, 1]))

    @property
    def mouth_left(self) -> tuple[float, float]:
        """Viewer-left mouth corner."""
        return (float(self.points[3, 0]), float(self.points[3, 1]))

    @property
    def mouth_right(self) -> tuple[float, float]:
        """Viewer-right mouth corner."""
        return (float(self.points[4, 0]), float(self.points[4, 1]))

    @property
    def interocular_distance(self) -> float:
        """Pixel distance between the eye centres.

        The standard scale reference for facial measurements; every normalised
        landmark metric in this codebase divides by it.
        """
        return float(np.linalg.norm(self.points[1] - self.points[0]))

    @property
    def eye_center(self) -> tuple[float, float]:
        """Midpoint between the two eyes."""
        mid = (self.points[0] + self.points[1]) / 2.0
        return (float(mid[0]), float(mid[1]))

    @property
    def mouth_center(self) -> tuple[float, float]:
        """Midpoint between the two mouth corners."""
        mid = (self.points[3] + self.points[4]) / 2.0
        return (float(mid[0]), float(mid[1]))

    @property
    def roll_degrees(self) -> float:
        """In-plane rotation inferred from the eye line, in degrees.

        Positive means the head is tilted so the viewer-right eye sits lower.
        This is a direct geometric measurement and is used as a sanity check on
        the PnP solve rather than as its replacement.
        """
        dx = self.points[1, 0] - self.points[0, 0]
        dy = self.points[1, 1] - self.points[0, 1]
        return float(math.degrees(math.atan2(float(dy), float(dx))))

    def translate(self, dx: float, dy: float) -> Landmarks5:
        """Shift every point by ``(dx, dy)``."""
        return Landmarks5(self.points + np.array([dx, dy], dtype=np.float32))

    def scale(self, factor_x: float, factor_y: float | None = None) -> Landmarks5:
        """Scale every point about the image origin."""
        fy = factor_x if factor_y is None else factor_y
        return Landmarks5(self.points * np.array([factor_x, fy], dtype=np.float32))

    def to_box_relative(self, box: BoundingBox) -> FloatArray:
        """Express the points in ``[0, 1]`` coordinates relative to ``box``.

        Enables comparison against the canonical face template regardless of
        the face's size or position in the frame.
        """
        origin = np.array([box.x1, box.y1], dtype=np.float32)
        size = np.array([max(box.width, EPSILON), max(box.height, EPSILON)], dtype=np.float32)
        return ((self.points - origin) / size).astype(np.float32)

    def as_list(self) -> list[dict[str, Any]]:
        """Serialisable representation used in the API response."""
        return [
            {"name": name, "x": round(float(point[0]), 2), "y": round(float(point[1]), 2)}
            for name, point in zip(FIVE_POINT_LANDMARK_NAMES, self.points, strict=True)
        ]

    def is_plausible(self) -> bool:
        """Cheap sanity check that the points form a face-like configuration.

        Rejects the degenerate landmark sets some detectors emit on textured
        non-face regions: collapsed points, eyes below the mouth, or a nose
        outside the eye-to-mouth band. Costs microseconds and removes a class
        of false positives before the expensive pose solve runs.
        """
        interocular = self.interocular_distance
        if interocular < 1.0:
            return False
        eye_y = (self.points[0, 1] + self.points[1, 1]) / 2.0
        mouth_y = (self.points[3, 1] + self.points[4, 1]) / 2.0
        if mouth_y <= eye_y:
            return False
        nose_y = self.points[2, 1]
        # The nose tip must sit between the eye line and the mouth line, with a
        # generous tolerance of half the interocular distance for extreme pitch.
        margin = interocular * 0.5
        if not (eye_y - margin) <= nose_y <= (mouth_y + margin):
            return False
        # Vertical face extent should be a sane multiple of the eye separation.
        vertical_span = mouth_y - eye_y
        return bool(0.25 <= (vertical_span / interocular) <= 3.0)


# --------------------------------------------------------------------------- #
# Non-maximum suppression
# --------------------------------------------------------------------------- #


def iou_matrix(boxes_a: npt.NDArray[Any], boxes_b: npt.NDArray[Any]) -> FloatArray:
    """Pairwise IoU between two sets of boxes.

    Args:
        boxes_a: ``(N, 4)`` array of ``[x1, y1, x2, y2]``.
        boxes_b: ``(M, 4)`` array of ``[x1, y1, x2, y2]``.

    Returns:
        An ``(N, M)`` float32 matrix of IoU values.
    """
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    matrix: FloatArray = (inter / np.maximum(union, EPSILON)).astype(np.float32)
    return matrix


def nms(
    boxes: npt.NDArray[Any],
    scores: npt.NDArray[Any],
    iou_threshold: float,
    *,
    top_k: int | None = None,
) -> npt.NDArray[np.int64]:
    """Greedy non-maximum suppression.

    A vectorised implementation rather than a Python loop over survivors: face
    detectors routinely emit several thousand candidates before thresholding
    and the naive version dominates the detector's latency on CPU.

    Args:
        boxes: ``(N, 4)`` array of ``[x1, y1, x2, y2]``.
        scores: ``(N,)`` array of confidences.
        iou_threshold: Boxes overlapping a kept box by more than this are
            suppressed.
        top_k: Consider only the ``top_k`` highest-scoring candidates.

    Returns:
        Indices of the surviving boxes, ordered by descending score.
    """
    box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    if box_array.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if box_array.shape[0] != score_array.shape[0]:
        raise ValueError(
            f"boxes and scores disagree: {box_array.shape[0]} vs {score_array.shape[0]}"
        )

    order = np.argsort(-score_array)
    if top_k is not None:
        order = order[:top_k]

    x1, y1, x2, y2 = (box_array[:, i] for i in range(4))
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]

        left = np.maximum(x1[current], x1[rest])
        top = np.maximum(y1[current], y1[rest])
        right = np.minimum(x2[current], x2[rest])
        bottom = np.minimum(y2[current], y2[rest])

        inter = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
        overlap = inter / np.maximum(areas[current] + areas[rest] - inter, EPSILON)
        order = rest[overlap <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: npt.NDArray[Any],
    scores: npt.NDArray[Any],
    class_ids: npt.NDArray[Any],
    iou_threshold: float,
) -> npt.NDArray[np.int64]:
    """Class-aware NMS: boxes of different classes never suppress each other.

    Implemented with the standard coordinate-offset trick so a single
    :func:`nms` pass handles every class at once.
    """
    box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if box_array.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    classes = np.asarray(class_ids, dtype=np.float32).reshape(-1)
    stride = float(box_array.max()) + 1.0
    offset = (classes * stride)[:, None]
    return nms(box_array + offset, scores, iou_threshold)


# --------------------------------------------------------------------------- #
# Anchor-free distance decoding (SCRFD / FCOS family)
# --------------------------------------------------------------------------- #


def distance2bbox(
    anchor_points: npt.NDArray[Any],
    distances: npt.NDArray[Any],
    max_shape: tuple[int, int] | None = None,
) -> FloatArray:
    """Decode ``(left, top, right, bottom)`` distances into corner coordinates.

    SCRFD is anchor-free: each feature-map cell predicts how far its centre is
    from the four edges of the box it is responsible for. This inverts that
    parameterisation.

    Args:
        anchor_points: ``(N, 2)`` array of cell-centre ``(x, y)`` coordinates.
        distances: ``(N, 4)`` array of ``[left, top, right, bottom]`` offsets.
        max_shape: Optional ``(height, width)`` to clamp the result to.

    Returns:
        An ``(N, 4)`` float32 array of ``[x1, y1, x2, y2]``.
    """
    points = np.asarray(anchor_points, dtype=np.float32).reshape(-1, 2)
    offsets = np.asarray(distances, dtype=np.float32).reshape(-1, 4)

    x1 = points[:, 0] - offsets[:, 0]
    y1 = points[:, 1] - offsets[:, 1]
    x2 = points[:, 0] + offsets[:, 2]
    y2 = points[:, 1] + offsets[:, 3]

    if max_shape is not None:
        height, width = max_shape
        x1 = np.clip(x1, 0, width)
        y1 = np.clip(y1, 0, height)
        x2 = np.clip(x2, 0, width)
        y2 = np.clip(y2, 0, height)

    stacked: FloatArray = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)
    return stacked


def distance2kps(
    anchor_points: npt.NDArray[Any],
    distances: npt.NDArray[Any],
    max_shape: tuple[int, int] | None = None,
) -> FloatArray:
    """Decode per-keypoint ``(dx, dy)`` offsets into absolute coordinates.

    Args:
        anchor_points: ``(N, 2)`` array of cell-centre ``(x, y)`` coordinates.
        distances: ``(N, 2K)`` array of interleaved ``dx, dy`` per keypoint.
        max_shape: Optional ``(height, width)`` to clamp the result to.

    Returns:
        An ``(N, K, 2)`` float32 array of keypoint coordinates.
    """
    points = np.asarray(anchor_points, dtype=np.float32).reshape(-1, 2)
    offsets = np.asarray(distances, dtype=np.float32)
    if offsets.ndim != 2 or offsets.shape[1] % 2 != 0:
        raise ValueError(f"Keypoint distances must be (N, 2K); got {offsets.shape}")

    num_points = offsets.shape[1] // 2
    coords = offsets.reshape(-1, num_points, 2) + points[:, None, :]

    if max_shape is not None:
        height, width = max_shape
        coords[..., 0] = np.clip(coords[..., 0], 0, width)
        coords[..., 1] = np.clip(coords[..., 1], 0, height)

    return coords.astype(np.float32)


def build_anchor_centers(
    height: int, width: int, stride: int, num_anchors: int
) -> FloatArray:
    """Generate the anchor-point grid for one FPN level.

    Args:
        height: Feature-map height.
        width: Feature-map width.
        stride: Downsampling factor of this level relative to the input.
        num_anchors: Anchors per cell (2 for the SCRFD models used here).

    Returns:
        An ``(height * width * num_anchors, 2)`` array of ``(x, y)`` centres in
        network-input pixel space, ordered to match the flattened network
        output exactly.
    """
    grid_y, grid_x = np.mgrid[:height, :width]
    centers = np.stack([grid_x, grid_y], axis=-1).astype(np.float32) * float(stride)
    centers = centers.reshape(-1, 2)
    if num_anchors > 1:
        centers = np.repeat(centers, num_anchors, axis=0)
    return centers


def point_in_polygon_area(points: npt.NDArray[Any]) -> float:
    """Signed area of a simple polygon via the shoelace formula.

    Used by the occlusion analyser to measure the area of the landmark-derived
    facial regions.
    """
    poly = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


__all__ = [
    "BoundingBox",
    "Landmarks5",
    "batched_nms",
    "build_anchor_centers",
    "distance2bbox",
    "distance2kps",
    "iou_matrix",
    "nms",
    "point_in_polygon_area",
]
