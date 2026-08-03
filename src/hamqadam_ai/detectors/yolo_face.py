"""YOLOv8-face detector - first fallback.

Model selection
---------------
Kept as the first fallback rather than the primary for two reasons:

* SCRFD outperforms YOLOv8-n on WIDER FACE *hard* at comparable cost, and hard
  is the subset that resembles a badly-lit phone selfie.
* The YOLO-face exports in circulation are community-produced with no
  canonical, signed release, so their provenance is weaker.

It earns its place in the chain because its failure modes are *different* from
SCRFD's. It is markedly better on large, close-up, off-angle faces - a face
filling 70% of the frame at 40 degrees of yaw is a case where the anchor-free
FPN sometimes produces a loose box and YOLO does not. When SCRFD's weights are
unavailable this adapter degrades the service by very little.

Output layout
-------------
A YOLOv8-pose export emits a single ``(1, 4 + 1 + 3K, N)`` tensor, transposed
here to ``(N, 20)`` for a five-keypoint face model:

===========  =========================================
Columns      Meaning
===========  =========================================
``0:4``      ``cx, cy, w, h`` in network pixels
``4``        objectness (single class, so no class row)
``5:20``     ``x, y, visibility`` per keypoint
===========  =========================================
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import ModelSpec
from hamqadam_ai.detectors.base import BgrImage, FaceDetector, RawDetection
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.models.base import LoadedModel
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5, nms
from hamqadam_ai.utils.image_ops import build_blob, letterbox

log = get_logger(__name__)

#: Below this per-keypoint visibility the point is treated as unobserved.
#: YOLO-pose emits a coordinate for every keypoint regardless of whether it saw
#: one, so an unfiltered landmark set on a profile face contains invented points.
_KEYPOINT_VISIBILITY_FLOOR = 0.30


class YoloFaceDetector(FaceDetector):
    """YOLOv8-pose face detector with five keypoints.

    Args:
        model: The loaded ONNX session wrapper.
        spec: The model declaration, supplying pre-processing constants.
        score_threshold: Minimum objectness for a candidate.
        nms_iou_threshold: IoU above which overlapping boxes are suppressed.
        input_size: Network input ``(width, height)``.
        max_candidates: Cap on candidates carried into NMS.
    """

    provides_landmarks = True

    def __init__(
        self,
        model: LoadedModel,
        spec: ModelSpec,
        *,
        score_threshold: float,
        nms_iou_threshold: float,
        input_size: tuple[int, int],
        max_candidates: int = 200,
    ) -> None:
        super().__init__(
            name="yolo",
            version=spec.version,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_candidates=max_candidates,
        )
        self._model = model
        self._spec = spec
        self._input_size = input_size
        self._num_keypoints = int(spec.output.get("num_keypoints", 5))
        self._mean = spec.input.mean_tuple
        self._scale = spec.input.scale
        self._swap_rb = spec.input.swap_rb

        log.info(
            "detector.yolo.ready",
            version=spec.version,
            input_size=list(input_size),
            keypoints=self._num_keypoints,
        )

    def _detect_impl(self, image: BgrImage) -> list[RawDetection]:
        """Pre-process, run the network and decode into source coordinates."""
        # Centred letterbox with the 114 grey YOLO trains against.
        padded, transform = letterbox(
            image, self._input_size, pad_value=114, center=True
        )
        blob = build_blob(
            padded,
            self._input_size,
            mean=self._mean,
            scale=self._scale,
            swap_rb=self._swap_rb,
        )

        outputs = self._model.run({self._model.input_names[0]: blob})
        boxes, scores, keypoints, visibility = self._decode(outputs[0])

        if boxes.shape[0] == 0:
            return []

        keep = nms(boxes, scores, self.nms_iou_threshold, top_k=self.max_candidates)
        if keep.size == 0:
            return []

        kept_boxes = transform.unmap_array(boxes[keep].reshape(-1, 2, 2)).reshape(-1, 4)
        kept_scores = scores[keep]
        kept_keypoints = (
            transform.unmap_array(keypoints[keep]) if keypoints is not None else None
        )
        kept_visibility = visibility[keep] if visibility is not None else None

        height, width = image.shape[:2]
        detections: list[RawDetection] = []
        for index in range(kept_boxes.shape[0]):
            box = BoundingBox.from_array(kept_boxes[index]).clip(width, height)
            if box.width < 1.0 or box.height < 1.0:
                continue

            landmarks: Landmarks5 | None = None
            confidences: tuple[float, ...] | None = None
            if kept_keypoints is not None and self._num_keypoints >= 5:
                points: npt.NDArray[np.float32] | None = kept_keypoints[index][:5]
                if kept_visibility is not None:
                    point_visibility = kept_visibility[index][:5]
                    confidences = tuple(float(v) for v in point_visibility)
                    # YOLO-pose emits a coordinate for every keypoint whether
                    # or not it saw one, so an unfiltered set on a profile
                    # face contains invented points.
                    if float(point_visibility.min()) < _KEYPOINT_VISIBILITY_FLOOR:
                        points = None
                if points is not None:
                    candidate = Landmarks5(points)
                    landmarks = candidate if candidate.is_plausible() else None

            detections.append(
                RawDetection(
                    box=box,
                    confidence=float(kept_scores[index]),
                    landmarks=landmarks,
                    landmark_confidence=confidences if landmarks is not None else None,
                )
            )

        return detections

    def _decode(
        self, raw: npt.NDArray[Any]
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32] | None,
        npt.NDArray[np.float32] | None,
    ]:
        """Convert the packed YOLO tensor into boxes, scores and keypoints.

        Args:
            raw: The ``(1, C, N)`` output tensor, where ``C = 5 + 3K``.

        Returns:
            ``(boxes_xyxy, scores, keypoints, visibility)`` in network pixel
            space, already filtered by the score threshold.
        """
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim == 3:
            array = array[0]

        # Exports differ: some emit (C, N), others (N, C). The channel count is
        # small and fixed, so the short axis identifies the layout.
        expected_channels = 5 + 3 * self._num_keypoints
        if array.shape[0] == expected_channels:
            predictions = array.T
        elif array.shape[1] == expected_channels:
            predictions = array
        else:
            raise ValueError(
                f"YOLO output has shape {array.shape}; expected one axis of "
                f"length {expected_channels} for {self._num_keypoints} keypoints."
            )

        scores = predictions[:, 4]
        positive = np.flatnonzero(scores >= self.score_threshold)
        if positive.size == 0:
            empty = np.empty((0, 4), dtype=np.float32)
            return empty, np.empty((0,), dtype=np.float32), None, None

        selected = predictions[positive]

        # cx, cy, w, h -> x1, y1, x2, y2
        centers = selected[:, 0:2]
        sizes = selected[:, 2:4]
        half = sizes / 2.0
        boxes = np.concatenate([centers - half, centers + half], axis=1).astype(np.float32)

        keypoint_block = selected[:, 5:].reshape(selected.shape[0], self._num_keypoints, 3)
        keypoints = keypoint_block[:, :, 0:2].astype(np.float32)
        visibility = keypoint_block[:, :, 2].astype(np.float32)

        return boxes, selected[:, 4].astype(np.float32), keypoints, visibility

    async def detect_async(self, image: BgrImage) -> list[RawDetection]:
        """Detect using the registry's bounded inference thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._model.executor, self.detect, image)

    def close(self) -> None:
        """No-op; the session is owned by the model registry."""
        return


__all__ = ["YoloFaceDetector"]
