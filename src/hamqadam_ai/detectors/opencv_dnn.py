"""OpenCV ResNet-10 SSD face detector - second fallback.

Model selection
---------------
A 2017-vintage Caffe SSD with a ResNet-10 backbone, distributed with OpenCV.
It is markedly weaker than SCRFD: no landmarks, poor recall below ~40 px, and
a bias towards frontal faces. It sits in the chain anyway because it is a
*different lineage entirely* - a different framework, a different training set
and a different failure surface from the two anchor-free detectors above it.
When both ONNX models are unavailable, this one still gives a correct box on
the overwhelming majority of cooperative selfies.

Landmarks
---------
None. The adapter reports ``provides_landmarks = False`` and returns ``None``
rather than inventing keypoints. Downstream, that means pose and occlusion
analysis are skipped for this detector and the visibility score is computed
from the components that remain - a documented, explicit degradation which is
surfaced as a warning on the result.

The model is loaded through ``cv2.dnn`` rather than the ONNX registry because
it is a Caffe artefact; the adapter therefore owns its own lifecycle.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import ModelSpec
from hamqadam_ai.core.exceptions import (
    InferenceError,
    ModelLoadError,
    ModelNotLoadedError,
)
from hamqadam_ai.detectors.base import BgrImage, FaceDetector, RawDetection
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.utils.geometry import BoundingBox, nms

log = get_logger(__name__)


def caffe_importer_available() -> bool:
    """Whether this OpenCV build can read a Caffe model.

    OpenCV 5.0 dropped the legacy Caffe importer. Probing for the symbol
    rather than parsing a version string keeps the check correct across
    distributions and custom builds.
    """
    return hasattr(cv2.dnn, "readNetFromCaffe")


class OpenCvDnnDetector(FaceDetector):
    """Caffe ResNet-10 SSD face detector, box-only.

    Args:
        weights_path: Path to the ``.caffemodel`` file.
        config_path: Path to the ``deploy.prototxt`` file.
        spec: The model declaration, supplying pre-processing constants.
        score_threshold: Minimum confidence for a candidate.
        nms_iou_threshold: IoU above which overlapping boxes are suppressed.
        prefer_cuda: Attempt the CUDA DNN backend. Silently falls back to CPU
            when the OpenCV build has no CUDA support, which the standard
            ``opencv-python-headless`` wheel does not.
        max_candidates: Cap on candidates carried into NMS.
    """

    provides_landmarks = False

    def __init__(
        self,
        weights_path: Path,
        config_path: Path,
        spec: ModelSpec,
        *,
        score_threshold: float,
        nms_iou_threshold: float,
        prefer_cuda: bool = False,
        max_candidates: int = 200,
    ) -> None:
        super().__init__(
            name="opencv_dnn",
            version=spec.version,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_candidates=max_candidates,
        )
        if not caffe_importer_available():
            raise ModelLoadError(
                "This OpenCV build has no Caffe importer, so the ResNet-SSD "
                "detector cannot be loaded. OpenCV 5.0 removed "
                "cv2.dnn.readNetFromCaffe; install the 4.x line to use this "
                "adapter. The detector chain will skip it.",
                details={"opencv_version": cv2.__version__},
            )
        if not weights_path.is_file():
            raise ModelLoadError(
                f"OpenCV DNN weights not found at {weights_path}.",
                details={"path": str(weights_path)},
            )
        if not config_path.is_file():
            raise ModelLoadError(
                f"OpenCV DNN prototxt not found at {config_path}.",
                details={"path": str(config_path)},
            )

        self._net: cv2.dnn.Net | None = None
        try:
            self._net = cv2.dnn.readNetFromCaffe(str(config_path), str(weights_path))
        except cv2.error as exc:
            raise ModelLoadError(
                f"OpenCV could not parse the Caffe model: {exc}",
                details={"weights": str(weights_path), "config": str(config_path)},
                cause=exc,
            ) from exc

        self._backend = "cpu"
        if prefer_cuda:
            self._backend = self._try_cuda()

        self._input_size = spec.input.size
        mean = spec.input.mean
        self._mean: tuple[float, float, float] = (
            (float(mean), float(mean), float(mean))
            if isinstance(mean, int | float)
            else (float(mean[0]), float(mean[1]), float(mean[2]))
        )
        self._scale = spec.input.scale
        self._swap_rb = spec.input.swap_rb

        log.info(
            "detector.opencv_dnn.ready",
            version=spec.version,
            input_size=list(self._input_size),
            backend=self._backend,
        )

    @property
    def _network(self) -> cv2.dnn.Net:
        """The live network, raising if the adapter has been closed.

        `close()` drops the reference so OpenCV can free its buffers, which
        makes the attribute genuinely optional. Routing every use through this
        accessor turns a use-after-close from an AttributeError deep inside a
        forward pass into a typed, diagnosable error.
        """
        if self._net is None:
            raise ModelNotLoadedError(
                "The OpenCV DNN detector has been closed.",
                details={"detector": self.name},
            )
        return self._net

    def _try_cuda(self) -> str:
        """Switch the DNN backend to CUDA when the build supports it."""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
                return "cpu"
            self._network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except (cv2.error, AttributeError):
            # opencv-python-headless is built without CUDA; this is expected.
            return "cpu"
        return "cuda"

    def _detect_impl(self, image: BgrImage) -> list[RawDetection]:
        """Run the SSD and map its normalised outputs to source pixels."""
        height, width = image.shape[:2]

        # The SSD was trained on a fixed 300x300 stretch, not a letterbox.
        # Preserving the aspect ratio here would put the face outside the
        # distribution the model learned and measurably drops recall.
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=self._scale,
            size=self._input_size,
            mean=self._mean,
            swapRB=self._swap_rb,
            crop=False,
        )

        try:
            network = self._network
            network.setInput(blob)
            raw = network.forward()
        except cv2.error as exc:
            raise InferenceError(
                f"OpenCV DNN forward pass failed: {exc}",
                details={"detector": self.name, "image_shape": [height, width]},
                cause=exc,
            ) from exc

        boxes, scores = self._decode(
            np.asarray(raw, dtype=np.float32), width, height
        )
        if boxes.shape[0] == 0:
            return []

        keep = nms(boxes, scores, self.nms_iou_threshold, top_k=self.max_candidates)
        detections: list[RawDetection] = []
        for index in keep:
            box = BoundingBox.from_array(boxes[index]).clip(width, height)
            if box.width < 1.0 or box.height < 1.0:
                continue
            detections.append(
                RawDetection(box=box, confidence=float(scores[index]), landmarks=None)
            )
        return detections

    def _decode(
        self, raw: npt.NDArray[np.float32], width: int, height: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Convert the ``(1, 1, N, 7)`` SSD tensor to pixel boxes and scores.

        Row layout is ``[batch_id, class_id, confidence, x1, y1, x2, y2]`` with
        the coordinates normalised to ``[0, 1]``.
        """
        detections = np.asarray(raw, dtype=np.float32).reshape(-1, 7)
        confidences = detections[:, 2]
        positive = np.flatnonzero(confidences >= self.score_threshold)
        if positive.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        selected = detections[positive]
        scale = np.array([width, height, width, height], dtype=np.float32)
        boxes = selected[:, 3:7] * scale
        return boxes.astype(np.float32), selected[:, 2].astype(np.float32)

    def close(self) -> None:
        """Drop the network so OpenCV can free its buffers."""
        self._net = None


__all__ = ["OpenCvDnnDetector"]
