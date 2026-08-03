"""Haar cascade face detector - the terminal fallback.

Why a 2001 algorithm is in a 2026 production system
---------------------------------------------------
Because it is the only detector in the chain that **cannot fail to be
available**. The cascade XML ships inside the ``opencv-python`` wheel, so it
needs no download, no network access at build time, no model store and no
GPU. Whatever else goes wrong - a wiped volume, a firewall blocking the model
CDN, a corrupted artefact failing its digest check - this adapter still runs.

That property is worth a great deal in an identity-verification service. The
alternative to a weak answer here is HTTP 503 on every request, and the fraud
engine can weigh a low-confidence detection appropriately while a total outage
gives it nothing to weigh at all.

Its limitations are real and are reported honestly:

* Viola-Jones has no notion of a confidence score. The
  ``detectMultiScale3`` stage weights are used as a proxy, squashed through a
  logistic. They are a *ranking* signal, not a calibrated probability, and the
  visibility scorer treats them as such.
* Recall collapses beyond ~25 degrees of yaw and in low light.
* It produces no landmarks. This adapter recovers an eye pair from a second
  cascade and *constructs* the remaining three points from the canonical face
  template. Those points are flagged ``landmarks_derived=True`` so that the
  pose estimator refuses to run a PnP solve on them - a derived nose tip would
  report perfect frontality for a face at 30 degrees of yaw.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.constants import ARCFACE_REFERENCE_LANDMARKS_112
from hamqadam_ai.core.exceptions import (
    InferenceError,
    ModelLoadError,
    ModelNotLoadedError,
)
from hamqadam_ai.detectors.base import BgrImage, FaceDetector, RawDetection
from hamqadam_ai.detectors.cascades import (
    EYE_TREE_EYEGLASSES,
    FRONTAL_FACE,
)
from hamqadam_ai.detectors.cascades import (
    cascade_path as resolve_cascade,
)
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5, nms
from hamqadam_ai.utils.image_ops import to_grayscale

log = get_logger(__name__)


def cascade_api_available() -> bool:
    """Whether this OpenCV build still exposes the Haar cascade API.

    OpenCV 5.0 removed ``cv2.CascadeClassifier`` along with the rest of the
    legacy objdetect module. Probing for the symbol keeps the check correct
    across distributions and custom builds, and lets the detector chain skip
    this adapter cleanly instead of dying with an ``AttributeError`` deep
    inside construction.
    """
    return hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "CASCADE_SCALE_IMAGE")

#: Canonical template normalised so the eye centres sit at the origin, one unit
#: apart horizontally. Used to place the nose and mouth once an eye pair is
#: known. Derived from the ArcFace 112x112 reference, so a face built this way
#: aligns consistently with a model-predicted one.
_TEMPLATE_EYE_NORMALISED: npt.NDArray[np.float32] = (
    lambda ref: (
        (ref - (ref[0] + ref[1]) / 2.0) / float(np.linalg.norm(ref[1] - ref[0]))
    ).astype(np.float32)
)(ARCFACE_REFERENCE_LANDMARKS_112)

#: Logistic steepness and midpoint mapping a cascade stage weight to [0, 1].
#: Calibrated so a typical clean frontal detection (weight ~4-6) lands around
#: 0.75-0.90 and a marginal one (weight ~1) lands near 0.35 - deliberately
#: below the default 0.60 policy floor, so weak Haar hits do not auto-pass.
_WEIGHT_STEEPNESS = 0.55
_WEIGHT_MIDPOINT = 2.6


class HaarCascadeDetector(FaceDetector):
    """Viola-Jones cascade face detector with an eye-cascade landmark recovery.

    Args:
        cascade_path: Path to the frontal-face cascade XML. Defaults to the
            copy bundled with OpenCV.
        eye_cascade_path: Path to the eye cascade XML used for landmark
            recovery. Landmark recovery is skipped when it cannot be loaded.
        version: Version string reported on results.
        score_threshold: Minimum mapped confidence for a candidate.
        nms_iou_threshold: IoU above which overlapping boxes are suppressed.
        scale_factor: Pyramid step. 1.1 is thorough and slow; 1.15 is a good
            compromise for a fallback path that is already off the happy path.
        min_neighbors: Overlapping-detection votes required to keep a window.
        min_size_ratio: Minimum face side as a fraction of the image's shorter
            side, which bounds the pyramid and keeps latency predictable.
        max_candidates: Cap on candidates carried into NMS.
    """

    provides_landmarks = True

    def __init__(
        self,
        *,
        cascade_path: Path | None = None,
        eye_cascade_path: Path | None = None,
        version: str = "opencv-haarcascade-frontalface-default",
        score_threshold: float = 0.30,
        nms_iou_threshold: float = 0.40,
        scale_factor: float = 1.15,
        min_neighbors: int = 5,
        min_size_ratio: float = 0.06,
        max_candidates: int = 200,
    ) -> None:
        super().__init__(
            name="haar",
            version=version,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_candidates=max_candidates,
        )
        if not cascade_api_available():
            raise ModelLoadError(
                "This OpenCV build has no CascadeClassifier, so the Haar "
                "detector cannot be loaded. OpenCV 5.0 removed the legacy "
                "objdetect API; install the 4.x line to retain the terminal "
                "fallback.",
                details={"opencv_version": cv2.__version__},
            )

        face_xml = cascade_path or resolve_cascade(FRONTAL_FACE)
        eye_xml = eye_cascade_path or resolve_cascade(EYE_TREE_EYEGLASSES)

        if face_xml is None or not face_xml.is_file():
            raise ModelLoadError(
                "The Haar frontal-face cascade could not be located. It is "
                "vendored at hamqadam_ai/detectors/cascades/; a packaging step "
                "has stripped it.",
                details={"expected": FRONTAL_FACE},
            )

        self._cascade: cv2.CascadeClassifier | None = cv2.CascadeClassifier(
            str(face_xml)
        )
        if self._cascade.empty():
            raise ModelLoadError(
                f"OpenCV could not parse the Haar face cascade at {face_xml}.",
                details={"path": str(face_xml)},
            )

        self._eye_cascade: cv2.CascadeClassifier | None = (
            cv2.CascadeClassifier(str(eye_xml)) if eye_xml is not None else None
        )
        if self._eye_cascade is not None and self._eye_cascade.empty():
            log.warning(
                "detector.haar.eye_cascade_unavailable",
                path=str(eye_xml),
                note="landmarks will not be recovered; pose analysis will be skipped",
            )
            self._eye_cascade = None

        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size_ratio = min_size_ratio
        self.provides_landmarks = self._eye_cascade is not None

        log.info(
            "detector.haar.ready",
            version=version,
            cascade=face_xml.name,
            eye_cascade=(
                eye_xml.name
                if (eye_xml is not None and self._eye_cascade is not None)
                else None
            ),
            vendored=str(face_xml.parent).endswith("cascades"),
        )

    @property
    def _face_cascade(self) -> cv2.CascadeClassifier:
        """The live frontal-face cascade, raising if the adapter is closed."""
        if self._cascade is None:
            raise ModelNotLoadedError(
                "The Haar cascade detector has been closed.",
                details={"detector": self.name},
            )
        return self._cascade

    def _detect_impl(self, image: BgrImage) -> list[RawDetection]:
        """Run the cascade and recover landmarks where an eye pair is found."""
        gray = to_grayscale(image)
        # Histogram equalisation is close to mandatory for Viola-Jones: the
        # cascade thresholds were tuned on normalised intensities and recall
        # drops sharply on under- or over-exposed input without it.
        equalised: npt.NDArray[np.uint8] = np.asarray(
            cv2.equalizeHist(gray), dtype=np.uint8
        )

        height, width = gray.shape[:2]
        minimum = max(24, int(min(height, width) * self._min_size_ratio))

        try:
            rects, _levels, weights = self._face_cascade.detectMultiScale3(
                equalised,
                scaleFactor=self._scale_factor,
                minNeighbors=self._min_neighbors,
                minSize=(minimum, minimum),
                flags=cv2.CASCADE_SCALE_IMAGE,
                outputRejectLevels=True,
            )
        except cv2.error as exc:
            raise InferenceError(
                f"Haar cascade detection failed: {exc}",
                details={"detector": self.name, "image_shape": [height, width]},
                cause=exc,
            ) from exc

        if len(rects) == 0:
            return []

        boxes = np.array(
            [[x, y, x + w, y + h] for (x, y, w, h) in rects], dtype=np.float32
        )
        scores = np.array(
            [_weight_to_confidence(float(w)) for w in np.asarray(weights).reshape(-1)],
            dtype=np.float32,
        )

        above = np.flatnonzero(scores >= self.score_threshold)
        if above.size == 0:
            return []
        boxes = boxes[above]
        scores = scores[above]

        keep = nms(boxes, scores, self.nms_iou_threshold, top_k=self.max_candidates)

        detections: list[RawDetection] = []
        for index in keep:
            box = BoundingBox.from_array(boxes[index]).clip(width, height)
            if box.width < 1.0 or box.height < 1.0:
                continue
            landmarks = self._recover_landmarks(equalised, box)
            detections.append(
                RawDetection(
                    box=box,
                    confidence=float(scores[index]),
                    landmarks=landmarks,
                    landmarks_derived=landmarks is not None,
                )
            )
        return detections

    def _recover_landmarks(
        self, gray: npt.NDArray[np.uint8], box: BoundingBox
    ) -> Landmarks5 | None:
        """Locate an eye pair and construct the five-point set around it.

        Returns ``None`` unless exactly the two most confident eye candidates
        form a plausible pair: roughly level, separated by a sensible fraction
        of the face width, and both in the upper half of the face box.
        """
        if self._eye_cascade is None:
            return None

        x1, y1, x2, y2 = box.as_int_tuple()
        # Eyes live in the upper 60% of a face box; searching the lower half
        # mostly finds nostrils and mouth corners.
        roi_y2 = y1 + int((y2 - y1) * 0.60)
        roi = gray[max(0, y1) : max(0, roi_y2), max(0, x1) : max(0, x2)]
        if roi.size == 0 or roi.shape[0] < 12 or roi.shape[1] < 24:
            return None

        min_eye = max(8, int(box.width * 0.12))
        try:
            eyes = self._eye_cascade.detectMultiScale(
                roi,
                scaleFactor=1.10,
                minNeighbors=4,
                minSize=(min_eye, min_eye),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except cv2.error:
            return None

        if len(eyes) < 2:
            return None

        # Take the two largest candidates and order them left-to-right.
        ordered = sorted(eyes, key=lambda rect: int(rect[2]) * int(rect[3]), reverse=True)[:2]
        centres = sorted(
            [
                (float(x1 + ex + ew / 2.0), float(y1 + ey + eh / 2.0))
                for (ex, ey, ew, eh) in ordered
            ],
            key=lambda point: point[0],
        )
        left_eye, right_eye = centres

        separation = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
        if separation < box.width * 0.20 or separation > box.width * 0.75:
            return None
        # Reject a "pair" that is actually one eye detected twice, or an eye
        # paired with an eyebrow: a real pair is close to level.
        vertical_offset = abs(right_eye[1] - left_eye[1])
        if vertical_offset > separation * 0.45:
            return None

        return self._construct_from_eyes(left_eye, right_eye)

    @staticmethod
    def _construct_from_eyes(
        left_eye: tuple[float, float], right_eye: tuple[float, float]
    ) -> Landmarks5:
        """Place the canonical template on a known eye pair.

        The similarity transform that maps the template's eye centres onto the
        detected pair is applied to all five template points. The eyes are
        therefore exact; the nose and mouth are a statistical average face
        rotated and scaled to fit. This is why the result is flagged as derived.
        """
        left = np.array(left_eye, dtype=np.float32)
        right = np.array(right_eye, dtype=np.float32)

        midpoint = (left + right) / 2.0
        delta = right - left
        separation = float(np.linalg.norm(delta))
        angle = math.atan2(float(delta[1]), float(delta[0]))

        cos_a = math.cos(angle) * separation
        sin_a = math.sin(angle) * separation
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        points = (_TEMPLATE_EYE_NORMALISED @ rotation.T) + midpoint
        return Landmarks5(points.astype(np.float32))

    def close(self) -> None:
        """Drop the cascades."""
        self._cascade = None
        self._eye_cascade = None


def _weight_to_confidence(weight: float) -> float:
    """Squash a cascade stage weight into a ``[0, 1]`` pseudo-confidence.

    Viola-Jones emits no probability. ``detectMultiScale3`` returns the
    accumulated stage weight of the winning window, an unbounded real where
    larger is stronger. A logistic maps it onto the unit interval so the
    detector's output is at least *comparable* with the neural detectors'
    scores, while remaining honestly uncalibrated.
    """
    return 1.0 / (1.0 + math.exp(-_WEIGHT_STEEPNESS * (weight - _WEIGHT_MIDPOINT)))


__all__ = ["HaarCascadeDetector"]
