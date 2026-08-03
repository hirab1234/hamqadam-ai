"""MODULE 1 - Face detection.

Architecture
------------
A single port, :class:`FaceDetector`, with four interchangeable adapters
arranged in a fallback chain:

============  ====================================  ==========  ==========
Adapter       Model                                 Landmarks   Role
============  ====================================  ==========  ==========
``scrfd``     SCRFD-10GF (InsightFace buffalo_l)    5 points    primary
``yolo``      YOLOv8-n face                         5 points    fallback 1
``opencv_dnn``ResNet-10 SSD                         none        fallback 2
``haar``      Haar cascade (bundled with OpenCV)    derived     terminal
============  ====================================  ==========  ==========

The chain exists because a verification service that returns HTTP 500 when a
model file is missing is strictly worse than one that returns a correct answer
with a lower-quality detector and says so. Every result reports which adapter
produced it and whether a fallback was used, so degradation is always visible
rather than silent.

Analysis layered on top of raw detection:

* :mod:`~hamqadam_ai.detectors.pose` - head orientation via PnP.
* :mod:`~hamqadam_ai.detectors.occlusion` - multi-signal obstruction analysis.
* :mod:`~hamqadam_ai.detectors.visibility` - composite visibility score.
* :mod:`~hamqadam_ai.detectors.policy` - accept/reject rules.
"""

from __future__ import annotations

from hamqadam_ai.detectors.base import DetectedFace, FaceDetector, RawDetection
from hamqadam_ai.detectors.factory import (
    DetectorChain,
    build_detector_chain,
    create_detector,
)
from hamqadam_ai.detectors.haar import HaarCascadeDetector
from hamqadam_ai.detectors.occlusion import OcclusionAnalyzer
from hamqadam_ai.detectors.opencv_dnn import OpenCvDnnDetector
from hamqadam_ai.detectors.policy import DetectionPolicy, PolicyVerdict
from hamqadam_ai.detectors.pose import PoseEstimator
from hamqadam_ai.detectors.scrfd import ScrfdDetector
from hamqadam_ai.detectors.visibility import VisibilityScorer
from hamqadam_ai.detectors.yolo_face import YoloFaceDetector

__all__ = [
    "DetectedFace",
    "DetectionPolicy",
    "DetectorChain",
    "FaceDetector",
    "HaarCascadeDetector",
    "OcclusionAnalyzer",
    "OpenCvDnnDetector",
    "PolicyVerdict",
    "PoseEstimator",
    "RawDetection",
    "ScrfdDetector",
    "VisibilityScorer",
    "YoloFaceDetector",
    "build_detector_chain",
    "create_detector",
]
