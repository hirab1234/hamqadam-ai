"""Immutable domain constants shared across every module.

Anything that is a *fact about the problem domain* lives here. Anything that an
operator might want to tune lives in ``configs/thresholds.yaml`` instead.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

import numpy as np

# --------------------------------------------------------------------------- #
# Roles an image can play in a verification request
# --------------------------------------------------------------------------- #


class ImageRole(StrEnum):
    """Semantic role of an input image inside a verification request.

    The role drives which policy is applied. A CNIC portrait is a low-DPI
    print and is held to a laxer quality bar than a live selfie, which is the
    biometric reference for the whole decision.
    """

    LIVE_SELFIE = "live_selfie"
    PROFILE_IMAGE = "profile_image"
    SECONDARY_IMAGE = "secondary_image"
    CNIC_IMAGE = "cnic_image"
    CNIC_PORTRAIT = "cnic_portrait"


class VerificationStatus(StrEnum):
    """Lifecycle state of a verification request.

    Mirrors the state machine agreed with the Backend in section 18 of the
    requirements document.
    """

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    FAILED = "FAILED"


class Recommendation(StrEnum):
    """Final AI recommendation returned to the Backend rules engine."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class RiskLevel(StrEnum):
    """Coarse fraud-risk banding."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class MatchDecision(StrEnum):
    """Outcome of a single biometric comparison."""

    STRONG_MATCH = "STRONG_MATCH"
    REVIEW = "REVIEW"
    FAILED = "FAILED"
    NOT_COMPARED = "NOT_COMPARED"


# --------------------------------------------------------------------------- #
# Facial geometry
# --------------------------------------------------------------------------- #


class FaceRegion(StrEnum):
    """Anatomical regions used by the occlusion analyser.

    The ordering of :data:`OCCLUSION_REGION_ORDER` is part of the model
    contract for the optional ONNX occlusion classifier and must not change
    without retraining that model.
    """

    FOREHEAD = "forehead"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    NOSE = "nose"
    MOUTH = "mouth"
    CHIN = "chin"


OCCLUSION_REGION_ORDER: Final[tuple[FaceRegion, ...]] = (
    FaceRegion.FOREHEAD,
    FaceRegion.LEFT_EYE,
    FaceRegion.RIGHT_EYE,
    FaceRegion.NOSE,
    FaceRegion.MOUTH,
    FaceRegion.CHIN,
)

#: Order of the five keypoints emitted by SCRFD, RetinaFace and YOLO-face.
#: "left"/"right" are from the *viewer's* perspective, matching the detectors.
FIVE_POINT_LANDMARK_NAMES: Final[tuple[str, ...]] = (
    "left_eye",
    "right_eye",
    "nose_tip",
    "mouth_left",
    "mouth_right",
)

#: Canonical destination points for ArcFace 112x112 alignment.
#: These exact coordinates are the InsightFace reference; changing them
#: invalidates every embedding ever produced by this service.
ARCFACE_REFERENCE_LANDMARKS_112: Final[np.ndarray] = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # mouth left
        [70.7299, 92.2041],  # mouth right
    ],
    dtype=np.float32,
)

#: Sparse 3D head model in millimetres, used by :mod:`hamqadam_ai.detectors.pose`
#: for the PnP solve. Values are the five-point subset of the widely used mean
#: adult face, and the axes follow the **OpenCV camera convention** so that no
#: axis flip is needed between the model and the image plane:
#:
#: * ``+X`` points to the viewer's right,
#: * ``+Y`` points **down** (matching image row order),
#: * ``+Z`` points away from the camera, into the scene.
#:
#: The origin is the nose tip, which is the most protruding landmark and
#: therefore has the smallest depth; the eyes and mouth sit further back at
#: positive Z. Getting this handedness wrong silently mirrors the recovered
#: yaw, which is why the convention is spelled out rather than assumed.
CANONICAL_FACE_3D_5PT: Final[np.ndarray] = np.array(
    [
        [-34.0, -32.0, 26.0],  # left eye centre  (viewer's left, above nose)
        [34.0, -32.0, 26.0],  # right eye centre (viewer's right, above nose)
        [0.0, 0.0, 0.0],  # nose tip         (origin, most protruding)
        [-26.0, 34.0, 22.0],  # left mouth corner  (below nose)
        [26.0, 34.0, 22.0],  # right mouth corner (below nose)
    ],
    dtype=np.float64,
)

#: Canonical 2D five-point template on a unit face box, used to measure how far
#: a detected landmark set deviates from a plausible face after alignment.
CANONICAL_FACE_2D_UNIT: Final[np.ndarray] = ARCFACE_REFERENCE_LANDMARKS_112 / 112.0


# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #

#: Container formats we are willing to decode. Anything else is rejected as
#: INVALID_IMAGE rather than handed to a decoder, which shrinks the attack
#: surface of the image-parsing stage considerably.
SUPPORTED_IMAGE_FORMATS: Final[frozenset[str]] = frozenset(
    {"JPEG", "PNG", "WEBP", "BMP"}
)

SUPPORTED_MIME_TYPES: Final[frozenset[str]] = frozenset(
    {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
)

#: Magic-byte signatures used to sniff the real format before decoding.
IMAGE_MAGIC_BYTES: Final[dict[str, tuple[bytes, ...]]] = {
    "JPEG": (b"\xff\xd8\xff",),
    "PNG": (b"\x89PNG\r\n\x1a\n",),
    "WEBP": (b"RIFF",),
    "BMP": (b"BM",),
}

#: Hard ceiling on decoded pixel count (width * height). A 178-megapixel PNG
#: that decompresses to 700 MB is a classic decompression-bomb DoS.
MAX_DECODED_PIXELS: Final[int] = 50_000_000

#: Hard ceiling on a single encoded image payload.
MAX_ENCODED_IMAGE_BYTES: Final[int] = 15 * 1024 * 1024

#: Absolute minimum accepted dimension on the short side.
MIN_IMAGE_DIMENSION: Final[int] = 64

#: Number of secondary profile images the Flutter app may supply.
MAX_SECONDARY_IMAGES: Final[int] = 4


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

#: All externally reported scores are on a 0-100 scale with one decimal, as
#: fixed by the sample response in section 16 of the requirements document.
SCORE_SCALE: Final[float] = 100.0
SCORE_DECIMALS: Final[int] = 2

#: Numerical guard used whenever we divide by a magnitude or a variance.
EPSILON: Final[float] = 1e-8

__all__ = [
    "ARCFACE_REFERENCE_LANDMARKS_112",
    "CANONICAL_FACE_2D_UNIT",
    "CANONICAL_FACE_3D_5PT",
    "EPSILON",
    "FIVE_POINT_LANDMARK_NAMES",
    "IMAGE_MAGIC_BYTES",
    "MAX_DECODED_PIXELS",
    "MAX_ENCODED_IMAGE_BYTES",
    "MAX_SECONDARY_IMAGES",
    "MIN_IMAGE_DIMENSION",
    "OCCLUSION_REGION_ORDER",
    "SCORE_DECIMALS",
    "SCORE_SCALE",
    "SUPPORTED_IMAGE_FORMATS",
    "SUPPORTED_MIME_TYPES",
    "FaceRegion",
    "ImageRole",
    "MatchDecision",
    "Recommendation",
    "RiskLevel",
    "VerificationStatus",
]
