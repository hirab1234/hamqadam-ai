"""Infrastructure helpers: image codecs, geometry, timing, hashing, scratch space.

Nothing here knows about faces, CNICs or verification. These are the reusable
primitives the capability modules are built from.
"""

from __future__ import annotations

from hamqadam_ai.utils.geometry import (
    BoundingBox,
    Landmarks5,
    batched_nms,
    distance2bbox,
    distance2kps,
    iou_matrix,
    nms,
)
from hamqadam_ai.utils.hashing import (
    average_hash,
    hamming_distance,
    sha256_bytes,
    sha256_file,
)
from hamqadam_ai.utils.image_io import (
    DecodedImage,
    decode_image,
    decode_image_b64,
    encode_image,
    load_image,
)
from hamqadam_ai.utils.image_ops import (
    LetterboxTransform,
    convert_to_bgr,
    crop_with_margin,
    ensure_min_size,
    letterbox,
    resize_long_side,
    to_grayscale,
)
from hamqadam_ai.utils.tempfiles import ScratchSpace, TempFileJanitor, scratch_space
from hamqadam_ai.utils.timing import StageTimings, Stopwatch, perf_timer

__all__ = [
    "BoundingBox",
    "DecodedImage",
    "Landmarks5",
    "LetterboxTransform",
    "ScratchSpace",
    "StageTimings",
    "Stopwatch",
    "TempFileJanitor",
    "average_hash",
    "batched_nms",
    "convert_to_bgr",
    "crop_with_margin",
    "decode_image",
    "decode_image_b64",
    "distance2bbox",
    "distance2kps",
    "encode_image",
    "ensure_min_size",
    "hamming_distance",
    "iou_matrix",
    "letterbox",
    "load_image",
    "nms",
    "perf_timer",
    "resize_long_side",
    "scratch_space",
    "sha256_bytes",
    "sha256_file",
    "to_grayscale",
]
