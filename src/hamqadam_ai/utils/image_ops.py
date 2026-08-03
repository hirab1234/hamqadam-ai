"""Image transforms shared by every capability module.

Two rules hold throughout:

* Any transform that changes the coordinate system returns a transform object
  capable of mapping detections back to source-image coordinates. Losing that
  mapping is the classic source of bounding boxes that are subtly wrong on
  non-square inputs.
* Downscaling uses ``INTER_AREA`` and upscaling uses ``INTER_LINEAR``. Using
  ``INTER_LINEAR`` to downscale aliases high-frequency detail and measurably
  inflates the blur score in Module 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.constants import EPSILON
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

BgrImage = npt.NDArray[np.uint8]
GrayImage = npt.NDArray[np.uint8]


def _interpolation_for(scale: float) -> int:
    """Pick the resampling kernel appropriate to the scale factor."""
    return cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR


def _as_u8(array: npt.NDArray[Any]) -> BgrImage:
    """Narrow a loosely-typed OpenCV result back to uint8.

    The cv2 stubs declare ``ndarray[Any, dtype[integer | floating]]`` for
    almost every operation, which defeats the precise annotations this
    module relies on. Narrowing here - rather than weakening the public
    signatures - keeps the image pipeline's contracts checkable. It is a
    no-op at runtime when the array is already uint8.
    """
    return np.asarray(array, dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Records how an image was letterboxed, so detections can be un-mapped.

    Attributes:
        scale: Uniform factor the source was multiplied by.
        pad_x: Horizontal padding added to the left edge, in target pixels.
        pad_y: Vertical padding added to the top edge, in target pixels.
        source_size: Original ``(width, height)``.
        target_size: Letterboxed ``(width, height)``.
    """

    scale: float
    pad_x: float
    pad_y: float
    source_size: tuple[int, int]
    target_size: tuple[int, int]

    def unmap_point(self, x: float, y: float) -> tuple[float, float]:
        """Map a point from letterboxed space back to source coordinates."""
        return ((x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale)

    def unmap_box(self, box: BoundingBox) -> BoundingBox:
        """Map a box from letterboxed space back to source coordinates."""
        x1, y1 = self.unmap_point(box.x1, box.y1)
        x2, y2 = self.unmap_point(box.x2, box.y2)
        return BoundingBox(x1, y1, x2, y2)

    def unmap_landmarks(self, landmarks: Landmarks5) -> Landmarks5:
        """Map five keypoints from letterboxed space back to source coordinates."""
        points = landmarks.points.copy()
        points[:, 0] = (points[:, 0] - self.pad_x) / self.scale
        points[:, 1] = (points[:, 1] - self.pad_y) / self.scale
        return Landmarks5(points)

    def unmap_array(self, coordinates: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        """Vectorised un-mapping of an ``(..., 2)`` array of points."""
        result = np.asarray(coordinates, dtype=np.float32).copy()
        result[..., 0] = (result[..., 0] - self.pad_x) / self.scale
        result[..., 1] = (result[..., 1] - self.pad_y) / self.scale
        return result


def letterbox(
    image: BgrImage,
    target_size: tuple[int, int],
    *,
    pad_value: int = 114,
    center: bool = True,
) -> tuple[BgrImage, LetterboxTransform]:
    """Resize preserving aspect ratio, padding the remainder.

    Distorting the aspect ratio to force a square input costs several points of
    detection recall on portrait-orientation phone photos, which is exactly the
    input distribution this service sees.

    Args:
        image: Source BGR array.
        target_size: Desired ``(width, height)``.
        pad_value: Fill value for the padded region. 114 is the YOLO/SCRFD
            training-time convention; a neutral grey rather than black avoids
            creating a hard synthetic edge next to the image content.
        center: Pad symmetrically. When False, all padding goes bottom-right,
            which is what some ONNX exports assume.

    Returns:
        The letterboxed image and the :class:`LetterboxTransform` that produced
        it.
    """
    source_height, source_width = image.shape[:2]
    target_width, target_height = target_size

    scale = min(target_width / max(source_width, 1), target_height / max(source_height, 1))
    new_width = max(1, int(round(source_width * scale)))
    new_height = max(1, int(round(source_height * scale)))

    resized = cv2.resize(
        image, (new_width, new_height), interpolation=_interpolation_for(scale)
    )

    pad_w = target_width - new_width
    pad_h = target_height - new_height
    if center:
        left = pad_w // 2
        top = pad_h // 2
    else:
        left = 0
        top = 0

    canvas = np.full((target_height, target_width, 3), pad_value, dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = resized

    return canvas, LetterboxTransform(
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
        source_size=(source_width, source_height),
        target_size=(target_width, target_height),
    )


def resize_long_side(image: BgrImage, max_long_side: int) -> tuple[BgrImage, float]:
    """Downscale so the longer edge is at most ``max_long_side``.

    A 4000x3000 phone photo costs four times the detection compute of a
    1920x1440 one for no measurable recall gain, because the face still spans
    hundreds of pixels. Images already within the limit are returned untouched
    with a scale of 1.0.

    Args:
        image: Source BGR array.
        max_long_side: Ceiling on the longer edge, in pixels.

    Returns:
        ``(image, scale)`` where multiplying a coordinate in the returned image
        by ``1 / scale`` recovers the original coordinate.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_long_side:
        return image, 1.0

    scale = max_long_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return _as_u8(resized), scale


def ensure_min_size(image: BgrImage, min_short_side: int) -> tuple[BgrImage, float]:
    """Upscale so the shorter edge is at least ``min_short_side``.

    Upscaling adds no information, but detectors have a minimum receptive-field
    size below which a face is simply invisible to them. Recovering a genuine
    detection from a small image beats reporting FACE_NOT_DETECTED; the quality
    module independently penalises the low native resolution.

    Returns:
        ``(image, scale)`` with the same convention as :func:`resize_long_side`.
    """
    height, width = image.shape[:2]
    shortest = min(height, width)
    if shortest >= min_short_side:
        return image, 1.0

    scale = min_short_side / float(max(shortest, 1))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return _as_u8(resized), scale


def crop_with_margin(
    image: BgrImage,
    box: BoundingBox,
    *,
    margin: float = 0.0,
    square: bool = False,
    pad_value: int = 114,
) -> tuple[BgrImage, BoundingBox]:
    """Crop a region, expanding by ``margin`` and padding beyond the frame.

    Padding rather than clipping matters for faces near an image edge: clipping
    silently changes the crop's aspect ratio and shifts the face off-centre,
    which shifts the alignment landmarks and degrades the embedding.

    Args:
        image: Source BGR array.
        box: Region of interest in source coordinates.
        margin: Fractional expansion applied before cropping.
        square: Expand to a centred square first.
        pad_value: Fill value for out-of-frame regions.

    Returns:
        ``(crop, effective_box)`` where ``effective_box`` is the region actually
        requested, in source coordinates, including any out-of-frame part.
    """
    height, width = image.shape[:2]

    region = box.expand(margin) if margin > 0.0 else box
    if square:
        region = region.to_square()

    x1, y1, x2, y2 = region.as_int_tuple()
    crop_width = max(1, x2 - x1)
    crop_height = max(1, y2 - y1)

    canvas = np.full((crop_height, crop_width, 3), pad_value, dtype=np.uint8)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(width, x2)
    src_y2 = min(height, y2)

    if src_x2 > src_x1 and src_y2 > src_y1:
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        canvas[
            dst_y1 : dst_y1 + (src_y2 - src_y1),
            dst_x1 : dst_x1 + (src_x2 - src_x1),
        ] = image[src_y1:src_y2, src_x1:src_x2]

    return canvas, BoundingBox(float(x1), float(y1), float(x2), float(y2))


def to_grayscale(image: npt.NDArray[Any]) -> GrayImage:
    """Convert a BGR image to single-channel uint8 luminance.

    Idempotent: an image that is already single-channel is returned unchanged.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0].astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[2] == 3:
        return _as_u8(
            cv2.cvtColor(array.astype(np.uint8, copy=False), cv2.COLOR_BGR2GRAY)
        )
    if array.ndim == 3 and array.shape[2] == 4:
        return _as_u8(
            cv2.cvtColor(array.astype(np.uint8, copy=False), cv2.COLOR_BGRA2GRAY)
        )
    raise ValueError(f"Cannot convert array of shape {array.shape} to grayscale")


def convert_to_bgr(image: npt.NDArray[Any]) -> BgrImage:
    """Coerce any common array layout to contiguous 3-channel BGR uint8."""
    raw = np.asarray(image)
    array: BgrImage = (
        raw.astype(np.uint8, copy=False)
        if raw.dtype == np.uint8
        else np.clip(raw, 0, 255).astype(np.uint8)
    )
    if array.ndim == 2:
        return np.ascontiguousarray(cv2.cvtColor(array, cv2.COLOR_GRAY2BGR))
    if array.ndim == 3:
        channels = array.shape[2]
        if channels == 1:
            return np.ascontiguousarray(cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR))
        if channels == 3:
            return np.ascontiguousarray(array)
        if channels == 4:
            return np.ascontiguousarray(cv2.cvtColor(array, cv2.COLOR_BGRA2BGR))
    raise ValueError(f"Cannot interpret array of shape {array.shape} as an image")


def build_blob(
    image: BgrImage,
    size: tuple[int, int],
    *,
    mean: tuple[float, float, float],
    scale: float,
    swap_rb: bool,
) -> npt.NDArray[np.float32]:
    """Produce an NCHW float32 tensor from a BGR image.

    Equivalent to ``cv2.dnn.blobFromImage`` but implemented explicitly so the
    exact normalisation is visible and unit-testable, and so no dependency on
    the optional ``cv2.dnn`` module is introduced.

    Args:
        image: Source BGR uint8 array, already at the network's input size.
        size: Expected ``(width, height)``; the image is resized if it differs.
        mean: Per-channel value subtracted before scaling, in BGR order.
        scale: Multiplier applied after mean subtraction.
        swap_rb: Swap the red and blue channels (BGR -> RGB).

    Returns:
        A ``(1, 3, H, W)`` float32 tensor.
    """
    target_width, target_height = size
    if (image.shape[1], image.shape[0]) != (target_width, target_height):
        current_scale = target_width / max(image.shape[1], 1)
        image = _as_u8(
            cv2.resize(
                image,
                (target_width, target_height),
                interpolation=_interpolation_for(current_scale),
            )
        )

    tensor = image.astype(np.float32)
    if swap_rb:
        tensor = tensor[:, :, ::-1]
        mean = (mean[2], mean[1], mean[0])

    tensor -= np.array(mean, dtype=np.float32)
    tensor *= float(scale)

    # HWC -> CHW -> NCHW, contiguous because ORT copies non-contiguous input.
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def align_face(
    image: BgrImage,
    landmarks: Landmarks5,
    reference: npt.NDArray[Any],
    output_size: tuple[int, int],
) -> BgrImage:
    """Warp a face onto a canonical landmark template.

    Estimates the similarity transform (rotation, uniform scale, translation -
    no shear) that best maps the detected landmarks onto ``reference``, then
    applies it. This is the standard ArcFace pre-processing step and is worth
    several points of verification accuracy over a naive box crop, because it
    removes in-plane rotation and normalises inter-ocular distance.

    Args:
        image: Source BGR array.
        landmarks: Five detected keypoints in source coordinates.
        reference: ``(5, 2)`` destination template, in output-size coordinates.
        output_size: Desired ``(width, height)``.

    Returns:
        The aligned crop.
    """
    source = np.asarray(landmarks.points, dtype=np.float32).reshape(5, 2)
    destination = np.asarray(reference, dtype=np.float32).reshape(5, 2)

    estimated = cv2.estimateAffinePartial2D(
        source, destination, method=cv2.LMEDS, refineIters=20
    )
    # The stubs declare a non-optional matrix; the call genuinely returns
    # None on a degenerate point set, so the cast restores the real contract
    # and keeps the fallback below type-checked rather than unreachable.
    matrix = cast("npt.NDArray[np.float32] | None", estimated[0])
    if matrix is None:
        # LMEDS can fail on near-degenerate landmark sets. Fall back to a
        # two-point similarity from the eye line, which always has a solution.
        matrix = _similarity_from_eyes(source, destination)

    return _as_u8(
        cv2.warpAffine(
            image,
            matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )
    )


def _similarity_from_eyes(
    source: npt.NDArray[np.float32], destination: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Closed-form similarity transform aligning two eye pairs."""
    src_vector = source[1] - source[0]
    dst_vector = destination[1] - destination[0]

    src_norm = float(np.linalg.norm(src_vector))
    dst_norm = float(np.linalg.norm(dst_vector))
    scale = dst_norm / max(src_norm, EPSILON)

    angle = float(
        np.arctan2(dst_vector[1], dst_vector[0]) - np.arctan2(src_vector[1], src_vector[0])
    )
    cos_a = float(np.cos(angle)) * scale
    sin_a = float(np.sin(angle)) * scale

    src_center = (source[0] + source[1]) / 2.0
    dst_center = (destination[0] + destination[1]) / 2.0

    return np.array(
        [
            [cos_a, -sin_a, dst_center[0] - (cos_a * src_center[0] - sin_a * src_center[1])],
            [sin_a, cos_a, dst_center[1] - (sin_a * src_center[0] + cos_a * src_center[1])],
        ],
        dtype=np.float32,
    )


def skin_mask(image: BgrImage) -> npt.NDArray[np.uint8]:
    """Binary skin-tone mask using an elliptical model in YCrCb space.

    Chrominance-only classification is deliberately illumination-invariant and
    generalises across skin tones far better than an RGB or HSV box, which is
    essential for a Pakistani user base. Used by the occlusion analyser to
    measure how much of each facial region shows skin rather than fabric,
    plastic or a hand.

    Args:
        image: Source BGR array.

    Returns:
        A ``(H, W)`` uint8 mask, 255 where the pixel is skin-like.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)

    # Ellipse inscribed in the Cr in [133, 173], Cb in [77, 127] region that the
    # skin-segmentation literature converges on across the multi-ethnic
    # benchmarks (Compaq / SFA / Pratheepan).
    #
    # Calibrated empirically: an earlier, tighter ellipse centred at Cb=112
    # scored genuine mouth and chin skin at only 0.22 coverage on a real
    # portrait, because its Cb band sat well above the true distribution. That
    # false deficit fed straight into the occlusion estimator. The centre and
    # axes below are the midpoint and half-width of the published ranges.
    center_cr, center_cb = 153.0, 102.0
    axis_cr, axis_cb = 20.0, 25.0

    cr_f = cr.astype(np.float32)
    cb_f = cb.astype(np.float32)
    distance = ((cr_f - center_cr) / axis_cr) ** 2 + ((cb_f - center_cb) / axis_cb) ** 2

    mask = (distance <= 1.0).astype(np.uint8) * 255

    # Close single-pixel gaps from JPEG chroma subsampling, then drop specks.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return np.asarray(
        cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel), dtype=np.uint8
    )


def gradient_energy(gray: GrayImage) -> npt.NDArray[np.float32]:
    """Per-pixel Sobel gradient magnitude, as float32.

    The core texture measure behind both occlusion detection (a mask or a hand
    is texturally flat compared with skin and eyes) and the Tenengrad focus
    metric in Module 2.
    """
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.asarray(cv2.magnitude(grad_x, grad_y), dtype=np.float32)


__all__ = [
    "BgrImage",
    "GrayImage",
    "LetterboxTransform",
    "align_face",
    "build_blob",
    "convert_to_bgr",
    "crop_with_margin",
    "ensure_min_size",
    "gradient_energy",
    "letterbox",
    "resize_long_side",
    "skin_mask",
    "to_grayscale",
]
