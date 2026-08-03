"""Hardened image decoding.

Decoding attacker-supplied bytes is the single largest attack surface in this
service, so the pipeline is deliberately paranoid and ordered cheapest-check-
first:

1. Size check on the raw bytes - rejects a 200 MB upload without touching it.
2. Magic-byte sniff - the declared content type is ignored entirely; only the
   real signature counts.
3. Header-only dimension probe - the pixel count is known *before* any pixel
   buffer is allocated, which is what actually stops a decompression bomb.
4. Decode.
5. Post-decode structural validation - channel count, dtype, finite values.

Decoded images are always ``uint8`` BGR ``(H, W, 3)``, the OpenCV convention
used throughout the codebase. Alpha is composited onto white rather than
dropped: dropping it turns a transparent PNG's background black, which
destroys the brightness statistics Module 2 depends on.
"""

from __future__ import annotations

import base64
import binascii
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFile, UnidentifiedImageError

from hamqadam_ai.core.constants import (
    IMAGE_MAGIC_BYTES,
    MAX_DECODED_PIXELS,
    MAX_ENCODED_IMAGE_BYTES,
    MIN_IMAGE_DIMENSION,
    SUPPORTED_IMAGE_FORMATS,
)
from hamqadam_ai.core.exceptions import (
    DecompressionBombError,
    ImageDecodeError,
    ImageTooLargeError,
    ImageTooSmallError,
    UnsupportedImageFormatError,
)

BgrImage = npt.NDArray[np.uint8]

# A truncated JPEG should raise, not silently yield a half-grey image that then
# fails face detection for reasons nobody can diagnose from the logs.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Pillow's own bomb guard. Ours is stricter and produces a typed error, but
# this stops Pillow allocating before our check runs on exotic formats.
Image.MAX_IMAGE_PIXELS = MAX_DECODED_PIXELS

#: Whitespace legal inside a MIME base64 body, stripped before strict decoding.
_BASE64_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")

_JPEG_TRAILER: Final[bytes] = b"\xff\xd9"
_PNG_TRAILER: Final[bytes] = b"IEND\xaeB`\x82"


@dataclass(frozen=True, slots=True)
class DecodedImage:
    """A validated, decoded image plus the provenance needed for auditing.

    Attributes:
        pixels: ``(H, W, 3)`` uint8 array in BGR channel order.
        source_format: Container format detected from the magic bytes.
        encoded_bytes: Size of the original encoded payload.
        had_alpha: Whether an alpha channel was composited away.
        exif_orientation: The EXIF orientation tag that was applied, if any.
    """

    pixels: BgrImage
    source_format: str
    encoded_bytes: int
    had_alpha: bool = False
    exif_orientation: int | None = None

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.pixels.shape[1])

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(height, width, channels)``."""
        return (self.height, self.width, int(self.pixels.shape[2]))

    @property
    def megapixels(self) -> float:
        """Pixel count in millions."""
        return (self.width * self.height) / 1_000_000.0

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height."""
        return self.width / max(self.height, 1)

    def describe(self) -> dict[str, Any]:
        """PII-free summary safe to attach to a log line."""
        return {
            "width": self.width,
            "height": self.height,
            "format": self.source_format,
            "encoded_bytes": self.encoded_bytes,
            "megapixels": round(self.megapixels, 2),
            "had_alpha": self.had_alpha,
        }


def sniff_format(data: bytes) -> str | None:
    """Identify the container format from its magic bytes.

    The caller-declared MIME type is never trusted; only the actual signature
    determines which decoder path runs.

    Args:
        data: The first bytes of the encoded payload.

    Returns:
        One of :data:`~hamqadam_ai.core.constants.SUPPORTED_IMAGE_FORMATS`, or
        ``None`` when nothing matches.
    """
    for name, signatures in IMAGE_MAGIC_BYTES.items():
        for signature in signatures:
            if data.startswith(signature):
                # RIFF is also the AVI/WAV container; WEBP has a second marker.
                if name == "WEBP" and data[8:12] != b"WEBP":
                    continue
                return name
    return None


def probe_dimensions(data: bytes) -> tuple[int, int]:
    """Read the image dimensions from the header without decoding pixels.

    This is the check that actually prevents a decompression bomb: Pillow's
    lazy ``open`` parses the header only, so a 60000x60000 PNG is rejected
    having allocated nothing.

    Args:
        data: The complete encoded payload.

    Returns:
        ``(width, height)`` in pixels.

    Raises:
        ImageDecodeError: if the header cannot be parsed.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            return int(image.width), int(image.height)
    except UnidentifiedImageError as exc:
        raise ImageDecodeError(
            "The payload header does not describe a recognisable image.",
            cause=exc,
        ) from exc
    except (OSError, ValueError) as exc:
        raise ImageDecodeError(
            f"Image header could not be parsed: {exc}", cause=exc
        ) from exc


def _looks_truncated(data: bytes, fmt: str) -> bool:
    """Cheap end-of-stream marker check for the two formats that have one."""
    if fmt == "JPEG":
        return not data.rstrip(b"\x00").endswith(_JPEG_TRAILER)
    if fmt == "PNG":
        return not data.endswith(_PNG_TRAILER)
    return False


def decode_image(
    data: bytes,
    *,
    role: str = "image",
    max_bytes: int = MAX_ENCODED_IMAGE_BYTES,
    max_pixels: int = MAX_DECODED_PIXELS,
    min_dimension: int = MIN_IMAGE_DIMENSION,
    apply_exif_orientation: bool = True,
) -> DecodedImage:
    """Validate and decode encoded image bytes into a BGR array.

    Args:
        data: The encoded payload.
        role: Semantic role of the image, echoed into error details so the
            Backend can tell the user *which* photo was rejected.
        max_bytes: Ceiling on the encoded payload size.
        max_pixels: Ceiling on ``width * height`` after decoding.
        min_dimension: Floor on the shorter side.
        apply_exif_orientation: Honour the EXIF orientation tag. Phone cameras
            store landscape sensor data plus a rotation flag; skipping this
            step feeds sideways faces to the detector.

    Returns:
        A validated :class:`DecodedImage`.

    Raises:
        ImageTooLargeError: payload exceeds ``max_bytes``.
        UnsupportedImageFormatError: magic bytes are unrecognised.
        DecompressionBombError: decoded pixel count exceeds ``max_pixels``.
        ImageTooSmallError: shorter side is below ``min_dimension``.
        ImageDecodeError: the bytes are corrupt, truncated or undecodable.
    """
    if not data:
        raise ImageDecodeError(
            "Received an empty image payload.", details={"role": role}
        )

    if len(data) > max_bytes:
        raise ImageTooLargeError(
            f"Encoded image is {len(data)} bytes, over the {max_bytes} byte limit.",
            details={"role": role, "size_bytes": len(data), "limit_bytes": max_bytes},
        )

    fmt = sniff_format(data[:32])
    if fmt is None or fmt not in SUPPORTED_IMAGE_FORMATS:
        raise UnsupportedImageFormatError(
            "Image format could not be identified from its signature.",
            details={
                "role": role,
                "supported": sorted(SUPPORTED_IMAGE_FORMATS),
                "leading_bytes": data[:8].hex(),
            },
        )

    if _looks_truncated(data, fmt):
        raise ImageDecodeError(
            f"{fmt} payload is missing its end-of-stream marker and is truncated.",
            details={"role": role, "format": fmt, "size_bytes": len(data)},
        )

    width, height = probe_dimensions(data)
    if width * height > max_pixels:
        raise DecompressionBombError(
            f"Image decodes to {width}x{height} = {width * height} pixels, "
            f"over the {max_pixels} pixel limit.",
            details={
                "role": role,
                "width": width,
                "height": height,
                "pixel_limit": max_pixels,
            },
        )
    if min(width, height) < min_dimension:
        raise ImageTooSmallError(
            f"Image is {width}x{height}; the shorter side must be at least "
            f"{min_dimension} pixels.",
            details={"role": role, "width": width, "height": height, "minimum": min_dimension},
        )

    pixels, had_alpha, orientation = _decode_pixels(
        data, role=role, apply_exif_orientation=apply_exif_orientation
    )
    _validate_pixels(pixels, role=role)

    return DecodedImage(
        pixels=pixels,
        source_format=fmt,
        encoded_bytes=len(data),
        had_alpha=had_alpha,
        exif_orientation=orientation,
    )


def _decode_pixels(
    data: bytes, *, role: str, apply_exif_orientation: bool
) -> tuple[BgrImage, bool, int | None]:
    """Decode to BGR uint8, compositing alpha and applying EXIF orientation.

    Pillow is used rather than ``cv2.imdecode`` because OpenCV silently ignores
    EXIF orientation for in-memory buffers and returns ``None`` on failure
    instead of raising something diagnosable.
    """
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image: Image.Image = opened
            orientation = _exif_orientation(image) if apply_exif_orientation else None
            if apply_exif_orientation and orientation not in (None, 1):
                from PIL import ImageOps

                image = ImageOps.exif_transpose(image) or image

            had_alpha = image.mode in {"RGBA", "LA", "PA"} or (
                image.mode == "P" and "transparency" in image.info
            )

            if had_alpha:
                rgba = image.convert("RGBA")
                # Composite onto white: a transparent background flattened to
                # black would skew every brightness and contrast metric.
                canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(canvas, rgba).convert("RGB")
            else:
                image = image.convert("RGB")

            rgb = np.asarray(image, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ImageDecodeError(
            "The payload is not a decodable image.",
            details={"role": role},
            cause=exc,
        ) from exc
    except (OSError, ValueError) as exc:
        raise ImageDecodeError(
            f"Image decoding failed: {exc}", details={"role": role}, cause=exc
        ) from exc

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ImageDecodeError(
            f"Decoded array has unexpected shape {rgb.shape}; expected (H, W, 3).",
            details={"role": role, "shape": list(rgb.shape)},
        )

    # np.ascontiguousarray: OpenCV requires a contiguous buffer and Pillow's
    # view of a transposed image is not.
    bgr = np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return bgr.astype(np.uint8, copy=False), had_alpha, orientation


def _exif_orientation(image: Image.Image) -> int | None:
    """Return the EXIF orientation tag (0x0112), or None when absent."""
    try:
        exif = image.getexif()
    except Exception:  # noqa: BLE001 - malformed EXIF must not fail the request
        return None
    if not exif:
        return None
    value = exif.get(0x0112)
    return int(value) if isinstance(value, int) else None


def _validate_pixels(pixels: BgrImage, *, role: str) -> None:
    """Reject decoded arrays that are structurally unusable.

    A uniform image passes every earlier check but carries no information: it
    is either a lens-cap frame or a synthetic probe, and running the full
    pipeline on it wastes several seconds to reach the same conclusion.
    """
    if pixels.dtype != np.uint8:
        raise ImageDecodeError(
            f"Decoded image has dtype {pixels.dtype}; expected uint8.",
            details={"role": role, "dtype": str(pixels.dtype)},
        )
    if pixels.size == 0:
        raise ImageDecodeError("Decoded image is empty.", details={"role": role})

    # Sample rather than scan: on a 12 MP image the full min/max costs ~10 ms
    # and a 64x64 sample answers the question just as well.
    sample = pixels[:: max(1, pixels.shape[0] // 64), :: max(1, pixels.shape[1] // 64)]
    if int(sample.max()) == int(sample.min()):
        raise ImageDecodeError(
            "Image contains a single uniform colour and carries no usable detail.",
            details={"role": role, "uniform_value": int(sample.min())},
        )


def decode_image_b64(payload: str, *, role: str = "image", **kwargs: Any) -> DecodedImage:
    """Decode a base64 (optionally data-URI) encoded image.

    Accepts both the bare base64 body and the ``data:image/jpeg;base64,...``
    form the Flutter app produces, plus URL-safe alphabets and missing padding.

    Args:
        payload: The base64 string.
        role: Semantic role, echoed into error details.
        **kwargs: Forwarded to :func:`decode_image`.

    Raises:
        ImageDecodeError: if the string is not valid base64.
    """
    text = payload.strip()
    if text.startswith("data:"):
        _, _, text = text.partition(",")
        if not text:
            raise ImageDecodeError(
                "Data URI contains no base64 body.", details={"role": role}
            )

    # Line breaks are legal in MIME base64 and several HTTP clients insert
    # them, but strict validation rejects them, so they are removed first.
    text = _BASE64_WHITESPACE.sub("", text)
    if not text:
        raise ImageDecodeError(
            "Payload contains no base64 data.", details={"role": role}
        )

    # URL-safe alphabet: translate rather than pass `altchars`, so that a
    # payload mixing both alphabets still decodes instead of half-failing.
    if "-" in text or "_" in text:
        text = text.replace("-", "+").replace("_", "/")

    # Restore stripped padding; several HTTP clients drop it.
    padding = (-len(text)) % 4
    if padding:
        text += "=" * padding

    try:
        # validate=True is essential. With the default the decoder *silently
        # discards* every character outside the base64 alphabet, so arbitrary
        # garbage decodes to arbitrary bytes and the failure then surfaces
        # several steps later as UNSUPPORTED_IMAGE_FORMAT. That misdirects the
        # Backend into telling the user to change image format when the real
        # problem is a corrupted upload.
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeError(
            "Payload is not valid base64.", details={"role": role}, cause=exc
        ) from exc

    return decode_image(data, role=role, **kwargs)


def load_image(path: str | Path, *, role: str = "image", **kwargs: Any) -> DecodedImage:
    """Read and decode an image from disk.

    Used by the evaluation harness and the demo scripts; the service itself
    never reads user images from the filesystem.

    Args:
        path: Filesystem path to the image.
        role: Semantic role, echoed into error details.
        **kwargs: Forwarded to :func:`decode_image`.

    Raises:
        ImageDecodeError: if the file is missing or unreadable.
    """
    file_path = Path(path)
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise ImageDecodeError(
            f"Could not read image file: {exc}",
            details={"role": role, "path": str(file_path)},
            cause=exc,
        ) from exc
    return decode_image(data, role=role, **kwargs)


def encode_image(
    pixels: npt.NDArray[Any], *, fmt: str = "JPEG", quality: int = 92
) -> bytes:
    """Encode a BGR array back to bytes.

    Only used for debug artefacts and the annotated demo output. No user image
    is ever encoded and persisted by the service itself.

    Args:
        pixels: ``(H, W, 3)`` BGR uint8 array.
        fmt: ``JPEG``, ``PNG`` or ``WEBP``.
        quality: Encoder quality for the lossy formats, 1-100.

    Raises:
        UnsupportedImageFormatError: for an unknown ``fmt``.
        ImageDecodeError: if the encoder rejects the array.
    """
    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt.upper())
    if extension is None:
        raise UnsupportedImageFormatError(
            f"Cannot encode to {fmt!r}.", details={"supported": ["JPEG", "PNG", "WEBP"]}
        )

    params: list[int] = []
    if extension == ".jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))]
    elif extension == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(np.clip(quality, 1, 100))]
    elif extension == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]

    ok, buffer = cv2.imencode(extension, np.ascontiguousarray(pixels), params)
    if not ok:
        raise ImageDecodeError(
            f"OpenCV failed to encode the array as {fmt}.",
            details={"format": fmt, "shape": list(np.shape(pixels))},
        )
    return bytes(buffer.tobytes())


__all__ = [
    "BgrImage",
    "DecodedImage",
    "decode_image",
    "decode_image_b64",
    "encode_image",
    "load_image",
    "probe_dimensions",
    "sniff_format",
]
