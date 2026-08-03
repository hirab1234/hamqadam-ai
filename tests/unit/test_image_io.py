"""Image codec hardening.

Decoding attacker-supplied bytes is the largest attack surface in the service.
Every rejection path below corresponds to a real class of hostile or malformed
input, and each must fail with a *specific* error code rather than a generic
exception, because the Backend switches on those codes to tell the user what
to fix.
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from hamqadam_ai.core.constants import MIN_IMAGE_DIMENSION
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import (
    DecompressionBombError,
    ImageDecodeError,
    ImageTooLargeError,
    ImageTooSmallError,
    UnsupportedImageFormatError,
)
from hamqadam_ai.utils.image_io import (
    decode_image,
    decode_image_b64,
    encode_image,
    probe_dimensions,
    sniff_format,
)


def textured(width: int = 320, height: int = 240) -> np.ndarray:
    """A deterministic non-uniform image that passes the detail check."""
    rng = np.random.default_rng(20260727)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:, :, 0] = np.linspace(10, 245, width, dtype=np.uint8)[None, :]
    base[:, :, 1] = np.linspace(20, 235, height, dtype=np.uint8)[:, None]
    base[:, :, 2] = rng.integers(0, 256, (height, width), dtype=np.uint8)
    return base


# --------------------------------------------------------------------------- #
# Format sniffing
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_round_trip_through_every_supported_format(fmt: str) -> None:
    encoded = encode_image(textured(), fmt=fmt)
    decoded = decode_image(encoded, role="test")
    assert decoded.source_format == fmt
    assert decoded.shape == (240, 320, 3)
    assert decoded.pixels.dtype == np.uint8


@pytest.mark.unit
def test_format_is_sniffed_from_magic_bytes_not_declared_type() -> None:
    """A caller-declared content type is never trusted."""
    assert sniff_format(encode_image(textured(), fmt="PNG")[:32]) == "PNG"
    assert sniff_format(encode_image(textured(), fmt="JPEG")[:32]) == "JPEG"
    assert sniff_format(b"GIF89a" + b"\x00" * 26) is None


@pytest.mark.unit
def test_riff_container_that_is_not_webp_is_rejected() -> None:
    """RIFF also fronts AVI and WAV; only the WEBP marker counts."""
    wav = b"RIFF" + (1000).to_bytes(4, "little") + b"WAVEfmt "
    assert sniff_format(wav) is None
    with pytest.raises(UnsupportedImageFormatError):
        decode_image(wav + b"\x00" * 512, role="test")


@pytest.mark.unit
def test_unsupported_format_names_what_is_accepted() -> None:
    with pytest.raises(UnsupportedImageFormatError) as excinfo:
        decode_image(b"GIF89a" + b"\x00" * 512, role="cnic_image")
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_IMAGE_FORMAT
    assert "JPEG" in excinfo.value.details["supported"]
    assert excinfo.value.details["role"] == "cnic_image"


# --------------------------------------------------------------------------- #
# Structural rejection
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_empty_payload_is_rejected() -> None:
    with pytest.raises(ImageDecodeError):
        decode_image(b"", role="test")


@pytest.mark.unit
def test_oversized_payload_is_rejected_before_decoding() -> None:
    with pytest.raises(ImageTooLargeError) as excinfo:
        decode_image(b"\xff\xd8\xff" + b"\x00" * 4096, role="test", max_bytes=1024)
    assert excinfo.value.details["limit_bytes"] == 1024


@pytest.mark.unit
def test_truncated_jpeg_is_detected_by_its_missing_trailer() -> None:
    full = encode_image(textured(), fmt="JPEG")
    with pytest.raises(ImageDecodeError, match="truncated"):
        decode_image(full[: len(full) // 2], role="test")


@pytest.mark.unit
def test_truncated_png_is_detected_by_its_missing_iend() -> None:
    full = encode_image(textured(), fmt="PNG")
    with pytest.raises(ImageDecodeError, match="truncated"):
        decode_image(full[: len(full) - 12], role="test")


@pytest.mark.unit
def test_corrupt_body_with_valid_header_is_rejected() -> None:
    """Right magic bytes, right trailer, garbage in between."""
    payload = b"\xff\xd8\xff\xe0" + b"\x5a" * 2048 + b"\xff\xd9"
    with pytest.raises(ImageDecodeError):
        decode_image(payload, role="test")


@pytest.mark.unit
def test_uniform_image_carries_no_usable_detail() -> None:
    flat = np.full((200, 200, 3), 137, dtype=np.uint8)
    with pytest.raises(ImageDecodeError, match="uniform"):
        decode_image(encode_image(flat, fmt="PNG"), role="test")


# --------------------------------------------------------------------------- #
# Resource-exhaustion guards
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_dimensions_are_probed_without_allocating_pixels() -> None:
    encoded = encode_image(textured(1280, 720), fmt="PNG")
    assert probe_dimensions(encoded) == (1280, 720)


@pytest.mark.unit
def test_decompression_bomb_is_refused_on_the_header_alone() -> None:
    """A large, highly-compressible PNG must be rejected before decoding."""
    bomb = np.zeros((4000, 4000, 3), dtype=np.uint8)
    bomb[::7, ::11] = 255  # keep it non-uniform but still tiny once deflated
    encoded = encode_image(bomb, fmt="PNG")

    with pytest.raises(DecompressionBombError) as excinfo:
        decode_image(encoded, role="test", max_pixels=1_000_000)
    assert excinfo.value.details["width"] == 4000
    assert excinfo.value.details["pixel_limit"] == 1_000_000


@pytest.mark.unit
def test_image_below_the_dimension_floor_is_rejected() -> None:
    tiny = textured(32, 32)
    with pytest.raises(ImageTooSmallError) as excinfo:
        decode_image(encode_image(tiny, fmt="PNG"), role="live_selfie")
    assert excinfo.value.details["minimum"] == MIN_IMAGE_DIMENSION


# --------------------------------------------------------------------------- #
# Colour handling
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_transparent_png_is_composited_onto_white_not_black() -> None:
    """Flattening alpha to black would destroy every brightness statistic."""
    rgba = np.zeros((120, 120, 4), dtype=np.uint8)
    rgba[:, :, 3] = 0  # fully transparent
    rgba[40:80, 40:80] = (200, 30, 30, 255)  # an opaque red square

    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    decoded = decode_image(buffer.getvalue(), role="test")

    assert decoded.had_alpha is True
    corner = decoded.pixels[5, 5]
    assert corner.tolist() == [255, 255, 255], "transparent area should be white"


@pytest.mark.unit
def test_decoded_output_is_bgr_channel_order() -> None:
    """A pure-red RGB source must come back as BGR (0, 0, 255)."""
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[:, :, 0] = 220  # red in RGB
    rgb[0, 0] = (5, 5, 5)  # break uniformity

    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    decoded = decode_image(buffer.getvalue(), role="test")

    assert decoded.pixels[50, 50].tolist() == [0, 0, 220]


@pytest.mark.unit
def test_grayscale_source_is_expanded_to_three_channels() -> None:
    gray = np.linspace(0, 255, 128 * 128, dtype=np.uint8).reshape(128, 128)
    buffer = io.BytesIO()
    Image.fromarray(gray, mode="L").save(buffer, format="PNG")
    decoded = decode_image(buffer.getvalue(), role="test")
    assert decoded.shape[2] == 3


@pytest.mark.unit
def test_exif_orientation_is_applied() -> None:
    """Phone cameras store landscape sensor data plus a rotation flag."""
    portrait = textured(200, 100)  # wide
    buffer = io.BytesIO()
    image = Image.fromarray(cv2.cvtColor(portrait, cv2.COLOR_BGR2RGB))
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise
    image.save(buffer, format="JPEG", exif=exif)

    decoded = decode_image(buffer.getvalue(), role="live_selfie")
    assert decoded.exif_orientation == 6
    assert decoded.height > decoded.width, "orientation tag should have been applied"


@pytest.mark.unit
def test_exif_orientation_can_be_disabled() -> None:
    portrait = textured(200, 100)
    buffer = io.BytesIO()
    image = Image.fromarray(cv2.cvtColor(portrait, cv2.COLOR_BGR2RGB))
    exif = image.getexif()
    exif[0x0112] = 6
    image.save(buffer, format="JPEG", exif=exif)

    decoded = decode_image(
        buffer.getvalue(), role="test", apply_exif_orientation=False
    )
    assert decoded.width > decoded.height


# --------------------------------------------------------------------------- #
# Base64 transport
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_plain_base64_is_decoded() -> None:
    encoded = base64.b64encode(encode_image(textured(), fmt="JPEG")).decode()
    assert decode_image_b64(encoded, role="test").width == 320


@pytest.mark.unit
def test_data_uri_prefix_is_stripped() -> None:
    body = base64.b64encode(encode_image(textured(), fmt="PNG")).decode()
    assert decode_image_b64(f"data:image/png;base64,{body}", role="test").width == 320


@pytest.mark.unit
def test_missing_base64_padding_is_restored() -> None:
    """Several HTTP clients strip '=' padding in transit."""
    body = base64.b64encode(encode_image(textured(), fmt="JPEG")).decode()
    assert decode_image_b64(body.rstrip("="), role="test").width == 320


@pytest.mark.unit
def test_mime_line_breaks_are_tolerated() -> None:
    """MIME base64 is line-wrapped at 76 characters; strict decoding must not
    trip over that."""
    body = base64.encodebytes(encode_image(textured(), fmt="JPEG")).decode()
    assert "\n" in body
    assert decode_image_b64(body, role="test").width == 320


@pytest.mark.unit
def test_url_safe_alphabet_is_accepted() -> None:
    body = base64.urlsafe_b64encode(encode_image(textured(), fmt="PNG")).decode()
    assert decode_image_b64(body, role="test").width == 320


@pytest.mark.unit
def test_invalid_base64_is_reported_as_a_decode_error() -> None:
    """Must be a base64 error, not a format error.

    Python's decoder silently discards non-alphabet characters unless
    ``validate=True``, which would turn corrupt transport into a misleading
    UNSUPPORTED_IMAGE_FORMAT and send the user off to convert their JPEG.
    """
    with pytest.raises(ImageDecodeError) as excinfo:
        decode_image_b64("!!! definitely not base64 !!!", role="test")
    assert excinfo.value.code is ErrorCode.IMAGE_DECODE_FAILED
    assert "base64" in excinfo.value.message


@pytest.mark.unit
def test_whitespace_only_base64_body_is_rejected() -> None:
    with pytest.raises(ImageDecodeError, match="no base64 data"):
        decode_image_b64("   \n\t  ", role="test")


@pytest.mark.unit
def test_empty_data_uri_body_is_rejected() -> None:
    with pytest.raises(ImageDecodeError, match="no base64 body"):
        decode_image_b64("data:image/png;base64,", role="test")


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_describe_is_pii_free_and_serialisable() -> None:
    decoded = decode_image(encode_image(textured(), fmt="JPEG"), role="test")
    summary = decoded.describe()
    assert set(summary) == {
        "width",
        "height",
        "format",
        "encoded_bytes",
        "megapixels",
        "had_alpha",
    }
    assert all(isinstance(v, int | float | str | bool) for v in summary.values())


@pytest.mark.unit
def test_encode_rejects_an_unknown_format() -> None:
    with pytest.raises(UnsupportedImageFormatError):
        encode_image(textured(), fmt="TIFF")
