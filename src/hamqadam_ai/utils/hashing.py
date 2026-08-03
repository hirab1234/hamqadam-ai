"""Cryptographic and perceptual hashing.

Two unrelated jobs share this module:

* **Integrity** - SHA-256 of model artefacts, verified before a file is handed
  to ONNX Runtime. A tampered detector is a silent, total compromise of the
  verification decision, so this check is not optional in production.
* **Content identity** - perceptual hashes of images, used as embedding-cache
  keys and by the fraud engine to spot the same photo resubmitted across
  accounts with trivial re-encoding.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.utils.image_ops import to_grayscale

#: Read models in 1 MiB chunks so a 300 MB artefact never lands in memory twice.
_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = _CHUNK_SIZE) -> str:
    """Return the lowercase hex SHA-256 digest of a file, read in chunks.

    Args:
        path: File to digest.
        chunk_size: Read granularity in bytes.

    Raises:
        OSError: if the file cannot be read.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> bool:
    """Constant-time comparison of a file's digest against ``expected``.

    Uses :func:`hashlib.compare_digest` rather than ``==`` so the comparison
    does not leak digest content through its timing.
    """
    import hmac

    return hmac.compare_digest(sha256_file(path).lower(), expected.strip().lower())


def sha256_array(array: npt.NDArray[Any]) -> str:
    """Digest a numpy array's contents, shape and dtype.

    Including the shape and dtype in the digest prevents two arrays with the
    same bytes but different interpretations from colliding, which would
    otherwise be possible for a reshaped buffer.
    """
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def content_key(data: bytes, *, namespace: str = "img") -> str:
    """Build a short, collision-resistant cache key for a byte payload.

    Args:
        data: The payload, typically encoded image bytes.
        namespace: Prefix separating key spaces in a shared Redis instance.

    Returns:
        ``"<namespace>:<32 hex chars>"``. 128 bits of a SHA-256 digest is well
        beyond the collision risk of any realistic cache population.
    """
    return f"{namespace}:{sha256_bytes(data)[:32]}"


# --------------------------------------------------------------------------- #
# Perceptual hashing
# --------------------------------------------------------------------------- #


def average_hash(image: npt.NDArray[Any], *, hash_size: int = 8) -> int:
    """Compute the aHash perceptual hash of an image.

    Downsamples to ``hash_size x hash_size`` and thresholds each pixel against
    the mean. Cheap, and robust to re-encoding, mild rescaling and small
    brightness shifts - exactly the transformations someone applies when
    resubmitting a stolen photo.

    Args:
        image: Source image, colour or grayscale.
        hash_size: Edge length of the reduced image; the hash is
            ``hash_size**2`` bits.

    Returns:
        The hash as an integer, comparable with :func:`hamming_distance`.
    """
    gray = to_grayscale(image)
    reduced = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = float(reduced.mean())
    bits = (reduced > mean).flatten()

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def difference_hash(image: npt.NDArray[Any], *, hash_size: int = 8) -> int:
    """Compute the dHash perceptual hash of an image.

    Encodes the sign of the horizontal gradient between adjacent pixels. More
    discriminative than aHash on images with a flat histogram - such as a
    document photographed against a plain background - because it keys off
    structure rather than absolute intensity.

    Args:
        image: Source image, colour or grayscale.
        hash_size: The reduced image is ``(hash_size + 1) x hash_size``.

    Returns:
        The hash as an integer.
    """
    gray = to_grayscale(image)
    reduced = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (reduced[:, 1:] > reduced[:, :-1]).flatten()

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def perceptual_hash(image: npt.NDArray[Any], *, hash_size: int = 8) -> int:
    """Compute the pHash perceptual hash using a discrete cosine transform.

    The most robust of the three: retaining only the low-frequency DCT
    coefficients makes it largely invariant to gamma changes, JPEG artefacts
    and moderate cropping. Correspondingly the most expensive, so it is applied
    only where the fraud engine needs high confidence.

    Args:
        image: Source image, colour or grayscale.
        hash_size: Number of low-frequency coefficients kept per axis.

    Returns:
        The hash as an integer.
    """
    gray = to_grayscale(image)
    # Transform at 4x the hash size so the retained coefficients are genuinely
    # low-frequency relative to the source detail.
    transform_size = hash_size * 4
    reduced = cv2.resize(
        gray, (transform_size, transform_size), interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    dct = cv2.dct(reduced)
    low_frequency = dct[:hash_size, :hash_size]

    # The DC term encodes overall brightness and would dominate the median.
    coefficients = low_frequency.flatten()
    median = float(np.median(coefficients[1:]))
    bits = coefficients > median

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Number of differing bits between two hashes.

    For 64-bit perceptual hashes the conventional interpretation is:
    0-5 near-identical, 6-10 probably the same source image, above 10 different.
    """
    return int(bin(hash_a ^ hash_b).count("1"))


def hash_to_hex(value: int, *, bits: int = 64) -> str:
    """Render a hash integer as fixed-width hex for storage and logging."""
    return format(value, f"0{bits // 4}x")


__all__ = [
    "average_hash",
    "content_key",
    "difference_hash",
    "hamming_distance",
    "hash_to_hex",
    "perceptual_hash",
    "sha256_array",
    "sha256_bytes",
    "sha256_file",
    "verify_sha256",
]
