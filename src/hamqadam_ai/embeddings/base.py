"""The face-embedder port and its value objects.

Contract every adapter honours
------------------------------
* Input is an **aligned** crop at the model's native size, produced by
  :mod:`hamqadam_ai.embeddings.alignment`. An adapter never does its own
  detection or alignment - that separation is what lets the CNIC portrait
  path in Module 6 reuse the identical embedder with a different crop source.
* Output vectors are **L2-normalised**, so cosine similarity is a dot product.
* The pre-normalisation magnitude is preserved as :attr:`FaceEmbedding.raw_norm`
  rather than discarded.
* Embedding is **deterministic**. The same aligned crop must produce a
  bit-identical vector on every call, on pain of a cached and an uncached
  comparison disagreeing.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.constants import EPSILON, ImageRole

FloatArray = npt.NDArray[np.float32]
BgrImage = npt.NDArray[np.uint8]


def l2_normalise(vector: npt.NDArray[Any]) -> FloatArray:
    """Scale a vector to unit length.

    Args:
        vector: Any real-valued vector.

    Returns:
        The unit-length vector as float32. A zero or near-zero vector is
        returned unchanged rather than amplified into numerical noise - a
        degenerate embedding should stay obviously degenerate, and dividing by
        1e-30 would turn it into a confident-looking direction pointing
        nowhere in particular.
    """
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= EPSILON:
        return array
    return (array / norm).astype(np.float32)


def cosine_similarity(left: npt.NDArray[Any], right: npt.NDArray[Any]) -> float:
    """Cosine similarity between two vectors, in ``[-1, 1]``.

    Normalises defensively rather than assuming its inputs are already unit
    length, because a caller passing a raw vector would otherwise get a
    silently wrong number instead of an error.
    """
    a = l2_normalise(left)
    b = l2_normalise(right)
    if a.shape != b.shape:
        raise ValueError(
            f"Cannot compare embeddings of different dimensions: "
            f"{a.shape[0]} vs {b.shape[0]}"
        )
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    """A face's biometric template plus the provenance needed to trust it.

    Attributes:
        vector: L2-normalised float32 vector of :attr:`dimension` elements.
        raw_norm: L2 norm of the network output *before* normalisation.
            ArcFace's angular-margin loss never constrains this magnitude, and
            empirically it tracks face quality - the observation MagFace and
            AdaFace build on. Reported as an independent confidence signal.
        confidence: :attr:`raw_norm` mapped through the configured ramp into
            ``[0, 1]``.
        model_key: Registry key of the model that produced the vector.
        model_version: Pinned version of that model. Two embeddings with
            different versions are **not comparable** and the matcher refuses
            to compare them.
        aligned: Whether the crop was warped onto the landmark template. False
            means the box-only fallback ran and accuracy is materially lower.
        alignment_residual: Mean landmark-to-template distance after warping,
            in units of interocular distance. ``None`` for the fallback path.
        flip_averaged: Whether the mirrored crop was embedded and averaged in.
        cache_hit: Whether this came from the cache rather than a forward pass.
        role: Which image in the verification request this face came from.
    """

    vector: FloatArray
    raw_norm: float
    confidence: float
    model_key: str
    model_version: str
    aligned: bool = True
    alignment_residual: float | None = None
    flip_averaged: bool = False
    cache_hit: bool = False
    role: ImageRole | None = None

    def __post_init__(self) -> None:
        array = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        object.__setattr__(self, "vector", array)

    @property
    def dimension(self) -> int:
        """Number of elements in the vector."""
        return int(self.vector.shape[0])

    @property
    def is_degenerate(self) -> bool:
        """True when the vector carries no usable direction.

        A near-zero network output cannot be normalised into a meaningful
        direction, so any similarity computed against it is noise.
        """
        return float(np.linalg.norm(self.vector)) <= EPSILON

    def similarity_to(self, other: FaceEmbedding) -> float:
        """Cosine similarity against another embedding.

        Raises:
            ValueError: if the two came from different model versions.
                Embeddings from different networks occupy unrelated spaces and
                their cosine similarity is meaningless - returning a number
                would be worse than refusing, because the number looks valid.
        """
        if self.model_version != other.model_version:
            raise ValueError(
                f"Refusing to compare embeddings from different model versions: "
                f"{self.model_version!r} vs {other.model_version!r}. Vectors from "
                f"different networks occupy unrelated spaces."
            )
        return cosine_similarity(self.vector, other.vector)

    def as_list(self) -> list[float]:
        """The vector as plain floats, for JSON transport and Qdrant upsert."""
        return [float(value) for value in self.vector]

    def describe(self) -> dict[str, Any]:
        """PII-free summary for logging.

        Never includes the vector itself: an embedding is biometric data and
        is reversible enough - through model-inversion attacks - to be treated
        as personal data under GDPR Article 9.
        """
        return {
            "dimension": self.dimension,
            "raw_norm": round(self.raw_norm, 3),
            "confidence": round(self.confidence, 3),
            "aligned": self.aligned,
            "residual": (
                round(self.alignment_residual, 4)
                if self.alignment_residual is not None
                else None
            ),
            "flip_averaged": self.flip_averaged,
            "cache_hit": self.cache_hit,
            "model_version": self.model_version,
            "role": str(self.role) if self.role else None,
        }


@dataclass(slots=True)
class EmbeddingRequest:
    """One face queued for embedding.

    Carries the source image plus the geometry needed to align it. Batched
    callers submit a list of these so the adapter can coalesce the aligned
    crops into a single forward pass.

    Attributes:
        image: Full source image, BGR uint8.
        box: Face bounds in source coordinates.
        landmarks: Five keypoints. Alignment falls back to the box when absent.
        role: Which image in the verification request this is.
        label: Free-form identifier echoed back on the result, so a caller can
            reassociate results with its own bookkeeping.
    """

    image: BgrImage
    box: Any = None  # BoundingBox; loose to avoid a circular import
    landmarks: Any = None  # Landmarks5
    role: ImageRole | None = None
    label: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class FaceEmbedder(abc.ABC):
    """Abstract face-embedding model.

    Args:
        name: Short adapter identifier.
        model_key: Registry key of the underlying artefact.
        model_version: Pinned version string.
        dimension: Length of the produced vector.
    """

    def __init__(
        self,
        *,
        name: str,
        model_key: str,
        model_version: str,
        dimension: int,
    ) -> None:
        self.name = name
        self.model_key = model_key
        self.model_version = model_version
        self.dimension = dimension

    # -- Subclass contract ------------------------------------------------ #

    @abc.abstractmethod
    def embed_aligned(self, crops: Sequence[BgrImage]) -> list[tuple[FloatArray, float]]:
        """Embed pre-aligned crops.

        Args:
            crops: Aligned BGR crops at the model's native input size.

        Returns:
            One ``(normalised_vector, raw_norm)`` pair per crop, in order.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources held by the adapter."""

    # -- Public surface ---------------------------------------------------- #

    async def embed_aligned_async(
        self, crops: Sequence[BgrImage]
    ) -> list[tuple[FloatArray, float]]:
        """Embed off the event loop.

        The default hands off to the default thread pool; the ONNX adapter
        overrides this to use the registry's bounded inference pool.
        """
        return await asyncio.to_thread(self.embed_aligned, crops)

    def __enter__(self) -> FaceEmbedder:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"version={self.model_version!r}, dim={self.dimension})"
        )


__all__ = [
    "BgrImage",
    "EmbeddingRequest",
    "FaceEmbedder",
    "FaceEmbedding",
    "FloatArray",
    "cosine_similarity",
    "l2_normalise",
]
