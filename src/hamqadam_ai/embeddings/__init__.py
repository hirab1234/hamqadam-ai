"""MODULE 3 - face embedding generation.

Turns a detected face into the 512-dimensional vector everything downstream
depends on. Matching, CNIC comparison, duplicate search and therefore the final
recommendation are all functions of these numbers, so this module has more
leverage over the verification outcome than any other.

Structure
---------
``base``
    The :class:`~hamqadam_ai.embeddings.base.FaceEmbedder` port and the
    :class:`~hamqadam_ai.embeddings.base.FaceEmbedding` value object.

``alignment``
    Warps a detected face onto ArcFace's canonical five-point template. This
    is not pre-processing housekeeping - it is the single largest controllable
    factor in recognition accuracy after the model itself.

``arcface``
    The ONNX adapter. Handles batching, flip augmentation and the raw-norm
    quality signal.

``cache``
    Content-addressed embedding cache, keyed on the aligned crop.

Why the vectors are L2-normalised
---------------------------------
ArcFace is trained with an angular margin on the hypersphere: the loss only
ever sees the *direction* of the embedding, never its magnitude. Comparing
un-normalised vectors therefore mixes a quantity the model optimised (angle)
with one it did not (length), and cosine similarity on normalised vectors
reduces to a dot product, which the vector database can index directly.

The discarded magnitude is not thrown away - it is reported as ``raw_norm``,
because it turns out to correlate with face quality.
"""

from __future__ import annotations

from hamqadam_ai.embeddings.alignment import (
    AlignmentResult,
    align_face_for_recognition,
    alignment_residual,
)
from hamqadam_ai.embeddings.arcface import ArcFaceEmbedder
from hamqadam_ai.embeddings.base import (
    EmbeddingRequest,
    FaceEmbedder,
    FaceEmbedding,
    cosine_similarity,
    l2_normalise,
)
from hamqadam_ai.embeddings.cache import (
    EmbeddingCache,
    InMemoryEmbeddingCache,
    NullEmbeddingCache,
    build_cache,
)

__all__ = [
    "AlignmentResult",
    "ArcFaceEmbedder",
    "EmbeddingCache",
    "EmbeddingRequest",
    "FaceEmbedder",
    "FaceEmbedding",
    "InMemoryEmbeddingCache",
    "NullEmbeddingCache",
    "align_face_for_recognition",
    "alignment_residual",
    "build_cache",
    "cosine_similarity",
    "l2_normalise",
]
