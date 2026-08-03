"""MODULE 3 service - orchestrates alignment, caching and batched inference.

Pipeline for a set of faces::

    align each face  ->  probe cache  ->  batch the misses through ArcFace
                                      ->  score confidence from the residual
                                      ->  populate cache
                                      ->  EmbeddingResult per face

Design notes
------------
**Batching helps, but less than one might assume.** A single 112x112 forward
pass costs ~148 ms on this CPU and a batch of eight costs ~121 ms per face -
a 1.22x speed-up, not an order of magnitude. ONNX Runtime already parallelises
one ResNet-50 across all cores, so batching mostly saves per-call overhead
rather than unlocking idle compute. It is still worth doing on a seven-image
request, which is why the service always takes a *list* and the single-face
entry point is a thin wrapper over the batch path.

**Confidence comes from the alignment residual, not the embedding magnitude.**
The magnitude was the first design, on the MagFace premise that it tracks face
quality. Measurement refuted that for this artefact - see
:meth:`EmbeddingService._confidence_for`.

**Cache probing happens after alignment, not before.** The key is the hash of
the aligned crop, so alignment must run first. Alignment is a warp and costs
well under a millisecond, so paying it on a cache hit is a good trade for a key
that is exactly right.

**Failures are per-face.** One image whose landmarks are unusable produces a
failed :class:`~hamqadam_ai.schemas.embedding.EmbeddingResult` in its slot; the
other six still embed. The pipeline needs a complete picture, not the first
error.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import Settings, get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import (
    HamqadamError,
    ModelNotLoadedError,
)
from hamqadam_ai.embeddings.alignment import AlignmentResult, align_face_for_recognition
from hamqadam_ai.embeddings.arcface import ArcFaceEmbedder
from hamqadam_ai.embeddings.base import (
    BgrImage,
    EmbeddingRequest,
    FaceEmbedder,
    FaceEmbedding,
)
from hamqadam_ai.embeddings.cache import EmbeddingCache, build_cache, crop_cache_key
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.models.registry import ModelRegistry, get_registry
from hamqadam_ai.quality.scoring import ramp_score
from hamqadam_ai.schemas.common import AnalysisWarning
from hamqadam_ai.schemas.detection import FaceDetectionResult
from hamqadam_ai.schemas.embedding import EmbeddingBatchResult, EmbeddingResult
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

log = get_logger(__name__)

_LANDMARK_ORDER = ("left_eye", "right_eye", "nose_tip", "mouth_left", "mouth_right")


class EmbeddingService:
    """Produces face embeddings for one or more detected faces.

    Args:
        embedder: The recognition adapter.
        cache: Embedding cache.
        settings: Service configuration.
    """

    __slots__ = ("_cache", "_config", "_embedder", "_settings")

    def __init__(
        self,
        *,
        embedder: FaceEmbedder,
        cache: EmbeddingCache,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._cache = cache
        self._settings = settings
        self._config = settings.embedding

    # -- Public surface ----------------------------------------------------- #

    def embed(
        self,
        image: BgrImage,
        *,
        role: ImageRole = ImageRole.PROFILE_IMAGE,
        detection: FaceDetectionResult | None = None,
        box: BoundingBox | None = None,
        landmarks: Landmarks5 | None = None,
        include_vector: bool = False,
    ) -> EmbeddingResult:
        """Embed the primary face in one image.

        Args:
            image: Full source image, BGR uint8.
            role: Which image in the verification request this is.
            detection: The Module 1 result. Supplies box and landmarks;
                strongly preferred over passing them individually.
            box: Face bounds, when no detection result is available.
            landmarks: Five keypoints, when no detection result is available.
            include_vector: Return the vector in the response. Off by default
                because an embedding is biometric data, not a diagnostic.

        Returns:
            The result. A failure is reported through ``success`` and
            ``error_code`` rather than raised.
        """
        resolved_box, resolved_landmarks = self._resolve_face(detection, box, landmarks)
        request = EmbeddingRequest(
            image=image, box=resolved_box, landmarks=resolved_landmarks, role=role
        )
        batch = self.embed_many([request], include_vectors=include_vector)
        return batch.results[0]

    def embed_many(
        self,
        requests: Sequence[EmbeddingRequest],
        *,
        include_vectors: bool = False,
    ) -> EmbeddingBatchResult:
        """Embed several faces in one batched pass.

        Args:
            requests: The faces to embed.
            include_vectors: Return vectors in the responses.

        Returns:
            One result per request, in submission order.
        """
        started = time.perf_counter()
        if not requests:
            return EmbeddingBatchResult(results=[], total_duration_ms=0.0)

        # Stage 1: align. Cheap, and produces the cache key.
        alignments: list[AlignmentResult | HamqadamError] = [
            self._safe_align(request) for request in requests
        ]

        # Stage 2: probe the cache for every successful alignment.
        keys: list[str | None] = []
        cached: list[tuple[npt.NDArray[np.float32], float] | None] = []
        for alignment in alignments:
            if isinstance(alignment, HamqadamError):
                keys.append(None)
                cached.append(None)
                continue
            key = crop_cache_key(
                alignment.crop,
                model_version=self._embedder.model_version,
                flip_augmented=self._config.flip_augmentation,
            )
            keys.append(key)
            cached.append(self._cache.get(key))

        # Stage 3: one batched forward pass over the misses.
        pending_indices = [
            index
            for index, alignment in enumerate(alignments)
            if not isinstance(alignment, HamqadamError) and cached[index] is None
        ]
        computed: dict[int, tuple[npt.NDArray[np.float32], float]] = {}
        inference_error: HamqadamError | None = None

        if pending_indices:
            crops = [
                alignments[index].crop  # type: ignore[union-attr]
                for index in pending_indices
            ]
            try:
                produced = self._embedder.embed_aligned(crops)
            except HamqadamError as exc:
                inference_error = exc
                log.error(
                    "embedding.batch_failed",
                    count=len(crops),
                    reason=exc.message,
                    error_code=str(exc.code),
                )
            else:
                for index, value in zip(pending_indices, produced, strict=True):
                    computed[index] = value
                    cache_key = keys[index]
                    if cache_key is not None:
                        self._cache.put(cache_key, value)

        # Stage 4: assemble.
        results: list[EmbeddingResult] = []
        for index, request in enumerate(requests):
            results.append(
                self._build_result(
                    request=request,
                    alignment=alignments[index],
                    cached=cached[index],
                    computed=computed.get(index),
                    inference_error=inference_error,
                    include_vector=include_vectors,
                )
            )

        total_ms = (time.perf_counter() - started) * 1000.0
        batch = EmbeddingBatchResult(
            results=results,
            total_duration_ms=total_ms,
            cache_hits=sum(1 for entry in cached if entry is not None),
            forward_passes=len(pending_indices),
        )

        log.info(
            "embedding.batch_completed",
            requested=len(requests),
            succeeded=batch.succeeded,
            cache_hits=batch.cache_hits,
            forward_passes=batch.forward_passes,
            duration_ms=round(total_ms, 1),
        )
        return batch

    def embed_to_vector(
        self,
        image: BgrImage,
        *,
        role: ImageRole = ImageRole.PROFILE_IMAGE,
        detection: FaceDetectionResult | None = None,
        box: BoundingBox | None = None,
        landmarks: Landmarks5 | None = None,
    ) -> FaceEmbedding:
        """Embed one face and return the internal value object.

        The path the rest of the pipeline uses: Modules 4, 6 and 8 need the
        :class:`~hamqadam_ai.embeddings.base.FaceEmbedding` itself, not the
        API-facing schema.

        Raises:
            HamqadamError: if no embedding could be produced. This entry point
                is fail-fast by design - a caller asking for a vector has
                nothing useful to do with a failure result.
        """
        resolved_box, resolved_landmarks = self._resolve_face(detection, box, landmarks)
        alignment = align_face_for_recognition(
            image,
            box=resolved_box,
            landmarks=resolved_landmarks,
            output_size=self._config.alignment.output_size,
            max_residual=self._config.alignment.max_alignment_residual,
            allow_box_fallback=self._config.alignment.allow_box_fallback,
            box_margin=self._config.alignment.box_fallback_margin,
        )

        key = crop_cache_key(
            alignment.crop,
            model_version=self._embedder.model_version,
            flip_augmented=self._config.flip_augmentation,
        )
        entry = self._cache.get(key)
        cache_hit = entry is not None
        if entry is None:
            entry = self._embedder.embed_aligned([alignment.crop])[0]
            self._cache.put(key, entry)

        vector, raw_norm = entry
        return FaceEmbedding(
            vector=vector,
            raw_norm=raw_norm,
            confidence=self._confidence_for(alignment),
            model_key=self._embedder.model_key,
            model_version=self._embedder.model_version,
            aligned=alignment.aligned,
            alignment_residual=alignment.residual,
            flip_averaged=self._config.flip_augmentation,
            cache_hit=cache_hit,
            role=role,
        )

    async def embed_async(
        self,
        image: BgrImage,
        *,
        role: ImageRole = ImageRole.PROFILE_IMAGE,
        detection: FaceDetectionResult | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """Embed one face without blocking the event loop."""
        return await asyncio.to_thread(
            self.embed, image, role=role, detection=detection, **kwargs
        )

    async def embed_many_async(
        self,
        requests: Sequence[EmbeddingRequest],
        *,
        include_vectors: bool = False,
    ) -> EmbeddingBatchResult:
        """Batched embedding without blocking the event loop.

        The whole batch runs on one worker thread rather than being fanned out:
        the forward pass is already internally parallel and ONNX Runtime's own
        thread pool would contend with a fan-out here, making it slower.
        """
        return await asyncio.to_thread(
            self.embed_many, list(requests), include_vectors=include_vectors
        )

    def describe(self) -> dict[str, Any]:
        """Serialisable summary of the active configuration, for ``/health``."""
        return {
            "model_key": self._embedder.model_key,
            "model_version": self._embedder.model_version,
            "dimension": self._embedder.dimension,
            "flip_augmentation": self._config.flip_augmentation,
            "max_batch": self._config.batch.max_size,
            "alignment": {
                "output_size": list(self._config.alignment.output_size),
                "max_residual": self._config.alignment.max_alignment_residual,
                "box_fallback": self._config.alignment.allow_box_fallback,
            },
            "cache": self._cache.stats,
        }

    def close(self) -> None:
        """Release the embedder and drop cached vectors.

        Clearing the cache matters: it holds biometric templates, and they
        should not outlive the process that needed them.
        """
        self._cache.clear()
        self._embedder.close()

    # -- Internals ----------------------------------------------------------- #

    @staticmethod
    def _resolve_face(
        detection: FaceDetectionResult | None,
        box: BoundingBox | None,
        landmarks: Landmarks5 | None,
    ) -> tuple[BoundingBox | None, Landmarks5 | None]:
        """Extract the primary face's geometry, preferring explicit overrides."""
        if box is not None or landmarks is not None:
            return box, landmarks
        if detection is None or detection.primary_face is None:
            return None, None

        primary = detection.primary_face
        resolved_box = BoundingBox(
            primary.bounding_box.x1,
            primary.bounding_box.y1,
            primary.bounding_box.x2,
            primary.bounding_box.y2,
        )

        resolved_landmarks: Landmarks5 | None = None
        named = {landmark.name: landmark for landmark in primary.landmarks}
        if all(name in named for name in _LANDMARK_ORDER):
            resolved_landmarks = Landmarks5(
                np.array(
                    [[named[name].x, named[name].y] for name in _LANDMARK_ORDER],
                    dtype=np.float32,
                )
            )
        return resolved_box, resolved_landmarks

    def _safe_align(self, request: EmbeddingRequest) -> AlignmentResult | HamqadamError:
        """Align one face, returning the error inline rather than raising."""
        try:
            return align_face_for_recognition(
                request.image,
                box=request.box,
                landmarks=request.landmarks,
                output_size=self._config.alignment.output_size,
                max_residual=self._config.alignment.max_alignment_residual,
                allow_box_fallback=self._config.alignment.allow_box_fallback,
                box_margin=self._config.alignment.box_fallback_margin,
            )
        except ValueError as exc:
            return HamqadamError(
                str(exc),
                code=ErrorCode.FACE_NOT_DETECTED,
                details={"role": str(request.role) if request.role else None},
                cause=exc,
            )
        except Exception as exc:  # noqa: BLE001 - one face must not fail the batch
            log.warning("embedding.alignment_failed", reason=str(exc))
            return HamqadamError(
                f"Face alignment failed: {exc}",
                code=ErrorCode.AI_SERVICE_ERROR,
                cause=exc,
            )

    def _confidence_for(self, alignment: AlignmentResult) -> float:
        """Derive a [0, 1] confidence from how well the face aligned.

        Not from the embedding magnitude. That was the first design, on the
        MagFace premise that ArcFace's pre-normalisation norm tracks face
        quality, and measurement refuted it for this artefact: across a
        twelve-step degradation ladder the norm moved only between 22.8 and
        25.5, and a 21-pixel blur measured *higher* than the pristine face.
        Those papers train for the property with a magnitude-aware loss;
        vanilla ArcFace leaves the magnitude unconstrained.

        The alignment residual does carry the signal - it sits in a tight
        0.034-0.061 band for every correctly-detected face and climbs as the
        landmarks stop describing a plausible one.
        """
        if not alignment.aligned or alignment.residual is None:
            return self._config.quality.box_fallback_confidence
        return ramp_score(alignment.residual, self._config.quality.alignment_residual)

    def _build_result(
        self,
        *,
        request: EmbeddingRequest,
        alignment: AlignmentResult | HamqadamError,
        cached: tuple[npt.NDArray[np.float32], float] | None,
        computed: tuple[npt.NDArray[np.float32], float] | None,
        inference_error: HamqadamError | None,
        include_vector: bool,
    ) -> EmbeddingResult:
        """Assemble one API result from the staged outcomes."""
        role = request.role or ImageRole.PROFILE_IMAGE

        if isinstance(alignment, HamqadamError):
            return EmbeddingResult(
                success=False,
                role=role,
                error_code=alignment.code,
                error_message=alignment.message,
                model_key=self._embedder.model_key,
                model_version=self._embedder.model_version,
            )

        entry = cached if cached is not None else computed
        if entry is None:
            error = inference_error or HamqadamError(
                "The embedding was neither cached nor computed.",
                code=ErrorCode.INFERENCE_FAILED,
            )
            return EmbeddingResult(
                success=False,
                role=role,
                aligned=alignment.aligned,
                alignment_residual=alignment.residual,
                error_code=error.code,
                error_message=error.message,
                model_key=self._embedder.model_key,
                model_version=self._embedder.model_version,
            )

        vector, raw_norm = entry
        confidence = self._confidence_for(alignment)
        low_confidence = confidence < self._config.quality.min_confidence

        warnings: list[AnalysisWarning] = []
        if not alignment.aligned:
            warnings.append(
                AnalysisWarning(
                    code="EMBEDDING_BOX_ALIGNED",
                    message=(
                        "No usable landmarks, so the face was centred from its "
                        "bounding box instead of warped onto the recognition "
                        "template. Scale is normalised but in-plane rotation is "
                        "not, and match scores from this embedding are less "
                        "reliable."
                    ),
                    stage="embedding",
                    detail={"reason": alignment.reason},
                )
            )
        if low_confidence and alignment.aligned:
            warnings.append(
                AnalysisWarning(
                    code="EMBEDDING_LOW_CONFIDENCE",
                    message=(
                        f"The face aligned poorly onto the recognition template "
                        f"(residual {alignment.residual:.3f}), so the landmarks may "
                        f"not describe the face accurately. The embedding is still "
                        f"usable but downstream match scores carry less weight."
                    ),
                    stage="embedding",
                    detail={
                        "alignment_residual": round(alignment.residual or 0.0, 4),
                        "confidence": round(confidence, 3),
                        "minimum": self._config.quality.min_confidence,
                    },
                )
            )

        return EmbeddingResult(
            success=True,
            role=role,
            dimension=int(vector.shape[0]),
            vector=[float(value) for value in vector] if include_vector else None,
            raw_norm=round(float(raw_norm), 4),
            confidence=round(confidence, 4),
            confidence_score=round(confidence * 100.0, 2),
            low_confidence=low_confidence,
            aligned=alignment.aligned,
            alignment_residual=(
                round(alignment.residual, 5) if alignment.residual is not None else None
            ),
            flip_averaged=self._config.flip_augmentation,
            cache_hit=cached is not None,
            model_key=self._embedder.model_key,
            model_version=self._embedder.model_version,
            warnings=warnings,
        )


def build_embedding_service(
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
) -> EmbeddingService:
    """Wire up an :class:`EmbeddingService` from configuration.

    Args:
        settings: Service configuration. Loaded from the environment when
            omitted.
        registry: The model registry. The process-wide one is used when
            omitted.

    Returns:
        A ready service.

    Raises:
        ModelNotLoadedError: if the recogniser could not be loaded. Unlike
            detection there is no fallback chain here: a face embedding from a
            different model is not a degraded answer, it is an incomparable
            one, so failing loudly is the only correct behaviour.
    """
    settings = settings or get_settings()
    registry = registry or get_registry(settings)
    config = settings.embedding

    spec = settings.model_spec(config.model)
    model = registry.try_get(config.model)
    if model is None:
        raise ModelNotLoadedError(
            f"The face recogniser {config.model!r} could not be loaded, so no "
            f"embeddings can be produced. Run `python scripts/download_models.py`. "
            f"There is deliberately no fallback: an embedding from a different "
            f"model occupies a different space and cannot be compared with the "
            f"vectors already stored.",
            details={"model": config.model, "path": spec.path},
        )

    embedder = ArcFaceEmbedder(
        model,
        spec,
        model_key=config.model,
        max_batch=config.batch.max_size,
        flip_augmentation=config.flip_augmentation,
    )
    cache = build_cache(config.cache, dimension=embedder.dimension)

    log.info(
        "embedding.service_ready",
        model=config.model,
        version=spec.version,
        dimension=embedder.dimension,
        cache_backend=cache.stats.get("backend"),
    )

    return EmbeddingService(embedder=embedder, cache=cache, settings=settings)


__all__ = ["EmbeddingService", "build_embedding_service"]
