"""ArcFace ONNX embedder - the recognition backbone.

Model selection
---------------
``w600k_r50.onnx`` from the InsightFace ``buffalo_l`` pack: a ResNet-50
trained with ArcFace loss on Glint360K (360k identities, 17M images). Chosen
over the alternatives for reasons specific to this deployment:

* **Accuracy at a tractable cost.** It reaches ~99.8% on LFW and, more
  relevantly, ~97% TAR@FAR=1e-4 on IJB-C - the benchmark whose protocol
  actually resembles selfie-versus-document verification. The r100 variant is
  marginally better and roughly twice the compute, which on the CPU-only
  fallback path this service must support is not a trade worth making.
* **Demographic coverage.** Glint360K is substantially more balanced across
  ethnicities than the older MS1M-based models, which matters directly for a
  Pakistani user base.
* **Same pack as the detector.** Detector and recogniser ship in one versioned
  artefact, so they cannot drift apart across deployments.

A quirk of this artefact
------------------------
The exported graph declares its output shape as ``[1, 512]`` while the
computation is genuinely batch-correct - a batch of four returns ``(4, 512)``.
ONNX Runtime notices the mismatch and logs a warning per call. The session is
configured at ``log_severity_level=3`` so it stays quiet, and the decoder reads
the real array shape rather than trusting the declared one. Worth knowing
before someone "fixes" the batching on the strength of that warning.

Flip augmentation
-----------------
Embedding the mirrored crop as well and averaging the two vectors is standard
InsightFace evaluation practice. It costs exactly 2x inference, so it is
configurable, and the measured effect on this model is recorded in
``docs/modules/03_face_embeddings.md`` rather than asserted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import ModelSpec
from hamqadam_ai.core.exceptions import InferenceError
from hamqadam_ai.embeddings.alignment import mirror
from hamqadam_ai.embeddings.base import BgrImage, FaceEmbedder, FloatArray, l2_normalise
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.models.base import LoadedModel

log = get_logger(__name__)


class ArcFaceEmbedder(FaceEmbedder):
    """ArcFace ResNet-50 embedder backed by ONNX Runtime.

    Args:
        model: The loaded ONNX session wrapper.
        spec: The model declaration, supplying pre-processing constants.
        model_key: Registry key, recorded on every embedding.
        max_batch: Maximum crops per forward pass.
        flip_augmentation: Embed the mirrored crop too and average.
    """

    def __init__(
        self,
        model: LoadedModel,
        spec: ModelSpec,
        *,
        model_key: str = "face_embedder_arcface",
        max_batch: int = 16,
        flip_augmentation: bool = True,
    ) -> None:
        dimension = int(spec.output.get("embedding_dim", 512))
        super().__init__(
            name="arcface",
            model_key=model_key,
            model_version=spec.version,
            dimension=dimension,
        )
        self._model = model
        self._spec = spec
        self._max_batch = max(1, min(max_batch, spec.runtime.max_batch or max_batch))
        self._flip_augmentation = flip_augmentation

        self._input_size = spec.input.size
        self._mean = spec.input.mean_tuple
        self._scale = spec.input.scale
        self._swap_rb = spec.input.swap_rb
        self._input_name = model.input_names[0]

        log.info(
            "embedder.arcface.ready",
            version=spec.version,
            dimension=dimension,
            input_size=list(self._input_size),
            max_batch=self._max_batch,
            flip_augmentation=flip_augmentation,
        )

    @property
    def flip_augmentation(self) -> bool:
        """Whether the mirrored crop is embedded and averaged in."""
        return self._flip_augmentation

    @property
    def max_batch(self) -> int:
        """Maximum crops coalesced into one forward pass."""
        return self._max_batch

    # -- Inference ---------------------------------------------------------- #

    def embed_aligned(
        self, crops: Sequence[BgrImage]
    ) -> list[tuple[FloatArray, float]]:
        """Embed a sequence of aligned crops.

        Args:
            crops: Aligned BGR crops at the model's input size.

        Returns:
            One ``(unit_vector, raw_norm)`` pair per crop, in input order.

        Raises:
            InferenceError: if a forward pass fails.
        """
        if not crops:
            return []

        # With flip augmentation each crop contributes two tensors. They are
        # interleaved into one flat list so a single batching pass covers both,
        # rather than running the whole set twice.
        tensors: list[FloatArray] = []
        for crop in crops:
            tensors.append(self._preprocess(crop))
            if self._flip_augmentation:
                tensors.append(self._preprocess(mirror(crop)))

        raw = self._run_batched(tensors)
        stride = 2 if self._flip_augmentation else 1

        results: list[tuple[FloatArray, float]] = []
        for index in range(len(crops)):
            block = raw[index * stride : (index + 1) * stride]
            summed = block.sum(axis=0)
            # The reported norm is the mean of the contributing magnitudes, not
            # the norm of the sum: two near-parallel vectors would otherwise
            # report roughly double the magnitude either one actually had.
            magnitudes = [float(np.linalg.norm(vector)) for vector in block]
            results.append(
                (l2_normalise(summed), float(np.mean(magnitudes)))
            )
        return results

    def _run_batched(self, tensors: Sequence[FloatArray]) -> npt.NDArray[np.float32]:
        """Run every tensor through the network, in chunks of ``max_batch``."""
        outputs: list[npt.NDArray[np.float32]] = []

        for start in range(0, len(tensors), self._max_batch):
            chunk = tensors[start : start + self._max_batch]
            batch = np.concatenate(chunk, axis=0)
            try:
                result = self._model.run({self._input_name: batch})[0]
            except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
                raise InferenceError(
                    f"ArcFace forward pass failed on a batch of {len(chunk)}: {exc}",
                    details={
                        "model": self.model_key,
                        "version": self.model_version,
                        "batch_size": len(chunk),
                    },
                    cause=exc,
                ) from exc

            # Read the real shape rather than the declared one: this export
            # advertises [1, 512] regardless of the batch it was given.
            array = np.asarray(result, dtype=np.float32).reshape(len(chunk), -1)
            if array.shape[1] != self.dimension:
                raise InferenceError(
                    f"ArcFace returned {array.shape[1]}-dimensional vectors, "
                    f"but the model is declared as {self.dimension}-dimensional.",
                    details={
                        "model": self.model_key,
                        "declared": self.dimension,
                        "actual": int(array.shape[1]),
                    },
                )
            outputs.append(array)

        return np.concatenate(outputs, axis=0)

    def _preprocess(self, crop: BgrImage) -> FloatArray:
        """Turn an aligned crop into the network's NCHW input tensor."""
        from hamqadam_ai.utils.image_ops import build_blob

        return build_blob(
            crop,
            self._input_size,
            mean=self._mean,
            scale=self._scale,
            swap_rb=self._swap_rb,
        )

    async def embed_aligned_async(
        self, crops: Sequence[BgrImage]
    ) -> list[tuple[FloatArray, float]]:
        """Embed using the registry's bounded inference thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._model.executor, self.embed_aligned, list(crops)
        )

    def close(self) -> None:
        """No-op; the session is owned by the model registry."""
        return


__all__ = ["ArcFaceEmbedder"]
