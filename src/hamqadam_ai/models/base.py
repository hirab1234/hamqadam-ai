"""Abstract inference contracts.

Defines what "a loaded model" means independently of the runtime that backs it.
ONNX Runtime is the production backend, but the OpenCV DNN and Haar cascade
fallbacks are not ONNX at all, and a future Triton or TensorRT-native backend
should slot in without touching a single detector.

The contract is deliberately narrow - ``run`` plus metadata. Pre- and
post-processing belong to the capability module that understands the tensors,
not to the runtime wrapper.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import ModelSpec

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ModelHandle:
    """Immutable identity of a loaded artefact, reported in the API response.

    Attributes:
        key: Registry key, e.g. ``face_detector_scrfd``.
        version: Pinned version string from ``models.yaml``.
        kind: Backend family that loaded it.
        device: Resolved device family the session runs on.
        providers: Execution providers actually in use, in preference order.
        sha256: Verified digest, or ``None`` when verification was disabled.
        load_time_ms: How long loading and warm-up took.
    """

    key: str
    version: str
    kind: str
    device: str
    providers: tuple[str, ...]
    sha256: str | None = None
    load_time_ms: float = 0.0

    def describe(self) -> dict[str, Any]:
        """Serialisable summary for ``/health`` and ``model_versions``."""
        return {
            "key": self.key,
            "version": self.version,
            "kind": self.kind,
            "device": self.device,
            "providers": list(self.providers),
            "sha256": self.sha256[:16] if self.sha256 else None,
            "load_time_ms": round(self.load_time_ms, 1),
        }


@runtime_checkable
class InferenceBackend(Protocol):
    """Minimal structural contract every model backend satisfies."""

    @property
    def handle(self) -> ModelHandle:
        """Identity and provenance of this loaded model."""
        ...

    def run(self, inputs: dict[str, npt.NDArray[Any]]) -> list[npt.NDArray[Any]]:
        """Execute a forward pass synchronously."""
        ...

    def close(self) -> None:
        """Release the underlying session and its device memory."""
        ...


@dataclass
class LoadedModel(abc.ABC):
    """Base class for a loaded model artefact.

    Subclasses implement :meth:`_run_sync`; this class supplies the async
    bridge, input validation, warm-up and inference statistics that every
    backend needs identically.

    Args:
        spec: The declaration this model was loaded from.
        handle: Resolved identity, filled in by the loader.
        executor: Thread pool that blocking forward passes are dispatched to.
    """

    spec: ModelSpec
    handle: ModelHandle
    executor: ThreadPoolExecutor | None = None

    _calls: int = field(default=0, init=False, repr=False)
    _total_seconds: float = field(default=0.0, init=False, repr=False)
    _failures: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    # -- Subclass contract ------------------------------------------------ #

    @abc.abstractmethod
    def _run_sync(self, inputs: dict[str, npt.NDArray[Any]]) -> list[npt.NDArray[Any]]:
        """Execute one forward pass. Called on a worker thread."""

    @abc.abstractmethod
    def _close_backend(self) -> None:
        """Release backend-specific resources."""

    @property
    @abc.abstractmethod
    def input_names(self) -> tuple[str, ...]:
        """Names of the model's input tensors, in declaration order."""

    @property
    @abc.abstractmethod
    def output_names(self) -> tuple[str, ...]:
        """Names of the model's output tensors, in declaration order."""

    @property
    @abc.abstractmethod
    def input_shapes(self) -> dict[str, tuple[int | str | None, ...]]:
        """Declared shape of each input; ``None``/``str`` entries are dynamic."""

    # -- Public surface ---------------------------------------------------- #

    def run(self, inputs: dict[str, npt.NDArray[Any]]) -> list[npt.NDArray[Any]]:
        """Execute a forward pass, recording timing and failure statistics.

        Args:
            inputs: Mapping of input tensor name to array.

        Returns:
            Output tensors in the model's declared output order.

        Raises:
            ModelNotLoadedError: if the model has been closed.
            InferenceError: if the backend raises.
        """
        import time

        from hamqadam_ai.core.exceptions import InferenceError, ModelNotLoadedError

        if self._closed:
            raise ModelNotLoadedError(
                f"Model {self.handle.key!r} has been closed.",
                details={"model": self.handle.key},
            )

        started = time.perf_counter()
        try:
            outputs = self._run_sync(inputs)
        except Exception as exc:
            self._failures += 1
            raise InferenceError(
                f"Forward pass failed for model {self.handle.key!r}: {exc}",
                details={
                    "model": self.handle.key,
                    "version": self.handle.version,
                    "input_shapes": {
                        name: list(np.shape(array)) for name, array in inputs.items()
                    },
                },
                cause=exc,
            ) from exc
        finally:
            self._calls += 1
            self._total_seconds += time.perf_counter() - started

        return outputs

    async def run_async(
        self, inputs: dict[str, npt.NDArray[Any]]
    ) -> list[npt.NDArray[Any]]:
        """Execute a forward pass off the event loop.

        ONNX Runtime releases the GIL inside its kernels, so dispatching to a
        bounded thread pool gives genuine parallelism across concurrent
        requests while keeping the event loop responsive. The pool is bounded
        rather than unbounded because each in-flight session holds an arena
        allocation, and an unbounded pool will OOM a container under load long
        before it saturates the CPU.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.run, inputs)

    def run_single(self, tensor: npt.NDArray[Any]) -> list[npt.NDArray[Any]]:
        """Convenience wrapper for the common single-input case."""
        return self.run({self.input_names[0]: tensor})

    async def run_single_async(self, tensor: npt.NDArray[Any]) -> list[npt.NDArray[Any]]:
        """Async counterpart of :meth:`run_single`."""
        return await self.run_async({self.input_names[0]: tensor})

    def warmup(self, iterations: int = 2) -> float:
        """Run synthetic forward passes to force lazy initialisation.

        The first inference on a fresh session pays for arena allocation,
        kernel selection and, on CUDA, cuDNN algorithm autotuning. On a CPU
        SCRFD session that is 300-800 ms - unacceptable as a tail latency on a
        real user's first verification. Warming up at start-up moves that cost
        into the readiness window instead.

        Args:
            iterations: Number of synthetic passes.

        Returns:
            Total warm-up time in seconds. Zero when no synthetic input could
            be constructed because every dimension is dynamic.
        """
        import time

        if iterations <= 0:
            return 0.0

        synthetic = self._synthetic_inputs()
        if synthetic is None:
            return 0.0

        started = time.perf_counter()
        for _ in range(iterations):
            try:
                self._run_sync(synthetic)
            except Exception:  # noqa: BLE001 - warm-up must never block start-up
                return time.perf_counter() - started
        elapsed = time.perf_counter() - started

        # Warm-up passes are infrastructure, not traffic; keep them out of the
        # statistics so p50 latency is not skewed by synthetic input.
        self._calls = 0
        self._total_seconds = 0.0
        return elapsed

    def _synthetic_inputs(self) -> dict[str, npt.NDArray[Any]] | None:
        """Build a zero-filled input set matching the declared shapes.

        Dynamic dimensions are resolved from the model spec where possible
        (batch 1, the configured input size), which covers every model this
        service loads.
        """
        configured_width, configured_height = self.spec.input.size
        tensors: dict[str, npt.NDArray[Any]] = {}

        for name in self.input_names:
            shape = self.input_shapes.get(name)
            if shape is None:
                return None
            resolved: list[int] = []
            for axis, dimension in enumerate(shape):
                if isinstance(dimension, int) and dimension > 0:
                    resolved.append(dimension)
                elif axis == 0:
                    resolved.append(1)
                elif len(shape) == 4 and axis == 1:
                    resolved.append(3)
                elif len(shape) == 4 and axis == 2:
                    resolved.append(configured_height)
                elif len(shape) == 4 and axis == 3:
                    resolved.append(configured_width)
                else:
                    return None
            tensors[name] = np.zeros(tuple(resolved), dtype=np.float32)

        return tensors or None

    def close(self) -> None:
        """Release the backend. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._close_backend()

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._closed

    @property
    def stats(self) -> dict[str, Any]:
        """Inference counters exposed on the health endpoint."""
        average_ms = (self._total_seconds / self._calls * 1000.0) if self._calls else 0.0
        return {
            "calls": self._calls,
            "failures": self._failures,
            "average_latency_ms": round(average_ms, 2),
            "total_seconds": round(self._total_seconds, 3),
        }

    def __enter__(self) -> LoadedModel:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def validate_batch(
    tensors: Sequence[npt.NDArray[Any]], max_batch: int
) -> list[Sequence[npt.NDArray[Any]]]:
    """Split a sequence of tensors into chunks no larger than ``max_batch``.

    Batching amortises the per-call ONNX Runtime overhead, which dominates for
    small models such as the 112x112 ArcFace network: seven separate calls cost
    roughly 40% more than one batch of seven. The chunk size is bounded because
    a large batch spikes peak memory and delays the first result.

    Args:
        tensors: Individual input tensors, each without a batch dimension.
        max_batch: Maximum tensors per chunk.

    Returns:
        A list of chunks, preserving input order.
    """
    if max_batch < 1:
        raise ValueError("max_batch must be at least 1")
    return [list(tensors[i : i + max_batch]) for i in range(0, len(tensors), max_batch)]


__all__ = [
    "FloatArray",
    "InferenceBackend",
    "LoadedModel",
    "ModelHandle",
    "validate_batch",
]
