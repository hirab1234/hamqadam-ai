"""ONNX Runtime backend.

Wraps ``onnxruntime.InferenceSession`` with the settings a production service
needs and a development notebook does not: deterministic provider selection,
bounded thread counts, graph optimisation, IO-binding-friendly contiguous
inputs and silenced verbose logging.

The one design decision worth calling out: sessions are created **eagerly at
start-up**, not lazily on first request. Lazy loading makes the first
verification of every cold pod 2-4 seconds slower and, worse, makes a missing
model file surface as a 500 on a real user's request instead of as a failed
readiness probe.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import ModelSpec, RuntimeConfig
from hamqadam_ai.core.device import DevicePlan, default_thread_counts
from hamqadam_ai.core.exceptions import DependencyUnavailableError, ModelLoadError
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.models.base import LoadedModel, ModelHandle

log = get_logger(__name__)

_GRAPH_OPTIMIZATION_LEVELS = {
    "disable_all": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


def _import_onnxruntime() -> Any:
    """Import onnxruntime, converting the ImportError into a typed error."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - the wheel is a hard dependency
        raise DependencyUnavailableError(
            "onnxruntime",
            purpose="ONNX model inference",
            extra="onnxruntime  (or onnxruntime-gpu for CUDA)",
            cause=exc,
        ) from exc
    return ort


def build_session(
    model_path: Path,
    plan: DevicePlan,
    runtime: RuntimeConfig,
    *,
    model_key: str,
) -> Any:
    """Create a configured :class:`onnxruntime.InferenceSession`.

    Args:
        model_path: Path to the ``.onnx`` file.
        plan: Resolved execution-provider plan.
        runtime: Runtime tuning configuration.
        model_key: Registry key, used in error messages.

    Returns:
        The live session.

    Raises:
        ModelLoadError: if the file is missing or ONNX Runtime rejects it.
    """
    ort = _import_onnxruntime()

    if not model_path.is_file():
        raise ModelLoadError(
            f"Model file for {model_key!r} not found at {model_path}. "
            "Run `python scripts/download_models.py` to fetch it.",
            details={"model": model_key, "path": str(model_path)},
        )

    options = ort.SessionOptions()
    intra, inter = default_thread_counts(runtime)
    if intra > 0:
        options.intra_op_num_threads = intra
    if inter > 0:
        options.inter_op_num_threads = inter

    options.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel,
        _GRAPH_OPTIMIZATION_LEVELS[runtime.graph_optimization_level],
    )
    # Sequential execution: the pipeline already runs several images
    # concurrently through the thread pool, so parallel intra-session execution
    # would oversubscribe the CPU and make tail latency worse, not better.
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = True

    # 3 = ERROR. ORT's default emits a warning per unsupported node, which on a
    # CPU-only host means dozens of lines per model load.
    options.log_severity_level = 3

    providers, provider_options = plan.as_session_args()

    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers,
            provider_options=provider_options,
        )
    except Exception as exc:
        # A provider can be *present* but fail to initialise - no CUDA driver,
        # a cuDNN version mismatch. Falling back to CPU keeps the service alive
        # and the degradation is visible in the logs and on /health.
        if providers != ["CPUExecutionProvider"]:
            log.warning(
                "model.provider_init_failed",
                model=model_key,
                providers=providers,
                reason=str(exc),
                action="falling_back_to_cpu",
            )
            try:
                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as cpu_exc:
                raise ModelLoadError(
                    f"ONNX Runtime could not load {model_key!r} on any provider: {cpu_exc}",
                    details={"model": model_key, "path": str(model_path)},
                    cause=cpu_exc,
                ) from cpu_exc
        else:
            raise ModelLoadError(
                f"ONNX Runtime could not load {model_key!r}: {exc}",
                details={"model": model_key, "path": str(model_path)},
                cause=exc,
            ) from exc

    return session


class OnnxModel(LoadedModel):
    """A loaded ONNX model backed by an ONNX Runtime session.

    Args:
        spec: The model declaration.
        handle: Resolved identity.
        session: A live ``InferenceSession``.
        executor: Thread pool for async dispatch.
    """

    def __init__(
        self,
        spec: ModelSpec,
        handle: ModelHandle,
        session: Any,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        super().__init__(spec=spec, handle=handle, executor=executor)
        self._session = session
        self._input_names: tuple[str, ...] = tuple(
            item.name for item in session.get_inputs()
        )
        self._output_names: tuple[str, ...] = tuple(
            item.name for item in session.get_outputs()
        )
        self._input_shapes: dict[str, tuple[int | str | None, ...]] = {
            item.name: tuple(item.shape) for item in session.get_inputs()
        }
        self._input_dtypes: dict[str, str] = {
            item.name: item.type for item in session.get_inputs()
        }

    @property
    def input_names(self) -> tuple[str, ...]:
        """Input tensor names in declaration order."""
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        """Output tensor names in declaration order."""
        return self._output_names

    @property
    def input_shapes(self) -> dict[str, tuple[int | str | None, ...]]:
        """Declared input shapes; string or ``None`` entries are dynamic axes."""
        return dict(self._input_shapes)

    @property
    def session(self) -> Any:
        """The underlying ONNX Runtime session, for advanced use."""
        return self._session

    def _run_sync(self, inputs: dict[str, npt.NDArray[Any]]) -> list[npt.NDArray[Any]]:
        """Execute the session, coercing inputs to the expected dtype and layout."""
        prepared = {
            name: self._prepare(name, array) for name, array in inputs.items()
        }
        return list(self._session.run(list(self._output_names), prepared))

    def _prepare(self, name: str, array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Coerce one input tensor to a contiguous array of the declared dtype.

        ONNX Runtime raises an opaque error on a dtype mismatch and silently
        copies non-contiguous buffers, so both are handled here where the fix
        is cheap and the diagnostics are good.
        """
        declared = self._input_dtypes.get(name, "tensor(float)")
        target = _ORT_DTYPE_MAP.get(declared, np.float32)
        result = np.asarray(array)
        if result.dtype != target:
            result = result.astype(target, copy=False)
        return np.ascontiguousarray(result)

    def _close_backend(self) -> None:
        """Drop the session reference so ORT can free its arenas."""
        self._session = None


_ORT_DTYPE_MAP: dict[str, Any] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
    "tensor(bool)": np.bool_,
}


def load_onnx_model(
    spec: ModelSpec,
    model_path: Path,
    plan: DevicePlan,
    runtime: RuntimeConfig,
    *,
    model_key: str,
    sha256: str | None,
    executor: ThreadPoolExecutor | None = None,
    warmup_iterations: int = 0,
) -> OnnxModel:
    """Load, wrap and warm up an ONNX model.

    Args:
        spec: The model declaration.
        model_path: Verified path to the ``.onnx`` file.
        plan: Resolved execution-provider plan.
        runtime: Runtime tuning configuration.
        model_key: Registry key.
        sha256: Verified digest, recorded in the handle for audit.
        executor: Thread pool for async dispatch.
        warmup_iterations: Synthetic passes to run before returning.

    Returns:
        A ready-to-use :class:`OnnxModel`.
    """
    started = time.perf_counter()
    session = build_session(model_path, plan, runtime, model_key=model_key)

    actual_providers = tuple(session.get_providers())
    handle = ModelHandle(
        key=model_key,
        version=spec.version,
        kind="onnx",
        device=plan.family,
        providers=actual_providers,
        sha256=sha256,
    )

    model = OnnxModel(spec=spec, handle=handle, session=session, executor=executor)
    warmup_seconds = model.warmup(warmup_iterations) if warmup_iterations else 0.0

    load_ms = (time.perf_counter() - started) * 1000.0
    model.handle = ModelHandle(
        key=handle.key,
        version=handle.version,
        kind=handle.kind,
        device=handle.device,
        providers=handle.providers,
        sha256=handle.sha256,
        load_time_ms=load_ms,
    )

    log.info(
        "model.loaded",
        model=model_key,
        version=spec.version,
        providers=list(actual_providers),
        inputs={name: list(shape) for name, shape in model.input_shapes.items()},
        outputs=list(model.output_names),
        load_ms=round(load_ms, 1),
        warmup_ms=round(warmup_seconds * 1000.0, 1),
    )
    return model


__all__ = ["OnnxModel", "build_session", "load_onnx_model"]
