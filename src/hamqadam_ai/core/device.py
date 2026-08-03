"""Hardware and execution-provider resolution for ONNX Runtime.

The service must run unchanged on a CUDA server, a DirectML Windows box, an
Apple-silicon laptop and a plain CPU container. This module resolves the
configured ``runtime.device`` against what ONNX Runtime actually reports as
available and produces a concrete, ordered provider list.

The resolution is deliberately *silent-degrading*: a provider that is
configured but unavailable is dropped with a warning, never an exception, and
``CPUExecutionProvider`` is always appended as the terminal fallback. A
verification service that refuses to boot because a GPU is missing is worse
than one that boots slower.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from hamqadam_ai.core.config import ProviderSpec, RuntimeConfig, Settings

#: ONNX Runtime provider name -> the device family it belongs to.
_PROVIDER_FAMILY: dict[str, str] = {
    "TensorrtExecutionProvider": "cuda",
    "CUDAExecutionProvider": "cuda",
    "DmlExecutionProvider": "directml",
    "CoreMLExecutionProvider": "coreml",
    "ROCMExecutionProvider": "cuda",
    "CPUExecutionProvider": "cpu",
}

#: Order in which device families are probed when ``runtime.device`` is ``auto``.
_AUTO_PROBE_ORDER: tuple[str, ...] = ("cuda", "directml", "coreml", "cpu")


@dataclass(frozen=True, slots=True)
class DevicePlan:
    """A fully resolved execution plan for one ONNX Runtime session.

    Attributes:
        family: Resolved device family (``cuda``, ``directml``, ``coreml``,
            ``cpu``).
        providers: Provider names in preference order, always ending in
            ``CPUExecutionProvider``.
        provider_options: Per-provider options, positionally aligned with
            :attr:`providers`, as ONNX Runtime's ``InferenceSession`` expects.
        requested: The device the operator asked for, before resolution.
        degraded: True when the requested family was unavailable and the plan
            fell back to a weaker one. Surfaced in ``/health`` so a silently
            CPU-bound production node is visible rather than merely slow.
        available: Every provider ONNX Runtime reported in this process.
    """

    family: str
    providers: tuple[str, ...]
    provider_options: tuple[dict[str, Any], ...]
    requested: str
    degraded: bool
    available: tuple[str, ...]

    @property
    def is_gpu(self) -> bool:
        """True when the plan will execute on an accelerator."""
        return self.family != "cpu"

    @property
    def primary_provider(self) -> str:
        """The provider ONNX Runtime will try first."""
        return self.providers[0]

    def as_session_args(self) -> tuple[list[str], list[dict[str, Any]]]:
        """Return ``(providers, provider_options)`` for ``InferenceSession``."""
        return list(self.providers), [dict(opts) for opts in self.provider_options]

    def describe(self) -> dict[str, Any]:
        """Serialisable summary for health checks and start-up logging."""
        return {
            "requested_device": self.requested,
            "resolved_family": self.family,
            "providers": list(self.providers),
            "degraded": self.degraded,
            "available_providers": list(self.available),
        }


@lru_cache(maxsize=1)
def available_providers() -> tuple[str, ...]:
    """Return the execution providers ONNX Runtime reports in this process.

    Cached: the answer cannot change without reloading the shared library, and
    the underlying call is not free.

    Returns:
        Provider names, or ``("CPUExecutionProvider",)`` when ONNX Runtime is
        not importable at all (unit-test environments without the wheel).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return ("CPUExecutionProvider",)
    try:
        return tuple(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - defensive: a broken ORT build must not crash boot
        return ("CPUExecutionProvider",)


def available_families() -> tuple[str, ...]:
    """Return the device families backed by at least one available provider."""
    families = {
        _PROVIDER_FAMILY[name]
        for name in available_providers()
        if name in _PROVIDER_FAMILY
    }
    families.add("cpu")
    return tuple(family for family in _AUTO_PROBE_ORDER if family in families)


def _cuda_visible() -> bool:
    """Return False when CUDA is explicitly masked off by the environment.

    ONNX Runtime happily reports ``CUDAExecutionProvider`` as available even
    when ``CUDA_VISIBLE_DEVICES=""`` makes every device invisible; session
    creation then fails at request time instead of at boot. Checking the
    variable up front turns a runtime error into a clean start-up degradation.
    """
    masked = os.environ.get("CUDA_VISIBLE_DEVICES")
    return masked is None or masked.strip() not in {"", "-1"}


def resolve_device(
    runtime: RuntimeConfig,
    provider_preferences: dict[str, list[ProviderSpec]] | None = None,
) -> DevicePlan:
    """Resolve ``runtime.device`` into a concrete, ordered provider plan.

    Args:
        runtime: The runtime section of the settings.
        provider_preferences: Per-family provider preference lists, normally
            :attr:`Settings.execution_providers`. When omitted a sane built-in
            default is used, so the function stays usable in isolation.

    Returns:
        A :class:`DevicePlan` whose provider list is guaranteed non-empty and
        guaranteed to end in ``CPUExecutionProvider``.
    """
    preferences = provider_preferences or _default_preferences()
    present = available_providers()
    requested = runtime.device

    # `tensorrt` is a request for the CUDA family with TensorRT ranked first;
    # it is not a separate family as far as provider lookup is concerned.
    # Typed as a plain str from here on: the resolution below may widen it
    # to any available family, which is no longer the caller's Literal.
    target_family: str = "cuda" if requested == "tensorrt" else requested

    if target_family == "auto":
        target_family = _auto_select(present)
        degraded = False
    else:
        supported = available_families()
        if target_family not in supported or (target_family == "cuda" and not _cuda_visible()):
            target_family = "cpu"
            degraded = requested != "cpu"
        else:
            degraded = False

    specs = preferences.get(target_family) or preferences.get("cpu") or []
    providers: list[str] = []
    options: list[dict[str, Any]] = []

    for spec in specs:
        if spec.name not in present:
            continue
        if requested != "tensorrt" and spec.name == "TensorrtExecutionProvider":
            # TensorRT engine building costs minutes on a cold cache. Only opt
            # in when the operator asked for it by name.
            continue
        resolved = _resolve_options(spec, runtime)
        providers.append(spec.name)
        options.append(resolved)

    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
        options.append({})

    if len(providers) == 1 and providers[0] == "CPUExecutionProvider":
        target_family = "cpu"
        degraded = degraded or requested not in {"cpu", "auto"}

    return DevicePlan(
        family=target_family,
        providers=tuple(providers),
        provider_options=tuple(options),
        requested=requested,
        degraded=degraded,
        available=present,
    )


def _auto_select(present: tuple[str, ...]) -> str:
    """Pick the best available family when the operator said ``auto``."""
    for family in _AUTO_PROBE_ORDER:
        if family == "cpu":
            continue
        if family == "cuda" and not _cuda_visible():
            continue
        if any(_PROVIDER_FAMILY.get(name) == family for name in present):
            return family
    return "cpu"


def _resolve_options(spec: ProviderSpec, runtime: RuntimeConfig) -> dict[str, Any]:
    """Materialise provider options, injecting the configured GPU ordinal."""
    options = dict(spec.options)
    if spec.name in {"CUDAExecutionProvider", "DmlExecutionProvider", "ROCMExecutionProvider"}:
        options["device_id"] = runtime.gpu_device_id
    if spec.name == "TensorrtExecutionProvider":
        options.setdefault("device_id", runtime.gpu_device_id)
    return options


def _default_preferences() -> dict[str, list[ProviderSpec]]:
    """Built-in provider ordering, used when configuration supplies none."""
    return {
        "cuda": [
            ProviderSpec(name="TensorrtExecutionProvider", options={"trt_fp16_enable": True}),
            ProviderSpec(name="CUDAExecutionProvider"),
            ProviderSpec(name="CPUExecutionProvider"),
        ],
        "directml": [
            ProviderSpec(name="DmlExecutionProvider"),
            ProviderSpec(name="CPUExecutionProvider"),
        ],
        "coreml": [
            ProviderSpec(name="CoreMLExecutionProvider"),
            ProviderSpec(name="CPUExecutionProvider"),
        ],
        "cpu": [ProviderSpec(name="CPUExecutionProvider")],
    }


def resolve_from_settings(settings: Settings) -> DevicePlan:
    """Convenience wrapper resolving the plan straight from :class:`Settings`."""
    return resolve_device(settings.runtime, settings.execution_providers)


def default_thread_counts(runtime: RuntimeConfig) -> tuple[int, int]:
    """Return ``(intra_op, inter_op)`` thread counts for an ORT session.

    A value of ``0`` for ``intra_op_threads`` means "let ONNX Runtime decide",
    which is right on a dedicated node. Inside a CPU-limited container ORT sees
    the *host* core count and oversubscribes badly, so the cgroup CPU quota is
    honoured when one is present.
    """
    intra = runtime.intra_op_threads
    if intra == 0:
        quota = _cgroup_cpu_quota()
        if quota is not None:
            intra = max(1, quota)
    return intra, runtime.inter_op_threads


def _cgroup_cpu_quota() -> int | None:
    """Return the container's CPU limit in whole cores, or None if unlimited."""
    # cgroup v2
    v2 = "/sys/fs/cgroup/cpu.max"
    try:
        with open(v2, encoding="utf-8") as handle:  # noqa: PTH123 - procfs, not a data path
            quota_raw, period_raw = handle.read().split()
        if quota_raw != "max":
            return max(1, int(int(quota_raw) / int(period_raw)))
        return None
    except (OSError, ValueError):
        pass
    # cgroup v1
    try:
        with open(  # noqa: PTH123
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="utf-8"
        ) as handle:
            quota = int(handle.read().strip())
        with open(  # noqa: PTH123
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8"
        ) as handle:
            period = int(handle.read().strip())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return None


__all__ = [
    "DevicePlan",
    "available_families",
    "available_providers",
    "default_thread_counts",
    "resolve_device",
    "resolve_from_settings",
]
