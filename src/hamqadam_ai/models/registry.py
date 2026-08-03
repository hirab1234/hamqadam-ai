"""The versioned model registry.

Single point of truth for "which artefact, at which version, on which device".
Responsibilities:

* Resolve a registry key to a filesystem path under the model store.
* Verify the artefact's SHA-256 against the pinned digest before it is loaded.
* Build and cache one session per key for the process lifetime.
* Own the bounded thread pool that blocking forward passes are dispatched to.
* Report readiness: a required model that failed to load makes the service
  unready rather than letting it serve degraded results silently.
* Produce the ``model_versions`` block returned in every API response.

Thread-safe. Concurrent first-touch of the same key blocks on a per-key lock so
one artefact is never loaded twice.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hamqadam_ai.core.config import ModelSpec, Settings, get_settings
from hamqadam_ai.core.device import DevicePlan, resolve_from_settings
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import (
    ConfigurationError,
    ModelChecksumError,
    ModelLoadError,
    ModelNotLoadedError,
)
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.models.base import LoadedModel
from hamqadam_ai.models.onnx_backend import load_onnx_model
from hamqadam_ai.utils.hashing import sha256_file

log = get_logger(__name__)


@dataclass(slots=True)
class ModelStatus:
    """Load outcome for one registry key, surfaced on ``/health``."""

    key: str
    version: str
    enabled: bool
    required: bool
    state: str  # not_loaded | loaded | failed | disabled | missing_file
    path: str | None = None
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable summary."""
        payload: dict[str, Any] = {
            "key": self.key,
            "version": self.version,
            "state": self.state,
            "enabled": self.enabled,
            "required": self.required,
        }
        if self.error:
            payload["error"] = self.error
        if self.detail:
            payload["detail"] = self.detail
        return payload


class ModelRegistry:
    """Loads, verifies, caches and reports on every model artefact.

    Args:
        settings: Service configuration. Loaded from the environment when
            omitted.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._plan: DevicePlan = resolve_from_settings(self._settings)
        self._models: dict[str, LoadedModel] = {}
        self._status: dict[str, ModelStatus] = {}
        self._key_locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=self._settings.runtime.inference_workers,
            thread_name_prefix="hqai-infer",
        )
        self._closed = False

        for key, spec in self._settings.models.items():
            self._status[key] = ModelStatus(
                key=key,
                version=spec.version,
                enabled=spec.enabled,
                required=spec.required,
                state="disabled" if not spec.enabled else "not_loaded",
                path=str(self.resolve_path(key)) if spec.enabled else None,
            )

    # -- Paths and verification ------------------------------------------- #

    @property
    def store_root(self) -> Path:
        """Absolute path to the model store."""
        return self._settings.storage.resolved_model_dir

    @property
    def device_plan(self) -> DevicePlan:
        """The execution plan every session in this registry uses."""
        return self._plan

    @property
    def executor(self) -> ThreadPoolExecutor:
        """The shared inference thread pool."""
        return self._executor

    def resolve_path(self, key: str) -> Path:
        """Return the absolute path of an artefact under the model store.

        Args:
            key: Registry key.

        Raises:
            ConfigurationError: if the key is undeclared, or if the declared
                path escapes the model store. The latter is a hard failure
                rather than a warning: a config file that can point the loader
                at an arbitrary filesystem location is a code-execution vector.
        """
        spec = self._settings.model_spec(key)
        candidate = Path(spec.path)
        resolved = (
            candidate if candidate.is_absolute() else self.store_root / candidate
        )
        resolved = resolved.resolve()

        if not candidate.is_absolute():
            store = self.store_root.resolve()
            try:
                resolved.relative_to(store)
            except ValueError as exc:
                raise ConfigurationError(
                    f"Model {key!r} declares a path that escapes the model store.",
                    details={"model": key, "declared": spec.path, "store": str(store)},
                ) from exc
        return resolved

    def verify_artifact(self, key: str) -> str | None:
        """Verify an artefact's digest against the pinned value.

        Args:
            key: Registry key.

        Returns:
            The computed digest, or ``None`` when the spec pins no digest and
            verification is therefore not possible.

        Raises:
            ModelLoadError: if the file is missing.
            ModelChecksumError: if the digest does not match.
        """
        spec = self._settings.model_spec(key)
        path = self.resolve_path(key)

        if not path.is_file():
            raise ModelLoadError(
                f"Model artefact for {key!r} is missing.",
                details={
                    "model": key,
                    "path": str(path),
                    "hint": "Run: python scripts/download_models.py",
                },
            )

        if not spec.sha256:
            if self._settings.model_store.verify_checksum and spec.required:
                log.warning(
                    "model.checksum_unpinned",
                    model=key,
                    path=str(path),
                    note="required model has no pinned sha256; integrity is unverified",
                )
            return None

        digest = sha256_file(path)
        if digest.lower() != spec.sha256.strip().lower():
            if self._settings.model_store.verify_checksum:
                raise ModelChecksumError(
                    f"Model {key!r} failed integrity verification and was refused.",
                    details={
                        "model": key,
                        "path": str(path),
                        "expected_sha256": spec.sha256,
                        "actual_sha256": digest,
                    },
                )
            log.warning(
                "model.checksum_mismatch_ignored",
                model=key,
                expected=spec.sha256[:16],
                actual=digest[:16],
                note="verify_checksum is disabled; loading anyway",
            )
        return digest

    # -- Loading ----------------------------------------------------------- #

    def _lock_for(self, key: str) -> threading.Lock:
        """Return the per-key load lock, creating it on first use."""
        with self._registry_lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def get(self, key: str) -> LoadedModel:
        """Return a loaded model, loading it on first access.

        Args:
            key: Registry key.

        Returns:
            The loaded model.

        Raises:
            ModelNotLoadedError: if the model is disabled or the registry is
                closed.
            ModelLoadError: if loading fails.
        """
        if self._closed:
            raise ModelNotLoadedError(
                "The model registry has been closed.", details={"model": key}
            )

        cached = self._models.get(key)
        if cached is not None:
            return cached

        spec = self._settings.model_spec(key)
        if not spec.enabled:
            raise ModelNotLoadedError(
                f"Model {key!r} is disabled in configuration.",
                details={"model": key},
            )

        with self._lock_for(key):
            cached = self._models.get(key)
            if cached is not None:
                return cached
            model = self._load(key, spec)
            self._models[key] = model
            return model

    def try_get(self, key: str) -> LoadedModel | None:
        """Return a loaded model, or ``None`` if it is unavailable.

        Used by the detector fallback chain, which must be able to skip a
        detector whose weights were never downloaded without that being an
        error at all.
        """
        try:
            return self.get(key)
        except (ModelLoadError, ModelNotLoadedError, ConfigurationError) as exc:
            log.info(
                "model.unavailable",
                model=key,
                reason=str(exc),
                error_code=str(getattr(exc, "code", "?")),
            )
            return None

    def _load(self, key: str, spec: ModelSpec) -> LoadedModel:
        """Verify and load one artefact, recording the outcome in the status map."""
        status = self._status[key]
        try:
            digest = self.verify_artifact(key)
            path = self.resolve_path(key)

            if spec.kind != "onnx":
                raise ModelLoadError(
                    f"Model {key!r} has kind {spec.kind!r}, which the registry does "
                    "not load directly. Non-ONNX artefacts are owned by their "
                    "capability adapter.",
                    details={"model": key, "kind": spec.kind},
                )

            model = load_onnx_model(
                spec=spec,
                model_path=path,
                plan=self._plan,
                runtime=self._settings.runtime,
                model_key=key,
                sha256=digest,
                executor=self._executor,
                warmup_iterations=(
                    self._settings.runtime.warmup_iterations
                    if self._settings.runtime.warmup_on_startup
                    else 0
                ),
            )
        except (ModelLoadError, ModelChecksumError, ConfigurationError) as exc:
            # An absent *optional* artefact is an expected deployment state,
            # not a fault - the fallback chain exists precisely for it. Logging
            # it at ERROR trains operators to ignore genuine errors, so the
            # level tracks whether the model is actually required.
            missing_optional = (
                not spec.required and exc.code is ErrorCode.MODEL_LOAD_FAILED
            )
            status.state = "missing_file" if missing_optional else "failed"
            status.error = exc.message
            status.detail = exc.details
            emit = log.info if missing_optional else log.error
            emit(
                "model.load_failed",
                model=key,
                required=spec.required,
                error_code=str(exc.code),
                reason=exc.message,
            )
            raise

        status.state = "loaded"
        status.error = None
        status.detail = model.handle.describe()
        return model

    def preload(self, keys: list[str] | None = None) -> dict[str, ModelStatus]:
        """Eagerly load models at start-up.

        A required model that fails to load raises: the pod should crash-loop
        with a clear error rather than accept traffic it cannot serve. An
        optional model that fails is logged and the service continues.

        Args:
            keys: Specific keys to load. Defaults to every enabled model.

        Returns:
            The status map after loading.

        Raises:
            ModelLoadError: if a *required* model could not be loaded.
        """
        targets = keys if keys is not None else [
            key for key, spec in self._settings.models.items() if spec.enabled
        ]

        for key in targets:
            spec = self._settings.models.get(key)
            if spec is None or not spec.enabled:
                continue
            try:
                self.get(key)
            except (ModelLoadError, ModelNotLoadedError, ConfigurationError):
                if spec.required:
                    raise
                log.warning(
                    "model.optional_unavailable",
                    model=key,
                    note="service will run without this model",
                )

        loaded = sum(1 for s in self._status.values() if s.state == "loaded")
        log.info(
            "registry.preload_complete",
            loaded=loaded,
            total=len(targets),
            device=self._plan.family,
            providers=list(self._plan.providers),
            degraded=self._plan.degraded,
        )
        return dict(self._status)

    # -- Reporting ---------------------------------------------------------- #

    def is_loaded(self, key: str) -> bool:
        """Whether ``key`` currently has a live session."""
        return key in self._models

    @property
    def ready(self) -> bool:
        """True when every *required* enabled model has loaded successfully."""
        return all(
            status.state == "loaded"
            for status in self._status.values()
            if status.required and status.enabled
        )

    def status_report(self) -> dict[str, Any]:
        """Full readiness report for the health endpoint."""
        return {
            "ready": self.ready,
            "device": self._plan.describe(),
            "models": [status.as_dict() for status in self._status.values()],
            "inference_workers": self._settings.runtime.inference_workers,
        }

    def version_map(self) -> dict[str, str]:
        """Map of model key to version, for the response's ``model_versions``.

        Only *loaded* models appear: reporting the version of an artefact that
        never loaded would misrepresent which code actually produced the result.
        """
        return {
            key: model.handle.version for key, model in sorted(self._models.items())
        }

    def stats(self) -> dict[str, dict[str, Any]]:
        """Per-model inference counters."""
        return {key: model.stats for key, model in self._models.items()}

    def __iter__(self) -> Iterator[tuple[str, LoadedModel]]:
        """Iterate over currently-loaded ``(key, model)`` pairs."""
        return iter(self._models.items())

    # -- Lifecycle ---------------------------------------------------------- #

    @property
    def closed(self) -> bool:
        """Whether this registry has been shut down.

        A closed registry cannot load or serve anything: the thread pool is gone
        and the flag is never cleared. `get_registry` reads this to avoid handing
        out an instance that can only fail.
        """
        return self._closed

    def close(self) -> None:
        """Release every session and shut down the thread pool. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for key, model in list(self._models.items()):
            try:
                model.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                log.warning("model.close_failed", model=key, reason=str(exc))
        self._models.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)
        log.info("registry.closed")

    def __enter__(self) -> ModelRegistry:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Process-wide accessor
# --------------------------------------------------------------------------- #

_registry: ModelRegistry | None = None
_registry_singleton_lock = threading.Lock()


def get_registry(settings: Settings | None = None) -> ModelRegistry:
    """Return the process-wide registry, creating it on first call.

    A **closed** registry is replaced rather than returned. `close()` marks the
    instance dead permanently and shuts down its thread pool, but it cannot
    clear this global - it has no reference to it - so anything that closed the
    registry directly rather than through `reset_registry` left every later
    caller holding a corpse. Every subsequent model load then failed with
    "could not be loaded", which points at a missing weights file and not at a
    lifecycle bug.

    That is reachable in production, not only from tests: the API's shutdown
    releases its services' sessions, so a second `create_app` in the same
    process - two apps mounted together, an in-process restart, an embedded
    consumer alongside the server - would find nothing loadable. It surfaced
    here because an in-process `TestClient` shutdown silently disabled every
    test that ran afterwards.

    Rebuilding is the right response rather than raising: the caller asked for a
    working registry, weights are on disk, and reloading them is exactly what a
    first call would have done.
    """
    global _registry
    if _registry is not None and not _registry.closed:
        return _registry
    with _registry_singleton_lock:
        if _registry is None or _registry.closed:
            _registry = ModelRegistry(settings)
    return _registry


def reset_registry() -> None:
    """Close and discard the process-wide registry.

    Used by the FastAPI shutdown hook and by tests that need a clean slate.
    """
    global _registry
    with _registry_singleton_lock:
        if _registry is not None:
            _registry.close()
        _registry = None


__all__ = ["ModelRegistry", "ModelStatus", "get_registry", "reset_registry"]
