"""Model loading, versioning and inference infrastructure.

Every model artefact the service uses is reached through :class:`ModelRegistry`,
which resolves its path from ``configs/models.yaml``, verifies its SHA-256
digest, builds an ONNX Runtime session on the resolved device, warms it up and
caches it for the process lifetime.

Nothing in this package knows what a face is. It is the boundary between
"a tensor goes in, a tensor comes out" and the capability modules that give
those tensors meaning.
"""

from __future__ import annotations

from hamqadam_ai.models.base import InferenceBackend, LoadedModel, ModelHandle
from hamqadam_ai.models.onnx_backend import OnnxModel, build_session
from hamqadam_ai.models.registry import ModelRegistry, get_registry, reset_registry

__all__ = [
    "InferenceBackend",
    "LoadedModel",
    "ModelHandle",
    "ModelRegistry",
    "OnnxModel",
    "build_session",
    "get_registry",
    "reset_registry",
]
