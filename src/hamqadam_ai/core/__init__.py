"""Domain core: configuration, error taxonomy, execution context and retries.

Nothing in this package performs network or model I/O. It is safe to import
from any layer and from any test without side effects.
"""

from __future__ import annotations

from hamqadam_ai.core.config import (
    Settings,
    get_settings,
    reload_settings,
)
from hamqadam_ai.core.constants import (
    FIVE_POINT_LANDMARK_NAMES,
    FaceRegion,
    ImageRole,
)
from hamqadam_ai.core.errors import ErrorCode, ErrorSeverity
from hamqadam_ai.core.exceptions import (
    ConfigurationError,
    DependencyUnavailableError,
    FaceNotDetectedError,
    HamqadamError,
    ImageDecodeError,
    ImageValidationError,
    ModelLoadError,
    MultipleFacesDetectedError,
    PipelineTimeoutError,
    QualityRejectedError,
)

__all__ = [
    "FIVE_POINT_LANDMARK_NAMES",
    "ConfigurationError",
    "DependencyUnavailableError",
    "ErrorCode",
    "ErrorSeverity",
    "FaceNotDetectedError",
    "FaceRegion",
    "HamqadamError",
    "ImageDecodeError",
    "ImageRole",
    "ImageValidationError",
    "ModelLoadError",
    "MultipleFacesDetectedError",
    "PipelineTimeoutError",
    "QualityRejectedError",
    "Settings",
    "get_settings",
    "reload_settings",
]
