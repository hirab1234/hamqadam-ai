"""Exception hierarchy.

A single root, :class:`HamqadamError`, carries an :class:`~hamqadam_ai.core.errors.ErrorCode`
plus a structured ``details`` mapping. The API layer can therefore turn any
internal failure into a well-formed error response without a chain of
``isinstance`` checks, and structured logging gets machine-readable context for
free.

Detail dictionaries must never contain image bytes, embeddings or PII. The
logging redaction processor is a safety net, not a licence.
"""

from __future__ import annotations

from typing import Any

from hamqadam_ai.core.errors import ErrorCode, ErrorSeverity


class HamqadamError(Exception):
    """Base class for every error raised by this service.

    Args:
        message: Human-readable description. Falls back to the code's default.
        code: The machine-readable error code. Subclasses supply a default.
        details: Structured, JSON-serialisable context. Must be PII-free.
        cause: The originating exception, preserved for the traceback chain.
    """

    #: Overridden by subclasses so callers can raise them with no arguments.
    default_code: ErrorCode = ErrorCode.AI_SERVICE_ERROR

    def __init__(
        self,
        message: str | None = None,
        *,
        code: ErrorCode | None = None,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.code: ErrorCode = code or self.default_code
        self.message: str = message or self.code.default_message
        self.details: dict[str, Any] = dict(details or {})
        self.cause: BaseException | None = cause
        super().__init__(self.message)
        if cause is not None:
            self.__cause__ = cause

    # -- Convenience accessors mirroring the code metadata --------------- #

    @property
    def http_status(self) -> int:
        """HTTP status this error maps to."""
        return self.code.http_status

    @property
    def severity(self) -> ErrorSeverity:
        """Severity band of the underlying error code."""
        return self.code.severity

    @property
    def retryable(self) -> bool:
        """Whether retrying the identical operation could plausibly succeed."""
        return self.code.retryable

    def to_dict(self) -> dict[str, Any]:
        """Render as the JSON error envelope returned by the API."""
        payload: dict[str, Any] = {
            "error_code": str(self.code),
            "message": self.message,
            "severity": str(self.severity),
            "retryable": self.retryable,
        }
        if self.details:
            payload["details"] = self.details
        return payload

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(code={self.code!s}, message={self.message!r})"


# --------------------------------------------------------------------------- #
# Configuration & infrastructure
# --------------------------------------------------------------------------- #


class ConfigurationError(HamqadamError):
    """The service configuration is internally inconsistent or incomplete."""

    default_code = ErrorCode.CONFIGURATION_ERROR


class ModelLoadError(HamqadamError):
    """A model artefact is missing, corrupt or could not be initialised."""

    default_code = ErrorCode.MODEL_LOAD_FAILED


class ModelChecksumError(ModelLoadError):
    """A model artefact's SHA-256 digest did not match the pinned value."""

    default_code = ErrorCode.MODEL_CHECKSUM_MISMATCH


class ModelNotLoadedError(HamqadamError):
    """A model was requested before it finished loading, or it failed to load."""

    default_code = ErrorCode.MODEL_NOT_LOADED


class InferenceError(HamqadamError):
    """A model forward pass raised."""

    default_code = ErrorCode.INFERENCE_FAILED


class DependencyUnavailableError(HamqadamError):
    """An optional third-party package required for this code path is absent.

    Raised instead of letting an ``ImportError`` escape, so the caller can fall
    back to an alternative adapter and the operator gets an actionable message
    naming the missing distribution.
    """

    default_code = ErrorCode.DEPENDENCY_UNAVAILABLE

    def __init__(
        self,
        package: str,
        *,
        purpose: str,
        extra: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        # An extra names a *group* in this project's own metadata, not a package,
        # so it has to be installed through the distribution. `pip install infra`
        # would either fail or fetch something unrelated from PyPI, which is
        # worse than no hint at all - a remediation that does not work costs the
        # reader the time they spend trusting it.
        hint = (
            f"pip install 'hamqadam-ai-verification[{extra}]'"
            if extra
            else f"pip install {package}"
        )
        super().__init__(
            f"Optional dependency {package!r} is required for {purpose}. Install it with: {hint}",
            details={"package": package, "purpose": purpose, "install_hint": hint},
            cause=cause,
        )


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


class ImageValidationError(HamqadamError):
    """The supplied image failed a pre-decode structural check."""

    default_code = ErrorCode.INVALID_IMAGE


class ImageDecodeError(ImageValidationError):
    """The image bytes could not be decoded into a pixel array."""

    default_code = ErrorCode.IMAGE_DECODE_FAILED


class UnsupportedImageFormatError(ImageValidationError):
    """The container format is outside the supported allow-list."""

    default_code = ErrorCode.UNSUPPORTED_IMAGE_FORMAT


class ImageTooLargeError(ImageValidationError):
    """The encoded payload or decoded pixel count exceeds the hard ceiling."""

    default_code = ErrorCode.IMAGE_TOO_LARGE


class ImageTooSmallError(ImageValidationError):
    """The image is below the minimum usable resolution."""

    default_code = ErrorCode.IMAGE_TOO_SMALL


class DecompressionBombError(ImageValidationError):
    """The image expands to an unreasonable pixel count and was refused."""

    default_code = ErrorCode.DECOMPRESSION_BOMB


class RequestValidationError(HamqadamError):
    """The request payload violated the API schema."""

    default_code = ErrorCode.VALIDATION_ERROR


# --------------------------------------------------------------------------- #
# Face analysis outcomes
# --------------------------------------------------------------------------- #


class FaceAnalysisError(HamqadamError):
    """Base class for negative outcomes of the face-analysis stage."""

    default_code = ErrorCode.FACE_NOT_DETECTED


class FaceNotDetectedError(FaceAnalysisError):
    """No face survived the detection policy for a given image role."""

    default_code = ErrorCode.FACE_NOT_DETECTED


class MultipleFacesDetectedError(FaceAnalysisError):
    """More than one qualifying face is present in a single-person image."""

    default_code = ErrorCode.MULTIPLE_FACES_DETECTED


class FaceOccludedError(FaceAnalysisError):
    """The face is obstructed beyond the configured tolerance."""

    default_code = ErrorCode.FACE_OCCLUDED


class FacePoseOutOfRangeError(FaceAnalysisError):
    """Head orientation exceeds the configured hard pose limits."""

    default_code = ErrorCode.FACE_POSE_OUT_OF_RANGE


class FaceTooSmallError(FaceAnalysisError):
    """The face box is too small in absolute or relative terms."""

    default_code = ErrorCode.FACE_TOO_SMALL


class QualityRejectedError(HamqadamError):
    """Image quality is below the minimum required for reliable analysis."""

    default_code = ErrorCode.LOW_IMAGE_QUALITY


# --------------------------------------------------------------------------- #
# Pipeline control flow
# --------------------------------------------------------------------------- #


class PipelineTimeoutError(HamqadamError):
    """The verification pipeline exceeded its wall-clock budget."""

    default_code = ErrorCode.PROCESSING_TIMEOUT


class VectorStoreError(HamqadamError):
    """The vector database rejected an operation or was unreachable."""

    default_code = ErrorCode.VECTOR_DB_ERROR


class CacheError(HamqadamError):
    """The cache backend rejected an operation or was unreachable."""

    default_code = ErrorCode.CACHE_ERROR


class RateLimitExceededError(HamqadamError):
    """The caller exhausted its request budget."""

    default_code = ErrorCode.RATE_LIMITED

    def __init__(self, retry_after_seconds: float, *, scope: str = "global") -> None:
        super().__init__(
            f"Rate limit exceeded for scope {scope!r}. "
            f"Retry after {retry_after_seconds:.1f}s.",
            details={"retry_after_seconds": round(retry_after_seconds, 3), "scope": scope},
        )
        self.retry_after_seconds = retry_after_seconds


class AuthenticationError(HamqadamError):
    """Missing or invalid API credentials."""

    default_code = ErrorCode.UNAUTHORIZED


__all__ = [
    "AuthenticationError",
    "CacheError",
    "ConfigurationError",
    "DecompressionBombError",
    "DependencyUnavailableError",
    "FaceAnalysisError",
    "FaceNotDetectedError",
    "FaceOccludedError",
    "FacePoseOutOfRangeError",
    "FaceTooSmallError",
    "HamqadamError",
    "ImageDecodeError",
    "ImageTooLargeError",
    "ImageTooSmallError",
    "ImageValidationError",
    "InferenceError",
    "ModelChecksumError",
    "ModelLoadError",
    "ModelNotLoadedError",
    "MultipleFacesDetectedError",
    "PipelineTimeoutError",
    "QualityRejectedError",
    "RateLimitExceededError",
    "RequestValidationError",
    "UnsupportedImageFormatError",
    "VectorStoreError",
]
