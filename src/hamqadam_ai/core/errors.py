"""Canonical error taxonomy.

Section 19 of the requirements document fixes a set of error codes that the
Backend switches on. Those codes are reproduced here verbatim and are treated
as a frozen public contract - renaming one is a breaking API change.

Additional codes are namespaced below the contract block. They exist so the
service can be diagnosed precisely without forcing every internal failure into
the generic ``AI_SERVICE_ERROR`` bucket, and every one of them carries an
explicit HTTP status and retryability flag.
"""

from __future__ import annotations

from enum import StrEnum
from http import HTTPStatus
from typing import Final, NamedTuple


class ErrorSeverity(StrEnum):
    """How an error should be treated by the caller and by alerting.

    ``CLIENT``
        The request was malformed or the supplied media is unusable. Retrying
        the identical payload will fail identically. Never pages an on-call.

    ``BUSINESS``
        The pipeline ran correctly and produced a negative outcome (no face,
        duplicate account). This is a *result*, not a fault; it is surfaced
        with HTTP 200 inside the analysis response and only raised as an
        exception when a caller explicitly opts into fail-fast behaviour.

    ``TRANSIENT``
        A dependency wobbled. Safe to retry with backoff.

    ``FATAL``
        The service is misconfigured or a model is missing. Fails readiness.
    """

    CLIENT = "client"
    BUSINESS = "business"
    TRANSIENT = "transient"
    FATAL = "fatal"


class ErrorDefinition(NamedTuple):
    """Static metadata attached to every :class:`ErrorCode`."""

    http_status: int
    severity: ErrorSeverity
    retryable: bool
    message: str


class ErrorCode(StrEnum):
    """Stable machine-readable error identifiers.

    Use :meth:`definition` to obtain the HTTP status, severity, retryability
    and default human-readable message for a code.
    """

    # ------------------------------------------------------------------ #
    # Contract codes - requirements document, section 19. DO NOT RENAME.
    # ------------------------------------------------------------------ #
    INVALID_IMAGE = "INVALID_IMAGE"
    FACE_NOT_DETECTED = "FACE_NOT_DETECTED"
    MULTIPLE_FACES_DETECTED = "MULTIPLE_FACES_DETECTED"
    LOW_IMAGE_QUALITY = "LOW_IMAGE_QUALITY"
    CNIC_OCR_FAILED = "CNIC_OCR_FAILED"
    FACE_MISMATCH = "FACE_MISMATCH"
    DUPLICATE_FACE_DETECTED = "DUPLICATE_FACE_DETECTED"
    AI_SERVICE_ERROR = "AI_SERVICE_ERROR"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"

    # ------------------------------------------------------------------ #
    # Extended codes - internal precision, additive and backwards compatible.
    # ------------------------------------------------------------------ #

    # Input / media
    UNSUPPORTED_IMAGE_FORMAT = "UNSUPPORTED_IMAGE_FORMAT"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    DECOMPRESSION_BOMB = "DECOMPRESSION_BOMB"
    VALIDATION_ERROR = "VALIDATION_ERROR"

    # Face analysis
    FACE_OCCLUDED = "FACE_OCCLUDED"
    FACE_POSE_OUT_OF_RANGE = "FACE_POSE_OUT_OF_RANGE"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    FACE_TRUNCATED = "FACE_TRUNCATED"
    FACE_NOT_VISIBLE = "FACE_NOT_VISIBLE"
    CNIC_FACE_NOT_FOUND = "CNIC_FACE_NOT_FOUND"

    # Document analysis
    #
    # Distinct from CNIC_OCR_FAILED, and the distinction is the whole point:
    # unreadable text asks the user to retake the photograph, whereas the
    # wrong document asks them to find a different one. Collapsing the two
    # sends half of them round a loop that cannot succeed.
    CNIC_NOT_RECOGNISED = "CNIC_NOT_RECOGNISED"

    # Infrastructure
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_CHECKSUM_MISMATCH = "MODEL_CHECKSUM_MISMATCH"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    VECTOR_DB_ERROR = "VECTOR_DB_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    QUEUE_ERROR = "QUEUE_ERROR"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"

    # Transport / policy
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"

    @property
    def definition(self) -> ErrorDefinition:
        """Return the static metadata for this code."""
        return _ERROR_DEFINITIONS[self]

    @property
    def http_status(self) -> int:
        """HTTP status code to return when this error terminates a request."""
        return _ERROR_DEFINITIONS[self].http_status

    @property
    def severity(self) -> ErrorSeverity:
        """Severity band, driving alerting and retry behaviour."""
        return _ERROR_DEFINITIONS[self].severity

    @property
    def retryable(self) -> bool:
        """Whether an identical retry has a realistic chance of succeeding."""
        return _ERROR_DEFINITIONS[self].retryable

    @property
    def default_message(self) -> str:
        """Human-readable default description, safe to show to an operator."""
        return _ERROR_DEFINITIONS[self].message


_C = ErrorSeverity.CLIENT
_B = ErrorSeverity.BUSINESS
_T = ErrorSeverity.TRANSIENT
_F = ErrorSeverity.FATAL

_ERROR_DEFINITIONS: Final[dict[ErrorCode, ErrorDefinition]] = {
    # -- Contract codes ------------------------------------------------- #
    ErrorCode.INVALID_IMAGE: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _C, False,
        "The supplied image could not be interpreted as a valid picture.",
    ),
    ErrorCode.FACE_NOT_DETECTED: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _B, False,
        "No human face was found in the image.",
    ),
    ErrorCode.MULTIPLE_FACES_DETECTED: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _B, False,
        "More than one person is present; a single-person image is required.",
    ),
    ErrorCode.LOW_IMAGE_QUALITY: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _B, False,
        "Image quality is below the minimum required for reliable analysis.",
    ),
    ErrorCode.CNIC_OCR_FAILED: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _B, False,
        "No readable text could be extracted from the CNIC image.",
    ),
    ErrorCode.FACE_MISMATCH: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _B, False,
        "The compared faces do not belong to the same person.",
    ),
    ErrorCode.DUPLICATE_FACE_DETECTED: ErrorDefinition(
        HTTPStatus.CONFLICT, _B, False,
        "This face already exists on another account.",
    ),
    ErrorCode.AI_SERVICE_ERROR: ErrorDefinition(
        HTTPStatus.INTERNAL_SERVER_ERROR, _T, True,
        "An unexpected error occurred inside the AI verification service.",
    ),
    ErrorCode.PROCESSING_TIMEOUT: ErrorDefinition(
        HTTPStatus.GATEWAY_TIMEOUT, _T, True,
        "Verification did not finish within the allotted time budget.",
    ),

    # -- Input / media -------------------------------------------------- #
    ErrorCode.UNSUPPORTED_IMAGE_FORMAT: ErrorDefinition(
        HTTPStatus.UNSUPPORTED_MEDIA_TYPE, _C, False,
        "Image format is not supported. Use JPEG, PNG, WEBP or BMP.",
    ),
    ErrorCode.IMAGE_TOO_LARGE: ErrorDefinition(
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE, _C, False,
        "Image exceeds the maximum permitted size.",
    ),
    ErrorCode.IMAGE_TOO_SMALL: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _C, False,
        "Image resolution is below the minimum permitted size.",
    ),
    ErrorCode.IMAGE_DECODE_FAILED: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _C, False,
        "The image bytes are corrupt or truncated.",
    ),
    ErrorCode.DECOMPRESSION_BOMB: ErrorDefinition(
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE, _C, False,
        "The image expands to an unreasonable number of pixels and was refused.",
    ),
    ErrorCode.VALIDATION_ERROR: ErrorDefinition(
        HTTPStatus.BAD_REQUEST, _C, False,
        "The request payload failed schema validation.",
    ),

    # -- Face analysis -------------------------------------------------- #
    ErrorCode.FACE_OCCLUDED: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _B, False,
        "The face is significantly obstructed.",
    ),
    ErrorCode.FACE_POSE_OUT_OF_RANGE: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _B, False,
        "Head orientation is too far from frontal.",
    ),
    ErrorCode.FACE_TOO_SMALL: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _B, False,
        "The detected face occupies too few pixels for reliable recognition.",
    ),
    ErrorCode.FACE_TRUNCATED: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _B, False,
        "The face is cut off by the edge of the frame.",
    ),
    ErrorCode.FACE_NOT_VISIBLE: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _B, False,
        "Overall face visibility is below the acceptable threshold.",
    ),
    ErrorCode.CNIC_FACE_NOT_FOUND: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _B, False,
        "No portrait could be located on the CNIC image.",
    ),

    # -- Document analysis ---------------------------------------------- #
    ErrorCode.CNIC_NOT_RECOGNISED: ErrorDefinition(
        HTTPStatus.UNPROCESSABLE_ENTITY, _B, False,
        "The image is readable but is not a Pakistani CNIC.",
    ),

    # -- Infrastructure ------------------------------------------------- #
    ErrorCode.MODEL_NOT_LOADED: ErrorDefinition(
        HTTPStatus.SERVICE_UNAVAILABLE, _F, True,
        "A required model is not loaded; the service is not ready.",
    ),
    ErrorCode.MODEL_LOAD_FAILED: ErrorDefinition(
        HTTPStatus.SERVICE_UNAVAILABLE, _F, False,
        "A model artefact could not be loaded.",
    ),
    ErrorCode.MODEL_CHECKSUM_MISMATCH: ErrorDefinition(
        HTTPStatus.SERVICE_UNAVAILABLE, _F, False,
        "A model artefact failed integrity verification and was refused.",
    ),
    ErrorCode.INFERENCE_FAILED: ErrorDefinition(
        HTTPStatus.INTERNAL_SERVER_ERROR, _T, True,
        "A model forward pass failed.",
    ),
    ErrorCode.VECTOR_DB_ERROR: ErrorDefinition(
        HTTPStatus.SERVICE_UNAVAILABLE, _T, True,
        "The vector database is unreachable or returned an error.",
    ),
    ErrorCode.CACHE_ERROR: ErrorDefinition(
        HTTPStatus.SERVICE_UNAVAILABLE, _T, True,
        "The cache backend is unreachable or returned an error.",
    ),
    ErrorCode.QUEUE_ERROR: ErrorDefinition(
        HTTPStatus.SERVICE_UNAVAILABLE, _T, True,
        "The message broker is unreachable or returned an error.",
    ),
    ErrorCode.DEPENDENCY_UNAVAILABLE: ErrorDefinition(
        HTTPStatus.SERVICE_UNAVAILABLE, _T, True,
        "A required optional dependency is not installed in this environment.",
    ),
    ErrorCode.CONFIGURATION_ERROR: ErrorDefinition(
        HTTPStatus.INTERNAL_SERVER_ERROR, _F, False,
        "The service configuration is invalid.",
    ),

    # -- Transport / policy --------------------------------------------- #
    ErrorCode.UNAUTHORIZED: ErrorDefinition(
        HTTPStatus.UNAUTHORIZED, _C, False,
        "Missing or invalid API credentials.",
    ),
    ErrorCode.FORBIDDEN: ErrorDefinition(
        HTTPStatus.FORBIDDEN, _C, False,
        "The credentials presented are not permitted to perform this action.",
    ),
    ErrorCode.RATE_LIMITED: ErrorDefinition(
        HTTPStatus.TOO_MANY_REQUESTS, _C, True,
        "Request rate limit exceeded.",
    ),
    ErrorCode.PAYLOAD_TOO_LARGE: ErrorDefinition(
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE, _C, False,
        "The request body exceeds the maximum permitted size.",
    ),
}

#: The subset of codes frozen by the requirements document. Used by a
#: contract test that fails the build if one of them is ever removed.
CONTRACT_ERROR_CODES: Final[frozenset[ErrorCode]] = frozenset(
    {
        ErrorCode.INVALID_IMAGE,
        ErrorCode.FACE_NOT_DETECTED,
        ErrorCode.MULTIPLE_FACES_DETECTED,
        ErrorCode.LOW_IMAGE_QUALITY,
        ErrorCode.CNIC_OCR_FAILED,
        ErrorCode.FACE_MISMATCH,
        ErrorCode.DUPLICATE_FACE_DETECTED,
        ErrorCode.AI_SERVICE_ERROR,
        ErrorCode.PROCESSING_TIMEOUT,
    }
)

__all__ = ["CONTRACT_ERROR_CODES", "ErrorCode", "ErrorDefinition", "ErrorSeverity"]
