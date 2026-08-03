"""Pydantic contracts for everything crossing the service boundary.

Schemas are the *only* thing the Backend integrates against, so they are
treated as a versioned public API: fields are added, never renamed or removed
within a major version, and every field carries a description that flows into
the generated OpenAPI document.
"""

from __future__ import annotations

from hamqadam_ai.schemas.common import (
    AnalysisWarning,
    ErrorEnvelope,
    ModelVersions,
    ProcessingTime,
    ScoreBreakdown,
    to_percent,
)
from hamqadam_ai.schemas.detection import (
    BoundingBoxModel,
    DetectedFaceModel,
    FaceDetectionResult,
    LandmarkModel,
    OcclusionReport,
    PoseEstimate,
    RegionOcclusion,
    VisibilityBreakdown,
)
from hamqadam_ai.schemas.duplicate import (
    DuplicateCandidate,
    DuplicateCheckResult,
    EnrolmentResult,
)
from hamqadam_ai.schemas.embedding import EmbeddingBatchResult, EmbeddingResult
from hamqadam_ai.schemas.fraud import (
    FamilyContributionModel,
    FraudRiskResult,
    FraudSignalModel,
)
from hamqadam_ai.schemas.matching import ComparisonResult, MatchingResult
from hamqadam_ai.schemas.ocr import CnicOcrResult, ExtractedField
from hamqadam_ai.schemas.profile import (
    AuthenticityFindingModel,
    AuthenticitySignalModel,
    ProfileAnalysisResult,
)
from hamqadam_ai.schemas.quality import MetricDetail, QualityResult
from hamqadam_ai.schemas.verification import (
    StageStatus,
    VerificationRequest,
    VerificationResult,
)

__all__ = [
    "BoundingBoxModel",
    "DetectedFaceModel",
    "DuplicateCandidate",
    "DuplicateCheckResult",
    "EmbeddingBatchResult",
    "EnrolmentResult",
    "EmbeddingResult",
    "CnicOcrResult",
    "ComparisonResult",
    "ErrorEnvelope",
    "ExtractedField",
    "FaceDetectionResult",
    "FamilyContributionModel",
    "FraudRiskResult",
    "FraudSignalModel",
    "LandmarkModel",
    "MatchingResult",
    "MetricDetail",
    "ModelVersions",
    "OcclusionReport",
    "PoseEstimate",
    "ProcessingTime",
    "QualityResult",
    "StageStatus",
    "VerificationRequest",
    "VerificationResult",
    "RegionOcclusion",
    "ScoreBreakdown",
    "VisibilityBreakdown",
    "AnalysisWarning",
    "AuthenticityFindingModel",
    "AuthenticitySignalModel",
    "ProfileAnalysisResult",
    "to_percent",
]
