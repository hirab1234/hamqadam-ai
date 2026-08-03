"""Application services.

A service orchestrates the capability layer and applies policy. It is the
boundary where "run these models" becomes "decide whether this image is
acceptable for identity verification".

Services never import FastAPI, never touch HTTP, and never read configuration
from the environment directly - everything arrives through the constructor.
That makes them directly unit-testable and reusable from the queue workers as
well as the HTTP surface.
"""

from __future__ import annotations

from hamqadam_ai.services.cnic_face_service import (
    CnicFaceService,
    build_cnic_face_service,
)
from hamqadam_ai.services.duplicate_service import (
    DuplicateService,
    build_duplicate_service,
)
from hamqadam_ai.services.embedding_service import (
    EmbeddingService,
    build_embedding_service,
)
from hamqadam_ai.services.face_detection_service import (
    FaceDetectionService,
    build_face_detection_service,
)
from hamqadam_ai.services.fraud_service import (
    FraudRiskService,
    build_fraud_service,
)
from hamqadam_ai.services.matching_service import (
    MatchingService,
    build_matching_service,
)
from hamqadam_ai.services.ocr_service import (
    OcrService,
    build_ocr_service,
)
from hamqadam_ai.services.profile_service import (
    ProfileAnalysisService,
    build_profile_service,
)
from hamqadam_ai.services.quality_service import (
    QualityService,
    build_quality_service,
)

__all__ = [
    "CnicFaceService",
    "DuplicateService",
    "EmbeddingService",
    "FaceDetectionService",
    "FraudRiskService",
    "MatchingService",
    "OcrService",
    "ProfileAnalysisService",
    "QualityService",
    "build_cnic_face_service",
    "build_duplicate_service",
    "build_embedding_service",
    "build_face_detection_service",
    "build_fraud_service",
    "build_matching_service",
    "build_ocr_service",
    "build_profile_service",
    "build_quality_service",
]
