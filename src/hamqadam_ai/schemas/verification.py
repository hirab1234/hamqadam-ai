"""MODULE 10 request and response contract - the whole verification.

This is the only schema the Backend has to understand. Everything else in
``schemas/`` is nested inside it, so a caller who wants the headline reads four
fields and a caller who wants to explain a rejection to a user has every stage's
findings underneath.

Recommendation, not decision
----------------------------
The field is called ``recommendation`` and the enum values are advice. Section
25 of the requirements document puts the account outcome with the Backend,
which knows things this service never will. Naming it ``decision`` would invite
a caller to treat it as one.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import Field, computed_field

from hamqadam_ai.core.constants import Recommendation, RiskLevel
from hamqadam_ai.schemas.cnic_face import CnicPortraitResult
from hamqadam_ai.schemas.common import (
    AnalysisWarning,
    ModelVersions,
    OutputModel,
    PercentScore,
    ProcessingTime,
    StrictModel,
    UnitScore,
)
from hamqadam_ai.schemas.detection import FaceDetectionResult
from hamqadam_ai.schemas.duplicate import DuplicateCheckResult
from hamqadam_ai.schemas.fraud import FraudRiskResult
from hamqadam_ai.schemas.matching import MatchingResult
from hamqadam_ai.schemas.ocr import CnicOcrResult
from hamqadam_ai.schemas.profile import ProfileAnalysisResult
from hamqadam_ai.schemas.quality import QualityResult


class VerificationRequest(StrictModel):
    """One verification, as the Backend submits it.

    Images arrive as base64 or as multipart uploads depending on the endpoint;
    this model carries whatever the transport decoded, plus the identifiers the
    service needs to correlate and to search.
    """

    verification_id: str = Field(
        description="The Backend's identifier for this attempt, echoed back."
    )
    user_reference: str | None = Field(
        default=None,
        description=(
            "The Backend's opaque identifier for the account. Used only to "
            "exclude the user's own template from the duplicate search - "
            "**pass it whenever it is known**, because without it a returning "
            "user matches themselves at similarity 1.0 and is reported as a "
            "duplicate of themselves."
        ),
    )
    enrol_on_success: bool | None = Field(
        default=None,
        description=(
            "Per-request override for gallery enrolment. **Leave unset** and "
            "the configured `duplicate.enrol_policy` decides, which is the "
            "normal case: one call to this endpoint verifies, checks the "
            "gallery and enrols, with no second request needed. Pass `false` "
            "to suppress enrolment for one submission, or `true` to force it "
            "where policy is `never`."
        ),
    )


class StageStatus(OutputModel):
    """Whether one pipeline stage ran, and what it cost."""

    stage: str = Field(description="Stage name.")
    ran: bool = Field(description="Whether it executed at all.")
    succeeded: bool = Field(
        default=False,
        description=(
            "Whether it produced a usable result. A stage can run and fail - "
            "'no face detected' is a completed stage with a negative finding."
        ),
    )
    duration_ms: float = Field(default=0.0, description="Wall-clock time.")
    error: str | None = Field(
        default=None, description="Why it could not run or could not finish."
    )


class VerificationResult(OutputModel):
    """MODULE 10 output - the complete verification.

    The four headline fields the Backend's rules engine reads are
    ``recommendation``, ``identity_confidence_score``, ``fraud_risk_score`` and
    ``fraud_risk_level``. Everything else exists so a decision can be explained
    to the person it was made about.
    """

    # -- Identity ---------------------------------------------------------- #
    verification_id: str = Field(description="Echoed from the request.")
    completed_at: dt.datetime = Field(description="When the analysis finished, UTC.")

    # -- The headline ------------------------------------------------------- #
    recommendation: Recommendation = Field(
        description=(
            "APPROVE, REJECT or MANUAL_REVIEW. **Advice, not an outcome** - "
            "the Backend's rules engine decides, using context this service "
            "does not have."
        )
    )
    recommendation_reasons: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Every condition the recommendation was based on.",
    )
    automated: bool = Field(
        default=False,
        description=(
            "Whether a rule fired, or the request simply fell through to the "
            "default. A reviewer seeing MANUAL_REVIEW needs to know which."
        ),
    )

    identity_confidence_score: PercentScore | None = Field(
        default=None,
        description=(
            "Module 4's fused confidence that every image is the same person. "
            "``null`` when no comparison was possible - which is different "
            "from zero."
        ),
    )
    fraud_risk_score: PercentScore = Field(
        default=0.0, description="Module 9's aggregated risk, 0-100."
    )
    fraud_risk_level: RiskLevel = Field(
        default=RiskLevel.LOW, description="LOW, MEDIUM or HIGH."
    )

    # -- Per-stage results -------------------------------------------------- #
    selfie_detection: FaceDetectionResult | None = Field(default=None)
    selfie_quality: QualityResult | None = Field(default=None)
    profile_analysis: ProfileAnalysisResult | None = Field(default=None)
    secondary_analyses: list[ProfileAnalysisResult] = Field(default_factory=list)
    cnic_ocr: CnicOcrResult | None = Field(default=None)
    cnic_portrait: CnicPortraitResult | None = Field(
        default=None,
        description=(
            "What Module 6 found on the card. The selfie-to-portrait "
            "*comparison* is not here - it lives in ``matching.comparisons`` "
            "with every other comparison, because reporting it twice would "
            "invite two answers to one question."
        ),
    )
    matching: MatchingResult | None = Field(default=None)
    duplicate: DuplicateCheckResult | None = Field(default=None)
    fraud: FraudRiskResult | None = Field(default=None)

    # -- Completeness -------------------------------------------------------- #
    stages: list[StageStatus] = Field(
        default_factory=list, description="What ran, what did not, and why."
    )
    complete: bool = Field(
        default=True,
        description=(
            "Whether every intended stage ran. A partial verification still "
            "produces a recommendation - it produces a more cautious one."
        ),
    )
    assessment_confidence: UnitScore = Field(
        default=1.0,
        description=(
            "Share of the intended evidence that was available. Not a "
            "confidence that the recommendation is right."
        ),
    )

    warnings: list[AnalysisWarning] = Field(
        default_factory=list, description="Non-fatal findings from every stage."
    )
    model_versions: ModelVersions | None = Field(
        default=None,
        description=(
            "Every model that contributed, so a disputed verification can be "
            "reproduced months later against the exact artefacts."
        ),
    )
    processing_time: ProcessingTime | None = Field(
        default=None, description="Wall-clock breakdown."
    )

    thresholds_validated: bool = Field(
        default=False,
        description=(
            "Whether the operating points behind this recommendation were "
            "derived from labelled data. Always false: Modules 4, 8 and 9 all "
            "report their thresholds as unvalidated, because the datasets that "
            "would calibrate them are not in this project."
        ),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def requires_human_review(self) -> bool:
        """Whether a person needs to look at this.

        Exposed rather than left to the caller to derive, because
        ``recommendation != APPROVE`` is not the same thing: a REJECT that was
        automated needs no reviewer, and a MANUAL_REVIEW always does.
        """
        return self.recommendation is Recommendation.MANUAL_REVIEW

    def summary(self) -> dict[str, Any]:
        """PII-free summary for logging."""
        return {
            "verification_id": self.verification_id,
            "recommendation": str(self.recommendation),
            "automated": self.automated,
            "identity_confidence": self.identity_confidence_score,
            "fraud_risk": self.fraud_risk_score,
            "fraud_level": str(self.fraud_risk_level),
            "complete": self.complete,
            "assessment_confidence": round(self.assessment_confidence, 3),
            "stages_failed": [s.stage for s in self.stages if s.ran and not s.succeeded],
            "stages_skipped": [s.stage for s in self.stages if not s.ran],
        }


__all__ = ["StageStatus", "VerificationRequest", "VerificationResult"]
