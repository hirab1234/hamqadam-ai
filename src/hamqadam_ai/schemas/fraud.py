"""MODULE 9 response contract - fraud risk.

Carries ``fraud_risk_score`` and ``fraud_risk_level``, and — more usefully than
either — the reasons behind them. A number a reviewer cannot decompose is a
number they have to either accept or ignore, and both are bad outcomes for
somebody being accused of dishonesty.

Reading the score
-----------------
``weights_validated`` is ``false``. The weights are a judgement about how much
the business should care about each finding, not probabilities derived from
labelled fraud, because no such data exists in this project. The *structure* -
which findings are independent evidence and which are the same fact seen twice
- is the part that has been reasoned about and measured.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from hamqadam_ai.core.constants import RiskLevel
from hamqadam_ai.schemas.common import OutputModel, PercentScore, UnitScore


class FraudSignalModel(OutputModel):
    """One piece of evidence, with what it weighed and why."""

    code: str = Field(description="Stable identifier from the producing module.")
    family: str = Field(
        description=(
            "Which kind of evidence this is. Findings sharing a family are "
            "treated as one fact seen several ways and do not compound."
        )
    )
    weight: UnitScore = Field(description="Evidence strength when certain.")
    confidence: UnitScore = Field(
        description="How sure the producing module was that it fired."
    )
    contribution: UnitScore = Field(description="weight x confidence.")
    decisive: bool = Field(
        default=False,
        description="Whether this alone raises the band to HIGH.",
    )
    stage: str = Field(default="", description="Which module produced it.")
    message: str = Field(description="What it means, for a reviewer.")
    detail: dict[str, Any] = Field(
        default_factory=dict, description="PII-free context."
    )


class FamilyContributionModel(OutputModel):
    """What one kind of evidence contributed to the score."""

    family: str = Field(description="The evidence family.")
    contribution: UnitScore = Field(description="Its strength after capping.")
    raw_contribution: UnitScore = Field(description="Before capping.")
    capped: bool = Field(description="Whether a configured ceiling reduced it.")
    driver: str = Field(description="Code of the strongest finding in the family.")
    signal_count: int = Field(
        ge=0,
        description=(
            "Findings in this family. More than one means the extras were "
            "treated as the same fact seen again, not as additional evidence."
        ),
    )


class FraudRiskResult(OutputModel):
    """MODULE 9 output for one verification request."""

    # -- Specification-mandated fields ---------------------------------- #
    fraud_risk_score: PercentScore = Field(
        description=(
            "0-100, higher meaning more suspicious. Combined as the strongest "
            "finding within each evidence family, then noisy-OR across "
            "families - so one fact reported four ways counts once."
        )
    )
    fraud_risk_level: RiskLevel = Field(description="LOW, MEDIUM or HIGH.")

    # -- Why -------------------------------------------------------------- #
    top_factors: list[str] = Field(
        default_factory=list,
        description="Finding codes driving the score, strongest first.",
    )
    families: list[FamilyContributionModel] = Field(
        default_factory=list,
        description="Per-family contributions, strongest first.",
    )
    signals: list[FraudSignalModel] = Field(
        default_factory=list, description="Every scored finding."
    )

    # -- How complete the evidence was ------------------------------------ #
    assessment_confidence: UnitScore = Field(
        default=1.0,
        description=(
            "Share of the intended checks that actually ran. **Not** a "
            "confidence that the score is correct - a measure of how much of "
            "the evidence was available to compute it."
        ),
    )
    unavailable_checks: list[str] = Field(
        default_factory=list,
        description=(
            "Checks that could not run. These contribute nothing to the "
            "score: an absent check is not evidence of innocence, and "
            "treating it as such is how a fraud engine gets quietly disabled "
            "by an outage."
        ),
    )
    unrecognised_findings: list[str] = Field(
        default_factory=list,
        description=(
            "Finding codes with no entry in the catalogue. A gap in this "
            "engine rather than a property of the request, surfaced rather "
            "than swallowed - an unscored finding is a hole no test would "
            "otherwise notice."
        ),
    )

    floored_by: str | None = Field(
        default=None,
        description=(
            "A decisive finding that raised the band regardless of the "
            "arithmetic. Null in the shipped configuration; nothing is marked "
            "decisive by default."
        ),
    )

    weights_validated: bool = Field(
        default=False,
        description=(
            "Whether the weights were derived from labelled fraud data. "
            "Always false. They are policy - a judgement about how much to "
            "care about each finding - not calibration."
        ),
    )

    duration_ms: float = Field(default=0.0, description="Wall-clock time.")

    def summary(self) -> dict[str, Any]:
        """PII-free summary for logging."""
        return {
            "score": self.fraud_risk_score,
            "level": str(self.fraud_risk_level),
            "factors": self.top_factors[:5],
            "signals": len(self.signals),
            "unavailable": self.unavailable_checks,
            "unrecognised": self.unrecognised_findings,
            "confidence": round(self.assessment_confidence, 3),
            "duration_ms": round(self.duration_ms, 1),
        }


__all__ = ["FamilyContributionModel", "FraudRiskResult", "FraudSignalModel"]
