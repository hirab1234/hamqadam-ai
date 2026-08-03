"""MODULE 9 service - what does everything the other modules found add up to?

The only module with no model, no images and no I/O. It takes the results of
Modules 1 through 8 and produces one number, one band, and the reasons for
both.

Why it is a separate module rather than a field on the pipeline
---------------------------------------------------------------
Because the aggregation is where the interesting mistakes live. Every upstream
module reports what it saw; none of them is in a position to know that four of
those reports are the same fact, or that a missing check is neither good news
nor bad. Putting that reasoning in one place, with the arithmetic written down,
is the difference between a risk score somebody can defend and a number that
came out of a spreadsheet.

What it deliberately does not do
--------------------------------
It does not decide. ``fraud_risk_level`` is evidence for the Backend's rules
engine, which owns the accept/reject decision and knows things this service
never will - a manual allow-list, a regulatory hold, an account's history.
Module 10 turns this into a *recommendation*; the Backend turns that into an
outcome.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from hamqadam_ai.core.config import Settings, get_settings
from hamqadam_ai.fraud_detection.aggregator import RiskAssessment, aggregate
from hamqadam_ai.fraud_detection.collector import SignalCollector
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.schemas.fraud import (
    FamilyContributionModel,
    FraudRiskResult,
    FraudSignalModel,
)

log = get_logger(__name__)


class FraudRiskService:
    """Aggregates every module's findings into one risk assessment.

    Args:
        settings: Service configuration.
    """

    __slots__ = ("_config", "_settings")

    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings
        self._config = settings.fraud

    # -- Public surface ------------------------------------------------------ #

    def assess(
        self,
        *,
        detection: Any = None,
        quality: Any = None,
        matching: Any = None,
        ocr: Any = None,
        cnic_face: Any = None,
        profile: Any = None,
        secondary_profiles: list[Any] | None = None,
        duplicate: Any = None,
    ) -> FraudRiskResult:
        """Combine whatever module results the caller has.

        Every argument is optional and ``None`` means "this check did not
        run", which is recorded as unavailable rather than treated as a pass.
        A partial verification still gets a score; it gets a lower
        ``assessment_confidence`` with it.

        Args:
            detection: Module 1 result for the live selfie.
            quality: Module 2 result for the live selfie.
            matching: Module 4 result.
            ocr: Module 5 result.
            cnic_face: Module 6 result.
            profile: Module 7 result for the profile image.
            secondary_profiles: Module 7 results for the additional images.
            duplicate: Module 8 result.

        Returns:
            The assessment. Never raises: a fraud score is the last thing that
            should take a verification down.
        """
        started = time.perf_counter()

        collector = SignalCollector(self._config.signal_weights)

        collector.collect_warnings(detection, stage="detection")
        collector.collect_warnings(quality, stage="quality")
        collector.collect_warnings(ocr, stage="ocr")
        collector.collect_warnings(profile, stage="profile")
        for index, secondary in enumerate(secondary_profiles or []):
            collector.collect_warnings(secondary, stage=f"secondary[{index}]")

        collector.collect_warnings(cnic_face, stage="cnic_face")
        collector.collect_cnic_face(cnic_face)
        collector.collect_matching(matching)
        collector.collect_duplicate(duplicate)

        assessment = aggregate(
            collector.signals,
            low_max=self._config.levels.low_max,
            medium_max=self._config.levels.medium_max,
            family_caps=dict(self._config.family_caps),
            unavailable=collector.unavailable,
            unrecognised=collector.unrecognised,
            expected_checks=self._config.expected_checks,
        )

        result = self._to_schema(
            assessment, duration_ms=(time.perf_counter() - started) * 1000.0
        )

        if result.unrecognised_findings:
            # Loud on purpose. An unscored finding is a hole in this engine,
            # and it will not show up as a failure anywhere else.
            log.warning(
                "fraud.unrecognised_findings",
                codes=result.unrecognised_findings,
                note="these findings contributed nothing to the risk score",
            )

        log.info("fraud.completed", **result.summary())
        return result

    def assess_from_signals(self, collector: SignalCollector) -> FraudRiskResult:
        """Score a collector the caller populated itself.

        For Module 10, which walks the pipeline and knows which stages ran, and
        for tests that want to drive the aggregation directly.
        """
        started = time.perf_counter()
        assessment = aggregate(
            collector.signals,
            low_max=self._config.levels.low_max,
            medium_max=self._config.levels.medium_max,
            family_caps=dict(self._config.family_caps),
            unavailable=collector.unavailable,
            unrecognised=collector.unrecognised,
            expected_checks=self._config.expected_checks,
        )
        return self._to_schema(
            assessment, duration_ms=(time.perf_counter() - started) * 1000.0
        )

    async def assess_async(self, **results: Any) -> FraudRiskResult:
        """Aggregate without blocking the event loop.

        Pure arithmetic over a handful of findings, so this is offered for
        interface symmetry rather than because it is slow.
        """
        return await asyncio.to_thread(lambda: self.assess(**results))

    def describe(self) -> dict[str, Any]:
        """Serialisable summary of the active configuration, for ``/health``."""
        from hamqadam_ai.fraud_detection.signals import BENIGN_CODES, CATALOGUE

        return {
            "levels": {
                "low_max": self._config.levels.low_max,
                "medium_max": self._config.levels.medium_max,
            },
            "family_caps": dict(self._config.family_caps),
            "catalogue_size": len(CATALOGUE),
            "benign_codes": len(BENIGN_CODES),
            "overrides": sorted(self._config.signal_weights),
            "combination": "strongest-within-family, then noisy-OR across",
            "weights_validated": False,
        }

    # -- Internals ----------------------------------------------------------- #

    def _to_schema(
        self, assessment: RiskAssessment, *, duration_ms: float
    ) -> FraudRiskResult:
        """Render the assessment as its response model."""
        return FraudRiskResult(
            fraud_risk_score=assessment.score,
            fraud_risk_level=assessment.level,
            top_factors=assessment.top_factors,
            families=[
                FamilyContributionModel(**family.as_dict())
                for family in assessment.families
            ],
            signals=[
                FraudSignalModel(**signal.as_dict()) for signal in assessment.signals
            ],
            assessment_confidence=assessment.assessment_confidence,
            unavailable_checks=assessment.unavailable,
            unrecognised_findings=assessment.unrecognised,
            floored_by=assessment.floored_by,
            weights_validated=False,
            duration_ms=duration_ms,
        )


def build_fraud_service(settings: Settings | None = None) -> FraudRiskService:
    """Wire up a :class:`FraudRiskService` from configuration.

    Args:
        settings: Service configuration. Loaded from the environment when
            omitted.

    Returns:
        A ready service. No models to load, so this cannot fail for want of
        weights.
    """
    settings = settings or get_settings()
    log.info(
        "fraud.service_ready",
        low_max=settings.fraud.levels.low_max,
        medium_max=settings.fraud.levels.medium_max,
        caps=dict(settings.fraud.family_caps),
    )
    return FraudRiskService(settings=settings)


__all__ = ["FraudRiskService", "build_fraud_service"]
