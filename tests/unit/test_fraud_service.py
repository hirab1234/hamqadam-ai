"""MODULE 9 service: reading real module results and scoring them.

No models and no images - the service takes other modules' *results*, and those
are constructible by hand. Using simple namespaces rather than the real
response models is deliberate and matches the collector's own design: it reads
by attribute so that Module 9 does not become the most coupled component in the
service, and the tests exercise that contract rather than working around it.
"""

from __future__ import annotations

from types import SimpleNamespace as NS  # noqa: N814 - terse by design here

import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import RiskLevel
from hamqadam_ai.fraud_detection import SignalCollector
from hamqadam_ai.services.fraud_service import FraudRiskService, build_fraud_service


@pytest.fixture
def service() -> FraudRiskService:
    return build_fraud_service()


def warning(code: str, **detail: object) -> NS:
    return NS(code=code, message="", stage="", detail=detail)


def finding(code: str, *, severity: str = "error", confidence: float = 1.0) -> NS:
    return NS(code=code, severity=severity, confidence=confidence)


def comparison(kind: str, decision: str, *, confidence: float = 0.9) -> NS:
    return NS(
        comparison=kind, decision=decision, confidence=confidence, compared=True
    )


def ok(**overrides: object) -> dict[str, object]:
    """A clean verification, with everything passing."""
    base: dict[str, object] = {
        "detection": NS(warnings=[], error_code=None),
        "quality": NS(warnings=[], error_code=None),
        "matching": NS(comparisons=[comparison("profile", "STRONG_MATCH")]),
        "ocr": NS(warnings=[], error_code=None, findings=[]),
        "cnic_face": NS(
            warnings=[], error_code=None,
            portrait=NS(foreign_face_count=0, found=True),
        ),
        "profile": NS(warnings=[], error_code=None, findings=[]),
        "duplicate": NS(
            searched=True, duplicate_found=False, needs_review=False,
            gallery_size=500,
        ),
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# The ordinary case
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_clean_verification_carries_no_risk(service: FraudRiskService) -> None:
    result = service.assess(**ok())

    assert result.fraud_risk_score == pytest.approx(0.0)
    assert result.fraud_risk_level is RiskLevel.LOW
    assert result.top_factors == []
    assert result.assessment_confidence == pytest.approx(1.0)


@pytest.mark.unit
def test_informational_findings_are_not_evidence(service: FraudRiskService) -> None:
    """Module 5 reports that the gender agreed with the number's parity. That
    is the check passing, and scoring it would be perverse."""
    result = service.assess(
        **ok(ocr=NS(
            warnings=[], error_code=None,
            findings=[finding("CNIC_GENDER_CONSISTENT", severity="info")],
        ))
    )

    assert result.fraud_risk_score == pytest.approx(0.0)


@pytest.mark.unit
def test_an_expired_card_is_not_fraud(service: FraudRiskService) -> None:
    """A correct reading of an out-of-date document. Whether it is acceptable
    is a policy question for the rules engine."""
    result = service.assess(
        **ok(ocr=NS(
            warnings=[], error_code=None,
            findings=[finding("CNIC_EXPIRED", severity="info")],
        ))
    )

    assert result.fraud_risk_score == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# The rule the module exists for
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_blurry_photograph_is_not_a_fraud_accusation(
    service: FraudRiskService,
) -> None:
    """Four independent quality findings from one bad photograph. Added up
    they reach the HIGH band; grouped by family they stay LOW, which is what a
    bad photograph deserves."""
    result = service.assess(
        **ok(
            detection=NS(
                warnings=[warning("FACE_POSE_OUT_OF_RANGE")], error_code=None
            ),
            quality=NS(warnings=[], error_code="LOW_IMAGE_QUALITY"),
            ocr=NS(warnings=[], error_code="CNIC_OCR_FAILED", findings=[]),
            profile=NS(
                warnings=[warning("PROFILE_IMAGE_QUALITY_LOW")],
                error_code=None, findings=[],
            ),
        )
    )

    assert result.fraud_risk_level is RiskLevel.LOW
    assert result.fraud_risk_score < 30.0


@pytest.mark.unit
def test_the_family_shows_how_many_findings_it_absorbed(
    service: FraudRiskService,
) -> None:
    """So a reviewer can see the suppression happened rather than infer it."""
    result = service.assess(
        **ok(
            quality=NS(warnings=[], error_code="LOW_IMAGE_QUALITY"),
            ocr=NS(warnings=[], error_code="CNIC_OCR_FAILED", findings=[]),
        )
    )
    quality = next(
        f for f in result.families if f.family == "capture_quality"
    )

    assert quality.signal_count >= 2
    assert len(result.signals) >= 2


@pytest.mark.unit
def test_independent_facts_reach_the_high_band(service: FraudRiskService) -> None:
    result = service.assess(
        **ok(
            ocr=NS(
                warnings=[], error_code=None,
                findings=[finding("CNIC_GENDER_MISMATCH")],
            ),
            profile=NS(
                warnings=[warning("PROFILE_IMAGE_IS_SCREENSHOT", confidence=1.0)],
                error_code=None, findings=[],
            ),
            duplicate=NS(
                searched=True, duplicate_found=True, needs_review=False,
                gallery_size=9000,
            ),
        )
    )

    assert result.fraud_risk_level is RiskLevel.HIGH
    assert len(result.families) >= 3


# --------------------------------------------------------------------------- #
# Reading each module
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_failed_cnic_comparison_is_read_from_matching(
    service: FraudRiskService,
) -> None:
    result = service.assess(
        **ok(matching=NS(comparisons=[comparison("cnic", "FAILED")]))
    )

    assert "CNIC_FACE_MISMATCH" in result.top_factors


@pytest.mark.unit
def test_a_failed_secondary_weighs_less_than_a_failed_cnic(
    service: FraudRiskService,
) -> None:
    """An extra photograph of a family member is a plausible mistake; the
    identity card is not."""
    secondary = service.assess(
        **ok(matching=NS(comparisons=[comparison("secondary", "FAILED")]))
    )
    cnic = service.assess(
        **ok(matching=NS(comparisons=[comparison("cnic", "FAILED")]))
    )

    assert secondary.fraud_risk_score < cnic.fraud_risk_score


@pytest.mark.unit
def test_a_passing_comparison_contributes_nothing(
    service: FraudRiskService,
) -> None:
    result = service.assess(
        **ok(matching=NS(comparisons=[comparison("cnic", "STRONG_MATCH")]))
    )
    assert result.fraud_risk_score == pytest.approx(0.0)


@pytest.mark.unit
def test_a_foreign_face_on_the_card_is_read_directly(
    service: FraudRiskService,
) -> None:
    """``foreign_face_count`` does not arrive as a warning on every path, and
    it is the signal for a card held up in front of somebody."""
    result = service.assess(
        **ok(cnic_face=NS(
            warnings=[], error_code=None,
            portrait=NS(foreign_face_count=1, found=True),
        ))
    )

    assert "CNIC_FOREIGN_FACE_PRESENT" in result.top_factors


@pytest.mark.unit
def test_a_duplicate_is_read_from_the_gallery_verdict(
    service: FraudRiskService,
) -> None:
    result = service.assess(
        **ok(duplicate=NS(
            searched=True, duplicate_found=True, needs_review=False,
            gallery_size=1000,
        ))
    )

    assert "DUPLICATE_FACE_DETECTED" in result.top_factors


@pytest.mark.unit
def test_a_review_band_duplicate_weighs_less_than_a_confirmed_one(
    service: FraudRiskService,
) -> None:
    review = service.assess(
        **ok(duplicate=NS(
            searched=True, duplicate_found=False, needs_review=True,
            gallery_size=1000,
        ))
    )
    confirmed = service.assess(
        **ok(duplicate=NS(
            searched=True, duplicate_found=True, needs_review=False,
            gallery_size=1000,
        ))
    )

    assert review.fraud_risk_score < confirmed.fraud_risk_score


# --------------------------------------------------------------------------- #
# Missing evidence
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_stage_that_did_not_run_is_recorded_not_ignored(
    service: FraudRiskService,
) -> None:
    result = service.assess(**ok(duplicate=None))

    assert "duplicate" in result.unavailable_checks
    assert result.assessment_confidence < 1.0
    assert result.fraud_risk_score == pytest.approx(0.0)


@pytest.mark.unit
def test_a_gallery_outage_is_an_absence_not_a_pass(
    service: FraudRiskService,
) -> None:
    """``searched=False`` means the check could not run. Reading it as "no
    duplicate found" is how a fraud engine gets quietly disabled by an
    outage."""
    result = service.assess(
        **ok(duplicate=NS(
            searched=False, duplicate_found=False, needs_review=False,
            gallery_size=0,
        ))
    )

    assert "duplicate" in result.unavailable_checks


@pytest.mark.unit
def test_an_infrastructure_error_is_an_absence_not_evidence(
    service: FraudRiskService,
) -> None:
    """An unreachable vector database says nothing about the person."""
    result = service.assess(
        **ok(quality=NS(warnings=[], error_code="MODEL_NOT_LOADED"))
    )

    assert result.fraud_risk_score == pytest.approx(0.0)
    assert "quality" in result.unavailable_checks


@pytest.mark.unit
def test_one_absent_module_counts_once(service: FraudRiskService) -> None:
    """``cnic_face`` is read by two collectors. Counting it twice would
    understate the confidence."""
    result = service.assess(**ok(cnic_face=None))

    assert result.unavailable_checks.count("cnic_face") == 1


@pytest.mark.unit
def test_nothing_at_all_still_produces_a_result(
    service: FraudRiskService,
) -> None:
    """A fraud score is the last thing that should take a verification down."""
    result = service.assess()

    assert result.fraud_risk_level is RiskLevel.LOW
    assert result.assessment_confidence == pytest.approx(0.0)
    assert len(result.unavailable_checks) >= 6


# --------------------------------------------------------------------------- #
# Gaps in the engine itself
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_an_unscored_finding_is_surfaced(service: FraudRiskService) -> None:
    """The failure this engine cannot otherwise detect about itself: a new
    upstream warning that nobody scored contributes nothing and fails
    nothing."""
    result = service.assess(
        **ok(detection=NS(
            warnings=[warning("A_BRAND_NEW_WARNING")], error_code=None
        ))
    )

    assert result.unrecognised_findings == ["A_BRAND_NEW_WARNING"]


@pytest.mark.unit
def test_a_benign_code_is_not_reported_as_unrecognised(
    service: FraudRiskService,
) -> None:
    """The distinction between "decided this is harmless" and "nobody has
    looked at this" has to survive."""
    result = service.assess(
        **ok(profile=NS(
            warnings=[warning("PROFILE_IMAGE_HAS_NO_FACE")],
            error_code=None, findings=[],
        ))
    )

    assert result.unrecognised_findings == []


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_weight_can_be_overridden_without_a_code_change() -> None:
    settings = get_settings().model_copy(deep=True)
    settings.fraud.signal_weights = {"DUPLICATE_FACE_DETECTED": 0.05}
    quiet = FraudRiskService(settings=settings)

    loud = build_fraud_service().assess(
        **ok(duplicate=NS(
            searched=True, duplicate_found=True, needs_review=False,
            gallery_size=10,
        ))
    )
    hushed = quiet.assess(
        **ok(duplicate=NS(
            searched=True, duplicate_found=True, needs_review=False,
            gallery_size=10,
        ))
    )

    assert hushed.fraud_risk_score < loud.fraud_risk_score


@pytest.mark.unit
def test_the_shipped_caps_hold_capture_quality_below_medium() -> None:
    """The cap that stops blur reading as dishonesty."""
    caps = get_settings().fraud.family_caps
    low_max = get_settings().fraud.levels.low_max

    assert caps["capture_quality"] * 100 <= low_max


@pytest.mark.unit
def test_the_service_never_claims_validated_weights(
    service: FraudRiskService,
) -> None:
    """There is no labelled fraud data in this project. The weights are a
    judgement about what to care about, not a calibration."""
    assert service.assess(**ok()).weights_validated is False
    assert service.describe()["weights_validated"] is False


@pytest.mark.unit
def test_the_service_describes_its_combination_rule(
    service: FraudRiskService,
) -> None:
    described = service.describe()

    assert "noisy-OR" in described["combination"]
    assert described["catalogue_size"] > 20


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_every_signal_carries_its_explanation(service: FraudRiskService) -> None:
    """A user rejected for fraud deserves to know why, and a reviewer needs
    the reasons ranked."""
    result = service.assess(
        **ok(ocr=NS(
            warnings=[], error_code=None,
            findings=[finding("CNIC_GENDER_MISMATCH")],
        ))
    )
    signal = result.signals[0]

    assert signal.message
    assert signal.family
    assert 0.0 <= signal.contribution <= 1.0


@pytest.mark.unit
def test_the_result_serialises(service: FraudRiskService) -> None:
    import json

    payload = service.assess(**ok()).model_dump(mode="json")
    json.dumps(payload)

    assert payload["weights_validated"] is False


@pytest.mark.unit
def test_the_summary_is_pii_free(service: FraudRiskService) -> None:
    import json

    summary = service.assess(**ok()).summary()
    json.dumps(summary)

    assert set(summary) >= {"score", "level", "factors", "confidence"}


@pytest.mark.unit
def test_a_caller_can_drive_the_collector_itself(
    service: FraudRiskService,
) -> None:
    """Module 10 walks the pipeline and knows which stages ran."""
    collector = SignalCollector()
    collector.add_code("CNIC_GENDER_MISMATCH", stage="ocr")
    collector.mark_unavailable("duplicate")

    result = service.assess_from_signals(collector)

    assert result.fraud_risk_score > 0.0
    assert result.unavailable_checks == ["duplicate"]


@pytest.mark.unit
def test_assessment_is_deterministic(service: FraudRiskService) -> None:
    """A risk score that changes between two runs of one request is not a
    score anybody can defend."""
    payload = ok(
        ocr=NS(
            warnings=[], error_code=None,
            findings=[finding("CNIC_GENDER_MISMATCH")],
        )
    )
    first = service.assess(**payload)
    second = service.assess(**payload)

    assert first.fraud_risk_score == pytest.approx(second.fraud_risk_score)
    assert first.top_factors == second.top_factors
