"""Comparison, identity fusion and the MODULE 4 service surface.

Needs no model weights. The embeddings here are synthetic vectors constructed
to sit at chosen cosine similarities, which is what lets the fusion rules be
pinned exactly rather than sampled from whatever a real model happens to
produce.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import MatchDecision
from hamqadam_ai.embeddings.base import FaceEmbedding, l2_normalise
from hamqadam_ai.matching.aggregator import IdentityAggregator
from hamqadam_ai.matching.comparator import ComparisonType, FaceComparator, MatchOutcome
from hamqadam_ai.schemas.matching import MatchingResult
from hamqadam_ai.services.matching_service import build_matching_service

DIMENSION = 512
VERSION = "arcface-test-v1"


def embedding_at(
    similarity: float,
    *,
    reference: FaceEmbedding | None = None,
    confidence: float = 0.95,
    aligned: bool = True,
    version: str = VERSION,
) -> FaceEmbedding:
    """Build an embedding sitting at a chosen cosine from a reference.

    Constructs ``cos(t)*u + sin(t)*v`` for orthonormal ``u`` and ``v``, so the
    resulting similarity is exact rather than approximate.
    """
    if reference is None:
        vector = np.zeros(DIMENSION, dtype=np.float32)
        vector[0] = 1.0
    else:
        base = np.asarray(reference.vector, dtype=np.float64)
        # An orthonormal companion to the reference direction.
        seed = np.zeros(DIMENSION, dtype=np.float64)
        seed[1] = 1.0
        companion = seed - np.dot(seed, base) * base
        norm = np.linalg.norm(companion)
        if norm < 1e-9:
            seed = np.zeros(DIMENSION, dtype=np.float64)
            seed[2] = 1.0
            companion = seed - np.dot(seed, base) * base
            norm = np.linalg.norm(companion)
        companion = companion / norm

        clamped = max(-1.0, min(1.0, similarity))
        angle = math.acos(clamped)
        vector = (math.cos(angle) * base + math.sin(angle) * companion).astype(
            np.float32
        )

    return FaceEmbedding(
        vector=l2_normalise(vector),
        raw_norm=22.5,
        confidence=confidence,
        model_key="face_embedder_arcface",
        model_version=version,
        aligned=aligned,
        alignment_residual=0.04 if aligned else None,
    )


@pytest.fixture
def reference() -> FaceEmbedding:
    """The live-selfie reference."""
    return embedding_at(1.0)


@pytest.fixture
def comparator() -> FaceComparator:
    """A comparator on the real configuration."""
    return FaceComparator(get_settings().matching)


@pytest.fixture
def aggregator() -> IdentityAggregator:
    """An aggregator on the real configuration."""
    return IdentityAggregator(get_settings().matching)


@pytest.fixture
def service():  # noqa: ANN201 - pytest fixture
    """The matching service. Needs no weights, so it always builds."""
    return build_matching_service()


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_constructed_similarity_is_exact(reference: FaceEmbedding) -> None:
    """Guards the test helper itself: if this drifts, everything below is
    measuring the wrong thing."""
    for target in (0.0, 0.3, 0.62, 0.9):
        candidate = embedding_at(target, reference=reference)
        assert reference.similarity_to(candidate) == pytest.approx(target, abs=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("similarity", "expected"),
    [
        (0.95, MatchDecision.STRONG_MATCH),
        (0.62, MatchDecision.STRONG_MATCH),
        (0.50, MatchDecision.REVIEW),
        (0.20, MatchDecision.FAILED),
    ],
)
def test_profile_decisions(
    comparator: FaceComparator,
    reference: FaceEmbedding,
    similarity: float,
    expected: MatchDecision,
) -> None:
    outcome = comparator.compare(
        reference,
        embedding_at(similarity, reference=reference),
        ComparisonType.PROFILE,
    )
    assert outcome.compared is True
    assert outcome.decision is expected


@pytest.mark.unit
def test_the_cnic_operating_point_is_laxer(
    comparator: FaceComparator, reference: FaceEmbedding
) -> None:
    """A similarity of 0.50 fails as a profile comparison and passes as a CNIC
    one. That is the point of having separate thresholds."""
    candidate = embedding_at(0.50, reference=reference)

    profile = comparator.compare(reference, candidate, ComparisonType.PROFILE)
    cnic = comparator.compare(reference, candidate, ComparisonType.CNIC)

    assert profile.decision is MatchDecision.REVIEW
    assert cnic.decision is MatchDecision.STRONG_MATCH


@pytest.mark.unit
def test_a_missing_reference_is_not_compared(comparator: FaceComparator) -> None:
    outcome = comparator.compare(None, embedding_at(1.0), ComparisonType.PROFILE)
    assert outcome.compared is False
    assert outcome.decision is MatchDecision.NOT_COMPARED
    assert outcome.score == 0.0
    assert outcome.reason is not None


@pytest.mark.unit
def test_a_missing_candidate_is_not_compared(
    comparator: FaceComparator, reference: FaceEmbedding
) -> None:
    outcome = comparator.compare(reference, None, ComparisonType.CNIC)
    assert outcome.compared is False
    assert outcome.decision is MatchDecision.NOT_COMPARED


@pytest.mark.unit
def test_not_compared_is_distinct_from_failed(
    comparator: FaceComparator, reference: FaceEmbedding
) -> None:
    """"There was no CNIC portrait" and "the CNIC portrait is a different
    person" are opposite findings, and collapsing them would let a missing
    document read as a passing one."""
    absent = comparator.compare(reference, None, ComparisonType.CNIC)
    mismatched = comparator.compare(
        reference, embedding_at(0.05, reference=reference), ComparisonType.CNIC
    )

    assert absent.decision is MatchDecision.NOT_COMPARED
    assert absent.failed is False
    assert mismatched.decision is MatchDecision.FAILED
    assert mismatched.failed is True


@pytest.mark.unit
def test_cross_version_comparison_is_refused(
    comparator: FaceComparator, reference: FaceEmbedding
) -> None:
    """Vectors from different networks occupy unrelated spaces. A cosine
    between them is a number, but it is not a similarity."""
    other = embedding_at(0.9, reference=reference, version="arcface-v2")
    outcome = comparator.compare(reference, other, ComparisonType.PROFILE)

    assert outcome.compared is False
    assert outcome.reason is not None
    assert "model versions" in outcome.reason


@pytest.mark.unit
def test_a_degenerate_embedding_is_refused(
    comparator: FaceComparator, reference: FaceEmbedding
) -> None:
    degenerate = FaceEmbedding(
        vector=np.zeros(DIMENSION, dtype=np.float32),
        raw_norm=0.0,
        confidence=0.0,
        model_key="face_embedder_arcface",
        model_version=VERSION,
    )
    outcome = comparator.compare(reference, degenerate, ComparisonType.PROFILE)
    assert outcome.compared is False


@pytest.mark.unit
def test_the_pair_inherits_the_weaker_confidence(
    comparator: FaceComparator,
) -> None:
    """A comparison is only as trustworthy as its worse side. An excellent
    selfie against a box-aligned profile photo is limited by the bad one."""
    strong = embedding_at(1.0, confidence=0.98)
    weak = embedding_at(0.9, reference=strong, confidence=0.30, aligned=False)

    outcome = comparator.compare(strong, weak, ComparisonType.PROFILE)

    assert outcome.confidence == pytest.approx(0.30)
    assert outcome.low_confidence is True


@pytest.mark.unit
def test_the_thresholds_are_reported_on_the_outcome(
    comparator: FaceComparator, reference: FaceEmbedding
) -> None:
    outcome = comparator.compare(
        reference, embedding_at(0.7, reference=reference), ComparisonType.CNIC
    )
    assert outcome.detail["strong_match_threshold"] == 0.42
    assert outcome.detail["review_threshold"] == 0.30


# --------------------------------------------------------------------------- #
# Identity fusion
# --------------------------------------------------------------------------- #


def outcome(
    comparison: ComparisonType,
    score: float,
    decision: MatchDecision,
    *,
    confidence: float = 0.95,
    low: bool = False,
) -> MatchOutcome:
    """A synthetic outcome, for testing the aggregator in isolation."""
    return MatchOutcome(
        comparison=comparison,
        compared=True,
        similarity=0.7,
        score=score,
        decision=decision,
        confidence=confidence,
        low_confidence=low,
    )


@pytest.mark.unit
def test_all_strong_matches_fuse_high(aggregator: IdentityAggregator) -> None:
    assessment = aggregator.aggregate(
        [
            outcome(ComparisonType.PROFILE, 95.0, MatchDecision.STRONG_MATCH),
            outcome(ComparisonType.CNIC, 90.0, MatchDecision.STRONG_MATCH),
            outcome(ComparisonType.SECONDARY, 92.0, MatchDecision.STRONG_MATCH),
        ]
    )
    assert assessment.available is True
    assert assessment.identity_confidence is not None
    assert assessment.identity_confidence > 88.0


@pytest.mark.unit
def test_the_cnic_carries_the_most_weight(aggregator: IdentityAggregator) -> None:
    """It is the only state-issued anchor in the request."""
    weights = get_settings().matching.identity.weights
    assert weights["cnic"] > weights["profile"] > weights["secondary"]


@pytest.mark.unit
def test_a_failed_cnic_caps_the_identity_confidence(
    aggregator: IdentityAggregator,
) -> None:
    """The rule that stops somebody uploading three selfies of themselves
    alongside a stranger's identity document and scoring highly on internal
    consistency."""
    assessment = aggregator.aggregate(
        [
            outcome(ComparisonType.PROFILE, 98.0, MatchDecision.STRONG_MATCH),
            outcome(ComparisonType.SECONDARY, 97.0, MatchDecision.STRONG_MATCH),
            outcome(ComparisonType.CNIC, 5.0, MatchDecision.FAILED),
        ]
    )
    cap = get_settings().matching.identity.cnic_failure_cap

    assert assessment.identity_confidence is not None
    assert assessment.identity_confidence <= cap
    assert assessment.capped_by == "cnic_failure"
    assert any("capped" in reason for reason in assessment.reasons)


@pytest.mark.unit
def test_the_cap_does_not_apply_when_the_cnic_merely_needs_review(
    aggregator: IdentityAggregator,
) -> None:
    """REVIEW is inconclusive, not contradictory."""
    assessment = aggregator.aggregate(
        [
            outcome(ComparisonType.PROFILE, 95.0, MatchDecision.STRONG_MATCH),
            outcome(ComparisonType.CNIC, 60.0, MatchDecision.REVIEW),
        ]
    )
    assert assessment.capped_by is None


@pytest.mark.unit
def test_the_cap_does_not_apply_when_there_is_no_cnic(
    aggregator: IdentityAggregator,
) -> None:
    assessment = aggregator.aggregate(
        [
            outcome(ComparisonType.PROFILE, 95.0, MatchDecision.STRONG_MATCH),
            MatchOutcome.not_compared(ComparisonType.CNIC, reason="no portrait"),
        ]
    )
    assert assessment.capped_by is None
    assert assessment.identity_confidence is not None
    assert assessment.identity_confidence > 90.0


@pytest.mark.unit
def test_weights_renormalise_around_missing_comparisons(
    aggregator: IdentityAggregator,
) -> None:
    """A request without secondary images must not be penalised for it."""
    assessment = aggregator.aggregate(
        [
            outcome(ComparisonType.PROFILE, 80.0, MatchDecision.STRONG_MATCH),
            outcome(ComparisonType.CNIC, 80.0, MatchDecision.STRONG_MATCH),
        ]
    )
    assert assessment.identity_confidence == pytest.approx(80.0, abs=0.1)
    assert sum(assessment.effective_weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert "secondary" not in assessment.effective_weights


@pytest.mark.unit
def test_secondary_images_are_averaged_not_maximised(
    aggregator: IdentityAggregator,
) -> None:
    """A mismatching secondary image is a fraud signal, not noise to discard.
    `max` would silently ignore it."""
    assert get_settings().matching.identity.secondary_aggregation == "mean"

    assessment = aggregator.aggregate(
        [
            outcome(ComparisonType.SECONDARY, 95.0, MatchDecision.STRONG_MATCH),
            outcome(ComparisonType.SECONDARY, 95.0, MatchDecision.STRONG_MATCH),
            outcome(ComparisonType.SECONDARY, 5.0, MatchDecision.FAILED),
        ]
    )
    assert assessment.secondary_aggregate == pytest.approx(65.0, abs=0.1)


@pytest.mark.unit
def test_the_worst_secondary_is_reported_separately(
    aggregator: IdentityAggregator,
) -> None:
    """So Module 9 can weigh it as a risk signal without it dominating the
    identity score."""
    assessment = aggregator.aggregate(
        [
            outcome(ComparisonType.SECONDARY, 95.0, MatchDecision.STRONG_MATCH),
            outcome(ComparisonType.SECONDARY, 12.0, MatchDecision.FAILED),
        ]
    )
    assert assessment.secondary_worst == pytest.approx(12.0)
    assert assessment.any_failed is True


@pytest.mark.unit
def test_a_low_confidence_comparison_is_down_weighted(
    aggregator: IdentityAggregator,
) -> None:
    strong = aggregator.aggregate(
        [
            outcome(ComparisonType.PROFILE, 100.0, MatchDecision.STRONG_MATCH),
            outcome(ComparisonType.CNIC, 20.0, MatchDecision.REVIEW),
        ]
    )
    degraded = aggregator.aggregate(
        [
            outcome(ComparisonType.PROFILE, 100.0, MatchDecision.STRONG_MATCH),
            outcome(
                ComparisonType.CNIC, 20.0, MatchDecision.REVIEW,
                confidence=0.2, low=True,
            ),
        ]
    )
    # Down-weighting the weak low-scoring CNIC lets the strong profile
    # dominate more, so the fused confidence rises.
    assert degraded.identity_confidence is not None
    assert strong.identity_confidence is not None
    assert degraded.identity_confidence > strong.identity_confidence


@pytest.mark.unit
def test_too_few_comparisons_reports_unavailable(
    aggregator: IdentityAggregator,
) -> None:
    """Reporting a number derived from nothing would be worse than reporting
    its absence."""
    assessment = aggregator.aggregate(
        [
            MatchOutcome.not_compared(ComparisonType.PROFILE, reason="no selfie"),
            MatchOutcome.not_compared(ComparisonType.CNIC, reason="no selfie"),
        ]
    )
    assert assessment.available is False
    assert assessment.identity_confidence is None
    assert assessment.reasons


@pytest.mark.unit
def test_the_headline_score_prefers_the_profile_comparison(
    aggregator: IdentityAggregator,
) -> None:
    assessment = aggregator.aggregate(
        [
            outcome(ComparisonType.PROFILE, 88.0, MatchDecision.STRONG_MATCH),
            outcome(ComparisonType.CNIC, 70.0, MatchDecision.STRONG_MATCH),
        ]
    )
    assert assessment.face_match_score == pytest.approx(88.0)


@pytest.mark.unit
def test_the_headline_falls_back_rather_than_reporting_zero(
    aggregator: IdentityAggregator,
) -> None:
    """Zero would be indistinguishable from a total mismatch."""
    assessment = aggregator.aggregate(
        [
            MatchOutcome.not_compared(ComparisonType.PROFILE, reason="not supplied"),
            outcome(ComparisonType.SECONDARY, 77.0, MatchDecision.STRONG_MATCH),
            outcome(ComparisonType.CNIC, 66.0, MatchDecision.STRONG_MATCH),
        ]
    )
    assert assessment.face_match_score == pytest.approx(77.0)


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_service_returns_every_specified_field(service, reference) -> None:  # noqa: ANN001
    result = service.match(
        selfie=reference,
        profile=embedding_at(0.90, reference=reference),
        secondaries=[
            embedding_at(0.85, reference=reference),
            embedding_at(0.80, reference=reference),
        ],
        cnic=embedding_at(0.65, reference=reference),
    )

    assert isinstance(result, MatchingResult)
    assert result.face_match_score > 0
    assert result.profile_face_match_score is not None
    assert len(result.secondary_face_match_scores) == 2
    assert result.cnic_face_match_score is not None
    assert result.identity_confidence_score is not None
    assert result.identity_available is True


@pytest.mark.unit
def test_the_result_is_json_serialisable(service, reference) -> None:  # noqa: ANN001
    result = service.match(
        selfie=reference, profile=embedding_at(0.9, reference=reference)
    )
    json.dumps(result.model_dump(mode="json"))


@pytest.mark.unit
def test_a_missing_selfie_disables_everything(service) -> None:  # noqa: ANN001
    """Reported, not raised: the pipeline still needs the quality and OCR
    findings for the same request."""
    result = service.match(
        selfie=None,
        profile=embedding_at(1.0),
        cnic=embedding_at(1.0),
    )
    assert result.identity_available is False
    assert result.identity_confidence_score is None
    assert all(not entry.compared for entry in result.comparisons)
    assert any(w.code == "MATCHING_NO_REFERENCE" for w in result.warnings)


@pytest.mark.unit
def test_secondary_labels_are_echoed_back(service, reference) -> None:  # noqa: ANN001
    result = service.match(
        selfie=reference,
        secondaries=[
            embedding_at(0.9, reference=reference),
            embedding_at(0.8, reference=reference),
        ],
        secondary_labels=["upload_a", "upload_b"],
    )
    labels = [
        entry.target_label
        for entry in result.comparisons
        if entry.comparison == "secondary"
    ]
    assert labels == ["upload_a", "upload_b"]


@pytest.mark.unit
def test_a_failed_secondary_is_not_hidden(service, reference) -> None:  # noqa: ANN001
    result = service.match(
        selfie=reference,
        profile=embedding_at(0.95, reference=reference),
        secondaries=[
            embedding_at(0.92, reference=reference),
            embedding_at(0.05, reference=reference),
        ],
        cnic=embedding_at(0.7, reference=reference),
    )
    assert result.any_comparison_failed is True
    assert result.secondary_worst_score is not None
    assert result.secondary_worst_score < 20.0


@pytest.mark.unit
def test_an_unembeddable_secondary_is_reported_not_skipped(
    service, reference
) -> None:  # noqa: ANN001
    """So the caller can see which upload failed."""
    result = service.match(
        selfie=reference,
        secondaries=[embedding_at(0.9, reference=reference), None],
    )
    secondaries = [e for e in result.comparisons if e.comparison == "secondary"]
    assert len(secondaries) == 2
    assert secondaries[0].compared is True
    assert secondaries[1].compared is False


@pytest.mark.unit
def test_cnic_identity_match_is_reported(service, reference) -> None:  # noqa: ANN001
    matched = service.match(
        selfie=reference, cnic=embedding_at(0.8, reference=reference)
    )
    mismatched = service.match(
        selfie=reference, cnic=embedding_at(0.05, reference=reference)
    )
    absent = service.match(selfie=reference, profile=embedding_at(0.9, reference=reference))

    assert matched.cnic_identity_match is True
    assert mismatched.cnic_identity_match is False
    assert absent.cnic_identity_match is None


@pytest.mark.unit
def test_the_response_admits_the_thresholds_are_unvalidated(
    service, reference
) -> None:  # noqa: ANN001
    """More useful than a note in a document nobody reads at 3am."""
    result = service.match(
        selfie=reference, profile=embedding_at(0.9, reference=reference)
    )
    assert result.thresholds_validated is False
    assert any(
        w.code == "MATCHING_THRESHOLDS_UNVALIDATED" for w in result.warnings
    )


@pytest.mark.unit
def test_both_the_score_and_the_raw_similarity_are_reported(
    service, reference
) -> None:  # noqa: ANN001
    """The score is for reading; the similarity is what thresholds apply to
    and what a recalibration needs."""
    result = service.match(
        selfie=reference, profile=embedding_at(0.75, reference=reference)
    )
    entry = next(e for e in result.comparisons if e.comparison == "profile")
    assert entry.similarity == pytest.approx(0.75, abs=1e-4)
    assert entry.score > entry.similarity * 100.0  # calibration lifts it


@pytest.mark.unit
def test_describe_reports_the_configuration(service) -> None:  # noqa: ANN001
    described = service.describe()
    json.dumps(described)
    assert described["thresholds"]["cnic"]["strong_match"] == 0.42
    assert described["thresholds_validated"] is False


@pytest.mark.unit
def test_compare_pair_is_available_for_other_modules(service, reference) -> None:  # noqa: ANN001
    """Module 6 needs one comparison, not a whole verification fusion."""
    result = service.compare_pair(
        reference, embedding_at(0.8, reference=reference), ComparisonType.CNIC
    )
    assert result.compared is True
    assert result.decision is MatchDecision.STRONG_MATCH
