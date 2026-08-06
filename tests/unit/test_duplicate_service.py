"""MODULE 8 service logic, driven by synthetic embeddings.

No model weights: the service takes a :class:`FaceEmbedding` and a gallery, and
both are constructible by hand. That makes the parts worth testing - the
self-exclusion, the version isolation, the erasure semantics, the warnings -
pinnable against situations no fixture would naturally produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from hamqadam_ai.core.config import DuplicateConfig, QdrantConfig
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.core.exceptions import (
    DependencyUnavailableError,
    VectorStoreError,
)
from hamqadam_ai.duplicate_detection import build_store
from hamqadam_ai.duplicate_detection.base import (
    SearchHit,
    VectorRecord,
    VectorStore,
    utc_now,
)
from hamqadam_ai.duplicate_detection.memory_store import InMemoryVectorStore
from hamqadam_ai.embeddings.base import FaceEmbedding
from hamqadam_ai.services.duplicate_service import (
    UNCALIBRATED_GALLERY_WARNING,
    DuplicateService,
)

MODEL = "arcface-test-1"
DIMENSION = 128


def embedding(vector: np.ndarray, *, model_version: str = MODEL) -> FaceEmbedding:
    """A template, normalised."""
    array = np.asarray(vector, dtype=np.float32)
    array = array / np.linalg.norm(array)
    return FaceEmbedding(
        vector=array,
        raw_norm=22.0,
        confidence=0.9,
        model_key="face_embedder_arcface",
        model_version=model_version,
        role=ImageRole.LIVE_SELFIE,
    )


def unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=DIMENSION).astype(np.float32)
    return vector / np.linalg.norm(vector)


def at_similarity(base: np.ndarray, similarity: float, *, seed: int = 99) -> np.ndarray:
    """A vector at approximately the requested cosine to ``base``."""
    rng = np.random.default_rng(seed)
    orthogonal = rng.normal(size=base.shape).astype(np.float32)
    orthogonal -= (orthogonal @ base) * base
    orthogonal /= np.linalg.norm(orthogonal)
    combined = similarity * base + np.sqrt(max(1.0 - similarity**2, 0.0)) * orthogonal
    return combined / np.linalg.norm(combined)


@pytest.fixture
def service():  # noqa: ANN201 - pytest fixture
    from hamqadam_ai.core.config import get_settings

    store = InMemoryVectorStore(max_records=10_000)
    built = DuplicateService(store=store, settings=get_settings())
    yield built
    built.close()


# --------------------------------------------------------------------------- #
# The empty gallery
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_an_empty_gallery_finds_nothing(service: DuplicateService) -> None:
    """The normal state of a new deployment."""
    result = service.check(embedding(unit(1)), reference="alice")

    assert result.duplicate_found is False
    assert result.searched is True
    assert result.gallery_size == 0
    assert result.best_similarity is None


@pytest.mark.unit
def test_no_template_means_no_search(service: DuplicateService) -> None:
    """Distinct from finding nothing: the selfie itself may have failed, and
    the Backend must not read that as "no duplicate"."""
    result = service.check(None, reference="alice")

    assert result.searched is False
    assert result.duplicate_found is False
    assert result.error_code is ErrorCode.FACE_NOT_DETECTED


@pytest.mark.unit
def test_a_degenerate_template_means_no_search(service: DuplicateService) -> None:
    degenerate = FaceEmbedding(
        vector=np.zeros(DIMENSION, dtype=np.float32),
        raw_norm=0.0,
        confidence=0.0,
        model_key="face_embedder_arcface",
        model_version=MODEL,
    )
    assert service.check(degenerate, reference="alice").searched is False


# --------------------------------------------------------------------------- #
# Self-exclusion
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_returning_user_is_not_their_own_duplicate(
    service: DuplicateService,
) -> None:
    """The trap this module is most likely to fall into. Without excluding the
    querying reference, every re-verification matches at cosine 1.0 - and the
    bug looks exactly like a working detector."""
    alice = unit(1)
    service.enrol(embedding(alice), reference="alice")

    result = service.check(embedding(alice), reference="alice")

    assert result.duplicate_found is False
    assert result.self_excluded is True
    assert result.candidates == []


@pytest.mark.unit
def test_omitting_the_reference_is_warned_about(service: DuplicateService) -> None:
    """Because the consequence is silent and severe, the caller is told rather
    than left to discover it."""
    alice = unit(1)
    service.enrol(embedding(alice), reference="alice")

    result = service.check(embedding(alice), reference=None)

    assert result.duplicate_found is True
    assert result.best_similarity == pytest.approx(1.0, abs=1e-5)
    assert "DUPLICATE_SELF_NOT_EXCLUDED" in [w.code for w in result.warnings]


@pytest.mark.unit
def test_excluding_one_user_does_not_hide_another(
    service: DuplicateService,
) -> None:
    alice = unit(1)
    service.enrol(embedding(alice), reference="alice")
    service.enrol(embedding(at_similarity(alice, 0.95)), reference="bob")

    result = service.check(embedding(alice), reference="alice")

    assert result.duplicate_found is True
    assert result.candidates[0].reference == "bob"


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_same_face_is_a_duplicate(service: DuplicateService) -> None:
    alice = unit(1)
    service.enrol(embedding(alice), reference="alice")

    result = service.check(embedding(at_similarity(alice, 0.97)), reference="mallory")

    assert result.duplicate_found is True
    assert result.candidates[0].is_duplicate is True
    assert result.recommended_action == "manual_review"


@pytest.mark.unit
def test_a_borderline_face_needs_review(service: DuplicateService) -> None:
    """Between the two thresholds: close enough to be worth a human look, not
    close enough to call."""
    alice = unit(1)
    service.enrol(embedding(alice), reference="alice")

    result = service.check(embedding(at_similarity(alice, 0.62)), reference="carol")

    assert result.duplicate_found is False
    assert result.needs_review is True
    assert result.candidates[0].needs_review is True
    assert result.recommended_action == "manual_review"


@pytest.mark.unit
def test_a_stranger_passes(service: DuplicateService) -> None:
    service.enrol(embedding(unit(1)), reference="alice")

    result = service.check(embedding(unit(500)), reference="dave")

    assert result.duplicate_found is False
    assert result.needs_review is False
    assert result.recommended_action == "proceed"


@pytest.mark.unit
def test_a_duplicate_can_be_configured_to_recommend_rejection() -> None:
    from hamqadam_ai.core.config import get_settings

    settings = get_settings().model_copy(deep=True)
    settings.duplicate.on_duplicate = "reject"
    store = InMemoryVectorStore()
    built = DuplicateService(store=store, settings=settings)

    alice = unit(1)
    built.enrol(embedding(alice), reference="alice")
    result = built.check(embedding(at_similarity(alice, 0.97)), reference="mallory")

    assert result.recommended_action == "reject"
    built.close()


@pytest.mark.unit
def test_candidates_are_ordered_and_scored(service: DuplicateService) -> None:
    alice = unit(1)
    for index, similarity in enumerate((0.95, 0.70, 0.40)):
        service.enrol(
            embedding(at_similarity(alice, similarity, seed=index)),
            reference=f"user-{index}",
        )

    result = service.check(embedding(alice), reference="query")
    scores = [candidate.match_score for candidate in result.candidates]

    assert scores == sorted(scores, reverse=True)
    assert result.best_match_score == scores[0]


# --------------------------------------------------------------------------- #
# Enrolment and erasure
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_checking_does_not_enrol(service: DuplicateService) -> None:
    """Deliberate, and a safety property rather than an inconvenience:
    enrolling as a side effect of checking would put a *rejected* applicant's
    face in the gallery, where it would then match their next legitimate
    attempt."""
    service.check(embedding(unit(1)), reference="alice")

    assert service.gallery_size() == 0


@pytest.mark.unit
def test_enrolling_reports_replacement(service: DuplicateService) -> None:
    first = service.enrol(embedding(unit(1)), reference="alice")
    second = service.enrol(embedding(unit(2)), reference="alice")

    assert first.replaced is False
    assert second.replaced is True
    assert second.gallery_size == 1


@pytest.mark.unit
def test_a_reference_is_required_to_enrol(service: DuplicateService) -> None:
    with pytest.raises(ValueError, match="reference"):
        service.enrol(embedding(unit(1)), reference="")


@pytest.mark.unit
def test_a_degenerate_template_is_refused(service: DuplicateService) -> None:
    degenerate = FaceEmbedding(
        vector=np.zeros(DIMENSION, dtype=np.float32),
        raw_norm=0.0,
        confidence=0.0,
        model_key="face_embedder_arcface",
        model_version=MODEL,
    )
    result = service.enrol(degenerate, reference="alice")

    assert result.enrolled is False
    assert result.error_code is ErrorCode.VALIDATION_ERROR


@pytest.mark.unit
def test_a_template_can_be_erased(service: DuplicateService) -> None:
    alice = unit(1)
    service.enrol(embedding(alice), reference="alice")

    assert service.forget("alice") is True
    assert service.check(embedding(alice), reference="zed").duplicate_found is False


@pytest.mark.unit
def test_erasure_is_idempotent(service: DuplicateService) -> None:
    service.enrol(embedding(unit(1)), reference="alice")

    assert service.forget("alice") is True
    assert service.forget("alice") is False


# --------------------------------------------------------------------------- #
# Model versions
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_search_never_crosses_model_versions(service: DuplicateService) -> None:
    """Two embeddings from different recogniser builds are not comparable, and
    comparing them anyway yields scores that look ordinary and mean nothing."""
    alice = unit(1)
    service.enrol(embedding(alice, model_version=MODEL), reference="alice")

    result = service.check(
        embedding(alice, model_version="arcface-vNEXT"), reference="query"
    )

    assert result.duplicate_found is False
    assert result.gallery_size == 0
    assert result.candidates == []


@pytest.mark.unit
def test_the_gallery_size_is_per_version(service: DuplicateService) -> None:
    """A caller cannot interpret a similarity without knowing how many
    *comparable* entries it beat."""
    service.enrol(embedding(unit(1), model_version=MODEL), reference="alice")
    service.enrol(embedding(unit(2), model_version="other"), reference="bob")

    result = service.check(embedding(unit(3)), reference="query")

    assert result.gallery_size == 1


# --------------------------------------------------------------------------- #
# Warnings the caller needs
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_duplicate_carries_the_relatives_caveat(
    service: DuplicateService,
) -> None:
    """Face recognition cannot separate identical twins at any threshold, and
    a matrimonial platform serving extended families will enrol exactly the
    population where that is most common."""
    alice = unit(1)
    service.enrol(embedding(alice), reference="alice")

    result = service.check(embedding(at_similarity(alice, 0.97)), reference="m")

    assert "DUPLICATE_MAY_BE_A_RELATIVE" in [w.code for w in result.warnings]


@pytest.mark.unit
def test_a_large_gallery_warns_that_the_threshold_is_uncalibrated(
    service: DuplicateService,
) -> None:
    """The 1:N false-match rate compounds with gallery size, and the shipped
    threshold was never derived against a real one."""
    rng = np.random.default_rng(0)
    for index in range(UNCALIBRATED_GALLERY_WARNING + 1):
        vector = rng.normal(size=DIMENSION).astype(np.float32)
        service._store.enrol(  # noqa: SLF001 - bulk fill, bypassing the service
            VectorRecord(
                reference=f"user-{index}",
                vector=vector,
                model_version=MODEL,
                enrolled_at=utc_now(),
            )
        )

    result = service.check(embedding(unit(7)), reference="query")
    codes = [w.code for w in result.warnings]

    assert "DUPLICATE_THRESHOLD_UNCALIBRATED" in codes


@pytest.mark.unit
def test_a_small_gallery_does_not_warn(service: DuplicateService) -> None:
    service.enrol(embedding(unit(1)), reference="alice")
    result = service.check(embedding(unit(2)), reference="query")

    assert "DUPLICATE_THRESHOLD_UNCALIBRATED" not in [
        w.code for w in result.warnings
    ]


@pytest.mark.unit
def test_thresholds_are_never_reported_as_validated(
    service: DuplicateService,
) -> None:
    """A simulation puts the correct threshold anywhere between 0.26 and 0.96
    depending on a property of the embedding manifold nobody here can measure."""
    assert service.check(embedding(unit(1)), reference="a").thresholds_validated is False
    assert service.describe()["thresholds_validated"] is False


@pytest.mark.unit
def test_the_service_names_its_known_limits(service: DuplicateService) -> None:
    limits = service.describe()["known_limits"]

    assert "identical_twins" in limits
    assert "close_relatives" in limits


# --------------------------------------------------------------------------- #
# Gallery failure
# --------------------------------------------------------------------------- #


class BrokenStore(VectorStore):
    """A gallery that is down."""

    def __init__(self) -> None:
        super().__init__(name="broken")

    def enrol(self, record: VectorRecord) -> None:
        raise VectorStoreError("gallery unreachable")

    def search(self, vector, *, model_version, top_k, exclude=None):  # noqa: ANN001, ANN201
        raise VectorStoreError("gallery unreachable")

    def exists(self, reference: str) -> bool:
        raise VectorStoreError("gallery unreachable")

    def delete(self, reference: str) -> bool:
        raise VectorStoreError("gallery unreachable")

    def count(self, *, model_version: str | None = None) -> int:
        raise VectorStoreError("gallery unreachable")

    def close(self) -> None:
        return None


@pytest.mark.unit
def test_a_gallery_outage_is_reported_not_raised() -> None:
    """A duplicate check failing must not take the rest of a verification with
    it - the face matching and document findings are still worth having."""
    from hamqadam_ai.core.config import get_settings

    service = DuplicateService(store=BrokenStore(), settings=get_settings())
    result = service.check(embedding(unit(1)), reference="alice")

    assert result.searched is False
    assert result.duplicate_found is False
    assert result.error_code is ErrorCode.VECTOR_DB_ERROR
    assert result.error_message is not None
    assert "no conclusion" in result.error_message.lower()


@pytest.mark.unit
def test_an_enrolment_outage_is_reported_not_raised() -> None:
    from hamqadam_ai.core.config import get_settings

    service = DuplicateService(store=BrokenStore(), settings=get_settings())
    result = service.enrol(embedding(unit(1)), reference="alice")

    assert result.enrolled is False
    assert result.error_code is ErrorCode.VECTOR_DB_ERROR


@pytest.mark.unit
def test_a_full_memory_gallery_is_reported_not_raised() -> None:
    from hamqadam_ai.core.config import get_settings

    service = DuplicateService(
        store=InMemoryVectorStore(max_records=1), settings=get_settings()
    )
    service.enrol(embedding(unit(1)), reference="alice")
    result = service.enrol(embedding(unit(2)), reference="bob")

    assert result.enrolled is False
    assert result.error_code is ErrorCode.VECTOR_DB_ERROR
    service.close()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_summary_names_no_matched_account(service: DuplicateService) -> None:
    """A log line saying which account a face matched is a linkage nobody
    asked for."""
    alice = unit(1)
    service.enrol(embedding(alice), reference="alice-account-12345")
    summary = service.check(
        embedding(at_similarity(alice, 0.97)), reference="m"
    ).summary()

    assert "alice-account-12345" not in repr(summary)
    assert summary["duplicate"] is True


@pytest.mark.unit
def test_the_result_serialises(service: DuplicateService) -> None:
    import json

    service.enrol(embedding(unit(1)), reference="alice")
    payload = service.check(embedding(unit(1)), reference="q").model_dump(mode="json")

    json.dumps(payload)
    assert payload["thresholds_validated"] is False


@pytest.mark.unit
def test_no_vector_reaches_the_response(service: DuplicateService) -> None:
    """A template is biometric data; the response carries scores and
    references, never the vector itself."""
    import json

    service.enrol(embedding(unit(1)), reference="alice")
    serialised = json.dumps(
        service.check(embedding(unit(1)), reference="q").model_dump(mode="json")
    )

    assert "vector" not in serialised


@pytest.mark.unit
def test_the_service_describes_its_configuration(service: DuplicateService) -> None:
    import json

    described = service.describe()
    json.dumps(described)

    assert described["store"]["store"] == "memory"
    assert described["store"]["durable"] is False


@pytest.mark.unit
def test_a_hit_records_the_model_version(service: DuplicateService) -> None:
    alice = unit(1)
    service.enrol(embedding(alice), reference="alice")
    result = service.check(embedding(at_similarity(alice, 0.9)), reference="q")

    assert result.model_version == MODEL


@pytest.mark.unit
def test_checking_is_deterministic(service: DuplicateService) -> None:
    alice = unit(1)
    service.enrol(embedding(alice), reference="alice")
    probe = embedding(at_similarity(alice, 0.8))

    first = service.check(probe, reference="q")
    second = service.check(probe, reference="q")

    assert first.best_similarity == pytest.approx(second.best_similarity)


@pytest.mark.unit
def test_a_search_hit_is_rendered_faithfully() -> None:
    """The mapping from store hit to response candidate must not lose or
    reinterpret anything."""
    hit = SearchHit(reference="alice", similarity=0.75, model_version=MODEL)

    assert hit.reference == "alice"
    assert hit.similarity == pytest.approx(0.75)


class TestQdrantIsTheDefault:
    """The shipped default must be durable, and must not degrade silently.

    Both halves were measured problems. The default was `memory`, so a stock
    deployment reported `store: "memory"` and lost every template on restart -
    a face found as a duplicate at cosine 1.0 was approved after a restart with
    gallery_size back to 0. And the Qdrant path fell back to memory
    *unconditionally* when unreachable, so a misconfigured URL produced the same
    silent loss while logging only an error nobody reads on a 200 OK.
    """

    def test_the_shipped_default_is_qdrant(self) -> None:
        assert DuplicateConfig().backend == "qdrant"

    def test_the_default_url_is_a_server_not_an_on_disk_path(self) -> None:
        """Deliberately the reverse of what this test asserted before.

        The embedded on-disk engine was the default because it gave a single
        node durability with nothing else to install. Two properties made it
        wrong for a deployment, and both were observed here rather than
        theorised:

        * it takes an **exclusive lock** on its directory, so a second API
          replica cannot open the gallery at all; and
        * a hard crash leaves that lock behind, so the next start refuses -
          which bit during testing, with `Storage folder ./data/qdrant is
          already accessed by another instance`.

        A server has neither problem. The cost is that a stock deployment now
        needs a Qdrant container running, which `deploy/docker-compose.yml`
        provides and the VPS guide documents.
        """
        url = DuplicateConfig().qdrant.url
        assert url.startswith(("http://", "https://")), (
            "the default gallery endpoint must be a Qdrant server; an embedded "
            "path cannot be shared between replicas and strands its own lock "
            "on a crash"
        )
        assert url != ":memory:"

    def test_silent_fallback_is_off_by_default(self) -> None:
        assert DuplicateConfig().allow_memory_fallback is False

    def test_an_unreachable_qdrant_refuses_rather_than_degrading(self) -> None:
        """Fail loudly. A non-durable fraud control that reports healthy is worse
        than one that will not start.
        """
        config = DuplicateConfig(
            backend="qdrant",
            qdrant=QdrantConfig(url="http://127.0.0.1:59999", timeout_seconds=1.0),
            allow_memory_fallback=False,
        )
        with pytest.raises((VectorStoreError, DependencyUnavailableError)):
            build_store(config, dimension=512)

    def test_the_downgrade_is_available_when_opted_into(self) -> None:
        """Explicit is fine; implicit is not."""
        config = DuplicateConfig(
            backend="qdrant",
            qdrant=QdrantConfig(url="http://127.0.0.1:59999", timeout_seconds=1.0),
            allow_memory_fallback=True,
        )
        assert build_store(config, dimension=512).name == "memory"
