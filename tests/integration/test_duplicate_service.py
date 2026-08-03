"""MODULE 8 against real ArcFace templates.

The unit tests drive the service with synthetic vectors, which pins the logic
but proves nothing about whether real embeddings actually behave the way the
thresholds assume. These use the recogniser.

Only two public-domain reference faces are available, so what can honestly be
checked is the extremes: the same face found, a different face not found, and
the end-to-end path intact through both store adapters. That is not a
validation of the operating point and this file does not pretend otherwise -
see ``scripts/calibrate_duplicate_threshold.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.duplicate_detection.memory_store import InMemoryVectorStore
from hamqadam_ai.embeddings.base import FaceEmbedding
from hamqadam_ai.services import build_embedding_service, build_face_detection_service
from hamqadam_ai.services.duplicate_service import DuplicateService
from tests.fixtures.profile_images import (
    _jpeg,
    alternate_photo,
    reference_photo,
)

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder():  # noqa: ANN201 - pytest fixture
    try:
        return build_embedding_service()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"embedding service unavailable: {exc}")


@pytest.fixture(scope="module")
def detector():  # noqa: ANN201 - pytest fixture
    try:
        return build_face_detection_service()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"face detector unavailable: {exc}")


def template(embedder, detector, image: BgrImage) -> FaceEmbedding:
    """Embed one photograph the way the pipeline would."""
    detection = detector.detect(image, role=ImageRole.LIVE_SELFIE)
    return embedder.embed_to_vector(
        image, role=ImageRole.LIVE_SELFIE, detection=detection
    )


@pytest.fixture(scope="module")
def alice(embedder, detector) -> FaceEmbedding:
    image = reference_photo()
    if image is None:
        pytest.skip("no public-domain reference photograph installed")
    return template(embedder, detector, image)


@pytest.fixture(scope="module")
def alice_again(embedder, detector) -> FaceEmbedding:
    """The same person, re-encoded - a second capture of one face."""
    image = reference_photo()
    if image is None:
        pytest.skip("no public-domain reference photograph installed")
    return template(embedder, detector, _jpeg(image, quality=70))


@pytest.fixture(scope="module")
def bob(embedder, detector) -> FaceEmbedding:
    image = alternate_photo()
    if image is None:
        pytest.skip("no second reference photograph installed")
    return template(embedder, detector, image)


def make_service(store=None) -> DuplicateService:  # noqa: ANN001
    from hamqadam_ai.core.config import get_settings

    return DuplicateService(
        store=store or InMemoryVectorStore(max_records=1000),
        settings=get_settings(),
    )


# --------------------------------------------------------------------------- #
# Real templates
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_the_same_person_enrolling_twice_is_caught(
    alice: FaceEmbedding, alice_again: FaceEmbedding
) -> None:
    """The whole purpose of the module: one person, two accounts."""
    service = make_service()
    service.enrol(alice, reference="account-1")

    result = service.check(alice_again, reference="account-2")

    assert result.duplicate_found is True
    assert result.candidates[0].reference == "account-1"
    assert result.best_similarity is not None
    assert result.best_similarity > result.duplicate_threshold
    service.close()


@pytest.mark.integration
def test_a_different_person_is_not_a_duplicate(
    alice: FaceEmbedding, bob: FaceEmbedding
) -> None:
    service = make_service()
    service.enrol(alice, reference="account-1")

    result = service.check(bob, reference="account-2")

    assert result.duplicate_found is False
    assert result.needs_review is False
    assert result.recommended_action == "proceed"
    service.close()


@pytest.mark.integration
def test_the_two_cases_are_far_apart(
    alice: FaceEmbedding, alice_again: FaceEmbedding, bob: FaceEmbedding
) -> None:
    """Separation is what makes a threshold meaningful at all. A threshold
    between two overlapping distributions is a coin toss with a number on it."""
    service = make_service()
    service.enrol(alice, reference="account-1")

    same = service.check(alice_again, reference="q1")
    different = service.check(bob, reference="q2")

    assert same.best_similarity is not None
    assert different.best_similarity is not None
    assert same.best_similarity - different.best_similarity > 0.5
    service.close()


@pytest.mark.integration
def test_a_returning_user_is_not_flagged(alice: FaceEmbedding) -> None:
    """Real templates, real self-exclusion. Without it every re-verification
    is a duplicate of itself."""
    service = make_service()
    service.enrol(alice, reference="account-1")

    result = service.check(alice, reference="account-1")

    assert result.duplicate_found is False
    assert result.self_excluded is True
    service.close()


@pytest.mark.integration
def test_re_encoding_barely_moves_the_template(
    alice: FaceEmbedding, alice_again: FaceEmbedding
) -> None:
    """A photograph that has been through a messaging app must still match
    itself, or the module would miss the most common duplicate of all."""
    from hamqadam_ai.embeddings.base import cosine_similarity

    assert cosine_similarity(alice.vector, alice_again.vector) > 0.9


# --------------------------------------------------------------------------- #
# Both adapters agree
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["memory", "qdrant"])
def test_both_stores_reach_the_same_verdict(
    alice: FaceEmbedding,
    alice_again: FaceEmbedding,
    bob: FaceEmbedding,
    backend: str,
) -> None:
    """Two adapters disagreeing about a duplicate would give a deployment a
    different answer depending on how it was configured."""
    if backend == "qdrant":
        pytest.importorskip("qdrant_client")
        from hamqadam_ai.duplicate_detection.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore(
            url=":memory:", collection="int", dimension=alice.dimension
        )
    else:
        store = InMemoryVectorStore(max_records=100)

    service = make_service(store)
    service.enrol(alice, reference="account-1")

    assert service.check(alice_again, reference="q").duplicate_found is True
    assert service.check(bob, reference="q").duplicate_found is False
    service.close()


@pytest.mark.integration
def test_the_stores_agree_on_the_similarity(
    alice: FaceEmbedding, alice_again: FaceEmbedding
) -> None:
    """Not just the verdict but the number. A calibration derived against one
    store has to transfer to the other."""
    pytest.importorskip("qdrant_client")
    from hamqadam_ai.duplicate_detection.qdrant_store import QdrantVectorStore

    memory = make_service(InMemoryVectorStore(max_records=100))
    qdrant = make_service(
        QdrantVectorStore(url=":memory:", collection="agree", dimension=alice.dimension)
    )
    for service in (memory, qdrant):
        service.enrol(alice, reference="account-1")

    from_memory = memory.check(alice_again, reference="q").best_similarity
    from_qdrant = qdrant.check(alice_again, reference="q").best_similarity

    assert from_memory is not None
    assert from_qdrant is not None
    assert from_memory == pytest.approx(from_qdrant, abs=1e-4)
    memory.close()
    qdrant.close()


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_erasing_a_template_stops_it_matching(
    alice: FaceEmbedding, alice_again: FaceEmbedding
) -> None:
    """Right to erasure, end to end. A service that stores face templates and
    cannot erase one cannot lawfully be deployed."""
    service = make_service()
    service.enrol(alice, reference="account-1")
    assert service.check(alice_again, reference="q").duplicate_found is True

    service.forget("account-1")

    assert service.check(alice_again, reference="q").duplicate_found is False
    assert service.gallery_size() == 0
    service.close()


@pytest.mark.integration
def test_a_rejected_applicant_is_not_enrolled_by_checking(
    alice: FaceEmbedding,
) -> None:
    """``check`` never writes. Enrolling as a side effect would put a rejected
    applicant's face in the gallery, where it would match their next
    legitimate attempt."""
    service = make_service()
    service.check(alice, reference="account-1")

    assert service.gallery_size() == 0
    service.close()


@pytest.mark.integration
def test_the_response_carries_no_template(alice: FaceEmbedding) -> None:
    import json

    service = make_service()
    service.enrol(alice, reference="account-1")
    serialised = json.dumps(
        service.check(alice, reference="q").model_dump(mode="json")
    )

    assert "vector" not in serialised
    service.close()


@pytest.mark.integration
def test_the_operating_point_is_reported_as_unvalidated(
    alice: FaceEmbedding,
) -> None:
    """No real gallery has been used, and simulation puts the correct
    threshold anywhere between 0.26 and 0.99 depending on a property of the
    embedding manifold that cannot be measured here."""
    service = make_service()
    result = service.check(alice, reference="q")

    assert result.thresholds_validated is False
    service.close()
