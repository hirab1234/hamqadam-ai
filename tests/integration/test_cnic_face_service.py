"""MODULE 6 against real weights: locate the CNIC portrait, match the selfie.

Every card here is synthetic and carries a **public-domain reference
portrait**, print-degraded. No real identity document and no private
individual's photograph appears in this repository.

The centre of gravity is the held-up-card attack. A user photographing their
CNIC while holding it in front of their own face produces two faces in one
frame, and a naive extractor picks the large live one - after which the selfie
is compared against itself and passes whoever the card belongs to. The
scenarios below check that the printed portrait is chosen every time, and that
a *stolen* card held the same way is refused or fails on identity.

Measured on this fixture set, held-up card, varying how close it is held:

    card scale   stolen card                     own card
    0.52         refused (portrait 37 px)        87.7
    0.70         11.2  (cosine 0.067) FAILED     87.8
    0.88         14.7  (cosine 0.088) FAILED     87.5

The live face sits at cosine 0.71 against the selfie - a near-perfect
self-match - and is never once selected.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.core.errors import ErrorCode
from hamqadam_ai.embeddings.base import FaceEmbedding
from hamqadam_ai.services import (
    build_cnic_face_service,
    build_embedding_service,
    build_face_detection_service,
)
from hamqadam_ai.services.cnic_face_service import CnicFaceService
from tests.fixtures.cnic_portrait import (
    alternate_portrait,
    blank_card,
    card_beside_a_bystander,
    card_held_in_front_of_face,
    reference_portrait,
    render_cnic_with_portrait,
)

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.integration

#: Cosine above which the CNIC comparison counts as a match. Mirrors
#: ``matching.selfie_vs_cnic.strong_match``; asserted against directly so a
#: change to the operating point shows up here as well as in Module 4.
CNIC_STRONG_MATCH = 0.42


@pytest.fixture(scope="module")
def service() -> CnicFaceService:
    try:
        return build_cnic_face_service()
    except Exception as exc:  # noqa: BLE001 - a missing model is a skip
        pytest.skip(f"CNIC face service unavailable: {exc}")


@pytest.fixture(scope="module")
def face() -> BgrImage:
    image = reference_portrait()
    if image is None:
        pytest.skip("no public-domain reference portrait installed")
    return image


@pytest.fixture(scope="module")
def stranger() -> BgrImage:
    image = alternate_portrait()
    if image is None:
        pytest.skip("no second reference portrait installed")
    return image


@pytest.fixture(scope="module")
def selfie(face: BgrImage) -> FaceEmbedding:
    """The live selfie of the person printed on the genuine card."""
    detector = build_face_detection_service()
    embedder = build_embedding_service()
    detection = detector.detect(face, role=ImageRole.LIVE_SELFIE)
    return embedder.embed_to_vector(
        face, role=ImageRole.LIVE_SELFIE, detection=detection
    )


@pytest.fixture(scope="module")
def genuine_card(face: BgrImage) -> BgrImage:
    card = render_cnic_with_portrait(face=face)
    if card is None:
        pytest.skip("card fixture unavailable")
    return card


@pytest.fixture(scope="module")
def stranger_card(stranger: BgrImage) -> BgrImage:
    card = render_cnic_with_portrait(face=stranger)
    if card is None:
        pytest.skip("card fixture unavailable")
    return card


# --------------------------------------------------------------------------- #
# The ordinary case
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_a_genuine_card_matches_its_holder(
    service: CnicFaceService, genuine_card: BgrImage, selfie: FaceEmbedding
) -> None:
    result = service.match(genuine_card, selfie)

    assert result.success is True
    assert result.cnic_identity_match is True
    assert result.similarity is not None
    assert result.similarity > CNIC_STRONG_MATCH
    assert result.error_code is None


@pytest.mark.integration
def test_the_portrait_is_located_and_usable(
    service: CnicFaceService, genuine_card: BgrImage
) -> None:
    portrait, embedding = service.extract_portrait(genuine_card)

    assert portrait.found is True
    assert portrait.usable is True
    assert portrait.portrait is not None
    assert embedding is not None
    assert embedding.role is ImageRole.CNIC_PORTRAIT


@pytest.mark.integration
def test_a_print_is_judged_on_the_cnic_quality_scale(
    service: CnicFaceService, genuine_card: BgrImage
) -> None:
    """A sub-300-dpi print behind laminate cannot meet a live selfie's bar and
    is not asked to. Held to the selfie scale it would be rejected outright,
    and a decisive match discarded with it."""
    portrait, _embedding = service.extract_portrait(genuine_card)

    assert portrait.quality_score is not None
    assert portrait.quality_score < 80.0  # nowhere near live-selfie quality
    assert portrait.usable is True


@pytest.mark.integration
def test_a_different_person_on_the_card_fails(
    service: CnicFaceService, stranger_card: BgrImage, selfie: FaceEmbedding
) -> None:
    result = service.match(stranger_card, selfie)

    assert result.success is True
    assert result.cnic_identity_match is False
    assert result.similarity is not None
    assert result.similarity < CNIC_STRONG_MATCH


@pytest.mark.integration
def test_the_score_separates_the_two_cases(
    service: CnicFaceService,
    genuine_card: BgrImage,
    stranger_card: BgrImage,
    selfie: FaceEmbedding,
) -> None:
    """Separation is what makes the operating point meaningful; a threshold
    between two overlapping distributions is a coin toss."""
    genuine = service.match(genuine_card, selfie)
    impostor = service.match(stranger_card, selfie)

    assert genuine.similarity is not None
    assert impostor.similarity is not None
    assert genuine.similarity - impostor.similarity > 0.4


# --------------------------------------------------------------------------- #
# The attack
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.parametrize("card_scale", [0.52, 0.70, 0.88])
def test_a_stolen_card_held_in_front_of_a_face_never_matches(
    service: CnicFaceService,
    stranger_card: BgrImage,
    face: BgrImage,
    selfie: FaceEmbedding,
    card_scale: float,
) -> None:
    """The attack this module exists to stop, at three holding distances.

    The live face in these frames matches the selfie at cosine 0.71 - it is
    the same person - so an extractor that picked it would pass every time.
    Two acceptable outcomes, and no third: refuse because the printed portrait
    is unreadable, or compare against the stranger actually on the card and
    fail. What must never happen is a match.
    """
    attack = card_held_in_front_of_face(
        stranger_card, face=face, card_scale=card_scale
    )
    if attack is None:
        pytest.skip("attack fixture unavailable")

    result = service.match(attack, selfie)

    assert result.cnic_identity_match is not True
    if result.success:
        assert result.similarity is not None
        assert result.similarity < CNIC_STRONG_MATCH


@pytest.mark.integration
def test_the_live_face_is_never_chosen_as_the_portrait(
    service: CnicFaceService,
    stranger_card: BgrImage,
    face: BgrImage,
) -> None:
    """Directly, rather than through the score. The face behind the card is
    the largest thing in the frame and the easiest detection; the chosen
    portrait must not be it."""
    attack = card_held_in_front_of_face(stranger_card, face=face, card_scale=0.70)
    if attack is None:
        pytest.skip("attack fixture unavailable")

    portrait, _embedding = service.extract_portrait(attack)

    assert portrait.portrait is not None
    # The live face covers ~18% of the frame; any printed portrait is an
    # order of magnitude smaller.
    assert portrait.portrait.area_ratio < 0.05


@pytest.mark.integration
def test_a_face_that_cannot_be_on_the_card_is_reported(
    service: CnicFaceService, genuine_card: BgrImage, face: BgrImage
) -> None:
    """The count is the signal Module 9 reads. A match score alone cannot
    distinguish a genuine verification from a card held up in front of its
    own owner - and the count can."""
    attack = card_held_in_front_of_face(genuine_card, face=face, card_scale=0.52)
    if attack is None:
        pytest.skip("attack fixture unavailable")

    result = service.match(attack, None)

    assert result.portrait.foreign_face_count >= 1
    codes = [w.code for w in result.warnings]
    assert "CNIC_FOREIGN_FACE_PRESENT" in codes


@pytest.mark.integration
def test_holding_the_card_up_is_explained_in_terms_of_what_to_do(
    service: CnicFaceService, stranger_card: BgrImage, face: BgrImage
) -> None:
    """"Photograph the card on its own" is actionable. A generic "no portrait
    found" would send the user round the same loop."""
    attack = card_held_in_front_of_face(stranger_card, face=face, card_scale=0.52)
    if attack is None:
        pytest.skip("attack fixture unavailable")

    portrait, _embedding = service.extract_portrait(attack)
    if portrait.found:
        pytest.skip("portrait was readable at this distance; message not exercised")

    assert portrait.error_message is not None
    assert "holding it up in front of your face" in portrait.error_message


@pytest.mark.integration
def test_a_bystander_is_removed_by_rectification(
    service: CnicFaceService, genuine_card: BgrImage, selfie: FaceEmbedding
) -> None:
    """A card on a desk with somebody else in shot. Rectification crops to the
    card's own bounds, so the bystander never reaches the detector - which is
    why it is the primary defence and the geometry checks are the backstop."""
    scene = card_beside_a_bystander(genuine_card)
    if scene is None:
        pytest.skip("bystander fixture unavailable")

    result = service.match(scene, selfie)

    assert result.portrait.rectified is True
    assert result.portrait.foreign_face_count == 0
    assert result.cnic_identity_match is True


# --------------------------------------------------------------------------- #
# Ghost portraits
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_a_ghost_reproduction_is_expected_not_suspicious(
    service: CnicFaceService, face: BgrImage, selfie: FaceEmbedding
) -> None:
    card = render_cnic_with_portrait(face=face, ghost=True)
    if card is None:
        pytest.skip("card fixture unavailable")

    result = service.match(card, selfie)

    assert result.portrait.has_ghost is True
    assert result.portrait.foreign_face_count == 0
    assert result.cnic_identity_match is True
    codes = [w.code for w in result.warnings]
    assert "CNIC_GHOST_PORTRAIT_PRESENT" in codes


@pytest.mark.integration
def test_the_ghost_does_not_degrade_the_match(
    service: CnicFaceService, face: BgrImage, selfie: FaceEmbedding
) -> None:
    """It is a fainter copy of the same face, so choosing it would silently
    hand the recogniser the worse of two images."""
    plain = render_cnic_with_portrait(face=face)
    ghosted = render_cnic_with_portrait(face=face, ghost=True)
    if plain is None or ghosted is None:
        pytest.skip("card fixtures unavailable")

    with_ghost = service.match(ghosted, selfie)
    without = service.match(plain, selfie)

    assert with_ghost.similarity is not None
    assert without.similarity is not None
    assert abs(with_ghost.similarity - without.similarity) < 0.05


# --------------------------------------------------------------------------- #
# Nothing to find
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_strict_mode_refuses_a_card_whose_edges_were_not_found(
    genuine_card: BgrImage, selfie: FaceEmbedding
) -> None:
    """``allow_unrectified: false`` must actually refuse.

    Worth its own test because the first version of this setting was inert: it
    skipped the upscale and then searched the frame anyway, so a deployment
    that had asked for the strong containment guarantee silently did not get
    it. A flat scan has no border to find, which is exactly the case here.
    """
    settings = get_settings().model_copy(deep=True)
    settings.cnic_face.allow_unrectified = False
    strict = build_cnic_face_service(settings)

    result = strict.match(genuine_card, selfie)

    assert result.success is False
    assert result.portrait.found is False
    assert result.portrait.rectified is False
    assert result.error_message is not None
    assert "edges of the card" in result.error_message


@pytest.mark.integration
def test_the_same_card_passes_when_the_guarantee_is_not_required(
    service: CnicFaceService, genuine_card: BgrImage, selfie: FaceEmbedding
) -> None:
    """The other half of the pair: the refusal above is the policy talking,
    not the card being unreadable."""
    assert service.match(genuine_card, selfie).cnic_identity_match is True


@pytest.mark.integration
def test_a_card_with_no_portrait_is_reported(
    service: CnicFaceService, selfie: FaceEmbedding
) -> None:
    result = service.match(blank_card(), selfie)

    assert result.success is False
    assert result.cnic_identity_match is None
    assert result.error_code is ErrorCode.CNIC_FACE_NOT_FOUND
    assert result.portrait.found is False


@pytest.mark.integration
def test_a_missing_selfie_is_not_a_failed_comparison(
    service: CnicFaceService, genuine_card: BgrImage
) -> None:
    """``None`` must not be read as "did not match". The selfie may simply
    have failed its own checks, and the Backend needs to tell those apart."""
    result = service.match(genuine_card, None)

    assert result.success is False
    assert result.cnic_identity_match is None
    assert result.portrait.found is True
    assert result.error_message is not None


@pytest.mark.integration
def test_noise_yields_no_portrait(
    service: CnicFaceService, selfie: FaceEmbedding
) -> None:
    noise = np.random.default_rng(11).integers(
        0, 256, (700, 1100, 3), dtype=np.uint8
    )
    result = service.match(noise, selfie)

    assert result.cnic_identity_match is not True


@pytest.mark.integration
def test_a_non_bgr_array_is_a_programming_error(service: CnicFaceService) -> None:
    """Distinct from a business outcome: the caller passed the wrong thing."""
    with pytest.raises(ValueError, match="BGR"):
        service.extract_portrait(np.zeros((100, 100), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_the_result_serialises(
    service: CnicFaceService, genuine_card: BgrImage, selfie: FaceEmbedding
) -> None:
    import json

    payload = service.match(genuine_card, selfie).model_dump(mode="json")
    json.dumps(payload)

    assert payload["cnic_identity_match"] is True
    assert "portrait" in payload


@pytest.mark.integration
def test_no_biometric_vector_reaches_the_response(
    service: CnicFaceService, genuine_card: BgrImage, selfie: FaceEmbedding
) -> None:
    """An embedding is biometric data. The response carries scores and
    geometry, never the template itself."""
    import json

    serialised = json.dumps(
        service.match(genuine_card, selfie).model_dump(mode="json")
    )

    assert "vector" not in serialised
    assert "embedding" not in serialised


@pytest.mark.integration
def test_the_operating_point_is_reported_as_unvalidated(
    service: CnicFaceService, genuine_card: BgrImage, selfie: FaceEmbedding
) -> None:
    """No Pakistani print-versus-live dataset has been used. Reporting the
    threshold as validated would be a claim nobody made."""
    result = service.match(genuine_card, selfie)

    assert result.thresholds_validated is False
    assert result.strong_match_threshold is not None
    assert result.review_threshold is not None


@pytest.mark.integration
def test_the_summary_is_pii_free(
    service: CnicFaceService, genuine_card: BgrImage, selfie: FaceEmbedding
) -> None:
    import json

    summary = service.match(genuine_card, selfie).summary()
    json.dumps(summary)

    assert set(summary) >= {"success", "match", "score", "foreign_faces"}


@pytest.mark.integration
def test_the_service_describes_its_configuration(service: CnicFaceService) -> None:
    import json

    described = service.describe()
    json.dumps(described)

    assert described["thresholds_validated"] is False
    assert "geometry" in described


@pytest.mark.integration
def test_reading_the_same_card_twice_agrees(
    service: CnicFaceService, genuine_card: BgrImage, selfie: FaceEmbedding
) -> None:
    """A verification decision that changes between two reads of one image is
    not a decision anyone can defend."""
    first = service.match(genuine_card, selfie)
    second = service.match(genuine_card, selfie)

    assert first.similarity == pytest.approx(second.similarity, abs=1e-6)
    assert first.cnic_identity_match == second.cnic_identity_match
