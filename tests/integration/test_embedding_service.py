"""MODULE 3 against the real ArcFace weights.

Pins the behaviour the calibration run measured, so a future change to
alignment, batching or caching that silently degrades recognition fails here
rather than in production. Skips cleanly when the model store is not
populated.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.embeddings.alignment import align_face_for_recognition
from hamqadam_ai.embeddings.base import EmbeddingRequest, cosine_similarity
from hamqadam_ai.models.registry import get_registry
from hamqadam_ai.services import build_embedding_service, build_face_detection_service
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

BgrImage = npt.NDArray[np.uint8]

pytestmark = pytest.mark.integration

_LANDMARK_ORDER = ("left_eye", "right_eye", "nose_tip", "mouth_left", "mouth_right")


@pytest.fixture(scope="module")
def portrait() -> BgrImage:
    """Public-domain reference portrait."""
    matplotlib = pytest.importorskip("matplotlib")
    path = (
        Path(matplotlib.__file__).parent
        / "mpl-data" / "sample_data" / "grace_hopper.jpg"
    )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        pytest.skip("reference portrait unavailable")
    return image


@pytest.fixture(scope="module")
def impostor() -> BgrImage:
    """A genuinely different person, for separation testing."""
    skimage_data = pytest.importorskip("skimage.data")
    return cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR)


@pytest.fixture(scope="module")
def detector():  # noqa: ANN201 - pytest fixture
    """Face detection, skipping when unavailable."""
    settings = get_settings()
    try:
        return build_face_detection_service(settings, get_registry(settings))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no face detector available: {exc}")


@pytest.fixture(scope="module")
def service():  # noqa: ANN201 - pytest fixture
    """The embedding service, skipping when the recogniser is absent."""
    settings = get_settings()
    try:
        return build_embedding_service(settings, get_registry(settings))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ArcFace weights unavailable: {exc}")


def geometry(detection) -> tuple[BoundingBox | None, Landmarks5 | None]:  # noqa: ANN001
    """Extract box and landmarks from a detection result."""
    if detection.primary_face is None:
        return None, None
    bb = detection.primary_face.bounding_box
    box = BoundingBox(bb.x1, bb.y1, bb.x2, bb.y2)
    named = {lm.name: (lm.x, lm.y) for lm in detection.primary_face.landmarks}
    marks = (
        Landmarks5(np.array([named[n] for n in _LANDMARK_ORDER], dtype=np.float32))
        if all(n in named for n in _LANDMARK_ORDER)
        else None
    )
    return box, marks


@pytest.fixture(scope="module")
def reference(service, detector, portrait):  # noqa: ANN001, ANN201 - fixture
    """The reference embedding for the portrait."""
    detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    if detection.primary_face is None:
        pytest.skip("no face detected in the reference portrait")
    box, marks = geometry(detection)
    return service.embed_to_vector(
        portrait, role=ImageRole.LIVE_SELFIE, box=box, landmarks=marks
    )


# --------------------------------------------------------------------------- #
# The basic contract
# --------------------------------------------------------------------------- #


def test_an_embedding_has_the_declared_shape(reference) -> None:  # noqa: ANN001
    assert reference.dimension == 512
    assert reference.vector.dtype == np.float32
    assert not reference.is_degenerate


def test_the_vector_is_unit_length(reference) -> None:  # noqa: ANN001
    """Cosine similarity reduces to a dot product, which Qdrant indexes."""
    assert float(np.linalg.norm(reference.vector)) == pytest.approx(1.0, abs=1e-5)


def test_a_face_is_identical_to_itself(reference) -> None:  # noqa: ANN001
    assert reference.similarity_to(reference) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic(service, detector, portrait) -> None:  # noqa: ANN001
    """What makes the cache sound: the same crop must give the same bytes."""
    detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    box, marks = geometry(detection)
    crop = align_face_for_recognition(portrait, box=box, landmarks=marks).crop

    first = service._embedder.embed_aligned([crop])[0][0]  # noqa: SLF001
    second = service._embedder.embed_aligned([crop])[0][0]  # noqa: SLF001

    np.testing.assert_array_equal(first, second)


# --------------------------------------------------------------------------- #
# Separation: the property the whole system rests on
# --------------------------------------------------------------------------- #


def test_a_different_person_scores_far_below_a_genuine_pair(
    service, detector, reference, impostor
) -> None:  # noqa: ANN001
    """Measured on this pair: genuine self-similarity bottoms out at 0.90 under
    heavy degradation while the impostor sits near zero - a margin of ~0.90.
    """
    detection = detector.detect(impostor, role=ImageRole.PROFILE_IMAGE)
    if detection.primary_face is None:
        pytest.skip("no face detected in the impostor image")

    box, marks = geometry(detection)
    other = service.embed_to_vector(
        impostor, role=ImageRole.PROFILE_IMAGE, box=box, landmarks=marks
    )
    score = reference.similarity_to(other)

    assert score < 0.35, f"impostor scored {score:.4f}, far too close to genuine"


@pytest.mark.parametrize(
    ("label", "transform", "floor"),
    [
        ("jpeg_q30", lambda i: _jpeg(i, 30), 0.90),
        ("blur_5", lambda i: cv2.GaussianBlur(i, (5, 5), 0), 0.90),
        ("blur_11", lambda i: cv2.GaussianBlur(i, (11, 11), 0), 0.85),
        ("dark", lambda i: np.clip(i * 0.4, 0, 255).astype(np.uint8), 0.85),
        ("noise", lambda i: _add_noise(i, 20.0), 0.85),
    ],
)
def test_degradation_preserves_identity(
    service, detector, reference, portrait, label: str, transform, floor: float
) -> None:  # noqa: ANN001
    """The same person through a single controlled defect must stay recognisable.

    Floors are set below the measured values with headroom, so this catches a
    genuine regression rather than tracking noise.
    """
    degraded = transform(portrait)
    detection = detector.detect(degraded, role=ImageRole.LIVE_SELFIE)
    if detection.primary_face is None:
        pytest.skip(f"{label}: the detector lost the face")

    box, marks = geometry(detection)
    embedding = service.embed_to_vector(
        degraded, role=ImageRole.LIVE_SELFIE, box=box, landmarks=marks
    )
    score = reference.similarity_to(embedding)

    assert score >= floor, f"{label}: self-similarity fell to {score:.4f}"


def _jpeg(image: BgrImage, quality: int) -> BgrImage:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _add_noise(image: BgrImage, sigma: float) -> BgrImage:
    rng = np.random.default_rng(5)
    return np.clip(
        image.astype(np.float32) + rng.normal(0.0, sigma, image.shape), 0, 255
    ).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Alignment, measured end to end
# --------------------------------------------------------------------------- #


def test_alignment_materially_changes_the_embedding(
    service, detector, portrait
) -> None:  # noqa: ANN001
    """Measured: the same face through the template and through the box
    fallback scores only 0.786 against itself."""
    detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    box, marks = geometry(detection)
    if marks is None:
        pytest.skip("no landmarks available")

    aligned = align_face_for_recognition(portrait, box=box, landmarks=marks)
    boxed = align_face_for_recognition(portrait, box=box, landmarks=None)

    pairs = service._embedder.embed_aligned([aligned.crop, boxed.crop])  # noqa: SLF001
    score = cosine_similarity(pairs[0][0], pairs[1][0])

    assert score < 0.95, (
        "the box fallback produced a near-identical embedding, which suggests "
        "the landmark warp is not actually being applied"
    )


def test_alignment_rescues_a_rotated_capture(
    service, detector, portrait
) -> None:  # noqa: ANN001
    """The strongest argument for doing alignment properly.

    Measured on a 25-degree rotation: template-aligned scores 0.985 against the
    upright original, box-only 0.573 - the latter below any sane impostor
    threshold, so the system would call the person a stranger purely because
    alignment was skipped.
    """
    height, width = portrait.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 25.0, 1.0)
    rotated = cv2.warpAffine(portrait, matrix, (width, height), borderValue=(114,) * 3)

    upright_detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    rotated_detection = detector.detect(rotated, role=ImageRole.LIVE_SELFIE)
    if rotated_detection.primary_face is None:
        pytest.skip("the detector lost the rotated face")

    up_box, up_marks = geometry(upright_detection)
    rot_box, rot_marks = geometry(rotated_detection)
    if up_marks is None or rot_marks is None:
        pytest.skip("no landmarks available")

    upright = align_face_for_recognition(portrait, box=up_box, landmarks=up_marks).crop
    rot_aligned = align_face_for_recognition(
        rotated, box=rot_box, landmarks=rot_marks
    ).crop
    rot_boxed = align_face_for_recognition(rotated, box=rot_box, landmarks=None).crop

    vectors = service._embedder.embed_aligned(  # noqa: SLF001
        [upright, rot_aligned, rot_boxed]
    )
    with_alignment = cosine_similarity(vectors[0][0], vectors[1][0])
    without_alignment = cosine_similarity(vectors[0][0], vectors[2][0])

    assert with_alignment > 0.90
    assert with_alignment > without_alignment + 0.20


def test_residuals_stay_in_the_measured_band(
    service, detector, portrait
) -> None:  # noqa: ANN001
    """Measured 0.034-0.061 across every degradation of a detected face. The
    configured `good` anchor sits at 0.06, so a drift here would start
    flagging healthy captures as low-confidence."""
    detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    box, marks = geometry(detection)
    if marks is None:
        pytest.skip("no landmarks available")

    embedding = service.embed_to_vector(
        portrait, role=ImageRole.LIVE_SELFIE, box=box, landmarks=marks
    )
    assert embedding.alignment_residual is not None
    assert 0.0 <= embedding.alignment_residual < 0.15
    assert embedding.aligned is True
    assert embedding.confidence > 0.8


def test_the_box_fallback_is_flagged_low_confidence(
    service, detector, portrait
) -> None:  # noqa: ANN001
    detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    box, _ = geometry(detection)

    result = service.embed(
        portrait, role=ImageRole.LIVE_SELFIE, box=box, landmarks=None
    )

    assert result.success is True
    assert result.aligned is False
    assert result.low_confidence is True
    assert result.trustworthy is False
    assert any(w.code == "EMBEDDING_BOX_ALIGNED" for w in result.warnings)


# --------------------------------------------------------------------------- #
# The service surface
# --------------------------------------------------------------------------- #


def test_the_vector_is_withheld_unless_requested(
    service, detector, portrait
) -> None:  # noqa: ANN001
    """An embedding is biometric data, not a diagnostic."""
    detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    withheld = service.embed(portrait, detection=detection)
    included = service.embed(portrait, detection=detection, include_vector=True)

    assert withheld.vector is None
    assert included.vector is not None
    assert len(included.vector) == 512


def test_the_summary_never_contains_the_vector(
    service, detector, portrait
) -> None:  # noqa: ANN001
    detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    summary = service.embed(portrait, detection=detection, include_vector=True).summary()
    assert "vector" not in summary


def test_batching_returns_results_in_order(
    service, detector, portrait, impostor
) -> None:  # noqa: ANN001
    requests = []
    for image, role in (
        (portrait, ImageRole.LIVE_SELFIE),
        (impostor, ImageRole.PROFILE_IMAGE),
        (cv2.GaussianBlur(portrait, (5, 5), 0), ImageRole.SECONDARY_IMAGE),
    ):
        detection = detector.detect(image, role=role)
        box, marks = geometry(detection)
        requests.append(
            EmbeddingRequest(image=image, box=box, landmarks=marks, role=role)
        )

    batch = service.embed_many(requests)

    assert len(batch.results) == 3
    assert [r.role for r in batch.results] == [
        ImageRole.LIVE_SELFIE,
        ImageRole.PROFILE_IMAGE,
        ImageRole.SECONDARY_IMAGE,
    ]
    assert batch.succeeded == 3


def test_batching_agrees_with_one_at_a_time(
    service, detector, portrait
) -> None:  # noqa: ANN001
    """A batched forward pass must produce the same vectors as separate ones,
    or a cached and an uncached comparison would disagree."""
    detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    box, marks = geometry(detection)
    if marks is None:
        pytest.skip("no landmarks available")

    crops = [
        align_face_for_recognition(portrait, box=box, landmarks=marks).crop,
        align_face_for_recognition(
            cv2.GaussianBlur(portrait, (5, 5), 0), box=box, landmarks=marks
        ).crop,
    ]

    batched = service._embedder.embed_aligned(crops)  # noqa: SLF001
    singly = [service._embedder.embed_aligned([crop])[0] for crop in crops]  # noqa: SLF001

    for (batch_vec, _), (single_vec, _) in zip(batched, singly, strict=True):
        assert cosine_similarity(batch_vec, single_vec) > 0.9999


def test_a_failure_is_isolated_to_its_own_slot(
    service, detector, portrait
) -> None:  # noqa: ANN001
    """One bad face must not cost the other six."""
    detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    box, marks = geometry(detection)

    requests = [
        EmbeddingRequest(
            image=portrait, box=box, landmarks=marks, role=ImageRole.LIVE_SELFIE
        ),
        # Neither landmarks nor a box: nothing to align from.
        EmbeddingRequest(image=portrait, box=None, landmarks=None,
                         role=ImageRole.SECONDARY_IMAGE),
        EmbeddingRequest(
            image=portrait, box=box, landmarks=marks, role=ImageRole.PROFILE_IMAGE
        ),
    ]

    batch = service.embed_many(requests)

    assert batch.results[0].success is True
    assert batch.results[1].success is False
    assert batch.results[1].error_code is not None
    assert batch.results[2].success is True


def test_the_cache_returns_the_identical_vector(
    service, detector, portrait
) -> None:  # noqa: ANN001
    """A cached and an uncached comparison must never disagree."""
    service._cache.clear()  # noqa: SLF001
    detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    box, marks = geometry(detection)

    cold = service.embed_to_vector(portrait, box=box, landmarks=marks)
    warm = service.embed_to_vector(portrait, box=box, landmarks=marks)

    assert cold.cache_hit is False
    assert warm.cache_hit is True
    np.testing.assert_array_equal(cold.vector, warm.vector)
    assert cold.raw_norm == pytest.approx(warm.raw_norm)


def test_the_cache_makes_a_repeat_dramatically_cheaper(
    service, detector, portrait
) -> None:  # noqa: ANN001
    """Measured 591 ms cold against 2 ms warm for a three-face batch."""
    detection = detector.detect(portrait, role=ImageRole.LIVE_SELFIE)
    box, marks = geometry(detection)
    requests = [
        EmbeddingRequest(image=portrait, box=box, landmarks=marks,
                         role=ImageRole.LIVE_SELFIE)
    ]

    service._cache.clear()  # noqa: SLF001
    cold = service.embed_many(requests)
    warm = service.embed_many(requests)

    assert cold.forward_passes == 1
    assert cold.cache_hits == 0
    assert warm.forward_passes == 0
    assert warm.cache_hits == 1
    assert warm.total_duration_ms < cold.total_duration_ms


def test_describe_reports_the_active_configuration(service) -> None:  # noqa: ANN001
    import json

    described = service.describe()
    assert described["dimension"] == 512
    assert described["model_version"]
    assert "cache" in described
    json.dumps(described)
