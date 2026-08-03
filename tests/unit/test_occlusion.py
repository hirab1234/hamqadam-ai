"""Unit tests for the occlusion analyser.

Every test works on a synthetic face with known landmarks, then applies a
controlled perturbation - a flat rectangle over a region, a global darkening -
and asserts that the analyser's response moves in the direction physics
demands. That is the right level of assertion for an estimator: exact
probabilities are not contractual, but "covering both eyes must raise the eye
regions' occlusion probability" is.
"""

from __future__ import annotations

import numpy as np
import pytest

from hamqadam_ai.core.config import OcclusionConfig
from hamqadam_ai.core.constants import FaceRegion
from hamqadam_ai.detectors.occlusion import (
    CANONICAL_SIZE,
    OcclusionAnalyzer,
    OcclusionResult,
)
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5
from tests.conftest import SyntheticFace, make_synthetic_face, occlude_rectangle


@pytest.fixture
def analyzer() -> OcclusionAnalyzer:
    """Analyser with the default configured thresholds."""
    return OcclusionAnalyzer(OcclusionConfig())


@pytest.fixture
def clean_face() -> SyntheticFace:
    """A frontal, unobstructed synthetic face."""
    return make_synthetic_face(image_size=512, distance=380.0)


# --------------------------------------------------------------------------- #
# Baseline behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_clean_face_is_not_occluded(
    analyzer: OcclusionAnalyzer, clean_face: SyntheticFace
) -> None:
    """An unobstructed face must not be flagged."""
    result = analyzer.analyze(clean_face.image, clean_face.box, clean_face.landmarks)

    assert result.method == "geometric"
    assert not result.occluded
    assert result.overall_score < OcclusionConfig().overall_threshold


@pytest.mark.unit
def test_all_six_regions_are_measured(
    analyzer: OcclusionAnalyzer, clean_face: SyntheticFace
) -> None:
    """Every anatomical region must appear in the report."""
    result = analyzer.analyze(clean_face.image, clean_face.box, clean_face.landmarks)

    assert set(result.regions) == {str(region) for region in FaceRegion}


@pytest.mark.unit
def test_scores_are_bounded(
    analyzer: OcclusionAnalyzer, clean_face: SyntheticFace
) -> None:
    """Probabilities and coverage fractions must stay in [0, 1]."""
    result = analyzer.analyze(clean_face.image, clean_face.box, clean_face.landmarks)

    assert 0.0 <= result.overall_score <= 1.0
    assert 0.0 <= result.symmetry_delta <= 1.0
    for evidence in result.regions.values():
        assert 0.0 <= evidence.probability <= 1.0
        assert 0.0 <= evidence.skin_coverage <= 1.0
        assert 0.0 <= evidence.colour_uniformity <= 1.0
        assert evidence.texture_energy >= 0.0


# --------------------------------------------------------------------------- #
# Detecting real obstructions
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_sunglasses_raise_eye_region_occlusion(
    analyzer: OcclusionAnalyzer, clean_face: SyntheticFace
) -> None:
    """A dark flat bar across both eyes must raise both eye probabilities."""
    interocular = clean_face.landmarks.interocular_distance
    centre_x, centre_y = clean_face.landmarks.eye_center
    obstructed = occlude_rectangle(
        clean_face.image,
        (centre_x, centre_y),
        (int(interocular * 2.1), int(interocular * 0.75)),
        colour=(25, 25, 25),
    )

    before = analyzer.analyze(clean_face.image, clean_face.box, clean_face.landmarks)
    after = analyzer.analyze(obstructed, clean_face.box, clean_face.landmarks)

    for region in (FaceRegion.LEFT_EYE, FaceRegion.RIGHT_EYE):
        key = str(region)
        assert after.regions[key].probability > before.regions[key].probability
        assert after.regions[key].occluded, f"{key} should be flagged"

    assert after.overall_score > before.overall_score
    assert after.occluded


@pytest.mark.unit
def test_surgical_mask_raises_lower_face_occlusion(
    analyzer: OcclusionAnalyzer, clean_face: SyntheticFace
) -> None:
    """A flat covering over nose, mouth and chin must flag the lower face."""
    interocular = clean_face.landmarks.interocular_distance
    mouth_x, mouth_y = clean_face.landmarks.mouth_center
    obstructed = occlude_rectangle(
        clean_face.image,
        (mouth_x, mouth_y),
        (int(interocular * 2.2), int(interocular * 1.8)),
        colour=(205, 200, 190),
    )

    result = analyzer.analyze(obstructed, clean_face.box, clean_face.landmarks)

    assert result.regions[str(FaceRegion.MOUTH)].occluded
    assert result.occluded
    assert str(FaceRegion.MOUTH) in result.occluded_regions


@pytest.mark.unit
def test_one_sided_occlusion_raises_symmetry_delta(
    analyzer: OcclusionAnalyzer, clean_face: SyntheticFace
) -> None:
    """A hand over one eye is the case per-region averages alone would miss."""
    interocular = clean_face.landmarks.interocular_distance
    obstructed = occlude_rectangle(
        clean_face.image,
        clean_face.landmarks.left_eye,
        (int(interocular * 0.85), int(interocular * 0.85)),
        colour=(150, 170, 200),
    )

    before = analyzer.analyze(clean_face.image, clean_face.box, clean_face.landmarks)
    after = analyzer.analyze(obstructed, clean_face.box, clean_face.landmarks)

    assert after.symmetry_delta > before.symmetry_delta


@pytest.mark.unit
def test_flat_covering_lowers_texture_energy(
    analyzer: OcclusionAnalyzer, clean_face: SyntheticFace
) -> None:
    """The texture signal must respond to flatness, which is what it measures."""
    interocular = clean_face.landmarks.interocular_distance
    obstructed = occlude_rectangle(
        clean_face.image,
        clean_face.landmarks.mouth_center,
        (int(interocular * 1.6), int(interocular * 0.9)),
        colour=(160, 180, 200),
    )

    before = analyzer.analyze(clean_face.image, clean_face.box, clean_face.landmarks)
    after = analyzer.analyze(obstructed, clean_face.box, clean_face.landmarks)

    key = str(FaceRegion.MOUTH)
    assert after.regions[key].texture_energy < before.regions[key].texture_energy


@pytest.mark.unit
def test_non_skin_covering_lowers_skin_coverage(
    analyzer: OcclusionAnalyzer, clean_face: SyntheticFace
) -> None:
    """A blue covering must drop skin coverage in the region it covers."""
    interocular = clean_face.landmarks.interocular_distance
    obstructed = occlude_rectangle(
        clean_face.image,
        clean_face.landmarks.nose_tip,
        (int(interocular * 0.9), int(interocular * 0.9)),
        colour=(200, 60, 20),
    )

    before = analyzer.analyze(clean_face.image, clean_face.box, clean_face.landmarks)
    after = analyzer.analyze(obstructed, clean_face.box, clean_face.landmarks)

    key = str(FaceRegion.NOSE)
    assert after.regions[key].skin_coverage < before.regions[key].skin_coverage


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_missing_landmarks_returns_unavailable(analyzer: OcclusionAnalyzer) -> None:
    """Without landmarks the analyser must say so rather than guess."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    result = analyzer.analyze(image, BoundingBox(10, 10, 150, 190), None)

    assert result.method == "unavailable"
    assert not result.occluded
    assert result.overall_score == 0.0
    assert result.regions == {}


@pytest.mark.unit
def test_collinear_landmarks_return_unavailable(
    analyzer: OcclusionAnalyzer, clean_face: SyntheticFace
) -> None:
    """A degenerate landmark set makes the canonical warp singular."""
    collinear = Landmarks5(
        np.array(
            [[100, 100], [200, 100], [150, 100], [120, 100], [180, 100]],
            dtype=np.float32,
        )
    )
    result = analyzer.analyze(clean_face.image, clean_face.box, collinear)

    assert result.method == "unavailable"


@pytest.mark.unit
def test_analysis_is_invariant_to_head_roll(analyzer: OcclusionAnalyzer) -> None:
    """The canonical warp exists to make roll irrelevant. Verify that it does."""
    upright = make_synthetic_face(roll=0.0, image_size=512, distance=380.0)
    tilted = make_synthetic_face(roll=22.0, image_size=512, distance=380.0)

    a = analyzer.analyze(upright.image, upright.box, upright.landmarks)
    b = analyzer.analyze(tilted.image, tilted.box, tilted.landmarks)

    assert a.occluded == b.occluded
    assert b.overall_score == pytest.approx(a.overall_score, abs=0.15)


@pytest.mark.unit
def test_analysis_is_invariant_to_face_scale(analyzer: OcclusionAnalyzer) -> None:
    """A face near or far must score comparably; the warp normalises scale."""
    near = make_synthetic_face(image_size=512, distance=320.0)
    far = make_synthetic_face(image_size=512, distance=520.0)

    a = analyzer.analyze(near.image, near.box, near.landmarks)
    b = analyzer.analyze(far.image, far.box, far.landmarks)

    assert a.occluded == b.occluded
    assert b.overall_score == pytest.approx(a.overall_score, abs=0.15)


@pytest.mark.unit
def test_forehead_covering_is_damped(
    analyzer: OcclusionAnalyzer, clean_face: SyntheticFace
) -> None:
    """Hair and headscarves are normal, so a covered forehead alone must not
    flag the whole face as obstructed."""
    interocular = clean_face.landmarks.interocular_distance
    eye_x, eye_y = clean_face.landmarks.eye_center
    obstructed = occlude_rectangle(
        clean_face.image,
        (eye_x, eye_y - interocular * 0.95),
        (int(interocular * 2.4), int(interocular * 1.0)),
        colour=(45, 40, 38),
    )

    result = analyzer.analyze(obstructed, clean_face.box, clean_face.landmarks)

    assert not result.occluded


@pytest.mark.unit
def test_unavailable_factory_is_neutral() -> None:
    """The sentinel result must not imply a clean face."""
    result = OcclusionResult.unavailable()

    assert result.method == "unavailable"
    assert result.overall_score == 0.0
    assert result.occluded_regions == []


@pytest.mark.unit
def test_canonical_frame_size_is_consistent() -> None:
    """Region boxes must fit inside the canonical frame they index into."""
    from hamqadam_ai.detectors.occlusion import _REGION_BOXES  # noqa: PLC0415

    for region, (x1, y1, x2, y2) in _REGION_BOXES.items():
        assert 0 <= x1 < x2 <= CANONICAL_SIZE, region
        assert 0 <= y1 < y2 <= CANONICAL_SIZE, region
