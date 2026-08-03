"""The eight quality analysers, tested against controlled degradations.

Every test here works the same way: take an image, apply *one* known defect,
and assert the responsible metric moves in the right direction while the others
stay put. That cross-check matters as much as the direction - a blur metric
that also fires on darkness is not measuring blur.

Images are synthetic where the property under test can be constructed exactly
(a known noise sigma, a known block pattern) and the shared procedural face
fixture where facial geometry is needed. No real identity document ever enters
the repository.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.quality.artifacts import (
    DistortionAnalyzer,
    PixelationAnalyzer,
    landmark_anisotropy,
)
from hamqadam_ai.quality.base import (
    QualityContext,
    percentile_range,
    shannon_entropy,
)
from hamqadam_ai.quality.blur import (
    BlurAnalyzer,
    SharpnessAnalyzer,
    laplacian_variance,
    motion_blur_anisotropy,
    normalised_laplacian,
    tenengrad,
    worst_axis_laplacian,
    worst_axis_tenengrad,
)
from hamqadam_ai.quality.exposure import (
    BrightnessAnalyzer,
    ContrastAnalyzer,
    clipping_fractions,
    illumination_uniformity,
)
from hamqadam_ai.quality.noise import (
    NoiseAnalyzer,
    estimate_noise_sigma,
    flat_block_sigma,
    immerkaer_sigma,
    local_contrast,
)
from hamqadam_ai.quality.resolution import ResolutionAnalyzer
from hamqadam_ai.quality.spectral import (
    banding_ratio,
    blockiness,
    chromatic_aberration,
    radial_power_spectrum,
)
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

BgrImage = npt.NDArray[np.uint8]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def textured(width: int = 512, height: int = 512, seed: int = 11) -> BgrImage:
    """A broadband test image with detail at every spatial frequency."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (height, width), dtype=np.uint8)
    # Layer smooth structure under the noise so the image has both low- and
    # high-frequency content, like a real photograph.
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    smooth = (
        128
        + 60 * np.sin(x / 40.0)
        + 40 * np.cos(y / 25.0)
    )
    blended = np.clip(0.55 * smooth + 0.45 * base, 0, 255).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)


def context_for(image: BgrImage, **kwargs: object) -> QualityContext:
    """Build a context with the real configured sizes."""
    config = get_settings().quality
    return QualityContext(
        image=image,
        canonical_size=config.canonical_face_size,
        crop_margin=config.face_crop_margin,
        analysis_long_side=config.global_analysis_long_side,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def quality_config():  # noqa: ANN201 - pytest fixture
    """The real quality configuration."""
    return get_settings().quality


# --------------------------------------------------------------------------- #
# Focus primitives
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_laplacian_variance_falls_with_blur() -> None:
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    values = [
        laplacian_variance(cv2.GaussianBlur(gray, (k, k), 0)) if k else laplacian_variance(gray)
        for k in (0, 3, 9, 21)
    ]
    assert values == sorted(values, reverse=True)


@pytest.mark.unit
def test_tenengrad_falls_with_blur() -> None:
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    sharp = tenengrad(gray)
    blurred = tenengrad(cv2.GaussianBlur(gray, (11, 11), 0))
    assert blurred < sharp * 0.5


@pytest.mark.unit
def test_focus_measures_are_zero_on_an_empty_array() -> None:
    empty = np.zeros((0, 0), dtype=np.uint8)
    assert laplacian_variance(empty) == 0.0
    assert tenengrad(empty) == 0.0
    assert normalised_laplacian(empty) == 0.0
    assert motion_blur_anisotropy(empty) == 0.0


@pytest.mark.unit
def test_normalised_laplacian_removes_the_contrast_dependence() -> None:
    """Halving contrast quarters the raw Laplacian but should barely move the
    normalised one."""
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    dim = np.clip(gray.astype(np.float32) * 0.5 + 64, 0, 255).astype(np.uint8)

    raw_ratio = laplacian_variance(dim) / laplacian_variance(gray)
    normalised_ratio = normalised_laplacian(dim) / normalised_laplacian(gray)

    assert raw_ratio < 0.4
    assert normalised_ratio == pytest.approx(1.0, abs=0.15)


@pytest.mark.unit
def test_motion_blur_is_reported_as_anisotropic() -> None:
    """Directional smear versus isotropic defocus - different remedies."""
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)

    kernel = np.zeros((21, 21), dtype=np.float32)
    kernel[10, :] = 1.0 / 21.0
    horizontal = cv2.filter2D(gray, -1, kernel)
    defocus = cv2.GaussianBlur(gray, (21, 21), 0)

    assert motion_blur_anisotropy(horizontal) > 0.35
    assert motion_blur_anisotropy(defocus) < 0.15


# --------------------------------------------------------------------------- #
# BlurAnalyzer
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_blur_score_falls_monotonically_with_blur(quality_config) -> None:
    analyser = BlurAnalyzer(quality_config.blur)
    base = textured()
    scores = [
        analyser.analyse(
            context_for(cv2.GaussianBlur(base, (k, k), 0) if k else base)
        ).score
        for k in (0, 3, 9, 21)
    ]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0.85
    assert scores[-1] < 0.25


@pytest.mark.unit
def test_blur_reports_every_raw_measurement(quality_config) -> None:
    result = BlurAnalyzer(quality_config.blur).analyse(context_for(textured()))
    assert {
        "laplacian_variance",
        "tenengrad",
        "high_frequency_ratio",
        "motion_anisotropy",
    } <= set(result.measurements)
    assert result.limiting_factor in {"laplacian", "tenengrad", "spectral"}


@pytest.mark.unit
def test_blur_notes_directional_motion(quality_config) -> None:
    """The note fires only when the image is *both* soft and directional, so
    the test needs a smear heavy enough to trip the score gate too."""
    kernel = np.zeros((41, 41), dtype=np.float32)
    kernel[20, :] = 1.0 / 41.0
    smeared = cv2.filter2D(textured(), -1, kernel)

    result = BlurAnalyzer(quality_config.blur).analyse(context_for(smeared))
    assert result.score < 0.5
    assert result.measurements["motion_anisotropy"] > 0.35
    assert result.note is not None
    assert "motion" in result.note.lower()


@pytest.mark.unit
def test_worst_axis_tenengrad_exposes_directional_blur() -> None:
    """The isotropic measure hides half of a one-directional smear.

    Measured on the test image, a 41 px horizontal blur leaves the *summed*
    Tenengrad at a healthy level because every vertical edge survives, while
    the weaker axis collapses. Scoring the weaker axis removes the blind spot.
    """
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    kernel = np.zeros((41, 41), dtype=np.float32)
    kernel[20, :] = 1.0 / 41.0
    smeared = cv2.filter2D(gray, -1, kernel)

    isotropic_ratio = tenengrad(smeared) / tenengrad(gray)
    worst_axis_ratio = worst_axis_tenengrad(smeared) / worst_axis_tenengrad(gray)

    assert worst_axis_ratio < isotropic_ratio / 2.0


@pytest.mark.unit
def test_worst_axis_matches_the_isotropic_measure_when_sharp() -> None:
    """On isotropic content the correction must be close to a no-op."""
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    assert worst_axis_tenengrad(gray) == pytest.approx(tenengrad(gray), rel=0.35)
    assert worst_axis_laplacian(gray) == pytest.approx(
        laplacian_variance(gray), rel=0.35
    )


@pytest.mark.unit
def test_worst_axis_laplacian_collapses_on_directional_blur() -> None:
    """Measured: 2665 isotropic against 38 on the weaker axis."""
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    kernel = np.zeros((41, 41), dtype=np.float32)
    kernel[20, :] = 1.0 / 41.0
    smeared = cv2.filter2D(gray, -1, kernel)

    assert worst_axis_laplacian(smeared) < laplacian_variance(smeared) / 20.0


@pytest.mark.unit
def test_worst_axis_laplacian_tracks_isotropic_defocus() -> None:
    """Defocus removes both axes equally, so the correction must not fire."""
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    defocused = cv2.GaussianBlur(gray, (9, 9), 0)
    assert worst_axis_laplacian(defocused) == pytest.approx(
        laplacian_variance(defocused), rel=0.40
    )


# --------------------------------------------------------------------------- #
# SharpnessAnalyzer
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_sharpness_is_unmeasured_without_a_face(quality_config) -> None:
    """It is a face metric. Reporting a number without a face would be a
    fabrication, so it is excluded from the composite instead."""
    result = SharpnessAnalyzer(quality_config.sharpness).analyse(
        context_for(textured())
    )
    assert result.measured is False
    assert result.note is not None


@pytest.mark.unit
def test_sharpness_falls_with_face_blur(quality_config, synthetic_face) -> None:
    analyser = SharpnessAnalyzer(quality_config.sharpness)
    scores = []
    for kernel in (0, 5, 15):
        image = (
            cv2.GaussianBlur(synthetic_face.image, (kernel, kernel), 0)
            if kernel
            else synthetic_face.image
        )
        scores.append(
            analyser.analyse(
                context_for(
                    image,
                    face_box=synthetic_face.box,
                    landmarks=synthetic_face.landmarks,
                )
            ).score
        )
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_sharpness_detects_a_blurred_face_in_a_sharp_scene(
    quality_config, synthetic_face
) -> None:
    """The failure global blur cannot see: subject motion.

    The scene stays pin sharp, only the face is smeared.
    """
    height, width = synthetic_face.image.shape[:2]
    x1, y1, x2, y2 = synthetic_face.box.as_int_tuple()
    x1, y1 = max(0, x1), max(0, y1)

    # The fixture draws its face on flat grey. Against a featureless
    # background any face out-textures the scene and the ratio caps at 1.0, so
    # a textured backdrop is required for the comparison to mean anything.
    backdrop = textured(width, height, seed=44)
    sharp = backdrop.copy()
    sharp[y1:y2, x1:x2] = synthetic_face.image[y1:y2, x1:x2]

    blurred_face = sharp.copy()
    blurred_face[y1:y2, x1:x2] = cv2.GaussianBlur(sharp[y1:y2, x1:x2], (21, 21), 0)

    analyser = SharpnessAnalyzer(quality_config.sharpness)
    reference = analyser.analyse(
        context_for(sharp, face_box=synthetic_face.box, landmarks=synthetic_face.landmarks)
    )
    degraded = analyser.analyse(
        context_for(
            blurred_face,
            face_box=synthetic_face.box,
            landmarks=synthetic_face.landmarks,
        )
    )

    assert degraded.score < reference.score
    ratio = degraded.measurements.get("face_vs_scene_ratio")
    assert ratio is not None, "a textured backdrop should make the ratio computable"
    assert ratio < reference.measurements["face_vs_scene_ratio"]


@pytest.mark.unit
def test_sharpness_warns_when_the_crop_was_upsampled(quality_config) -> None:
    """A face smaller than the analysis size is interpolated, which inflates
    apparent sharpness. The caller must be told."""
    small = textured(120, 120)
    context = context_for(small, face_box=BoundingBox(10, 10, 70, 70))
    assert context.canonical_upsampled is True

    result = SharpnessAnalyzer(quality_config.sharpness).analyse(context)
    assert result.note is not None
    assert "upscaled" in result.note.lower() or "interpolat" in result.note.lower()


# --------------------------------------------------------------------------- #
# Exposure
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_clipping_fractions_count_crushed_and_blown_pixels() -> None:
    gray = np.full((100, 100), 128, dtype=np.uint8)
    gray[:10, :] = 0
    gray[90:, :] = 255
    shadows, highlights = clipping_fractions(gray)
    assert shadows == pytest.approx(0.10)
    assert highlights == pytest.approx(0.10)


@pytest.mark.unit
def test_brightness_band_rejects_both_extremes(quality_config) -> None:
    analyser = BrightnessAnalyzer(quality_config.brightness)
    base = textured()

    good = analyser.analyse(context_for(base)).score
    dark = analyser.analyse(
        context_for(np.clip(base * 0.2, 0, 255).astype(np.uint8))
    ).score
    bright = analyser.analyse(
        context_for(np.clip(base.astype(np.float32) * 1.4 + 90, 0, 255).astype(np.uint8))
    ).score

    assert good > dark
    assert good > bright


@pytest.mark.unit
def test_brightness_notes_underexposure(quality_config) -> None:
    dark = np.clip(textured() * 0.18, 0, 255).astype(np.uint8)
    result = BrightnessAnalyzer(quality_config.brightness).analyse(context_for(dark))
    assert result.score < 0.7
    assert result.note is not None
    assert "dark" in result.note.lower() or "underexposed" in result.note.lower()


@pytest.mark.unit
def test_illumination_uniformity_detects_side_lighting() -> None:
    """Strong side light gives a healthy mean and a healthy std while being
    genuinely hard to recognise, so it needs its own signal."""
    even = np.full((120, 120), 140, dtype=np.uint8)
    gradient = np.tile(
        np.linspace(20, 240, 120, dtype=np.uint8)[None, :], (120, 1)
    )
    assert illumination_uniformity(even) > 0.95
    assert illumination_uniformity(gradient) < 0.4


@pytest.mark.unit
def test_contrast_falls_when_the_image_is_flattened(quality_config) -> None:
    analyser = ContrastAnalyzer(quality_config.contrast)
    base = textured()
    flat = np.clip(base.astype(np.float32) * 0.22 + 110, 0, 255).astype(np.uint8)

    assert analyser.analyse(context_for(base)).score > analyser.analyse(
        context_for(flat)
    ).score


@pytest.mark.unit
def test_entropy_falls_under_posterisation() -> None:
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    posterised = ((gray // 42) * 42).astype(np.uint8)
    assert shannon_entropy(posterised) < shannon_entropy(gray) - 1.0


@pytest.mark.unit
def test_percentile_range_ignores_outliers() -> None:
    """One hot pixel must not report a full 0-255 range on a flat image."""
    gray = np.full((100, 100), 128, dtype=np.uint8)
    gray[0, 0] = 0
    gray[0, 1] = 255
    assert percentile_range(gray) < 5.0


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("sigma", [3.0, 8.0, 15.0])
def test_immerkaer_recovers_a_known_sigma_on_a_smooth_image(sigma: float) -> None:
    """Unbiased where there is no texture to confuse it."""
    rng = np.random.default_rng(3)
    smooth = np.full((320, 320), 128.0, dtype=np.float32)
    noisy = np.clip(smooth + rng.normal(0.0, sigma, smooth.shape), 0, 255).astype(
        np.uint8
    )
    assert immerkaer_sigma(noisy) == pytest.approx(sigma, rel=0.20)


@pytest.mark.unit
def test_flat_block_estimator_beats_immerkaer_on_a_textured_image() -> None:
    """Texture can only inflate a noise estimate, never depress it - which is
    why the analyser takes the minimum of the two."""
    rng = np.random.default_rng(5)
    gray = cv2.cvtColor(textured(seed=21), cv2.COLOR_BGR2GRAY)
    sigma = 4.0
    noisy = np.clip(
        gray.astype(np.float32) + rng.normal(0.0, sigma, gray.shape), 0, 255
    ).astype(np.uint8)

    global_estimate = immerkaer_sigma(noisy)
    robust_estimate = flat_block_sigma(noisy)

    assert global_estimate > sigma * 1.5, "texture should inflate the global estimate"
    assert robust_estimate < global_estimate


@pytest.mark.unit
def test_estimate_noise_sigma_returns_the_minimum() -> None:
    gray = cv2.cvtColor(textured(seed=31), cv2.COLOR_BGR2GRAY)
    chosen, global_estimate, robust = estimate_noise_sigma(gray)
    assert chosen == min(global_estimate, robust)


@pytest.mark.unit
def test_noise_score_falls_as_noise_rises(quality_config) -> None:
    analyser = NoiseAnalyzer(quality_config.noise)
    rng = np.random.default_rng(9)
    base = textured()

    scores = []
    for sigma in (0.0, 6.0, 18.0):
        noisy = (
            np.clip(base.astype(np.float32) + rng.normal(0.0, sigma, base.shape), 0, 255)
            .astype(np.uint8)
            if sigma
            else base
        )
        scores.append(analyser.analyse(context_for(noisy)).score)

    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_local_contrast_is_zero_on_a_flat_patch() -> None:
    assert local_contrast(np.full((64, 64), 100, dtype=np.uint8)) == pytest.approx(
        0.0, abs=1e-3
    )


# --------------------------------------------------------------------------- #
# Spectral primitives
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_high_frequency_energy_falls_with_blur() -> None:
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    sharp = radial_power_spectrum(gray).energy_above(0.25)
    blurred = radial_power_spectrum(cv2.GaussianBlur(gray, (11, 11), 0)).energy_above(
        0.25
    )
    assert blurred < sharp * 0.5


@pytest.mark.unit
def test_radial_profile_is_normalised_and_ordered() -> None:
    spectrum = radial_power_spectrum(cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY))
    assert spectrum.normalised_radii[0] == 0.0
    assert spectrum.normalised_radii[-1] == pytest.approx(1.0)
    assert spectrum.profile.size == spectrum.normalised_radii.size


@pytest.mark.unit
def test_energy_above_is_monotone_in_the_cutoff() -> None:
    spectrum = radial_power_spectrum(cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY))
    values = [spectrum.energy_above(c) for c in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert values == sorted(values, reverse=True)


# --------------------------------------------------------------------------- #
# Blockiness
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_blockiness_fires_on_a_synthetic_block_pattern() -> None:
    pattern = np.tile(
        np.repeat(np.array([40, 200], dtype=np.uint8), 8), (256, 16)
    )[:256, :256]
    assert blockiness(pattern) > 1.0


@pytest.mark.unit
@pytest.mark.parametrize("shift", [0, 1, 3, 5, 7])
def test_blockiness_is_phase_invariant(shift: int) -> None:
    """The correctness requirement, not a refinement.

    A face crop taken at an arbitrary offset shifts the codec grid by
    ``offset % 8``. An implementation that only tests phase 0 reports a
    severely blocked JPEG as clean seven times out of eight.
    """
    pattern = np.tile(
        np.repeat(np.array([40, 200], dtype=np.uint8), 8), (256, 16)
    )[:256, :256]
    shifted = np.roll(pattern, shift, axis=1)
    assert blockiness(shifted) == pytest.approx(blockiness(pattern), rel=0.05)


@pytest.mark.unit
def test_blockiness_is_low_on_a_smooth_image() -> None:
    rng = np.random.default_rng(0)
    smooth = cv2.GaussianBlur(
        rng.integers(0, 256, (256, 256), dtype=np.uint8), (9, 9), 0
    )
    assert blockiness(smooth) < 0.1


@pytest.mark.unit
def test_blockiness_rises_as_jpeg_quality_falls() -> None:
    base = textured(256, 256)
    values = []
    for quality in (95, 50, 20, 8):
        ok, buffer = cv2.imencode(
            ".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        assert ok
        decoded = cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        values.append(blockiness(decoded))
    assert values[-1] > values[0]


@pytest.mark.unit
def test_blockiness_ignores_a_too_small_image() -> None:
    assert blockiness(np.zeros((10, 10), dtype=np.uint8)) == 0.0


# --------------------------------------------------------------------------- #
# Banding and chromatic aberration
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_banding_fires_on_posterisation() -> None:
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    posterised = ((gray // 40) * 40).astype(np.uint8)
    assert banding_ratio(posterised) > banding_ratio(gray) + 0.5


@pytest.mark.unit
def test_banding_does_not_fire_on_a_legitimately_dark_image() -> None:
    """A low-key photograph uses few levels, but contiguous ones."""
    gray = cv2.cvtColor(textured(), cv2.COLOR_BGR2GRAY)
    dark = (gray.astype(np.float32) * 0.25).astype(np.uint8)
    assert banding_ratio(dark) < 0.3


@pytest.mark.unit
def test_chromatic_aberration_fires_on_channel_misalignment() -> None:
    """Lateral CA displaces red relative to blue, fringing only at edges."""
    base = textured(256, 256)
    fringed = base.copy()
    fringed[:, :, 2] = np.roll(base[:, :, 2], 3, axis=1)

    assert chromatic_aberration(fringed) > chromatic_aberration(base)


@pytest.mark.unit
def test_chromatic_aberration_ignores_a_flat_colour_cast() -> None:
    """A red jumper against a blue wall has a huge R-B difference everywhere,
    not just at edges - that is subject colour, not an optical artefact."""
    base = textured(256, 256)
    tinted = base.copy().astype(np.int16)
    tinted[:, :, 2] = np.clip(tinted[:, :, 2] + 45, 0, 255)
    tinted[:, :, 0] = np.clip(tinted[:, :, 0] - 45, 0, 255)

    assert chromatic_aberration(tinted.astype(np.uint8)) == pytest.approx(
        chromatic_aberration(base), abs=2.0
    )


@pytest.mark.unit
def test_chromatic_aberration_of_a_grayscale_array_is_zero() -> None:
    assert chromatic_aberration(np.zeros((64, 64), dtype=np.uint8)) == 0.0


# --------------------------------------------------------------------------- #
# Geometric anisotropy
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_anisotropy_is_near_zero_for_the_canonical_template() -> None:
    from hamqadam_ai.core.constants import CANONICAL_FACE_2D_UNIT

    template = Landmarks5((CANONICAL_FACE_2D_UNIT * 200.0).astype(np.float32))
    anisotropy, major, minor = landmark_anisotropy(template)
    assert anisotropy < 0.02
    assert major == pytest.approx(minor, rel=0.02)


@pytest.mark.unit
@pytest.mark.parametrize("stretch", [1.3, 1.6, 2.0])
def test_anisotropy_grows_with_horizontal_stretch(stretch: float) -> None:
    from hamqadam_ai.core.constants import CANONICAL_FACE_2D_UNIT

    points = (CANONICAL_FACE_2D_UNIT * 200.0).astype(np.float32).copy()
    points[:, 0] *= stretch
    anisotropy, _, _ = landmark_anisotropy(Landmarks5(points))
    assert anisotropy == pytest.approx(np.log(stretch), abs=0.05)


@pytest.mark.unit
def test_anisotropy_is_invariant_to_uniform_scale_and_rotation() -> None:
    """Only *non-uniform* scaling is distortion. A bigger or tilted face is not."""
    from hamqadam_ai.core.constants import CANONICAL_FACE_2D_UNIT

    points = (CANONICAL_FACE_2D_UNIT * 200.0).astype(np.float32)
    angle = np.radians(23.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    transformed = (points * 2.4) @ rotation.T + np.array([50.0, 90.0], dtype=np.float32)

    anisotropy, _, _ = landmark_anisotropy(Landmarks5(transformed.astype(np.float32)))
    assert anisotropy < 0.03


@pytest.mark.unit
def test_distortion_skips_anisotropy_on_a_turned_head(
    quality_config, synthetic_face
) -> None:
    """Yaw compresses the face through perspective, which is geometrically
    indistinguishable from an editor's squeeze."""
    context = context_for(
        synthetic_face.image,
        face_box=synthetic_face.box,
        landmarks=synthetic_face.landmarks,
        yaw_degrees=40.0,
    )
    result = DistortionAnalyzer(quality_config.distortion).analyse(context)

    assert "geometric" not in result.sub_scores
    assert result.note is not None
    assert "yaw" in result.note.lower()


@pytest.mark.unit
def test_distortion_measures_anisotropy_on_a_frontal_head(
    quality_config, synthetic_face
) -> None:
    context = context_for(
        synthetic_face.image,
        face_box=synthetic_face.box,
        landmarks=synthetic_face.landmarks,
        yaw_degrees=3.0,
    )
    result = DistortionAnalyzer(quality_config.distortion).analyse(context)
    assert "geometric" in result.sub_scores


# --------------------------------------------------------------------------- #
# Pixelation and resolution analysers
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_pixelation_falls_with_jpeg_quality(quality_config) -> None:
    analyser = PixelationAnalyzer(quality_config.pixelation)
    base = textured(384, 384)

    scores = []
    for quality in (95, 30, 8):
        ok, buffer = cv2.imencode(
            ".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        assert ok
        scores.append(
            analyser.analyse(context_for(cv2.imdecode(buffer, cv2.IMREAD_COLOR))).score
        )

    assert scores == sorted(scores, reverse=True)
    assert scores[0] - scores[-1] > 0.3


@pytest.mark.unit
def test_resolution_rewards_a_larger_face(quality_config) -> None:
    analyser = ResolutionAnalyzer(quality_config.resolution)
    image = textured(800, 800)

    small = analyser.analyse(context_for(image, face_box=BoundingBox(10, 10, 60, 60)))
    large = analyser.analyse(context_for(image, face_box=BoundingBox(10, 10, 310, 310)))

    assert large.score > small.score
    assert small.measurements["face_short_side"] == pytest.approx(50.0)


@pytest.mark.unit
def test_resolution_penalises_lost_detail(quality_config) -> None:
    """An enlarged thumbnail has the pixel count but not the information."""
    analyser = ResolutionAnalyzer(quality_config.resolution)
    base = textured(512, 512)
    small = cv2.resize(base, (128, 128), interpolation=cv2.INTER_AREA)
    enlarged = cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC)

    box = BoundingBox(100, 100, 400, 400)
    native = analyser.analyse(context_for(base, face_box=box))
    upscaled = analyser.analyse(context_for(enlarged, face_box=box))

    assert upscaled.measurements["detail_energy"] < native.measurements["detail_energy"]
    assert upscaled.score < native.score


@pytest.mark.unit
def test_resolution_uses_interocular_distance_when_available(
    quality_config, synthetic_face
) -> None:
    result = ResolutionAnalyzer(quality_config.resolution).analyse(
        context_for(
            synthetic_face.image,
            face_box=synthetic_face.box,
            landmarks=synthetic_face.landmarks,
        )
    )
    assert "interocular" in result.sub_scores
    assert result.measurements["interocular_pixels"] > 0.0


@pytest.mark.unit
def test_resolution_without_a_face_uses_detail_alone(quality_config) -> None:
    result = ResolutionAnalyzer(quality_config.resolution).analyse(
        context_for(textured())
    )
    assert set(result.sub_scores) == {"detail"}
    assert result.note is not None


# --------------------------------------------------------------------------- #
# Failure containment
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_safe_analyse_converts_a_crash_into_an_unmeasured_result(
    quality_config,
) -> None:
    """One analyser hitting a numerical edge case must not deprive the caller
    of the other seven dimensions."""

    class Exploding(BlurAnalyzer):
        def analyse(self, context: QualityContext):  # noqa: ANN201, ARG002
            raise RuntimeError("synthetic failure")

    result = Exploding(quality_config.blur).safe_analyse(context_for(textured()))
    assert result.measured is False
    assert result.note is not None
    assert "RuntimeError" in result.note


@pytest.mark.unit
@pytest.mark.parametrize(
    "analyser_name",
    ["blur", "brightness", "contrast", "noise", "resolution", "pixelation", "distortion"],
)
def test_analysers_survive_a_uniform_image(quality_config, analyser_name: str) -> None:
    """A degenerate but legal input must degrade, not raise."""
    analysers = {
        "blur": BlurAnalyzer(quality_config.blur),
        "brightness": BrightnessAnalyzer(quality_config.brightness),
        "contrast": ContrastAnalyzer(quality_config.contrast),
        "noise": NoiseAnalyzer(quality_config.noise),
        "resolution": ResolutionAnalyzer(quality_config.resolution),
        "pixelation": PixelationAnalyzer(quality_config.pixelation),
        "distortion": DistortionAnalyzer(quality_config.distortion),
    }
    flat = np.full((256, 256, 3), 128, dtype=np.uint8)
    result = analysers[analyser_name].safe_analyse(context_for(flat))
    assert 0.0 <= result.score <= 1.0


@pytest.mark.unit
@pytest.mark.parametrize("size", [(64, 64), (17, 240), (300, 11)])
def test_analysers_survive_extreme_aspect_ratios(
    quality_config, size: tuple[int, int]
) -> None:
    image = textured(size[0], size[1], seed=7)
    for analyser in (
        BlurAnalyzer(quality_config.blur),
        BrightnessAnalyzer(quality_config.brightness),
        ContrastAnalyzer(quality_config.contrast),
        NoiseAnalyzer(quality_config.noise),
        PixelationAnalyzer(quality_config.pixelation),
    ):
        result = analyser.safe_analyse(context_for(image))
        assert 0.0 <= result.score <= 1.0
