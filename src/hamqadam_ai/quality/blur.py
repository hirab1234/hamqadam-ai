"""Focus analysis: whole-image blur and face-region sharpness.

Two analysers that answer genuinely different questions:

:class:`BlurAnalyzer`
    *Is the photograph in focus?* Measured on the resolution-normalised
    analysis image. Catches camera shake and defocus.

:class:`SharpnessAnalyzer`
    *Is the face in focus?* Measured on the eye band of the canonical face
    crop, plus the ratio of face focus to scene focus. Catches the case global
    blur misses entirely - a pin-sharp background with a motion-blurred
    subject, which is what a moving hand or a turning head produces and which
    is also a common artefact of a replayed video attack.

Three complementary focus measures
----------------------------------
No single focus operator is reliable on its own:

* **Laplacian variance** is the classic, most sensitive measure, but it scales
  with local contrast, so a low-contrast sharp image scores like a
  high-contrast blurred one.
* **Tenengrad** (squared Sobel magnitude) is less sensitive to impulse noise,
  because squaring a first derivative amplifies edges rather than isolated
  pixels the way a second derivative does.
* **Spectral high-frequency ratio** is a *ratio*, so it is largely independent
  of both contrast and content. It is what disambiguates "genuinely smooth
  subject" from "out of focus" - the failure case that defeats the other two.

Both derivative measures are evaluated **per axis, and scored on the weaker
one**. Every isotropic focus operator conceals half of a directional blur, and
the effect is large enough to matter: a 41-pixel horizontal smear leaves the
summed Laplacian variance at 2665 - comfortably "sharp" - while its weaker axis
reads 38. On genuinely isotropic content the two agree to within about 10%, so
the correction costs nothing where it is not needed.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import BlurConfig, SharpnessConfig
from hamqadam_ai.core.constants import EPSILON
from hamqadam_ai.quality.base import (
    GrayImage,
    MetricResult,
    QualityContext,
    QualityMetric,
)
from hamqadam_ai.quality.scoring import (
    limiting_component,
    ramp_score,
    weighted_mean,
)
from hamqadam_ai.quality.spectral import HIGH_FREQUENCY_CUTOFF


def laplacian_variance(gray: GrayImage) -> float:
    """Variance of the Laplacian - the standard focus measure.

    A sharp image has strong second derivatives at edges and therefore a wide
    spread of Laplacian responses; a blurred one has them all near zero.

    Args:
        gray: Single-channel uint8 image.

    Returns:
        The variance, in squared intensity units. Scale-dependent, which is
        why callers measure on a canonical-size crop.
    """
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())


def tenengrad(gray: GrayImage) -> float:
    """Mean squared Sobel gradient magnitude, summed over both axes.

    Less sensitive than the Laplacian to impulse noise: a first derivative
    amplifies edges, whereas a second derivative amplifies isolated hot pixels
    just as strongly as real structure.

    Args:
        gray: Single-channel uint8 image.

    Returns:
        The mean squared gradient magnitude.
    """
    if gray.size == 0:
        return 0.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(grad_x * grad_x + grad_y * grad_y))


def worst_axis_laplacian(gray: GrayImage) -> float:
    """Laplacian variance restricted to the axis carrying the least detail.

    The Laplacian is the sum of the two second partial derivatives, so like
    every isotropic operator it conceals directional blur. Splitting it and
    scoring the weaker half is dramatic in effect - measured on the test image
    a 41-pixel horizontal smear reads 2665 isotropically and **38** on its
    weaker axis, a factor of seventy.

    On isotropic content the two agree closely (88582 against 78853 on a sharp
    image, 99 against 77 after a 9x9 Gaussian), so the same configured anchors
    remain valid and no separate calibration is needed.

    Args:
        gray: Single-channel uint8 image.

    Returns:
        Twice the weaker axis's second-derivative variance.
    """
    if gray.size == 0:
        return 0.0
    second_x = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=3)
    second_y = cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=3)
    return 2.0 * min(float(second_x.var()), float(second_y.var()))


def worst_axis_tenengrad(gray: GrayImage) -> float:
    """Tenengrad restricted to whichever axis carries the *least* detail.

    Every isotropic focus measure systematically under-reports directional
    blur, and the effect is large: a 41-pixel horizontal smear on the test
    image leaves the Laplacian variance at 2665 - comfortably in "sharp"
    territory - because horizontal blurring removes only horizontal-frequency
    content and every vertical edge survives untouched. Averaging the two axes
    hides exactly half the defect.

    Taking the weaker axis and doubling it (so the scale matches the isotropic
    sum for a genuinely isotropic image) removes that blind spot with no
    special-casing: a sharp photograph is sharp along both axes and is
    unaffected, while a smeared one is scored on the axis that was destroyed.

    Args:
        gray: Single-channel uint8 image.

    Returns:
        Twice the weaker axis's mean squared gradient, directly comparable
        with :func:`tenengrad` on isotropic input.
    """
    if gray.size == 0:
        return 0.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy_x = float(np.mean(grad_x * grad_x))
    energy_y = float(np.mean(grad_y * grad_y))
    return 2.0 * min(energy_x, energy_y)


def normalised_laplacian(gray: GrayImage) -> float:
    """Laplacian variance divided by intensity variance.

    Removes the contrast dependence of the raw measure, which matters when
    comparing a well-lit face against one photographed in dim light. Reported
    as supporting evidence rather than scored directly, because the
    normalisation also suppresses genuine detail in high-contrast images.
    """
    if gray.size == 0:
        return 0.0
    intensity_variance = float(np.var(gray.astype(np.float32)))
    return laplacian_variance(gray) / max(intensity_variance, EPSILON)


def motion_blur_anisotropy(gray: GrayImage) -> float:
    """How directional the image's gradient energy is, in ``[0, 1]``.

    Motion blur suppresses detail along one axis only, leaving a strongly
    anisotropic gradient field; defocus is isotropic. Distinguishing the two
    matters because they have different remedies - "hold the camera still"
    versus "tap to focus" - and because directional blur is a weak signal of a
    photographed screen.

    Returns:
        0.0 for perfectly isotropic gradients, approaching 1.0 for a purely
        one-directional smear.
    """
    if gray.size == 0:
        return 0.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    energy_x = float(np.mean(grad_x * grad_x))
    energy_y = float(np.mean(grad_y * grad_y))
    total = energy_x + energy_y
    if total <= EPSILON:
        return 0.0
    return float(abs(energy_x - energy_y) / total)


class BlurAnalyzer(QualityMetric):
    """Whole-image focus.

    Args:
        config: The blur section of the quality configuration.
    """

    def __init__(self, config: BlurConfig) -> None:
        super().__init__("blur")
        self._config = config

    def analyse(self, context: QualityContext) -> MetricResult:
        """Measure focus on the resolution-normalised analysis image."""
        gray = context.analysis_gray
        if gray.size == 0:
            return MetricResult.unmeasured(self.name, "image has no pixels")

        isotropic_laplacian = laplacian_variance(gray)
        isotropic_tenengrad = tenengrad(gray)
        # Both derivative terms are scored on their *weaker axis*, not on the
        # average of the two. An isotropic operator conceals exactly half of
        # any directional blur, and the effect is not marginal: see
        # `worst_axis_laplacian` for the measured seventy-fold difference.
        directional_laplacian = worst_axis_laplacian(gray)
        directional_tenengrad = worst_axis_tenengrad(gray)
        spectral = context.radial_spectrum.energy_above(HIGH_FREQUENCY_CUTOFF)
        anisotropy = motion_blur_anisotropy(gray)

        sub_scores = {
            "laplacian": ramp_score(
                directional_laplacian, self._config.laplacian_variance
            ),
            "tenengrad": ramp_score(directional_tenengrad, self._config.tenengrad),
            "spectral": ramp_score(spectral, self._config.high_frequency_ratio),
        }
        score = weighted_mean(sub_scores, self._config.weights)

        note = None
        if anisotropy > 0.35:
            note = (
                "Blur is strongly directional, which indicates camera or subject "
                "motion rather than defocus. Note that the Laplacian term "
                "over-reports sharpness in this case, because the unblurred axis "
                "retains its edges."
            )

        return MetricResult(
            name=self.name,
            score=score,
            measurements={
                "laplacian_variance": isotropic_laplacian,
                "laplacian_worst_axis": directional_laplacian,
                "tenengrad": isotropic_tenengrad,
                "tenengrad_worst_axis": directional_tenengrad,
                "high_frequency_ratio": spectral,
                "normalised_laplacian": normalised_laplacian(gray),
                "motion_anisotropy": anisotropy,
                "analysis_width": float(gray.shape[1]),
                "analysis_height": float(gray.shape[0]),
            },
            sub_scores=sub_scores,
            limiting_factor=limiting_component(sub_scores, self._config.weights),
            note=note,
        )


class SharpnessAnalyzer(QualityMetric):
    """Face-region focus, measured on the eye band.

    Args:
        config: The sharpness section of the quality configuration.
    """

    def __init__(self, config: SharpnessConfig) -> None:
        super().__init__("sharpness")
        self._config = config

    def analyse(self, context: QualityContext) -> MetricResult:
        """Measure focus on the face, and relative to the rest of the scene."""
        eye_band = context.eye_band
        if eye_band is None or eye_band.size == 0:
            return MetricResult.unmeasured(
                self.name, "no face was detected, so face sharpness is undefined"
            )

        eye_laplacian = laplacian_variance(eye_band)
        relative = self._face_vs_scene(context)

        sub_scores = {
            "eye_region": ramp_score(
                eye_laplacian, self._config.eye_region_laplacian
            ),
        }
        measurements: dict[str, float] = {
            "eye_region_laplacian": eye_laplacian,
            "eye_band_height": float(eye_band.shape[0]),
        }

        if relative is not None:
            sub_scores["relative"] = ramp_score(
                relative, self._config.face_vs_scene_ratio
            )
            measurements["face_vs_scene_ratio"] = relative

        canonical = context.canonical_gray
        if canonical is not None:
            measurements["canonical_laplacian"] = laplacian_variance(canonical)
            measurements["canonical_tenengrad"] = tenengrad(canonical)

        score = weighted_mean(sub_scores, self._config.weights)

        note = None
        if context.canonical_upsampled:
            note = (
                "The detected face was smaller than the canonical analysis size and "
                "was upscaled; interpolation inflates apparent sharpness, so this "
                "score is optimistic. See the resolution metric for true detail."
            )
        elif relative is not None and relative < 0.4:
            note = (
                "The face is markedly less sharp than the rest of the scene, which "
                "indicates subject motion rather than a focus error."
            )

        return MetricResult(
            name=self.name,
            score=score,
            measurements=measurements,
            sub_scores=sub_scores,
            limiting_factor=limiting_component(sub_scores, self._config.weights),
            note=note,
        )

    @staticmethod
    def _face_vs_scene(context: QualityContext) -> float | None:
        """Ratio of face-region focus to background focus.

        Both sides are measured on the *same* image at the *same* sampling
        density, which is the only way the ratio means anything. Comparing the
        canonical crop against the full image would compare two different
        scales and produce a number driven by the resampling, not the optics.

        Returns:
            The ratio, or ``None`` when there is too little background to
            compare against - a tight head-and-shoulders crop, typically.
        """
        box = context.analysis_face_box
        if box is None:
            return None

        gray = context.analysis_gray
        x1, y1, x2, y2 = box.as_int_tuple()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(gray.shape[1], x2)
        y2 = min(gray.shape[0], y2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None

        face_region = gray[y1:y2, x1:x2]

        background = gray.copy()
        background[y1:y2, x1:x2] = 0
        mask = np.ones(gray.shape, dtype=bool)
        mask[y1:y2, x1:x2] = False
        if mask.sum() < gray.size * 0.15:
            # The face fills the frame; there is no meaningful background.
            return None

        face_focus = tenengrad(face_region)
        scene_focus = _masked_tenengrad(gray, mask)
        if scene_focus <= EPSILON:
            return None

        # Capped at 1.0: a face sharper than its background is perfectly
        # normal - it is what a portrait with shallow depth of field looks
        # like - and must not score above "ideal".
        return float(min(1.0, face_focus / scene_focus))


def _masked_tenengrad(gray: GrayImage, mask: npt.NDArray[np.bool_]) -> float:
    """Tenengrad restricted to the pixels selected by ``mask``."""
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = grad_x * grad_x + grad_y * grad_y
    selected = energy[mask]
    return float(selected.mean()) if selected.size else 0.0


__all__ = [
    "BlurAnalyzer",
    "SharpnessAnalyzer",
    "laplacian_variance",
    "motion_blur_anisotropy",
    "normalised_laplacian",
    "tenengrad",
    "worst_axis_laplacian",
    "worst_axis_tenengrad",
]
