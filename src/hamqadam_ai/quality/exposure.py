"""Exposure and tonal analysis: brightness and contrast.

Both analysers measure the **face region** in preference to the whole frame,
which is a deliberate departure from generic image-quality practice. A selfie
taken against a bright window has a perfectly healthy global histogram and a
silhouetted, unusable face; a portrait against a dark wall reads as
underexposed globally while the face itself is lit correctly. What matters for
identity verification is the exposure of the biometric, not of the scene.

Clipping is measured separately from the mean because the two fail
differently. Clipped pixels carry *no recoverable information* - a blown
highlight is not "a bit bright", it is a hole in the data - so even a few
percent is disqualifying while the mean still looks healthy.
"""

from __future__ import annotations

import cv2
import numpy as np

from hamqadam_ai.core.config import BrightnessConfig, ContrastConfig
from hamqadam_ai.quality.base import (
    GrayImage,
    MetricResult,
    QualityContext,
    QualityMetric,
    percentile_range,
    shannon_entropy,
)
from hamqadam_ai.quality.scoring import (
    band_score,
    limiting_component,
    ramp_score,
    weighted_mean,
)

#: Luminance at or below which a pixel is treated as crushed to black.
SHADOW_CLIP_LEVEL = 4

#: Luminance at or above which a pixel is treated as blown to white.
HIGHLIGHT_CLIP_LEVEL = 251


def clipping_fractions(gray: GrayImage) -> tuple[float, float]:
    """Fraction of pixels crushed to black and blown to white.

    Args:
        gray: Single-channel uint8 image.

    Returns:
        ``(shadow_fraction, highlight_fraction)``, each in ``[0, 1]``.
    """
    if gray.size == 0:
        return 0.0, 0.0
    total = float(gray.size)
    shadows = float(np.count_nonzero(gray <= SHADOW_CLIP_LEVEL)) / total
    highlights = float(np.count_nonzero(gray >= HIGHLIGHT_CLIP_LEVEL)) / total
    return shadows, highlights


def illumination_uniformity(gray: GrayImage) -> float:
    """How evenly the region is lit, in ``[0, 1]``.

    Splits the region into a 3x3 grid and compares the brightest cell against
    the darkest. Strong side-lighting - one half of the face in shadow -
    produces a healthy mean and a healthy standard deviation while being
    genuinely difficult for a recogniser, so it needs its own signal.

    Returns:
        1.0 for perfectly even illumination, approaching 0.0 for a face lit
        from one side only.
    """
    if gray.size == 0:
        return 1.0
    height, width = gray.shape[:2]
    if height < 9 or width < 9:
        return 1.0

    cell_h, cell_w = height // 3, width // 3
    means = [
        float(gray[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w].mean())
        for r in range(3)
        for c in range(3)
    ]
    brightest = max(means)
    darkest = min(means)
    if brightest <= 0.0:
        return 0.0
    return float(max(0.0, darkest / brightest))


def _region_for(context: QualityContext) -> tuple[GrayImage, str]:
    """Return the region exposure is judged on, preferring the face."""
    canonical = context.canonical_gray
    if canonical is not None and canonical.size > 0:
        return canonical, "face"
    return context.analysis_gray, "image"


class BrightnessAnalyzer(QualityMetric):
    """Exposure: mean luminance plus shadow and highlight clipping.

    Args:
        config: The brightness section of the quality configuration.
    """

    def __init__(self, config: BrightnessConfig) -> None:
        super().__init__("brightness")
        self._config = config

    def analyse(self, context: QualityContext) -> MetricResult:
        """Measure exposure of the face region, or the image without one."""
        gray, region = _region_for(context)
        if gray.size == 0:
            return MetricResult.unmeasured(self.name, "image has no pixels")

        mean_luma = float(gray.mean())
        shadows, highlights = clipping_fractions(gray)
        uniformity = illumination_uniformity(gray)

        sub_scores = {
            "mean": band_score(mean_luma, self._config.mean_luma),
            "shadow_clip": ramp_score(shadows, self._config.shadow_clipping),
            "highlight_clip": ramp_score(highlights, self._config.highlight_clipping),
        }

        # Mean luminance GATES the clipping terms rather than averaging with
        # them, because that is the physical relationship. Under a plain
        # weighted mean a silhouetted face scored 45: its mean term was zero
        # but both clipping terms were perfect, and two thirds of the weight
        # outvoted the one measurement that mattered. Whether a silhouette's
        # shadows are technically clipped is meaningless - there is no signal
        # in that region either way.
        retention = weighted_mean(
            {
                "shadow_clip": sub_scores["shadow_clip"],
                "highlight_clip": sub_scores["highlight_clip"],
            },
            self._config.weights,
        )
        score = sub_scores["mean"] * (
            1.0 - self._config.clipping_influence * (1.0 - retention)
        )

        notes: list[str] = []
        if mean_luma < self._config.mean_luma.min_acceptable:
            # Below the acceptable band the image is underexposed whether or
            # not shadows have clipped; an earlier version required clipping
            # too and so said nothing about a uniformly dark photograph.
            notes.append("The image is far too dark to assess reliably.")
        elif mean_luma > self._config.mean_luma.max_acceptable:
            notes.append("The image is far too bright to assess reliably.")
        elif mean_luma < self._config.mean_luma.ideal_low:
            notes.append("The image is underexposed.")
        elif mean_luma > self._config.mean_luma.ideal_high:
            notes.append("The image is overexposed.")

        if shadows > 0.05:
            notes.append("Shadow detail has been crushed to black and is unrecoverable.")
        if highlights > 0.03:
            notes.append("Highlight detail has been blown out and is unrecoverable.")
        if uniformity < 0.35:
            notes.append(
                "The subject is lit strongly from one side; parts of the face are "
                "much darker than others."
            )

        note = " ".join(notes) if notes else None

        return MetricResult(
            name=self.name,
            score=score,
            measurements={
                "mean_luma": mean_luma,
                "median_luma": float(np.median(gray)),
                "shadow_clipping": shadows,
                "highlight_clipping": highlights,
                "illumination_uniformity": uniformity,
                "measured_on_face": 1.0 if region == "face" else 0.0,
            },
            sub_scores=sub_scores,
            limiting_factor=limiting_component(
                sub_scores,
                # `mean` gates the whole metric, so for the purpose of naming
                # the limiting factor it carries the dominant notional weight.
                {"mean": 0.6, "shadow_clip": 0.2, "highlight_clip": 0.2},
            ),
            note=note,
        )


class ContrastAnalyzer(QualityMetric):
    """Tonal separation: RMS, dynamic range and histogram entropy.

    Args:
        config: The contrast section of the quality configuration.
    """

    def __init__(self, config: ContrastConfig) -> None:
        super().__init__("contrast")
        self._config = config

    def analyse(self, context: QualityContext) -> MetricResult:
        """Measure tonal separation of the face region, or of the image."""
        gray, region = _region_for(context)
        if gray.size == 0:
            return MetricResult.unmeasured(self.name, "image has no pixels")

        rms = float(gray.std())
        dynamic = percentile_range(gray, low=1.0, high=99.0)
        entropy = shannon_entropy(gray)

        sub_scores = {
            "rms": ramp_score(rms, self._config.rms),
            "dynamic_range": ramp_score(dynamic, self._config.dynamic_range),
            "entropy": ramp_score(entropy, self._config.entropy),
        }
        score = weighted_mean(sub_scores, self._config.weights)

        note = None
        if entropy < self._config.entropy.floor and rms >= self._config.rms.floor:
            # A wide spread across very few distinct levels is the signature of
            # a posterised or heavily-processed image, not a genuine capture.
            note = (
                "Tonal spread looks acceptable but very few distinct levels are in "
                "use, which suggests the image has been posterised or heavily "
                "re-processed."
            )
        elif score < 0.35:
            note = "The image is flat and washed out, with little tonal separation."

        return MetricResult(
            name=self.name,
            score=score,
            measurements={
                "rms_contrast": rms,
                "dynamic_range_p1_p99": dynamic,
                "entropy_bits": entropy,
                "michelson": _michelson_contrast(gray),
                "measured_on_face": 1.0 if region == "face" else 0.0,
            },
            sub_scores=sub_scores,
            limiting_factor=limiting_component(sub_scores, self._config.weights),
            note=note,
        )


def _michelson_contrast(gray: GrayImage) -> float:
    """Michelson contrast from the 5th and 95th luminance percentiles.

    Reported as supporting evidence. Bounded to ``[0, 1]`` regardless of
    exposure, which makes it easier to compare across images than RMS, though
    it is too insensitive to drive a score on its own.
    """
    if gray.size == 0:
        return 0.0
    low, high = np.percentile(gray, [5.0, 95.0])
    denominator = float(high + low)
    if denominator <= 0.0:
        return 0.0
    return float(max(0.0, (high - low) / denominator))


def equalise_preview(gray: GrayImage) -> GrayImage:
    """CLAHE-equalised copy, for the annotated demo output only.

    Never used in scoring: adaptive equalisation would erase precisely the
    exposure defects this module exists to measure.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalised: GrayImage = clahe.apply(gray).astype(np.uint8, copy=False)
    return equalised


__all__ = [
    "HIGHLIGHT_CLIP_LEVEL",
    "SHADOW_CLIP_LEVEL",
    "BrightnessAnalyzer",
    "ContrastAnalyzer",
    "clipping_fractions",
    "equalise_preview",
    "illumination_uniformity",
]
