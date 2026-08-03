"""Sensor-noise estimation.

The hard part is that noise and texture are indistinguishable to any purely
local operator: a high-pass filter responds identically to grain and to hair.
Two estimators are combined and the *lower* is taken, because texture can only
ever inflate an estimate, never depress it.

:func:`immerkaer_sigma`
    Fast global estimator. Convolves with a kernel chosen to annihilate any
    locally-linear intensity ramp, so a smooth gradient contributes nothing
    and only the non-smooth residual - noise plus texture - survives.

:func:`flat_block_sigma`
    Robust estimator. Divides the image into blocks and estimates from the
    *flattest* decile only. Those blocks contain the least texture, so their
    residual is dominated by genuine sensor noise. This is the estimator that
    is right on a detailed image; Immerkaer over-reports badly there.

Both are reported, and the score is driven by the minimum. On a real portrait
the two typically differ by 2-4x, which is exactly the texture contamination
the robust estimator is designed to remove.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import NoiseConfig
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

#: Immerkaer's kernel. It is the difference of two Laplacians, constructed so
#: that its response to any locally-linear or locally-quadratic surface is
#: exactly zero - which is what makes it blind to smooth shading and sensitive
#: only to the high-frequency residual.
_IMMERKAER_KERNEL: npt.NDArray[np.float32] = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32
)

#: Normalising constant for the kernel above, from Immerkaer (1996):
#: sigma = sqrt(pi/2) / (6 * (W-2) * (H-2)) * sum |I * M|
_IMMERKAER_SCALE = math.sqrt(math.pi / 2.0) / 6.0

#: Edge length of the blocks the robust estimator works on. Large enough for
#: Immerkaer's kernel to have a stable interior once the border is trimmed,
#: small enough that flat regions of a real face - a cheek, a forehead - fit
#: inside one.
_BLOCK_SIZE = 32

#: Share of the flattest blocks used by the robust estimator.
_FLAT_BLOCK_QUANTILE = 0.10


def immerkaer_sigma(gray: GrayImage) -> float:
    """Estimate additive-noise sigma over the whole image.

    Fast and unbiased on a smooth image, but inflated by texture, since the
    kernel cannot tell grain from detail.

    Args:
        gray: Single-channel uint8 image.

    Returns:
        Estimated sigma on the 0-255 intensity scale.
    """
    height, width = gray.shape[:2]
    if height < 3 or width < 3:
        return 0.0

    response = cv2.filter2D(
        gray.astype(np.float32), cv2.CV_32F, _IMMERKAER_KERNEL, borderType=cv2.BORDER_REPLICATE
    )
    # Trim the replicated border, where the kernel's cancellation property
    # does not hold and the response is dominated by the edge extension.
    interior = response[1:-1, 1:-1]
    if interior.size == 0:
        return 0.0
    return float(_IMMERKAER_SCALE * np.abs(interior).mean())


def flat_block_sigma(
    gray: GrayImage,
    *,
    block_size: int = _BLOCK_SIZE,
    quantile: float = _FLAT_BLOCK_QUANTILE,
) -> float:
    """Estimate noise sigma from the flattest blocks only.

    Applies :func:`immerkaer_sigma` independently to each cell of a grid and
    returns a low quantile of the resulting distribution. The flattest blocks
    contain the least texture, so their estimate is the closest available proxy
    for pure sensor noise.

    Using the *same* estimator locally, rather than a different residual
    statistic, is deliberate. An earlier version measured the standard
    deviation of a median-filter residual and rescaled it by the MAD constant
    1.4826 - two unrelated corrections stacked on each other, which made the
    "robust" estimate *exceed* the global one on textured input and inverted
    the whole point of having it.

    Args:
        gray: Single-channel uint8 image.
        block_size: Grid cell edge length in pixels. Must be large enough for
            Immerkaer's kernel to have a stable interior.
        quantile: Fraction of blocks, ordered by their estimate, to average.

    Returns:
        Estimated sigma on the 0-255 intensity scale.
    """
    height, width = gray.shape[:2]
    if height < block_size * 2 or width < block_size * 2:
        return immerkaer_sigma(gray)

    rows = height // block_size
    columns = width // block_size

    estimates = [
        immerkaer_sigma(
            gray[
                r * block_size : (r + 1) * block_size,
                c * block_size : (c + 1) * block_size,
            ]
        )
        for r in range(rows)
        for c in range(columns)
    ]
    if not estimates:
        return immerkaer_sigma(gray)

    ordered = np.sort(np.asarray(estimates, dtype=np.float32))
    count = max(1, int(ordered.size * quantile))
    return float(np.mean(ordered[:count]))


def estimate_noise_sigma(gray: GrayImage) -> tuple[float, float, float]:
    """Estimate noise sigma with both estimators.

    Args:
        gray: Single-channel uint8 image.

    Returns:
        ``(chosen, immerkaer, flat_block)``. ``chosen`` is the minimum, because
        texture can only inflate an estimate and never depress it, so the lower
        of two estimates is the one less contaminated by image content.
    """
    global_estimate = immerkaer_sigma(gray)
    robust_estimate = flat_block_sigma(gray)
    return min(global_estimate, robust_estimate), global_estimate, robust_estimate


def local_contrast(gray: GrayImage) -> float:
    """Median local standard deviation over a 7x7 neighbourhood.

    The signal term of the noise-to-signal ratio. Using the median rather than
    the mean keeps a few very high-contrast edges from dominating.
    """
    if gray.size == 0:
        return 0.0
    values = gray.astype(np.float32)
    mean = cv2.blur(values, (7, 7))
    mean_of_squares = cv2.blur(values * values, (7, 7))
    variance = np.maximum(mean_of_squares - mean * mean, 0.0)
    return float(np.median(np.sqrt(variance)))


class NoiseAnalyzer(QualityMetric):
    """Sensor noise, absolute and relative to local signal contrast.

    Args:
        config: The noise section of the quality configuration.
    """

    def __init__(self, config: NoiseConfig) -> None:
        super().__init__("noise")
        self._config = config

    def analyse(self, context: QualityContext) -> MetricResult:
        """Estimate noise on the face when available, otherwise on the image.

        Preferring the face region is deliberate: noise in a dark background
        is irrelevant to whether the biometric can be extracted, and a noisy
        backdrop behind a cleanly-lit face should not fail the image.
        """
        gray = context.canonical_gray
        region = "face"
        if gray is None or gray.size == 0:
            gray = context.analysis_gray
            region = "image"
        if gray.size == 0:
            return MetricResult.unmeasured(self.name, "image has no pixels")

        sigma, global_sigma, robust_sigma = estimate_noise_sigma(gray)
        contrast = local_contrast(gray)
        ratio = sigma / max(contrast, EPSILON)

        sub_scores = {
            "sigma": ramp_score(sigma, self._config.sigma),
            "ratio": ramp_score(ratio, self._config.noise_to_signal),
        }
        score = weighted_mean(sub_scores, self._config.weights)

        note = None
        if global_sigma > robust_sigma * 3.0:
            note = (
                "The global and robust noise estimates diverge sharply, which "
                "indicates a highly textured subject rather than a noisy sensor. "
                "The robust estimate was used."
            )

        return MetricResult(
            name=self.name,
            score=score,
            measurements={
                "sigma": sigma,
                "sigma_immerkaer": global_sigma,
                "sigma_flat_block": robust_sigma,
                "local_contrast": contrast,
                "noise_to_signal": ratio,
                "measured_on_face": 1.0 if region == "face" else 0.0,
            },
            sub_scores=sub_scores,
            limiting_factor=limiting_component(sub_scores, self._config.weights),
            note=note,
        )


__all__ = [
    "NoiseAnalyzer",
    "estimate_noise_sigma",
    "flat_block_sigma",
    "immerkaer_sigma",
    "local_contrast",
]
