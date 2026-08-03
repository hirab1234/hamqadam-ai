"""Detecting a photograph taken of a screen.

Why moire is the right signal
-----------------------------
A display is a regular grid of pixels; a camera sensor is another regular
grid. Photograph one with the other at any angle and the two periodicities
beat against each other, producing interference at their sum and difference
frequencies. That is moire, and in the Fourier domain it is unmistakable: an
**isolated, symmetric pair of peaks away from the axes**, sitting on top of an
otherwise smoothly decaying spectrum.

Natural images have no such structure. A photograph's spectrum falls off
roughly as a power law and is smooth; the exceptions - fabric weave, brick,
window blinds - are usually axis-aligned, which is why the axes themselves are
excluded from the search along with the DC neighbourhood.

Measured peak prominence above the locally-smoothed spectrum:

    genuine photograph            1.410
    quality-12 JPEG               1.456
    twice-recompressed            1.437
    screenshot                    1.428
    photograph of a print         1.411
    synthetic render              1.443
    photograph of a screen        3.461 - 4.220   (display pitch 3 to 8 px)

Everything that is not a screen capture sits in a band 0.05 wide; a screen
capture sits at two and a half times its top. That is the widest separation
any measurement in this module achieves, which is why this detector is
allowed to trigger on its own evidence alone.

Its limits, stated plainly
--------------------------
Moire depends on the display grid surviving into the captured image. A capture
made far enough away, or downscaled hard enough afterwards, loses the grid
frequency to the resampling filter and this detector goes quiet. It reports
what it found; a null result is not evidence of a genuine capture, and the
aggregator treats it accordingly.
"""

from __future__ import annotations

import cv2
import numpy as np

from hamqadam_ai.authenticity.base import (
    SPECTRAL_SIZE,
    AuthenticityContext,
    AuthenticityDetector,
    AuthenticitySignal,
    ramp,
)
from hamqadam_ai.core.config import MoireDetectorConfig


class MoireDetector(AuthenticityDetector):
    """Finds the periodic interference left by photographing a display.

    Args:
        config: Search band, anchors and the trigger threshold.
    """

    __slots__ = ("_config",)

    def __init__(self, config: MoireDetectorConfig) -> None:
        super().__init__(name="screen_recapture")
        self._config = config

    def analyse(self, context: AuthenticityContext) -> AuthenticitySignal:
        """Find the strongest isolated off-axis peak in the spectrum."""
        if context.degenerate:
            return AuthenticitySignal(
                name=self.name,
                triggered=False,
                confidence=0.0,
                note="image is uniform or too small to analyse",
            )

        spectrum = context.log_spectrum
        radius = context.radius_map
        centre = SPECTRAL_SIZE // 2

        ys, xs = np.mgrid[0:SPECTRAL_SIZE, 0:SPECTRAL_SIZE]
        band = (
            (radius > SPECTRAL_SIZE * self._config.min_radius_fraction)
            & (radius < SPECTRAL_SIZE * self._config.max_radius_fraction)
        )
        # Exclude the axes. JPEG's 8x8 blocking puts a comb of energy exactly
        # there, and so does any axis-aligned texture, so a peak on an axis is
        # the one place a false positive is likely.
        band &= (np.abs(xs - centre) > self._config.axis_exclusion) & (
            np.abs(ys - centre) > self._config.axis_exclusion
        )

        if not band.any():
            return AuthenticitySignal(
                name=self.name,
                triggered=False,
                confidence=0.0,
                note="the configured search band is empty",
            )

        # Prominence above the locally-smoothed spectrum rather than the raw
        # magnitude: an image with more energy everywhere would otherwise
        # score higher everywhere, which measures brightness, not periodicity.
        background = cv2.GaussianBlur(spectrum, (0, 0), self._config.background_sigma)
        excess = np.where(band, spectrum - background, 0.0)

        peak_index = int(np.argmax(excess))
        peak_y, peak_x = divmod(peak_index, SPECTRAL_SIZE)
        prominence = float(excess[peak_y, peak_x])

        symmetry = _mirror_prominence(excess, peak_y, peak_x, centre)

        confidence = ramp(
            prominence,
            floor=self._config.prominence_floor,
            ceiling=self._config.prominence_ceiling,
        )
        # A real interference pattern is symmetric about DC, because a real
        # signal's spectrum is. A lone peak with no mirror is more likely one
        # bright artefact than a periodic structure.
        confidence *= 0.5 + 0.5 * ramp(symmetry, floor=0.30, ceiling=0.85)

        return AuthenticitySignal(
            name=self.name,
            triggered=confidence >= self._config.min_confidence,
            confidence=confidence,
            measurements={
                "peak_prominence": prominence,
                "peak_radius_fraction": float(radius[peak_y, peak_x] / centre),
                "mirror_symmetry": symmetry,
            },
        )


def _mirror_prominence(
    excess: np.ndarray, peak_y: int, peak_x: int, centre: int
) -> float:
    """How strong the peak's reflection through DC is, relative to the peak.

    Searched in a small neighbourhood rather than at the exact mirror pixel,
    because the resample and the window shift it by a pixel or two.
    """
    mirror_y = 2 * centre - peak_y
    mirror_x = 2 * centre - peak_x
    size = excess.shape[0]
    if not (0 <= mirror_y < size and 0 <= mirror_x < size):
        return 0.0

    half = 3
    y0, y1 = max(mirror_y - half, 0), min(mirror_y + half + 1, size)
    x0, x1 = max(mirror_x - half, 0), min(mirror_x + half + 1, size)
    neighbourhood = excess[y0:y1, x0:x1]
    if neighbourhood.size == 0:
        return 0.0

    peak = float(excess[peak_y, peak_x])
    if peak <= 0.0:
        return 0.0
    return float(min(max(neighbourhood.max() / peak, 0.0), 1.0))


__all__ = ["MoireDetector"]
