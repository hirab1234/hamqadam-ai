"""Detecting a photograph of a printed photograph, and rendered artwork.

Two detectors share this file because they are the same question asked twice:
does this image carry the high-frequency detail a camera sensor puts into
everything it sees? Print destroys it by reproducing at a few hundred dpi;
rendering never creates it.

Print recapture
---------------
Measured share of spectral energy above half Nyquist:

    twice-recompressed            0.0206
    genuine photograph            0.0178
    synthetic render              0.0153
    quality-12 JPEG               0.0121
    photograph of a print         0.0021

An order of magnitude below the genuine case, and - importantly - well below
the quality-12 JPEG that is the hard negative for every other detector here.
Corroborated by two secondary signals a print also shows: a compressed tonal
range, because ink reaches neither true black nor paper white, and a uniform
border where the print's edge or its mount is in frame.

Rendered artwork
----------------
A vector avatar or cartoon is built from constant fills, so it fails a
different test: it reuses a tiny palette. Measured distinct colours per pixel:

    twice-recompressed            0.304
    genuine photograph            0.286
    screenshot                    0.235
    photograph of a print         0.195
    quality-12 JPEG               0.173
    synthetic render              0.000

Zero to three decimal places, against 0.17 for the most degraded real
photograph. Combined with a near-total flat-block fraction - 0.911 against
0.355 for that same JPEG - the separation is unambiguous.

Note the asymmetry with the screenshot detector: there, flat blocks were
useless because JPEG produces them too. Here they are useful, because 0.911 is
nowhere near 0.355. The same measurement is diagnostic at one magnitude and
worthless at another, which is why every anchor in this module is written down
next to the population it separates.
"""

from __future__ import annotations

import numpy as np

from hamqadam_ai.authenticity.base import (
    SPECTRAL_SIZE,
    AuthenticityContext,
    AuthenticityDetector,
    AuthenticitySignal,
    ramp,
)
from hamqadam_ai.core.config import (
    PrintRecaptureDetectorConfig,
    SyntheticDetectorConfig,
)


class PrintRecaptureDetector(AuthenticityDetector):
    """Finds the signature of a photograph reproduced on paper.

    Args:
        config: Anchors and the trigger threshold.
    """

    __slots__ = ("_config",)

    def __init__(self, config: PrintRecaptureDetectorConfig) -> None:
        super().__init__(name="print_recapture")
        self._config = config

    def analyse(self, context: AuthenticityContext) -> AuthenticitySignal:
        """Measure high-frequency loss, tonal compression and border."""
        if context.degenerate:
            return AuthenticitySignal(
                name=self.name,
                triggered=False,
                confidence=0.0,
                note="image is uniform or too small to analyse",
            )

        high_frequency = _high_frequency_share(context)
        tonal = _tonal_span(context)
        border = _uniform_border_fraction(context)

        # Lower high-frequency share means more print-like, so the anchors run
        # downwards.
        confidence = ramp(
            high_frequency,
            floor=self._config.high_frequency_floor,
            ceiling=self._config.high_frequency_ceiling,
        )
        if confidence > 0.0:
            corroboration = min(
                1.0,
                0.6 * ramp(tonal, floor=self._config.tonal_span_floor,
                           ceiling=self._config.tonal_span_ceiling)
                + 0.4 * ramp(border, floor=0.01, ceiling=0.12),
            )
            confidence = min(1.0, confidence * (0.75 + 0.25 * (1.0 + corroboration)))

        return AuthenticitySignal(
            name=self.name,
            triggered=confidence >= self._config.min_confidence,
            confidence=confidence,
            measurements={
                "high_frequency_share": high_frequency,
                "tonal_span": tonal,
                "uniform_border_fraction": border,
            },
        )


class SyntheticImageDetector(AuthenticityDetector):
    """Finds rendered artwork: avatars, cartoons, flat-shaded illustration.

    Args:
        config: Anchors and the trigger threshold.
    """

    __slots__ = ("_config",)

    def __init__(self, config: SyntheticDetectorConfig) -> None:
        super().__init__(name="synthetic_image")
        self._config = config

    def analyse(self, context: AuthenticityContext) -> AuthenticitySignal:
        """Measure palette size and flat-region share."""
        if context.degenerate:
            return AuthenticitySignal(
                name=self.name,
                triggered=False,
                confidence=0.0,
                note="image is uniform or too small to analyse",
            )

        palette = _unique_colour_ratio(context, self._config.colour_sample_size)
        flat = _flat_block_fraction(context)

        # Both must agree. A tiny palette alone fires on a heavily posterised
        # photograph; a high flat fraction alone fires on a quality-12 JPEG,
        # measured at 0.355. Requiring both is what separates 0.911-and-0.000
        # from 0.355-and-0.173.
        palette_term = ramp(
            palette,
            floor=self._config.palette_floor,
            ceiling=self._config.palette_ceiling,
        )
        flat_term = ramp(
            flat,
            floor=self._config.flat_floor,
            ceiling=self._config.flat_ceiling,
        )
        confidence = float(np.sqrt(max(palette_term, 0.0) * max(flat_term, 0.0)))

        return AuthenticitySignal(
            name=self.name,
            triggered=confidence >= self._config.min_confidence,
            confidence=confidence,
            measurements={
                "unique_colour_ratio": palette,
                "flat_block_fraction": flat,
            },
        )


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #


def _high_frequency_share(context: AuthenticityContext) -> float:
    """Fraction of spectral power above half Nyquist.

    Computed on the unwindowed spectrum of the fixed-size resample: the window
    the moire detector needs would itself suppress high frequencies near the
    frame edge, which is exactly the quantity being measured here.
    """
    transformed = np.fft.fftshift(np.fft.fft2(context.spectral_grey))
    power = np.abs(transformed) ** 2
    radius = context.radius_map
    centre = SPECTRAL_SIZE // 2

    # Exclude DC and its immediate neighbours: on a near-flat image they carry
    # essentially all the energy and the ratio becomes meaningless.
    total = float(power[radius > 2.0].sum())
    if total <= 0.0:
        return 0.0
    high = float(power[radius > centre * 0.5].sum())
    return high / total


def _tonal_span(context: AuthenticityContext) -> float:
    """Spread between the 2nd and 98th luminance percentiles, normalised.

    Ink cannot reach true black or paper white, so a print's tonal range is
    compressed. Percentiles rather than min and max, so one specular highlight
    or one dead pixel does not restore the full span.
    """
    low, high = np.percentile(context.grey_f32, [2.0, 98.0])
    return float((high - low) / 255.0)


def _uniform_border_fraction(context: AuthenticityContext) -> float:
    """Share of the frame occupied by a uniform border around the content.

    A photographed print usually shows its own white margin or its mount.
    Measured as contiguous constant rows and columns at the frame edges.
    """
    rows = context.constant_row_mask
    columns = _constant_column_mask(context)

    def leading(mask: np.ndarray) -> int:
        count = 0
        while count < mask.size and bool(mask[count]):
            count += 1
        return count

    def trailing(mask: np.ndarray) -> int:
        count = 0
        while count < mask.size and bool(mask[mask.size - 1 - count]):
            count += 1
        return count

    row_border = leading(rows) + trailing(rows)
    column_border = leading(columns) + trailing(columns)

    vertical = row_border / max(context.height, 1)
    horizontal = column_border / max(context.width, 1)
    return float(min(max(vertical, horizontal), 1.0))


def _constant_column_mask(context: AuthenticityContext) -> np.ndarray:
    """Columns constant across the full image height."""
    columns = context.grey.astype(np.int16)
    return np.asarray(columns.max(axis=0) - columns.min(axis=0) <= 2)


def _unique_colour_ratio(context: AuthenticityContext, sample: int) -> float:
    """Distinct colours per sampled pixel.

    Sampled on a fixed stride rather than randomly, so the measurement is
    deterministic - a verification score that changes between two runs of the
    same image is not one anybody can defend.
    """
    flat = context.image.reshape(-1, 3)
    if flat.shape[0] > sample:
        stride = flat.shape[0] // sample
        flat = flat[::stride][:sample]

    packed = (
        (flat[:, 0].astype(np.int32) << 16)
        | (flat[:, 1].astype(np.int32) << 8)
        | flat[:, 2].astype(np.int32)
    )
    if packed.size == 0:
        return 0.0
    return float(np.unique(packed).size) / float(packed.size)


def _flat_block_fraction(context: AuthenticityContext, size: int = 16) -> float:
    """Share of blocks with essentially no variation."""
    blocks = context.blocks(size)
    if blocks.shape[0] == 0:
        return 0.0
    return float(np.mean(blocks.std(axis=1) <= 0.5))


__all__ = ["PrintRecaptureDetector", "SyntheticImageDetector"]
