"""Detecting a screenshot of an application, rather than a photograph.

The measurement that works, and the one that does not
-----------------------------------------------------
The obvious idea is that rendered UI has flat regions no camera could produce.
**Measured, that is false.** A quality-12 JPEG of a genuine photograph has
*exactly-zero-variance* 8x8 blocks across 35% of the frame, against 36% for a
real screenshot - JPEG quantisation zeroes every AC coefficient in a smooth
block, so flatness alone cannot tell the two apart at all.

What does work is a **full-width constant row**. A photograph's content varies
horizontally somewhere along any given row, so quantisation cannot flatten the
whole span; rendered chrome is written from a constant and does. Measured:

    genuine photograph            0.000
    quality-12 JPEG               0.000     <- the hard negative, separated
    twice-recompressed            0.000
    photograph of a screen        0.000
    photograph of a print         0.068
    screenshot                    0.335 - 0.454

And the false positive that measurement then found
--------------------------------------------------
Social apps pad uploads to a square. A letterboxed photograph scores 0.333 on
that measure - identical to a screenshot - so the first version of this
detector would have flagged a large fraction of perfectly honest uploads.

The fix is to ask *where* the constant rows are. Padding is contiguous at the
frame edges; application chrome is distributed through the interior. Stripping
the contiguous constant borders before measuring separates them completely:

    genuine photograph            0.000
    letterboxed photograph        0.000    <- false positive removed
    square-padded then JPEG       0.000    <- false positive removed
    padded at the top only        0.000
    photograph of a print         0.000
    screenshot                    0.175 - 0.187

Exact device resolutions are deliberately **not** used. A genuine photograph
resized to 1080x1920 matches one, so the test convicts the innocent and any
attacker can defeat it by cropping a single row.
"""

from __future__ import annotations

import numpy as np

from hamqadam_ai.authenticity.base import (
    AuthenticityContext,
    AuthenticityDetector,
    AuthenticitySignal,
    ramp,
)
from hamqadam_ai.core.config import ScreenshotDetectorConfig


class ScreenshotDetector(AuthenticityDetector):
    """Finds application chrome: constant rows through the image interior.

    Args:
        config: Anchors and the trigger threshold.
    """

    __slots__ = ("_config",)

    def __init__(self, config: ScreenshotDetectorConfig) -> None:
        super().__init__(name="screenshot")
        self._config = config

    def analyse(self, context: AuthenticityContext) -> AuthenticitySignal:
        """Measure interior constant rows and corroborating chrome structure."""
        if context.degenerate:
            return AuthenticitySignal(
                name=self.name,
                triggered=False,
                confidence=0.0,
                note="image is uniform or too small to analyse",
            )

        mask = context.constant_row_mask
        interior, pad_top, pad_bottom = _interior_fraction(mask)

        if interior is None:
            return AuthenticitySignal(
                name=self.name,
                triggered=False,
                confidence=0.0,
                measurements={
                    "padding_rows_top": float(pad_top),
                    "padding_rows_bottom": float(pad_bottom),
                },
                note="the whole image is constant rows; nothing to measure",
            )

        edges = _full_width_edges(context)
        chrome = _chrome_bands(mask, context.height)

        confidence = ramp(
            interior,
            floor=self._config.interior_rows_floor,
            ceiling=self._config.interior_rows_ceiling,
        )
        # Corroboration only. Full-width dividers and chrome bands both fire on
        # a padded photograph, so they may sharpen a verdict the interior rows
        # already support - never create one.
        if confidence > 0.0:
            corroboration = min(
                1.0,
                0.5 * ramp(float(edges), floor=1.0, ceiling=6.0)
                + 0.5 * ramp(chrome, floor=0.20, ceiling=0.80),
            )
            confidence = min(1.0, confidence * (1.0 + 0.25 * corroboration))

        return AuthenticitySignal(
            name=self.name,
            triggered=confidence >= self._config.min_confidence,
            confidence=confidence,
            measurements={
                "interior_constant_rows": interior,
                "full_width_edges": float(edges),
                "chrome_band_fraction": chrome,
                "padding_rows_top": float(pad_top),
                "padding_rows_bottom": float(pad_bottom),
            },
        )


def _interior_fraction(
    mask: np.ndarray,
) -> tuple[float | None, int, int]:
    """Constant-row fraction after stripping contiguous constant borders.

    Args:
        mask: One boolean per row, True where the row is constant.

    Returns:
        ``(interior_fraction, padding_top, padding_bottom)``. The fraction is
        ``None`` when nothing survives the strip, which means the image is
        entirely constant rows - a degenerate case rather than a screenshot.
    """
    count = int(mask.size)
    top = 0
    while top < count and bool(mask[top]):
        top += 1
    bottom = count
    while bottom > top and bool(mask[bottom - 1]):
        bottom -= 1

    interior = mask[top:bottom]
    if interior.size < 16:
        return None, top, count - bottom
    return float(interior.mean()), top, count - bottom


def _full_width_edges(context: AuthenticityContext) -> int:
    """Horizontal edges spanning almost the whole width: UI dividers.

    A photograph can contain one - a horizon, a table edge - but rarely
    several, and never several that are pixel-straight.
    """
    import cv2

    vertical_gradient = np.abs(
        cv2.Sobel(context.grey, cv2.CV_32F, 0, 1, ksize=3)
    )
    strong = vertical_gradient > 40.0
    return int(np.sum(strong.mean(axis=1) > 0.95))


def _chrome_bands(mask: np.ndarray, height: int) -> float:
    """Constant-row share within the top and bottom sixth of the frame.

    Status and navigation bars live at the extremes. Reported as
    corroboration rather than evidence, because padding lives there too.
    """
    band = max(int(height * 0.06), 4)
    if mask.size < 2 * band:
        return 0.0
    extremes = np.concatenate([mask[:band], mask[-band:]])
    return float(extremes.mean())


__all__ = ["ScreenshotDetector"]
