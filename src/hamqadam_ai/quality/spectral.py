"""Frequency-domain analysis, shared by three metric families.

The radially-averaged power spectrum answers three separate questions from one
FFT, which is why it lives here rather than inside any single analyser:

* **Blur** - how much of the total energy sits above a quarter of Nyquist.
  Nearly content-independent, so it distinguishes a genuinely smooth subject
  from an out-of-focus one in a way the Laplacian cannot.
* **Resolution** - the share of energy above half Nyquist: how much genuine
  detail an image carries relative to its nominal pixel count. An enlarged
  thumbnail has the pixel count but not the information.
* **Pixelation** - phase-invariant blockiness on the codec's 8x8 grid, which is
  a spatial rather than spectral measure but belongs with the other
  compression diagnostics.

An earlier version also tried to infer an explicit *upscale factor* from a
cliff in the spectrum. It was withdrawn after measurement showed no clean cliff
survives on already-compressed photographs; see
:meth:`RadialSpectrum.detail_energy` for the full reasoning.

Everything here is pure NumPy and free of configuration, so it is directly
unit-testable against synthetic signals of known bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.constants import EPSILON

FloatArray = npt.NDArray[np.float32]
GrayImage = npt.NDArray[np.uint8]

#: Longest side the image is reduced to before the FFT. A 1024x1024 transform
#: costs ~50 ms on CPU and reveals nothing about global focus that a 512x512
#: one does not; the metrics derived here are all ratios, so the reduction
#: does not bias them.
SPECTRAL_ANALYSIS_SIZE = 512

#: Fraction of Nyquist above which energy counts as "high frequency" for the
#: blur metric. A quarter is low enough to survive JPEG's chroma subsampling
#: and high enough that a defocused image has essentially nothing there.
HIGH_FREQUENCY_CUTOFF = 0.25

#: Band lower bound for the "genuine detail" measure used by the resolution
#: metric. Half Nyquist is the frequency an image upscaled 2x cannot populate
#: from real information.
DETAIL_CUTOFF = 0.50

#: Ceiling on the blockiness ratio. Beyond this the artefact is total and the
#: exact figure carries no further information, so it is capped to keep the
#: score mapping's anchors on a sane scale.
_MAX_BLOCKINESS = 4.0


@dataclass(frozen=True, slots=True)
class RadialSpectrum:
    """A radially-averaged power spectrum, normalised to unit total energy.

    Attributes:
        profile: Mean power at each integer radius, index 0 being DC.
        normalised_radii: Each bin's radius as a fraction of Nyquist, in
            ``[0, 1]``.
        size: Edge length of the square transform the profile came from.
    """

    profile: FloatArray
    normalised_radii: FloatArray
    size: int

    def energy_above(self, cutoff: float) -> float:
        """Fraction of total energy at or above ``cutoff`` x Nyquist.

        Args:
            cutoff: Threshold as a fraction of Nyquist, in ``[0, 1]``.

        Returns:
            The energy fraction in ``[0, 1]``.
        """
        # Weight each radial bin by its circumference: a bin at radius r
        # aggregates ~2*pi*r frequency samples, and ignoring that would
        # massively over-weight the low-frequency bins.
        weights = np.maximum(self.normalised_radii, EPSILON)
        weighted = self.profile * weights
        total = float(weighted.sum())
        if total <= 0.0:
            return 0.0
        mask = self.normalised_radii >= cutoff
        return float(weighted[mask].sum() / total)

    def detail_energy(self, cutoff: float = 0.5) -> float:
        """Share of energy in the top band - the "real detail" measure.

        This replaced an earlier attempt to infer an explicit *upscale factor*
        by locating a cliff in the spectrum. That attempt was withdrawn after
        measurement: on real, already-JPEG-compressed photographs a 2x cubic
        upscale produces no clean cliff at half Nyquist, because interpolation
        ringing and codec noise put energy back above the theoretical cutoff.
        The measured profile of a 2x-upscaled portrait was indistinguishable in
        shape from a mildly blurred one.

        What *can* be measured reliably is how much genuine high-frequency
        detail an image carries relative to its nominal pixel count. Blur and
        upscaling both reduce it, and for the purpose this service exists for -
        "does this face carry enough information to recognise" - the two are
        the same defect. Reporting the measurable quantity under an honest name
        is better than reporting an inferred one that does not survive contact
        with real inputs.

        Args:
            cutoff: Band lower bound as a fraction of Nyquist.

        Returns:
            Energy fraction in ``[0, 1]``. Measured on a sharp 512x600 portrait
            this is around 4e-3; a 9x9 Gaussian blur drops it to 9e-6.
        """
        return self.energy_above(cutoff)


def radial_power_spectrum(
    gray: GrayImage, *, analysis_size: int = SPECTRAL_ANALYSIS_SIZE
) -> RadialSpectrum:
    """Compute the radially-averaged power spectrum of an image.

    Args:
        gray: Single-channel uint8 image.
        analysis_size: The image is resampled to this square size first, so the
            returned radii are comparable across inputs of any shape.

    Returns:
        The radial profile and its normalised radii.
    """
    size = int(analysis_size)
    resized = cv2.resize(
        gray,
        (size, size),
        interpolation=cv2.INTER_AREA if max(gray.shape) > size else cv2.INTER_LINEAR,
    ).astype(np.float32)

    # A 2D Hann window. Without it the implicit discontinuity at the image
    # border produces a strong cross artefact through the spectrum that is
    # indistinguishable from genuine high-frequency detail - which would make
    # every image look sharp.
    window_1d = np.hanning(size).astype(np.float32)
    resized = resized * np.outer(window_1d, window_1d)

    spectrum = np.fft.fftshift(np.fft.fft2(resized))
    power = (np.abs(spectrum) ** 2).astype(np.float32)

    centre = size // 2
    y_indices, x_indices = np.indices((size, size))
    radius = np.hypot(x_indices - centre, y_indices - centre).astype(np.int32)

    bin_count = centre + 1
    radius_flat = radius.ravel()
    power_flat = power.ravel()
    inside = radius_flat < bin_count

    totals = np.bincount(
        radius_flat[inside], weights=power_flat[inside], minlength=bin_count
    )
    counts = np.bincount(radius_flat[inside], minlength=bin_count)
    profile = (totals / np.maximum(counts, 1)).astype(np.float32)

    radii = (np.arange(bin_count, dtype=np.float32) / max(centre, 1)).astype(np.float32)
    return RadialSpectrum(profile=profile, normalised_radii=radii, size=size)


def high_frequency_ratio(
    gray: GrayImage, *, cutoff: float = HIGH_FREQUENCY_CUTOFF
) -> float:
    """Fraction of spectral energy above ``cutoff`` x Nyquist.

    Convenience wrapper for callers that need only the blur term.
    """
    return radial_power_spectrum(gray).energy_above(cutoff)


def blockiness(gray: GrayImage, *, period: int = 8) -> float:
    """Excess edge energy on the JPEG block grid, relative to off-grid.

    JPEG quantises 8x8 blocks independently, so heavy compression leaves a
    step discontinuity at every block boundary. Measuring the mean absolute
    first difference *on* the grid and dividing by the same quantity *off* the
    grid isolates that artefact from the image's own content: a picture of a
    brick wall has strong periodic edges but they do not align to the codec's
    grid.

    Phase invariance
    ----------------
    The grid's *phase* is searched rather than assumed. This is not a
    refinement, it is required for correctness: a face crop taken at an
    arbitrary offset shifts the grid by ``offset % 8``, and an implementation
    that only tests phase 0 reports a severely blocked quality-8 JPEG as
    perfectly clean seven times out of eight. The maximum excess across all
    phases is taken, since only one phase can be the real grid and the others
    measure content.

    Args:
        gray: Single-channel uint8 image.
        period: Block size. 8 for JPEG; 16 would target some video codecs.

    Returns:
        Excess ratio, floored at 0 and capped at :data:`_MAX_BLOCKINESS`.
        Measured on the reference portrait: 0.18 for the source JPEG, 0.61 at
        quality 30, 2.4 at quality 8.
    """
    if gray.shape[0] < period * 3 or gray.shape[1] < period * 3:
        return 0.0

    values = gray.astype(np.float32)

    # Vertical block boundaries are differences between adjacent columns.
    column_diff = np.abs(np.diff(values, axis=1)).mean(axis=0)
    row_diff = np.abs(np.diff(values, axis=0)).mean(axis=1)

    def excess_at(differences: npt.NDArray[np.floating[Any]], phase: int) -> float:
        """Excess on-grid energy for one candidate grid phase."""
        indices = np.arange(differences.size)
        on_grid = (indices % period) == phase
        if not on_grid.any() or on_grid.all():
            return 0.0

        grid_mean = float(differences[on_grid].mean())
        other_mean = float(differences[~on_grid].mean())

        if other_mean <= EPSILON:
            # Zero off-grid variation with non-zero on-grid variation is the
            # *strongest* possible blocking signal - a perfectly flat image
            # made entirely of hard-edged blocks. An earlier version returned
            # 0.0 here, which reported a synthetic block pattern as artefact
            # free: the guard was protecting the division and accidentally
            # inverted the metric at its own extreme.
            return _MAX_BLOCKINESS if grid_mean > EPSILON else 0.0

        return min(_MAX_BLOCKINESS, max(0.0, (grid_mean - other_mean) / other_mean))

    def best_excess(differences: npt.NDArray[np.floating[Any]]) -> float:
        if differences.size < period * 3:
            return 0.0
        return max(excess_at(differences, phase) for phase in range(period))

    return float((best_excess(column_diff) + best_excess(row_diff)) / 2.0)


def banding_ratio(gray: GrayImage) -> float:
    """Share of unoccupied histogram levels inside the image's tonal range.

    Posterisation - from aggressive re-saving, a heavy filter, or an 8-bit
    export of an already-quantised source - leaves comb-like gaps in the
    histogram. Restricting the count to the occupied range is what stops a
    legitimately low-key or high-key photograph from being flagged: a dark
    image uses few levels, but the ones it uses are contiguous.

    Returns:
        Fraction of empty levels within the occupied range, in ``[0, 1]``.
    """
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    occupied = np.flatnonzero(histogram > 0)
    if occupied.size < 8:
        # Fewer than eight distinct levels is itself severe posterisation.
        return 1.0

    span = histogram[occupied[0] : occupied[-1] + 1]
    if span.size <= 2:
        return 1.0
    return float(np.count_nonzero(span == 0) / span.size)


def chromatic_aberration(image: npt.NDArray[np.uint8]) -> float:
    """Colour fringing at high-contrast edges, in 0-255 units.

    Lateral chromatic aberration displaces the red and blue channels relative
    to each other, producing coloured fringes that appear only at sharp edges.
    Comparing the mean red-blue divergence *on* strong edges against the same
    quantity in flat regions isolates the optical artefact from the subject's
    own colour: a red jumper against a blue wall has a huge R-B difference
    everywhere, not just at edges.

    Returns:
        Excess R-B divergence at edges. Around 0-2 for a clean phone capture,
        above 10 for heavy fringing or a badly resampled image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return 0.0

    blue = image[:, :, 0].astype(np.float32)
    red = image[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    magnitude = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    if magnitude.size == 0:
        return 0.0

    edge_threshold = float(np.percentile(magnitude, 97.0))
    flat_threshold = float(np.percentile(magnitude, 40.0))
    if edge_threshold <= flat_threshold:
        return 0.0

    divergence = np.abs(red - blue)
    edge_mask = magnitude >= edge_threshold
    flat_mask = magnitude <= flat_threshold

    if not edge_mask.any() or not flat_mask.any():
        return 0.0

    return float(max(0.0, divergence[edge_mask].mean() - divergence[flat_mask].mean()))


__all__ = [
    "HIGH_FREQUENCY_CUTOFF",
    "SPECTRAL_ANALYSIS_SIZE",
    "RadialSpectrum",
    "banding_ratio",
    "blockiness",
    "chromatic_aberration",
    "high_frequency_ratio",
    "radial_power_spectrum",
]
