"""The quality-metric port and its shared value objects.

Contract every analyser honours
-------------------------------
* It takes a :class:`QualityContext` - the pre-computed views of the image that
  several analysers need in common - and returns one :class:`MetricResult`.
* It **never raises for difficult input**. A degenerate image yields a
  ``MetricResult`` with a low score and a note explaining why, because the
  pipeline must still produce a complete quality report for the other six
  images in the request.
* It reports every raw measurement alongside the derived score. Returning only
  the score makes a rejection impossible to explain.

Why a shared context
--------------------
Grayscale conversion, the canonical face crop and the gradient field are each
needed by three or four analysers. Computing them once per image rather than
once per analyser removes roughly 60% of the module's arithmetic, and it
guarantees every metric is measured on identical pixels - which matters,
because a metric computed on a slightly different crop is not comparable with
its own configured anchors.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.constants import EPSILON
from hamqadam_ai.quality.spectral import RadialSpectrum, radial_power_spectrum
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5
from hamqadam_ai.utils.image_ops import (
    crop_with_margin,
    gradient_energy,
    resize_long_side,
    to_grayscale,
)

BgrImage = npt.NDArray[np.uint8]
GrayImage = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One quality dimension: its score, its evidence and its verdict.

    Attributes:
        name: Metric family name, matching the aggregation weight key.
        score: Normalised quality in ``[0, 1]``, 1 being best.
        measurements: Raw values behind the score, keyed by sub-metric. These
            are the numbers an engineer recalibrates against, so they are
            reported verbatim rather than rounded away.
        sub_scores: Per-sub-metric normalised scores, before weighting.
        limiting_factor: The sub-metric contributing the largest deficit.
        measured: False when the metric could not be computed at all - no
            landmarks for geometric anisotropy, for instance. An unmeasured
            metric is excluded from the composite rather than scored zero.
        note: Human-readable explanation, present when something is unusual.
    """

    name: str
    score: float
    measurements: dict[str, float] = field(default_factory=dict)
    sub_scores: dict[str, float] = field(default_factory=dict)
    limiting_factor: str | None = None
    measured: bool = True
    note: str | None = None

    @classmethod
    def unmeasured(cls, name: str, reason: str) -> MetricResult:
        """Build a result for a metric that could not be computed.

        Scored 1.0 but flagged ``measured=False``. The score is never used - the
        aggregator drops unmeasured metrics and renormalises the remaining
        weights - but a neutral value avoids any accidental contribution should
        a future caller read it directly.
        """
        return cls(name=name, score=1.0, measured=False, note=reason)

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the API response."""
        payload: dict[str, Any] = {
            "score": round(self.score * 100.0, 2),
            "measured": self.measured,
        }
        if self.measurements:
            payload["measurements"] = {
                key: round(float(value), 4) for key, value in self.measurements.items()
            }
        if self.sub_scores:
            payload["sub_scores"] = {
                key: round(float(value), 4) for key, value in self.sub_scores.items()
            }
        if self.limiting_factor:
            payload["limiting_factor"] = self.limiting_factor
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass
class QualityContext:
    """Pre-computed views of one image, shared across every analyser.

    Constructed once per image by
    :class:`~hamqadam_ai.services.quality_service.QualityService`. All the
    expensive derived views are :func:`functools.cached_property`, so an
    analyser that does not need the FFT never pays for it.

    Args:
        image: The full source image, BGR uint8.
        face_box: The primary face bounds in source coordinates, when a face
            was detected. Face-region metrics degrade gracefully without it.
        landmarks: The five keypoints, when available. Required for the eye
            band and for geometric anisotropy.
        yaw_degrees: Head yaw, used to suppress the anisotropy measurement on
            a turned head where perspective foreshortening is
            indistinguishable from a stretched image.
        canonical_size: Edge length the face crop is resampled to.
        crop_margin: Fractional expansion applied before cropping the face.
        analysis_long_side: Longest side the global analysis image is reduced
            to before spectral and block work.
    """

    image: BgrImage
    face_box: BoundingBox | None = None
    landmarks: Landmarks5 | None = None
    yaw_degrees: float | None = None
    canonical_size: int = 160
    crop_margin: float = 0.15
    analysis_long_side: int = 1024

    # -- Source-level views ------------------------------------------------ #

    @cached_property
    def height(self) -> int:
        """Source image height in pixels."""
        return int(self.image.shape[0])

    @cached_property
    def width(self) -> int:
        """Source image width in pixels."""
        return int(self.image.shape[1])

    @cached_property
    def has_face(self) -> bool:
        """Whether a face box is available for region-specific metrics."""
        return self.face_box is not None

    # -- Global analysis view ---------------------------------------------- #

    @cached_property
    def analysis_image(self) -> BgrImage:
        """The source reduced to ``analysis_long_side`` for global metrics.

        Spectral analysis and block-artefact detection on a 12 MP image cost
        several hundred milliseconds and reveal nothing a 1 MP view does not.
        The reduction is *not* applied to focus metrics, which are measured on
        the canonical face crop instead.
        """
        reduced, _ = resize_long_side(self.image, self.analysis_long_side)
        return reduced

    @cached_property
    def analysis_gray(self) -> GrayImage:
        """Luminance of :attr:`analysis_image`."""
        return to_grayscale(self.analysis_image)

    @cached_property
    def analysis_gradient(self) -> FloatArray:
        """Sobel gradient magnitude of :attr:`analysis_gray`."""
        return gradient_energy(self.analysis_gray)

    @cached_property
    def analysis_scale(self) -> float:
        """Factor mapping source coordinates into :attr:`analysis_image`."""
        return float(self.analysis_image.shape[1]) / float(max(self.width, 1))

    @cached_property
    def analysis_face_box(self) -> BoundingBox | None:
        """The face box expressed in :attr:`analysis_image` coordinates.

        Needed so the face-versus-scene focus comparison is made at a single
        sampling density. Comparing a metric computed on the canonical crop
        against one computed on the full image would compare two different
        scales and produce a ratio that means nothing.
        """
        if self.face_box is None:
            return None
        scaled = self.face_box.scale(self.analysis_scale)
        return scaled.clip(
            self.analysis_image.shape[1], self.analysis_image.shape[0]
        )

    @cached_property
    def radial_spectrum(self) -> RadialSpectrum:
        """Radially-averaged power spectrum of the analysis image.

        Cached because three separate metric families need it - blur's
        high-frequency ratio, resolution's effective ratio and pixelation's
        upscale factor - and the transform is the single most expensive
        operation in the module.
        """
        return radial_power_spectrum(self.analysis_gray)

    # -- Face-region views -------------------------------------------------- #

    @cached_property
    def face_crop(self) -> BgrImage | None:
        """The face region at source resolution, expanded by the margin.

        ``None`` when no face was detected.
        """
        if self.face_box is None:
            return None
        crop, _ = crop_with_margin(
            self.image, self.face_box, margin=self.crop_margin, square=True
        )
        return crop

    @cached_property
    def face_native_short_side(self) -> float:
        """Shorter side of the detector box in genuine source pixels.

        Reported before any resampling, because it is the number that decides
        whether the embedding stage has enough real detail to work with.
        """
        return self.face_box.short_side if self.face_box is not None else 0.0

    @cached_property
    def canonical_face(self) -> BgrImage | None:
        """The face crop resampled to a fixed square size.

        This is the single most important design decision in the module. Focus
        measures scale with sampling density, so the same face photographed at
        4000 px and at 300 px produces Laplacian variances an order of
        magnitude apart. Resampling to a fixed size makes one set of configured
        anchors valid for every input resolution.

        Downscaling uses ``INTER_AREA`` and upscaling ``INTER_LINEAR``: using
        the wrong kernel to downscale aliases high-frequency detail and
        measurably inflates the apparent sharpness of a blurred image.
        """
        crop = self.face_crop
        if crop is None or crop.size == 0:
            return None
        size = self.canonical_size
        current = max(crop.shape[0], crop.shape[1])
        interpolation = cv2.INTER_AREA if current > size else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (size, size), interpolation=interpolation)
        return resized.astype(np.uint8, copy=False)

    @cached_property
    def canonical_gray(self) -> GrayImage | None:
        """Luminance of :attr:`canonical_face`."""
        canonical = self.canonical_face
        return None if canonical is None else to_grayscale(canonical)

    @cached_property
    def face_crop_gray(self) -> GrayImage | None:
        """Luminance of the face crop at its **native** resolution.

        Distinct from :attr:`canonical_gray` and needed for exactly one thing:
        block-artefact detection. Blockiness keys off the codec's 8x8 pixel
        grid, and any resampling destroys that alignment - measured on the
        canonical crop, a quality-8 JPEG reads as perfectly artefact-free
        because the grid has been interpolated away. Every other metric wants
        the scale-normalised view; this one must have the original pixels.
        """
        crop = self.face_crop
        return None if crop is None or crop.size == 0 else to_grayscale(crop)

    @cached_property
    def canonical_upsampled(self) -> bool:
        """Whether producing the canonical crop required upscaling.

        A face crop smaller than the canonical size has been stretched, which
        *raises* its apparent Laplacian variance relative to its true detail.
        The resolution analyser reports the true size separately, but callers
        reading the focus metrics need to know the sample was interpolated.
        """
        crop = self.face_crop
        if crop is None:
            return False
        return bool(max(crop.shape[0], crop.shape[1]) < self.canonical_size)

    @cached_property
    def eye_band(self) -> GrayImage | None:
        """The horizontal band of the canonical crop containing the eyes.

        The eye region carries the highest spatial frequencies on a face -
        lashes, iris boundary, lid edges - which makes it far more diagnostic
        of focus than the cheeks or forehead. Cropping to it is what lets the
        sharpness metric distinguish a soft-focus portrait from a sharp one
        where the subject simply has smooth skin.

        Derived from the landmark positions when they are available, falling
        back to the anatomical average band otherwise.
        """
        canonical = self.canonical_gray
        if canonical is None:
            return None

        size = self.canonical_size
        if self.landmarks is not None and self.face_box is not None:
            crop = self.face_crop
            if crop is not None and crop.shape[0] > 0:
                # Map the eye centre from source coordinates into the square,
                # margin-expanded crop and then into the canonical resample.
                expanded = self.face_box.expand(self.crop_margin).to_square()
                scale = size / max(expanded.width, EPSILON)
                _, eye_y = self.landmarks.eye_center
                centre = (eye_y - expanded.y1) * scale
                half = max(8.0, size * 0.12)
                top = int(np.clip(centre - half, 0, size - 4))
                bottom = int(np.clip(centre + half, top + 4, size))
                return canonical[top:bottom, :]

        # Anatomical fallback: the eyes sit around 38-52% down a face crop.
        top = int(size * 0.30)
        bottom = int(size * 0.56)
        return canonical[top:bottom, :]

    def describe(self) -> dict[str, Any]:
        """PII-free summary of what this context makes available."""
        return {
            "width": self.width,
            "height": self.height,
            "has_face": self.has_face,
            "has_landmarks": self.landmarks is not None,
            "face_short_side": round(self.face_native_short_side, 1),
            "canonical_upsampled": self.canonical_upsampled,
        }


class QualityMetric(abc.ABC):
    """Abstract analyser for one quality dimension.

    Args:
        name: Family name, matching the key used in the aggregation weights.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def analyse(self, context: QualityContext) -> MetricResult:
        """Measure this dimension.

        Implementations must not raise for difficult input; return a low score
        with an explanatory note instead. :meth:`safe_analyse` enforces that
        for unexpected exceptions, but a deliberate degradation should be
        handled explicitly so the note is informative.
        """

    def safe_analyse(self, context: QualityContext) -> MetricResult:
        """Run :meth:`analyse`, converting an unexpected failure into a result.

        One analyser hitting a numerical edge case must not deprive the caller
        of the other seven dimensions.
        """
        try:
            return self.analyse(context)
        except Exception as exc:  # noqa: BLE001 - degrade, never fail the request
            from hamqadam_ai.logging.setup import get_logger

            get_logger(__name__).warning(
                "quality.metric_failed",
                metric=self.name,
                reason=str(exc),
                exception=type(exc).__name__,
            )
            return MetricResult.unmeasured(
                self.name, f"analysis failed: {type(exc).__name__}"
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"


def percentile_range(gray: GrayImage, low: float = 1.0, high: float = 99.0) -> float:
    """Span between two luminance percentiles.

    Preferred over ``max - min`` for dynamic range: a single hot pixel or one
    blown specular highlight would otherwise report a full 0-255 range on an
    image that is visibly flat.
    """
    lower, upper = np.percentile(gray, [low, high])
    return float(upper - lower)


def shannon_entropy(gray: GrayImage) -> float:
    """Shannon entropy of the luminance histogram, in bits.

    Complements the standard deviation. A posterised or washed-out image can
    show an acceptable std while occupying only a handful of distinct levels;
    entropy catches that because it counts *how many* levels are genuinely in
    use, not merely how far apart they are.
    """
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(histogram.sum())
    if total <= 0.0:
        return 0.0
    probabilities = histogram / total
    nonzero = probabilities[probabilities > 0.0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


__all__ = [
    "BgrImage",
    "FloatArray",
    "GrayImage",
    "MetricResult",
    "QualityContext",
    "QualityMetric",
    "percentile_range",
    "shannon_entropy",
]
