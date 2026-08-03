"""MODULE 2 - image and face quality assessment.

Architecture
------------
Eight independent metric families, each a self-contained analyser producing a
:class:`~hamqadam_ai.quality.base.MetricResult`, combined by a single
:class:`~hamqadam_ai.quality.aggregator.QualityAggregator`.

===============  ==========================================================
Family           What it measures
===============  ==========================================================
``blur``         Whole-image focus: Laplacian, Tenengrad, spectral
``sharpness``    Face-region focus, specifically the eye band
``brightness``   Mean luminance plus shadow / highlight clipping
``contrast``     RMS, dynamic range, histogram entropy
``noise``        Additive sensor noise, absolute and signal-relative
``resolution``   Genuine as-captured resolution, not the nominal pixel count
``pixelation``   Block artefacts and inferred upscaling
``distortion``   Posterisation, colour fringing, geometric anisotropy
===============  ==========================================================

Two properties hold throughout:

**Scale normalisation.** Focus metrics are computed on the face crop resampled
to a canonical size. A raw Laplacian variance is not comparable across
resolutions, so a single set of thresholds could not otherwise be valid for
both a 300 px and a 4000 px source.

**Every raw measurement is reported alongside its score.** A composite of 42
that cannot be explained is useless to a human reviewer and to the user, who
needs to know whether to move into better light or clean the lens.
"""

from __future__ import annotations

from hamqadam_ai.quality.aggregator import QualityAggregator
from hamqadam_ai.quality.artifacts import DistortionAnalyzer, PixelationAnalyzer
from hamqadam_ai.quality.base import (
    MetricResult,
    QualityContext,
    QualityMetric,
)
from hamqadam_ai.quality.blur import BlurAnalyzer, SharpnessAnalyzer
from hamqadam_ai.quality.exposure import BrightnessAnalyzer, ContrastAnalyzer
from hamqadam_ai.quality.noise import NoiseAnalyzer, estimate_noise_sigma
from hamqadam_ai.quality.resolution import ResolutionAnalyzer, detail_energy
from hamqadam_ai.quality.scoring import band_score, ramp_score, smoothstep

__all__ = [
    "BlurAnalyzer",
    "BrightnessAnalyzer",
    "ContrastAnalyzer",
    "DistortionAnalyzer",
    "MetricResult",
    "NoiseAnalyzer",
    "PixelationAnalyzer",
    "QualityAggregator",
    "QualityContext",
    "QualityMetric",
    "ResolutionAnalyzer",
    "SharpnessAnalyzer",
    "band_score",
    "detail_energy",
    "estimate_noise_sigma",
    "ramp_score",
    "smoothstep",
]
