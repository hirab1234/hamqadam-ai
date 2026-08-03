"""Resolution analysis - genuine detail, not the nominal pixel count.

The distinction is the whole point. A 200x200 face crop upscaled to 1000x1000
reports a face 1000 px across and will sail past any pixel-count check, while
carrying exactly the detail of the 200 px original. That matters directly:
ArcFace is trained at 112x112, and feeding it an interpolated crop produces a
confident embedding of information that was never captured.

Three measurements:

* **Face short side** - the detector box in genuine source pixels.
* **Interocular distance** - the standard biometric scale reference.
  ISO/IEC 19794-5 asks for 60 px in a token image; 40 is workable for a
  consumer selfie flow, below ~22 the embedding is unreliable.
* **Detail energy** - the share of spectral energy above half Nyquist. This is
  the measurement that catches an upscaled thumbnail, and it needs no
  reference image to do it.

A note on what this metric does *not* claim
-------------------------------------------
An earlier version reported an explicit "inferred upscale factor". It was
removed after measurement: on real, already-compressed photographs a 2x cubic
upscale produces no clean spectral cliff, because interpolation ringing and
codec noise repopulate the band above the theoretical cutoff. Its radial
profile was indistinguishable in shape from a mildly blurred capture.

So this analyser measures how much genuine detail is present relative to the
nominal pixel count, and says exactly that. Blur and upscaling both reduce it;
for the question this service exists to answer - *does this face carry enough
information to recognise* - they are the same defect, and pretending to
distinguish them would be a claim the data does not support.
"""

from __future__ import annotations

from hamqadam_ai.core.config import ResolutionConfig
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
from hamqadam_ai.quality.spectral import DETAIL_CUTOFF, radial_power_spectrum


def detail_energy(gray: GrayImage) -> float:
    """Share of spectral energy above half Nyquist.

    Convenience wrapper for callers outside the analyser.

    Returns:
        Around 4e-3 for a sharp 512x600 portrait, 5e-4 after a 2x upscale, and
        below 1e-5 once the image is genuinely blurred.
    """
    return radial_power_spectrum(gray).detail_energy(DETAIL_CUTOFF)


class ResolutionAnalyzer(QualityMetric):
    """Genuine, as-captured resolution of the face.

    Args:
        config: The resolution section of the quality configuration.
    """

    def __init__(self, config: ResolutionConfig) -> None:
        super().__init__("resolution")
        self._config = config

    def analyse(self, context: QualityContext) -> MetricResult:
        """Measure the face's real detail content."""
        detail = context.radial_spectrum.detail_energy(DETAIL_CUTOFF)

        measurements: dict[str, float] = {
            "detail_energy": detail,
            "image_width": float(context.width),
            "image_height": float(context.height),
            "megapixels": float(context.width * context.height) / 1_000_000.0,
        }
        sub_scores: dict[str, float] = {
            "detail": ramp_score(detail, self._config.detail_energy),
        }

        if not context.has_face:
            # Without a face the two size terms are undefined. The spectral
            # term still applies and the weights renormalise around it.
            return MetricResult(
                name=self.name,
                score=weighted_mean(sub_scores, self._config.weights),
                measurements=measurements,
                sub_scores=sub_scores,
                limiting_factor="detail",
                note=(
                    "No face was detected, so only the image-wide detail content "
                    "could be assessed."
                ),
            )

        face_short_side = context.face_native_short_side
        measurements["face_short_side"] = face_short_side
        measurements["face_area_ratio"] = float(
            (context.face_box.area if context.face_box else 0.0)
            / max(context.width * context.height, EPSILON)
        )
        sub_scores["face_size"] = ramp_score(
            face_short_side, self._config.face_short_side
        )

        if context.landmarks is not None:
            interocular = context.landmarks.interocular_distance
            measurements["interocular_pixels"] = interocular
            sub_scores["interocular"] = ramp_score(
                interocular, self._config.interocular
            )

        score = weighted_mean(sub_scores, self._config.weights)

        note = None
        if sub_scores["detail"] < 0.4 and sub_scores.get("face_size", 0.0) > 0.8:
            note = (
                f"The face is nominally {face_short_side:.0f} px across but carries "
                f"little genuine high-frequency detail. The image has most likely "
                f"been enlarged from a smaller original, or is out of focus - the "
                f"spectrum cannot distinguish the two."
            )
        elif face_short_side < self._config.face_short_side.floor:
            note = (
                "The face occupies too few pixels for a reliable biometric "
                "comparison."
            )
        elif context.canonical_upsampled:
            note = (
                "The face crop is smaller than the analysis size and had to be "
                "interpolated; focus metrics for this image are optimistic."
            )

        return MetricResult(
            name=self.name,
            score=score,
            measurements=measurements,
            sub_scores=sub_scores,
            limiting_factor=limiting_component(sub_scores, self._config.weights),
            note=note,
        )


__all__ = ["ResolutionAnalyzer", "detail_energy"]
