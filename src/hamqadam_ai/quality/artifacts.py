"""Compression and processing artefacts: pixelation and distortion.

:class:`PixelationAnalyzer`
    Block artefacts from aggressive JPEG, measured image-wide and again on the
    canonical face crop. The two differ substantially - the reference portrait
    reads 0.18 globally but 0.04 on the face - because blocking concentrates in
    the flat background regions where the codec has least to encode, and it is
    the face figure that matters for recognition.

    An earlier version also reported an inferred upscale factor here. It was
    removed after measurement showed the spectrum cannot separate upscaling
    from blur on already-compressed inputs; see
    :mod:`hamqadam_ai.quality.resolution` for the full reasoning.

:class:`DistortionAnalyzer`
    Posterisation, colour fringing, and geometric anisotropy.

Geometric anisotropy deserves an explanation. Fitting the detected landmarks
onto the canonical face template with a *full* affine and decomposing the
linear part by SVD yields two singular values - the effective scale along each
principal axis. On an undistorted face they are nearly equal; on an image
stretched non-uniformly they are not, and the log ratio measures it directly.

Crucially this is only valid on a near-frontal face. Head yaw compresses the
face horizontally through ordinary perspective, which is geometrically
indistinguishable from a horizontal squeeze applied in an editor. The metric
is therefore suppressed beyond a configured yaw and reported as unmeasured
rather than guessed at.
"""

from __future__ import annotations

import math
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import DistortionConfig, PixelationConfig
from hamqadam_ai.core.constants import CANONICAL_FACE_2D_UNIT, EPSILON
from hamqadam_ai.quality.base import (
    MetricResult,
    QualityContext,
    QualityMetric,
)
from hamqadam_ai.quality.scoring import (
    limiting_component,
    ramp_score,
    weighted_mean,
)
from hamqadam_ai.quality.spectral import (
    banding_ratio,
    blockiness,
    chromatic_aberration,
)
from hamqadam_ai.utils.geometry import Landmarks5


class PixelationAnalyzer(QualityMetric):
    """JPEG block artefacts, image-wide and on the face.

    Args:
        config: The pixelation section of the quality configuration.
    """

    def __init__(self, config: PixelationConfig) -> None:
        super().__init__("pixelation")
        self._config = config

    def analyse(self, context: QualityContext) -> MetricResult:
        """Measure JPEG block artefacts, image-wide and on the face."""
        gray = context.analysis_gray
        if gray.size == 0:
            return MetricResult.unmeasured(self.name, "image has no pixels")

        global_blocking = blockiness(gray)
        measurements: dict[str, float] = {"blockiness": global_blocking}
        sub_scores: dict[str, float] = {
            "blockiness": ramp_score(global_blocking, self._config.blockiness),
        }

        # Blocking is far more damaging on the face than in a flat background.
        # Measured on the NATIVE-resolution crop, never the canonical resample:
        # blockiness keys off the codec's 8x8 pixel grid, and interpolating the
        # crop destroys that alignment. Measured on the canonical view a
        # quality-8 JPEG reads as perfectly artefact-free.
        face_gray = context.face_crop_gray
        if face_gray is not None and face_gray.size > 0:
            face_blocking = blockiness(face_gray)
            measurements["face_blockiness"] = face_blocking
            sub_scores["face_blockiness"] = ramp_score(
                face_blocking, self._config.blockiness
            )

        score = weighted_mean(sub_scores, self._config.weights)
        worst = max(measurements.values())

        note = None
        if worst > self._config.blockiness.floor * 0.6:
            note = (
                "Visible block artefacts; the image has been heavily compressed "
                "and fine detail has been quantised away."
            )

        return MetricResult(
            name=self.name,
            score=score,
            measurements=measurements,
            sub_scores=sub_scores,
            limiting_factor=limiting_component(sub_scores, self._config.weights),
            note=note,
        )


def landmark_anisotropy(landmarks: Landmarks5) -> tuple[float, float, float]:
    """Measure non-uniform scaling in a landmark set.

    Fits a full affine mapping the canonical five-point template onto the
    detected landmarks, then takes the SVD of its 2x2 linear part. The two
    singular values are the effective scale factors along the transform's
    principal axes; their log ratio is a signed, scale-invariant measure of how
    much the face has been stretched along one axis relative to the other.

    The log ratio is used rather than a plain ratio so that a 2x horizontal
    stretch and a 2x vertical stretch give the same magnitude.

    Args:
        landmarks: The five detected keypoints.

    Returns:
        ``(anisotropy, scale_major, scale_minor)``. Anisotropy is
        ``|log(major / minor)|``, so 0.0 means perfectly uniform.
    """
    source = (CANONICAL_FACE_2D_UNIT * 100.0).astype(np.float32)
    target = np.asarray(landmarks.points, dtype=np.float32).reshape(5, 2)

    estimated = cv2.estimateAffine2D(
        source, target, method=cv2.LMEDS, refineIters=20
    )
    # The cv2 stubs declare a non-optional matrix, but the function genuinely
    # returns None on a degenerate point set. The cast restores the real
    # contract so the guard below is type-checked rather than reported as
    # unreachable - deleting the guard to please the stub would crash on the
    # exact input it exists to handle.
    matrix = cast("npt.NDArray[np.float64] | None", estimated[0])
    if matrix is None:
        return 0.0, 1.0, 1.0

    linear = np.asarray(matrix[:, :2], dtype=np.float64)
    singular_values = np.linalg.svd(linear, compute_uv=False)
    major = float(max(singular_values))
    minor = float(min(singular_values))
    if minor <= EPSILON:
        return 1.0, major, minor

    return float(abs(math.log(major / minor))), major, minor


class DistortionAnalyzer(QualityMetric):
    """Posterisation, colour fringing and geometric anisotropy.

    Args:
        config: The distortion section of the quality configuration.
    """

    def __init__(self, config: DistortionConfig) -> None:
        super().__init__("distortion")
        self._config = config

    def analyse(self, context: QualityContext) -> MetricResult:
        """Measure processing artefacts and geometric stretching."""
        gray = context.analysis_gray
        if gray.size == 0:
            return MetricResult.unmeasured(self.name, "image has no pixels")

        banding = banding_ratio(gray)
        fringing = chromatic_aberration(context.analysis_image)

        measurements: dict[str, float] = {
            "banding_ratio": banding,
            "chromatic_aberration": fringing,
        }
        sub_scores: dict[str, float] = {
            "banding": ramp_score(banding, self._config.banding),
            "chromatic": ramp_score(fringing, self._config.chromatic_aberration),
        }

        # Notes accumulate. An earlier version assigned rather than appended,
        # so a banding note silently discarded the explanation of *why* the
        # geometric term had been skipped - losing exactly the disclosure that
        # made the omission honest.
        notes: list[str] = []

        geometric_note = self._add_geometric_term(context, measurements, sub_scores)
        if geometric_note:
            notes.append(geometric_note)

        if banding > self._config.banding.floor * 0.7:
            notes.append(
                "The luminance histogram is heavily combed, which indicates the "
                "image has been posterised by repeated re-saving or a filter."
            )

        score = weighted_mean(sub_scores, self._config.weights)
        note = " ".join(notes) if notes else None

        return MetricResult(
            name=self.name,
            score=score,
            measurements=measurements,
            sub_scores=sub_scores,
            limiting_factor=limiting_component(sub_scores, self._config.weights),
            note=note,
        )

    def _add_geometric_term(
        self,
        context: QualityContext,
        measurements: dict[str, float],
        sub_scores: dict[str, float],
    ) -> str | None:
        """Add the anisotropy term when the head is frontal enough to trust it.

        Returns a note when the term had to be skipped, so the omission is
        visible in the response rather than silent.
        """
        if context.landmarks is None:
            return None

        yaw = context.yaw_degrees
        if yaw is not None and abs(yaw) > self._config.max_yaw_for_anisotropy:
            measurements["geometric_anisotropy_skipped_yaw"] = float(abs(yaw))
            return (
                f"Geometric distortion was not assessed: at {abs(yaw):.0f} degrees "
                f"of yaw, perspective foreshortening is indistinguishable from a "
                f"horizontally squeezed image."
            )

        anisotropy, major, minor = landmark_anisotropy(context.landmarks)
        measurements["geometric_anisotropy"] = anisotropy
        measurements["scale_major"] = major
        measurements["scale_minor"] = minor
        sub_scores["geometric"] = ramp_score(
            anisotropy, self._config.geometric_anisotropy
        )

        if anisotropy > self._config.geometric_anisotropy.floor * 0.7:
            stretch = math.exp(anisotropy)
            return (
                f"The facial geometry is stretched by roughly {stretch:.2f}x along "
                f"one axis, which indicates the image was resized non-uniformly."
            )
        return None


__all__ = ["DistortionAnalyzer", "PixelationAnalyzer", "landmark_anisotropy"]
