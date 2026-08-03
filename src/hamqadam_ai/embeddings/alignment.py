"""Warping a detected face onto ArcFace's canonical input frame.

Why this module exists separately
---------------------------------
Alignment is the single largest controllable factor in recognition accuracy
after the choice of model. ArcFace was trained exclusively on faces warped onto
a fixed five-point template at 112x112; feeding it a plain box crop costs
several points of verification accuracy, and feeding it a *differently* warped
crop costs more. Because it matters that much, and because Module 6 needs to
run the identical transform on a CNIC portrait, it is a first-class module
rather than a private helper inside the adapter.

Similarity, not affine
----------------------
The warp is estimated as a **similarity** transform - rotation, uniform scale
and translation, four degrees of freedom. A full affine has six and would
additionally shear and independently scale the axes to fit the template
exactly. That sounds like a better fit and is actively harmful: it normalises
away the inter-landmark geometry that distinguishes one person's face from
another's, which is exactly the signal the recogniser depends on. The residual
error a similarity transform leaves behind is information, not noise, and
:func:`alignment_residual` reports it.

The fallback path
-----------------
A landmark-free detector (the OpenCV DNN or a Haar detection with no eye pair)
still produces a usable box. The face can be centred in the template frame from
the box alone, which recovers most of the scale normalisation but none of the
rotation normalisation. Results from that path are flagged ``aligned=False``
throughout, never silently mixed in with properly aligned ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.constants import ARCFACE_REFERENCE_LANDMARKS_112, EPSILON
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5
from hamqadam_ai.utils.image_ops import crop_with_margin

BgrImage = npt.NDArray[np.uint8]

#: Fill colour for regions of the template frame that fall outside the source
#: image. Mid-grey rather than black, which would create a hard synthetic edge
#: the network has never seen in training.
_BORDER_VALUE = (114, 114, 114)


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """An aligned crop and how well the warp fitted.

    Attributes:
        crop: The aligned BGR image at the template size.
        aligned: True when the landmark template was used; False for the
            box-only fallback.
        residual: Mean landmark-to-template distance after warping, in units
            of interocular distance. ``None`` for the fallback path.
        matrix: The 2x3 transform applied, retained so a caller can map
            coordinates back into the source frame.
        reason: Why the fallback was taken, when it was.
    """

    crop: BgrImage
    aligned: bool
    residual: float | None = None
    matrix: npt.NDArray[np.float32] | None = None
    reason: str | None = None

    @property
    def trustworthy(self) -> bool:
        """Whether the warp fitted well enough for the crop to be relied on."""
        return self.aligned and self.residual is not None


def estimate_similarity_transform(
    landmarks: Landmarks5,
    template: npt.NDArray[np.float32] = ARCFACE_REFERENCE_LANDMARKS_112,
) -> npt.NDArray[np.float32] | None:
    """Fit the similarity transform mapping detected landmarks onto a template.

    Uses the closed-form Umeyama solution rather than an iterative estimator.
    It is exact for the least-squares similarity fit, has no failure mode on
    well-conditioned input, and - unlike ``cv2.estimateAffinePartial2D`` with
    RANSAC or LMEDS - is fully deterministic, which the embedding cache relies
    on: a non-deterministic warp would produce a different crop hash, and
    therefore a different embedding, for the same face on every call.

    Args:
        landmarks: The five detected keypoints in source coordinates.
        template: ``(5, 2)`` destination points in the output frame.

    Returns:
        A ``(2, 3)`` float32 transform, or ``None`` if the landmark set is
        degenerate (all points coincident).
    """
    source = np.asarray(landmarks.points, dtype=np.float64).reshape(5, 2)
    destination = np.asarray(template, dtype=np.float64).reshape(5, 2)

    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)

    source_centred = source - source_mean
    destination_centred = destination - destination_mean

    source_variance = float((source_centred**2).sum() / source.shape[0])
    if source_variance <= EPSILON:
        return None

    covariance = (destination_centred.T @ source_centred) / source.shape[0]
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)

    # Guard against the reflection case. Without it a mirrored landmark set
    # would be "fitted" by flipping the face, which the recogniser would then
    # embed as a different person.
    correction = np.eye(2)
    if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) < 0:
        correction[1, 1] = -1.0

    rotation = u_matrix @ correction @ vt_matrix
    scale = float((singular_values * np.diag(correction)).sum() / source_variance)
    translation = destination_mean - scale * (rotation @ source_mean)

    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:, :2] = (scale * rotation).astype(np.float32)
    matrix[:, 2] = translation.astype(np.float32)
    return matrix


def alignment_residual(
    landmarks: Landmarks5,
    matrix: npt.NDArray[np.float32],
    template: npt.NDArray[np.float32] = ARCFACE_REFERENCE_LANDMARKS_112,
) -> float:
    """Mean landmark-to-template distance after warping, scale-normalised.

    Expressed in units of interocular distance *in the template frame*, so the
    figure is comparable across faces of any size. A well-detected frontal face
    lands around 0.02-0.08; a turned head is higher because a rigid transform
    genuinely cannot fit a rotated 3D object onto a frontal template; a garbage
    landmark set is higher still.

    That conflation is deliberate and documented rather than corrected for: the
    residual is used as a *trust* signal, and a heavily turned head is in fact
    less trustworthy for recognition, so both causes should raise it.

    Args:
        landmarks: The five detected keypoints in source coordinates.
        matrix: The ``(2, 3)`` transform returned by
            :func:`estimate_similarity_transform`.
        template: The destination template the transform was fitted to.

    Returns:
        Mean residual in interocular units, non-negative.
    """
    source = np.asarray(landmarks.points, dtype=np.float64).reshape(5, 2)
    destination = np.asarray(template, dtype=np.float64).reshape(5, 2)

    homogeneous = np.hstack([source, np.ones((5, 1), dtype=np.float64)])
    projected = homogeneous @ np.asarray(matrix, dtype=np.float64).T

    distances = np.linalg.norm(projected - destination, axis=1)
    template_interocular = float(np.linalg.norm(destination[1] - destination[0]))
    return float(distances.mean() / max(template_interocular, EPSILON))


def align_face_for_recognition(
    image: BgrImage,
    *,
    box: BoundingBox | None = None,
    landmarks: Landmarks5 | None = None,
    output_size: tuple[int, int] = (112, 112),
    max_residual: float = 0.28,
    allow_box_fallback: bool = True,
    box_margin: float = 0.10,
) -> AlignmentResult:
    """Produce the crop the recogniser expects.

    Prefers the landmark template. Falls back to a box-centred crop when
    landmarks are unavailable, when the transform cannot be estimated, or when
    the residual shows the landmark set does not describe a plausible face.

    Args:
        image: Full source image, BGR uint8.
        box: Face bounds, required for the fallback path.
        landmarks: Five keypoints, required for the template path.
        output_size: Template frame size. Must be ``(112, 112)`` for ArcFace.
        max_residual: Residual above which the landmark fit is rejected.
        allow_box_fallback: Whether the box-only path may be used.
        box_margin: Fractional expansion applied in the fallback path.

    Returns:
        The aligned crop with its provenance.

    Raises:
        ValueError: if neither landmarks nor a box were supplied, or if the
            fallback is needed but disabled.
    """
    if landmarks is None and box is None:
        raise ValueError("Alignment needs either landmarks or a bounding box")

    if landmarks is not None:
        result = _align_from_landmarks(
            image, landmarks, output_size=output_size, max_residual=max_residual
        )
        if result is not None:
            return result
        reason = "landmark alignment failed its residual check"
    else:
        reason = "no landmarks were available"

    if not allow_box_fallback or box is None:
        raise ValueError(
            f"Cannot align this face: {reason}, and the box fallback is "
            f"{'unavailable' if box is None else 'disabled'}."
        )

    return _align_from_box(image, box, output_size=output_size, margin=box_margin, reason=reason)


def _align_from_landmarks(
    image: BgrImage,
    landmarks: Landmarks5,
    *,
    output_size: tuple[int, int],
    max_residual: float,
) -> AlignmentResult | None:
    """Warp onto the template, or return ``None`` if the fit is not credible."""
    # The reference template is defined at 112x112 in absolute pixels; scale it
    # if a caller has configured a different frame.
    scale_x = output_size[0] / 112.0
    scale_y = output_size[1] / 112.0
    template = (
        ARCFACE_REFERENCE_LANDMARKS_112 * np.array([scale_x, scale_y], dtype=np.float32)
    ).astype(np.float32)

    matrix = estimate_similarity_transform(landmarks, template)
    if matrix is None:
        return None

    residual = alignment_residual(landmarks, matrix, template)
    if residual > max_residual:
        return None

    crop = cv2.warpAffine(
        image,
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_BORDER_VALUE,
    )
    return AlignmentResult(
        crop=np.asarray(crop, dtype=np.uint8),
        aligned=True,
        residual=residual,
        matrix=matrix,
    )


def _align_from_box(
    image: BgrImage,
    box: BoundingBox,
    *,
    output_size: tuple[int, int],
    margin: float,
    reason: str,
) -> AlignmentResult:
    """Centre the box in the template frame - the degraded path.

    Recovers scale normalisation but not rotation: an in-plane tilted head goes
    into the network tilted, and ArcFace has no rotation invariance to speak of
    because its training data was always upright after alignment.
    """
    crop, _ = crop_with_margin(image, box, margin=margin, square=True)
    resized = cv2.resize(
        crop,
        output_size,
        interpolation=(
            cv2.INTER_AREA if crop.shape[0] > output_size[1] else cv2.INTER_LINEAR
        ),
    )
    return AlignmentResult(
        crop=np.asarray(resized, dtype=np.uint8),
        aligned=False,
        residual=None,
        matrix=None,
        reason=reason,
    )


def mirror(crop: BgrImage) -> BgrImage:
    """Horizontally mirror an aligned crop, for flip augmentation."""
    return np.asarray(cv2.flip(crop, 1), dtype=np.uint8)


__all__ = [
    "AlignmentResult",
    "align_face_for_recognition",
    "alignment_residual",
    "estimate_similarity_transform",
    "mirror",
]
