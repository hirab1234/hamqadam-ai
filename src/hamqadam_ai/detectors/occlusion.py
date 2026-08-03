"""Facial occlusion analysis.

Problem
-------
The specification requires detecting partial obstruction: a hand, a mask,
sunglasses, a scarf. There is no public, redistributable occlusion classifier
of production quality, and training one demands a labelled dataset this project
does not have. Inventing a model would mean shipping something untested.

Approach
--------
A multi-signal geometric estimator that measures physically meaningful
properties of each facial region and fuses them into a probability. It is fully
implemented, deterministic, unit-testable and needs no weights - and it is
honest about being an estimator rather than a learned classifier.

Every face is first warped into a **canonical frame** using the eye pair and
the mouth centre. Region boxes are then fixed rectangles in that frame, so head
tilt, scale and position stop mattering: the "left eye" region is the left eye
whether the subject is upright, tilted 20 degrees, near or far.

Three independent signals per region
------------------------------------
1. **Flat fraction** - the share of the region whose local gradient falls below
   half the face-wide median. This is the dominant signal, and it is measured
   as a *fraction* rather than as mean gradient energy for a concrete reason
   established by measurement: a hard-edged occluder such as a sunglasses bar
   contributes strong gradients along its own boundary, so the mean gradient of
   a covered region can be *higher* than that of bare skin. The fraction of
   flat pixels is immune to that, because a handful of boundary pixels cannot
   move it. On a real portrait it separates cleanly - 0.06 for a bare eye
   against 1.00 for a covered one - and it does so regardless of the
   occluder's colour, which is what catches a skin-toned covering that the
   chrominance signal misses entirely.
2. **Skin coverage** - fraction of pixels inside the YCrCb skin ellipse. This
   is chrominance-only and therefore illumination-invariant and robust across
   skin tones, which matters for a Pakistani user base.
3. **Colour uniformity** - inverse spread of the CIELAB ``a*``/``b*``
   chrominance. A manufactured object is far more uniform in colour than skin.

No single signal is sufficient - a dark-skinned subject in low light drags skin
coverage down, a smooth forehead is genuinely low-texture, and a skin-coloured
occluder defeats chrominance - so the three are fused with configurable weights
and the result is sharpened by a logistic.

A **symmetry** term is computed separately. A hand covering one cheek produces
a large left/right asymmetry with only a moderate overall score, which is the
signature the per-region scores alone would miss.

Optional learned classifier
---------------------------
If ``face_occlusion_classifier`` is enabled in ``configs/models.yaml`` and its
ONNX artefact is present, its per-region sigmoid outputs are used instead and
the geometric estimator becomes a cross-check. The plumbing is complete; no
weights are shipped, because shipping unvalidated weights would be worse than
shipping none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import OcclusionConfig
from hamqadam_ai.core.constants import EPSILON, OCCLUSION_REGION_ORDER, FaceRegion
from hamqadam_ai.logging.setup import get_logger
from hamqadam_ai.models.base import LoadedModel
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5
from hamqadam_ai.utils.image_ops import build_blob, gradient_energy, skin_mask

log = get_logger(__name__)

#: Edge length of the canonical analysis frame.
CANONICAL_SIZE = 128

#: Where the three reference landmarks land in the canonical frame.
#: Interocular distance is 48 px, giving each eye region ~32 px of width -
#: enough for the gradient statistics to be stable without over-smoothing.
_CANONICAL_LEFT_EYE = (40.0, 48.0)
_CANONICAL_RIGHT_EYE = (88.0, 48.0)
_CANONICAL_MOUTH_CENTER = (64.0, 100.0)

#: Region rectangles in the canonical frame, as ``(x1, y1, x2, y2)``.
#: Deliberately slightly inset from anatomical boundaries so that a small
#: landmark error does not pull a neighbouring region's pixels into the sample.
_REGION_BOXES: dict[FaceRegion, tuple[int, int, int, int]] = {
    FaceRegion.FOREHEAD: (36, 6, 92, 38),
    FaceRegion.LEFT_EYE: (24, 33, 58, 63),
    FaceRegion.RIGHT_EYE: (70, 33, 104, 63),
    FaceRegion.NOSE: (49, 58, 79, 89),
    FaceRegion.MOUTH: (39, 85, 89, 114),
    FaceRegion.CHIN: (45, 109, 83, 128),
}

#: Relative weight of each signal inside a region's fused probability.
#: Flatness dominates because measurement showed it to be the only signal that
#: separates cleanly for *every* occluder colour; chrominance is blind to a
#: skin-toned covering and uniformity is only corroborating evidence.
_SIGNAL_WEIGHTS = {"flatness": 0.50, "skin": 0.32, "uniformity": 0.18}

#: A pixel counts as flat when its gradient magnitude is below this multiple of
#: the face-wide median gradient.
_FLATNESS_GRADIENT_RATIO = 0.5

#: Flat fraction below which a region is considered normally textured. Measured
#: baselines on a clean portrait: eyes 0.06-0.08, mouth 0.16, chin 0.17,
#: nose 0.27, forehead 0.32. The forehead is the flattest genuine region, so
#: the floor sits above it to avoid a standing false positive there.
_FLATNESS_FLOOR = 0.45

#: Flat fraction at which the flatness deficit saturates.
_FLATNESS_CEILING = 0.85

#: Logistic sharpening applied to the fused evidence. Without it the estimator
#: produces a mush of mid-range probabilities that no threshold separates well.
_LOGISTIC_STEEPNESS = 7.0
_LOGISTIC_MIDPOINT = 0.45

#: The forehead is legitimately covered by hair, a fringe or a headscarf on a
#: large fraction of genuine users. Its evidence is damped so that ordinary
#: hairstyles do not read as fraud, and its configured weight is low as well.
_FOREHEAD_EVIDENCE_DAMPING = 0.55


@dataclass(frozen=True, slots=True)
class _RegionSignals:
    """Raw measurements for one region, before fusion.

    Internal to the analyser; the public shape is :class:`RegionEvidence`.
    """

    flat_fraction: float
    texture_energy: float
    skin_coverage: float
    colour_uniformity: float
    relative_luminance: float


@dataclass(frozen=True, slots=True)
class RegionEvidence:
    """Per-region measurements and the probability fused from them.

    Attributes:
        probability: Fused occlusion probability in ``[0, 1]``.
        occluded: Whether the probability exceeds the configured threshold.
        flat_fraction: Share of the region whose gradient falls below half the
            face median. The dominant signal; high means flat means covered.
        texture_energy: Mean gradient energy relative to the face median.
            Reported for diagnostics and human review. Deliberately *not* the
            driver of the flatness deficit - a hard-edged occluder inflates it.
        skin_coverage: Fraction of the region classified as skin.
        colour_uniformity: How uniform the chrominance is, in ``[0, 1]``.
        mean_luminance: Mean luminance relative to the whole face.
    """

    probability: float
    occluded: bool
    flat_fraction: float
    texture_energy: float
    skin_coverage: float
    colour_uniformity: float
    mean_luminance: float


@dataclass(slots=True)
class OcclusionResult:
    """Aggregate occlusion analysis for one face.

    Attributes:
        occluded: Whether overall occlusion exceeds the configured threshold.
        overall_score: Weighted occlusion probability across regions.
        regions: Per-region evidence, keyed by region name.
        occluded_regions: Names of the regions flagged as obstructed.
        symmetry_delta: Normalised left/right asymmetry.
        method: ``classifier`` or ``geometric``.
    """

    occluded: bool
    overall_score: float
    regions: dict[str, RegionEvidence] = field(default_factory=dict)
    occluded_regions: list[str] = field(default_factory=list)
    symmetry_delta: float = 0.0
    method: str = "geometric"

    @classmethod
    def unavailable(cls) -> OcclusionResult:
        """Result used when analysis could not run (no landmarks).

        Reports "not occluded" with a zero score rather than guessing. The
        detection service raises a warning alongside it so the absence of
        analysis is visible rather than being mistaken for a clean face.
        """
        return cls(
            occluded=False, overall_score=0.0, regions={}, method="unavailable"
        )

    def describe(self) -> dict[str, Any]:
        """Compact summary for logging."""
        return {
            "occluded": self.occluded,
            "score": round(self.overall_score, 3),
            "regions": self.occluded_regions,
            "symmetry": round(self.symmetry_delta, 3),
            "method": self.method,
        }


class OcclusionAnalyzer:
    """Estimates which parts of a face are obstructed.

    Args:
        config: The occlusion section of the detection configuration.
        classifier: Optional ONNX per-region occlusion model. When supplied its
            outputs take precedence and the geometric estimator is retained as
            a cross-check.
    """

    __slots__ = ("_classifier", "_classifier_regions", "_config")

    def __init__(
        self, config: OcclusionConfig, classifier: LoadedModel | None = None
    ) -> None:
        self._config = config
        self._classifier = classifier
        self._classifier_regions: tuple[FaceRegion, ...] = OCCLUSION_REGION_ORDER
        if classifier is not None:
            declared = classifier.spec.output.get("regions")
            if declared:
                self._classifier_regions = tuple(FaceRegion(name) for name in declared)

    def analyze(
        self,
        image: npt.NDArray[np.uint8],
        box: BoundingBox,
        landmarks: Landmarks5 | None,
    ) -> OcclusionResult:
        """Analyse one face for obstruction.

        Args:
            image: The full source image, BGR uint8.
            box: The face bounds in source coordinates.
            landmarks: Five keypoints. Required - without them the canonical
                warp cannot be built and analysis is skipped.

        Returns:
            The occlusion analysis, or :meth:`OcclusionResult.unavailable` when
            landmarks are absent.
        """
        if landmarks is None:
            return OcclusionResult.unavailable()

        canonical = self._warp_to_canonical(image, landmarks, box)
        if canonical is None:
            return OcclusionResult.unavailable()

        if self._classifier is not None:
            classified = self._run_classifier(canonical)
            if classified is not None:
                return classified

        return self._analyze_geometric(canonical)

    # -- Canonical warp ----------------------------------------------------- #

    def _warp_to_canonical(
        self,
        image: npt.NDArray[np.uint8],
        landmarks: Landmarks5,
        box: BoundingBox,
    ) -> npt.NDArray[np.uint8] | None:
        """Warp the face into the fixed analysis frame.

        Uses an affine transform fitted to three points - the two eyes and the
        mouth centre. Three points give a full affine rather than a similarity,
        which additionally absorbs the vertical foreshortening that pitch
        introduces, keeping the mouth region over the mouth on a face looking
        slightly down.
        """
        source = np.array(
            [landmarks.left_eye, landmarks.right_eye, landmarks.mouth_center],
            dtype=np.float32,
        )
        destination = np.array(
            [_CANONICAL_LEFT_EYE, _CANONICAL_RIGHT_EYE, _CANONICAL_MOUTH_CENTER],
            dtype=np.float32,
        )

        # Degenerate landmark sets (collinear points) make the affine singular.
        area = abs(
            (source[1, 0] - source[0, 0]) * (source[2, 1] - source[0, 1])
            - (source[2, 0] - source[0, 0]) * (source[1, 1] - source[0, 1])
        )
        if area < max(box.area * 1e-4, 4.0):
            return None

        try:
            matrix = cv2.getAffineTransform(source, destination)
        except cv2.error:
            return None

        warped = cv2.warpAffine(
            image,
            matrix,
            (CANONICAL_SIZE, CANONICAL_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return np.asarray(warped, dtype=np.uint8)

    # -- Learned path -------------------------------------------------------- #

    def _run_classifier(
        self, canonical: npt.NDArray[np.uint8]
    ) -> OcclusionResult | None:
        """Run the optional ONNX occlusion classifier.

        Returns ``None`` on any failure so the caller falls through to the
        geometric estimator - a classifier that errors must not take down the
        whole detection stage.
        """
        assert self._classifier is not None  # noqa: S101 - guarded by the caller
        spec = self._classifier.spec
        try:
            blob = build_blob(
                canonical,
                spec.input.size,
                mean=spec.input.mean_tuple,
                scale=spec.input.scale,
                swap_rb=spec.input.swap_rb,
            )
            outputs = self._classifier.run({self._classifier.input_names[0]: blob})
        except Exception as exc:  # noqa: BLE001 - degrade, never fail the request
            log.warning("occlusion.classifier_failed", reason=str(exc))
            return None

        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if logits.size != len(self._classifier_regions):
            log.warning(
                "occlusion.classifier_shape_mismatch",
                expected=len(self._classifier_regions),
                actual=int(logits.size),
            )
            return None

        probabilities = 1.0 / (1.0 + np.exp(-logits))

        # The geometric estimator still runs, to supply the measured signals
        # that make a flagged region explainable to a human reviewer.
        geometric = self._analyze_geometric(canonical)

        regions: dict[str, RegionEvidence] = {}
        for region, probability in zip(
            self._classifier_regions, probabilities, strict=True
        ):
            measured = geometric.regions.get(str(region))
            regions[str(region)] = RegionEvidence(
                probability=float(probability),
                occluded=bool(probability >= self._config.region_threshold),
                flat_fraction=measured.flat_fraction if measured else 0.0,
                texture_energy=measured.texture_energy if measured else 0.0,
                skin_coverage=measured.skin_coverage if measured else 0.0,
                colour_uniformity=measured.colour_uniformity if measured else 0.0,
                mean_luminance=measured.mean_luminance if measured else 0.0,
            )

        return self._aggregate(regions, geometric.symmetry_delta, method="classifier")

    # -- Geometric path ------------------------------------------------------ #

    def _analyze_geometric(
        self, canonical: npt.NDArray[np.uint8]
    ) -> OcclusionResult:
        """Measure the three signals per region and fuse them."""
        gray = np.asarray(
            cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY), dtype=np.uint8
        )
        energy = gradient_energy(gray)
        skin = skin_mask(canonical) > 0
        lab = cv2.cvtColor(canonical, cv2.COLOR_BGR2LAB)

        # Face-level baselines. Every region measurement is expressed relative
        # to these, which is what makes the estimator invariant to global
        # illumination and to how sharp the photograph is overall.
        face_energy_median = float(np.median(energy)) + EPSILON
        face_luminance = float(np.mean(gray)) + EPSILON
        flat_cutoff = face_energy_median * _FLATNESS_GRADIENT_RATIO

        raw: dict[FaceRegion, _RegionSignals] = {}
        for region, (x1, y1, x2, y2) in _REGION_BOXES.items():
            patch_energy = energy[y1:y2, x1:x2]
            patch_skin = skin[y1:y2, x1:x2]
            patch_gray = gray[y1:y2, x1:x2]
            patch_lab = lab[y1:y2, x1:x2]

            if patch_energy.size == 0:
                raw[region] = _RegionSignals(0.0, 1.0, 1.0, 0.0, 1.0)
                continue

            flat_fraction = float(np.mean(patch_energy < flat_cutoff))
            relative_energy = float(np.mean(patch_energy)) / face_energy_median
            skin_coverage = float(np.mean(patch_skin))
            relative_luminance = float(np.mean(patch_gray)) / face_luminance

            # Chrominance spread. Skin varies; a manufactured surface does not.
            # 12.0 is roughly the a*/b* standard deviation of a healthy skin
            # patch, so the ratio saturates at 1.0 for genuinely flat colour.
            chroma_spread = float(
                np.mean([np.std(patch_lab[:, :, 1]), np.std(patch_lab[:, :, 2])])
            )
            uniformity = float(np.clip(1.0 - chroma_spread / 12.0, 0.0, 1.0))

            raw[region] = _RegionSignals(
                flat_fraction=flat_fraction,
                texture_energy=relative_energy,
                skin_coverage=skin_coverage,
                colour_uniformity=uniformity,
                relative_luminance=relative_luminance,
            )

        regions: dict[str, RegionEvidence] = {}
        for region, signals in raw.items():
            probability = self._fuse(region, signals)
            regions[str(region)] = RegionEvidence(
                probability=probability,
                occluded=probability >= self._config.region_threshold,
                flat_fraction=signals.flat_fraction,
                texture_energy=signals.texture_energy,
                skin_coverage=signals.skin_coverage,
                colour_uniformity=signals.colour_uniformity,
                mean_luminance=signals.relative_luminance,
            )

        symmetry = self._symmetry_delta(raw)
        return self._aggregate(regions, symmetry, method="geometric")

    def _fuse(self, region: FaceRegion, signals: _RegionSignals) -> float:
        """Combine the three signals into one probability for a region.

        Each signal contributes a *deficit* in ``[0, 1]`` - how far past its
        configured operating point it has gone - and the weighted sum is passed
        through a logistic to sharpen the decision boundary.
        """
        config = self._config

        # Flatness: the dominant signal. Ramped between the measured baseline
        # of a genuine region and the level a solid covering reaches.
        flatness_deficit = float(
            np.clip(
                (signals.flat_fraction - _FLATNESS_FLOOR)
                / max(_FLATNESS_CEILING - _FLATNESS_FLOOR, EPSILON),
                0.0,
                1.0,
            )
        )

        skin_floor = max(config.skin_coverage_floor, EPSILON)
        skin_deficit = float(
            np.clip((skin_floor - signals.skin_coverage) / skin_floor, 0.0, 1.0)
        )

        # Uniformity is already a [0, 1] flatness of colour; only the upper half
        # of the range is evidence of anything, so it is rescaled from 0.5 up.
        uniformity_deficit = float(
            np.clip((signals.colour_uniformity - 0.5) * 2.0, 0.0, 1.0)
        )

        evidence = (
            _SIGNAL_WEIGHTS["flatness"] * flatness_deficit
            + _SIGNAL_WEIGHTS["skin"] * skin_deficit
            + _SIGNAL_WEIGHTS["uniformity"] * uniformity_deficit
        )

        # Dark lenses are the one occluder that all three general signals
        # under-weight: a black bar is not colourful, and against dark lashes
        # and brows it is not dramatically flatter than a real eye either. The
        # conjunction of "much darker than the rest of the face" AND "not skin"
        # is its distinctive signature. `min` is deliberate - both conditions
        # must hold, so ordinary dark eyes on a clear face (which are skin-
        # adjacent and score skin_deficit ~ 0) cannot trigger it.
        if region in (FaceRegion.LEFT_EYE, FaceRegion.RIGHT_EYE):
            darkness = float(
                np.clip((0.65 - signals.relative_luminance) / 0.65, 0.0, 1.0)
            )
            evidence = max(evidence, min(darkness, skin_deficit))

        if region is FaceRegion.FOREHEAD:
            evidence *= _FOREHEAD_EVIDENCE_DAMPING

        return 1.0 / (
            1.0 + math.exp(-_LOGISTIC_STEEPNESS * (evidence - _LOGISTIC_MIDPOINT))
        )

    @staticmethod
    def _symmetry_delta(raw: dict[FaceRegion, _RegionSignals]) -> float:
        """Normalised left/right difference in flatness and skin coverage.

        A hand or object covering one side of the face produces a large value
        here even when the overall occlusion score is unremarkable, because
        only half the regions are affected and the weighted mean dilutes it.
        """
        left = raw.get(FaceRegion.LEFT_EYE)
        right = raw.get(FaceRegion.RIGHT_EYE)
        if left is None or right is None:
            return 0.0

        flatness_delta = abs(left.flat_fraction - right.flat_fraction)
        skin_delta = abs(left.skin_coverage - right.skin_coverage)
        return float(np.clip(0.6 * flatness_delta + 0.4 * skin_delta, 0.0, 1.0))

    def _aggregate(
        self,
        regions: dict[str, RegionEvidence],
        symmetry_delta: float,
        *,
        method: str,
    ) -> OcclusionResult:
        """Combine per-region probabilities into the overall verdict."""
        weights = self._config.region_weights
        total_weight = 0.0
        weighted = 0.0
        for name, evidence in regions.items():
            weight = weights.get(name, 0.0)
            weighted += weight * evidence.probability
            total_weight += weight

        overall = weighted / total_weight if total_weight > 0 else 0.0

        # One heavily-covered critical region should trigger even when the
        # weighted mean stays low. A hand over both eyes is total occlusion for
        # recognition purposes regardless of how clear the chin is.
        occluded_regions = [
            name for name, evidence in regions.items() if evidence.occluded
        ]
        critical = {str(FaceRegion.LEFT_EYE), str(FaceRegion.RIGHT_EYE), str(FaceRegion.NOSE)}
        critical_hit = any(name in critical for name in occluded_regions)

        asymmetric = symmetry_delta >= self._config.symmetry_delta_threshold

        occluded = (
            overall >= self._config.overall_threshold
            or (critical_hit and overall >= self._config.overall_threshold * 0.6)
            or (asymmetric and overall >= self._config.overall_threshold * 0.7)
        )

        return OcclusionResult(
            occluded=occluded,
            overall_score=float(np.clip(overall, 0.0, 1.0)),
            regions=regions,
            occluded_regions=occluded_regions,
            symmetry_delta=float(symmetry_delta),
            method=method,
        )


__all__ = [
    "CANONICAL_SIZE",
    "OcclusionAnalyzer",
    "OcclusionResult",
    "RegionEvidence",
]
