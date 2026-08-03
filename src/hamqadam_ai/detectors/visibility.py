"""Composite face-visibility scoring.

``face_visibility_score`` is one of the three values the specification requires
from Module 1, and it is the number a human reviewer looks at first. A single
opaque figure is not enough, so the scorer returns the weighted composite *and*
every component that fed it, letting the API answer "why is this 41?" with
"because the face is 22% of the frame it should be and one eye is obscured".

Five components, all in ``[0, 1]`` where 1 is best:

``detector_confidence``
    The detector's own objectness. Included because a marginal detection is
    itself evidence that something about the face is unclear.

``occlusion``
    ``1 - overall_occlusion_score``.

``pose``
    ``1 - pose_deviation_score``.

``face_size``
    How close the face's share of the frame is to the ideal band. Scored with a
    plateau rather than a ramp: anywhere inside the band is equally good, and
    both too-small and too-large fall away from it.

``framing``
    ``1 - truncation_ratio``, penalising a face cut off by the frame edge.

Weights come from ``configs/thresholds.yaml`` and are normalised to sum to 1.0
at load time, so an operator can raise one without rebalancing the rest.

Missing components
------------------
The landmark-free detectors produce no pose or occlusion analysis. Rather than
substituting a neutral 1.0 - which would flatter a face nobody actually
examined - the affected weights are **redistributed** across the components
that were measured, and the result carries a warning. A face scored on three
components is scored honestly on three components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hamqadam_ai.core.config import VisibilityConfig
from hamqadam_ai.core.constants import EPSILON

#: Ideal share of the frame for a face, as a fraction of total image area.
#: Below the lower bound the face has too few pixels for a reliable embedding;
#: above the upper bound the crop usually clips the hairline and chin, and is
#: frequently a photograph of a photograph.
_IDEAL_AREA_BAND = (0.06, 0.45)

#: Face area ratio at which the size score reaches zero on the small side.
_AREA_FLOOR = 0.004

#: Face area ratio at which the size score reaches zero on the large side.
_AREA_CEILING = 0.95


@dataclass(frozen=True, slots=True)
class VisibilityResult:
    """A composite visibility score and its provenance.

    Attributes:
        score: Composite visibility in ``[0, 1]``.
        components: Each measured component, in ``[0, 1]``.
        weights: The weight actually applied to each component, after any
            redistribution for missing components.
        limiting_factor: The component contributing the largest weighted
            deficit - the single thing most worth telling the user to fix.
        missing: Components that could not be measured.
        acceptable: Whether the score clears the configured minimum.
    """

    score: float
    components: dict[str, float]
    weights: dict[str, float]
    limiting_factor: str
    missing: tuple[str, ...]
    acceptable: bool

    def describe(self) -> dict[str, Any]:
        """Compact summary for logging."""
        return {
            "score": round(self.score, 3),
            "limiting": self.limiting_factor,
            "missing": list(self.missing),
            "acceptable": self.acceptable,
        }


class VisibilityScorer:
    """Computes the composite face-visibility score.

    Args:
        config: The visibility section of the detection configuration.
    """

    __slots__ = ("_config",)

    def __init__(self, config: VisibilityConfig) -> None:
        self._config = config

    def score(
        self,
        *,
        detector_confidence: float,
        occlusion_score: float | None,
        pose_deviation: float | None,
        face_area_ratio: float,
        truncation_ratio: float,
    ) -> VisibilityResult:
        """Compute visibility from the measured signals.

        Args:
            detector_confidence: Detector objectness in ``[0, 1]``.
            occlusion_score: Overall occlusion in ``[0, 1]``, or ``None`` when
                occlusion analysis could not run.
            pose_deviation: Pose deviation in ``[0, 1]``, or ``None`` when pose
                estimation could not run.
            face_area_ratio: Face area divided by image area.
            truncation_ratio: Fraction of the face box outside the frame.

        Returns:
            The composite score with its full breakdown.
        """
        components: dict[str, float] = {
            "detector_confidence": _clamp(detector_confidence),
            "face_size": self._size_score(face_area_ratio),
            "framing": _clamp(1.0 - truncation_ratio),
        }
        missing: list[str] = []

        if occlusion_score is None:
            missing.append("occlusion")
        else:
            components["occlusion"] = _clamp(1.0 - occlusion_score)

        if pose_deviation is None:
            missing.append("pose")
        else:
            components["pose"] = _clamp(1.0 - pose_deviation)

        weights = self._redistribute(set(missing))

        total = sum(weights[name] * components[name] for name in components)
        limiting = self._limiting_factor(components, weights)

        return VisibilityResult(
            score=_clamp(total),
            components=components,
            weights=weights,
            limiting_factor=limiting,
            missing=tuple(missing),
            acceptable=total >= self._config.min_acceptable,
        )

    def _redistribute(self, missing: set[str]) -> dict[str, float]:
        """Spread the weight of unmeasured components over the measured ones.

        Redistribution is proportional, so the relative importance of the
        surviving components is preserved.
        """
        weights = dict(self._config.weights)
        if not missing:
            return weights

        forfeited = sum(weights.pop(name, 0.0) for name in missing)
        remaining = sum(weights.values())
        if remaining <= EPSILON:
            # Pathological config where every measured component has zero
            # weight. Fall back to a uniform split so the score stays defined.
            share = 1.0 / max(len(weights), 1)
            return dict.fromkeys(weights, share)

        factor = 1.0 + forfeited / remaining
        return {name: weight * factor for name, weight in weights.items()}

    @staticmethod
    def _size_score(area_ratio: float) -> float:
        """Score how well the face fills the frame.

        A plateau over the ideal band with linear ramps outside it. Ramping
        from a single ideal point would penalise a perfectly good portrait for
        being 8% of the frame instead of 20%, which is not a real defect.
        """
        low, high = _IDEAL_AREA_BAND
        if low <= area_ratio <= high:
            return 1.0
        if area_ratio < low:
            if area_ratio <= _AREA_FLOOR:
                return 0.0
            return _clamp((area_ratio - _AREA_FLOOR) / max(low - _AREA_FLOOR, EPSILON))
        if area_ratio >= _AREA_CEILING:
            return 0.0
        return _clamp((_AREA_CEILING - area_ratio) / max(_AREA_CEILING - high, EPSILON))

    @staticmethod
    def _limiting_factor(
        components: dict[str, float], weights: dict[str, float]
    ) -> str:
        """Return the component responsible for the largest weighted deficit."""
        if not components:
            return "unknown"
        deficits = {
            name: weights.get(name, 0.0) * (1.0 - value)
            for name, value in components.items()
        }
        return max(deficits, key=lambda name: deficits[name])


def _clamp(value: float) -> float:
    """Clamp a float into ``[0, 1]``."""
    return float(min(1.0, max(0.0, value)))


__all__ = ["VisibilityResult", "VisibilityScorer"]
