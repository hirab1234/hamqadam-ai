"""Diagnostic for the occlusion signal set.

Prints the raw per-region measurements for a clean face and for occluded
variants, so the discriminative power of each signal can be compared directly
and thresholds calibrated against real pixels rather than intuition.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.detectors.occlusion import _REGION_BOXES, OcclusionAnalyzer
from hamqadam_ai.models.registry import get_registry
from hamqadam_ai.services import build_face_detection_service
from hamqadam_ai.utils.geometry import Landmarks5
from hamqadam_ai.utils.image_io import load_image
from hamqadam_ai.utils.image_ops import gradient_energy, skin_mask


def portrait() -> Path:
    """Public-domain test portrait bundled with matplotlib."""
    import matplotlib

    return (
        Path(matplotlib.__file__).parent
        / "mpl-data" / "sample_data" / "grace_hopper.jpg"
    )


def bar_over_eyes(
    image: np.ndarray,
    marks: dict[str, tuple[float, float]],
    colour: tuple[int, int, int],
    height_scale: float = 0.72,
    width_scale: float = 2.0,
) -> np.ndarray:
    """Paste a rotated bar centred on the eye line."""
    canvas = image.copy()
    left, right = marks["left_eye"], marks["right_eye"]
    sep = math.hypot(right[0] - left[0], right[1] - left[1])
    centre = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    rect = ((centre[0], centre[1]), (sep * width_scale, sep * height_scale), angle)
    cv2.fillConvexPoly(canvas, cv2.boxPoints(rect).astype(np.int32), colour)
    return canvas


def measure(analyzer: OcclusionAnalyzer, image, box, landmarks) -> dict:  # noqa: ANN001
    """Return per-region raw statistics, including candidate new signals."""
    canonical = analyzer._warp_to_canonical(image, landmarks, box)  # noqa: SLF001
    gray = cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY)
    energy = gradient_energy(gray)
    skin = skin_mask(canonical) > 0
    lab = cv2.cvtColor(canonical, cv2.COLOR_BGR2LAB)

    face_median = float(np.median(energy)) + 1e-8
    face_luma = float(np.mean(gray)) + 1e-8
    flat_cut = face_median * 0.5

    rows = {}
    for region, (x1, y1, x2, y2) in _REGION_BOXES.items():
        pe = energy[y1:y2, x1:x2]
        rows[str(region)] = {
            "mean_grad": float(np.mean(pe)) / face_median,
            "median_grad": float(np.median(pe)) / face_median,
            "flat_frac": float(np.mean(pe < flat_cut)),
            "skin": float(np.mean(skin[y1:y2, x1:x2])),
            "luma": float(np.mean(gray[y1:y2, x1:x2])) / face_luma,
            "chroma_sd": float(
                np.mean([np.std(lab[y1:y2, x1:x2, 1]), np.std(lab[y1:y2, x1:x2, 2])])
            ),
        }
    return rows


def main() -> int:
    """Print a signal comparison table."""
    settings = get_settings()
    service = build_face_detection_service(settings, get_registry(settings))
    base = load_image(portrait(), role="diag").pixels

    result = service.detect(base)
    face = result.primary_face
    assert face is not None
    marks = {lm.name: (lm.x, lm.y) for lm in face.landmarks}
    landmarks = Landmarks5(
        np.array(
            [marks[n] for n in
             ("left_eye", "right_eye", "nose_tip", "mouth_left", "mouth_right")],
            dtype=np.float32,
        )
    )
    from hamqadam_ai.utils.geometry import BoundingBox

    box = BoundingBox(
        face.bounding_box.x1, face.bounding_box.y1,
        face.bounding_box.x2, face.bounding_box.y2,
    )

    analyzer = OcclusionAnalyzer(settings.detection.occlusion)

    variants = {
        "clean": base,
        "black_bar": bar_over_eyes(base, marks, (18, 18, 20)),
        "grey_bar": bar_over_eyes(base, marks, (120, 120, 120)),
        "skin_bar": bar_over_eyes(base, marks, (150, 175, 205)),
    }

    for signal in ("mean_grad", "median_grad", "flat_frac", "skin", "luma", "chroma_sd"):
        print()
        print(f"=== {signal} ===")
        header = f"{'variant':<12s}" + "".join(f"{r:>11s}" for r in _REGION_BOXES)
        print(header)
        for name, image in variants.items():
            rows = measure(analyzer, image, box, landmarks)
            line = f"{name:<12s}"
            for region in _REGION_BOXES:
                line += f"{rows[str(region)][signal]:>11.3f}"
            print(line)

    print()
    print("=== overall skin coverage of the whole canonical face (clean) ===")
    canonical = analyzer._warp_to_canonical(base, landmarks, box)  # noqa: SLF001
    print(f"  full-frame skin fraction: {float(np.mean(skin_mask(canonical) > 0)):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
