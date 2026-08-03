"""Inspect the raw quality signals before any score mapping is applied.

Answers three questions the calibration table raised:

1. What are the true dynamic ranges of the focus measures?
2. Does the radial spectrum actually separate native from upscaled capture?
3. Is the blockiness measure responding to codec artefacts or to content?
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.models.registry import get_registry
from hamqadam_ai.quality.base import QualityContext
from hamqadam_ai.quality.blur import laplacian_variance, tenengrad
from hamqadam_ai.quality.spectral import blockiness, radial_power_spectrum
from hamqadam_ai.services import build_face_detection_service
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5

BgrImage = npt.NDArray[np.uint8]


def portrait() -> BgrImage:
    """Public-domain reference portrait."""
    import matplotlib

    return cv2.imread(
        str(
            Path(matplotlib.__file__).parent
            / "mpl-data" / "sample_data" / "grace_hopper.jpg"
        ),
        cv2.IMREAD_COLOR,
    )


def upscale(image: BgrImage, factor: float) -> BgrImage:
    """Downscale then re-enlarge."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (int(w / factor), int(h / factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def jpeg(image: BgrImage, q: int) -> BgrImage:
    """JPEG round-trip."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main() -> int:
    """Print the diagnostic tables."""
    settings = get_settings()
    detector = build_face_detection_service(settings, get_registry(settings))
    base = portrait()

    cases = {
        "pristine": base,
        "gauss_3": cv2.GaussianBlur(base, (3, 3), 0),
        "gauss_9": cv2.GaussianBlur(base, (9, 9), 0),
        "gauss_21": cv2.GaussianBlur(base, (21, 21), 0),
        "upscaled_2x": upscale(base, 2.0),
        "upscaled_4x": upscale(base, 4.0),
        "jpeg_q8": jpeg(base, 8),
    }

    contexts: dict[str, QualityContext] = {}
    for name, image in cases.items():
        detection = detector.detect(image, role=ImageRole.LIVE_SELFIE)
        box = marks = None
        if detection.primary_face is not None:
            bb = detection.primary_face.bounding_box
            box = BoundingBox(bb.x1, bb.y1, bb.x2, bb.y2)
            named = {lm.name: (lm.x, lm.y) for lm in detection.primary_face.landmarks}
            if len(named) == 5:
                marks = Landmarks5(
                    np.array(
                        [
                            named[n]
                            for n in (
                                "left_eye", "right_eye", "nose_tip",
                                "mouth_left", "mouth_right",
                            )
                        ],
                        dtype=np.float32,
                    )
                )
        contexts[name] = QualityContext(
            image=image,
            face_box=box,
            landmarks=marks,
            canonical_size=settings.quality.canonical_face_size,
            crop_margin=settings.quality.face_crop_margin,
            analysis_long_side=settings.quality.global_analysis_long_side,
        )

    names = list(cases)
    width = 13

    print("=" * 100)
    print("1. FOCUS MEASURES - true dynamic range")
    print("=" * 100)
    print(f"{'measure':<34s}" + "".join(f"{n:>{width}s}" for n in names))
    print("-" * (34 + width * len(names)))

    rows: list[tuple[str, list[float]]] = []
    rows.append(
        ("analysis lap_var", [laplacian_variance(contexts[n].analysis_gray) for n in names])
    )
    rows.append(
        ("analysis tenengrad", [tenengrad(contexts[n].analysis_gray) for n in names])
    )
    rows.append(
        (
            "canonical lap_var",
            [
                laplacian_variance(contexts[n].canonical_gray)
                if contexts[n].canonical_gray is not None
                else float("nan")
                for n in names
            ],
        )
    )
    rows.append(
        (
            "eye_band lap_var",
            [
                laplacian_variance(contexts[n].eye_band)
                if contexts[n].eye_band is not None
                else float("nan")
                for n in names
            ],
        )
    )
    rows.append(
        (
            "eye_band tenengrad",
            [
                tenengrad(contexts[n].eye_band)
                if contexts[n].eye_band is not None
                else float("nan")
                for n in names
            ],
        )
    )
    for label, values in rows:
        print(f"{label:<34s}" + "".join(f"{v:>{width}.1f}" for v in values))

    print()
    print("=" * 100)
    print("2. RADIAL SPECTRUM - does it separate native from upscaled?")
    print("=" * 100)
    spectra = {n: radial_power_spectrum(contexts[n].analysis_gray) for n in names}

    print(f"{'measure':<34s}" + "".join(f"{n:>{width}s}" for n in names))
    print("-" * (34 + width * len(names)))
    for cutoff in (0.10, 0.25, 0.40, 0.55, 0.70, 0.85):
        values = [spectra[n].energy_above(cutoff) for n in names]
        print(f"{'energy_above(' + f'{cutoff:.2f}' + ')':<34s}"
              + "".join(f"{v:>{width}.6f}" for v in values))
    print(f"{'cutoff_ratio()':<34s}"
          + "".join(f"{spectra[n].cutoff_ratio():>{width}.4f}" for n in names))

    print()
    print("  normalised log10 profile at selected radii (relative to r=0.05):")
    reference_index = max(1, int(0.05 * (spectra[names[0]].profile.size - 1)))
    print(f"{'  radius':<34s}" + "".join(f"{n:>{width}s}" for n in names))
    for fraction in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99):
        line = f"{'  r=' + f'{fraction:.2f}':<34s}"
        for n in names:
            profile = spectra[n].profile
            index = min(profile.size - 1, int(fraction * (profile.size - 1)))
            ref = max(profile[reference_index], 1e-12)
            line += f"{np.log10(max(profile[index], 1e-12) / ref):>{width}.2f}"
        print(line)

    print()
    print("=" * 100)
    print("3. BLOCKINESS - codec artefact or content?")
    print("=" * 100)
    print(f"{'measure':<34s}" + "".join(f"{n:>{width}s}" for n in names))
    print("-" * (34 + width * len(names)))
    print(f"{'analysis blockiness':<34s}"
          + "".join(f"{blockiness(contexts[n].analysis_gray):>{width}.4f}" for n in names))
    print(f"{'canonical blockiness':<34s}"
          + "".join(
              f"{blockiness(contexts[n].canonical_gray):>{width}.4f}"
              if contexts[n].canonical_gray is not None else f"{'-':>{width}s}"
              for n in names
          ))
    synthetic = np.tile(
        np.repeat(np.array([40, 200], dtype=np.uint8), 8), (256, 16)
    )[:256, :256]
    print(f"{'synthetic 8px blocks (control)':<34s}{blockiness(synthetic):>{width}.4f}")
    smooth = cv2.GaussianBlur(
        np.random.default_rng(0).integers(0, 256, (256, 256), dtype=np.uint8), (9, 9), 0
    )
    print(f"{'smooth noise (control)':<34s}{blockiness(smooth):>{width}.4f}")

    print()
    print("=" * 100)
    print("4. FACE GEOMETRY")
    print("=" * 100)
    for n in names:
        ctx = contexts[n]
        band = ctx.eye_band
        print(
            f"  {n:<13s} face_short={ctx.face_native_short_side:6.1f}  "
            f"canonical_upsampled={ctx.canonical_upsampled}  "
            f"eye_band={band.shape if band is not None else None}  "
            f"analysis={ctx.analysis_gray.shape}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
