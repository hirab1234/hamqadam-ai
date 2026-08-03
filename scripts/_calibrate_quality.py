"""Measure every quality metric against a real photograph and its degradations.

Prints the raw measurement table the anchor points in
``configs/thresholds.yaml`` are derived from. Run this after changing any
metric implementation - anchors calibrated against intuition rather than
pixels are how a quality gate ends up rejecting good photographs.

    python scripts/_calibrate_quality.py
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
from hamqadam_ai.services import build_face_detection_service, build_quality_service

BgrImage = npt.NDArray[np.uint8]


def portrait() -> BgrImage:
    """Public-domain reference portrait bundled with matplotlib."""
    import matplotlib

    path = (
        Path(matplotlib.__file__).parent
        / "mpl-data" / "sample_data" / "grace_hopper.jpg"
    )
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def jpeg(image: BgrImage, quality: int) -> BgrImage:
    """Round-trip through JPEG at a given quality."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def upscale(image: BgrImage, factor: float) -> BgrImage:
    """Downscale then re-enlarge, simulating an upscaled thumbnail."""
    height, width = image.shape[:2]
    small = cv2.resize(
        image,
        (int(width / factor), int(height / factor)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


def add_noise(image: BgrImage, sigma: float) -> BgrImage:
    """Add zero-mean Gaussian noise of a known sigma."""
    rng = np.random.default_rng(7)
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def motion_blur(image: BgrImage, length: int) -> BgrImage:
    """Horizontal motion blur of a given kernel length."""
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(image, -1, kernel)


def posterise(image: BgrImage, levels: int) -> BgrImage:
    """Quantise to a small number of luminance levels."""
    step = 256 // levels
    return ((image // step) * step).astype(np.uint8)


def stretch(image: BgrImage, factor: float) -> BgrImage:
    """Stretch horizontally and KEEP the new aspect ratio.

    An earlier version resized out and back again, which is a round trip that
    restores the original proportions - it was measuring resampling loss, not
    geometric distortion, and unsurprisingly the anisotropy metric did not fire.
    """
    height, width = image.shape[:2]
    return cv2.resize(
        image, (int(width * factor), height), interpolation=cv2.INTER_CUBIC
    )


def variants(base: BgrImage) -> dict[str, BgrImage]:
    """The degradation ladder the anchors are calibrated against."""
    return {
        "pristine": base,
        "blur_gauss_3": cv2.GaussianBlur(base, (3, 3), 0),
        "blur_gauss_9": cv2.GaussianBlur(base, (9, 9), 0),
        "blur_gauss_21": cv2.GaussianBlur(base, (21, 21), 0),
        "motion_15": motion_blur(base, 15),
        "dark_x0.35": np.clip(base * 0.35, 0, 255).astype(np.uint8),
        "bright_x1.9": np.clip(base * 1.9, 0, 255).astype(np.uint8),
        "flat_contrast": np.clip(base * 0.30 + 110, 0, 255).astype(np.uint8),
        "noise_sigma_8": add_noise(base, 8.0),
        "noise_sigma_20": add_noise(base, 20.0),
        "jpeg_q90": jpeg(base, 90),
        "jpeg_q30": jpeg(base, 30),
        "jpeg_q8": jpeg(base, 8),
        "upscaled_2x": upscale(base, 2.0),
        "upscaled_4x": upscale(base, 4.0),
        "posterised_6": posterise(base, 6),
        "stretched_1.3x": stretch(base, 1.3),
        "stretched_1.6x": stretch(base, 1.6),
    }


ROWS = [
    ("blur", "laplacian_variance"),
    ("blur", "tenengrad"),
    ("blur", "high_frequency_ratio"),
    ("blur", "motion_anisotropy"),
    ("sharpness", "eye_region_laplacian"),
    ("sharpness", "face_vs_scene_ratio"),
    ("brightness", "mean_luma"),
    ("brightness", "shadow_clipping"),
    ("brightness", "highlight_clipping"),
    ("contrast", "rms_contrast"),
    ("contrast", "dynamic_range_p1_p99"),
    ("contrast", "entropy_bits"),
    ("noise", "sigma"),
    ("noise", "sigma_immerkaer"),
    ("noise", "sigma_flat_block"),
    ("noise", "noise_to_signal"),
    ("resolution", "face_short_side"),
    ("resolution", "interocular_pixels"),
    ("resolution", "detail_energy"),
    ("pixelation", "blockiness"),
    ("pixelation", "face_blockiness"),
    ("distortion", "banding_ratio"),
    ("distortion", "chromatic_aberration"),
    ("distortion", "geometric_anisotropy"),
]

SCORE_ROWS = [
    "blur",
    "sharpness",
    "brightness",
    "contrast",
    "noise",
    "resolution",
    "pixelation",
    "distortion",
]


def main() -> int:
    """Print the raw-measurement and score tables."""
    settings = get_settings()
    detector = build_face_detection_service(settings, get_registry(settings))
    quality = build_quality_service(settings)

    base = portrait()
    cases = variants(base)

    detections = {}
    results = {}
    for name, image in cases.items():
        detection = detector.detect(image, role=ImageRole.LIVE_SELFIE)
        detections[name] = detection
        results[name] = quality.assess(
            image, role=ImageRole.LIVE_SELFIE, detection=detection
        )

    names = list(cases)
    width = max(len(n) for n in names) + 1

    print("=" * 100)
    print("RAW MEASUREMENTS")
    print("=" * 100)
    print(f"(showing first 4 of {len(names)} variants per block)")

    for start in range(0, len(names), 4):
        block = names[start : start + 4]
        print()
        print(f"{'metric':<38s}" + "".join(f"{n[:12]:>13s}" for n in block))
        print("-" * (38 + 13 * len(block)))
        for family, key in ROWS:
            line = f"{family + '.' + key:<38s}"
            for name in block:
                value = results[name].metrics.get(family)
                raw = value.measurements.get(key) if value else None
                line += f"{raw:>13.4f}" if raw is not None else f"{'-':>13s}"
            print(line)

    print()
    print("=" * 100)
    print("SCORES (0-100) AND VERDICT")
    print("=" * 100)
    print(
        f"{'variant':<{width}s}{'overall':>9s}{'arith':>8s}{'usable':>8s}"
        + "".join(f"{f[:7]:>9s}" for f in SCORE_ROWS)
        + f"  {'limiting':<12s} face"
    )
    print("-" * (width + 33 + 9 * len(SCORE_ROWS) + 20))
    for name in names:
        result = results[name]
        line = (
            f"{name:<{width}s}"
            f"{result.image_quality_score:>9.1f}"
            f"{result.arithmetic_score:>8.1f}"
            f"{'yes' if result.usable else 'NO':>8s}"
        )
        for family in SCORE_ROWS:
            detail = result.metrics.get(family)
            line += f"{detail.score:>9.1f}" if detail and detail.measured else f"{'-':>9s}"
        line += f"  {str(result.limiting_factor):<12s}"
        line += " yes" if detections[name].face_detected else " no"
        print(line)

    print()
    print("=" * 100)
    print("TIMING")
    print("=" * 100)
    durations = [r.duration_ms for r in results.values()]
    print(f"  quality assessment: min {min(durations):.1f} ms  "
          f"median {sorted(durations)[len(durations) // 2]:.1f} ms  "
          f"max {max(durations):.1f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
