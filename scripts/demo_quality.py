"""MODULE 2 demonstration - run quality assessment on real images.

Usage
-----
::

    # Built-in reference portrait plus a ladder of controlled degradations
    python scripts/demo_quality.py

    # Your own images
    python scripts/demo_quality.py path/to/selfie.jpg path/to/cnic.jpg

    # As a specific role, with the annotated overlay written out
    python scripts/demo_quality.py selfie.jpg --role live_selfie --annotate out/

    # Machine-readable, for piping into jq
    python scripts/demo_quality.py selfie.jpg --json

Detection runs first when model weights are available, so quality is measured
on the face that will actually be used. Without weights the module still works
and reports the face-dependent dimensions as unmeasured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.logging import configure_logging
from hamqadam_ai.models.registry import get_registry
from hamqadam_ai.schemas.detection import FaceDetectionResult
from hamqadam_ai.schemas.quality import QualityResult
from hamqadam_ai.services import build_face_detection_service, build_quality_service
from hamqadam_ai.utils.image_io import load_image

BgrImage = npt.NDArray[np.uint8]

BAR_WIDTH = 28

DIMENSIONS = (
    "blur",
    "sharpness",
    "brightness",
    "contrast",
    "noise",
    "resolution",
    "pixelation",
    "distortion",
)


def reference_portrait() -> BgrImage:
    """Public-domain portrait bundled with matplotlib, used when no path is given."""
    import matplotlib

    path = (
        Path(matplotlib.__file__).parent
        / "mpl-data" / "sample_data" / "grace_hopper.jpg"
    )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:  # pragma: no cover - depends on the matplotlib install
        raise SystemExit(f"Could not read the reference portrait at {path}")
    return image


def degradations(base: BgrImage) -> dict[str, BgrImage]:
    """A ladder of single, controlled defects for the built-in demonstration."""

    def jpeg(quality: int) -> BgrImage:
        ok, buffer = cv2.imencode(".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else base

    def motion(length: int) -> BgrImage:
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        return cv2.filter2D(base, -1, kernel)

    def upscaled(factor: float) -> BgrImage:
        height, width = base.shape[:2]
        small = cv2.resize(
            base,
            (int(width / factor), int(height / factor)),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)

    rng = np.random.default_rng(7)
    return {
        "original": base,
        "out of focus": cv2.GaussianBlur(base, (21, 21), 0),
        "camera shake": motion(15),
        "underexposed": np.clip(base * 0.35, 0, 255).astype(np.uint8),
        "washed out": np.clip(base * 0.30 + 110, 0, 255).astype(np.uint8),
        "high ISO noise": np.clip(
            base.astype(np.float32) + rng.normal(0, 20, base.shape), 0, 255
        ).astype(np.uint8),
        "over-compressed": jpeg(8),
        "enlarged thumbnail": upscaled(4.0),
    }


def bar(score: float, width: int = BAR_WIDTH) -> str:
    """Render a 0-100 score as a text bar."""
    filled = int(round(width * max(0.0, min(100.0, score)) / 100.0))
    return "#" * filled + "." * (width - filled)


def print_report(label: str, result: QualityResult) -> None:
    """Print the human-readable assessment for one image."""
    verdict = "USABLE" if result.usable else "REJECTED"
    print()
    print("=" * 78)
    print(f"  {label}")
    print("=" * 78)
    print(
        f"  overall {result.image_quality_score:6.1f} / 100"
        f"   (plain mean {result.arithmetic_score:5.1f})"
        f"   required {result.min_required:.0f}"
        f"   -> {verdict}"
    )
    print(
        f"  {result.image_width}x{result.image_height}"
        f"   face_analysed={result.face_analysed}"
        f"   {result.duration_ms:.0f} ms"
    )
    print()

    for name in DIMENSIONS:
        detail = result.metrics.get(name)
        if detail is None:
            continue
        if not detail.measured:
            print(f"    {name:<12s} {'unmeasured':>6s}  {'-' * BAR_WIDTH}")
            continue
        marker = " <-- limiting" if name == result.limiting_factor else ""
        critical = "  CRITICAL" if name in result.critical_failures else ""
        print(
            f"    {name:<12s} {detail.score:6.1f}  {bar(detail.score)}"
            f"{marker}{critical}"
        )

    if result.critical_failures:
        print()
        print(f"  critical failures : {', '.join(result.critical_failures)}")
    if result.unmeasured:
        print(f"  unmeasured        : {', '.join(result.unmeasured)}")

    notes = [w.message for w in result.warnings if w.code == "QUALITY_OBSERVATION"]
    if notes:
        print()
        print("  observations:")
        for note in notes:
            wrapped = _wrap(note, 70)
            for index, line in enumerate(wrapped):
                print(f"    - {line}" if index == 0 else f"      {line}")

    if result.error_message:
        print()
        print("  rejection reason:")
        for line in _wrap(result.error_message, 70):
            print(f"    {line}")


def _wrap(text: str, width: int) -> list[str]:
    """Naive word wrap, avoiding a textwrap import for four lines of output."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def print_measurements(result: QualityResult) -> None:
    """Print every raw measurement behind the scores."""
    print()
    print("  raw measurements:")
    for name in DIMENSIONS:
        detail = result.metrics.get(name)
        if detail is None or not detail.measurements:
            continue
        print(f"    {name}:")
        for key, value in sorted(detail.measurements.items()):
            print(f"      {key:<28s} {value:>14.5f}")


def annotate(
    image: BgrImage, result: QualityResult, detection: FaceDetectionResult | None
) -> BgrImage:
    """Draw the verdict and the dimension bars onto a copy of the image."""
    canvas = image.copy()
    height, width = canvas.shape[:2]

    if detection is not None and detection.primary_face is not None:
        box = detection.primary_face.bounding_box
        colour = (0, 200, 0) if result.usable else (0, 0, 220)
        cv2.rectangle(
            canvas,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            colour,
            max(2, width // 400),
        )

    panel_height = 26 * (len(DIMENSIONS) + 3)
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    scale = max(0.45, min(0.8, width / 1200.0))

    verdict = "USABLE" if result.usable else "REJECTED"
    colour = (0, 220, 0) if result.usable else (0, 0, 255)
    cv2.putText(
        panel,
        f"quality {result.image_quality_score:.1f}/100  (min {result.min_required:.0f})"
        f"  {verdict}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        1,
        cv2.LINE_AA,
    )

    row = 2
    for name in DIMENSIONS:
        detail = result.metrics.get(name)
        if detail is None:
            continue
        text = (
            f"{name:<12s} {'  --  ' if not detail.measured else f'{detail.score:6.1f}'}"
            f"  {bar(detail.score if detail.measured else 0.0, 20)}"
        )
        if name == result.limiting_factor:
            text += "  <- limiting"
        cv2.putText(
            panel,
            text,
            (10, 24 + row * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.75,
            (200, 200, 200) if detail.measured else (110, 110, 110),
            1,
            cv2.LINE_AA,
        )
        row += 1

    return np.vstack([canvas, panel])


def build_services():  # noqa: ANN201 - returns a heterogeneous tuple
    """Construct the detection and quality services.

    Detection is optional: without model weights the quality module still runs
    and reports the face-dependent dimensions as unmeasured.
    """
    settings = get_settings()
    quality = build_quality_service(settings)
    try:
        detector = build_face_detection_service(settings, get_registry(settings))
    except Exception as exc:  # noqa: BLE001 - the demo must run without weights
        print(f"  (face detection unavailable: {exc})")
        print("  (face-dependent dimensions will be reported as unmeasured)")
        detector = None
    return quality, detector


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Demonstrate MODULE 2 image and face quality assessment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("images", nargs="*", type=Path, help="Image files to assess.")
    parser.add_argument(
        "--role",
        default="profile_image",
        choices=[role.value for role in ImageRole],
        help="Image role, which selects the acceptance threshold.",
    )
    parser.add_argument(
        "--annotate",
        type=Path,
        metavar="DIR",
        help="Write annotated overlays into this directory.",
    )
    parser.add_argument(
        "--measurements", action="store_true", help="Print every raw measurement."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv)

    configure_logging(get_settings())
    role = ImageRole(args.role)

    print("=" * 78)
    print("HAMQADAM AI - MODULE 2: IMAGE AND FACE QUALITY")
    print("=" * 78)
    quality, detector = build_services()

    if args.images:
        cases = {}
        for path in args.images:
            if not path.is_file():
                print(f"  skipping {path}: not a file")
                continue
            cases[path.name] = load_image(path).pixels
        if not cases:
            parser.error("none of the supplied paths could be read")
    else:
        print("  no images supplied; using the reference portrait and a")
        print("  ladder of single controlled degradations")
        cases = degradations(reference_portrait())

    payloads: dict[str, dict] = {}
    for label, image in cases.items():
        detection = None
        if detector is not None:
            detection = detector.detect(image, role=role)

        result = quality.assess(image, role=role, detection=detection)
        payloads[label] = result.model_dump(mode="json")

        if not args.json:
            print_report(label, result)
            if args.measurements:
                print_measurements(result)

        if args.annotate:
            args.annotate.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
            target = args.annotate / f"{safe}.jpg"
            cv2.imwrite(str(target), annotate(image, result, detection))
            if not args.json:
                print(f"\n  annotated -> {target}")

    if args.json:
        print(json.dumps(payloads, indent=2))
        return 0

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'image':<22s}{'score':>8s}{'verdict':>11s}  limiting")
    print("  " + "-" * 60)
    for label, payload in payloads.items():
        verdict = "USABLE" if payload["usable"] else "REJECTED"
        print(
            f"  {label[:21]:<22s}{payload['image_quality_score']:>8.1f}"
            f"{verdict:>11s}  {payload['limiting_factor'] or '-'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
