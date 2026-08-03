"""MODULE 1 demonstration - real inference on real images.

Runs the complete face-detection service and prints the standardised result for
a series of scenarios, including every rejection path the specification
requires. Renders annotated output images so the boxes, landmarks, pose axes
and occlusion verdict can be inspected visually.

Usage::

    python scripts/demo_face_detection.py
    python scripts/demo_face_detection.py --image path/to/photo.jpg
    python scripts/demo_face_detection.py --output-dir reports/demo

Test imagery
------------
With no ``--image`` the demo uses ``grace_hopper.jpg``, the US Navy portrait of
Rear Admiral Grace Hopper that ships inside matplotlib. It is a genuine
photograph of a real face, it is in the public domain as a work of the US
federal government, and it needs no download - so this demo is reproducible on
any machine without shipping a real identity document into the repository.

Derived scenarios (crowd, occluded, tiny, truncated) are constructed from that
same image by composition, so every rejection path is exercised against real
detector output rather than a mock.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.constants import ImageRole
from hamqadam_ai.logging import configure_logging
from hamqadam_ai.models.registry import get_registry
from hamqadam_ai.schemas.detection import FaceDetectionResult
from hamqadam_ai.services import build_face_detection_service
from hamqadam_ai.utils.image_io import encode_image, load_image

BgrImage = npt.NDArray[np.uint8]

_BOX_PRIMARY = (80, 220, 90)
_BOX_REJECTED = (60, 60, 220)
_BOX_BYSTANDER = (170, 170, 170)
_TEXT = (255, 255, 255)


def default_image() -> Path:
    """Locate the public-domain portrait bundled with matplotlib."""
    import matplotlib

    path = (
        Path(matplotlib.__file__).parent
        / "mpl-data"
        / "sample_data"
        / "grace_hopper.jpg"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "matplotlib's sample portrait is unavailable. Pass --image explicitly."
        )
    return path


# --------------------------------------------------------------------------- #
# Scenario construction
# --------------------------------------------------------------------------- #


def scenario_clean(base: BgrImage) -> BgrImage:
    """The source photograph, unmodified."""
    return base.copy()


def scenario_no_face(base: BgrImage) -> BgrImage:
    """A textured landscape with no person in it."""
    height, width = base.shape[:2]
    rng = np.random.default_rng(11)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # A gradient sky over textured ground: plenty of edges, no face.
    canvas[:, :, 0] = np.linspace(180, 90, height, dtype=np.uint8)[:, None]
    canvas[:, :, 1] = np.linspace(150, 110, height, dtype=np.uint8)[:, None]
    canvas[:, :, 2] = np.linspace(110, 130, height, dtype=np.uint8)[:, None]
    ground = int(height * 0.62)
    canvas[ground:] = (60, 95, 70)
    noise = rng.normal(0, 18, canvas.shape)
    return np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def scenario_two_people(base: BgrImage) -> BgrImage:
    """The same face twice, side by side - the single-person rule must fire."""
    height, width = base.shape[:2]
    canvas = np.full((height, width * 2, 3), 100, dtype=np.uint8)
    canvas[:, :width] = base
    canvas[:, width:] = cv2.flip(base, 1)
    return canvas


def scenario_sunglasses(base: BgrImage, service) -> BgrImage:  # noqa: ANN001
    """A dark bar over the eyes, positioned from the real detected landmarks."""
    result = service.detect(base, role=ImageRole.LIVE_SELFIE)
    canvas = base.copy()
    if result.primary_face is None or not result.primary_face.landmarks:
        return canvas

    marks = {lm.name: (lm.x, lm.y) for lm in result.primary_face.landmarks}
    left = marks.get("left_eye")
    right = marks.get("right_eye")
    if left is None or right is None:
        return canvas

    separation = math.hypot(right[0] - left[0], right[1] - left[1])
    centre = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))

    rect = ((centre[0], centre[1]), (separation * 2.0, separation * 0.72), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(canvas, box, (18, 18, 20))
    return canvas


def scenario_face_mask(base: BgrImage, service) -> BgrImage:  # noqa: ANN001
    """A flat covering over nose, mouth and chin."""
    result = service.detect(base, role=ImageRole.LIVE_SELFIE)
    canvas = base.copy()
    if result.primary_face is None or not result.primary_face.landmarks:
        return canvas

    marks = {lm.name: (lm.x, lm.y) for lm in result.primary_face.landmarks}
    left = marks["left_eye"]
    right = marks["right_eye"]
    nose = marks["nose_tip"]
    separation = math.hypot(right[0] - left[0], right[1] - left[1])

    rect = (
        (nose[0], nose[1] + separation * 0.45),
        (separation * 2.1, separation * 1.9),
        math.degrees(math.atan2(right[1] - left[1], right[0] - left[0])),
    )
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(canvas, box, (208, 202, 190))
    return canvas


def scenario_tiny_face(base: BgrImage) -> BgrImage:
    """The face shrunk into a corner of a large frame."""
    small = cv2.resize(base, None, fx=0.13, fy=0.13, interpolation=cv2.INTER_AREA)
    canvas = np.full((1200, 1600, 3), 130, dtype=np.uint8)
    rng = np.random.default_rng(3)
    canvas = np.clip(
        canvas.astype(np.float32) + rng.normal(0, 10, canvas.shape), 0, 255
    ).astype(np.uint8)
    canvas[60 : 60 + small.shape[0], 60 : 60 + small.shape[1]] = small
    return canvas


def scenario_truncated(base: BgrImage) -> BgrImage:
    """Half the face cropped away by the frame edge."""
    height, width = base.shape[:2]
    return np.ascontiguousarray(base[:, : int(width * 0.42)])


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def annotate(image: BgrImage, result: FaceDetectionResult) -> BgrImage:
    """Draw boxes, landmarks, pose axes and a verdict banner."""
    canvas = image.copy()
    scale = max(0.45, min(1.1, image.shape[1] / 900.0))

    for face in result.faces:
        box = face.bounding_box
        if face.is_primary:
            colour = _BOX_PRIMARY
        elif face.is_bystander:
            colour = _BOX_BYSTANDER
        else:
            colour = _BOX_REJECTED
        thickness = 3 if face.is_primary else 2

        cv2.rectangle(
            canvas,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            colour,
            thickness,
        )

        label = f"{face.confidence:.2f} vis={face.face_visibility_score:.0f}"
        if face.rejection_reasons:
            label += f" [{face.rejection_reasons[0]}]"
        elif face.is_bystander:
            label += " [bystander]"
        _label(canvas, label, (int(box.x1), int(box.y1) - 6), colour, scale * 0.55)

        for landmark in face.landmarks:
            cv2.circle(canvas, (int(landmark.x), int(landmark.y)), 2, (60, 220, 255), -1)

        if face.pose is not None and face.is_primary:
            _draw_pose_axes(canvas, face)

        if face.occlusion is not None and face.occlusion.occluded_regions:
            _label(
                canvas,
                "occluded: " + ",".join(face.occlusion.occluded_regions),
                (int(box.x1), int(box.y2) + 16),
                (60, 120, 240),
                scale * 0.50,
            )

    verdict = "PASS" if result.passed else f"REJECT {result.error_code}"
    banner = (
        f"{verdict} | {result.detector} | faces={result.face_count} | "
        f"vis={result.face_visibility_score:.1f} | {result.duration_ms:.0f}ms"
    )
    _banner(canvas, banner, _BOX_PRIMARY if result.passed else _BOX_REJECTED, scale)
    return canvas


def _draw_pose_axes(canvas: BgrImage, face) -> None:  # noqa: ANN001
    """Draw a small yaw/pitch/roll gnomon at the nose tip."""
    marks = {lm.name: (lm.x, lm.y) for lm in face.landmarks}
    nose = marks.get("nose_tip")
    if nose is None:
        return

    length = max(28.0, face.bounding_box.width * 0.28)
    yaw = math.radians(face.pose.yaw)
    pitch = math.radians(face.pose.pitch)
    roll = math.radians(face.pose.roll)

    origin = (int(nose[0]), int(nose[1]))
    # X axis (red) shows roll and yaw; Y axis (green) shows roll and pitch.
    x_tip = (
        int(nose[0] + length * math.cos(yaw) * math.cos(roll)),
        int(nose[1] + length * math.cos(yaw) * math.sin(roll)),
    )
    y_tip = (
        int(nose[0] - length * math.cos(pitch) * math.sin(roll)),
        int(nose[1] - length * math.cos(pitch) * math.cos(roll)),
    )
    cv2.arrowedLine(canvas, origin, x_tip, (60, 60, 230), 2, tipLength=0.25)
    cv2.arrowedLine(canvas, origin, y_tip, (60, 230, 60), 2, tipLength=0.25)
    _label(
        canvas,
        f"y{face.pose.yaw:+.0f} p{face.pose.pitch:+.0f} r{face.pose.roll:+.0f}",
        (origin[0] + 8, origin[1] + 18),
        (255, 240, 200),
        0.45,
    )


def _label(
    canvas: BgrImage,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float,
) -> None:
    """Draw text with a dark backing plate so it stays readable."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), _ = cv2.getTextSize(text, font, scale, 1)
    x, y = origin
    y = max(y, height + 4)
    cv2.rectangle(canvas, (x - 2, y - height - 4), (x + width + 4, y + 4), (25, 25, 25), -1)
    cv2.putText(canvas, text, (x, y), font, scale, colour, 1, cv2.LINE_AA)


def _banner(
    canvas: BgrImage, text: str, colour: tuple[int, int, int], scale: float
) -> None:
    """Draw the verdict banner across the top of the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    height = int(30 * scale)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], height), (20, 20, 20), -1)
    cv2.putText(
        canvas, text, (8, int(height * 0.72)), font, scale * 0.55, colour, 1, cv2.LINE_AA
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def report(name: str, result: FaceDetectionResult) -> None:
    """Print the specification-mandated fields plus supporting evidence."""
    print()
    print("=" * 78)
    print(f"SCENARIO: {name}")
    print("=" * 78)
    print(f"  face_detected          : {result.face_detected}")
    print(f"  face_count             : {result.face_count}")
    print(f"  face_visibility_score  : {result.face_visibility_score}")
    print(f"  passed                 : {result.passed}")
    if result.error_code:
        print(f"  error_code             : {result.error_code}")
        print(f"  error_message          : {result.error_message}")
    print(f"  detector               : {result.detector} ({result.detector_version})")
    print(f"  raw_detections         : {result.raw_detection_count}")
    print(f"  duration_ms            : {result.duration_ms:.1f}")

    face = result.primary_face
    if face is not None:
        box = face.bounding_box
        print("  --- primary face ---")
        print(
            f"  bounding_box           : "
            f"({box.x1:.0f}, {box.y1:.0f}) -> ({box.x2:.0f}, {box.y2:.0f})  "
            f"{box.width:.0f}x{box.height:.0f}px"
        )
        print(f"  confidence             : {face.confidence:.4f}")
        print(f"  face_area_ratio        : {face.face_area_ratio:.4f}")
        print(f"  truncation_ratio       : {face.truncation_ratio:.4f}")
        if face.pose is not None:
            print(
                f"  pose                   : yaw={face.pose.yaw:+.1f} "
                f"pitch={face.pose.pitch:+.1f} roll={face.pose.roll:+.1f} "
                f"[{face.pose.method}] frontal={face.pose.frontal}"
            )
            if face.pose.reprojection_error is not None:
                print(f"  reprojection_error     : {face.pose.reprojection_error:.2f}px")
        if face.occlusion is not None:
            print(
                f"  occlusion              : score={face.occlusion.overall_score:.3f} "
                f"occluded={face.occlusion.occluded} "
                f"symmetry={face.occlusion.symmetry_delta:.3f}"
            )
            for region in face.occlusion.regions:
                flag = "OCCLUDED" if region.occluded else "clear   "
                print(
                    f"      {region.region:<10s} {flag}  "
                    f"p={region.occlusion_probability:.3f}  "
                    f"flat={region.flat_fraction:.2f}  "
                    f"texture={region.texture_energy:.2f}  "
                    f"skin={region.skin_coverage:.2f}"
                )
        if face.visibility_breakdown is not None:
            breakdown = face.visibility_breakdown
            print(
                f"  visibility components  : conf={breakdown.detector_confidence:.3f} "
                f"occl={breakdown.occlusion:.3f} pose={breakdown.pose:.3f} "
                f"size={breakdown.face_size:.3f} framing={breakdown.framing:.3f}"
            )
            print(f"  limiting_factor        : {breakdown.limiting_factor}")

    if result.warnings:
        print("  --- warnings ---")
        for warning in result.warnings:
            print(f"      [{warning.code}] {warning.message}")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Demonstrate Module 1 face detection.")
    parser.add_argument("--image", type=Path, help="Portrait to analyse.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/demo/module1"),
        help="Where annotated images are written.",
    )
    parser.add_argument("--json", action="store_true", help="Dump the full result JSON.")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)

    source_path = args.image or default_image()
    base = load_image(source_path, role="demo").pixels

    print("=" * 78)
    print("HAMQADAM AI - MODULE 1 FACE DETECTION DEMONSTRATION")
    print("=" * 78)
    print(f"  source image : {source_path}")
    print(f"  dimensions   : {base.shape[1]}x{base.shape[0]}")

    registry = get_registry(settings)
    service = build_face_detection_service(settings, registry)
    print(f"  detector     : {json.dumps(service.describe()['chain'], indent=2)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenarios: list[tuple[str, Callable[[], BgrImage], ImageRole]] = [
        ("clean_portrait", lambda: scenario_clean(base), ImageRole.LIVE_SELFIE),
        ("no_face", lambda: scenario_no_face(base), ImageRole.PROFILE_IMAGE),
        ("two_people", lambda: scenario_two_people(base), ImageRole.PROFILE_IMAGE),
        ("sunglasses", lambda: scenario_sunglasses(base, service), ImageRole.LIVE_SELFIE),
        ("face_mask", lambda: scenario_face_mask(base, service), ImageRole.LIVE_SELFIE),
        ("tiny_face", lambda: scenario_tiny_face(base), ImageRole.PROFILE_IMAGE),
        ("truncated", lambda: scenario_truncated(base), ImageRole.PROFILE_IMAGE),
    ]

    outcomes: list[tuple[str, FaceDetectionResult]] = []
    for name, build, role in scenarios:
        image = build()
        result = service.detect(image, role=role, analyse_all_faces=True)
        outcomes.append((name, result))
        report(name, result)

        destination = args.output_dir / f"{name}.jpg"
        destination.write_bytes(encode_image(annotate(image, result), fmt="JPEG"))

        if args.json:
            print("  --- full JSON ---")
            print(
                json.dumps(
                    result.model_dump(mode="json", exclude={"faces"}),
                    indent=2,
                    default=str,
                )
            )

    # ---- Latency ------------------------------------------------------- #
    print()
    print("=" * 78)
    print("LATENCY (clean portrait, 30 iterations after 3 warm-up)")
    print("=" * 78)
    for _ in range(3):
        service.detect(base, role=ImageRole.LIVE_SELFIE)
    samples: list[float] = []
    for _ in range(30):
        started = time.perf_counter()
        service.detect(base, role=ImageRole.LIVE_SELFIE)
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    print(f"  device   : {registry.device_plan.family} "
          f"({', '.join(registry.device_plan.providers)})")
    print(f"  min      : {samples[0]:7.1f} ms")
    print(f"  p50      : {samples[len(samples) // 2]:7.1f} ms")
    print(f"  p95      : {samples[int(len(samples) * 0.95)]:7.1f} ms")
    print(f"  max      : {samples[-1]:7.1f} ms")

    # ---- Summary -------------------------------------------------------- #
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'scenario':<18s} {'passed':<8s} {'faces':<7s} {'vis':<7s} error")
    for name, result in outcomes:
        print(
            f"  {name:<18s} {str(result.passed):<8s} {result.face_count:<7d} "
            f"{result.face_visibility_score:<7.1f} {result.error_code or '-'}"
        )
    print()
    print(f"  annotated images written to {args.output_dir.resolve()}")

    service.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
