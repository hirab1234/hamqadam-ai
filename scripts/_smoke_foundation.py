"""Foundation smoke check: configuration, logging, redaction, geometry, scratch space.

Run with:  python scripts/_smoke_foundation.py

Exercises every part of Module 0 that has no model-weight dependency, so it can
be used as a fast post-deploy sanity check on any host.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from hamqadam_ai.core.config import get_settings
from hamqadam_ai.core.context import RequestContext, request_context
from hamqadam_ai.core.device import resolve_from_settings
from hamqadam_ai.logging import configure_logging, get_logger
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5, nms
from hamqadam_ai.utils.image_io import decode_image, encode_image
from hamqadam_ai.utils.tempfiles import scratch_space
from hamqadam_ai.utils.timing import StageTimings


def main() -> int:
    """Run the checks and return a process exit code."""
    settings = get_settings()
    configure_logging(settings)
    log = get_logger("smoke")

    print("=" * 72)
    print("CONFIGURATION")
    print("=" * 72)
    print(f"  app              : {settings.app.name} v{settings.app.version}")
    print(f"  environment      : {settings.app.environment}")
    print(f"  config_dir       : {settings.config_dir_}")
    print(f"  models declared  : {sorted(settings.models)}")
    print(f"  detector chain   : {settings.detection.fallback_chain}")
    visibility = settings.detection.visibility.weights
    print(f"  visibility wts   : sum={sum(visibility.values()):.6f} {visibility}")
    occlusion = settings.detection.occlusion.region_weights
    print(f"  occlusion wts    : sum={sum(occlusion.values()):.6f}")
    print(f"  model store      : {settings.storage.resolved_model_dir}")
    print(f"  scratch root     : {settings.storage.resolved_temp_dir}")

    print()
    print("=" * 72)
    print("DEVICE RESOLUTION")
    print("=" * 72)
    plan = resolve_from_settings(settings)
    for key, value in plan.describe().items():
        print(f"  {key:18s}: {value}")

    print()
    print("=" * 72)
    print("GEOMETRY")
    print("=" * 72)
    box_a = BoundingBox(10, 10, 110, 110)
    box_b = BoundingBox(60, 60, 160, 160)
    print(f"  box_a area       : {box_a.area:.1f}")
    print(f"  iou(a, b)        : {box_a.iou(box_b):.6f}  (expected 0.142857)")
    print(f"  a.to_square()    : {BoundingBox(10, 10, 110, 210).to_square().as_tuple()}")
    print(f"  truncation @80x80: {box_a.truncation_ratio(80, 80):.4f}")

    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    print(f"  nms(iou=0.4)     : {nms(boxes, scores, 0.4).tolist()}  (expected [0, 2])")

    landmarks = Landmarks5(
        np.array(
            [[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]],
            dtype=np.float32,
        )
    )
    print(f"  interocular      : {landmarks.interocular_distance:.3f}")
    print(f"  roll (deg)       : {landmarks.roll_degrees:.3f}")
    print(f"  plausible        : {landmarks.is_plausible()}  (expected True)")

    flipped = Landmarks5(
        np.array(
            [[38.3, 92.4], [73.5, 92.2], [56.0, 71.7], [41.5, 51.7], [70.7, 51.5]],
            dtype=np.float32,
        )
    )
    print(f"  upside-down      : {flipped.is_plausible()}  (expected False)")

    print()
    print("=" * 72)
    print("IMAGE CODEC")
    print("=" * 72)
    gradient = np.zeros((240, 320, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.linspace(0, 255, 320, dtype=np.uint8)[None, :]
    gradient[:, :, 1] = np.linspace(0, 255, 240, dtype=np.uint8)[:, None]
    encoded = encode_image(gradient, fmt="PNG")
    decoded = decode_image(encoded, role="smoke")
    print(f"  encoded bytes    : {len(encoded)}")
    print(f"  decoded          : {decoded.describe()}")

    for label, payload in (
        ("empty payload", b""),
        ("not an image", b"this is plainly not an image at all"),
        ("truncated jpeg", b"\xff\xd8\xff\xe0" + b"\x00" * 512),
    ):
        try:
            decode_image(payload, role="smoke")
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            code = getattr(exc, "code", type(exc).__name__)
            print(f"  reject {label:15s}: {code}")
        else:
            print(f"  reject {label:15s}: NOT REJECTED  <-- FAILURE")
            return 1

    uniform = np.full((200, 200, 3), 137, dtype=np.uint8)
    try:
        decode_image(encode_image(uniform, fmt="PNG"), role="smoke")
    except Exception as exc:  # noqa: BLE001
        print(f"  reject {'uniform image':15s}: {getattr(exc, 'code', '?')}")
    else:
        print(f"  reject {'uniform image':15s}: NOT REJECTED  <-- FAILURE")
        return 1

    print()
    print("=" * 72)
    print("SCRATCH SPACE (no permanent storage)")
    print("=" * 72)
    with scratch_space(settings.storage.resolved_temp_dir, label="smoke") as space:
        written = space.write_bytes("payload.bin", b"sensitive-cnic-bytes" * 64)
        scratch_dir = space.path
        print(f"  created          : {scratch_dir.name}")
        print(f"  file exists      : {written.is_file()}  size={written.stat().st_size}")
        try:
            space.file("../escape.txt")
        except ValueError:
            print("  path traversal   : blocked")
    print(f"  after close      : dir_exists={Path(scratch_dir).exists()}  (expected False)")

    print()
    print("=" * 72)
    print("TIMING")
    print("=" * 72)
    timings = StageTimings()
    with timings.measure("detection"):
        _ = np.linalg.svd(np.random.default_rng(0).random((120, 120)))
    with timings.measure("quality"):
        _ = np.fft.fft2(np.random.default_rng(1).random((256, 256)))
    timings.finish()
    report = timings.as_dict()
    print(f"  total            : {report['total']} {report['unit']}")
    for name, entry in report["stages"].items():
        print(f"  stage {name:12s}: {entry['duration']} ms x{entry['calls']}")

    print()
    print("=" * 72)
    print("PII REDACTION (the next lines are the raw log output)")
    print("=" * 72)
    context = RequestContext.create(verification_id="VER-123456", user_id="user-42")
    with request_context(context):
        log.info(
            "smoke.redaction_probe",
            cnic_number="35202-1234567-1",
            full_name="Ubaid Malik",
            ocr_dump="Name: Ubaid Malik  CNIC 35202-1234567-1  mail ubaid@example.com",
            authorization="Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
            image=b"\x00" * 4096,
            embedding=np.zeros(512, dtype=np.float32),
            face_count=1,
            confidence=0.973,
        )

    print()
    print("Foundation smoke check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
