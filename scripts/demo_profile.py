"""MODULE 7 demonstration - is this a genuine photograph of you?

Usage
-----
::

    # Built-in scenarios, including the hard negatives
    python scripts/demo_profile.py

    # Your own image
    python scripts/demo_profile.py --image me.jpg

    # Just the authenticity detectors, no face models needed
    python scripts/demo_profile.py --image me.jpg --detectors-only

    # Machine-readable
    python scripts/demo_profile.py --json

Every built-in image is a public-domain reference photograph or is
synthesised. No private individual's photograph appears in this repository.

What the run demonstrates
-------------------------
Four detectors catch four different ways a profile photo can fail to be what
it claims. The interesting part is the middle block: the **hard negatives**, a
heavily-compressed photograph and a square-padded one. Both look superficially
like the things being detected - a quality-12 JPEG has as many flat blocks as
a real screenshot, and a padded upload has as many constant rows - and both
must come through clean, because they are what ordinary uploads look like
after a messaging app has had them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hamqadam_ai.authenticity import AuthenticityAssessment  # noqa: E402
from hamqadam_ai.logging import configure_logging  # noqa: E402
from hamqadam_ai.schemas.profile import ProfileAnalysisResult  # noqa: E402
from hamqadam_ai.services import build_profile_service  # noqa: E402
from hamqadam_ai.services.profile_service import ProfileAnalysisService  # noqa: E402
from hamqadam_ai.utils.image_io import load_image  # noqa: E402

BgrImage = npt.NDArray[np.uint8]


def build_scenarios() -> list[tuple[str, str, BgrImage]]:
    """``(group, label, image)`` for every built-in case."""
    from tests.fixtures.profile_images import (
        as_heavily_compressed,
        as_print_recapture,
        as_recompressed,
        as_screen_recapture,
        as_screenshot,
        as_synthetic_render,
        flat_colour,
        landscape_photo,
        reference_photo,
    )

    photo = reference_photo()
    if photo is None:
        raise SystemExit(
            "No public-domain reference photograph is installed.\n"
            "Install one with:  pip install matplotlib scikit-image"
        )

    def letterbox(image: BgrImage, pad: float = 0.25) -> BgrImage:
        rows = int(image.shape[0] * pad)
        return cv2.copyMakeBorder(
            image, rows, rows, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    scenarios: list[tuple[str, str, BgrImage]] = [
        ("genuine", "a real photograph", photo),
        ("hard negative", "JPEG quality 12", as_heavily_compressed(photo)),
        ("hard negative", "recompressed 92 then 74", as_recompressed(photo)),
        ("hard negative", "square-padded upload", letterbox(photo)),
        ("impostor", "screenshot of an app", as_screenshot(photo)),
        ("impostor", "photograph of a screen", as_screen_recapture(photo)),
        ("impostor", "photograph of a print", as_print_recapture(photo)),
        ("impostor", "cartoon avatar", as_synthetic_render()),
        ("evasion", "screenshot, cropped 10%", _crop(as_screenshot(photo), 0.10)),
        ("evasion", "screen capture, downscaled 4x",
         cv2.resize(as_screen_recapture(photo), None, fx=0.25, fy=0.25,
                    interpolation=cv2.INTER_AREA)),
        ("not a portrait", "a landscape", landscape_photo()),
        ("not a portrait", "a flat colour field", flat_colour()),
    ]
    return [(g, label, image) for g, label, image in scenarios if image is not None]


def _crop(image: BgrImage, fraction: float) -> BgrImage:
    """Trim a fraction off the top and bottom."""
    height = image.shape[0]
    return image[int(height * fraction / 2):int(height * (1 - fraction / 2))]


def summarise(group: str, label: str, result: ProfileAnalysisResult) -> str:
    """One line per scenario."""
    finding = result.findings[0].code if result.findings else "-"
    return (
        f"{group:<15s} {label:<30s} "
        f"{'OK  ' if result.usable_as_profile else 'FAIL'}  "
        f"auth {result.authenticity_score:>5.1f}  "
        f"faces {result.face_count}  "
        f"qual {result.image_quality_score:>5.1f}  "
        f"{result.duration_ms:>6.0f} ms  {finding}"
    )


def render_detail(result: ProfileAnalysisResult) -> str:
    """Every reading behind one verdict."""
    lines: list[str] = ["  detector readings"]
    for signal in result.signals:
        marker = "->" if signal.triggered else "  "
        if not signal.measured:
            lines.append(f"   {marker} {signal.name:<18s} not measured: {signal.note}")
            continue
        values = "  ".join(
            f"{key}={value:.4g}" for key, value in signal.measurements.items()
        )
        lines.append(
            f"   {marker} {signal.name:<18s} conf {signal.confidence:>5.3f}   {values}"
        )

    lines.append("  subject")
    lines.append(f"    face detected      {result.face_detected}")
    lines.append(f"    faces             {result.face_count}")
    lines.append(f"    group photo       {result.is_group_photo}")
    lines.append(f"    visibility        {result.face_visibility_score:.1f}")

    lines.append("  quality")
    lines.append(f"    score             {result.image_quality_score:.1f}")
    lines.append(f"    usable            {result.quality_usable}")

    if result.error_message:
        lines.append(f"  error   {result.error_code}: {result.error_message}")

    return "\n".join(lines)


def render_assessment(assessment: AuthenticityAssessment) -> str:
    """The detectors-only view."""
    lines = [f"  authenticity score  {assessment.score:.1f}", "  readings"]
    for signal in assessment.signals:
        marker = "->" if signal.triggered else "  "
        if not signal.measured:
            lines.append(f"   {marker} {signal.name:<18s} not measured: {signal.note}")
            continue
        values = "  ".join(
            f"{key}={value:.4g}" for key, value in signal.measurements.items()
        )
        lines.append(
            f"   {marker} {signal.name:<18s} conf {signal.confidence:>5.3f}   {values}"
        )
    for finding in assessment.findings:
        lines.append(f"  finding  {finding.code}: {finding.message}")
    return "\n".join(lines)


def run_builtin(service: ProfileAnalysisService, *, as_json: bool) -> int:
    """Analyse every built-in scenario."""
    scenarios = build_scenarios()
    results = [(g, label, service.analyse(image)) for g, label, image in scenarios]

    if as_json:
        print(json.dumps(
            {label: result.model_dump(mode="json") for _g, label, result in results},
            indent=2,
        ))
        return 0

    print()
    print("Public-domain reference photographs and synthesised impostors")
    print("=" * 118)
    for group, label, result in results:
        print(summarise(group, label, result))

    print()
    print("A genuine photograph, in full")
    print("-" * 118)
    print(render_detail(results[0][2]))

    print()
    print("A screenshot, in full")
    print("-" * 118)
    screenshot = next(r for _g, label, r in results if label == "screenshot of an app")
    print(render_detail(screenshot))

    print()
    print("Known evasions, reported honestly")
    print("-" * 118)
    print(
        "  Cropping a screenshot puts its flat chrome against the new frame edge,\n"
        "  where it is stripped as padding - the same strip that stops a\n"
        "  square-padded upload being falsely accused. The two are geometrically\n"
        "  identical. Downscaling a screen capture destroys the display grid the\n"
        "  moire detector needs. Both are documented limits, not surprises."
    )

    # The hard negatives are the ones that must NOT be flagged.
    wrongly_accused = [
        label for group, label, result in results
        if group in {"genuine", "hard negative"} and result.findings
    ]
    missed = [
        label for group, label, result in results
        if group == "impostor" and not result.findings
    ]

    print()
    if wrongly_accused:
        print(f"FALSE POSITIVES: {', '.join(wrongly_accused)}")
    if missed:
        print(f"MISSED IMPOSTORS: {', '.join(missed)}")
    if wrongly_accused or missed:
        return 1

    print(
        "Every genuine photograph and hard negative passed clean; "
        "every impostor was caught with actionable guidance."
    )
    return 0


def run_single(
    service: ProfileAnalysisService,
    path: Path,
    *,
    as_json: bool,
    detectors_only: bool,
) -> int:
    """Analyse one image from disk."""
    image = load_image(path)

    if detectors_only:
        assessment = service.assess_authenticity(image)
        if as_json:
            print(json.dumps(assessment.describe(), indent=2))
            return 0
        print()
        print(f"{path.name}")
        print(render_assessment(assessment))
        return 0 if assessment.clean else 1

    result = service.analyse(image)
    if as_json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return 0

    print()
    print(summarise("", path.name, result))
    print(render_detail(result))
    return 0 if result.usable_as_profile else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Analyse a photograph for use as a profile picture."
    )
    parser.add_argument("--image", type=Path, help="Analyse this image.")
    parser.add_argument(
        "--detectors-only",
        action="store_true",
        help="Run only the authenticity detectors, which need no model weights.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    arguments = parser.parse_args(argv)

    configure_logging()

    try:
        service = build_profile_service()
    except Exception as exc:  # noqa: BLE001 - a missing model is a clean exit
        print(f"Profile service unavailable: {exc}", file=sys.stderr)
        print("Fetch the weights with:  python scripts/download_models.py",
              file=sys.stderr)
        return 2

    if arguments.image:
        return run_single(
            service,
            arguments.image,
            as_json=arguments.json,
            detectors_only=arguments.detectors_only,
        )
    return run_builtin(service, as_json=arguments.json)


if __name__ == "__main__":
    raise SystemExit(main())
