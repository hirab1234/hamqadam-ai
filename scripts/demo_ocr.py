"""MODULE 5 demonstration - read a Pakistani CNIC.

Usage
-----
::

    # Built-in scenarios: a clean card and six realistic degradations
    python scripts/demo_ocr.py

    # Your own image
    python scripts/demo_ocr.py --image card.jpg

    # Machine-readable
    python scripts/demo_ocr.py --json

Every built-in card is synthetic and describes a wholly fictitious holder. No
real identity document is included in this repository, and none should be
added: the service is built specifically to avoid retaining that data.

The run walks a clean scan, a photograph on a desk, dim light, laminate
glare, a low-resolution capture, heavy JPEG compression, sensor noise, and
each of the three rotations - then two cards that are not readable as a CNIC
at all, to show what the caller gets back when the upload is wrong.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hamqadam_ai.logging import configure_logging  # noqa: E402
from hamqadam_ai.schemas.ocr import CnicOcrResult  # noqa: E402
from hamqadam_ai.services.ocr_service import OcrService, build_ocr_service  # noqa: E402
from hamqadam_ai.utils.image_io import load_image  # noqa: E402

BgrImage = npt.NDArray[np.uint8]


def build_scenarios() -> list[tuple[str, BgrImage]]:
    """The built-in cards, in the order they are demonstrated."""
    from tests.fixtures.synthetic_cnic import (
        CnicSpec,
        dim_lighting,
        glare,
        jpeg_artifacts,
        low_resolution,
        photograph_on_desk,
        render_cnic,
        render_unrelated_document,
        rotated,
        sensor_noise,
    )

    card = render_cnic()
    expired = render_cnic(
        CnicSpec(
            date_of_issue=dt.date(2010, 5, 4), date_of_expiry=dt.date(2020, 5, 4)
        )
    )
    tampered = render_cnic(CnicSpec(gender="M", cnic_number="42101-8375926-4"))

    return [
        ("clean scan", card),
        ("on a desk, tilted", photograph_on_desk(card)),
        ("dim lighting", dim_lighting(card)),
        ("glare on laminate", glare(card)),
        ("low resolution", low_resolution(card)),
        ("jpeg quality 25", jpeg_artifacts(card)),
        ("sensor noise", sensor_noise(card)),
        ("rotated 90", rotated(card, 90)),
        ("rotated 180", rotated(card, 180)),
        ("rotated 270", rotated(card, 270)),
        ("expired card", expired),
        ("gender contradicts the number", tampered),
        ("a utility bill, not a CNIC", render_unrelated_document()),
        ("a blank page", np.full((640, 1012, 3), 245, dtype=np.uint8)),
    ]


def summarise(label: str, result: CnicOcrResult) -> str:
    """One line per card: did it read, how well, and how long it took."""
    status = "OK  " if result.success else "FAIL"
    return (
        f"{label:<32s} {status}  "
        f"conf {result.ocr_confidence_score:>5.1f}  "
        f"fields {len(result.fields_present)}/6  "
        f"rot {result.rotation_applied:>3d}  "
        f"passes {result.orientation_attempts}  "
        f"{result.duration_ms:>7.0f} ms"
    )


def render_detail(result: CnicOcrResult) -> str:
    """The full reading of one card, laid out for a human."""
    lines: list[str] = []

    def row(name: str, value: object, source: str | None = None) -> None:
        shown = "-" if value in (None, "") else str(value)
        suffix = f"   [{source}]" if source else ""
        lines.append(f"    {name:<18s} {shown}{suffix}")

    lines.append("  fields")
    row("identity number", result.cnic_number, result.fields["cnic_number"].source)
    row("name", result.name, result.fields["name"].source)
    row("father's name", result.father_name, result.fields["father_name"].source)
    row("gender", result.gender, result.fields["gender"].source)
    row("date of birth", result.date_of_birth, result.fields["date_of_birth"].source)
    row("date of issue", result.issue_date, result.fields["issue_date"].source)
    row(
        "date of expiry",
        "Lifetime" if result.is_lifetime else result.expiry_date,
        result.fields["expiry_date"].source,
    )

    lines.append("  derived from the number")
    row("province", result.province)
    row("implied gender", result.implied_gender)

    lines.append("  checks")
    row("internally consistent", result.consistent)
    row("expired", result.is_expired)
    # None rather than False when the gender was itself derived from the
    # number - agreeing with its own source proves nothing.
    row("gender cross-check", result.gender_cross_check_passed)

    if result.findings:
        lines.append("  findings")
        for finding in result.findings:
            lines.append(
                f"    {finding.severity.upper():<8s} {finding.code}: {finding.message}"
            )

    if result.error_message:
        lines.append(f"  error   {result.error_code}: {result.error_message}")

    lines.append("  provenance")
    row("engine", f"{result.engine} {result.engine_version}")
    row("lines detected", result.lines_detected)
    row("preprocessing", json.dumps(result.preprocessing, sort_keys=True))

    return "\n".join(lines)


def run_builtin(service: OcrService, *, as_json: bool) -> int:
    """Read every built-in scenario."""
    scenarios = build_scenarios()
    results = [(label, service.read(image)) for label, image in scenarios]

    if as_json:
        print(json.dumps(
            {label: result.model_dump(mode="json") for label, result in results},
            indent=2,
        ))
        return 0

    print()
    print("Synthetic Pakistani CNIC - fictitious holder, no real document used")
    print("=" * 78)
    for label, result in results:
        print(summarise(label, result))

    print()
    print("Full reading of the clean scan")
    print("-" * 78)
    print(render_detail(results[0][1]))

    tampered = next(
        result for label, result in results
        if label == "gender contradicts the number"
    )
    print()
    print("The cross-check that reads two independent parts of the card")
    print("-" * 78)
    print(
        "  The printed gender says M. The identity number ends in an even\n"
        "  digit, which by the CNIC's own construction means F. Nothing else\n"
        "  on the document reveals the disagreement."
    )
    print(render_detail(tampered))

    # Three scenarios are meant to fail, and failing is the demonstration:
    # a document that is not a CNIC, a blank page, and a card whose printed
    # gender contradicts its own identity number.
    expected_to_fail = {
        "gender contradicts the number",
        "a utility bill, not a CNIC",
        "a blank page",
    }
    unexpected = [
        label for label, result in results
        if not result.success and label not in expected_to_fail
    ]
    wrongly_accepted = [
        label for label, result in results
        if result.success and label in expected_to_fail
    ]

    print()
    if unexpected:
        print(f"Cards that should have read but did not: {', '.join(unexpected)}")
    if wrongly_accepted:
        print(f"Cards that should have been rejected: {', '.join(wrongly_accepted)}")
    if unexpected or wrongly_accepted:
        return 1

    print("Every genuine card read; every bad upload was rejected with guidance.")
    return 0


def run_single(service: OcrService, path: Path, *, as_json: bool) -> int:
    """Read one image from disk."""
    result = service.read(load_image(path))

    if as_json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return 0

    print()
    print(summarise(path.name, result))
    print(render_detail(result))
    return 0 if result.success else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Read a Pakistani CNIC and report every field with provenance."
    )
    parser.add_argument(
        "--image", type=Path, help="Read this image instead of the built-in cards."
    )
    parser.add_argument(
        "--engine", help="Force one engine: onnx_ppocr, paddleocr or easyocr."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    arguments = parser.parse_args(argv)

    configure_logging()

    try:
        service = build_ocr_service(
            engine_chain=[arguments.engine] if arguments.engine else None
        )
    except Exception as exc:  # noqa: BLE001 - a missing engine is a clean exit
        print(f"No OCR engine available: {exc}", file=sys.stderr)
        print(
            "Install the default engine with:  pip install 'hamqadam-ai[ocr]'",
            file=sys.stderr,
        )
        return 2

    try:
        if arguments.image:
            return run_single(service, arguments.image, as_json=arguments.json)
        return run_builtin(service, as_json=arguments.json)
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
