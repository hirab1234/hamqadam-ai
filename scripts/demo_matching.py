"""MODULE 4 demonstration - run a full verification match.

Usage
-----
::

    # Built-in scenarios, including the fraud cases
    python scripts/demo_matching.py

    # Your own images
    python scripts/demo_matching.py --selfie s.jpg --profile p.jpg --cnic c.jpg \\
        --secondary a.jpg --secondary b.jpg

    # Machine-readable
    python scripts/demo_matching.py --json

The built-in run walks four scenarios: a consistent identity, an impostor
profile photograph, a stranger's identity document, and one bad secondary
image among good ones.
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
from hamqadam_ai.embeddings.base import FaceEmbedding
from hamqadam_ai.logging import configure_logging
from hamqadam_ai.models.registry import get_registry
from hamqadam_ai.schemas.matching import MatchingResult
from hamqadam_ai.services import (
    build_embedding_service,
    build_face_detection_service,
    build_matching_service,
)
from hamqadam_ai.utils.geometry import BoundingBox, Landmarks5
from hamqadam_ai.utils.image_io import load_image

BgrImage = npt.NDArray[np.uint8]

_LANDMARK_ORDER = ("left_eye", "right_eye", "nose_tip", "mouth_left", "mouth_right")
BAR_WIDTH = 26


def person_a() -> BgrImage:
    """Public-domain portrait."""
    import matplotlib

    path = (
        Path(matplotlib.__file__).parent
        / "mpl-data" / "sample_data" / "grace_hopper.jpg"
    )
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("reference portrait unavailable")
    return image


def person_b() -> BgrImage | None:
    """A genuinely different identity."""
    try:
        from skimage import data

        return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001 - optional dependency
        return None


def jpeg(image: BgrImage, quality: int) -> BgrImage:
    """JPEG round-trip, standing in for a differently-captured upload."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else image


class Embedder:
    """Detection plus embedding, as one call."""

    def __init__(self) -> None:
        settings = get_settings()
        registry = get_registry(settings)
        self.detector = build_face_detection_service(settings, registry)
        self.service = build_embedding_service(settings, registry)

    def __call__(self, image: BgrImage, role: ImageRole) -> FaceEmbedding | None:
        """Embed the primary face, or return None when none was found."""
        detection = self.detector.detect(image, role=role)
        if detection.primary_face is None:
            return None
        bb = detection.primary_face.bounding_box
        box = BoundingBox(bb.x1, bb.y1, bb.x2, bb.y2)
        named = {lm.name: (lm.x, lm.y) for lm in detection.primary_face.landmarks}
        marks = (
            Landmarks5(np.array([named[n] for n in _LANDMARK_ORDER], dtype=np.float32))
            if all(n in named for n in _LANDMARK_ORDER)
            else None
        )
        return self.service.embed_to_vector(image, role=role, box=box, landmarks=marks)


def bar(score: float, width: int = BAR_WIDTH) -> str:
    """Render a 0-100 score as a text bar."""
    filled = int(round(width * max(0.0, min(100.0, score)) / 100.0))
    return "#" * filled + "." * (width - filled)


def print_result(title: str, result: MatchingResult) -> None:
    """Print one matching result."""
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)

    confidence = result.identity_confidence_score
    shown = f"{confidence:.1f}" if confidence is not None else "unavailable"
    print(f"  identity confidence : {shown}")
    print(f"  face_match_score    : {result.face_match_score:.1f}")
    if result.capped_by:
        print(f"  CAPPED BY           : {result.capped_by}")
    print()

    print(f"    {'comparison':<12s}{'score':>8s}{'cosine':>9s}  {'decision':<14s} bar")
    print("    " + "-" * 68)
    for entry in result.comparisons:
        if not entry.compared:
            print(
                f"    {entry.comparison:<12s}{'-':>8s}{'-':>9s}  "
                f"{'NOT_COMPARED':<14s} {entry.reason or ''}"
            )
            continue
        label = entry.target_label or entry.comparison
        print(
            f"    {label[:11]:<12s}{entry.score:>8.1f}{entry.similarity:>9.3f}  "
            f"{str(entry.decision):<14s} {bar(entry.score)}"
        )

    if result.secondary_worst_score is not None:
        print()
        print(f"  worst secondary     : {result.secondary_worst_score:.1f}")
    if result.contributions:
        print(f"  contributions       : {result.contributions}")

    notes = [w.message for w in result.warnings if w.code == "MATCHING_OBSERVATION"]
    if notes:
        print()
        print("  notes:")
        for note in notes:
            for line in _wrap(note, 68):
                print(f"    {line}")


def _wrap(text: str, width: int) -> list[str]:
    """Naive word wrap."""
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


def run_scenarios(embedder: Embedder, matcher) -> dict[str, MatchingResult]:  # noqa: ANN001
    """The four built-in scenarios."""
    a = person_a()
    b = person_b()

    selfie = embedder(a, ImageRole.LIVE_SELFIE)
    if selfie is None:
        raise SystemExit("no face detected in the reference portrait")

    own_profile = embedder(jpeg(a, 85), ImageRole.PROFILE_IMAGE)
    own_secondary = embedder(jpeg(a, 70), ImageRole.SECONDARY_IMAGE)
    own_cnic = embedder(jpeg(a, 40), ImageRole.CNIC_PORTRAIT)

    results: dict[str, MatchingResult] = {}

    results["1. Consistent identity (should APPROVE)"] = matcher.match(
        selfie=selfie,
        profile=own_profile,
        secondaries=[own_secondary],
        cnic=own_cnic,
        secondary_labels=["secondary_1"],
    )

    if b is None:
        print("  scikit-image unavailable; skipping the impostor scenarios")
        return results

    stranger_profile = embedder(b, ImageRole.PROFILE_IMAGE)
    stranger_cnic = embedder(b, ImageRole.CNIC_PORTRAIT)
    stranger_secondary = embedder(b, ImageRole.SECONDARY_IMAGE)

    results["2. Impostor profile photograph (should REJECT)"] = matcher.match(
        selfie=selfie,
        profile=stranger_profile,
        cnic=own_cnic,
    )

    results["3. Stranger's identity document (capped)"] = matcher.match(
        selfie=selfie,
        profile=own_profile,
        secondaries=[own_secondary],
        cnic=stranger_cnic,
        secondary_labels=["secondary_1"],
    )

    results["4. One bad secondary among good ones"] = matcher.match(
        selfie=selfie,
        profile=own_profile,
        secondaries=[own_secondary, stranger_secondary],
        cnic=own_cnic,
        secondary_labels=["genuine", "impostor"],
    )

    return results


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Demonstrate MODULE 4 face matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--selfie", type=Path, help="The live selfie.")
    parser.add_argument("--profile", type=Path, help="The main profile image.")
    parser.add_argument(
        "--secondary", type=Path, action="append", default=[],
        help="A secondary profile image; repeatable.",
    )
    parser.add_argument("--cnic", type=Path, help="The CNIC portrait.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)

    print("=" * 78)
    print("HAMQADAM AI - MODULE 4: FACE MATCHING")
    print("=" * 78)

    matcher = build_matching_service(settings)
    described = matcher.describe()
    print(f"  profile   strong_match : {described['thresholds']['profile']['strong_match']}")
    print(f"  cnic      strong_match : {described['thresholds']['cnic']['strong_match']}")
    print(f"  identity weights       : {described['identity']['weights']}")
    print(f"  thresholds validated   : {described['thresholds_validated']}")
    if not described["thresholds_validated"]:
        print()
        print("  NOTE: these are engineering defaults from the literature, not")
        print("  values derived from this deployment's own labelled data. Run")
        print("  scripts/evaluate_matching.py --corpus <dir> before production.")

    try:
        embedder = Embedder()
    except Exception as exc:  # noqa: BLE001 - explain rather than trace
        print()
        print(f"  Could not start: {exc}")
        print("  Run: python scripts/download_models.py")
        return 1

    if args.selfie:
        selfie = embedder(load_image(args.selfie).pixels, ImageRole.LIVE_SELFIE)
        if selfie is None:
            print(f"  no face detected in {args.selfie}")
            return 1
        result = matcher.match(
            selfie=selfie,
            profile=(
                embedder(load_image(args.profile).pixels, ImageRole.PROFILE_IMAGE)
                if args.profile
                else None
            ),
            secondaries=[
                embedder(load_image(path).pixels, ImageRole.SECONDARY_IMAGE)
                for path in args.secondary
            ],
            cnic=(
                embedder(load_image(args.cnic).pixels, ImageRole.CNIC_PORTRAIT)
                if args.cnic
                else None
            ),
            secondary_labels=[path.name for path in args.secondary],
        )
        results = {f"{args.selfie.name}": result}
    else:
        print()
        print("  no images supplied; running the built-in scenarios")
        results = run_scenarios(embedder, matcher)

    if args.json:
        print(
            json.dumps(
                {title: r.model_dump(mode="json") for title, r in results.items()},
                indent=2,
            )
        )
        return 0

    for title, result in results.items():
        print_result(title, result)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'scenario':<46s}{'identity':>10s}{'capped':>16s}  failed")
    print("  " + "-" * 74)
    for title, result in results.items():
        confidence = result.identity_confidence_score
        shown = f"{confidence:.1f}" if confidence is not None else "n/a"
        print(
            f"  {title[:45]:<46s}{shown:>10s}"
            f"{(result.capped_by or '-'):>16s}  "
            f"{'yes' if result.any_comparison_failed else 'no'}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
